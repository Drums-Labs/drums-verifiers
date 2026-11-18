"""drums_rl: one runner for prime-rl + Drums on your own GPUs, used unchanged by templates/modal (inside a Modal container)
and templates/gpu-box (inside the same image on any 8-GPU Linux machine). Standard library only.

    python3 drums_rl.py download                 fetch the pinned model into the weights volume (CPU)
    python3 drums_rl.py check                    imports, Drums health, render + `rl --dry-run` (CPU)
    python3 drums_rl.py scan                     the untrained model on the training tasks, K = 2, plus the pool report
    python3 drums_rl.py smoke                    one optimizer step end to end (all GPUs)
    python3 drums_rl.py train                    the run; resumes, moves partial checkpoints aside, stops at the budget
    python3 drums_rl.py evaluate --model M --split heldout --k 6 --tag T
    python3 drums_rl.py export [--step N]        a DCP checkpoint -> HF safetensors (CPU)
    python3 drums_rl.py spend                    the GPU-spend ledger
    python3 drums_rl.py render --out DIR         the rendered TOMLs, for reading or a dry run elsewhere
    python3 drums_rl.py pool-report RECORDS --group-size 8 [--last N]

Layout: weights in $DRUMS_RL_WEIGHTS (default /weights; HF cache under hf/), runs in $DRUMS_RL_RUNS (default /runs):
tasks/<split>/tasks.jsonl (copied there by `prepare`), <run name>/ (prime-rl's run dir), exports/, spend.jsonl.
The config is $DRUMS_RL_CONFIG (default ../rl-config.toml next to this folder). Secrets: DRUMS_URL and DRUMS_TOKEN come
from the environment only; nothing here reads, prints or writes them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRIME_RL = Path(os.environ.get("DRUMS_RL_PRIME_RL", "/app"))
UV_RUN = ["uv", "run", "--no-sync"]   # a plain `uv run` re-syncs the venv to prime-rl's lock and REMOVES drums-verifiers


def weights_dir() -> Path:
    return Path(os.environ.get("DRUMS_RL_WEIGHTS", "/weights"))


def runs_dir() -> Path:
    return Path(os.environ.get("DRUMS_RL_RUNS", "/runs"))


# --- config -----------------------------------------------------------------------------------------------------------

DEFAULTS = {
    "model": {}, "gpus": {"count": 8, "inference": 2, "trainer": 6, "region": "", "cpu": 32, "memory_gib": 256},
    "drums": {}, "tasks": {"desktops": 128},
    "run": {"name": "train", "seed": 1, "max_steps": 64, "ckpt_interval": 1, "recipe": "common/prime-rl-drums.toml"},
    "budget": {"usd": 0, "hours": 0}, "prices": {"cpu_core_hour": 0.0, "memory_gib_hour": 0.0, "box_usd_per_hour": 0},
}


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path or os.environ.get("DRUMS_RL_CONFIG") or HERE.parent / "rl-config.toml")
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    cfg = {k: {**v, **raw.get(k, {})} for k, v in DEFAULTS.items()}
    cfg.update({k: v for k, v in raw.items() if k not in cfg})
    g = cfg["gpus"]
    if g["inference"] + g["trainer"] != g["count"]:
        raise ValueError(f"[gpus] inference ({g['inference']}) + trainer ({g['trainer']}) must equal count ({g['count']})")
    if not cfg["model"].get("repo"):
        raise ValueError("[model] repo is required")
    cfg["_path"] = str(path)
    cfg["_dir"] = str(path.parent)
    return cfg


PIN_RE = re.compile(r"^drums-verifiers\s*(==\s*\d+\.\d+\.\d+\S*|@\s*git\+https://\S+@[0-9a-f]{40})$")


def image_pin_problem(spec: str, environ=None) -> str | None:
    """None when [image] drums_verifiers names a release (==X.Y.Z) or a full commit sha; else why it is refused.
    DRUMS_RL_ALLOW_UNPINNED=1 lets an unpinned spec through (a deliberate try, never a study)."""
    environ = os.environ if environ is None else environ
    if PIN_RE.match(spec.strip()):
        return None
    if environ.get("DRUMS_RL_ALLOW_UNPINNED") == "1":
        return None
    why = "still the placeholder PIN_A_COMMIT" if "PIN_A_COMMIT" in spec else "not a release (==X.Y.Z) or a 40-character commit sha"
    return (f"[image] drums_verifiers = {spec!r} is {why}. Put the commit you run there (see rl-config.toml), "
            "or set DRUMS_RL_ALLOW_UNPINNED=1 for a deliberate unpinned build")


def recipe_path(cfg: dict) -> Path:
    p = Path(cfg["run"]["recipe"])
    return p if p.is_absolute() else Path(cfg["_dir"]) / p


# --- rendering --------------------------------------------------------------------------------------------------------

PLACEHOLDER = re.compile(r'\["@([A-Z_]+)@"\]|"@([A-Z_]+)@"')


def render(template: str, values: dict) -> str:
    """Replace every "@NAME@" (a TOML string) and ["@NAME@"] (a one-item list) with the JSON of values[NAME]. JSON strings,
    numbers and lists of strings are valid TOML. A placeholder without a value is an error."""
    missing = []

    def sub(m):
        name = m.group(1) or m.group(2)
        if name not in values:
            missing.append(name)
            return m.group(0)
        v = values[name]
        if m.group(1) and not isinstance(v, list):
            v = [v]
        return json.dumps(v)
    out = PLACEHOLDER.sub(sub, template)
    if missing:
        raise KeyError(f"no value for {sorted(set(missing))}")
    return out


def agent_lines(recipe_text: str) -> str:
    """The recipe's env.taskset.task.* and env.agent.* lines (the evaluation runs the agent exactly as training does)."""
    keep = [line for line in recipe_text.splitlines() if line.startswith(("env.taskset.task.", "env.agent."))]
    return "\n".join(keep) + "\n"


def explode_tasks(split_dir: Path, out_dir: Path) -> list[str]:
    """tasks.jsonl -> <out_dir>/<id>.json (what drums-verifiers' tasks_dir reads). Idempotent; returns ids in file order."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = []
    with open(split_dir / "tasks.jsonl") as f:
        for line in f:
            if not line.strip():
                continue
            task = json.loads(line)
            ids.append(task["id"])
            p = out_dir / f"{task['id']}.json"
            text = json.dumps(task, sort_keys=True, ensure_ascii=False) + "\n"
            if not (p.exists() and p.read_text() == text):
                p.write_text(text)
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate task ids in {split_dir}")
    if not ids:
        raise ValueError(f"no tasks in {split_dir}/tasks.jsonl")
    return ids


def train_values(cfg: dict, ids: list[str], tasks_dir: Path, smoke: bool = False) -> dict:
    runs, g, r = runs_dir(), cfg["gpus"], cfg["run"]
    name = r["name"] + ("-smoke" if smoke else "")
    desktops = int(cfg["tasks"]["desktops"])
    inflight = 8 if smoke else desktops
    return {"MAX_STEPS": 1 if smoke else int(r["max_steps"]), "OUTPUT_DIR": str(runs), "RUN_NAME": name,
            "MODEL": cfg["model"]["repo"], "GPUS": int(g["count"]), "INFER_GPUS": int(g["inference"]),
            "TRAIN_GPUS": int(g["trainer"]), "CKPT_INTERVAL": int(r["ckpt_interval"]),
            "CONVERSION_DIR": str(runs / "conversion"), "BATCH_SIZE": 8 if smoke else 128,
            "MIN_INFLIGHT": 1 if smoke else max(1, (3 * desktops) // 4), "MAX_INFLIGHT": inflight,
            "SEED": int(r["seed"]), "TASKS_DIR": str(tasks_dir), "TASK_IDS": ids,
            "RECORD_PATH": str(runs / name / "drums-record.jsonl")}


def render_train(cfg: dict, ids: list[str], tasks_dir: Path, smoke: bool = False) -> str:
    return render(recipe_path(cfg).read_text(), train_values(cfg, ids, tasks_dir, smoke))


def render_eval(cfg: dict, model: str, ids: list[str], tasks_dir: Path, k: int, tag: str, inflight: int) -> tuple[str, str]:
    out = runs_dir() / "eval" / tag
    inf = render((HERE / "eval-inference.toml").read_text(),
                 {"OUTPUT_DIR": str(out / "inference"), "MODEL": model, "GPUS": 2})
    ev = (HERE / "eval.toml").read_text()
    marker = next(line for line in ev.splitlines() if line.startswith("# @AGENT_LINES@"))
    ev = ev.replace(marker, agent_lines(recipe_path(cfg).read_text()).rstrip("\n"))
    ev = render(ev, {"MODEL": model, "K": int(k), "OUTPUT_DIR": str(out), "TAG": tag, "INFLIGHT": int(inflight),
                     "TASKS_DIR": str(tasks_dir), "TASK_IDS": ids, "RECORD_PATH": str(out / f"{tag}.jsonl")})
    return inf, ev


# --- checkpoints ------------------------------------------------------------------------------------------------------

def step_number(p: Path) -> int | None:
    m = re.fullmatch(r"step_(\d+)", p.name)
    return int(m.group(1)) if m else None


def complete(step_dir: Path) -> bool:
    """A checkpoint is complete when the trainer's part finished (trainer/.metadata). D1: a preemption between the
    orchestrator's save and the trainer's left step_3 with only the orchestrator's progress, and prime-rl's --resume
    picked it by name and failed."""
    return (step_dir / "trainer" / ".metadata").exists()


def latest_complete(run_dir: Path) -> Path | None:
    steps = [(step_number(p), p) for p in (run_dir / "checkpoints").glob("step_*") if step_number(p) is not None]
    done = [p for n, p in sorted(steps) if complete(p)]
    return done[-1] if done else None


def move_partial_steps(run_dir: Path) -> list[str]:
    """Move every incomplete step_N out of checkpoints/ into checkpoints-partial/ (kept, never deleted), so --resume
    lands on the newest complete checkpoint. Returns the moved names."""
    ckpt = run_dir / "checkpoints"
    moved = []
    for p in sorted(ckpt.glob("step_*")) if ckpt.is_dir() else []:
        if step_number(p) is not None and not complete(p):
            dest = run_dir / "checkpoints-partial" / f"{p.name}-{int(time.time())}"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(p), str(dest))
            moved.append(p.name)
    return moved


# --- spend ledger and the budget --------------------------------------------------------------------------------------

BROAD_REGIONS = {"us", "eu", "ap", "uk", "ca", "jp", "au"}


def region_multiplier(region: str) -> float:
    """Modal bills a region pin on top of the base price: broad regions 1.15x, narrow ones 1.75x (read 2026-11-16)."""
    if not region:
        return 1.0
    return 1.15 if region in BROAD_REGIONS else 1.75


def usd_per_hour(cfg: dict, gpu_name: str, gpus: int, cpu: float, mem_gib: float) -> float:
    p = cfg["prices"]
    if os.environ.get("DRUMS_RL_PLATFORM") == "gpu-box":
        return float(p.get("box_usd_per_hour") or 0)
    rate = next((float(v) for k, v in p.items() if isinstance(v, (int, float)) and k.isupper() and k in gpu_name.upper()), None)
    if rate is None and gpus:
        raise ValueError(f"no price for GPU {gpu_name!r} in [prices]; add it (USD per GPU-hour)")
    hourly = gpus * (rate or 0) + cpu * float(p["cpu_core_hour"]) + mem_gib * float(p["memory_gib_hour"])
    return hourly * region_multiplier(cfg["gpus"].get("region") or "")


def read_ledger(path: Path) -> dict:
    """attempt id -> its last heartbeat."""
    out: dict = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                out[row["attempt"]] = row
    return out


def spent(path: Path, what: str | None = None) -> tuple[float, float]:
    """(USD, hours) over all attempts in the ledger, or only those of one kind (smoke, train, ...)."""
    usd = hours = 0.0
    for row in read_ledger(path).values():
        if what and row["what"] != what:
            continue
        h = max(0.0, row["t"] - row["t0"]) / 3600
        usd += h * row["usd_per_h"]
        hours += h
    return usd, hours


def gpu_info() -> tuple[str, int]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True,
                             timeout=60).stdout.strip().splitlines()
    except (OSError, subprocess.TimeoutExpired):
        out = []
    return (out[0].strip() if out else "none"), len(out)


class Meter:
    """Writes a heartbeat for this attempt to the ledger every `every_s` and, when given limits, calls stop() once the
    ledger's total spend reaches budget.usd or this kind's hours reach budget.hours."""

    def __init__(self, cfg: dict, what: str, cpu: float, mem_gib: float, limits: bool = False, every_s: float = 60.0):
        self.cfg, self.what, self.limits, self.every_s = cfg, what, limits, every_s
        self.path = runs_dir() / "spend.jsonl"
        self.gpu, self.gpus = gpu_info()
        self.row = {"attempt": f"{what}-{int(time.time())}-{os.getpid()}", "what": what, "gpu": self.gpu,
                    "gpus": self.gpus, "t0": time.time(), "t": time.time(),
                    "usd_per_h": usd_per_hour(cfg, self.gpu, self.gpus, cpu, mem_gib)}
        self.reason: str | None = None
        self._stop = threading.Event()
        self.on_limit = None

    def beat(self) -> None:
        self.row["t"] = time.time()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(self.row) + "\n")

    def over(self) -> str | None:
        b = self.cfg["budget"]
        usd, _ = spent(self.path)
        _, hours = spent(self.path, self.what)
        if b.get("usd") and usd >= float(b["usd"]):
            return f"budget reached: {usd:.2f} of {float(b['usd']):.2f} USD"
        if b.get("hours") and hours >= float(b["hours"]):
            return f"time budget reached: {hours:.2f} of {float(b['hours']):.2f} h of {self.what}"
        return None

    def _loop(self) -> None:
        while not self._stop.wait(self.every_s):
            self.beat()
            if self.limits and self.reason is None:
                self.reason = self.over()
                if self.reason and self.on_limit:
                    print(f"[drums_rl] {self.reason}; stopping", flush=True)
                    self.on_limit()

    def __enter__(self):
        self.beat()
        threading.Thread(target=self._loop, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self.beat()


# --- processes --------------------------------------------------------------------------------------------------------

def child_env(offline: bool = True) -> dict:
    env = {**os.environ, "HF_HOME": str(weights_dir() / "hf"), "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1"}
    if offline:
        env["HF_HUB_OFFLINE"] = "1"
    return env


def sh(cmd: list[str], check: bool = True, cwd: Path | None = None, offline: bool = True) -> int:
    print("$ " + " ".join(cmd), flush=True)
    rc = subprocess.run(cmd, cwd=cwd or PRIME_RL, env=child_env(offline)).returncode
    if check and rc != 0:
        raise SystemExit(f"command failed with exit code {rc}: {' '.join(cmd)}")
    return rc


def run_stoppable(cmd: list[str], meter: Meter) -> int:
    """Run cmd in its own process group; the meter's limit sends SIGTERM to the group, then SIGKILL after 120 s."""
    print("$ " + " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, cwd=PRIME_RL, env=child_env(), start_new_session=True)

    def stop():
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    meter.on_limit = stop
    return proc.wait()


def write_toml(text: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tomllib.loads(text)                      # never hand prime-rl a file that is not TOML
    path.write_text(text)
    return path


def tasks(split: str) -> tuple[list[str], Path]:
    src = runs_dir() / "tasks" / split
    if not (src / "tasks.jsonl").exists():
        raise SystemExit(f"{src}/tasks.jsonl is missing: run `prepare` first (it copies [tasks] {split} there)")
    out = runs_dir() / "tasks" / f"{split}-json"
    return explode_tasks(src, out), out


# --- commands ---------------------------------------------------------------------------------------------------------

def snapshot_dir(cfg: dict) -> Path:
    org_name = cfg["model"]["repo"].replace("/", "--")
    return weights_dir() / "hf" / "hub" / f"models--{org_name}" / "snapshots" / cfg["model"]["revision"]


def write_refs_main(snapshot: Path) -> Path:
    """snapshot_download(revision=<commit>) does not write refs/main; an offline lookup of the bare repo name (what vLLM
    and prime-rl do) then fails with LocalEntryNotFoundError although every file is there (D1 hit this)."""
    refs = snapshot.parent.parent / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    (refs / "main").write_text(snapshot.name)
    return refs / "main"


def cmd_download(cfg: dict, a) -> None:
    repo, rev = cfg["model"]["repo"], cfg["model"].get("revision")
    if not rev:
        raise SystemExit("[model] revision: pin a commit (the HF API lists it), so every run reads the same weights")
    code = ("import sys; from huggingface_hub import snapshot_download; "
            f"print(snapshot_download({repo!r}, revision={rev!r}, cache_dir={str(weights_dir() / 'hf' / 'hub')!r}, max_workers=8))")
    out = subprocess.run([str(PRIME_RL / ".venv/bin/python"), "-c", code], env=child_env(offline=False), check=True,
                         capture_output=True, text=True).stdout.strip().splitlines()[-1]
    snap = Path(out)
    write_refs_main(snap)
    lines = []
    for f in sorted(snap.iterdir()):
        h = hashlib.sha256()
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(64 << 20), b""):
                h.update(chunk)
        lines.append(f"{h.hexdigest()}  {f.name}")
    (weights_dir() / "SHA256SUMS").write_text(f"# {repo} revision {rev}\n" + "\n".join(lines) + "\n")
    print((weights_dir() / "SHA256SUMS").read_text())


def cmd_check(cfg: dict, a) -> None:
    problem = image_pin_problem(cfg.get("image", {}).get("drums_verifiers", ""))
    if problem:
        sys.exit(problem)
    sh([str(PRIME_RL / ".venv/bin/python"), "-c", "import drums, drums_verifiers, drums_chat, prime_rl, transformers; "
        "print('imports ok:', drums_verifiers.__name__, prime_rl.__name__, transformers.__version__)"])
    sh([str(PRIME_RL / ".venv/bin/python"), "-c", "import os, drums; c = drums.connect(os.environ['DRUMS_URL'], "
        "os.environ['DRUMS_TOKEN']); print('Drums health:', str(c.health())[:300])"], check=False)
    if not snapshot_dir(cfg).is_dir():
        print(f"weights not downloaded yet ({snapshot_dir(cfg)}); run `download`", flush=True)
    ids, tdir = tasks("train")
    p = write_toml(render_train(cfg, ids, tdir), Path("/tmp/drums-rl/train.toml"))
    check_harness(p.read_text())
    sh(UV_RUN + ["rl", "@", str(p), "--dry-run", "--output-dir", "/tmp/drums-rl/dry", "--run.name", "check"])


HARNESS_CHECK = """
import json, sys
from drums_chat import DrumsChatConfig
h = {k: v for k, v in json.loads(sys.argv[1]).items() if k != "id"}
unknown = sorted(set(h) - set(DrumsChatConfig.model_fields))
if unknown:
    sys.exit(f"the installed drums-verifiers does not know env.agent.harness {unknown}; prime-rl would ignore them "
             "silently. Install a newer drums-verifiers ([image] drums_verifiers) or remove those lines")
DrumsChatConfig(**h)
print("harness settings ok:", h)
"""


def check_harness(toml_text: str) -> None:
    """prime-rl accepts unknown env.agent.harness keys without a word (seen with drums-verifiers 0.2.2 and the window keys),
    so the run would quietly use the default window. Refuse instead."""
    t = tomllib.loads(toml_text)
    for src in t.get("orchestrator", {}).get("train", {}).get("source", []) or t.get("source", []):
        h = (src.get("env", {}).get("agent", {}) or {}).get("harness") or {}
        if h.get("id", "drums-chat") == "drums-chat":
            sh([str(PRIME_RL / ".venv/bin/python"), "-c", HARNESS_CHECK, json.dumps(h)])


def gpu_kind(cfg: dict) -> tuple[float, float]:
    return float(cfg["gpus"]["cpu"]), float(cfg["gpus"]["memory_gib"])


def cmd_smoke(cfg: dict, a) -> None:
    ids, tdir = tasks("train")
    p = write_toml(render_train(cfg, ids, tdir, smoke=True), runs_dir() / "configs" / "smoke.toml")
    check_harness(p.read_text())
    with Meter(cfg, "smoke", *gpu_kind(cfg)) as m:
        rc = run_stoppable(UV_RUN + ["rl", "@", str(p), "--clean"], m)
    if rc != 0:
        raise SystemExit(f"smoke failed: rl exited {rc}; logs under {runs_dir() / (cfg['run']['name'] + '-smoke')}/logs")
    print("smoke passed: one step, one weight sync, one checkpoint", flush=True)


def cmd_train(cfg: dict, a) -> int:
    run_dir = runs_dir() / cfg["run"]["name"]
    ids, tdir = tasks("train")
    p = write_toml(render_train(cfg, ids, tdir), runs_dir() / "configs" / "train.toml")
    check_harness(p.read_text())
    probe = Meter(cfg, "train", *gpu_kind(cfg))
    reason = probe.over()
    if reason:
        print(f"[drums_rl] {reason}: not starting (raise [budget] to continue)", flush=True)
        return 0
    args = []
    if run_dir.is_dir() and any(run_dir.iterdir()):
        moved = move_partial_steps(run_dir)
        if moved:
            print(f"[drums_rl] moved incomplete checkpoints aside: {moved} -> {run_dir / 'checkpoints-partial'}", flush=True)
        last = latest_complete(run_dir)
        print(f"[drums_rl] resuming {run_dir} from {last.name if last else 'scratch (no complete checkpoint)'}", flush=True)
        args.append("--resume")
    with Meter(cfg, "train", *gpu_kind(cfg), limits=True) as m:
        rc = run_stoppable(UV_RUN + ["rl", "@", str(p), *args, *a.extra], m)
    if m.reason:
        last = latest_complete(run_dir)
        print(f"[drums_rl] stopped: {m.reason}. Last complete checkpoint: {last}", flush=True)
        (run_dir / "STOPPED").write_text(json.dumps({"reason": m.reason, "last": str(last), "t": time.time()}) + "\n")
        return 0                              # a budget stop is the end of the run, not a failure to retry
    if rc != 0:
        raise SystemExit(f"rl exited {rc}; logs under {run_dir}/logs (a retry resumes from the last complete checkpoint)")
    return 0


def wait_for_server(url: str, proc: subprocess.Popen, timeout_s: int = 1800) -> None:
    import urllib.request
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            raise SystemExit(f"inference server exited early with code {proc.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    print(f"[drums_rl] inference up after {time.time() - t0:.0f} s", flush=True)
                    return
        except Exception:  # noqa: BLE001 - not up yet
            pass
        time.sleep(10)
    raise SystemExit(f"inference server not ready after {timeout_s} s")


def complete_export(model_dir: Path, snapshot: Path) -> None:
    """An export holds config + safetensors; vLLM also needs the tokenizer, chat template and image processor config."""
    for f in snapshot.iterdir():
        if f.name.endswith((".safetensors", ".index.json")) or f.name == "config.json":
            continue
        if not (model_dir / f.name).exists():
            shutil.copy(os.path.realpath(f), model_dir / f.name)


def cmd_evaluate(cfg: dict, a) -> dict:
    model = a.model if a.model != "base" else cfg["model"]["repo"]
    if model != cfg["model"]["repo"]:
        if not Path(model).is_dir():
            raise SystemExit(f"{model} is not a directory (pass `base` or an export from `export`)")
        complete_export(Path(model), snapshot_dir(cfg))
    ids, tdir = tasks(a.split)
    inf, ev = render_eval(cfg, model, ids, tdir, a.k, a.tag, a.inflight)
    out = runs_dir() / "eval" / a.tag
    ip, ep = write_toml(inf, out / "inference.toml"), write_toml(ev, out / "eval.toml")
    check_harness(ev)
    with Meter(cfg, f"eval:{a.tag}", 16, 64):
        server = subprocess.Popen(UV_RUN + ["inference", "@", str(ip)], cwd=PRIME_RL, env=child_env())
        try:
            wait_for_server("http://127.0.0.1:8000/v1/models", server)
            args = UV_RUN + ["eval", "@", str(ep)] + (["--resume"] if (out / a.tag).is_dir() else [])
            rc = sh(args, check=False)
        finally:
            server.terminate()
            try:
                server.wait(timeout=120)
            except subprocess.TimeoutExpired:
                server.kill()
    report = pool_report(read_records(out / f"{a.tag}.jsonl"), a.k)
    report.update(tag=a.tag, model=model, split=a.split, k=a.k, eval_exit_code=rc)
    (out / "result.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "per_task"}, indent=1), flush=True)
    return report


def cmd_scan(cfg: dict, a) -> dict:
    a.model, a.split, a.k, a.tag = "base", "train", 2, a.tag or "scan-step0"
    return cmd_evaluate(cfg, a)


def cmd_export(cfg: dict, a) -> Path:
    run_dir = runs_dir() / cfg["run"]["name"]
    step = run_dir / "checkpoints" / f"step_{a.step}" if a.step is not None else latest_complete(run_dir)
    if step is None or not complete(step):
        raise SystemExit(f"no complete checkpoint {'step_' + str(a.step) if a.step is not None else ''} under {run_dir}")
    # outside the checkpoint dir: keep_last rotates old step dirs away, an export inside one would go with it
    out = runs_dir() / "exports" / cfg["run"]["name"] / step.name
    out.parent.mkdir(parents=True, exist_ok=True)
    sh(UV_RUN + ["python", "tools/convert_dcp_to_bf16.py", str(step), str(out), "--workers", "2", "--overwrite"])
    complete_export(out, snapshot_dir(cfg))
    print(f"[drums_rl] exported {step} -> {out}", flush=True)
    return out


def cmd_spend(cfg: dict, a) -> None:
    path = runs_dir() / "spend.jsonl"
    rows = read_ledger(path)
    for r in rows.values():
        h = max(0.0, r["t"] - r["t0"]) / 3600
        print(f"{r['what']:24s} {r['gpus']} x {r['gpu']:28s} {h:7.2f} h  {h * r['usd_per_h']:9.2f} USD")
    usd, hours = spent(path)
    print(f"{'total':24s} {'':32s} {hours:7.2f} h  {usd:9.2f} USD   (budget {cfg['budget'].get('usd') or 'off'})")


def cmd_render(cfg: dict, a) -> None:
    out = Path(a.out)
    split_dir = Path(a.tasks) if a.tasks else runs_dir() / "tasks" / "train"
    ids = explode_tasks(split_dir, out / "train-json")
    write_toml(render_train(cfg, ids, out / "train-json"), out / "train.toml")
    write_toml(render_train(cfg, ids, out / "train-json", smoke=True), out / "smoke.toml")
    inf, ev = render_eval(cfg, cfg["model"]["repo"], ids, out / "train-json", 2, "scan-step0", 48)
    write_toml(inf, out / "eval-inference.toml")
    write_toml(ev, out / "eval.toml")
    print(f"rendered {len(ids)} tasks into {out}/{{train,smoke,eval-inference,eval}}.toml")


# --- the pool report --------------------------------------------------------------------------------------------------

def read_records(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    return [json.loads(line) for line in open(path) if line.strip()]


def pool_report(rows: list[dict], group_size: int, last: int = 0) -> dict:
    """Per task: scored episodes and success rate (valid episodes only). Pool numbers: mixed tasks (0 < rate < 1), pooled
    rate, and the expected share of GRPO groups with zero advantage, sum over tasks of p^G + (1 - p)^G, averaged: those
    groups give no gradient and prime-rl discards them. D1: 64 variants, the share went from few at step 1 to 93% at step
    5, and one optimizer step then needed up to 7,872 episodes."""
    rows = [r for r in rows if r.get("task")]
    if last:
        rows = rows[-last:]
    per: dict = {}
    for r in rows:
        v = per.setdefault(r["task"], {"task": r["task"], "episodes": 0, "scored": 0, "successes": 0, "invalid": 0,
                                        "errored": 0})
        v["episodes"] += 1
        if r.get("ok") is False or r.get("episode_failed"):
            v["errored"] += 1
        elif r.get("valid") is False or r.get("dropped"):
            v["invalid"] += 1
        elif r.get("reward") is not None:
            v["scored"] += 1
            v["successes"] += int(float(r["reward"]) >= 1.0)
    rates = []
    for v in per.values():
        v["rate"] = v["successes"] / v["scored"] if v["scored"] else None
        if v["rate"] is not None:
            rates.append(v["rate"])
    g = int(group_size)
    zero = [p ** g + (1 - p) ** g for p in rates]
    scored = sum(v["scored"] for v in per.values())
    return {"tasks": len(per), "episodes": len(rows), "scored": scored,
            "pooled_rate": (sum(v["successes"] for v in per.values()) / scored) if scored else None,
            "mixed": sum(1 for p in rates if 0 < p < 1), "floor": sum(1 for p in rates if p == 0),
            "ceiling": sum(1 for p in rates if p == 1), "group_size": g,
            "expected_zero_advantage_share": round(sum(zero) / len(zero), 3) if zero else None,
            "per_task": sorted(per.values(), key=lambda v: v["task"])}


def cmd_pool_report(cfg: dict | None, a) -> dict:
    rep = pool_report(read_records(Path(a.records)), a.group_size, a.last)
    print(json.dumps({k: v for k, v in rep.items() if k != "per_task"}, indent=1))
    return rep


# --- entry ------------------------------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ("download", "check", "smoke", "spend"):
        sp.add_parser(name)
    t = sp.add_parser("train")
    t.add_argument("extra", nargs=argparse.REMAINDER, help="extra prime-rl arguments, after --")
    s = sp.add_parser("scan")
    s.add_argument("--tag", default="")
    s.add_argument("--inflight", type=int, default=48)
    e = sp.add_parser("evaluate")
    e.add_argument("--model", default="base", help="`base` or an export directory")
    e.add_argument("--split", choices=("train", "heldout"), default="heldout")
    e.add_argument("--k", type=int, default=6)
    e.add_argument("--tag", required=True)
    e.add_argument("--inflight", type=int, default=48)
    x = sp.add_parser("export")
    x.add_argument("--step", type=int, default=None, help="default: the latest complete checkpoint")
    r = sp.add_parser("render")
    r.add_argument("--out", required=True)
    r.add_argument("--tasks", default="", help="a directory with tasks.jsonl (default: the runs volume's tasks/train)")
    p = sp.add_parser("pool-report")
    p.add_argument("records")
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--last", type=int, default=0, help="only the last N rows (a training record: the recent policy)")
    a = ap.parse_args(argv)
    if a.cmd == "pool-report":
        cmd_pool_report(None, a)
        return 0
    if getattr(a, "extra", None) and a.extra[:1] == ["--"]:
        a.extra = a.extra[1:]
    cfg = load_config(a.config)
    fn = {"download": cmd_download, "check": cmd_check, "smoke": cmd_smoke, "train": cmd_train, "scan": cmd_scan,
          "evaluate": cmd_evaluate, "export": cmd_export, "spend": cmd_spend, "render": cmd_render}[a.cmd]
    rc = fn(cfg, a)
    return rc if isinstance(rc, int) else 0


if __name__ == "__main__":
    sys.exit(main())
