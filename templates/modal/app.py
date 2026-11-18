"""prime-rl + Drums desktops in YOUR Modal account. Everything (image, volumes, GPUs, spend) is in your workspace; Drums
only serves the desktops, through the URL and key in a Modal Secret you create.

    modal run templates/modal/app.py::prepare                      volumes + tasks upload (CPU, local)
    modal run templates/modal/app.py::download                     the pinned weights into the weights volume (CPU)
    modal run templates/modal/app.py::check                        imports, Drums health, prime-rl dry run (CPU)
    modal run --detach templates/modal/app.py::launch --what scan  the untrained model on the training tasks (2 GPUs)
    modal run --detach templates/modal/app.py::launch --what smoke one step end to end (8 GPUs)
    modal run --detach templates/modal/app.py::launch --what train the run: resumes, stops itself at the budget
    modal run templates/modal/app.py::evaluate --model base --split heldout --k 6 --tag step0-heldout
    modal run templates/modal/app.py::export                       the last complete checkpoint -> HF safetensors (CPU)
    modal run templates/modal/app.py::spend                        what the GPUs cost so far (the ledger)

The one config file is templates/rl-config.toml (or $DRUMS_RL_CONFIG). The work itself is templates/common/drums_rl.py,
the same runner templates/gpu-box uses. Guide: docs/train-on-your-gpus.md.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
import tomllib
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
COMMON = HERE.parent / "common"
CONFIG_PATH = Path(os.environ.get("DRUMS_RL_CONFIG") or HERE.parent / "rl-config.toml")
IN_CONTAINER = "/opt/drums-rl"


def _config() -> dict:
    if modal.is_local():
        return tomllib.loads(CONFIG_PATH.read_text())
    return tomllib.loads(Path(IN_CONTAINER, "rl-config.toml").read_text())


CFG = _config()
G, IMG, M = CFG["gpus"], CFG["image"], CFG.get("modal", {})
REGION = G.get("region") or None


def _pin_problem(spec: str) -> str | None:
    """drums_rl.image_pin_problem (the runner is standard library only, so it loads here without the image)."""
    import importlib.util
    s = importlib.util.spec_from_file_location("drums_rl", COMMON / "drums_rl.py")
    r = importlib.util.module_from_spec(s)
    s.loader.exec_module(r)
    return r.image_pin_problem(spec)


if modal.is_local() and (_problem := _pin_problem(IMG["drums_verifiers"])):
    raise SystemExit(_problem)

app = modal.App(M.get("app", "drums-rl"))

image = modal.Image.from_dockerfile(
    COMMON / "Dockerfile", context_dir=COMMON,
    build_args={"PRIME_RL_COMMIT": IMG["prime_rl_commit"], "DRUMS_VERIFIERS": IMG["drums_verifiers"],
                "DRUMS_SDK": IMG["drums_sdk"]},
)
# The runner, its templates and the config are mounted when a container starts, so editing them needs no image rebuild.
image = image.add_local_dir(COMMON, f"{IN_CONTAINER}/common", ignore=["Dockerfile", "__pycache__"]) \
             .add_local_file(CONFIG_PATH, f"{IN_CONTAINER}/rl-config.toml")

weights = modal.Volume.from_name(M.get("weights_volume", "drums-rl-weights"), create_if_missing=True, version=2)
runs = modal.Volume.from_name(M.get("runs_volume", "drums-rl-runs"), create_if_missing=True, version=2)
VOLUMES = {"/weights": weights, "/runs": runs}
drums_secret = modal.Secret.from_name(CFG["drums"].get("modal_secret", "drums"), required_keys=["DRUMS_URL", "DRUMS_TOKEN"])
started = modal.Dict.from_name(f"{M.get('app', 'drums-rl')}-started", create_if_missing=True)
ENV = {"DRUMS_RL_CONFIG": f"{IN_CONTAINER}/rl-config.toml", "DRUMS_RL_WEIGHTS": "/weights", "DRUMS_RL_RUNS": "/runs",
       "DRUMS_RL_PLATFORM": "modal", "PYTHONUNBUFFERED": "1"}


def _runner(*args: str) -> int:
    """drums_rl.py in the image's system Python (it only needs the standard library; prime-rl runs in /app/.venv)."""
    runs.reload()
    weights.reload()
    cmd = [sys.executable, f"{IN_CONTAINER}/common/drums_rl.py", *args]
    print("$ " + " ".join(cmd), flush=True)
    rc = subprocess.run(cmd, env={**os.environ, **ENV}).returncode
    runs.commit()
    if rc != 0:
        raise RuntimeError(f"drums_rl {args[0]} exited {rc}")
    return rc


def _mark_started(what: str) -> None:
    started[what] = time.time()


_GPU_KW = dict(image=image, volumes=VOLUMES, secrets=[drums_secret], cpu=float(G["cpu"]), memory=int(G["memory_gib"]) * 1024,
               region=REGION, env=ENV)
_EVAL_KW = dict(image=image, volumes=VOLUMES, secrets=[drums_secret], cpu=16, memory=64 * 1024, region=REGION, env=ENV)
_CPU_KW = dict(image=image, volumes=VOLUMES, env=ENV)


@app.function(timeout=4 * 3600, cpu=8, memory=32 * 1024, **_CPU_KW)
def download() -> None:
    _runner("download")
    weights.commit()


@app.function(timeout=1800, cpu=4, memory=16 * 1024, secrets=[drums_secret], **_CPU_KW)
def check() -> None:
    _runner("check")


@app.function(gpu=G["modal"], timeout=2 * 3600, retries=0, **_GPU_KW)
def smoke() -> None:
    _mark_started("smoke")
    _runner("smoke")


# Retries: a preempted container is restarted by Modal with the same input; drums_rl then resumes from the newest
# COMPLETE checkpoint (partial step dirs are moved aside first). A budget stop returns normally, so it is not retried.
@app.function(gpu=G["modal"], timeout=24 * 3600, retries=modal.Retries(max_retries=2, initial_delay=60.0), **_GPU_KW)
def train(extra: str = "") -> None:
    _mark_started("train")
    _runner("train", *(["--", *extra.split()] if extra else []))


@app.function(gpu=G.get("modal_eval", ["H200:2", "B200:2"]), timeout=4 * 3600, retries=0, **_EVAL_KW)
def scan_remote() -> None:
    _mark_started("scan")
    _runner("scan")


@app.function(gpu=G.get("modal_eval", ["H200:2", "B200:2"]), timeout=4 * 3600, retries=0, **_EVAL_KW)
def evaluate_remote(model: str, split: str, k: int, tag: str, inflight: int) -> None:
    _mark_started(f"eval:{tag}")
    _runner("evaluate", "--model", model, "--split", split, "--k", str(k), "--tag", tag, "--inflight", str(inflight))


@app.function(timeout=4 * 3600, cpu=16, memory=128 * 1024, **_CPU_KW)
def export_remote(step: int = -1) -> None:
    _runner("export", *([] if step < 0 else ["--step", str(step)]))


@app.function(timeout=600, cpu=1, memory=1024, **_CPU_KW)
def spend_remote() -> None:
    _runner("spend")


# --- local entrypoints ------------------------------------------------------------------------------------------------

@app.local_entrypoint()
def prepare() -> None:
    """Upload [tasks] train and heldout (directories with tasks.jsonl) to the runs volume as tasks/train, tasks/heldout."""
    base = CONFIG_PATH.parent
    with runs.batch_upload(force=True) as up:
        for split in ("train", "heldout"):
            src = Path(CFG["tasks"][split])
            src = src if src.is_absolute() else base / src
            if not (src / "tasks.jsonl").exists():
                raise SystemExit(f"[tasks] {split} = {src}: no tasks.jsonl there (drums_tasks.split writes one)")
            up.put_file(src / "tasks.jsonl", f"/tasks/{split}/tasks.jsonl")
            print(f"uploaded {src / 'tasks.jsonl'} -> {runs.name}:/tasks/{split}/tasks.jsonl")


@app.local_entrypoint()
def launch(what: str = "train", extra: str = "", deadline_min: float = -1) -> None:
    """Start scan, smoke or train and wait until a container has really started; cancel the call when none has after
    [gpus] start_deadline_min (D1: 3 h in the queue for 2 x B200 without a container). Use `modal run --detach`, so the
    run continues after this command returns."""
    fns = {"train": lambda: train.spawn(extra), "smoke": smoke.spawn, "scan": scan_remote.spawn}
    if what not in fns:
        raise SystemExit(f"--what is one of {sorted(fns)}")
    deadline = float(G.get("start_deadline_min", 20)) if deadline_min < 0 else deadline_min
    started.pop(what, None)
    t0 = time.time()
    call = fns[what]()
    print(f"{what}: call {call.object_id} spawned on {G['modal'] if what != 'scan' else G.get('modal_eval')}; "
          f"waiting up to {deadline:.0f} min for a container", flush=True)
    while True:
        if started.get(what, 0) >= t0:
            print(f"{what}: container started after {(time.time() - t0) / 60:.1f} min. Logs: modal app logs "
                  f"{app.name}. The run continues in Modal (this command can exit with --detach).", flush=True)
            return
        try:
            call.get(timeout=0)
            print(f"{what}: finished", flush=True)
            return
        except TimeoutError:
            pass
        if deadline and time.time() - t0 > deadline * 60:
            call.cancel()
            raise SystemExit(f"{what}: no container after {deadline:.0f} min on {G['modal']}; call cancelled (nothing "
                             "billed for GPUs). Add a GPU type to [gpus] modal, drop a region pin, or try later.")
        time.sleep(15)


@app.local_entrypoint()
def evaluate(model: str = "base", split: str = "heldout", k: int = 6, tag: str = "step0-heldout", inflight: int = 48) -> None:
    evaluate_remote.remote(model, split, k, tag, inflight)


@app.local_entrypoint()
def export(step: int = -1) -> None:
    export_remote.remote(step)


@app.local_entrypoint()
def spend() -> None:
    spend_remote.remote()
