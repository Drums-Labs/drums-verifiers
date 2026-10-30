"""latency_probe.py URL MODEL OUT.jsonl STOP_EPOCH [INTERVAL_S]: the shared model endpoint's latency during a soak.

Every INTERVAL_S (default 30 s), until STOP_EPOCH: one fixed chat request (8 output tokens, temperature 0), timed end to
end, plus vLLM's queue gauges from GET <base>/metrics (requests running / waiting, KV cache use) when exposed. One JSON line
per probe. The load it adds is one short request per interval."""
import json
import re
import sys
import time
import urllib.request

URL, MODEL, OUT, STOP = sys.argv[1].rstrip("/"), sys.argv[2], sys.argv[3], float(sys.argv[4])
EVERY = float(sys.argv[5]) if len(sys.argv) > 5 else 30.0
BASE = URL[:-3] if URL.endswith("/v1") else URL
BODY = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": "Reply with the word ok."}], "max_tokens": 8,
                   "temperature": 0}).encode()
GAUGES = re.compile(r"^vllm:(num_requests_running|num_requests_waiting|gpu_cache_usage_perc|kv_cache_usage_perc)\{[^}]*\} ([0-9.eE+-]+)", re.M)
while time.time() < STOP:
    t0 = time.time(); row = {"t": round(t0, 3)}
    try:
        req = urllib.request.Request(URL + "/chat/completions", data=BODY, headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"})
        with urllib.request.urlopen(req, timeout=120) as r:
            body = json.loads(r.read())
        row.update(latency_s=round(time.time() - t0, 3), ok=True, completion_tokens=(body.get("usage") or {}).get("completion_tokens"))
    except Exception as e:  # noqa: BLE001
        row.update(latency_s=round(time.time() - t0, 3), ok=False, error=f"{type(e).__name__}: {e}"[:200])
    try:
        with urllib.request.urlopen(BASE + "/metrics", timeout=10) as r:
            m = {}
            for k, v in GAUGES.findall(r.read().decode()):
                m[k] = m.get(k, 0.0) + float(v)          # summed over data-parallel engines
            row["vllm"] = m
    except Exception as e:  # noqa: BLE001
        row["vllm_error"] = str(e)[:120]
    with open(OUT, "a") as f:
        f.write(json.dumps(row) + "\n")
    time.sleep(max(0.0, EVERY - (time.time() - t0)))
