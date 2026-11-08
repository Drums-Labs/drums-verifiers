"""Hybrid actions on a live desktop: GUI, browser, shell, files and LibreOffice in one env, and the same tools over MCP.

    python -I t6/hybrid_e2e.py --tasks TS_DIR --out OUT/hybrid.json [--n 3]

For n chrome_download tasks (setup: a local site, Chrome with its DevTools port):
  1. browser_tabs lists the start page; browser_open opens the file's URL, so Chrome downloads it to ~/Downloads;
  2. shell `sha256sum` and read_file (base64) both see the downloaded file, with the sha256 the factory recorded;
  3. write_file a CSV, office_convert it to xlsx, spreadsheet_read the xlsx back (LibreOffice), list_dir shows both;
  4. gui: pyautogui moves the mouse; screenshot is a 1920x1080 PNG;
  5. verify: the task's checker scores the browser's download 1.0, with a clean audit;
  6. MCP: `python -m drums.mcp --env <id>` over stdio: initialize, tools/list, tools/call shell + read_file + screenshot + verify.
"""
import argparse
import base64
import hashlib
import json
import os
import subprocess
import sys
import time

import drums
from drums.hybrid import HybridTools


def mcp_session(env_id, token_file):
    env = dict(os.environ, DRUMS_TOKEN_FILE=token_file)
    p = subprocess.Popen([sys.executable, "-m", "drums.mcp", "--env", env_id], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         text=True, env=env)
    msgs = [{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "shell", "arguments": {"command": "ls ~/Downloads"}}},
            {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "screenshot", "arguments": {}}},
            {"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {"name": "verify", "arguments": {}}}]
    out, _ = p.communicate("\n".join(json.dumps(m) for m in msgs) + "\n", timeout=300)
    rep = [json.loads(x) for x in out.splitlines() if x.strip()]
    by = {r["id"]: r for r in rep}
    return {"replies": len(rep), "tools": [t["name"] for t in by[2]["result"]["tools"]],
            "shell": json.loads(by[3]["result"]["content"][0]["text"]),
            "screenshot_type": by[4]["result"]["content"][0]["type"],
            "screenshot_bytes": len(base64.b64decode(by[4]["result"]["content"][0]["data"])),
            "verify": json.loads(by[5]["result"]["content"][0]["text"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--token-file", default=os.path.expanduser("~/.drums-token"))
    a = ap.parse_args()
    tasks = {json.loads(x)["id"]: json.loads(x) for x in open(os.path.join(a.tasks, "tasks.jsonl"))}
    sols = [json.loads(x) for x in open(os.path.join(a.tasks, "solutions.jsonl"))]
    picks = [s for s in sols if s["family"] == "chrome_download"][:a.n]
    c = drums.connect(token=open(a.token_file).read().strip())
    results = []
    for s in picks:
        t = tasks[s["id"]]
        want = t["evaluator"]["options"]["must_exist"]
        dst, sha = next(iter(want.items()))
        url = next(x["command"][1] for x in s["oracle"] if x["type"] == "execute")
        env = c.create(task=t, idle_timeout=900)
        env.wait_valid()
        h = HybridTools(env)
        r = {"task": s["id"], "env": env.id, "checks": {}}
        try:
            tabs0 = h.browser_tabs()
            r["checks"]["browser_tabs_start_page"] = any("index.html" in (x.get("url") or "") for x in tabs0.get("tabs", []))
            r["browser_open"] = h.browser_open(url)
            t0 = time.time()
            got = None
            while time.time() - t0 < 30:
                f = h.read_file(dst)
                if f.get("exists") and "base64" in f and hashlib.sha256(base64.b64decode(f["base64"])).hexdigest() == sha:
                    got = time.time() - t0
                    break
                time.sleep(1)
            r["download_visible_s"] = None if got is None else round(got, 2)
            r["checks"]["read_file_sees_browser_download"] = got is not None
            sh = h.shell(f"sha256sum '{dst}'")
            r["checks"]["shell_sees_browser_download"] = sh["stdout"].split(" ")[0] == sha
            h.write_file("~/Documents/t6_hybrid.csv", "item,qty\nanvil,3\nbeacon,5\n")
            conv = h.office_convert("~/Documents/t6_hybrid.csv", "xlsx")
            rows = h.spreadsheet_read("~/Documents/t6_hybrid.xlsx").get("rows")
            r["checks"]["office_convert_then_read"] = conv["ok"] and rows == [["item", "qty"], ["anvil", "3"], ["beacon", "5"]]
            names = [e["name"] for e in h.list_dir("~/Documents").get("entries", [])]
            r["checks"]["list_dir"] = {"t6_hybrid.csv", "t6_hybrid.xlsx"} <= set(names)
            r["checks"]["gui"] = h.gui("pyautogui.moveTo(50, 50)")["returncode"] == 0
            png = h.screenshot()
            r["checks"]["screenshot_png_1920x1080"] = png[:8] == b"\x89PNG\r\n\x1a\n" and drums._png_size(png) == (1920, 1080)
            rew = env.verify(strict=False)
            r["verify"] = {"score": rew.score, "valid": rew.valid, "trusted": rew.trusted, "audit_flags": rew.audit_flags}
            # the csv/xlsx this test wrote are outside the task's allow list: the audit must say so (scope_violation)
            r["checks"]["checker_scores_browser_download"] = rew.score == 1.0 and rew.valid
            r["checks"]["audit_flags_the_extra_files"] = "scope_violation" in rew.audit_flags
            m = mcp_session(env.id, a.token_file)
            r["mcp"] = m
            r["checks"]["mcp_tools_list"] = {"gui", "shell", "read_file", "write_file", "verify"} <= set(m["tools"])
            r["checks"]["mcp_shell_sees_download"] = os.path.basename(dst) in m["shell"]["stdout"]
            r["checks"]["mcp_screenshot_image"] = m["screenshot_type"] == "image" and m["screenshot_bytes"] > 10000
            r["checks"]["mcp_verify"] = m["verify"]["score"] == 1.0
        finally:
            env.close()
        r["pass"] = all(r["checks"].values())
        print(json.dumps({"task": r["task"], "pass": r["pass"], "checks": r["checks"]}), flush=True)
        results.append(r)
    out = {"n": len(results), "pass": sum(x["pass"] for x in results), "results": results}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({"n": out["n"], "pass": out["pass"]}))


if __name__ == "__main__":
    main()
