"""Verifiers named by hash (drums_tasks.upload): a task set names its verifier's sha256, and the host gets the file
before the first create. Standard library only: a stand-in drumsd keeps an in-memory verifier store."""
import hashlib
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from drums_tasks import upload as U  # noqa: E402


class Store(BaseHTTPRequestHandler):
    store, uploads, auth = {}, [], []

    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        sha = self.path.rsplit("/", 1)[-1]
        if sha in Store.store:
            return self._send(200, {"sha256": sha})
        self._send(404, {"error": "no verifier", "code": "verifier_missing"})

    def do_POST(self):
        data = self.rfile.read(int(self.headers["Content-Length"]))
        Store.auth.append(self.headers.get("Authorization"))
        sha = hashlib.sha256(data).hexdigest()
        Store.uploads.append((self.path, sha))
        Store.store[sha] = data
        self._send(201, {"sha256": sha, "bytes": len(data), "created": True})


class NotFound(Exception):
    def __init__(self, status):
        self.status = status


class Client:
    """The two things ensure_verifiers uses of the drums SDK's Client: url/token and call()."""

    def __init__(self, url):
        self.url, self.token = url, "t0ken"

    def call(self, method, path):
        import urllib.error
        import urllib.request
        try:
            with urllib.request.urlopen(urllib.request.Request(self.url + path, method=method)) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise NotFound(e.code) from None


@pytest.fixture()
def host():
    Store.store, Store.uploads, Store.auth = {}, [], []
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Store)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield Client(f"http://127.0.0.1:{srv.server_address[1]}")
    srv.shutdown()


def t6_sha():
    return hashlib.sha256(open(U.BUNDLED["drums_t6"], "rb").read()).hexdigest()


def lab_task(**ev):
    return {"id": "t6-x-0001", "evaluator": {"func": "drums_verifier", "verifier": "drums_t6", "function": "check",
                                             "redundant": {"verifier": "drums_t6", "function": "check_alt"}, **ev}}


def test_name_by_hash_adds_the_hash_and_keeps_the_name():
    t = lab_task()
    h = U.name_by_hash(t, U.bundled_sources())
    assert h["evaluator"]["verifier_sha256"] == t6_sha() and h["evaluator"]["verifier"] == "drums_t6"
    assert h["evaluator"]["redundant"]["verifier_sha256"] == t6_sha()
    assert "verifier_sha256" not in t["evaluator"], "the input is not changed"
    assert U.task_hashes(h) == [t6_sha()]
    osworld = {"id": "x", "evaluator": {"func": "compare_table"}}
    assert U.name_by_hash(osworld, U.bundled_sources()) == osworld and U.task_hashes(osworld) == []


def test_ensure_uploads_a_missing_verifier_once(host):
    t = U.name_by_hash(lab_task(), U.bundled_sources())
    cache = {}
    assert U.ensure_verifiers(host, [t, t], cache=cache) == {t6_sha(): "uploaded"}
    assert Store.uploads == [("/v1/verifiers?name=drums_t6", t6_sha())]
    assert Store.auth == ["Bearer t0ken"]
    assert U.ensure_verifiers(host, [t], cache=cache) == {t6_sha(): "uploaded"} and len(Store.uploads) == 1, "cached"
    assert U.ensure_verifiers(host, [t]) == {t6_sha(): "present"} and len(Store.uploads) == 1


def test_a_hash_with_no_local_file_says_so_and_a_local_file_is_found(host, tmp_path):
    p = tmp_path / "acme.py"
    p.write_bytes(b"def check(ctx):\n    return 1\n")
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    t = {"id": "acme-1", "evaluator": {"func": "drums_verifier", "verifier_sha256": sha, "function": "check"}}
    with pytest.raises(LookupError, match=sha):
        U.ensure_verifiers(host, [t])
    assert U.ensure_verifiers(host, [t], U.sources_from_files([str(p)])) == {sha: "uploaded"}
    assert Store.uploads[-1] == ("/v1/verifiers?name=acme", sha)


def test_no_token_over_plain_http_to_another_host():
    c = Client("http://203.0.113.9:7070")
    with pytest.raises(RuntimeError, match="plain http"):
        U.upload(c, b"x = 1\n")
