"""End-to-end bridge smoke test with a fake local Verus JSON-RPC server."""

import argparse
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import json
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading
import time

import solo_bridge as solo


ROOT = Path(__file__).resolve().parent
def test_address() -> str:
    # Deterministic public test address; no user's payout address or key is
    # included in the repository fixture.
    payload = bytes([60]) + hashlib.sha256(b"ccminer verus solo test").digest()[:20]
    value = int.from_bytes(payload + solo.sha256d(payload)[:4], "big")
    chars = []
    while value:
        value, digit = divmod(value, 58)
        chars.append(solo.BASE58[digit])
    return "".join(reversed(chars))


ADDRESS = test_address()
MERGED_BITS = "1e00ffff"  # easier than the VRSC target in the fixture
template = json.loads((ROOT / "test" / "fixtures.json").read_text())["template"]
old_script = next(script for value, script in solo.coinbase_outputs(
    bytes.fromhex(template["coinbasetxn"]["data"])) if value > 0)
new_script = solo.address_script(ADDRESS)
coinbase = bytes.fromhex(template["coinbasetxn"]["data"])
old_output = bytes([len(old_script)]) + old_script
assert coinbase.count(old_output) == 1
coinbase = coinbase.replace(old_output, bytes([len(new_script)]) + new_script, 1)
template["coinbasetxn"]["data"] = coinbase.hex()
template["coinbasetxn"]["hash"] = solo.sha256d(coinbase)[::-1].hex()
template["mergeminebits"] = MERGED_BITS
work = solo.build_work(template, ADDRESS)
assert int.from_bytes(work.target_be, "big") > int(template["target"], 16)
submitted = []


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        expected = "Basic " + base64.b64encode(b"solo:test").decode()
        if self.headers.get("Authorization") != expected:
            self.send_error(401)
            return
        length = int(self.headers["Content-Length"])
        message = json.loads(self.rfile.read(length))
        method = message["method"]
        if method == "getinfo":
            result = {"blocks": work.height - 1, "longestchain": work.height - 1,
                      "connections": 8}
        elif method == "getblocktemplate":
            assert message["params"][0]["miningdistribution"] == {ADDRESS: 1}
            result = template
        elif method == "submitmergedblock":
            raw = bytes.fromhex(message["params"][0])
            assert solo.check_candidate(work, raw[:1487].hex()) == raw[:1487]
            assert raw[1487:] == work.block_tail
            submitted.append(raw)
            result = None
        else:
            raise AssertionError(f"unexpected RPC method {method}")
        data = json.dumps({"result": result, "error": None,
                           "id": message["id"]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *_args):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("gpu", "cpu", "hybrid"), default="gpu")
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    with tempfile.TemporaryDirectory(dir=ROOT / "test") as folder:
        config = Path(folder) / "rpc.json"
        config.write_text(json.dumps({
            "url": f"http://127.0.0.1:{server.server_port}/",
            "user": "solo", "password": "test", "payout_address": ADDRESS,
        }))
        process = subprocess.Popen([
            sys.executable, str(ROOT / "solo_bridge.py"), "--config", str(config),
            "--miner", str(ROOT / "solo_gpu.exe"), "--mode", args.mode,
            "--cpu-threads", "4",
        ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        lines = queue.Queue()
        winner = None
        threading.Thread(target=lambda: [lines.put(x) for x in process.stdout],
                         daemon=True).start()
        deadline = time.monotonic() + 15
        try:
            while time.monotonic() < deadline:
                try:
                    line = lines.get(timeout=1).strip()
                except queue.Empty:
                    continue
                if line.startswith("WINNER "):
                    winner = line.split(" ", 1)[1]
                if line.startswith(("JOB", "READY", "GPU_CHECK", "WINNER", "SUBMIT", "SUBMIT_ERROR", "NODE")):
                    print(line)
                if line.startswith("SUBMIT "):
                    assert len(submitted) == 1
                    if args.mode == "hybrid":
                        assert winner in ("CPU", "GPU")
                        assert bool(submitted[0][1486] & 0x80) == (winner == "CPU")
                    digest = line.rsplit("hash=", 1)[1]
                    assert int(digest, 16) > int(template["target"], 16)
                    print(f"RPC_BRIDGE_OK mode={args.mode} "
                          f"submitted_bytes={len(submitted[0])}")
                    return
                if line.startswith("SUBMIT_ERROR"):
                    raise AssertionError(line)
            raise TimeoutError("bridge did not submit the easy target fixture")
        finally:
            process.terminate()
            process.wait(timeout=5)
            server.shutdown()


if __name__ == "__main__":
    main()
