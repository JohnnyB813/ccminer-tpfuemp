"""Compare scalar and paired VAES finalizers on the same saved Verus job."""

import argparse
import json
import os
from pathlib import Path
import queue
import re
import subprocess
import tempfile
import threading
import time

import solo_bridge


ROOT = Path(__file__).resolve().parent


def measure(miner: Path, job: Path, threads: int, vaes: bool) -> float:
    env = os.environ.copy()
    env["VERUS_SOLO_VAES"] = "1" if vaes else "0"
    process = subprocess.Popen(
        [str(miner.resolve()), "--verus-solo", str(job), "0",
         "--mode", "cpu", "--threads", str(threads)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=env,
    )
    lines: queue.Queue[str] = queue.Queue()
    assert process.stdout is not None
    threading.Thread(
        target=lambda: [lines.put(line.rstrip()) for line in process.stdout],
        daemon=True,
    ).start()
    rates = []
    deadline = time.monotonic() + 45
    try:
        while len(rates) < 2 and time.monotonic() < deadline:
            try:
                line = lines.get(timeout=1)
            except queue.Empty:
                if process.poll() is not None:
                    break
                continue
            if line.startswith(("READY", "RATE", "ERROR")):
                print(line, flush=True)
            match = re.match(r"RATE ([0-9.]+) MH/s", line)
            if match:
                rates.append(float(match.group(1)))
        if len(rates) != 2:
            raise RuntimeError(f"{'VAES' if vaes else 'scalar'} benchmark failed "
                               f"(exit={process.poll()}, rates={rates})")
        return sum(rates) / len(rates)
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--miner", type=Path, default=ROOT / "solo_gpu.exe")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.threads <= 256:
        parser.error("--threads must be 1..256")
    fixture = json.loads((ROOT / "test" / "fixtures.json").read_text())
    work = solo_bridge.build_work(fixture["template"], None, "00" * 32)
    with tempfile.TemporaryDirectory(dir=ROOT / "test") as folder:
        job = Path(folder) / "same-job.bin"
        work.job_file(job)
        scalar = measure(args.miner, job, args.threads, False)
        vaes = measure(args.miner, job, args.threads, True)
    print(f"CPU_FINALIZER_BENCH threads={args.threads} "
          f"scalar={scalar:.2f} MH/s vaes2={vaes:.2f} MH/s "
          f"ratio={vaes / scalar:.2f}x")


if __name__ == "__main__":
    main()
