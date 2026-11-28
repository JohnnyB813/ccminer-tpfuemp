"""Compare two GPU kernels on the same saved Verus solo job."""

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


def measure(miner: Path, job: Path, label: str) -> float:
    process = subprocess.Popen(
        [str(miner.resolve()), "--verus-solo", str(job), "0", "--mode", "gpu"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        env=os.environ.copy(),
    )
    lines: queue.Queue[str] = queue.Queue()
    assert process.stdout is not None
    threading.Thread(
        target=lambda: [lines.put(line.rstrip()) for line in process.stdout],
        daemon=True,
    ).start()
    rates = []
    checked = False
    deadline = time.monotonic() + 45
    try:
        while len(rates) < 2 and time.monotonic() < deadline:
            try:
                line = lines.get(timeout=1)
            except queue.Empty:
                if process.poll() is not None:
                    break
                continue
            if line.startswith(("READY", "GPU_CHECK", "RATE", "ERROR")):
                print(label, line, flush=True)
            if line in ("GPU_CHECK 128/128", "GPU_CHECK 258/258"):
                checked = True
            match = re.match(r"RATE ([0-9.]+) MH/s", line)
            if match:
                rates.append(float(match.group(1)))
        if not checked or len(rates) != 2:
            raise RuntimeError(f"{label} benchmark failed "
                               f"(exit={process.poll()}, checked={checked}, rates={rates})")
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
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    fixture = json.loads((ROOT / "test" / "fixtures.json").read_text())
    work = solo_bridge.build_work(fixture["template"], None, "00" * 32)
    with tempfile.TemporaryDirectory(dir=ROOT / "test") as folder:
        job = Path(folder) / "same-job.bin"
        work.job_file(job)
        baseline = measure(args.baseline, job, "baseline")
        candidate = measure(args.candidate, job, "candidate")
    print(f"GPU_KERNEL_BENCH baseline={baseline:.2f} MH/s "
          f"candidate={candidate:.2f} MH/s ratio={candidate / baseline:.2f}x")


if __name__ == "__main__":
    main()
