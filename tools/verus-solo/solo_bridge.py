"""Feed Verus getblocktemplate jobs to one ccminer CPU/GPU/hybrid binary."""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import queue
import secrets
import struct
import subprocess
import threading
import time
from urllib import error, request


ROOT = Path(__file__).resolve().parent
MAGIC = b"VRSOLO1" + bytes([0])
COMPACT_SOLUTION = bytes.fromhex("fd4005")
BASE58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def sha256d(data: bytes) -> bytes:
    return hashlib.sha256(hashlib.sha256(data).digest()).digest()


def address_script(address: str) -> bytes:
    if len(address) != 34 or not address.startswith("R"):
        raise ValueError("payout must be a Verus R-address")
    value = 0
    for char in address:
        if char not in BASE58:
            raise ValueError("payout address contains a non-Base58 character")
        value = value * 58 + BASE58.index(char)
    raw = value.to_bytes((value.bit_length() + 7) // 8, "big")
    raw = bytes([0]) * (len(address) - len(address.lstrip("1"))) + raw
    if len(raw) != 25 or raw[0] != 60 or sha256d(raw[:21])[:4] != raw[21:]:
        raise ValueError("invalid Verus R-address checksum or version")
    return bytes.fromhex("76a914") + raw[1:21] + bytes.fromhex("88ac")


def read_varint(data: bytes, offset: int) -> tuple[int, int]:
    if offset >= len(data):
        raise ValueError("truncated varint")
    first = data[offset]
    offset += 1
    size = {0xFD: 2, 0xFE: 4, 0xFF: 8}.get(first)
    if size is None:
        return first, offset
    if offset + size > len(data):
        raise ValueError("truncated varint body")
    return int.from_bytes(data[offset:offset + size], "little"), offset + size


def write_varint(value: int) -> bytes:
    if value < 0xFD:
        return bytes([value])
    if value <= 0xFFFF:
        return b"\xfd" + struct.pack("<H", value)
    if value <= 0xFFFFFFFF:
        return b"\xfe" + struct.pack("<I", value)
    return b"\xff" + struct.pack("<Q", value)


def compact_target(bits_hex: str) -> int:
    bits = bytes.fromhex(bits_hex)
    if len(bits) != 4:
        raise ValueError("expected four-byte compact target")
    exponent = bits[0]
    mantissa = int.from_bytes(bits[1:], "big")
    if not mantissa or mantissa & 0x800000:
        raise ValueError("invalid compact target mantissa")
    shift = 8 * (exponent - 3)
    value = mantissa << shift if shift >= 0 else mantissa >> -shift
    if not 0 < value < 1 << 256:
        raise ValueError("compact target is outside 256 bits")
    return value


def coinbase_outputs(tx: bytes) -> list[tuple[int, bytes]]:
    if len(tx) < 4:
        raise ValueError("truncated coinbase transaction")
    overwintered = bool(int.from_bytes(tx[:4], "little") & 0x80000000)
    pos = 8 if overwintered else 4
    inputs, pos = read_varint(tx, pos)
    for _ in range(inputs):
        if pos + 36 > len(tx):
            raise ValueError("truncated coinbase input")
        pos += 36
        script_size, pos = read_varint(tx, pos)
        if pos + script_size + 4 > len(tx):
            raise ValueError("truncated coinbase input script")
        pos += script_size + 4
    count, pos = read_varint(tx, pos)
    outputs = []
    for _ in range(count):
        if pos + 8 > len(tx):
            raise ValueError("truncated coinbase output")
        value = int.from_bytes(tx[pos:pos + 8], "little")
        pos += 8
        script_size, pos = read_varint(tx, pos)
        if pos + script_size > len(tx):
            raise ValueError("truncated coinbase output script")
        outputs.append((value, tx[pos:pos + script_size]))
        pos += script_size
    return outputs


def verify_payout(coinbase_hex: str, address: str) -> None:
    expected = address_script(address)
    paying = [(amount, script) for amount, script in coinbase_outputs(bytes.fromhex(coinbase_hex))
              if amount > 0]
    if not paying or any(script != expected for _, script in paying):
        raise ValueError("block template coinbase does not pay the requested address")


def merkle_root(template: dict) -> bytes:
    hashes = [template["coinbasetxn"]["hash"]]
    hashes.extend(tx["hash"] for tx in template["transactions"])
    level = [bytes.fromhex(value)[::-1] for value in hashes]
    while len(level) > 1:
        level = [sha256d(level[i] + level[min(i + 1, len(level) - 1)])
                 for i in range(0, len(level), 2)]
    return level[0]


def extra_nonce_offset(preimage: bytes) -> int:
    if len(preimage) != 1487 or preimage[140:143] != COMPACT_SOLUTION:
        raise ValueError("unsupported Verus solution size")
    sol = preimage[143:]
    version = int.from_bytes(sol[:4], "little")
    if version not in (7, 8):
        raise ValueError(f"solution version {version} is not VerusHash 2.2 PBaaS")
    headers = sol[5]
    extra_size = int.from_bytes(sol[6:8], "little")
    offset = 143 + 72 + 52 * headers
    if not headers or extra_size < 4 or offset + extra_size > 1472:
        raise ValueError("unsupported PBaaS solution extra data")
    return offset


@dataclass(frozen=True)
class Work:
    template_id: str
    height: int
    previous_hash: str
    solution_version: int
    merged_chains: int
    preimage: bytes
    target_be: bytes
    block_tail: bytes
    extra_offset: int

    def job_file(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".tmp")
        temp.write_bytes(MAGIC + self.target_be + self.preimage)
        os.replace(temp, path)


def build_work(template: dict, payout_address: str | None,
               override_target: str | None = None) -> Work:
    if payout_address:
        verify_payout(template["coinbasetxn"]["data"], payout_address)
    solution = bytes.fromhex(template["solution"])
    if len(solution) != 1344:
        raise ValueError("expected 1344-byte solution")
    nonce = bytes.fromhex(template.get("nonce") or "00" * 32)[::-1]
    if len(nonce) != 32:
        raise ValueError("expected 32-byte header nonce")
    header = (struct.pack("<I", template["version"] & 0xFFFFFFFF)
              + bytes.fromhex(template["previousblockhash"])[::-1]
              + merkle_root(template)
              + bytes.fromhex(template.get("finalsaplingroothash") or "00" * 32)[::-1]
              + struct.pack("<I", template["curtime"])
              + bytes.fromhex(template["bits"])[::-1]
              + nonce)
    if len(header) != 140:
        raise ValueError("malformed block header")
    preimage = header + COMPACT_SOLUTION + solution
    offset = extra_nonce_offset(preimage)
    primary_target = bytes.fromhex(template["target"])
    if len(primary_target) != 32:
        raise ValueError("expected 256-bit target")
    merged_chains = max(0, solution[5] - 1)
    if override_target:
        target = bytes.fromhex(override_target)
    else:
        # submitmergedblock can accept a hash that qualifies only for a PBaaS
        # chain, so scan to the easiest advertised merged target as well.
        merged_bits = template.get("merged_bits") or template.get("mergeminebits")
        merged_target = compact_target(merged_bits) if merged_bits else 0
        target = max(int.from_bytes(primary_target, "big"), merged_target).to_bytes(32, "big")
    if len(target) != 32:
        raise ValueError("expected 256-bit mining target")
    transactions = [bytes.fromhex(template["coinbasetxn"]["data"])]
    transactions.extend(bytes.fromhex(tx["data"]) for tx in template["transactions"])
    block_tail = write_varint(len(transactions)) + b"".join(transactions)
    # A fresh timestamp alone does not invalidate the old job. Verus Datum
    # likewise keeps work until the previous hash, merkle root, bits, or PBaaS
    # solution changes. Refresh the timestamp on a timed rebroadcast below.
    identity_preimage = preimage[:100] + bytes(4) + preimage[104:]
    template_id = hashlib.sha256(identity_preimage + target + block_tail).hexdigest()[:16]
    return Work(template_id, template["height"], template["previousblockhash"],
                int.from_bytes(solution[:4], "little"), merged_chains, preimage, target,
                block_tail, offset)


def check_candidate(work: Work, candidate_hex: str) -> bytes:
    candidate = bytes.fromhex(candidate_hex)
    if len(candidate) != len(work.preimage):
        raise ValueError("candidate has wrong preimage length")
    allowed = set(range(work.extra_offset, work.extra_offset + 4))
    allowed.update(range(1472, 1487))
    if any(a != b and i not in allowed
           for i, (a, b) in enumerate(zip(candidate, work.preimage))):
        raise ValueError("candidate changed fields outside the extra nonce and nonce")
    return candidate


class Rpc:
    def __init__(self, config: dict):
        self.url = config["url"]
        basic = f"{config['user']}:{config['password']}".encode("utf-8")
        self.authorization = "Basic " + base64.b64encode(basic).decode("ascii")

    def call(self, method: str, params: list | None = None):
        body = json.dumps({"jsonrpc": "1.0", "id": "verus-solo-gpu",
                           "method": method, "params": params or []}).encode("utf-8")
        req = request.Request(self.url, data=body,
                              headers={"Authorization": self.authorization,
                                       "Content-Type": "application/json"})
        try:
            with request.urlopen(req, timeout=20) as response:
                result = json.load(response)
        except error.HTTPError as exc:
            body = exc.read(2000).decode("utf-8", "replace")
            raise RuntimeError(f"RPC {method} HTTP {exc.code}: {body}") from exc
        if result.get("error"):
            raise RuntimeError(f"RPC {method}: {result['error']}")
        return result.get("result")


def read_output(process: subprocess.Popen, lines: queue.Queue[str]) -> None:
    assert process.stdout is not None
    for line in process.stdout:
        lines.put(line.rstrip("\r\n"))


def launch_miner(work: Work, executable: Path, mode: str, cpu_threads: int,
                 lines: queue.Queue[str]) -> subprocess.Popen:
    job = ROOT / "jobs" / f"{work.template_id}.bin"
    work.job_file(job)
    start = secrets.randbits(32)
    command = [str(executable), "--verus-solo", str(job), str(start),
               "--mode", mode, "--threads", str(cpu_threads)]
    process = subprocess.Popen(command,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1)
    reader = threading.Thread(target=read_output, args=(process, lines), daemon=True)
    reader.start()
    process.output_thread = reader
    return process


def stop_miner(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def run_fixture(path: Path, executable: Path, target_hex: str,
                mode: str, cpu_threads: int) -> None:
    fixture = json.loads(path.read_text(encoding="utf-8"))
    work = build_work(fixture["template"], None, target_hex)
    lines: queue.Queue[str] = queue.Queue()
    process = launch_miner(work, executable, mode, cpu_threads, lines)
    deadline = time.monotonic() + 60
    try:
        while time.monotonic() < deadline:
            try:
                line = lines.get(timeout=1)
            except queue.Empty:
                if process.poll() is not None:
                    break
                continue
            if line.startswith("FOUND "):
                _, preimage_hex, digest_hex = line.split(" ", 2)
                candidate = check_candidate(work, preimage_hex)
                print(f"FIXTURE_FOUND height={work.height} "
                      f"block_bytes={len(candidate + work.block_tail)} "
                      f"digest_le={digest_hex}")
                return
            print(line)
        raise RuntimeError(f"fixture miner did not find a candidate (exit={process.poll()})")
    finally:
        stop_miner(process)


def run_live(config_path: Path, executable: Path, mode: str,
             cpu_threads: int) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    payout_address = config["payout_address"]
    address_script(payout_address)
    rpc = Rpc(config)
    current: Work | None = None
    miner: subprocess.Popen | None = None
    lines: queue.Queue[str] = queue.Queue()
    last_sync_message = 0.0
    job_started = 0.0
    try:
        while True:
            # Drain a finished child's output before replacing the job. In particular,
            # FOUND is printed just before the GPU process exits.
            if miner is not None and miner.poll() is not None:
                miner.output_thread.join(timeout=2)
            while current is not None:
                try:
                    line = lines.get_nowait()
                except queue.Empty:
                    break
                if not line.startswith("FOUND "):
                    print(line, flush=True)
                    continue
                try:
                    _, preimage_hex, digest_hex = line.split(" ", 2)
                    candidate = check_candidate(current, preimage_hex)
                    if len(digest_hex) != 64:
                        raise ValueError("bad candidate digest")
                    info = rpc.call("getinfo")
                    if int(info.get("blocks", 0)) >= current.height:
                        print("STALE tip advanced before submission", flush=True)
                    else:
                        req = {"capabilities": ["coinbasetxn", "workid", "coinbase/append"],
                               "miningdistribution": {payout_address: 1}}
                        fresh = build_work(rpc.call("getblocktemplate", [req]), payout_address)
                        if fresh.template_id != current.template_id:
                            print("STALE template changed before submission", flush=True)
                        else:
                            method = "submitmergedblock" if current.solution_version > 6 else "submitblock"
                            response = rpc.call(method, [(candidate + current.block_tail).hex()])
                            print(f"SUBMIT height={current.height} response={response!r} "
                                  f"hash={bytes.fromhex(digest_hex)[::-1].hex()}", flush=True)
                except (OSError, RuntimeError, ValueError, KeyError) as exc:
                    print("SUBMIT_ERROR", exc, flush=True)
                stop_miner(miner)
                miner = None
                current = None
                break

            try:
                info = rpc.call("getinfo")
                blocks = int(info.get("blocks", 0))
                longest = int(info.get("longestchain", blocks))
                if longest - blocks > 2 or info.get("connections", 0) == 0:
                    stop_miner(miner)
                    miner = None
                    current = None
                    if time.monotonic() - last_sync_message > 15:
                        print(f"SYNC blocks={blocks} longest={longest} peers={info.get('connections')}",
                              flush=True)
                        last_sync_message = time.monotonic()
                    time.sleep(3)
                    continue
                req = {"capabilities": ["coinbasetxn", "workid", "coinbase/append"],
                       "miningdistribution": {payout_address: 1}}
                work = build_work(rpc.call("getblocktemplate", [req]), payout_address)
            except (OSError, RuntimeError, ValueError, KeyError) as exc:
                print("NODE", exc, flush=True)
                stop_miner(miner)
                miner = None
                current = None
                time.sleep(5)
                continue

            expired = (time.monotonic() - job_started >= 55 and
                       current is not None and work.preimage[100:104] != current.preimage[100:104])
            if (current is None or work.template_id != current.template_id or
                    miner is None or miner.poll() is not None or expired):
                stop_miner(miner)
                lines = queue.Queue()
                current = work
                miner = launch_miner(work, executable, mode, cpu_threads, lines)
                job_started = time.monotonic()
                print(f"JOB height={work.height} id={work.template_id} "
                      f"solution_v={work.solution_version} merged_chains={work.merged_chains} "
                      f"payout={payout_address}", flush=True)
            time.sleep(2)
    except KeyboardInterrupt:
        print("Stopping GPU miner", flush=True)
    finally:
        stop_miner(miner)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "private" / "rpc.json")
    parser.add_argument("--miner", "--gpu-exe", dest="miner", type=Path,
                        default=ROOT / "solo_gpu.exe")
    parser.add_argument("--mode", choices=("gpu", "cpu", "hybrid"), default="gpu")
    parser.add_argument("--cpu-threads", type=int,
                        default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--target", default="0000ffff" + "ff" * 28,
                        help="fixture-only 256-bit target, big endian hex")
    args = parser.parse_args()
    if not 1 <= args.cpu_threads <= 256:
        parser.error("--cpu-threads must be 1..256")
    if not args.miner.exists():
        raise SystemExit(f"miner missing: {args.miner}")
    if args.fixture:
        run_fixture(args.fixture, args.miner, args.target,
                    args.mode, args.cpu_threads)
    else:
        run_live(args.config, args.miner, args.mode, args.cpu_threads)


if __name__ == "__main__":
    main()
