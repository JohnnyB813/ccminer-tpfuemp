# Solo VerusHash 2.2 shortcut miner

This optional solo miner uses this repository's `algos/verus` implementation.
It reads a VerusCoin/PBaaS `getblocktemplate` from a local `verusd`, selects a
four-byte solution extra nonce that gives a verified `case 0x14` fixed point,
then tests the remaining 56 nonce bits on the GPU. The 32-round CLHash result
is precomputed once per selected job; the GPU runs the keyed Haraka tail. Every
GPU candidate is fully rehashed on the CPU before submission.

This is a separate executable from `ccminer -a verus`. On an ordinary Stratum
pool job the pool supplies the first seven of the 15 variable nonce bytes;
those bytes cannot generally be chosen to satisfy the shortcut. The normal
pool miner remains a full VerusHash 2.2 miner.

## Build and run on Windows

Install Visual Studio 2022 C++ tools, an NVIDIA driver, and the CUDA toolkit.
The default build targets `sm_89` (RTX 40 series); set `GPU_ARCH` to another
supported architecture before running `build_gpu.cmd` for a different GPU.

```powershell
cd tools/verus-solo
./build_gpu.cmd
```

Create `private/rpc.json` with your node's loopback RPC credentials and your
own Verus R-address:

```json
{
  "url": "http://127.0.0.1:27486/",
  "user": "replace-with-rpc-user",
  "password": "replace-with-rpc-password",
  "payout_address": "replace-with-your-R-address"
}
```

The bridge requires a synced node with peers, requests templates with the
specified payout address, and checks the coinbase outputs before mining.
Run it with Python 3.10 or newer:

```powershell
py solo_bridge.py --config private/rpc.json
```

It polls for new templates and uses `submitmergedblock` for PBaaS solutions.
If the node advertises a merged target, the miner also checks that easier
target. Other PBaaS chain rewards depend on those chains' node registrations
and payout settings. The code accepts solution versions 7 and 8 only and
refuses v2.3 work.

## Verify before live use

`test/fixtures.json` contains a real Verus `getblocktemplate` fixture. The
test below starts a fake local JSON-RPC node, checks the configured payout,
feeds a job to the CUDA miner, checks the GPU against the CPU reference, and
verifies the complete `submitmergedblock` payload. Its test target is easier
than the real network target; this is an offline end-to-end test.

```powershell
py test_bridge_rpc.py
```

The CUDA miner also checks 64 selected nonces against the full CPU VerusHash
and 128 GPU target words against the CPU before it starts its mining loop.
The source comments at `eligible()` and the low-64-bit nonce construction
explain why the shortcut is valid for selected jobs. The bridge and test use
only Python's standard library. Credentials and generated jobs stay under
ignored `private/` and `jobs/` directories.
