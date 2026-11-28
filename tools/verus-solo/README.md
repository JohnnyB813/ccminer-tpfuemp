# VerusHash 2.2 solo mining in ccminer

One `ccminer` binary supports `--mode gpu`, `--mode cpu`, and `--mode hybrid`
for node-prepared VerusCoin/PBaaS solo jobs. The Python bridge handles
`getblocktemplate`, refreshes work, and submits a validated candidate through
`submitmergedblock`; the same miner executable does all hashing in every mode.
The regular `ccminer -a verus` Stratum pool path remains available.

The miner selects a four-byte PBaaS solution extra nonce giving the verified
`case 0x14` fixed point. It computes the 276-step key once for that job and
shares its pristine key and fixed CLHash result across CPU and GPU searches.
On CPUs with VAES and AVX2, CPU-only and hybrid mode process two nonce suffixes
per Haraka512 call using 256-bit VAES. The miner checks 64 nonce pairs against
the scalar finalizer at job start, and fully rehashes every potential block.
Other CPUs use the existing scalar AES-NI finalizer. Set VERUS_SOLO_VAES=0
to compare scalar speed, or VERUS_SOLO_VAES=1 to require VAES.
In hybrid mode the GPU scans nonce suffix bit 55 = 0 and CPU threads scan bit
55 = 1, so they cannot repeat each other's nonces. Each potential block is
fully rehashed on the CPU before the bridge submits it. The miner rejects
solution versions other than 7 and 8; these are the VerusHash 2.2 formats
implemented here.

## Build

The main Windows `ccminer.vcxproj` includes the solo CUDA source. Build the
normal Release x64 project. It defaults to CUDA 11.8; if using a different
toolkit, pass `CudaBuildVersion` and a supported `DevArch` to MSBuild. For
example, on an RTX 4060 with CUDA 13.4:

```powershell
msbuild ccminer.vcxproj /p:Configuration=Release /p:Platform=x64 /p:CudaBuildVersion=13.4 /p:DevArch=89
```

`build_gpu.cmd` is a fast standalone build of the **same mining source**. It
produces `solo_gpu.exe`, whose three modes have the same job protocol. The
release packages rename this standalone binary to `ccminer-verus-solo` (with
`.exe` on Windows). It supports VerusHash 2.2 solo mining; the full `ccminer`
project also compiles this source and retains its existing pool algorithms.

On Linux, run `tools/verus-solo/build_linux.sh` with the CUDA toolkit, GCC,
and G++ installed. It produces `ccminer-verus-solo-linux-x86_64`. Set
`GPU_ARCH` to the NVIDIA architecture you intend to run; the default
`sm_89` targets RTX 40 series. CPU mode can run without an NVIDIA device.
The Linux release binary is built on Ubuntu 22.04 with CUDA 12.8.1 and needs
the usual Linux C++ runtime. The Windows release binary is built with CUDA
13.4 and the static MSVC runtime.

## Run with your solo node

Create `private/rpc.json` with your node RPC URL, credentials, and Verus
R-address. The bridge checks coinbase outputs against this address:

```json
{
  "url": "http://127.0.0.1:27486/",
  "user": "replace-with-rpc-user",
  "password": "replace-with-rpc-password",
  "payout_address": "replace-with-your-R-address"
}
```

Run Python 3.10 or newer and point `--miner` to the built ccminer binary.
If you passed `DevArch=89`, replace `Release` below with `Release-sm89`:

```powershell
py solo_bridge.py --config private/rpc.json --miner ..\..\x64\Release\ccminer.exe --mode gpu
py solo_bridge.py --config private/rpc.json --miner ..\..\x64\Release\ccminer.exe --mode cpu --cpu-threads 14
py solo_bridge.py --config private/rpc.json --miner ..\..\x64\Release\ccminer.exe --mode hybrid --cpu-threads 14
```

CPU mode enters before ccminer's normal CUDA device discovery. The bridge
requires a synced node with peers and requests templates with the configured
payout address. If a merged target is advertised, the bridge also checks it.
Other PBaaS chain rewards depend on those chains' node registrations and
payout settings. A Stratum pool normally owns the first seven nonce bytes,
so this selected nonce shortcut does not apply to ordinary pool work.

## Verify

`test/fixtures.json` is a real `getblocktemplate` fixture. For a fast local
check, build the standalone source and run all three fixture modes:

```powershell
./build_gpu.cmd
py solo_bridge.py --fixture test/fixtures.json --mode gpu
py solo_bridge.py --fixture test/fixtures.json --mode cpu --cpu-threads 4
py solo_bridge.py --fixture test/fixtures.json --mode hybrid --cpu-threads 4
py test_bridge_rpc.py --mode gpu
py test_bridge_rpc.py --mode cpu
py test_bridge_rpc.py --mode hybrid
```

The fake RPC test checks a complete `submitmergedblock` payload against the
fixture. Before mining, the binary checks 64 selected nonces against the full
host VerusHash; GPU modes also compare 128 GPU target words with the host.
Run py bench_cpu_finalizer.py --threads 4 to compare scalar and VAES2 CPU
scan rates on the same saved job. It uses an all-zero target, reports two
five-second samples for each path, and terminates both miners.
Generated jobs and RPC credentials stay under ignored `jobs/` and `private/`.
