# hoohash - HoohashV110 (PEPEPOW)

## Files

- `hoohash.cu` - scan driver (`scanhash_hoohash`, `free_hoohash`).
- `cuda_hoohashv110.cu` - device kernels, launchers and the init self-test.
- `hoohash_device.cuh` - matrix generation, the transforms and the branch predicate;
  BLAKE3 comes from the shared `cuda/blake3_device.cuh`.

## Consensus-critical strict floating point

`cuda_hoohashv110.cu` **must** be compiled with strict FP so the on-device
transcendentals match the consensus reference (glibc libm) bit-for-bit. Both build
systems carry this per file; do not "clean it up":

- MSBuild: `FastMath=false`, `FmadOptimization=false`,
  `--fmad=false --prec-div=true --prec-sqrt=true --ftz=false`.
- Autotools: dedicated rule `algos/hoohash/cuda_hoohashv110.o` with
  `--maxrregcount=128 --fmad=false --prec-div=true --prec-sqrt=true --ftz=false`.

There is no CPU re-verify of shares (a non-glibc libm would reject valid ones), so
the fail-closed init self-test (a real block's published digest) is the consensus
guard. Per `docs/coding-guideline.md` it runs on every target arch, since libdevice
code is compiled per `sm_*`.

The mining kernel lets each lane walk its own position in the 64x64 scan and evaluates
the expensive transform for all waiting lanes together (`HOO_WAIT_K`, tuned per arch);
results are bit-identical to the reference.
