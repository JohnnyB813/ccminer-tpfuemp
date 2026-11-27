#!/usr/bin/env bash
# Build the same solo hashing source as ccminer on Linux. The CPU object flags
# also go to nvcc's host compiler because the VerusHash reference uses AES-NI.
set -euo pipefail
cd "$(dirname "$0")"

GPU_ARCH="${GPU_ARCH:-sm_89}"
mkdir -p build-linux
nvcc -O3 -std=c++17 -arch="$GPU_ARCH" --use_fast_math \
    -DVERUS_SOLO_STANDALONE \
    -Xcompiler=-maes,-mpclmul,-mssse3,-mavx2,-pthread \
    -c solo_gpu.cu -o build-linux/solo_gpu.o

for source in verus_host haraka haraka_portable verus_clhash; do
    gcc -O3 -maes -mpclmul -mssse3 -mavx2 -I../../algos/verus \
        -c "../../algos/verus/${source}.c" -o "build-linux/${source}.o"
done

g++ -O3 -o ccminer-verus-solo-linux-x86_64 \
    build-linux/solo_gpu.o build-linux/verus_host.o build-linux/haraka.o \
    build-linux/haraka_portable.o build-linux/verus_clhash.o \
    -L"${CUDA_HOME:-/usr/local/cuda}/lib64" -lcudart_static -ldl -lrt -pthread
