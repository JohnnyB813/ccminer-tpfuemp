// Solo VerusHash 2.2 GPU miner. Input is a node block template prepared by
// solo_bridge.py; output is an unchanged block header/solution except for the
// four-byte PBaaS solution extra nonce and the last 15 solution nonce bytes.
// This is a separate solo entry point: a Stratum pool normally owns the first
// seven nonce bytes, so ccminer's ordinary -a verus path cannot select them.
#include <cuda_runtime.h>
#include "verus/verus_warp.cuh"
#include "verus/verus_host.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>

struct BenchJob { uint32_t cb[16]; };

static uint32_t le32(const uint8_t *p) {
    return uint32_t(p[0]) | uint32_t(p[1]) << 8 |
           uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24;
}

__device__ __forceinline__ void load_cb(uint32_t cb[16], const BenchJob *job,
                                         uint32_t n) {
    for (int k = 0; k < 16; ++k) cb[k] = job->cb[k];
    cb[10] |= n << 24;
    cb[11] |= n >> 8;
}

static void check(cudaError_t status, const char *what) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
}

template<class T> struct Device {
    T *p = nullptr;
    explicit Device(size_t n) { check(cudaMalloc(&p, n * sizeof(T)), "cudaMalloc"); }
    ~Device() { cudaFree(p); }
};

constexpr size_t PREIMAGE_BYTES = 1487;
constexpr size_t PREFIX_BYTES = 1472;
constexpr uint32_t BATCH = 1u << 20;
constexpr char JOB_MAGIC[8] = {'V','R','S','O','L','O','1','\0'};

__global__ void __launch_bounds__(1024, 1) solo_kernel(
    const uint32_t *gK, const uint32_t *gT0, const BenchJob *job,
    uint64_t inter, uint32_t first, uint32_t count, uint32_t target7,
    uint64_t *found, uint32_t *w7out)
{
    extern __shared__ __align__(16) uint8_t smem[];
    uint32_t *sT = (uint32_t *)smem;
    uint32_t *sK = sT + VW_T_WORDS;
    uint8_t *wsm = (uint8_t *)(sK + VP_KEY_SLOTS * 4);
    for (int i = threadIdx.x; i < VW_T_WORDS; i += blockDim.x) {
        const uint32_t v = gT0[i & 255], r = 8 * (i >> 8);
        sT[i] = r ? (v << r) | (v >> (32 - r)) : v;
    }
    for (int i = threadIdx.x; i < VP_KEY_SLOTS * 4; i += blockDim.x) sK[i] = gK[i];
    __syncthreads();
    uint8_t *wbase = wsm + (threadIdx.x >> 5) * VW_WARP_SMEM;
    uint32_t *ov = (uint32_t *)wbase;
    uint16_t *idx = (uint16_t *)(wbase + VP_OVL_SLOTS * 16);
    uint8_t *map = wbase + VP_OVL_SLOTS * 18;
    vw_key o;
    vw_key_init(&o, sK, map, idx, ov);
    const uint32_t gw = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const uint32_t nw = (gridDim.x * blockDim.x) >> 5;
    for (uint32_t i = gw; i < count; i += nw) {
        const uint32_t n = first + i;
        uint32_t cb[16];
        uint8_t out[32];
        load_cb(cb, job, n);
        // The selected nonce class returns the pristine key and the same
        // CLHash intermediate after 32 rounds. The host checked that claim
        // before launch, so each warp only needs the keyed Haraka tail.
        vp_hash_tail(out, 0, cb, inter, &o, sT, (vp_stats *)0);
        if (vw_lane() == 0) {
            const uint32_t w7 = (uint32_t)out[28] | (uint32_t)out[29] << 8 |
                                (uint32_t)out[30] << 16 | (uint32_t)out[31] << 24;
            if (w7out) w7out[i] = w7;
            if (found && w7 <= target7)
                atomicMin(reinterpret_cast<unsigned long long *>(found),
                          static_cast<unsigned long long>(n));
        }
    }
}

static void put32(uint8_t *p, uint32_t x) {
    for (int i = 0; i < 4; ++i) p[i] = uint8_t(x >> (8 * i));
}
static bool eligible(uint64_t s) {
    // case 0x14, one inner CLMUL pass, and an odd pbuf selector. The two
    // 9-bit key indices must alias: the final store then restores the key.
    // The active branch reads the nonce-dependent pbuf entry at rounds=0.
    return (s & 0x1c) == 0x14 && (s >> 61) == 0 &&
           (s & (1ULL << 28)) && (s & 1) &&
           ((s >> 5) & 511) == ((s >> 32) & 511);
}
static bool meets_target(const uint8_t hash[32], const uint8_t target_be[32]) {
    for (int i = 31; i >= 0; --i) {
        const uint8_t t = target_be[31 - i];
        if (hash[i] < t) return true;
        if (hash[i] > t) return false;
    }
    return true;
}
static void print_hex(const uint8_t *p, size_t n) {
    static const char digits[] = "0123456789abcdef";
    for (size_t i = 0; i < n; ++i) {
        std::cout.put(digits[p[i] >> 4]);
        std::cout.put(digits[p[i] & 15]);
    }
}
static void prepare_job(BenchJob &job, const uint8_t half[64], const uint8_t nonce[15]) {
    std::memcpy(job.cb, half, 64);
    std::memcpy(reinterpret_cast<uint8_t *>(job.cb) + 32, nonce, 15);
    std::memset(reinterpret_cast<uint8_t *>(job.cb) + 43, 0, 4);
}

int main(int argc, char **argv) {
    try {
        if (argc < 2 || argc > 3)
            throw std::runtime_error("usage: solo_gpu.exe job.bin [extra_nonce_start]");
        std::ifstream in(argv[1], std::ios::binary);
        if (!in) throw std::runtime_error("cannot open job file");
        char magic[8];
        uint8_t target_be[32];
        std::array<uint8_t, PREIMAGE_BYTES> raw{};
        in.read(magic, 8);
        in.read(reinterpret_cast<char *>(target_be), 32);
        in.read(reinterpret_cast<char *>(raw.data()), raw.size());
        if (!in || in.peek() != EOF || std::memcmp(magic, JOB_MAGIC, 8))
            throw std::runtime_error("invalid job file");
        if (raw[140] != 0xfd || raw[141] != 0x40 || raw[142] != 0x05)
            throw std::runtime_error("expected 1344-byte PBaaS solution");
        const uint8_t *sol = raw.data() + 143;
        const uint32_t solution_version = le32(sol);
        if (solution_version < 7 || solution_version > 8)
            throw std::runtime_error("this miner supports PBaaS VerusHash 2.2 solution v7/v8 only");
        const uint32_t headers = sol[5];
        const uint32_t extra_size = uint32_t(sol[6]) | uint32_t(sol[7]) << 8;
        const size_t extra_offset = 143 + 72 + 52 * headers;
        if (!headers || extra_size < 4 || extra_offset + extra_size > PREFIX_BYTES)
            throw std::runtime_error("unsupported PBaaS extra data layout");

        std::array<uint8_t, PREIMAGE_BYTES> canonical = raw;
        verus_host_clear_noncanonical(canonical.data());
        alignas(16) uint8_t keybuf[VERUS_KEYBUF_BYTES];
        uint8_t half[64];
        const uint32_t scan_start = argc == 3 ? uint32_t(std::strtoul(argv[2], nullptr, 0)) : 0;
        uint32_t chosen = 0;
        uint64_t selector = 0;
        bool found_seed = false;
        // The PBaaS solution extra nonce is inside the 1472-byte prologue.
        // Rolling it changes the job key and lets solo mining choose a
        // selector with the proven key-index collision.
        for (uint32_t attempt = 0; attempt < 2000000; ++attempt) {
            chosen = scan_start + attempt;
            put32(raw.data() + extra_offset, chosen);
            put32(canonical.data() + extra_offset, chosen);
            if (!verus_host_prologue(canonical.data(), PREFIX_BYTES, half, keybuf))
                throw std::runtime_error("VerusHash prologue failed");
            selector = vp_lo(reinterpret_cast<const vp128 *>(keybuf)[513]);
            if (eligible(selector)) { found_seed = true; break; }
        }
        if (!found_seed) {
            std::cout << "NO_ELIGIBLE 2000000" << std::endl;
            return 2;
        }
        const auto *key = reinterpret_cast<const vp128 *>(keybuf);
        const uint32_t pr = uint32_t(selector >> 5) & 511;
        const uint64_t b0 = uint64_t(le32(half)) | uint64_t(le32(half + 4)) << 32;
        // For selector&3 == 1, the CLMUL input uses b0 XOR nonce; for 3 it
        // uses nonce. XORing the low 64 bits with key[pr] makes that input's
        // low half zero. CLMUL_0x10 is then zero for all 2^56 upper nonces.
        // With aliasing stores, neither the accumulator nor key changes in
        // any of the 32 rounds. vp_finalize(key[513]) is job-constant.
        const uint64_t fixed = vp_lo(key[pr]) ^ ((selector & 3) == 1 ? b0 : 0);
        for (int i = 0; i < 8; ++i) {
            raw[PREFIX_BYTES + i] = uint8_t(fixed >> (8 * i));
            canonical[PREFIX_BYTES + i] = raw[PREFIX_BYTES + i];
        }
        const uint64_t inter = vp_finalize(key[513]);
        uint32_t table[256];
        vp_build_t0(table);
        std::mt19937_64 rng(uint64_t(chosen) ^ 0xd17ca62704aa19ceULL);
        for (int trial = 0; trial < 64; ++trial) {
            uint8_t nonce[15], fast[32], reference[32];
            std::memcpy(nonce, raw.data() + PREFIX_BYTES, 15);
            for (int j = 8; j < 15; ++j) nonce[j] = uint8_t(rng());
            uint32_t cb[16];
            std::memcpy(cb, half, 64);
            std::memcpy(reinterpret_cast<uint8_t *>(cb) + 32, nonce, 15);
            vp_ovl full_key, fast_key;
            vp_ovl_init(&full_key, key);
            vp_ovl_init(&fast_key, key);
            if (vp_clhash(&full_key, cb, table, nullptr) != inter)
                throw std::runtime_error("selected nonce class failed CPU CLHash check");
            vp_hash_tail(fast, 1, cb, inter, &fast_key, table, nullptr);
            verus_host_hash(reference, half, nonce, keybuf, 1);
            if (std::memcmp(fast, reference, 32))
                throw std::runtime_error("selected nonce class failed CPU digest check");
        }

        cudaDeviceProp prop{};
        check(cudaGetDeviceProperties(&prop, 0), "GPU device");
        const int tpb = 1024, blocks = prop.multiProcessorCount;
        const size_t smem = 1024u * vw_aes_tables(prop.major) + VP_KEY_SLOTS * 16u +
                            size_t(tpb / 32) * VW_WARP_SMEM;
        check(cudaFuncSetAttribute(solo_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   int(smem)), "GPU shared memory");
        Device<uint32_t> dK(VP_KEY_SLOTS * 4), dT(256), dW7(128);
        Device<uint64_t> dFound(1);
        Device<BenchJob> dJob(1);
        check(cudaMemcpy(dK.p, keybuf, VP_KEY_SLOTS * 16, cudaMemcpyHostToDevice), "key upload");
        check(cudaMemcpy(dT.p, table, sizeof(table), cudaMemcpyHostToDevice), "table upload");
        const uint32_t target7 = uint32_t(target_be[0]) << 24 | uint32_t(target_be[1]) << 16 |
                                 uint32_t(target_be[2]) << 8 | uint32_t(target_be[3]);
        const uint32_t outer_start = uint32_t(rng()) & 0xffffffu;
        std::cout << "READY GPU=" << prop.name << " selector=0x" << std::hex << selector
                  << " extra_nonce=0x" << chosen << " target7=0x" << target7 << std::dec
                  << " cpu_checks=64" << std::endl;
        uint64_t total = 0, since_report = 0;
        auto last_report = std::chrono::steady_clock::now();
        for (uint32_t outer_count = 0; outer_count < (1u << 24); ++outer_count) {
            const uint32_t outer = (outer_start + outer_count) & 0xffffffu;
            for (int i = 0; i < 3; ++i) {
                raw[PREFIX_BYTES + 8 + i] = uint8_t(outer >> (8 * i));
                canonical[PREFIX_BYTES + 8 + i] = raw[PREFIX_BYTES + 8 + i];
            }
            uint8_t *nonce = raw.data() + PREFIX_BYTES;
            BenchJob job{};
            prepare_job(job, half, nonce);
            check(cudaMemcpy(dJob.p, &job, sizeof job, cudaMemcpyHostToDevice), "job upload");
            if (outer_count == 0) {
                check(cudaMemset(dFound.p, 0xff, 8), "GPU test init");
                solo_kernel<<<blocks, tpb, smem>>>(dK.p, dT.p, dJob.p, inter, 0, 128,
                                                  0, nullptr, dW7.p);
                check(cudaGetLastError(), "GPU test launch");
                std::array<uint32_t, 128> words{};
                check(cudaMemcpy(words.data(), dW7.p, words.size() * 4,
                                 cudaMemcpyDeviceToHost), "GPU test copy");
                for (uint32_t i = 0; i < words.size(); ++i) {
                    uint8_t n[15], digest[32];
                    std::memcpy(n, nonce, 15);
                    put32(n + 11, i);
                    verus_host_hash(digest, half, n, keybuf, 1);
                    if (words[i] != le32(digest + 28))
                        throw std::runtime_error("GPU and CPU disagree on selected work");
                }
                std::cout << "GPU_CHECK 128/128" << std::endl;
            }
            for (uint64_t next = 0; next < (1ULL << 32); ) {
                const uint32_t count = uint32_t(std::min<uint64_t>(BATCH, (1ULL << 32) - next));
                const uint64_t end = next + count;
                while (next < end) {
                    const uint32_t first = uint32_t(next);
                    const uint32_t span = uint32_t(end - next);
                    check(cudaMemset(dFound.p, 0xff, 8), "candidate reset");
                    solo_kernel<<<blocks, tpb, smem>>>(dK.p, dT.p, dJob.p, inter, first, span,
                                                      target7, dFound.p, nullptr);
                    check(cudaGetLastError(), "mine launch");
                    uint64_t candidate = UINT64_MAX;
                    check(cudaMemcpy(&candidate, dFound.p, 8, cudaMemcpyDeviceToHost),
                          "candidate read");
                    if (candidate == UINT64_MAX) {
                        total += span;
                        since_report += span;
                        next = end;
                    } else {
                        if (uint64_t(candidate) < next || uint64_t(candidate) >= end)
                            throw std::runtime_error("GPU candidate outside batch");
                        const uint64_t checked = uint64_t(candidate) - next + 1;
                        total += checked;
                        since_report += checked;
                        next = uint64_t(candidate) + 1;
                        put32(raw.data() + PREFIX_BYTES + 11, candidate);
                        put32(canonical.data() + PREFIX_BYTES + 11, candidate);
                        uint8_t digest[32];
                        if (!verus_host_full(digest, canonical.data(), canonical.size()))
                            throw std::runtime_error("full CPU candidate hash failed");
                        if (meets_target(digest, target_be)) {
                            std::cout << "FOUND ";
                            print_hex(raw.data(), raw.size());
                            std::cout << ' ';
                            print_hex(digest, 32);
                            std::cout << std::endl;
                            return 0;
                        }
                    }
                }
                const auto now = std::chrono::steady_clock::now();
                const double seconds = std::chrono::duration<double>(now - last_report).count();
                if (seconds >= 5.0) {
                    std::cout << "RATE " << std::fixed << std::setprecision(2)
                              << (since_report / seconds / 1e6) << " MH/s total="
                              << total << std::endl;
                    last_report = now;
                    since_report = 0;
                }
            }
        }
        std::cout << "DONE nonce space exhausted" << std::endl;
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "ERROR " << e.what() << std::endl;
        return 1;
    }
}
