// Solo VerusHash 2.2 miner. A single selected job and key feed the GPU, CPU,
// or both. The bridge supplies node templates; ordinary Stratum mining keeps
// using ccminer's existing -a verus path because pools own nonce bytes.
#include <cuda_runtime.h>
#include "../../algos/verus/verus_warp.cuh"
#include "../../algos/verus/verus_host.h"
#include "solo_vaes2.h"
extern "C" {
#include "../../algos/verus/haraka.h"
}

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

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
// A larger launch amortizes the result reset and readback. On an RTX 4060
// this is a short kernel while preserving prompt job cancellation.
constexpr uint32_t BATCH = 1u << 22;
constexpr char JOB_MAGIC[8] = {'V','R','S','O','L','O','1','\0'};

__global__ void __launch_bounds__(512, 3) solo_kernel(
    const uint32_t *gK, const uint32_t *gT0, const BenchJob *job,
    uint64_t inter, uint32_t first, uint32_t count, uint32_t target7,
    uint64_t *found, uint32_t *w7out)
{
    extern __shared__ __align__(16) uint8_t smem[];
    uint32_t *sT = (uint32_t *)smem;
    uint32_t *sK = sT + VW_T_WORDS;
    for (int i = threadIdx.x; i < VW_T_WORDS; i += blockDim.x) {
        const uint32_t v = gT0[i & 255], r = 8 * (i >> 8);
        sT[i] = r ? (v << r) | (v >> (32 - r)) : v;
    }
    for (int i = threadIdx.x; i < VP_KEY_SLOTS * 4; i += blockDim.x) sK[i] = gK[i];
    __syncthreads();
    vw_pristine_key o{sK};
    const uint32_t gw = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const uint32_t nw = (gridDim.x * blockDim.x) >> 5;
    const uint32_t half = (threadIdx.x >> 4) & 1;
    const uint32_t pairs = (count + 1) >> 1;
    for (uint32_t pair = gw; pair < pairs; pair += nw) {
        // The Haraka warp already holds two copies of the four-state layout.
        // Give the two 16-lane halves different nonces. The final shuffle
        // stays inside each half, and the odd last pair is safely padded.
        const uint32_t i = pair * 2 + half;
        const uint32_t n = first + i;
        uint32_t cb[16];
        uint8_t out[32];
        load_cb(cb, job, n);
        // The selected nonce class returns the pristine key and the same
        // CLHash intermediate after 32 rounds. The host checked that claim
        // before launch, so each warp only needs the keyed Haraka tail.
        vp_hash_tail(out, 0, cb, inter, &o, sT, (vp_stats *)0);
        if ((vw_lane() & 15) == 0 && i < count) {
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

struct SoloResult {
    std::atomic<bool> stop{false};
    std::mutex mutex;
    bool found = false;
    std::array<uint8_t, PREIMAGE_BYTES> preimage{};
    std::array<uint8_t, 32> digest{};
    std::string worker;
    std::exception_ptr error;

    void publish(const char *source, const uint8_t *raw, const uint8_t hash[32]) {
        std::lock_guard<std::mutex> lock(mutex);
        if (found) return;
        std::memcpy(preimage.data(), raw, PREIMAGE_BYTES);
        std::memcpy(digest.data(), hash, 32);
        worker = source;
        found = true;
        stop.store(true, std::memory_order_relaxed);
    }
    void fail(std::exception_ptr cause) {
        std::lock_guard<std::mutex> lock(mutex);
        if (!error) error = cause;
        stop.store(true, std::memory_order_relaxed);
    }
};

// Each CPU thread owns its Haraka input. The key and CLHash intermediate are
// immutable after the selected class passes the 64 full-hash checks above.
// In hybrid mode, suffix bit 55 is one here and zero on the GPU.
struct CpuSearch {
    SoloResult &result;
    std::atomic<uint64_t> next{0}, done{0};
    std::atomic<unsigned> active{0};
    std::array<uint8_t, PREIMAGE_BYTES> base_raw{}, base_canonical{};
    std::vector<std::thread> workers;
    explicit CpuSearch(SoloResult &r) : result(r) {}
    ~CpuSearch() { result.stop.store(true); join(); }
    void join() {
        for (auto &worker : workers) if (worker.joinable()) worker.join();
    }
    void start(unsigned threads, bool hybrid, const uint8_t half[64],
               const uint8_t *keybuf, uint64_t inter,
               const std::array<uint8_t, PREIMAGE_BYTES> &raw,
               const std::array<uint8_t, PREIMAGE_BYTES> &canonical,
               const uint8_t target_be[32], uint32_t target7, bool use_vaes) {
        const uint64_t space = hybrid ? 1ULL << 55 : 1ULL << 56;
        const uint64_t offset = std::mt19937_64(std::random_device{}())() & (space - 1);
        base_raw = raw;
        base_canonical = canonical;
        active.store(threads);
        for (unsigned t = 0; t < threads; ++t) {
            workers.emplace_back([this, space, offset, hybrid, half, keybuf, inter,
                                  target_be, target7, use_vaes] {
                try {
                    alignas(16) uint8_t input[64], hash[32];
                    std::memcpy(input, half, 64);
                    std::memcpy(input + 32, base_raw.data() + PREFIX_BYTES, 15);
                    input[47] = uint8_t(inter);
                    for (int j = 0; j < 16; ++j)
                        input[48 + j] = uint8_t(inter >> (8 * ((j + 1) & 7)));
                    const auto *round_key = reinterpret_cast<const u128 *>(keybuf) +
                                            (inter & 511);
                    std::unique_ptr<void, void (*)(void *)> vaes(nullptr,
                                                                 verus_solo_vaes2_destroy);
                    if (use_vaes)
                        vaes.reset(verus_solo_vaes2_create(input, round_key));
                    auto verify_word = [&](uint64_t hi, uint32_t word) {
                        if (word > target7) return;
                        auto candidate = base_raw, verify = base_canonical;
                        for (int j = 0; j < 7; ++j) {
                            candidate[PREFIX_BYTES + 8 + j] = uint8_t(hi >> (8 * j));
                            verify[PREFIX_BYTES + 8 + j] = uint8_t(hi >> (8 * j));
                        }
                        uint8_t full[32];
                        if (!verus_host_full(full, verify.data(), verify.size()))
                            throw std::runtime_error("CPU candidate full hash failed");
                        if (meets_target(full, target_be))
                            result.publish("CPU", candidate.data(), full);
                    };
                    while (!result.stop.load(std::memory_order_relaxed)) {
                        const uint64_t first = next.fetch_add(4096, std::memory_order_relaxed);
                        if (first >= space) break;
                        const uint64_t end = std::min(first + 4096, space);
                        uint64_t processed = 0;
                        for (uint64_t i = first; i < end &&
                             !result.stop.load(std::memory_order_relaxed); ) {
                            const uint64_t hi0 = ((offset + i) & (space - 1)) |
                                                 (hybrid ? 1ULL << 55 : 0);
                            if (use_vaes && i + 1 < end) {
                                const uint64_t hi1 = ((offset + i + 1) & (space - 1)) |
                                                     (hybrid ? 1ULL << 55 : 0);
                                uint32_t words[2];
                                verus_solo_vaes2_words(vaes.get(), hi0, hi1, words);
                                processed += 2;
                                i += 2;
                                verify_word(hi0, words[0]);
                                if (!result.stop.load(std::memory_order_relaxed))
                                    verify_word(hi1, words[1]);
                            } else {
                                for (int j = 0; j < 7; ++j)
                                    input[40 + j] = uint8_t(hi0 >> (8 * j));
                                haraka512_keyed(hash, input, round_key);
                                ++processed;
                                ++i;
                                verify_word(hi0, le32(hash + 28));
                            }
                        }
                        done.fetch_add(processed, std::memory_order_relaxed);
                    }
                } catch (...) { result.fail(std::current_exception()); }
                active.fetch_sub(1);
            });
        }
    }
};

static bool emit_result(const SoloResult &result) {
    if (!result.found) return false;
    std::cout << "WINNER " << result.worker << std::endl;
    std::cout << "FOUND ";
    print_hex(result.preimage.data(), result.preimage.size());
    std::cout << ' ';
    print_hex(result.digest.data(), result.digest.size());
    std::cout << std::endl;
    return true;
}

int verus_solo_main(int argc, char **argv) {
    try {
        if (argc < 2)
            throw std::runtime_error("usage: ccminer --verus-solo job.bin [extra_nonce_start] [--mode gpu|cpu|hybrid] [--threads N]");
        std::string mode = "gpu";
        const unsigned cores = std::thread::hardware_concurrency();
        unsigned cpu_threads = cores > 2 ? cores - 2 : 1;
        uint32_t scan_start = 0;
        bool seed_set = false;
        for (int i = 2; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--mode" && i + 1 < argc) {
                mode = argv[++i];
                if (mode != "gpu" && mode != "cpu" && mode != "hybrid")
                    throw std::runtime_error("mode must be gpu, cpu, or hybrid");
            } else if (arg == "--threads" && i + 1 < argc) {
                cpu_threads = unsigned(std::stoul(argv[++i]));
                if (!cpu_threads || cpu_threads > 256)
                    throw std::runtime_error("threads must be 1..256");
            } else if (!seed_set && !arg.empty() && arg[0] != '-') {
                scan_start = uint32_t(std::stoull(arg, nullptr, 0));
                seed_set = true;
            } else {
                throw std::runtime_error("invalid argument: " + arg);
            }
        }
        bool use_vaes = mode != "gpu" && verus_solo_vaes2_available();
        if (mode != "gpu") {
            if (const char *setting = std::getenv("VERUS_SOLO_VAES")) {
                if (std::strcmp(setting, "0") == 0) use_vaes = false;
                else if (std::strcmp(setting, "1") == 0) {
                    if (!use_vaes)
                        throw std::runtime_error("VAES requested but CPU lacks VAES/AVX2");
                } else throw std::runtime_error("VERUS_SOLO_VAES must be 0 or 1");
            }
        }
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
            // Include byte carries and the upper edge of the 56-bit suffix.
            constexpr uint64_t edges[] = {0, 1, 0xffffff, 0x1000000,
                                          0xffffffff, 0x100000000,
                                          0xfffffffffffffe, 0xffffffffffffff};
            if (trial < 8)
                for (int j = 0; j < 7; ++j)
                    nonce[8 + j] = uint8_t(edges[trial] >> (8 * j));
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
            alignas(16) uint8_t cpu_input[64], cpu_word[32];
            std::memcpy(cpu_input, half, 64);
            std::memcpy(cpu_input + 32, nonce, 15);
            cpu_input[47] = uint8_t(inter);
            for (int j = 0; j < 16; ++j)
                cpu_input[48 + j] = uint8_t(inter >> (8 * ((j + 1) & 7)));
            haraka512_keyed(cpu_word, cpu_input,
                            reinterpret_cast<const u128 *>(keybuf) + (inter & 511));
            if (le32(cpu_word + 28) != le32(reference + 28))
                throw std::runtime_error("CPU shortcut target word disagrees with full hash");
            if (use_vaes) {
                uint64_t hi0 = 0;
                for (int j = 0; j < 7; ++j) hi0 |= uint64_t(nonce[8 + j]) << (8 * j);
                const uint64_t hi1 = (hi0 + 1 + uint64_t(trial) * 0x100000001ULL) &
                                     ((1ULL << 56) - 1);
                std::unique_ptr<void, void (*)(void *)> vaes(
                    verus_solo_vaes2_create(cpu_input,
                        reinterpret_cast<const u128 *>(keybuf) + (inter & 511)),
                    verus_solo_vaes2_destroy);
                uint32_t words[2];
                verus_solo_vaes2_words(vaes.get(), hi0, hi1, words);
                if (words[0] != le32(cpu_word + 28))
                    throw std::runtime_error("VAES first nonce disagrees with full hash");
                for (int j = 0; j < 7; ++j)
                    cpu_input[40 + j] = uint8_t(hi1 >> (8 * j));
                haraka512_keyed(cpu_word, cpu_input,
                    reinterpret_cast<const u128 *>(keybuf) + (inter & 511));
                if (words[1] != le32(cpu_word + 28))
                    throw std::runtime_error("VAES second nonce disagrees with scalar hash");
            }
        }

        const uint32_t target7 = uint32_t(target_be[0]) << 24 | uint32_t(target_be[1]) << 16 |
                                 uint32_t(target_be[2]) << 8 | uint32_t(target_be[3]);
        SoloResult result;
        CpuSearch cpu(result);
        if (mode == "cpu") {
            std::cout << "READY mode=cpu threads=" << cpu_threads
                      << " cpu_finalizer=" << (use_vaes ? "vaes2" : "scalar")
                      << " selector=0x" << std::hex << selector
                      << " extra_nonce=0x" << chosen << " target7=0x" << target7
                      << " fixed=0x" << fixed << " inter=0x" << inter
                      << std::dec << " cpu_checks=64" << std::endl;
            cpu.start(cpu_threads, false, half, keybuf, inter, raw, canonical,
                      target_be, target7, use_vaes);
            uint64_t previous = 0;
            auto last = std::chrono::steady_clock::now();
            while (cpu.active.load() && !result.stop.load()) {
                std::this_thread::sleep_for(std::chrono::seconds(1));
                const auto now = std::chrono::steady_clock::now();
                const double seconds = std::chrono::duration<double>(now - last).count();
                if (seconds >= 5) {
                    const uint64_t count = cpu.done.load();
                    std::cout << "RATE " << std::fixed << std::setprecision(2)
                              << (count - previous) / seconds / 1e6
                              << " MH/s total=" << count << std::endl;
                    previous = count;
                    last = now;
                }
            }
            cpu.join();
            if (result.error) std::rethrow_exception(result.error);
            if (!emit_result(result)) std::cout << "DONE nonce space exhausted" << std::endl;
            return 0;
        }

        cudaDeviceProp prop{};
        check(cudaGetDeviceProperties(&prop, 0), "GPU device");
        // With no per-warp write overlay, three 512-thread blocks can reside
        // on an Ada SM and fill its 48 warp slots.
        const int tpb = 512, blocks = prop.multiProcessorCount * 3;
        const size_t smem = 1024u * vw_aes_tables(prop.major) + VP_KEY_SLOTS * 16u;
        check(cudaFuncSetAttribute(solo_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   int(smem)), "GPU shared memory");
        Device<uint32_t> dK(VP_KEY_SLOTS * 4), dT(256), dW7(129);
        Device<uint64_t> dFound(1);
        Device<BenchJob> dJob(1);
        check(cudaMemcpy(dK.p, keybuf, VP_KEY_SLOTS * 16, cudaMemcpyHostToDevice), "key upload");
        check(cudaMemcpy(dT.p, table, sizeof(table), cudaMemcpyHostToDevice), "table upload");
        const uint32_t outer_start = uint32_t(rng()) & 0xffffffu;
        std::cout << "READY mode=" << mode << " GPU=" << prop.name
                  << " threads=" << (mode == "hybrid" ? cpu_threads : 0)
                  << (mode == "hybrid" ? (use_vaes ? " cpu_finalizer=vaes2" :
                                           " cpu_finalizer=scalar") : "")
                  << " selector=0x" << std::hex << selector
                  << " extra_nonce=0x" << chosen << " target7=0x" << target7 << std::dec
                  << " fixed=0x" << std::hex << fixed << " inter=0x" << inter
                  << std::dec << " cpu_checks=64" << std::endl;
        if (mode == "hybrid") {
            // CPU owns suffix bit 55 = 1; the GPU's inner counter stays below
            // 2^31 and therefore owns bit 55 = 0 for every outer value.
            cpu.start(cpu_threads, true, half, keybuf, inter, raw, canonical,
                      target_be, target7, use_vaes);
        }
        const uint64_t inner_end = mode == "hybrid" ? 1ULL << 31 : 1ULL << 32;
        uint64_t total = 0, since_report = 0;
        uint64_t cpu_previous = 0;
        auto last_report = std::chrono::steady_clock::now();
        for (uint32_t outer_count = 0; outer_count < (1u << 24) &&
                                      !result.stop.load(); ++outer_count) {
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
                // An odd count exercises the padded high half. The second
                // range exercises the full 32-bit inner nonce up to UINT32_MAX.
                for (const uint32_t test_first : {0u, 0xffffff7fu}) {
                    solo_kernel<<<blocks, tpb, smem>>>(dK.p, dT.p, dJob.p, inter,
                                                      test_first, 129, 0, nullptr, dW7.p);
                    check(cudaGetLastError(), "GPU test launch");
                    std::array<uint32_t, 129> words{};
                    check(cudaMemcpy(words.data(), dW7.p, words.size() * 4,
                                     cudaMemcpyDeviceToHost), "GPU test copy");
                    for (uint32_t i = 0; i < words.size(); ++i) {
                        uint8_t n[15], digest[32];
                        std::memcpy(n, nonce, 15);
                        put32(n + 11, test_first + i);
                        verus_host_hash(digest, half, n, keybuf, 1);
                        if (words[i] != le32(digest + 28))
                            throw std::runtime_error("GPU and CPU disagree on selected work");
                    }
                }
                std::cout << "GPU_CHECK 258/258" << std::endl;
            }
            for (uint64_t next = 0; next < inner_end && !result.stop.load(); ) {
                const uint32_t count = uint32_t(std::min<uint64_t>(BATCH, inner_end - next));
                const uint64_t end = next + count;
                while (next < end && !result.stop.load()) {
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
                            result.publish("GPU", raw.data(), digest);
                            break;
                        }
                    }
                }
                const auto now = std::chrono::steady_clock::now();
                const double seconds = std::chrono::duration<double>(now - last_report).count();
                if (seconds >= 5.0) {
                    const uint64_t cpu_total = cpu.done.load();
                    const double gpu_rate = since_report / seconds / 1e6;
                    const double cpu_rate = (cpu_total - cpu_previous) / seconds / 1e6;
                    std::cout << "RATE " << std::fixed << std::setprecision(2)
                              << (gpu_rate + cpu_rate) << " MH/s total="
                              << (total + cpu_total) << " gpu=" << gpu_rate
                              << " cpu=" << cpu_rate << std::endl;
                    last_report = now;
                    since_report = 0;
                    cpu_previous = cpu_total;
                }
            }
        }
        result.stop.store(true);
        cpu.join();
        if (result.error) std::rethrow_exception(result.error);
        if (!emit_result(result)) std::cout << "DONE nonce space exhausted" << std::endl;
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "ERROR " << e.what() << std::endl;
        return 1;
    }
}

#ifdef VERUS_SOLO_STANDALONE
int main(int argc, char **argv) {
    return argc > 1 && std::strcmp(argv[1], "--verus-solo") == 0
        ? verus_solo_main(argc - 1, argv + 1) : verus_solo_main(argc, argv);
}
#endif
