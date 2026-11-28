#include "solo_vaes2.h"

#include <immintrin.h>
#include <cstring>
#include <new>
#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace {

struct alignas(32) Context {
    __m256i s0, s1, s3;
    __m256i keys[40];
    uint32_t word0, word1, word3_fixed, feed;
};

#if defined(__GNUC__) || defined(__clang__)
#define SOLO_VAES_TARGET __attribute__((target("avx2,vaes")))
#else
#define SOLO_VAES_TARGET
#endif

SOLO_VAES_TARGET void init(Context &c, const uint8_t input[64],
                           const __m128i *keys) {
    c.s0 = _mm256_broadcastsi128_si256(
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(input)));
    c.s1 = _mm256_broadcastsi128_si256(
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(input + 16)));
    c.s3 = _mm256_broadcastsi128_si256(
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(input + 48)));
    std::memcpy(&c.word0, input + 32, 4);
    std::memcpy(&c.word1, input + 36, 4);
    std::memcpy(&c.word3_fixed, input + 44, 4);
    c.word3_fixed &= 0xff000000u; // byte 47 is intermediate, not nonce
    std::memcpy(&c.feed, input + 52, 4);
    for (int i = 0; i < 40; ++i)
        c.keys[i] = _mm256_broadcastsi128_si256(keys[i]);
}

SOLO_VAES_TARGET void words(const Context &c, uint64_t n0, uint64_t n1,
                            uint32_t out[2]) {
    // Verus nonce bytes 8..14 occupy input[40..46], or words 2 and 3 of
    // state s2. The high byte of word 3 is the fixed CLHash intermediate.
    const __m128i lo = _mm_set_epi32(int(c.word3_fixed | (n0 >> 32 & 0xffffffu)),
                                     int(n0), int(c.word1), int(c.word0));
    const __m128i hi = _mm_set_epi32(int(c.word3_fixed | (n1 >> 32 & 0xffffffu)),
                                     int(n1), int(c.word1), int(c.word0));
    __m256i s0 = c.s0, s1 = c.s1, s3 = c.s3, tmp;
    __m256i s2 = _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
#define VAES4(rci) \
    s0 = _mm256_aesenc_epi128(s0, c.keys[(rci)]); \
    s1 = _mm256_aesenc_epi128(s1, c.keys[(rci) + 1]); \
    s2 = _mm256_aesenc_epi128(s2, c.keys[(rci) + 2]); \
    s3 = _mm256_aesenc_epi128(s3, c.keys[(rci) + 3]); \
    s0 = _mm256_aesenc_epi128(s0, c.keys[(rci) + 4]); \
    s1 = _mm256_aesenc_epi128(s1, c.keys[(rci) + 5]); \
    s2 = _mm256_aesenc_epi128(s2, c.keys[(rci) + 6]); \
    s3 = _mm256_aesenc_epi128(s3, c.keys[(rci) + 7]);
#define VMIX4 \
    tmp = _mm256_unpacklo_epi32(s0, s1); \
    s0 = _mm256_unpackhi_epi32(s0, s1); \
    s1 = _mm256_unpacklo_epi32(s2, s3); \
    s2 = _mm256_unpackhi_epi32(s2, s3); \
    s3 = _mm256_unpacklo_epi32(s0, s2); \
    s0 = _mm256_unpackhi_epi32(s0, s2); \
    s2 = _mm256_unpackhi_epi32(s1, tmp); \
    s1 = _mm256_unpacklo_epi32(s1, tmp);
    VAES4(0); VMIX4;
    VAES4(8); VMIX4;
    VAES4(16); VMIX4;
    VAES4(24);
    // The existing scalar haraka512_keyed computes only digest word 7.
    // Its fourth MIX4_LAST leaves just s2, and round five AESes just s2.
    tmp = _mm256_unpacklo_epi32(s0, s1);
    s1 = _mm256_unpacklo_epi32(s2, s3);
    s2 = _mm256_unpackhi_epi32(s1, tmp);
    s2 = _mm256_aesenc_epi128(s2, c.keys[34]);
    s2 = _mm256_aesenc_epi128(s2, c.keys[38]);
#undef VAES4
#undef VMIX4
    const __m128i a = _mm256_castsi256_si128(s2);
    const __m128i b = _mm256_extracti128_si256(s2, 1);
    out[0] = uint32_t(_mm_cvtsi128_si32(_mm_srli_si128(a, 8))) ^ c.feed;
    out[1] = uint32_t(_mm_cvtsi128_si32(_mm_srli_si128(b, 8))) ^ c.feed;
}

} // namespace

extern "C" bool verus_solo_vaes2_available() {
#if defined(_MSC_VER)
    int info[4];
    __cpuid(info, 0);
    if (info[0] < 7) return false;
    __cpuid(info, 1);
    if (!(info[2] & (1 << 27)) || !(info[2] & (1 << 28))) return false;
    if ((_xgetbv(0) & 6) != 6) return false;
    __cpuidex(info, 7, 0);
    return (info[1] & (1 << 5)) && (info[2] & (1 << 9));
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("vaes");
#else
    return false;
#endif
}

extern "C" void *verus_solo_vaes2_create(const uint8_t input[64],
                                           const void *round_keys) {
    auto *c = new Context;
    init(*c, input, reinterpret_cast<const __m128i *>(round_keys));
    return c;
}

extern "C" void verus_solo_vaes2_words(const void *context, uint64_t nonce0,
                                         uint64_t nonce1, uint32_t result[2]) {
    words(*static_cast<const Context *>(context), nonce0, nonce1, result);
}

extern "C" void verus_solo_vaes2_destroy(void *context) {
    delete static_cast<Context *>(context);
}
