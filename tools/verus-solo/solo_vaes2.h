#pragma once

#include <cstdint>

// The context holds two Haraka512 states in the low/high 128-bit VAES lanes.
// Only nonce bytes 8..14 vary; all other input bytes and 40 round keys are
// prepared once per CPU worker. Call words only when available() is true.
extern "C" bool verus_solo_vaes2_available();
extern "C" void *verus_solo_vaes2_create(const uint8_t input[64],
                                          const void *round_keys);
extern "C" void verus_solo_vaes2_words(const void *context, uint64_t nonce0,
                                        uint64_t nonce1, uint32_t words[2]);
extern "C" void verus_solo_vaes2_destroy(void *context);
