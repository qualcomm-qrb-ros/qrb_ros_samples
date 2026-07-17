// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear

#pragma once
#include <cstdint>

namespace tts
{
namespace dict_en
{

// ── language parameters ───────────────────────────────────────────────────
constexpr uint32_t kInputLanguageCode = 0;  // EN
constexpr bool kIsBertEnabled = true;
constexpr int64_t kSid = 0;
constexpr float kSdpRatio = 0.2f;
constexpr float kNoiseScale = 0.667f;
constexpr float kNoiseScaleW = 0.8f;

// ── dict sizes ────────────────────────────────────────────────────────────
constexpr uint32_t kDictNumWords = 167122u;
constexpr uint32_t kWordListSize = 1480004u;
constexpr uint32_t kDictNumPhones = 1135140u;
constexpr uint32_t kDictNumPhonesOffset = 167122u;

// ── dict arrays (defined in dict_en.cpp) ──────────────────────────────────
extern const uint32_t kDictWordOffset[kDictNumWords];
extern const char kDictAllWords[kWordListSize];
extern const uint32_t kDictPhonesOffset[kDictNumPhonesOffset];
extern const uint8_t kPhonesList[kDictNumPhones];
extern const uint8_t kTonesList[kDictNumPhones];
extern const uint8_t kPhonesPerWord[kDictNumWords];

// ── replacement dict sizes ────────────────────────────────────────────────
constexpr uint32_t kNumReplEntries = 36u;
constexpr uint32_t kOrigWordListSize = 205u;
constexpr uint32_t kReplWordListSize = 308u;

// ── replacement dict arrays (defined in dict_en.cpp) ──────────────────────
extern const uint32_t kOrigWordOffset[kNumReplEntries];
extern const char kOrigWordList[kOrigWordListSize];
extern const uint32_t kReplWordOffset[kNumReplEntries];
extern const char kReplWordList[kReplWordListSize];

}  // namespace dict_en
}  // namespace tts
