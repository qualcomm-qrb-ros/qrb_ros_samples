// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear

#pragma once

#include <cstdint>

#include "audio_types.hpp"
#include "bert.hpp"

namespace audio
{

// Type aliases so existing code that references the old names still compiles.
using TokenVocabEntry = ::VocabRecord;
using TokenBuffer = ::TokenBuffer;

struct ModelFileHeader
{
  uint32_t magic;
  int16_t versionMajor;
  int16_t versionMinor;
  uint32_t size;
  uint32_t padding;
};

inline constexpr uint32_t kMagicMelo0 = 0x4D454C30;  // 'MEL0'

}  // namespace audio
