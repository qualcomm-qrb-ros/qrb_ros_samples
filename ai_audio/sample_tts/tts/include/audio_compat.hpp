// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear

#pragma once
#include "audio_types.hpp"

namespace tts
{
using SynthStatus = ::audio::Status;
using SynthConfig = ::audio::Config;
using PcmOutput = ::audio::AudioBuffer;
using SegmentedText = ::audio::SegmentResult;

using TtsResult = ::audio::Status;
using TtsConfig = ::audio::Config;
using TtsStatusParam = ::audio::AudioBuffer;
using SplitSentence = ::audio::SegmentResult;

inline constexpr uint32_t TTS_MAX_TEXT_SIZE = ::audio::kMaxTextBytes;
inline constexpr uint32_t TTS_MAX_PCM_SIZE = ::audio::kMaxAudioBytes;
inline constexpr uint32_t BERT_MAX_TOKENS = ::audio::kTokenSlots;
inline constexpr uint32_t MAX_SPLIT_SENTENCES = ::audio::kMaxSentences;
inline constexpr uint32_t MAX_TOKEN_LEN = ::audio::kMaxPieceLen;
inline constexpr uint32_t MAX_SEQ_LEN = ::audio::kMaxPhoneSeq;
inline constexpr uint32_t DESIRED_SENTENCE_LEN = ::audio::kSoftBreakLen;
inline constexpr uint32_t MAX_SENTENCE_LEN = ::audio::kHardBreakLen;
}  // namespace tts
