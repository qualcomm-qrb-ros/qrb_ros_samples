// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear

#pragma once
#include "audio_compat.hpp"

namespace tts
{

void split_struct_reset(SplitSentence & s);

TtsResult split_sentences(SplitSentence & out,
    char * text,
    uint16_t text_len,
    uint16_t min_len,
    uint16_t max_len);

}  // namespace tts
