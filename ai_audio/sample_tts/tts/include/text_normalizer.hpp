// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear

#pragma once
#include "audio_compat.hpp"

namespace tts
{

int text_normalize_en(const char * text, char * output, size_t output_size);
char * text_lower(char * text);
char * text_upper(char * text);

}  // namespace tts
