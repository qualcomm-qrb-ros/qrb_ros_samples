// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear

#pragma once
#include <string>

#include "audio_types.hpp"

namespace audio
{

std::string findConfigPath();
Config loadDefaultConfig();
Config loadConfig(const std::string & path);

}  // namespace audio
