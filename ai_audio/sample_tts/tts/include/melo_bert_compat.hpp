// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear

#pragma once

#include <cstdio>

/* Language codes — must match melo_api.h enum order */
constexpr int melo_en = 0;
constexpr int melo_zh = 1;
constexpr int melo_de = 2;
constexpr int melo_es = 3;
constexpr int melo_ru = 4;
constexpr int melo_ko = 5;
constexpr int melo_fr = 6;
constexpr int melo_ja = 7;
constexpr int melo_pt = 8;
constexpr int melo_tr = 9;
constexpr int melo_pl = 10;

/* From melo_calibration_api.h */
constexpr int MELO_MAX_TTS_CHAR_SIZE = 1024;

/* Logging — non-Android, non-DSP path from melo.h */
#define MELO_ERRLOG(fmt, ...)                                                                      \
  do {                                                                                             \
    printf("%s:%d %s ", __FILE__, __LINE__, __func__);                                             \
    printf(fmt, ##__VA_ARGS__);                                                                    \
  } while (0)
