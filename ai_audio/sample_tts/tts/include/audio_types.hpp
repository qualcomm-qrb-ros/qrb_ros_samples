// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace audio
{

// ─── constants
// ────────────────────────────────────────────────────────────────

inline constexpr uint32_t kMaxTextBytes = 1024;
inline constexpr uint32_t kMaxAudioBytes = 40960;
inline constexpr uint32_t kTokenSlots = 400;
inline constexpr uint32_t kMaxSentences = 2044;
inline constexpr uint32_t kMaxPieceLen = 128;
inline constexpr uint32_t kMaxPhoneSeq = 512;
inline constexpr uint32_t kSoftBreakLen = 75;
inline constexpr uint32_t kHardBreakLen = 150;
inline constexpr uint32_t kBertDim = 768;
inline constexpr uint32_t kSpeakerVecDim = 256;
inline constexpr uint32_t kLatentChannels = 192;
inline constexpr uint32_t kDecoderWindow = 40;
inline constexpr uint32_t kDecoderOverlap = 12;
inline constexpr uint32_t kHopSamples = 512;
inline constexpr uint32_t kMaxFrames = 1536;
inline constexpr uint32_t kMaxPhones = 512;
inline constexpr uint32_t kBatchSize = 1;
inline constexpr uint32_t kMaxFallbackPhones = 100;
inline constexpr uint32_t kMaxUtf8Bytes = 4;

inline constexpr float kDefaultRate = 1.0f;
inline constexpr float kDefaultPitch = 0.0f;
inline constexpr float kDefaultGain = 0.0f;
inline constexpr uint32_t kDefaultSampleRate = 44100;

// ─── status
// ───────────────────────────────────────────────────────────────────

enum class Status : int32_t
{
  Ok = 0,
  Success = 0,
  MoreChunks = 1,
  OutputIncomplete = 1,
  Fail = 2,
  NeedInput = 3,
  NeedMore = 3,
  Detected = 4,
  InvalidArg = 5,
  InvalidParam = 5,
  NotInitialized = 6,
  NotPrepared = 6,
  NullInput = 7,
  NullRef = 7,
  SizeMismatch = 8,
  WrongSize = 8,
  ModelMismatch = 9,
  WrongModel = 9,
  BufferOverflow = 10,
  FatalBufferOverflow = 10,
  Disabled = 11,
  LastError = 12,
  TimedOut = 13,
  Timeout = 13,
  VersionMismatch = 14,
  IncompatibleModelVersion = 14,
  NotSupported = 15,
  Unsupported = 15,
  OutOfMemory = 16,
  InsufficientScratchMem = 16,
};

// ─── PCM encoding
// ─────────────────────────────────────────────────────────────

enum class Encoding : uint32_t
{
  Raw16Bit = 0,
  Mp3 = 1,
  OggVorbis = 2,
  MuLaw = 3,
  ALaw = 4,
};

// ─── engine configuration
// ─────────────────────────────────────────────────────

struct Config
{
  Encoding encoding = Encoding::Raw16Bit;
  float speechRate = kDefaultRate;
  float pitch = kDefaultPitch;
  float volumeGain = kDefaultGain;
  uint32_t sampleRate = kDefaultSampleRate;
  uint32_t languageCode = 0;
  uint32_t modelGeneration = 1;

  std::string tokenizerPath;
  std::string normalizerPath;
  std::string bertModelPath;
  std::string encoderModelPath;
  std::string flowModelPath;
  std::string decoderModelPath;
  std::string backendLibPath;
  std::string systemLibPath;
};

// ─── audio output chunk
// ───────────────────────────────────────────────────────

struct AudioBuffer
{
  Status result = Status::Fail;
  uint32_t byteCount = 0;
  uint8_t samples[kMaxAudioBytes] = {};
};

// ─── sentence list
// ────────────────────────────────────────────────────────────

struct SegmentResult
{
  char buffer[kMaxSentences] = {};
  uint16_t beginIdx[kMaxSentences] = {};
  uint16_t endIdx[kMaxSentences] = {};
  uint32_t segmentCount = 0;
};

// ─── PCM format
// ───────────────────────────────────────────────────────────────

struct PcmFormat
{
  uint32_t sampleRate = 0;
  uint32_t bitsPerSample = 0;
  uint32_t channels = 0;
};

struct PcmFormatList
{
  uint32_t count = 0;
  PcmFormat entries[3] = {};
};
static_assert(sizeof(PcmFormatList) == 40, "PcmFormatList size mismatch");

// ─── hardware runtime config
// ──────────────────────────────────────────────────

struct HwConfig
{
  uint32_t encodingMode = 0;
  float speechRate = 1.0f;
  float pitch = 0.0f;
  float volumeGain = 0.0f;
  uint32_t sampleRate = 44100;
  uint32_t _reserved = 0;
  void * hwContext = nullptr;
};
static_assert(sizeof(HwConfig) == 32, "HwConfig size mismatch");

}  // namespace audio

// ─── global backward-compat aliases ──────────────────────────────────────────

using AudioStatus = audio::Status;
using AudioConfig = audio::Config;
using AudioChunk = audio::AudioBuffer;
using AudioSegments = audio::SegmentResult;
using AudioPcmFmt = audio::PcmFormat;
using AudioFmtList = audio::PcmFormatList;
using AudioHwCfg = audio::HwConfig;
using AudioEncoding = audio::Encoding;

inline constexpr uint32_t AUDIO_MAX_TEXT = audio::kMaxTextBytes;
inline constexpr uint32_t AUDIO_MAX_PCM = audio::kMaxAudioBytes;
inline constexpr uint32_t AUDIO_SAMPLE_RATE = audio::kDefaultSampleRate;
