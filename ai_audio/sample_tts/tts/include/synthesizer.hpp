// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear

#pragma once
#include <cstdint>

#include "audio_types.hpp"
#include "bert.hpp"

namespace audio
{

class InferenceRunner;

class Synthesizer
{
public:
  Synthesizer();
  ~Synthesizer();

  [[nodiscard]] Status initialize();
  [[nodiscard]] Status initialize(const Config & cfg);
  [[nodiscard]] Status process_text(const char * text, uint32_t textLen);
  [[nodiscard]] Status get_chunk(AudioBuffer & out);
  [[nodiscard]] Status speak(const char * text, uint32_t textLen);
  Status teardown();

private:
  Status boot(const Config & cfg, const char * tokPath, const char * normPath);
  Status reset_state();

  uint32_t m_sampleRate = 44100;
  uint32_t m_bitsPerSample = 16;
  uint32_t m_numChannels = 1;
  uint32_t m_numFormats = 1;
  uint32_t m_speechFormat = 0;
  PcmFormatList m_fmtList = {};

  uint32_t m_modelGen = 0;
  uint32_t m_langCode = 0;
  bool m_bertOn = false;
  HwConfig m_hwCfg = {};

  uint32_t m_phoneCount = 0;
  uint32_t m_toneCount = 0;
  uint32_t m_phones[kMaxPhoneSeq] = {};
  uint32_t m_tones[kMaxPhoneSeq] = {};

  int64_t m_speakerId = 0;
  float m_sdpRatio = 0.0f;
  float m_noiseScale = 0.667f;
  float m_noiseScaleW = 0.8f;
  float m_lengthScale = 1.0f;
  int64_t m_xLengths = 0;
  float m_gVec[kSpeakerVecDim] = {};
  float m_zMat[kLatentChannels * kMaxFrames] = {};
  float m_xMask[kMaxPhoneSeq] = {};
  float m_wCeil[kMaxPhoneSeq] = {};
  float m_yMask[kBatchSize * kMaxFrames] = {};
  float m_attn[kMaxFrames * kMaxPhoneSeq] = {};
  float m_bertEmb[kBertDim * kMaxPhoneSeq] = {};
  float m_mP[kMaxPhoneSeq * kLatentChannels] = {};
  float m_logsP[kMaxPhoneSeq * kLatentChannels] = {};
  uint32_t m_frameCount = 0;
  uint32_t m_totalFrames = 0;
  uint32_t m_decodeOffset = 0;

  uint32_t m_segIdx = 0;
  bool m_pipelineDone = false;
  bool m_decFinished = true;
  uint8_t m_textBuf[kMaxTextBytes] = {};
  uint32_t m_textLen = 0;
  uint8_t m_audioBuf[kMaxAudioBytes] = {};
  uint32_t m_audioBytes = 0;

  const char * m_dictWords = nullptr;
  const uint32_t * m_dictWordOffsets = nullptr;
  const uint32_t * m_dictPhoneOffsets = nullptr;
  const uint8_t * m_phoneList = nullptr;
  const uint8_t * m_toneList = nullptr;
  const uint8_t * m_phonesPerWord = nullptr;
  uint32_t m_dictWordCount = 0;
  uint32_t m_dictWordBytes = 0;
  uint32_t m_dictPhoneCount = 0;
  uint32_t m_dictPhoneOffCnt = 0;

  uint32_t m_replCount = 0;
  const uint32_t * m_replOrigOff = nullptr;
  const char * m_replOrigWords = nullptr;
  const uint32_t * m_replReplOff = nullptr;
  const char * m_replReplWords = nullptr;

  BertPipeline m_tokenizer = {};
  TokenBuffer m_preTokBuf = {};
  TokenBuffer m_outTokBuf = {};
  uint8_t m_tokWordCount[kTokenSlots] = {};
  uint8_t m_tokPhoneCount[kTokenSlots] = {};
  uint32_t m_word2ph[kTokenSlots] = {};
  uint32_t m_bertDistIdx = 0;
  char m_wordGroups[kTokenSlots * kMaxPieceLen] = {};
  uint32_t m_groupOffsets[kTokenSlots] = {};
  uint8_t * m_tokenizerData = nullptr;
  uint8_t * m_normData = nullptr;

  SegmentResult m_sentences = {};
  InferenceRunner * m_runner = nullptr;

  bool loadFiles(const char * tokPath, const char * normPath);
  Status buildTokenizer();
  Status tokenize(char * text);
  Status groupTokens(char * groups, uint32_t * offsets, int * count);
  Status buildPhoneSeq(char * groups, uint32_t * offsets, int * count);
  Status prepareSegment();
  void buildAlignPath(const float * durations, uint32_t phones, uint32_t frames);
  Status runBert();
  Status runEncoder();
  Status runFlow();
  Status runDecoder();
  Status runPipeline();

  void distributePhones(int nPhones, int nWords, uint8_t * out) const;
  Status lookupWord(const char * word, int * idx) const;
  Status letterToPhones(const char * word, uint32_t phones[], uint32_t tones[], int * count) const;
  static void boundedCopy(char * dst, const char * src, size_t cap);
  static int utf8Width(char lead);
};

}  // namespace audio
