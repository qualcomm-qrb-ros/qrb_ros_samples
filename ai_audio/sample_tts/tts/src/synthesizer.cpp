// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear

#include "synthesizer.hpp"

#include <pulse/error.h>
#include <pulse/simple.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "audio_types.hpp"
#include "bert.hpp"
#include "config_loader.hpp"
#include "dict_en.hpp"
#include "inference_engine.hpp"
#include "sentence_split.hpp"
#include "text_normalizer.hpp"

namespace audio
{

// ─── ctor / dtor
// ──────────────────────────────────────────────────────────────

Synthesizer::Synthesizer() {}

Synthesizer::~Synthesizer()
{
  delete m_runner;
  m_runner = nullptr;
  delete[] m_tokenizerData;
  delete[] m_normData;
}

// ─── boundedCopy ─────────────────────────────────────────────────────────────

void Synthesizer::boundedCopy(char * dst, const char * src, size_t cap)
{
  if (!dst || !src || cap == 0)
    return;
  size_t n = strlen(src);
  if (n >= cap)
    n = cap - 1;
  memcpy(dst, src, n);
  dst[n] = '\0';
}

// ─── utf8Width
// ────────────────────────────────────────────────────────────────

int Synthesizer::utf8Width(char lead)
{
  unsigned char c = static_cast<unsigned char>(lead);
  if ((c & 0x80) == 0)
    return 1;
  if ((c & 0xE0) == 0xC0)
    return 2;
  if ((c & 0xF0) == 0xE0)
    return 3;
  if ((c & 0xF8) == 0xF0)
    return 4;
  return 1;
}

// ─── initialize
// ───────────────────────────────────────────────────────────────

Status Synthesizer::initialize()
{
  return initialize(loadDefaultConfig());
}

Status Synthesizer::initialize(const Config & cfg)
{
  if (cfg.speechRate < 0.25f || cfg.speechRate > 4.0f)
    return Status::InvalidArg;
  if (cfg.pitch < -20.0f || cfg.pitch > 20.0f)
    return Status::InvalidArg;
  if (cfg.volumeGain < -96.0f || cfg.volumeGain > 16.0f)
    return Status::InvalidArg;
  if (cfg.sampleRate != 44100)
    return Status::InvalidArg;
  return boot(cfg, cfg.tokenizerPath.c_str(), cfg.normalizerPath.c_str());
}

// ─── teardown ────────────────────────────────────────────────────────────────

Status Synthesizer::teardown()
{
  delete m_runner;
  m_runner = nullptr;
  delete[] m_tokenizerData;
  m_tokenizerData = nullptr;
  delete[] m_normData;
  m_normData = nullptr;
  return reset_state();
}

// ─── process_text
// ─────────────────────────────────────────────────────────────

Status Synthesizer::process_text(const char * text, uint32_t length)
{
  if (!text || length == 0 || text[0] == '\0')
    return Status::NeedInput;

  m_textLen = (length >= kMaxTextBytes) ? kMaxTextBytes - 1 : length;
  strncpy(reinterpret_cast<char *>(m_textBuf), text, m_textLen);
  m_textBuf[m_textLen] = '\0';

  // Apply replacement dictionary
  char replaced[kMaxTextBytes] = {};
  {
    const char * src = reinterpret_cast<const char *>(m_textBuf);
    size_t wi = 0;
    size_t ri = 0;
    size_t srcLen = strlen(src);

    while (ri <= srcLen && wi < kMaxTextBytes - 1) {
      if (src[ri] == '\0') {
        replaced[wi++] = '\0';
        break;
      }
      bool found = false;
      for (uint32_t e = 0; e < m_replCount && !found; ++e) {
        const char * orig = m_replOrigWords + m_replOrigOff[e];
        size_t origLen = strlen(orig);
        if (origLen == 0)
          continue;
        if (ri + origLen <= srcLen) {
          bool match = true;
          for (size_t k = 0; k < origLen && match; ++k)
            match = (tolower(static_cast<unsigned char>(src[ri + k])) == orig[k]);
          if (match) {
            const char * repl = m_replReplWords + m_replReplOff[e];
            size_t replLen = strlen(repl);
            if (wi + replLen < kMaxTextBytes - 1) {
              memcpy(replaced + wi, repl, replLen);
              wi += replLen;
            }
            ri += origLen;
            found = true;
          }
        }
      }
      if (!found)
        replaced[wi++] = src[ri++];
    }
    replaced[wi < kMaxTextBytes ? wi : kMaxTextBytes - 1] = '\0';
  }

  char normalised[kMaxTextBytes] = {};
  if (tts::text_normalize_en(replaced, normalised, sizeof(normalised)) != 0)
    return Status::NotSupported;

  if (m_bertOn && m_tokenizerData && m_normData) {
    Status st = buildTokenizer();
    if (st != Status::Ok)
      return st;
  }

  m_segIdx = 0;
  uint16_t textLen16 = static_cast<uint16_t>(strlen(normalised));
  return tts::split_sentences(m_sentences, normalised, textLen16, kSoftBreakLen, kHardBreakLen);
}

// ─── get_chunk ───────────────────────────────────────────────────────────────

Status Synthesizer::get_chunk(AudioBuffer & output)
{
  m_lengthScale = 1.0f / m_hwCfg.speechRate;

  if (m_sentences.segmentCount == 0)
    return Status::NeedInput;

  Status st = Status::Ok;

  if (m_segIdx <= m_sentences.segmentCount) {
    if (!m_decFinished && m_frameCount != 0) {
      st = runPipeline();
      if (st != Status::Ok)
        return st;
    } else {
      st = prepareSegment();
      if (st != Status::Ok)
        return st;
      m_decFinished = true;
      st = runPipeline();
      if (st != Status::Ok)
        return st;
      ++m_segIdx;
    }

    output.byteCount = m_audioBytes;
    memcpy(output.samples, m_audioBuf, m_audioBytes);

    st = Status::MoreChunks;
    if (m_segIdx >= m_sentences.segmentCount && m_decFinished) {
      m_pipelineDone = true;
      output.result = Status::Ok;
      st = Status::Ok;
    }

    m_audioBytes = 0;
    memset(m_audioBuf, 0, sizeof(m_audioBuf));

    if (m_pipelineDone) {
      m_pipelineDone = false;
      m_decFinished = true;
    }
  } else {
    st = Status::Ok;
  }

  return st;
}

// ─── speak ───────────────────────────────────────────────────────────────────

Status Synthesizer::speak(const char * text, uint32_t textLen)
{
  Status st = process_text(text, textLen);
  if (st != Status::Ok)
    return st;

  std::vector<int16_t> pcm;
  int iter = 0;

  while (true) {
    AudioBuffer chunk{};
    st = get_chunk(chunk);

    if (chunk.byteCount > 0) {
      const auto * s = reinterpret_cast<const int16_t *>(chunk.samples);
      pcm.insert(pcm.end(), s, s + chunk.byteCount / sizeof(int16_t));
    }

    if (st == Status::Ok)
      break;
    if (st != Status::MoreChunks)
      return st;
    if (++iter > 200)
      return Status::TimedOut;
  }

  if (pcm.empty())
    return Status::Ok;

  pa_sample_spec ss{ PA_SAMPLE_S16LE, 44100, 1 };
  int err = 0;
  pa_simple * pa = pa_simple_new(
      nullptr, "tts1", PA_STREAM_PLAYBACK, nullptr, "speech", &ss, nullptr, nullptr, &err);
  if (!pa)
    return Status::Fail;

  pa_simple_write(pa, pcm.data(), pcm.size() * sizeof(int16_t), &err);
  pa_simple_drain(pa, &err);
  pa_simple_free(pa);
  return Status::Ok;
}

// ─── reset_state ─────────────────────────────────────────────────────────────

Status Synthesizer::reset_state()
{
  m_segIdx = 0;
  m_pipelineDone = false;
  m_decFinished = true;
  m_audioBytes = 0;
  m_frameCount = 0;
  m_sentences = SegmentResult{};
  return Status::Ok;
}

// ─── distributePhones ────────────────────────────────────────────────────────

void Synthesizer::distributePhones(int nPhones, int nWords, uint8_t * out) const
{
  for (int i = 0; i < nWords; ++i)
    out[i] = 0;
  for (int t = 0; t < nPhones; ++t) {
    int minVal = out[0], minIdx = 0;
    for (int i = 1; i < nWords; ++i) {
      if (out[i] < minVal) {
        minVal = out[i];
        minIdx = i;
      }
    }
    ++out[minIdx];
  }
}

// ─── lookupWord ──────────────────────────────────────────────────────────────

Status Synthesizer::lookupWord(const char * word, int * idx) const
{
  if (!word)
    return Status::Fail;
  *idx = -1;
  uint32_t lo = 0, hi = m_dictWordCount - 1;
  while (lo <= hi) {
    uint32_t mid = lo + (hi - lo) / 2;
    int cmp = strcmp(m_dictWords + m_dictWordOffsets[mid], word);
    if (cmp == 0) {
      *idx = static_cast<int>(mid);
      break;
    } else if (cmp < 0)
      lo = mid + 1;
    else
      hi = mid - 1;
  }
  return Status::Ok;
}

// ─── letterToPhones ──────────────────────────────────────────────────────────

Status Synthesizer::letterToPhones(const char * word,
    uint32_t phones[],
    uint32_t tones[],
    int * count) const
{
  *count = 0;
  int pos = 0;
  while (word[pos] != '\0') {
    int wid = 0;
    int blen = utf8Width(word[pos]);
    char letter[kMaxUtf8Bytes + 1] = {};
    for (int i = 0; i < blen && i < (int)kMaxUtf8Bytes; ++i)
      letter[i] = word[pos + i];
    lookupWord(letter, &wid);
    if (wid != -1 && *count < (int)kMaxFallbackPhones) {
      for (uint32_t j = 0; j < m_phonesPerWord[wid]; ++j) {
        phones[*count] = m_phoneList[m_dictPhoneOffsets[wid] + j];
        tones[*count] = m_toneList[m_dictPhoneOffsets[wid] + j];
        ++(*count);
      }
    }
    pos += blen;
  }
  return Status::Ok;
}

// ─── loadFiles ───────────────────────────────────────────────────────────────

bool Synthesizer::loadFiles(const char * tokPath, const char * normPath)
{
  auto readFile = [](const char * path) -> uint8_t * {
    if (!path || path[0] == '\0')
      return nullptr;
    FILE * f = fopen(path, "rb");
    if (!f)
      return nullptr;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    rewind(f);
    if (sz <= 0) {
      fclose(f);
      return nullptr;
    }
    auto * buf = new uint8_t[static_cast<size_t>(sz)];
    if (fread(buf, 1, static_cast<size_t>(sz), f) != static_cast<size_t>(sz)) {
      fclose(f);
      delete[] buf;
      return nullptr;
    }
    fclose(f);
    return buf;
  };

  if (tokPath)
    m_tokenizerData = readFile(tokPath);
  if (normPath)
    m_normData = readFile(normPath);
  return true;
}

// ─── buildTokenizer ──────────────────────────────────────────────────────────

Status Synthesizer::buildTokenizer()
{
  if (!m_tokenizerData)
    return Status::Fail;

  uint8_t * ptr = m_tokenizerData;

  uint32_t numFulls = *reinterpret_cast<uint32_t *>(ptr);
  ptr += 4;
  uint32_t numCacheFulls = *reinterpret_cast<uint32_t *>(ptr);
  ptr += 4;
  uint32_t numPartials = *reinterpret_cast<uint32_t *>(ptr);
  ptr += 4;
  uint32_t numCachePartials = *reinterpret_cast<uint32_t *>(ptr);
  ptr += 4;
  uint32_t sizeFullsVocab = *reinterpret_cast<uint32_t *>(ptr);
  ptr += 4;
  uint32_t sizePartialsVocab = *reinterpret_cast<uint32_t *>(ptr);
  ptr += 4;

  char * fullsVocab = reinterpret_cast<char *>(ptr);
  ptr += sizeFullsVocab;
  uint32_t * fullsCache = reinterpret_cast<uint32_t *>(ptr);
  ptr += numCacheFulls * sizeof(uint32_t);
  char * partsVocab = reinterpret_cast<char *>(ptr);
  ptr += sizePartialsVocab;
  uint32_t * partsCache = reinterpret_cast<uint32_t *>(ptr);
  ptr += numCachePartials * sizeof(uint32_t);
  VocabRecord * fullsInfo = reinterpret_cast<VocabRecord *>(ptr);
  ptr += numFulls * sizeof(VocabRecord);
  VocabRecord * partsInfo = reinterpret_cast<VocabRecord *>(ptr);

  m_tokenizer.setVocabStart(fullsVocab, fullsCache, fullsInfo, static_cast<int>(numFulls),
      static_cast<int>(numCacheFulls));
  m_tokenizer.setVocabCont(partsVocab, partsCache, partsInfo, static_cast<int>(numPartials),
      static_cast<int>(numCachePartials));

  m_tokenizer.setLanguage(static_cast<int32_t>(m_langCode));

  if (!m_normData)
    return Status::Fail;

  uint8_t * nptr = m_normData;
  nptr += sizeof(uint32_t);
  uint32_t sizeFixed = *reinterpret_cast<uint32_t *>(nptr);
  nptr += sizeof(uint32_t);
  nptr += sizeof(uint32_t);

  m_tokenizer.setNorm(static_cast<void *>(nptr), reinterpret_cast<uint32_t *>(nptr + sizeFixed));

  return Status::Ok;
}

// ─── tokenize ────────────────────────────────────────────────────────────────

Status Synthesizer::tokenize(char * text)
{
  m_preTokBuf.count = 0;
  m_outTokBuf.count = 0;
  m_tokenizer.setBuffers(&m_preTokBuf, &m_outTokBuf);

  int err = m_tokenizer.normalize(reinterpret_cast<uint8_t *>(text));
  if (err != 0)
    return Status::Fail;

  err = m_tokenizer.preTokenize(reinterpret_cast<const uint8_t *>(text));
  if (err != 0)
    return Status::Fail;

  err = m_tokenizer.tokenize(1);
  if (err != 0)
    return Status::Fail;

  m_tokenizer.addSpecialTokens(1, 1);
  return Status::Ok;
}

// ─── groupTokens ─────────────────────────────────────────────────────────────

Status Synthesizer::groupTokens(char * groups, uint32_t * offsets, int * count)
{
  TokenBuffer * outBuf = m_tokenizer.outputStream();
  if (!outBuf || outBuf->count == 0)
    return Status::Fail;

  int tokenCount = outBuf->count;
  *count = 0;
  int currOffset = 0;
  int writeIdx = 0;

  for (int ri = 0; ri < tokenCount; ++ri) {
    int32_t tid = static_cast<int32_t>(outBuf->tokens[ri].tokenId);
    bool skip = (tid <= 998 && tid != 101 && tid != 102);
    if (!skip) {
      if (writeIdx != ri)
        outBuf->tokens[writeIdx] = outBuf->tokens[ri];
      ++writeIdx;
    }
  }
  outBuf->count = writeIdx;
  tokenCount = writeIdx;

  for (int i = 0; i < tokenCount; ++i) {
    const char * tok = reinterpret_cast<const char *>(outBuf->tokens[i].text);
    if (strncmp(tok, "##", 2) != 0) {
      offsets[*count] = static_cast<uint32_t>(currOffset);
      size_t tlen = strlen(tok);
      if (static_cast<size_t>(currOffset) + tlen + 1 <
          static_cast<size_t>(kTokenSlots * kMaxPieceLen)) {
        boundedCopy(groups + currOffset, tok, kMaxPieceLen);
        currOffset += static_cast<int>(tlen) + 1;
      }
      m_tokWordCount[*count] = 1;
      ++(*count);
    } else {
      if (*count > 0) {
        strncat(groups + offsets[*count - 1], tok + 2, kMaxPieceLen - 1);
        currOffset += static_cast<int>(strlen(tok + 2));
        ++m_tokWordCount[*count - 1];
      }
    }
  }
  return Status::Ok;
}

// ─── buildPhoneSeq
// ────────────────────────────────────────────────────────────

Status Synthesizer::buildPhoneSeq(char * groups, uint32_t * offsets, int * count)
{
  memset(m_tokPhoneCount, 0, sizeof(uint8_t) * kTokenSlots);
  memset(m_word2ph, 0, sizeof(uint32_t) * kTokenSlots);
  m_bertDistIdx = 0;

  m_phoneCount = 2;
  m_toneCount = 2;
  m_tones[0] = 0;
  m_tones[1] = 7;

  for (int i = 0; i < *count; ++i) {
    int numWords = m_tokWordCount[i];
    char * word = groups + offsets[i];

    char upper[kMaxTextBytes] = {};
    boundedCopy(upper, word, sizeof(upper));
    tts::text_upper(upper);

    int entryIdx = -1;
    lookupWord(upper, &entryIdx);

    if (entryIdx != -1) {
      int nPh = static_cast<int>(m_phonesPerWord[entryIdx]);
      for (uint32_t j = 0; j < m_phonesPerWord[entryIdx]; ++j) {
        m_phones[m_phoneCount++] = 0;
        m_phones[m_phoneCount++] = m_phoneList[m_dictPhoneOffsets[entryIdx] + j];
      }
      for (uint32_t j = 0; j < m_phonesPerWord[entryIdx]; ++j) {
        m_tones[m_toneCount++] = 0;
        m_tones[m_toneCount++] = m_toneList[m_dictPhoneOffsets[entryIdx] + j];
      }
      distributePhones(nPh, numWords, m_tokPhoneCount + m_bertDistIdx);

    } else if (i == 0 || i == *count - 1) {
      distributePhones(1, 1, m_tokPhoneCount + m_bertDistIdx);
    } else {
      uint32_t sph[kMaxFallbackPhones] = {};
      uint32_t stn[kMaxFallbackPhones] = {};
      int pcount = 0;
      letterToPhones(upper, sph, stn, &pcount);
      for (int k = 0; k < pcount; ++k) {
        m_phones[m_phoneCount++] = 0;
        m_phones[m_phoneCount++] = sph[k];
        m_tones[m_toneCount++] = 0;
        m_tones[m_toneCount++] = stn[k];
      }
      distributePhones(pcount, numWords, m_tokPhoneCount + m_bertDistIdx);
    }

    m_bertDistIdx += static_cast<uint32_t>(numWords);
  }

  for (uint32_t i = 0; i < m_bertDistIdx; ++i)
    m_word2ph[i] = m_tokPhoneCount[i] * 2;
  m_word2ph[0] += 1;

  m_phones[m_phoneCount++] = 0;
  m_phones[m_phoneCount++] = 0;
  m_phones[m_phoneCount++] = 0;
  m_tones[m_toneCount++] = 0;
  m_tones[m_toneCount++] = 7;
  m_tones[m_toneCount++] = 0;

  return Status::Ok;
}

// ─── prepareSegment ──────────────────────────────────────────────────────────

Status Synthesizer::prepareSegment()
{
  uint32_t si = m_sentences.beginIdx[m_segIdx];
  uint32_t ei = m_sentences.endIdx[m_segIdx];
  uint32_t clen = ei - si + 1;

  char segText[kMaxTextBytes] = {};
  boundedCopy(segText, m_sentences.buffer + si, clen + 1);

  memset(m_phones, 0, sizeof(m_phones));
  memset(m_tones, 0, sizeof(m_tones));

  Status st = tokenize(segText);
  if (st != Status::Ok)
    return st;

  memset(m_groupOffsets, 0, sizeof(uint32_t) * kTokenSlots);
  memset(m_wordGroups, 0, sizeof(char) * kTokenSlots * kMaxPieceLen);
  memset(m_tokWordCount, 0, sizeof(uint8_t) * kTokenSlots);
  int groupCount = 0;

  st = groupTokens(m_wordGroups, m_groupOffsets, &groupCount);
  if (st != Status::Ok)
    return st;

  return buildPhoneSeq(m_wordGroups, m_groupOffsets, &groupCount);
}

// ─── buildAlignPath ──────────────────────────────────────────────────────────

void Synthesizer::buildAlignPath(const float * durations, uint32_t phonemes, uint32_t frames)
{
  for (uint32_t j = 0; j < kMaxFrames; ++j)
    m_yMask[j] = (j < frames) ? 1.0f : 0.0f;

  float cumDur[kMaxPhoneSeq + 1] = {};
  cumDur[0] = 0.0f;
  for (uint32_t i = 0; i < phonemes && i < kMaxPhoneSeq; ++i)
    cumDur[i + 1] = cumDur[i] + std::round(durations[i]);

  for (uint32_t i = 0; i < phonemes && i < kMaxPhoneSeq; ++i) {
    float prevCum = cumDur[i];
    float thisCum = cumDur[i + 1];
    for (uint32_t j = 0; j < kMaxFrames; ++j) {
      float curr = (j < static_cast<uint32_t>(thisCum)) ? 1.0f : 0.0f;
      float prev = (j < static_cast<uint32_t>(prevCum)) ? 1.0f : 0.0f;
      m_attn[j * kMaxPhoneSeq + i] = curr - prev;
    }
  }
}

// ─── runBert ─────────────────────────────────────────────────────────────────

Status Synthesizer::runBert()
{
  if (!m_runner)
    return Status::Ok;

  int numToks = m_outTokBuf.count;
  if (numToks <= 0)
    return Status::Ok;

  int32_t inputIds[200] = {};
  int32_t attnMask[200] = {};
  int32_t tokenTypes[200] = {};

  int n = std::min(numToks, 200);
  for (int i = 0; i < n; ++i) {
    inputIds[i] = static_cast<int32_t>(m_outTokBuf.tokens[i].tokenId);
    attnMask[i] = 1;
  }

  float hiddenStates[200 * kBertDim] = {};
  auto t0 = std::chrono::steady_clock::now();
  if (!m_runner->executeEmbedder(inputIds, attnMask, tokenTypes, hiddenStates))
    return Status::Fail;
  auto t1 = std::chrono::steady_clock::now();
  std::printf(
      "[timing] bert  : %6.2f ms\n", std::chrono::duration<double, std::milli>(t1 - t0).count());

  memset(m_bertEmb, 0, sizeof(m_bertEmb));
  uint32_t phoneIdx = 0;
  for (uint32_t tok = 0; tok < m_bertDistIdx; ++tok) {
    uint32_t phCount = m_word2ph[tok];
    for (uint32_t k = 0; k < phCount && phoneIdx < kMaxPhoneSeq; ++k, ++phoneIdx) {
      for (uint32_t dim = 0; dim < kBertDim; ++dim) {
        m_bertEmb[dim * kMaxPhoneSeq + phoneIdx] = hiddenStates[tok * kBertDim + dim];
      }
    }
  }
  return Status::Ok;
}

// ─── runEncoder ──────────────────────────────────────────────────────────────

Status Synthesizer::runEncoder()
{
  if (!m_runner)
    return Status::Ok;

  int32_t xArr[kMaxPhoneSeq] = {};
  int32_t tonesArr[kMaxPhoneSeq] = {};

  int32_t langId = 2;
  switch (m_langCode) {
    case 1:
      langId = 3;
      break;
    case 3:
      langId = 5;
      break;
    default:
      langId = 2;
      break;
  }

  for (uint32_t i = 0; i < kMaxPhoneSeq; ++i) {
    xArr[i] = static_cast<int32_t>(m_phones[i]);
    tonesArr[i] = static_cast<int32_t>(m_tones[i]);
  }

  float yLenOut = 0.0f;
  auto t0 = std::chrono::steady_clock::now();
  if (!m_runner->executeEncoder(xArr, static_cast<int32_t>(m_phoneCount), tonesArr, langId,
          static_cast<int32_t>(m_speakerId), m_bertEmb, m_sdpRatio, m_lengthScale, m_noiseScaleW,
          m_gVec, m_xMask, m_mP, m_logsP, m_wCeil, &yLenOut)) {
    return Status::Fail;
  }
  auto t1 = std::chrono::steady_clock::now();
  std::printf(
      "[timing] encoder: %6.2f ms\n", std::chrono::duration<double, std::milli>(t1 - t0).count());

  uint32_t computedFrames = 0;
  for (uint32_t i = 0; i < m_phoneCount; ++i)
    computedFrames += static_cast<uint32_t>(std::round(m_wCeil[i]));

  m_frameCount = computedFrames;
  m_totalFrames = computedFrames;
  m_decodeOffset = 0;

  memset(m_attn, 0, sizeof(m_attn));
  memset(m_yMask, 0, sizeof(m_yMask));
  buildAlignPath(m_wCeil, m_phoneCount, m_frameCount);
  return Status::Ok;
}

// ─── runFlow ─────────────────────────────────────────────────────────────────

Status Synthesizer::runFlow()
{
  if (!m_runner)
    return Status::Ok;

  auto t0 = std::chrono::steady_clock::now();
  if (!m_runner->executeFlow(m_mP, m_logsP, m_yMask, m_attn, m_gVec, m_noiseScale, m_zMat)) {
    return Status::Fail;
  }
  auto t1 = std::chrono::steady_clock::now();
  std::printf(
      "[timing] flow  : %6.2f ms\n", std::chrono::duration<double, std::milli>(t1 - t0).count());
  return Status::Ok;
}

// ─── runDecoder ──────────────────────────────────────────────────────────────

Status Synthesizer::runDecoder()
{
  if (!m_runner) {
    m_decFinished = true;
    return Status::Ok;
  }
  if (m_frameCount == 0) {
    m_decFinished = true;
    return Status::Ok;
  }

  const uint32_t kChunk = kDecoderWindow;
  const uint32_t kOvlap = kDecoderOverlap;
  const uint32_t kWin = kChunk + 2 * kOvlap;

  uint32_t off = m_decodeOffset;
  uint32_t wstart = (off >= kOvlap) ? (off - kOvlap) : 0;
  uint32_t wend = std::min(static_cast<uint32_t>(kMaxFrames), off + kChunk + kOvlap);
  uint32_t wsize = wend - wstart;

  float zWin[kLatentChannels * (kDecoderWindow + 2 * kDecoderOverlap)] = {};
  for (uint32_t ch = 0; ch < kLatentChannels; ++ch) {
    for (uint32_t t = 0; t < wsize; ++t)
      zWin[ch * kWin + t] = m_zMat[ch * kMaxFrames + wstart + t];
  }

  float audioOut[(kDecoderWindow + 2 * kDecoderOverlap) * kHopSamples] = {};
  auto t0 = std::chrono::steady_clock::now();
  if (!m_runner->executeDecoder(zWin, m_gVec, audioOut))
    return Status::Fail;
  auto t1 = std::chrono::steady_clock::now();
  std::printf("[timing] decoder: %6.2f ms  (offset=%u)\n",
      std::chrono::duration<double, std::milli>(t1 - t0).count(), off);

  m_decodeOffset += kChunk;
  m_decFinished = (m_decodeOffset >= m_totalFrames);

  uint32_t skipSamples = (off == 0) ? 0 : (kOvlap * kHopSamples);
  uint32_t validSamples = kChunk * kHopSamples;

  if (m_decFinished) {
    uint32_t lastFrames = m_totalFrames % kChunk;
    if (lastFrames != 0)
      validSamples = lastFrames * kHopSamples;
  }

  uint32_t pcmBytes = validSamples * sizeof(int16_t);
  if (m_audioBytes + pcmBytes > kMaxAudioBytes)
    pcmBytes = kMaxAudioBytes - m_audioBytes;

  int16_t * dst = reinterpret_cast<int16_t *>(m_audioBuf + m_audioBytes);
  for (uint32_t i = 0; i < pcmBytes / sizeof(int16_t); ++i) {
    dst[i] = static_cast<int16_t>(std::round(audioOut[skipSamples + i] * 32767.0f));
  }
  m_audioBytes += pcmBytes;
  return Status::Ok;
}

// ─── runPipeline ─────────────────────────────────────────────────────────────

Status Synthesizer::runPipeline()
{
  Status st;

  if (m_decFinished) {
    if (m_bertOn) {
      st = runBert();
      if (st != Status::Ok)
        return st;
    }
    st = runEncoder();
    if (st != Status::Ok)
      return st;
    st = runFlow();
    if (st != Status::Ok)
      return st;
  }

  return runDecoder();
}

// ─── boot ────────────────────────────────────────────────────────────────────

Status Synthesizer::boot(const Config & cfg, const char * tokPath, const char * normPath)
{
  if (cfg.languageCode != 0)
    return Status::NotSupported;

  m_modelGen = cfg.modelGeneration;
  m_langCode = tts::dict_en::kInputLanguageCode;
  m_bertOn = tts::dict_en::kIsBertEnabled;
  m_speakerId = tts::dict_en::kSid;
  m_sdpRatio = tts::dict_en::kSdpRatio;
  m_noiseScale = tts::dict_en::kNoiseScale;
  m_noiseScaleW = tts::dict_en::kNoiseScaleW;

  m_sampleRate = kDefaultSampleRate;
  m_bitsPerSample = 16;
  m_numChannels = 1;
  m_numFormats = 1;
  m_speechFormat = 0;

  m_dictWordCount = tts::dict_en::kDictNumWords;
  m_dictWordBytes = tts::dict_en::kWordListSize;
  m_dictPhoneCount = tts::dict_en::kDictNumPhones;
  m_dictPhoneOffCnt = tts::dict_en::kDictNumPhonesOffset;
  m_dictWordOffsets = tts::dict_en::kDictWordOffset;
  m_dictWords = tts::dict_en::kDictAllWords;
  m_dictPhoneOffsets = tts::dict_en::kDictPhonesOffset;
  m_phoneList = tts::dict_en::kPhonesList;
  m_toneList = tts::dict_en::kTonesList;
  m_phonesPerWord = tts::dict_en::kPhonesPerWord;

  m_replCount = tts::dict_en::kNumReplEntries;
  m_replOrigOff = tts::dict_en::kOrigWordOffset;
  m_replOrigWords = tts::dict_en::kOrigWordList;
  m_replReplOff = tts::dict_en::kReplWordOffset;
  m_replReplWords = tts::dict_en::kReplWordList;

  loadFiles(tokPath, normPath);

  m_hwCfg.encodingMode = static_cast<uint32_t>(cfg.encoding);
  m_hwCfg.speechRate = cfg.speechRate;
  m_hwCfg.pitch = cfg.pitch;
  m_hwCfg.volumeGain = cfg.volumeGain;
  m_hwCfg.sampleRate = cfg.sampleRate;
  m_hwCfg._reserved = 0;
  m_hwCfg.hwContext = nullptr;

  m_decFinished = true;
  m_pipelineDone = false;
  m_segIdx = 0;

  if (!cfg.bertModelPath.empty() && !cfg.encoderModelPath.empty() && !cfg.flowModelPath.empty() &&
      !cfg.decoderModelPath.empty()) {
    InferenceRunner::ModelPaths paths;
    paths.bertPath = cfg.bertModelPath;
    paths.encoderPath = cfg.encoderModelPath;
    paths.flowPath = cfg.flowModelPath;
    paths.decoderPath = cfg.decoderModelPath;
    if (!cfg.backendLibPath.empty())
      paths.backendLib = cfg.backendLibPath;
    if (!cfg.systemLibPath.empty())
      paths.systemLib = cfg.systemLibPath;

    m_runner = new InferenceRunner();
    if (!m_runner->setup(paths)) {
      delete m_runner;
      m_runner = nullptr;
      return Status::Fail;
    }
  }

  return Status::Ok;
}

}  // namespace audio
