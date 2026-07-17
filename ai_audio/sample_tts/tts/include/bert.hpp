// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear

#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "melo_bert_compat.hpp"

// ── compile-time capacities
// ────────────────────────────────────────────────────

inline constexpr uint32_t kTokenByteCap = 128;
inline constexpr uint32_t kStreamCap = 4096;
inline constexpr uint32_t kNormBufCap = 20;
inline constexpr uint32_t kUtf8MaxBytes = 4;

// ── tokenizer variety / algorithm
// ─────────────────────────────────────────────

enum class VocabKind : int32_t
{
  GPT2 = 0,
  MBart50 = 1,
  BertCased = 2,
  Unknown = 3,
};

enum class SplitKind : int32_t
{
  BytePair = 0,
  WordPiece = 1,
  Unigram = 2,
  Unknown = 3,
};

// ── single tokenised piece
// ──────────────────────────────────────────────────── 136 bytes: text[128] +
// textLen(4) + tokenId(4)

struct TokenPiece
{
  uint8_t text[kTokenByteCap] = {};
  int32_t textLen = 0;
  uint32_t tokenId = 0;
};

// ── flat array of token pieces
// ────────────────────────────────────────────────

struct TokenBuffer
{
  int32_t count = 0;
  TokenPiece tokens[kStreamCap] = {};
};

// ── compact vocab record (binary-file format; must stay 16 bytes)
// ───────────── Layout: offset(4) + id(4) + nChars(4) + nBytes(4)

struct VocabRecord
{
  uint32_t offset = 0;
  uint32_t id = 0;
  uint32_t nChars = 0;
  uint32_t nBytes = 0;
};

// ── unicode NFD normalizer tree reference
// ───────────────────────────────────── Layout: root(8) + leaf(8) = 16 bytes

struct NormTree
{
  void * root = nullptr;
  uint32_t * leaf = nullptr;
};

// ── vocabulary bank (pool + index for one subword role)
// ──────────────────────── Provides the three arrays needed for
// byte-length-indexed vocab lookup.

struct VocabBank
{
  const char * pool = nullptr;            // string pool
  const VocabRecord * records = nullptr;  // per-token metadata
  const uint32_t * byteCache = nullptr;   // cache[n] = first index of byte-len-n tokens
  int32_t count = 0;
  int32_t cacheLen = 0;
};

// ── unicode tree node (binary-compatible with old UnicodeNode)
// ───────────────── 16 bytes: codepoint(4) + category[2](2) + pad(2) +
// childCount(4) + childOffset(4)

struct UnicodeCrumb
{
  int32_t codepoint = 0;
  char category[2] = {};
  int32_t childCount = 0;
  int32_t childOffset = 0;
};

// ── tokenizer pipeline
// ────────────────────────────────────────────────────────
//
// Encapsulates all state for one tokenisation session.
// Call the set*() methods once at initialisation, then call the execution
// methods once per utterance in order: normalize → preTokenize → tokenize
// → addSpecialTokens.

class BertPipeline
{
public:
  // ── initialisation ─────────────────────────────────────────────────────

  void setLanguage(int32_t lang) { m_lang = lang; }

  void setNorm(void * root, uint32_t * leaf) { m_norm = { root, leaf }; }

  void setVocabStart(const char * pool,
      const uint32_t * cache,
      const VocabRecord * rec,
      int count,
      int cacheLen)
  {
    m_vocabStart = { pool, rec, cache, count, cacheLen };
  }

  void setVocabCont(const char * pool,
      const uint32_t * cache,
      const VocabRecord * rec,
      int count,
      int cacheLen)
  {
    m_vocabCont = { pool, rec, cache, count, cacheLen };
  }

  void setBuffers(TokenBuffer * staging, TokenBuffer * output)
  {
    m_staging = staging;
    m_output = output;
  }

  // ── per-utterance execution (call in this order) ───────────────────────

  int normalize(uint8_t * text) const;
  int preTokenize(const uint8_t * text) const;
  int tokenize(int leadSlots) const;
  void addSpecialTokens(int lead, int trail) const;

  // ── accessors ─────────────────────────────────────────────────────────

  TokenBuffer * outputStream() { return m_output; }
  const TokenBuffer * outputStream() const { return m_output; }
  int32_t language() const { return m_lang; }

private:
  int32_t m_lang = 0;
  NormTree m_norm = {};
  VocabBank m_vocabStart = {};
  VocabBank m_vocabCont = {};
  TokenBuffer * m_staging = nullptr;
  TokenBuffer * m_output = nullptr;
};
