// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear

#include "bert.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstring>
#include <string_view>

#include "melo_bert_compat.hpp"

namespace
{

// ── UTF-8 helpers
// ──────────────────────────────────────────────────────────────

struct Utf8
{
  static uint32_t decode(const uint8_t * src, int width)
  {
    switch (width) {
      case 1:
        return src[0];
      case 2:
        return ((src[0] & 0x1Fu) << 6) | (src[1] & 0x3Fu);
      case 3:
        return ((src[0] & 0x0Fu) << 12) | ((src[1] & 0x3Fu) << 6) | (src[2] & 0x3Fu);
      default:
        return ((src[0] & 0x07u) << 18) | ((src[1] & 0x3Fu) << 12) | ((src[2] & 0x3Fu) << 6) |
               (src[3] & 0x3Fu);
    }
  }

  static int encode(uint32_t cp, uint8_t * dst)
  {
    if (cp <= 0x007F) {
      dst[0] = static_cast<uint8_t>(cp);
      return 1;
    } else if (cp <= 0x07FF) {
      dst[0] = static_cast<uint8_t>(0xC0u | (cp >> 6 & 0x1Fu));
      dst[1] = static_cast<uint8_t>(0x80u | (cp & 0x3Fu));
      return 2;
    } else if (cp <= 0xFFFF) {
      dst[0] = static_cast<uint8_t>(0xE0u | (cp >> 12 & 0x0Fu));
      dst[1] = static_cast<uint8_t>(0x80u | (cp >> 6 & 0x3Fu));
      dst[2] = static_cast<uint8_t>(0x80u | (cp & 0x3Fu));
      return 3;
    } else if (cp <= 0x10FFFF) {
      dst[0] = static_cast<uint8_t>(0xF0u | (cp >> 18 & 0x07u));
      dst[1] = static_cast<uint8_t>(0x80u | (cp >> 12 & 0x3Fu));
      dst[2] = static_cast<uint8_t>(0x80u | (cp >> 6 & 0x3Fu));
      dst[3] = static_cast<uint8_t>(0x80u | (cp & 0x3Fu));
      return 4;
    }
    MELO_ERRLOG("Codepoint 0x%X out of range\n", cp);
    return 0;
  }

  static int scan(uint8_t * dst, const uint8_t * src, bool * trunc)
  {
    const uint8_t b = src[0];
    int n;
    if ((b >> 7) == 0x00)
      n = 1;
    else if ((b >> 5) == 0x06)
      n = 2;
    else if ((b >> 4) == 0x0E)
      n = 3;
    else if ((b >> 3) == 0x1E)
      n = 4;
    else {
      if ((b >> 6) == 0x02 && trunc)
        *trunc = true;
      return 0;
    }
    if (n > 1 && trunc)
      for (int i = 1; i < n; ++i)
        if (src[i] == '\0') {
          *trunc = true;
          return n;
        }
    for (int i = 0; i < n; ++i)
      dst[i] = src[i];
    return n;
  }
};

// ── CJK block membership
// ───────────────────────────────────────────────────────

struct CpRange
{
  uint32_t lo, hi;
};

static constexpr std::array<CpRange, 8> kCjkRanges{ {
    { 0x4E00, 0x9FFF },
    { 0x3400, 0x4DBF },
    { 0x20000, 0x2A6DF },
    { 0x2A700, 0x2B73F },
    { 0x2B740, 0x2B81F },
    { 0x2B820, 0x2CEAF },
    { 0xF900, 0xFAFF },
    { 0x2F800, 0x2FA1F },
} };

static bool isCjkBlock(uint32_t cp)
{
  for (const auto & r : kCjkRanges)
    if (cp >= r.lo && cp <= r.hi)
      return true;
  return false;
}

// ── Punctuation table
// ─────────────────────────────────────────────────────────

static constexpr const char * kPunctTable[] = {
  "!",
  "\"",
  "#",
  "$",
  "%",
  "&",
  "'",
  "(",
  ")",
  "*",
  "+",
  ",",
  "-",
  ".",
  "/",
  ":",
  ";",
  "<",
  "=",
  ">",
  "?",
  "@",
  "[",
  "\\",
  "]",
  "^",
  "_",
  "`",
  "{",
  "|",
  "}",
  "~",
  "\xE2\x80\x98",
  "\xE2\x80\x99",
  "\xE2\x80\x9A",
  "\xE2\x80\x9C",
  "\xE2\x80\x9D",
  "\xE2\x80\x9E",
  "\xE2\x80\xB2",
  "\xE2\x80\xB3",
  "\xE2\x81\x84",
  "\xEF\xBC\x81",
  "\xEF\xBC\x9A",
  "\xEF\xBC\x9F",
  "\xEF\xBD\x9E",
  "\xC2\xBF",
  "\xE3\x80\x82",
  "\xEF\xBC\x8C",
  "\xC2\xA1",
  "\xE2\x80\x94",
  "\xE2\x80\xA6",
  "\xEF\xBC\x8D",
  "\xE2\x80\x93",
  "\xEF\xBC\x88",
  "\xEF\xBC\x89",
  "\xE3\x80\x8A",
  "\xE3\x80\x8B",
  "\xE3\x80\x8C",
  "\xE3\x80\x8D",
  "\xE3\x80\x8E",
  "\xE3\x80\x8F",
};
static constexpr int kPunctCount = static_cast<int>(sizeof(kPunctTable) / sizeof(kPunctTable[0]));

static bool isPunct(const uint8_t * ch)
{
  std::string_view sv{ reinterpret_cast<const char *>(ch) };
  for (int i = 0; i < kPunctCount; ++i)
    if (sv == kPunctTable[i])
      return true;
  return false;
}

// ── Spanish uppercase accent → lowercase substitution
// ─────────────────────────

using AccentPair = std::pair<std::string_view, std::string_view>;

static constexpr AccentPair kEsAccentMap[] = {
  { "\xC3\x81", "\xC3\xA1" },
  { "\xC3\x89", "\xC3\xA9" },
  { "\xC3\x8D", "\xC3\xAD" },
  { "\xC3\x93", "\xC3\xB3" },
  { "\xC3\x9A", "\xC3\xBA" },
  { "\xC3\x91", "\xC3\xB1" },
  { "\xC3\x9C", "\xC3\xBC" },
};

static bool esAccentToLower(const uint8_t * ch, uint8_t * out, int * outLen)
{
  std::string_view sv{ reinterpret_cast<const char *>(ch) };
  for (const auto & [cap, low] : kEsAccentMap) {
    if (sv != cap)
      continue;
    if (out && outLen) {
      bool trunc = false;
      *outLen = Utf8::scan(out, reinterpret_cast<const uint8_t *>(low.data()), &trunc);
    }
    return true;
  }
  return false;
}

// ── Vocab bank lookup
// ───────────────────────────────────────────────────────── Returns the record
// index in bank matching probe, or -1.

static int lookupInBank(const uint8_t * probe, const VocabBank * bank, int pLen)
{
  const int lo = (pLen < bank->cacheLen) ? static_cast<int>(bank->byteCache[pLen]) : 0;
  const int hi =
      (pLen + 1 < bank->cacheLen) ? static_cast<int>(bank->byteCache[pLen + 1]) : bank->count;

  std::string_view target{ reinterpret_cast<const char *>(probe) };
  for (int k = lo; k < hi; ++k)
    if (target == (bank->pool + bank->records[k].offset))
      return k;
  return -1;
}

// ── Unicode category helpers
// ──────────────────────────────────────────────────

static bool nodeIsSpace(const UnicodeCrumb * n)
{
  std::string_view cat{ n->category, 2 };
  return (n->codepoint >= 0x09 && n->codepoint <= 0x0D) || n->codepoint == 0x85 || cat == "Zs" ||
         cat == "Zl" || cat == "Zp";
}

static bool nodeIsAccent(const UnicodeCrumb * n)
{
  return std::string_view{ n->category, 2 } == "Mn";
}

// ── Special token tables
// ───────────────────────────────────────────────────────

struct SpecToken
{
  const char * text;
  uint32_t id;
};

static constexpr SpecToken kSpecEN[] = { { "[PAD]", 0 }, { "[UNK]", 100 }, { "[CLS]", 101 },
  { "[SEP]", 102 } };
static constexpr SpecToken kSpecES[] = { { "[PAD]", 1 }, { "[UNK]", 3 }, { "[CLS]", 4 },
  { "[SEP]", 5 } };

static const SpecToken * specTableFor(int32_t lang)
{
  return (lang == melo_es) ? kSpecES : kSpecEN;
}

// ── Emit UNK token into a slot
// ────────────────────────────────────────────────

static void emitUnkPiece(TokenPiece & slot, int32_t lang, const TokenPiece & src)
{
  if (lang == melo_zh) {
    uint8_t ch[kUtf8MaxBytes + 1] = {};
    bool trunc = false;
    int nb = Utf8::scan(ch, src.text, &trunc);
    ch[nb] = '\0';
    strlcpy(reinterpret_cast<char *>(slot.text), reinterpret_cast<const char *>(ch), kTokenByteCap);
    slot.textLen = src.textLen;
    slot.tokenId = kSpecEN[1].id;
  } else {
    const SpecToken * tab = specTableFor(lang);
    strlcpy(reinterpret_cast<char *>(slot.text), tab[1].text, kTokenByteCap);
    slot.textLen = static_cast<int32_t>(strlen(tab[1].text));
    slot.tokenId = tab[1].id;
  }
}

// ── Copy a UTF-8 slice from src[start..end) into dst ─────────────────────────
// Returns byte count written; *lastWidth receives width of the last char.

static int copyUtf8Slice(const uint8_t * src, int start, int end, uint8_t * dst, int * lastWidth)
{
  int total = 0, last = 0;
  for (int idx = start; idx < end;) {
    bool trunc = false;
    int nb = Utf8::scan(dst + total, src + idx, &trunc);
    if (nb == 0) {
      ++idx;
      continue;
    }
    last = nb;
    total += nb;
    idx += nb;
  }
  dst[total] = '\0';
  if (lastWidth)
    *lastWidth = last;
  return total;
}

// ── Find the longest vocab prefix of src.text[at..src.textLen) ───────────────
// Fills `slot` on success and returns the end position (new `at`).
// Returns -1 when no prefix of any length is in the vocab bank.

static int findLongestMatch(const TokenPiece & src,
    int at,
    const VocabBank * bank,
    bool continuation,
    TokenPiece & slot)
{
  uint8_t probe[kTokenByteCap] = {};
  const int prefixLen = continuation ? 2 : 0;
  if (continuation) {
    probe[0] = '#';
    probe[1] = '#';
  }

  int back = src.textLen;
  int lastW = 0;

  while (back > at) {
    copyUtf8Slice(src.text, at, back, probe + prefixLen, &lastW);
    int pLen = static_cast<int>(strlen(reinterpret_cast<const char *>(probe)));
    int idx = lookupInBank(probe, bank, pLen);

    if (idx >= 0) {
      strlcpy(reinterpret_cast<char *>(slot.text), reinterpret_cast<const char *>(probe),
          kTokenByteCap);
      slot.textLen = static_cast<int32_t>(bank->records[idx].nBytes);
      slot.tokenId = bank->records[idx].id;
      return back;
    }

    if (lastW == 0)
      break;
    back -= lastW;
    memset(probe + prefixLen, 0, sizeof(probe) - prefixLen);
  }
  return -1;
}

// ── WordPiece tokenizer
// ───────────────────────────────────────────────────────

static int runWordPiece(const TokenBuffer * pre,
    const VocabBank * starts,
    const VocabBank * conts,
    TokenBuffer * out,
    int baseOff,
    int32_t lang)
{
  int placed = 0;

  for (int ti = 0; ti < pre->count; ++ti) {
    const TokenPiece & src = pre->tokens[ti];
    int at = 0;
    int firstSlot = placed;

    while (at < src.textLen) {
      const bool cont = (at > 0);
      const VocabBank * bank = cont ? conts : starts;

      int newAt = findLongestMatch(src, at, bank, cont, out->tokens[placed + baseOff]);
      if (newAt > 0) {
        ++placed;
        at = newAt;
      } else {
        if (lang == melo_zh) {
          emitUnkPiece(out->tokens[placed + baseOff], lang, src);
          ++placed;
        } else {
          emitUnkPiece(out->tokens[firstSlot + baseOff], lang, src);
          placed = firstSlot + 1;
          memset(&out->tokens[placed + baseOff], 0,
              (kStreamCap - placed - baseOff) * sizeof(TokenPiece));
        }
        break;
      }
    }
  }

  out->count = placed;
  return 0;
}

// ── Recursive decomposition
// ──────────────────────────────────────────────────── Traverses the Unicode
// NFD tree recursively (pre-order, left to right). Returns true when the caller
// should emit the original codepoint instead (i.e. when a non-spacing mark is
// encountered and stripAccents is false).

static bool decomposeRec(const UnicodeCrumb * node,
    const UnicodeCrumb * tree,
    const uint32_t * data,
    bool stripAccents,
    char * leafAcc,
    int leafCap)
{
  if (node->childCount == 0) {
    if (nodeIsSpace(node)) {
      strlcat(leafAcc, " ", leafCap);
      return false;
    }
    if (nodeIsAccent(node)) {
      return !stripAccents;
    }
    uint8_t ch[kUtf8MaxBytes + 1] = {};
    int nb = Utf8::encode(node->codepoint, ch);
    if (nb == 0)
      return false;
    ch[nb] = '\0';
    strlcat(leafAcc, reinterpret_cast<const char *>(ch), leafCap);
    return false;
  }

  const uint32_t * childIdx = data + (node->childOffset / sizeof(uint32_t));
  for (int ci = 0; ci < node->childCount; ++ci) {
    if (decomposeRec(&tree[childIdx[ci]], tree, data, stripAccents, leafAcc, leafCap))
      return true;
  }
  return false;
}

static int
resolveCodepoint(uint32_t cp, uint8_t * out, const NormTree * normTree, bool stripAccents)
{
  const UnicodeCrumb * tree = static_cast<const UnicodeCrumb *>(normTree->root);
  const uint32_t * data = normTree->leaf;

  const uint32_t * rootChildren = data + (tree->childOffset / sizeof(uint32_t));
  int lo = 0, hi = tree->childCount - 1;

  while (lo <= hi) {
    int mid = lo + (hi - lo) / 2;
    const UnicodeCrumb * node = &tree[rootChildren[mid]];

    if (cp == static_cast<uint32_t>(node->codepoint)) {
      char leafAcc[64] = {};
      bool keepOrig = decomposeRec(node, tree, data, stripAccents, leafAcc, sizeof(leafAcc));
      if (keepOrig) {
        uint8_t ch[kUtf8MaxBytes + 1] = {};
        int nb = Utf8::encode(cp, ch);
        ch[nb] = '\0';
        strlcat(reinterpret_cast<char *>(out), reinterpret_cast<const char *>(ch), kNormBufCap);
      } else {
        strlcat(reinterpret_cast<char *>(out), leafAcc, kNormBufCap);
      }
      return 0;
    }

    if (cp > static_cast<uint32_t>(node->codepoint))
      lo = mid + 1;
    else
      hi = mid - 1;
  }

  uint8_t ch[kUtf8MaxBytes + 1] = {};
  int nb = Utf8::encode(cp, ch);
  ch[nb] = '\0';
  strlcat(reinterpret_cast<char *>(out), reinterpret_cast<const char *>(ch), kNormBufCap);
  return 0;
}

}  // anonymous namespace

// ── BertPipeline method implementations
// ───────────────────────────────────────

int BertPipeline::normalize(uint8_t * text) const
{
  const int textLen = static_cast<int>(strlen(reinterpret_cast<const char *>(text)));
  const bool strip = true;
  uint8_t result[MELO_MAX_TTS_CHAR_SIZE] = {};

  for (int i = 0; i < textLen;) {
    uint8_t ch[kUtf8MaxBytes] = {};
    bool trunc = false;
    int nb = Utf8::scan(ch, text + i, &trunc);
    if (nb == 0) {
      ++i;
      continue;
    }

    uint32_t cp = Utf8::decode(ch, nb);
    uint8_t charBuf[kNormBufCap] = {};

    if (resolveCodepoint(cp, charBuf, &m_norm, strip) != 0)
      return -1;

    strlcat(reinterpret_cast<char *>(result), reinterpret_cast<const char *>(charBuf),
        MELO_MAX_TTS_CHAR_SIZE);
    i += nb;
  }

  strlcpy(reinterpret_cast<char *>(text), reinterpret_cast<const char *>(result),
      MELO_MAX_TTS_CHAR_SIZE);
  return 0;
}

int BertPipeline::preTokenize(const uint8_t * text) const
{
  const int textLen = static_cast<int>(strlen(reinterpret_cast<const char *>(text)));

  uint8_t accum[kTokenByteCap] = {};
  int accumLen = 0;
  int count = 0;
  bool inWord = false;

  auto flush = [&]() {
    if (accumLen > 0 && count < static_cast<int>(kStreamCap)) {
      TokenPiece & slot = m_staging->tokens[count++];
      memcpy(slot.text, accum, accumLen);
      slot.text[accumLen] = '\0';
      slot.textLen = accumLen;
      memset(accum, 0, accumLen);
      accumLen = 0;
    }
  };

  auto emitSingle = [&](const uint8_t * ch, int nb) {
    if (count < static_cast<int>(kStreamCap)) {
      TokenPiece & slot = m_staging->tokens[count++];
      memcpy(slot.text, ch, nb);
      slot.text[nb] = '\0';
      slot.textLen = nb;
    }
  };

  for (int i = 0; i < textLen;) {
    bool trunc = false;
    uint8_t ch[kUtf8MaxBytes + 1] = {};
    int nb = Utf8::scan(ch, text + i, &trunc);

    if (trunc) {
      ++i;
      continue;
    }
    if (nb == 0) {
      MELO_ERRLOG("Non-UTF8 byte at offset %d\n", i);
      return -1;
    }
    ch[nb] = '\0';

    const uint32_t cp = Utf8::decode(ch, nb);

    if (ch[0] == ' ') {
      if (inWord) {
        flush();
        inWord = false;
      }
    } else if (isPunct(ch) || isCjkBlock(cp)) {
      if (inWord) {
        flush();
        inWord = false;
      }
      emitSingle(ch, nb);
    } else {
      if (m_lang == melo_es) {
        uint8_t lower[kUtf8MaxBytes] = {};
        int lLen = 0;
        if (esAccentToLower(ch, lower, &lLen)) {
          if (accumLen + lLen < static_cast<int>(kTokenByteCap)) {
            memcpy(accum + accumLen, lower, lLen);
            accumLen += lLen;
          } else {
            MELO_ERRLOG("Segment overflow (ES accent)\n");
            return -1;
          }
          inWord = true;
          i += nb;
          continue;
        }
      }
      if (accumLen + nb < static_cast<int>(kTokenByteCap)) {
        memcpy(accum + accumLen, text + i, nb);
        accumLen += nb;
      } else {
        MELO_ERRLOG("Segment overflow at offset %d\n", i);
        return -1;
      }
      inWord = true;
    }
    i += nb;
  }

  if (inWord)
    flush();
  m_staging->count = count;
  return 0;
}

int BertPipeline::tokenize(int leadSlots) const
{
  return -runWordPiece(m_staging, &m_vocabStart, &m_vocabCont, m_output, leadSlots, m_lang);
}

void BertPipeline::addSpecialTokens(int lead, int trail) const
{
  if (m_lang != melo_en && m_lang != melo_zh && m_lang != melo_es)
    return;

  const SpecToken * tab = specTableFor(m_lang);

  TokenPiece & clsPiece = m_output->tokens[0];
  strlcpy(reinterpret_cast<char *>(clsPiece.text), tab[2].text, kTokenByteCap);
  clsPiece.textLen = static_cast<int32_t>(strlen(tab[2].text));
  clsPiece.tokenId = tab[2].id;

  TokenPiece & sepPiece = m_output->tokens[m_output->count + lead];
  strlcpy(reinterpret_cast<char *>(sepPiece.text), tab[3].text, kTokenByteCap);
  sepPiece.textLen = static_cast<int32_t>(strlen(tab[3].text));
  sepPiece.tokenId = tab[3].id;

  m_output->count += (lead + trail);
}
