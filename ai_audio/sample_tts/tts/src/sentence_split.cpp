// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear

#include "sentence_split.hpp"

#include <cctype>
#include <cstdint>
#include <cstring>

namespace tts
{

void split_struct_reset(SplitSentence & s)
{
  s.segmentCount = 0;
  memset(s.buffer, 0, sizeof(s.buffer));
  memset(s.beginIdx, 0, sizeof(s.beginIdx));
  memset(s.endIdx, 0, sizeof(s.endIdx));
}

// ─── Special character replacement table ─────────────────────────────────────

struct CharSubst
{
  const char * seq;
  int seqLen;
  char repl;
};  // repl='\0' → drop

static const CharSubst kSpecialChars[] = {
  { "\xE3\x80\x82", 3, '.' },   // U+3002 。→ .
  { "\xEF\xBC\x81", 3, '.' },   // U+FF01 ！→ .
  { "\xEF\xBC\x9F", 3, '.' },   // U+FF1F ？→ .
  { "\xEF\xBC\x9B", 3, '.' },   // U+FF1B ；→ .
  { "\xEF\xBC\x8C", 3, ',' },   // U+FF0C ，→ ,
  { "\xC2\xAB", 2, '\0' },      // U+00AB «  → drop
  { "\xC2\xBB", 2, '\0' },      // U+00BB »  → drop
  { "\xE2\x80\x9C", 3, '\0' },  // U+201C "  → drop
  { "\xE2\x80\x9D", 3, '\0' },  // U+201D "  → drop
  { "\xE2\x80\x98", 3, '\'' },  // U+2018 '  → apostrophe
  { "\xE2\x80\x99", 3, '\'' },  // U+2019 '  → apostrophe
};
static constexpr int kNumSpecial =
    static_cast<int>(sizeof(kSpecialChars) / sizeof(kSpecialChars[0]));

// ─── Normalize special/bracket chars in-place ────────────────────────────────

static void normalizeSpecials(char * text, uint16_t len)
{
  int ri = 0, wi = 0;
  while (ri < static_cast<int>(len)) {
    unsigned char c = static_cast<unsigned char>(text[ri]);
    if (c > 127) {
      bool matched = false;
      for (int k = 0; k < kNumSpecial; ++k) {
        const CharSubst & cs = kSpecialChars[k];
        if (strncmp(text + ri, cs.seq, cs.seqLen) == 0) {
          if (cs.repl != '\0')
            text[wi++] = cs.repl;
          ri += cs.seqLen;
          matched = true;
          break;
        }
      }
      if (!matched)
        text[wi++] = text[ri++];
    } else if (text[ri] == '<' || text[ri] == '>' || text[ri] == '(' || text[ri] == ')' ||
               text[ri] == '[' || text[ri] == ']' || text[ri] == '"') {
      ++ri;
    } else {
      text[wi++] = text[ri++];
    }
  }
  if (wi < len)
    text[wi++] = '\0';
  while (wi < len)
    text[wi++] = ' ';
}

// ─── Compact punctuation spacing ─────────────────────────────────────────────

static void compactPunct(const char * in, char * out, uint16_t inLen, uint16_t * outLen)
{
  uint16_t ii = 0, oi = 0;
  while (ii < inLen) {
    char c = in[ii];
    char nc = (ii + 1 < inLen) ? in[ii + 1] : '\0';

    if (c == '\n') {
      if (oi > 0 && out[oi - 1] != '\n')
        out[oi++] = in[ii];
      ++ii;
    } else if (c == ' ') {
      if (oi > 0 && out[oi - 1] != ' ')
        out[oi++] = in[ii];
      ++ii;
    } else if (c == '"' && nc == '"') {
      out[oi++] = c;
      ii += 2;
    } else if ((c == ',' || c == '.' || c == '?' || c == '!') && nc != ' ') {
      out[oi++] = in[ii++];
      out[oi++] = ' ';
    } else {
      out[oi++] = in[ii++];
    }
  }
  out[oi] = '\0';
  *outLen = static_cast<uint16_t>(strlen(out));
}

// ─── Boundary scanner state
// ───────────────────────────────────────────────────

struct BoundaryState
{
  char * txt;
  uint16_t * beg;
  uint16_t * end;
  uint16_t seg = 0;
  int16_t mark = -1;
  bool inQ = false;

  char step(int16_t & pos, int delta)
  {
    int dir = (delta < 0) ? -1 : 1;
    for (int i = 0, n = delta * dir; i < n; ++i) {
      pos += dir;
      if (txt[pos] == '"')
        inQ = !inQ;
    }
    return txt[pos];
  }

  void flush(int16_t pos)
  {
    end[seg] = static_cast<uint16_t>(pos);
    ++seg;
    beg[seg] = static_cast<uint16_t>(pos + 1);
    mark = -1;
  }
};

// ─── Scan sentence boundaries
// ─────────────────────────────────────────────────

static void scanBoundaries(BoundaryState & bs, uint16_t len, uint16_t minLen, uint16_t maxLen)
{
  int16_t pos = -1;

  while (pos < static_cast<int16_t>(len)) {
    char c = bs.step(pos, 1);
    int16_t segLen = pos - static_cast<int16_t>(bs.beg[bs.seg]) + 1;

    if (segLen >= static_cast<int16_t>(maxLen)) {
      if (bs.mark > -1 && segLen > static_cast<int16_t>(minLen) / 2) {
        int16_t d = pos - bs.mark;
        bs.step(pos, -d);
      } else {
        while (c != '!' && c != '?' && c != '.' && c != '\n' && c != ' ' && pos > 0 &&
               (pos - bs.beg[bs.seg] + 1 > static_cast<int16_t>(minLen))) {
          c = bs.step(pos, -1);
        }
      }
      bool adj = false;
      while ((static_cast<uint8_t>(c) & 0xC0) == 0x80) {
        c = bs.step(pos, -1);
        adj = true;
      }
      if ((static_cast<uint8_t>(c) >= 0xC0)) {
        c = bs.step(pos, -1);
        adj = true;
      }
      bs.flush(pos);
      if (adj)
        bs.step(pos, 3);

    } else if (!bs.inQ && ((c == '!' || c == '?' || c == '\n') ||
                              ((c == '.' || c == ',') &&
                                  (bs.txt[pos + 1] == '\n' || bs.txt[pos + 1] == ' ')))) {
      while (pos < len && (pos - bs.beg[bs.seg] + 1) < static_cast<int16_t>(maxLen) &&
             (bs.txt[pos + 1] == '!' || bs.txt[pos + 1] == '?' || bs.txt[pos + 1] == '.'))
        bs.step(pos, 1);

      bs.mark = pos;

      if ((pos - bs.beg[bs.seg] + 1) >= static_cast<int16_t>(minLen)) {
        bool adj = false;
        while ((static_cast<uint8_t>(c) & 0xC0) == 0x80) {
          c = bs.step(pos, -1);
          adj = true;
        }
        if ((static_cast<uint8_t>(c) >= 0xC0)) {
          c = bs.step(pos, -1);
          adj = true;
        }
        bs.flush(pos);
        if (adj)
          bs.step(pos, 3);
      }

    } else if (bs.inQ && bs.txt[pos + 1] == '"' &&
               (bs.txt[pos + 2] == '\n' || bs.txt[pos + 2] == ' ')) {
      bs.step(pos, 2);
      bs.mark = pos;
    }
  }

  if (bs.seg == 0) {
    bs.end[0] = len;
    bs.seg = 1;
  } else if (bs.beg[bs.seg] != 0 && bs.end[bs.seg] == 0) {
    bs.end[bs.seg] = len;
    ++bs.seg;
  }
}

// ─── Trim sentence edges
// ──────────────────────────────────────────────────────

static void trimEdges(SplitSentence & s)
{
  for (uint32_t i = 0; i < s.segmentCount; ++i) {
    auto & b = s.beginIdx[i];
    auto & e = s.endIdx[i];
    while (s.buffer[b] == ' ')
      ++b;
    while (e > b && s.buffer[e] == ' ')
      --e;
  }
}

// ─── Remove sentences with no content chars
// ───────────────────────────────────

static void pruneEmpty(SplitSentence & s)
{
  uint32_t valid = 0;
  for (uint32_t i = 0; i < s.segmentCount; ++i) {
    bool ok = false;
    for (int p = s.beginIdx[i]; p < s.endIdx[i] && !ok; ++p) {
      char c = s.buffer[p];
      ok = (c != ' ' && c != '.' && c != ',' && c != ';' && c != ':' && c != '!' && c != '?');
    }
    if (ok) {
      s.beginIdx[valid] = s.beginIdx[i];
      s.endIdx[valid] = s.endIdx[i];
      ++valid;
    }
  }
  s.segmentCount = valid;
}

// ─── Public entry point
// ───────────────────────────────────────────────────────

TtsResult split_sentences(SplitSentence & out,
    char * text,
    uint16_t text_len,
    uint16_t min_len,
    uint16_t max_len)
{
  split_struct_reset(out);

  if (!text || min_len == 0 || max_len == 0 || min_len > max_len || max_len > TTS_MAX_TEXT_SIZE ||
      text_len > TTS_MAX_TEXT_SIZE)
    return TtsResult::Fail;

  normalizeSpecials(text, text_len);

  uint16_t normLen = 0;
  compactPunct(text, out.buffer, text_len, &normLen);

  BoundaryState bs{ out.buffer, out.beginIdx, out.endIdx };
  scanBoundaries(bs, normLen, min_len, max_len);
  out.segmentCount = bs.seg;

  trimEdges(out);
  pruneEmpty(out);

  return TtsResult::Success;
}

}  // namespace tts
