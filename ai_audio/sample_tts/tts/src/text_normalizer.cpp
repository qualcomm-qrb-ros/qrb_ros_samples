// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear

#include "text_normalizer.hpp"

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace tts
{

// ─── Buffer helpers
// ───────────────────────────────────────────────────────────

static void scopy(char * dst, const char * src, size_t cap)
{
  if (!dst || !src || cap == 0)
    return;
  size_t n = strlen(src);
  if (n >= cap)
    n = cap - 1;
  memcpy(dst, src, n);
  dst[n] = '\0';
}

static void scat(char * dst, const char * src, size_t cap)
{
  if (!dst || !src || cap == 0)
    return;
  size_t d = strlen(dst);
  if (d >= cap - 1)
    return;
  scopy(dst + d, src, cap - d);
}

// ─── Case conversion
// ──────────────────────────────────────────────────────────

char * text_lower(char * text)
{
  for (char * p = text; *p; ++p)
    *p = static_cast<char>(tolower(static_cast<unsigned char>(*p)));
  return text;
}

char * text_upper(char * text)
{
  for (char * p = text; *p; ++p)
    *p = static_cast<char>(toupper(static_cast<unsigned char>(*p)));
  return text;
}

// ─── Symbol stripping
// ─────────────────────────────────────────────────────────

static void stripSymbols(char * text)
{
  struct SymSeq
  {
    const char * bytes;
    int len;
  };
  static const SymSeq kDrop[] = {
    { "\xE2\x84\xA2", 3 },  // ™
    { "\xC2\xAE", 2 },      // ®
    { "\xC2\xA9", 2 },      // ©
    { "\xE2\x84\xA0", 3 },  // ℠
    { "\xE2\x84\x97", 3 },  // ℗
    { "\xE2\x80\xA0", 3 },  // †
    { "\xE2\x80\xA1", 3 },  // ‡
    { "\xC2\xAA", 2 },      // ª
    { "\xC2\xBA", 2 },      // º
    { "\xC2\xA7", 2 },      // §
    { "\xC2\xB6", 2 },      // ¶
  };
  static constexpr int kNDrop = static_cast<int>(sizeof(kDrop) / sizeof(kDrop[0]));

  char buf[TTS_MAX_TEXT_SIZE] = {};
  const char * r = text;
  char * w = buf;
  size_t rem = TTS_MAX_TEXT_SIZE - 1;

  while (*r && rem > 0) {
    bool dropped = false;
    for (int i = 0; i < kNDrop && !dropped; ++i) {
      if (strncmp(r, kDrop[i].bytes, kDrop[i].len) == 0) {
        r += kDrop[i].len;
        dropped = true;
      }
    }
    if (dropped)
      continue;

    unsigned char b0 = static_cast<unsigned char>(*r);
    unsigned char b1 = (r[1] ? static_cast<unsigned char>(r[1]) : 0);
    unsigned char b2 = (r[2] ? static_cast<unsigned char>(r[2]) : 0);

    // superscript digits E2 81 B0..BF
    if (b0 == 0xE2 && b1 == 0x81 && b2 >= 0xB0 && b2 <= 0xBF) {
      r += 3;
      continue;
    }
    // subscript digits E2 82 80..9F
    if (b0 == 0xE2 && b1 == 0x82 && b2 >= 0x80 && b2 <= 0x9F) {
      r += 3;
      continue;
    }

    *w++ = *r++;
    --rem;
  }
  *w = '\0';
  scopy(text, buf, TTS_MAX_TEXT_SIZE);
}

// ─── Hyphen expansion
// ─────────────────────────────────────────────────────────

static void expandHyphens(char * text)
{
  for (char * p = text; *p; ++p)
    if (*p == '-')
      *p = ' ';
}

// ─── Number to words
// ──────────────────────────────────────────────────────────

static const char * kOnes[] = { "zero", "one", "two", "three", "four", "five", "six", "seven",
  "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
  "seventeen", "eighteen", "nineteen" };
static const char * kTens[] = { "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
  "eighty", "ninety" };
static const char * kScale[] = { "", "thousand", "million", "billion", "trillion" };

static void numWords(int64_t n, char * out, size_t sz)
{
  if (n == 0) {
    scopy(out, "zero", sz);
    return;
  }
  if (n < 0) {
    scopy(out, "minus ", sz);
    numWords(-n, out + strlen(out), sz - strlen(out));
    return;
  }
  if (n < 20) {
    scopy(out, kOnes[n], sz);
    return;
  }
  if (n < 100) {
    scopy(out, kTens[n / 10], sz);
    if (n % 10) {
      scat(out, " ", sz);
      scat(out, kOnes[n % 10], sz);
    }
    return;
  }
  if (n < 1000) {
    char rest[256] = {};
    if (n % 100 == 0)
      snprintf(out, sz, "%s hundred", kOnes[n / 100]);
    else {
      numWords(n % 100, rest, sizeof(rest));
      snprintf(out, sz, "%s hundred %s", kOnes[n / 100], rest);
    }
    return;
  }
  // >= 1000: collect groups bottom-up, assemble top-down
  char groups[5][256] = {};
  int ng = 0;
  int64_t rem = n;
  while (rem > 0 && ng < 5) {
    int g = static_cast<int>(rem % 1000);
    if (g != 0) {
      numWords(static_cast<int64_t>(g), groups[ng], sizeof(groups[ng]));
    }
    rem /= 1000;
    ++ng;
  }
  out[0] = '\0';
  for (int k = ng - 1; k >= 0; --k) {
    if (groups[k][0]) {
      if (out[0])
        scat(out, " ", sz);
      scat(out, groups[k], sz);
      if (k > 0) {
        scat(out, " ", sz);
        scat(out, kScale[k], sz);
      }
    }
  }
}

// ─── Ordinal number to words
// ──────────────────────────────────────────────────

static void numWordsOrd(int64_t n, char * out, size_t sz, bool isFinal)
{
  static const char * kOrdOnes[] = { "", "first", "second", "third", "fourth", "fifth", "sixth",
    "seventh", "eighth", "ninth" };
  static const char * kOrdTeens[] = { "tenth", "eleventh", "twelfth", "thirteenth", "fourteenth",
    "fifteenth", "sixteenth", "seventeenth", "eighteenth", "nineteenth" };
  static const char * kOrdTens[] = { "", "tenth", "twentieth", "thirtieth", "fortieth", "fiftieth",
    "sixtieth", "seventieth", "eightieth", "ninetieth" };

  if (n < 10) {
    snprintf(out, sz, "%s", isFinal ? kOrdOnes[n] : kOnes[n]);
  } else if (n < 20) {
    snprintf(out, sz, "%s", isFinal ? kOrdTeens[n - 10] : kOnes[n]);
  } else if (n < 100) {
    if (n % 10 == 0)
      snprintf(out, sz, "%s", isFinal ? kOrdTens[n / 10] : kTens[n / 10]);
    else
      snprintf(out, sz, "%s %s", kTens[n / 10], isFinal ? kOrdOnes[n % 10] : kOnes[n % 10]);
  } else if (n < 1000) {
    char tmp[256] = {};
    if (n % 100 == 0)
      snprintf(out, sz, "%s hundredth", kOnes[n / 100]);
    else {
      numWordsOrd(n % 100, tmp, sizeof(tmp), true);
      snprintf(out, sz, "%s hundred %s", kOnes[n / 100], tmp);
    }
  } else if (n < 1000000) {
    char a[256] = {}, b[256] = {};
    numWordsOrd(n / 1000, a, sizeof(a), false);
    if (n % 1000 == 0)
      snprintf(out, sz, "%s thousandth", a);
    else {
      numWordsOrd(n % 1000, b, sizeof(b), true);
      snprintf(out, sz, "%s thousand %s", a, b);
    }
  } else if (n < 1000000000) {
    char a[256] = {}, b[256] = {};
    numWordsOrd(n / 1000000, a, sizeof(a), false);
    if (n % 1000000 == 0)
      snprintf(out, sz, "%s millionth", a);
    else {
      numWordsOrd(n % 1000000, b, sizeof(b), true);
      snprintf(out, sz, "%s million %s", a, b);
    }
  } else {
    char a[256] = {}, b[256] = {};
    numWordsOrd(n / 1000000000, a, sizeof(a), false);
    if (n % 1000000000 == 0)
      snprintf(out, sz, "%s billionth", a);
    else {
      numWordsOrd(n % 1000000000, b, sizeof(b), true);
      snprintf(out, sz, "%s billion %s", a, b);
    }
  }
}

// ─── Time parsing
// ─────────────────────────────────────────────────────────────

enum class AmPmKind
{
  AM,
  PM,
  None
};

struct ParsedTime
{
  int hour = 0, minute = 0, second = 0;
  AmPmKind ampm = AmPmKind::None;
};

static AmPmKind parseAmPm(const char * s)
{
  if (strcmp(s, "am") == 0 || strcmp(s, "a.m.") == 0)
    return AmPmKind::AM;
  if (strcmp(s, "pm") == 0 || strcmp(s, "p.m.") == 0)
    return AmPmKind::PM;
  return AmPmKind::None;
}

static bool scanTime(const char * s, ParsedTime & t)
{
  unsigned h = 0, m = 0, sec = 0;
  const char * p = s;
  while (*p && isdigit((unsigned char)*p))
    h = h * 10 + (*p++) - '0';
  if (*p != ':')
    return false;
  ++p;
  while (*p && isdigit((unsigned char)*p))
    m = m * 10 + (*p++) - '0';
  if (*p == ':') {
    ++p;
    while (*p && isdigit((unsigned char)*p))
      sec = sec * 10 + (*p++) - '0';
    if (*p == '.') {
      ++p;
      while (*p && isdigit((unsigned char)*p))
        ++p;
    }
  }
  while (*p && isspace((unsigned char)*p))
    ++p;
  char ampmBuf[8] = {};
  if (*p)
    scopy(ampmBuf, p, sizeof(ampmBuf));
  if (h > 23 || m > 59 || sec > 59)
    return false;
  t = { (int)h, (int)m, (int)sec, parseAmPm(ampmBuf) };
  return true;
}

// ─── Time expansion
// ───────────────────────────────────────────────────────────

static void expandTime(char * text, char * output, size_t outSz)
{
  scopy(output, text, outSz);
  char * pos = output;

  while ((pos = strstr(pos, ":")) != nullptr) {
    if (pos == output || !isdigit((unsigned char)*(pos - 1)) ||
        !isdigit((unsigned char)*(pos + 1))) {
      ++pos;
      continue;
    }

    char * segStart = pos;
    while (segStart > output && isdigit((unsigned char)*(segStart - 1)))
      --segStart;

    char * segEnd = pos;
    while (isdigit((unsigned char)*(segEnd + 1)) || *(segEnd + 1) == ':' || *(segEnd + 1) == '.' ||
           *(segEnd + 1) == ' ')
      ++segEnd;
    if (strncmp(segEnd + 1, "am", 2) == 0 || strncmp(segEnd + 1, "pm", 2) == 0)
      segEnd += 2;
    else if (strncmp(segEnd + 1, "a.m.", 4) == 0 || strncmp(segEnd + 1, "p.m.", 4) == 0)
      segEnd += 4;

    char ts[50] = {};
    size_t tl = static_cast<size_t>(segEnd - segStart + 1);
    if (tl >= sizeof(ts))
      tl = sizeof(ts) - 1;
    scopy(ts, segStart, tl + 1);

    ParsedTime pt;
    if (!scanTime(ts, pt)) {
      pos = segEnd + 1;
      continue;
    }

    char hw[32] = {}, mw[32] = {}, sw[32] = {};
    numWords(pt.hour, hw, sizeof(hw));
    if (pt.minute > 0) {
      if (pt.minute < 10) {
        scopy(mw, "oh ", sizeof(mw));
        numWords(pt.minute, mw + 3, sizeof(mw) - 3);
      } else {
        numWords(pt.minute, mw, sizeof(mw));
      }
    }
    if (pt.second > 0)
      numWords(pt.second, sw, sizeof(sw));

    const char * amsfx = (pt.ampm == AmPmKind::AM) ? " a m" :
                         (pt.ampm == AmPmKind::PM) ? " p m" :
                                                     "";
    const char * trail = (pt.ampm == AmPmKind::None) ? " " : "";

    char expanded[128] = {};
    if (mw[0] && sw[0])
      snprintf(expanded, sizeof(expanded), "%s %s %s%s%s", hw, mw, sw, amsfx, trail);
    else if (mw[0])
      snprintf(expanded, sizeof(expanded), "%s %s%s%s", hw, mw, amsfx, trail);
    else
      snprintf(expanded, sizeof(expanded), "%s%s%s", hw, amsfx, trail);

    char tmp[TTS_MAX_TEXT_SIZE] = {};
    scopy(tmp, output, sizeof(tmp));
    size_t si = static_cast<size_t>(segStart - output);
    size_t ei = static_cast<size_t>(segEnd - output + 1);
    size_t elen = strlen(expanded);
    if (si + elen < outSz) {
      scopy(output + si, expanded, outSz - si);
      if (si + elen < outSz)
        scopy(output + si + elen, tmp + ei, outSz - si - elen);
    }
    pos = output + si + elen;
  }
}

// ─── Unit expansion
// ───────────────────────────────────────────────────────────

static void expandUnits(char * text)
{
  char buf[TTS_MAX_TEXT_SIZE] = {};
  const char * r = text;
  char * w = buf;
  size_t rem = TTS_MAX_TEXT_SIZE - 1;

  while (*r && rem > 0) {
    const char * sub = nullptr;
    size_t skip = 0;
    char prev = (r > text) ? *(r - 1) : '\0';
    char prev2 = (r > text + 1) ? *(r - 2) : '\0';
    bool sing = (prev == '1' && !isdigit((unsigned char)prev2));

    if (strncmp(r, "lbs", 3) == 0 && !isalnum((unsigned char)*(r + 3)) &&
        isdigit((unsigned char)prev)) {
      sub = sing ? " pound" : " pounds";
      skip = 3;
    } else if (strncmp(r, "kg", 2) == 0 && !isalnum((unsigned char)*(r + 2)) &&
               isdigit((unsigned char)prev)) {
      sub = sing ? " kilogram" : " kilograms";
      skip = 2;
    } else if (strncmp(r, "km", 2) == 0 && !isalnum((unsigned char)*(r + 2)) &&
               isdigit((unsigned char)prev)) {
      sub = sing ? " kilometer" : " kilometers";
      skip = 2;
    } else if (strncmp(r, "cm", 2) == 0 && !isalnum((unsigned char)*(r + 2)) &&
               isdigit((unsigned char)prev)) {
      sub = sing ? " centimeter" : " centimeters";
      skip = 2;
    } else if (strncmp(r, "mm", 2) == 0 && !isalnum((unsigned char)*(r + 2)) &&
               isdigit((unsigned char)prev)) {
      sub = sing ? " millimeter" : " millimeters";
      skip = 2;
    } else if (strncmp(r, "in", 2) == 0 && !isalnum((unsigned char)*(r + 2)) &&
               isdigit((unsigned char)prev)) {
      sub = sing ? " inch" : " inches";
      skip = 2;
    } else if (strncmp(r, "ft", 2) == 0 && !isalnum((unsigned char)*(r + 2)) &&
               isdigit((unsigned char)prev)) {
      sub = " feet";
      skip = 2;
    } else if (strncmp(r, "mi", 2) == 0 && !isalnum((unsigned char)*(r + 2)) &&
               isdigit((unsigned char)prev)) {
      sub = sing ? " mile" : " miles";
      skip = 2;
    } else if (*r == 'g' && !isalnum((unsigned char)*(r + 1)) && isdigit((unsigned char)prev)) {
      sub = sing ? " gram" : " grams";
      skip = 1;
    } else if (*r == 'm' && !isalnum((unsigned char)*(r + 1)) && isdigit((unsigned char)prev)) {
      sub = sing ? " meter" : " meters";
      skip = 1;
    } else if (*r == '%' && !isalnum((unsigned char)*(r + 1)) && isdigit((unsigned char)prev)) {
      sub = " percent";
      skip = 1;
    } else if (*r == '&' && *(r + 1) == '\0') {
      sub = " and";
      skip = 1;
    } else if (*r == '@') {
      sub = " at ";
      skip = 1;
    } else if (*r == '+') {
      sub = " plus ";
      skip = 1;
    } else if (strncmp(r, "a.m.", 4) == 0 && !isalnum((unsigned char)*(r + 4)) &&
               isdigit((unsigned char)prev)) {
      sub = "a.m.";
      skip = 4;
    } else if (strncmp(r, "p.m.", 4) == 0 && !isalnum((unsigned char)*(r + 4)) &&
               isdigit((unsigned char)prev)) {
      sub = "p.m.";
      skip = 4;
    } else if (strncmp(r, "am", 2) == 0 && !isalnum((unsigned char)*(r + 2)) &&
               isdigit((unsigned char)prev)) {
      sub = "am";
      skip = 2;
    } else if (strncmp(r, "pm", 2) == 0 && !isalnum((unsigned char)*(r + 2)) &&
               isdigit((unsigned char)prev)) {
      sub = "pm";
      skip = 2;
    }

    if (sub) {
      size_t sl = strlen(sub);
      if (rem < sl)
        break;
      scopy(w, sub, rem + 1);
      w += sl;
      rem -= sl;
      r += skip;
    } else {
      *w++ = *r++;
      --rem;
    }
  }
  *w = '\0';
  scopy(text, buf, TTS_MAX_TEXT_SIZE);
}

// ─── Number normalisation pipeline ───────────────────────────────────────────

static void stripGroupingSep(char * text)
{
  char *r = text, *w = text;
  bool inNum = false;
  while (*r) {
    if (isdigit((unsigned char)*r)) {
      inNum = true;
      *w++ = *r;
    } else if (*r == ',' && inNum && isdigit((unsigned char)*(r + 1))) { /* skip */
    } else {
      inNum = false;
      *w++ = *r;
    }
    ++r;
  }
  *w = '\0';
}

static void expandDecimal(char * text)
{
  char buf[TTS_MAX_TEXT_SIZE] = {};
  const char * r = text;
  char * w = buf;
  size_t rem = TTS_MAX_TEXT_SIZE - 1;

  while (*r && rem > 0) {
    if (*r == '.' && r > text && isdigit((unsigned char)*(r - 1)) &&
        isdigit((unsigned char)*(r + 1))) {
      static const char kPt[] = " point ";
      size_t pl = sizeof(kPt) - 1;
      if (rem < pl)
        break;
      scopy(w, kPt, rem + 1);
      w += pl;
      rem -= pl;
      ++r;
      while (isdigit((unsigned char)*r) && rem > 1) {
        *w++ = *r++;
        *w++ = ' ';
        rem -= 2;
      }
      if (w > buf && *(w - 1) == ' ') {
        --w;
        ++rem;
      }
    } else {
      *w++ = *r++;
      --rem;
    }
  }
  *w = '\0';
  scopy(text, buf, TTS_MAX_TEXT_SIZE);
}

static void expandCurrency(char * text)
{
  char buf[TTS_MAX_TEXT_SIZE] = {};
  const char * r = text;
  char * w = buf;
  size_t rem = TTS_MAX_TEXT_SIZE - 1;

  while (*r && rem > 0) {
    const char *unit = nullptr, *sub = nullptr;
    size_t symLen = 0;

    if (*r == '$') {
      unit = "dollar";
      sub = "cent";
      symLen = 1;
    } else if (strncmp(r, "\xC2\xA3", 2) == 0) {
      unit = "pound sterling";
      sub = "penny";
      symLen = 2;
    } else if (strncmp(r, "\xE2\x82\xAC", 3) == 0) {
      unit = "euro";
      sub = "cent";
      symLen = 3;
    } else if (strncmp(r, "\xC2\xA5", 2) == 0) {
      unit = "yen";
      sub = "sen";
      symLen = 2;
    }

    if (symLen > 0) {
      r += symLen;
      char nbuf[50] = {};
      char * np = nbuf;
      while ((isdigit((unsigned char)*r) || *r == ',' || *r == '.') &&
             np - nbuf < (int)sizeof(nbuf) - 1)
        *np++ = *r++;
      *np = '\0';
      double dv = atof(nbuf);
      uint64_t maj = static_cast<uint64_t>(dv);
      uint64_t min = static_cast<uint64_t>((dv - maj) * 100 + 0.5);
      char tmp[64] = {};
      if (maj > 0) {
        char wrd[128] = {};
        if (maj == 1)
          snprintf(tmp, sizeof(tmp), "one %s", unit);
        else {
          numWords(static_cast<int64_t>(maj), wrd, sizeof(wrd));
          snprintf(tmp, sizeof(tmp), "%s %ss", wrd, unit);
        }
        size_t tl = strlen(tmp);
        if (rem < tl)
          break;
        scopy(w, tmp, rem + 1);
        w += tl;
        rem -= tl;
      }
      if (min > 0) {
        char wrd[128] = {};
        if (maj > 0 && rem >= 5) {
          scopy(w, " and ", rem + 1);
          w += 5;
          rem -= 5;
        }
        if (min == 1)
          snprintf(tmp, sizeof(tmp), "one %s", sub);
        else {
          numWords(static_cast<int64_t>(min), wrd, sizeof(wrd));
          snprintf(tmp, sizeof(tmp), "%s %ss", wrd, sub);
        }
        size_t tl = strlen(tmp);
        if (rem < tl)
          break;
        scopy(w, tmp, rem + 1);
        w += tl;
        rem -= tl;
      }
    } else {
      *w++ = *r++;
      --rem;
    }
  }
  *w = '\0';
  scopy(text, buf, TTS_MAX_TEXT_SIZE);
}

static void expandOrdinals(char * text)
{
  char buf[TTS_MAX_TEXT_SIZE] = {};
  const char * r = text;
  char * w = buf;
  size_t rem = TTS_MAX_TEXT_SIZE - 1;

  while (*r && rem > 0) {
    if (isdigit((unsigned char)*r) || (*r == '-' && isdigit((unsigned char)*(r + 1)))) {
      int64_t n = strtoll(r, const_cast<char **>(&r), 10);
      if (strncmp(r, "st", 2) == 0 || strncmp(r, "nd", 2) == 0 || strncmp(r, "rd", 2) == 0 ||
          strncmp(r, "th", 2) == 0) {
        char wrd[128] = {};
        numWordsOrd(n, wrd, sizeof(wrd), true);
        size_t wl = strlen(wrd);
        if (rem < wl)
          break;
        scopy(w, wrd, rem + 1);
        w += wl;
        rem -= wl;
        r += 2;
      } else {
        size_t nl = snprintf(w, rem + 1, "%lld", (long long)n);
        w += nl;
        rem -= nl;
      }
    } else {
      *w++ = *r++;
      --rem;
    }
  }
  *w = '\0';
  scopy(text, buf, TTS_MAX_TEXT_SIZE);
}

static void expandNumbers(char * text)
{
  char buf[TTS_MAX_TEXT_SIZE] = {};
  const char * r = text;
  char * w = buf;
  size_t rem = TTS_MAX_TEXT_SIZE - 1;

  while (*r && rem > 0) {
    if (isdigit((unsigned char)*r) || (*r == '-' && isdigit((unsigned char)*(r + 1)))) {
      if (w > buf && *(w - 1) != ' ' && rem > 0) {
        *w++ = ' ';
        --rem;
      }
      int64_t n = strtoll(r, const_cast<char **>(&r), 10);
      int64_t absn = (n < 0) ? -n : n;
      char wrd[256] = {};
      numWords(absn, wrd, sizeof(wrd));
      size_t pl = (n < 0) ? 6 : 0;
      size_t wl = strlen(wrd);
      if (rem < pl + wl)
        break;
      if (n < 0) {
        scopy(w, "minus ", rem + 1);
        w += pl;
        rem -= pl;
      }
      scopy(w, wrd, rem + 1);
      w += wl;
      rem -= wl;
      if (w > buf && *(w - 1) == ' ') {
        --w;
        ++rem;
      }
      if (*r && *r != ' ' && rem > 0) {
        *w++ = ' ';
        --rem;
      }
    } else {
      *w++ = *r++;
      --rem;
    }
  }
  *w = '\0';
  scopy(text, buf, TTS_MAX_TEXT_SIZE);
}

// ─── Abbreviation expansion
// ───────────────────────────────────────────────────

struct AbbrevEntry
{
  const char * abbr;
  const char * full;
};
static const AbbrevEntry kAbbreviations[] = {
  { "wi-fi", "wifi" },
  { "mrs.", "misess" },
  { "ms.", "miss" },
  { "mr.", "mister" },
  { "dr.", "doctor" },
  { "st.", "saint" },
  { "co.", "company" },
  { "jr.", "junior" },
  { "maj.", "major" },
  { "gen.", "general" },
  { "drs.", "doctors" },
  { "rev.", "reverend" },
  { "lt.", "lieutenant" },
  { "hon.", "honorable" },
  { "sgt.", "sergeant" },
  { "capt.", "captain" },
  { "esq.", "esquire" },
  { "ltd.", "limited" },
  { "col.", "colonel" },
  { "ft.", "fort" },
};
static constexpr int kNAbbrev =
    static_cast<int>(sizeof(kAbbreviations) / sizeof(kAbbreviations[0]));

static void expandAbbreviations(char * text)
{
  char buf[TTS_MAX_TEXT_SIZE] = {};
  for (int ai = 0; ai < kNAbbrev; ++ai) {
    const AbbrevEntry & ae = kAbbreviations[ai];
    char *from = text, *loc;
    while ((loc = strstr(from, ae.abbr)) != nullptr) {
      if (loc > text && !isspace((unsigned char)*(loc - 1))) {
        from = loc + 1;
        continue;
      }
      size_t pfx = static_cast<size_t>(loc - text);
      size_t alen = strlen(ae.abbr);
      size_t flen = strlen(ae.full);
      size_t tail = strlen(loc + alen);
      size_t pad = isalpha((unsigned char)loc[alen]) ? 1 : 0;
      if (pfx + flen + pad + tail >= TTS_MAX_TEXT_SIZE)
        break;
      scopy(buf, text, pfx + 1);
      scat(buf, ae.full, sizeof(buf));
      if (pad)
        scat(buf, " ", sizeof(buf));
      scat(buf, loc + alen, sizeof(buf));
      scopy(text, buf, TTS_MAX_TEXT_SIZE);
      from = text + pfx + flen + pad;
    }
  }
}

// ─── Strip incomplete multi-byte chars at end
// ─────────────────────────────────

static void stripIncompleteUtf8(char * text)
{
  int len = static_cast<int>(strlen(text));
  int wi = 0, ri = 0;
  while (ri < len) {
    unsigned char c = static_cast<unsigned char>(text[ri]);
    if ((c & 0x80) == 0) {
      text[wi++] = text[ri++];
      continue;
    }
    int extra = 0;
    if ((c & 0xE0) == 0xC0)
      extra = 1;
    else if ((c & 0xF0) == 0xE0)
      extra = 2;
    else if ((c & 0xF8) == 0xF0)
      extra = 3;
    else {
      ++ri;
      continue;
    }
    if (ri + extra >= len)
      break;
    bool ok = true;
    for (int k = 1; k <= extra; ++k)
      if ((static_cast<unsigned char>(text[ri + k]) & 0xC0) != 0x80) {
        ok = false;
        break;
      }
    if (ok)
      for (int k = 0; k <= extra; ++k)
        text[wi++] = text[ri++];
    else
      ri += extra + 1;
  }
  text[wi] = '\0';
}

// ─── Main pipeline
// ────────────────────────────────────────────────────────────

int text_normalize_en(const char * text_in, char * output, size_t out_size)
{
  if (!text_in || !output || out_size == 0)
    return -1;

  char work[TTS_MAX_TEXT_SIZE] = {};
  scopy(work, text_in, sizeof(work));

  size_t wl = strlen(work);
  if (wl > 0 && work[wl - 1] == '\n')
    work[--wl] = '\0';
  for (char * p = work; *p; ++p)
    if (*p == '\n')
      *p = ' ';

  stripSymbols(work);
  expandHyphens(work);
  text_lower(work);

  expandTime(work, output, out_size);

  size_t ol = strlen(output);
  char termPunct = '\0';
  if (ol > 0 && (output[ol - 1] == '.' || output[ol - 1] == '!' || output[ol - 1] == '?')) {
    termPunct = output[--ol];
    output[ol] = '\0';
  }

  expandUnits(output);
  stripGroupingSep(output);
  expandCurrency(output);
  expandDecimal(output);
  expandOrdinals(output);
  expandNumbers(output);
  expandAbbreviations(output);
  stripIncompleteUtf8(output);

  if (termPunct != '\0' && strlen(output) < out_size - 1) {
    size_t nl = strlen(output);
    output[nl] = termPunct;
    output[nl + 1] = '\0';
  }

  return 0;
}

}  // namespace tts
