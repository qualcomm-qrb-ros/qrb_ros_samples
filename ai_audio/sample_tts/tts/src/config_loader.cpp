// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear

#include "config_loader.hpp"

#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <string>

namespace audio
{

static std::string trim(const std::string & s)
{
  const char * ws = " \t\r\n";
  size_t b = s.find_first_not_of(ws);
  if (b == std::string::npos)
    return {};
  size_t e = s.find_last_not_of(ws);
  return s.substr(b, e - b + 1);
}

static bool pathExists(const std::string & path)
{
  FILE * f = fopen(path.c_str(), "r");
  if (!f)
    return false;
  fclose(f);
  return true;
}

static std::string executableDir()
{
  char buf[4096] = {};
  ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
  if (n <= 0)
    return {};
  std::string p(buf, static_cast<size_t>(n));
  size_t pos = p.rfind('/');
  return (pos != std::string::npos) ? p.substr(0, pos + 1) : std::string();
}

std::string findConfigPath()
{
  static const char * kName = "tts_paths.cfg";
  const std::string dir = executableDir();

  const std::string candidates[] = {
    std::string("./") + kName,
    std::string("./config/") + kName,
    dir + kName,
    dir + "config/" + kName,
    TTS_DEFAULT_CONFIG_PATH,
  };

  for (const auto & p : candidates) {
    if (!p.empty() && pathExists(p))
      return p;
  }
  return TTS_DEFAULT_CONFIG_PATH;
}

Config loadDefaultConfig()
{
  return loadConfig(findConfigPath());
}

Config loadConfig(const std::string & configPath)
{
  Config cfg;
  FILE * f = fopen(configPath.c_str(), "r");
  if (!f)
    return cfg;

  char line[1024];
  while (fgets(line, sizeof(line), f)) {
    std::string s = trim(std::string(line));
    if (s.empty() || s[0] == '#')
      continue;

    size_t eq = s.find('=');
    if (eq == std::string::npos)
      continue;

    std::string key = trim(s.substr(0, eq));
    std::string val = trim(s.substr(eq + 1));

    if (key == "tokenizer_path")
      cfg.tokenizerPath = val;
    else if (key == "normalizer_path")
      cfg.normalizerPath = val;
    else if (key == "bert_model_path")
      cfg.bertModelPath = val;
    else if (key == "encoder_model_path")
      cfg.encoderModelPath = val;
    else if (key == "flow_model_path")
      cfg.flowModelPath = val;
    else if (key == "decoder_model_path")
      cfg.decoderModelPath = val;
    else if (key == "backend_lib_path")
      cfg.backendLibPath = val;
    else if (key == "system_lib_path")
      cfg.systemLibPath = val;
    else if (key == "speech_rate")
      cfg.speechRate = std::stof(val);
    else if (key == "sample_rate")
      cfg.sampleRate = static_cast<uint32_t>(std::stoul(val));
  }

  fclose(f);
  return cfg;
}

}  // namespace audio
