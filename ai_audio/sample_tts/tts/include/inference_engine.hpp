// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear

#pragma once
#include <QnnInterface.h>
#include <dlfcn.h>

#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "HTP/QnnHtpDevice.h"
#include "HTP/QnnHtpPerfInfrastructure.h"
#include "IOTensor.hpp"
#include "System/QnnSystemInterface.h"
#include "audio_types.hpp"

namespace audio
{

template <typename Fn>
static inline Fn loadQnnSym(void * handle, const char * sym)
{
  return reinterpret_cast<Fn>(dlsym(handle, sym));
}

using QnnIfaceGetProvidersFn = decltype(QnnInterface_getProviders);
using QnnSysIfaceGetProvidersFn = decltype(QnnSystemInterface_getProviders);

class InferenceRunner
{
public:
  struct ModelPaths
  {
    std::string bertPath;
    std::string encoderPath;
    std::string flowPath;
    std::string decoderPath;
    std::string backendLib = "/usr/lib/libQnnHtp.so";
    std::string systemLib = "/usr/lib/libQnnSystem.so";
  };

  InferenceRunner();
  ~InferenceRunner();

  bool setup(const ModelPaths & paths);

  bool executeEmbedder(const int32_t * tokenIds,
      const int32_t * attentionMask,
      const int32_t * tokenTypes,
      float * hiddenStatesOut);

  bool executeEncoder(const int32_t * phonemes,
      int32_t phoneCount,
      const int32_t * tones,
      int32_t langId,
      int32_t speakerId,
      const float * bertFeats,
      float sdpRatio,
      float lengthScale,
      float noiseScaleW,
      float * gOut,
      float * xMaskOut,
      float * mPOut,
      float * logsPOut,
      float * wCeilOut,
      float * yLenOut);

  bool executeFlow(const float * mP,
      const float * logsP,
      const float * yMask,
      const float * attnPath,
      const float * gVec,
      float noiseScale,
      float * zOut);

  bool executeDecoder(const float * zWindow, const float * gVec, float * audioOut);

private:
  struct GraphCtx
  {
    Qnn_ContextHandle_t ctx = nullptr;
    GraphInfo_t ** graphs = nullptr;
    uint32_t graphCount = 0;
    Qnn_Tensor_t * inputs = nullptr;
    Qnn_Tensor_t * outputs = nullptr;
  };

  QNN_INTERFACE_VER_TYPE m_iface = {};
  QNN_SYSTEM_INTERFACE_VER_TYPE m_sysIface = {};
  void * m_lib = nullptr;
  void * m_sysLib = nullptr;
  Qnn_BackendHandle_t m_backend = nullptr;
  Qnn_DeviceHandle_t m_device = nullptr;
  QnnHtpDevice_PerfInfrastructure_t m_perf = {};
  uint32_t m_perfId = 1;
  uint32_t m_nProviders = 0;
  bool m_hasDevice = false;

  GraphCtx m_embedder;
  GraphCtx m_encoder;
  GraphCtx m_flow;
  GraphCtx m_decoder;

  IOTensor m_io;

  bool openLibraries(const std::string & backLib, const std::string & sysLib);
  bool initBackend();
  bool initPerf();
  bool loadGraph(const std::string & path, GraphCtx & ctx);
  bool runGraph(GraphCtx & ctx);
  void releaseCtx(GraphCtx & ctx);
  void releaseDevice();
  void releaseBackend();

  bool copyGraphMetadata(const QnnSystemContext_BinaryInfo_t * info,
      GraphInfo_t **& graphs,
      uint32_t & count);
  bool copyGraphList(const QnnSystemContext_GraphInfo_t * src, uint32_t n, GraphInfo_t **& dst);
  bool copyGraphV1(const QnnSystemContext_GraphInfoV1_t * src, GraphInfo_t * dst);
  bool copyGraphV3(const QnnSystemContext_GraphInfoV3_t * src, GraphInfo_t * dst);
  bool copyTensorArray(const Qnn_Tensor_t * src, Qnn_Tensor_t *& dst, uint32_t n);

  static Qnn_Tensor_t * lookupTensor(Qnn_Tensor_t * arr, uint32_t n, const char * name);
  static Qnn_Tensor_t *
  resolveTensor(Qnn_Tensor_t * arr, uint32_t n, const char * name, uint32_t fallback);

  static uint16_t toFp16(float v);
  static float fromFp16(uint16_t h);

  static void setFloat(Qnn_Tensor_t * t, size_t i, float v);
  static void setInt32(Qnn_Tensor_t * t, size_t i, int32_t v);
  static float getFloat(const Qnn_Tensor_t * t, size_t i);
  static void fillFloat(Qnn_Tensor_t * t, const float * src, size_t n);
  static void fillInt32(Qnn_Tensor_t * t, const int32_t * src, size_t n);
  static void drainFloat(const Qnn_Tensor_t * t, float * dst, size_t n);

  void activateHighPerf();
  void deactivateHighPerf();

  static constexpr int kPerfLatencyLow = 40;
  static constexpr int kPerfLatencyHigh = 2000;
};

}  // namespace audio
