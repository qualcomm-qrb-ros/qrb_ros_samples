// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear

#include "inference_engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <set>
#include <vector>

#include "DataUtil.hpp"
#include "audio_types.hpp"

static std::set<uint32_t> s_perfIds;

static uint32_t getPerfConfigId()
{
  if (!s_perfIds.empty())
    return *s_perfIds.begin();
  return 1;
}

namespace audio
{

// ─── Constructor / Destructor ──────────────────────────────────────────────

InferenceRunner::InferenceRunner() {}

InferenceRunner::~InferenceRunner()
{
  releaseCtx(m_decoder);
  releaseCtx(m_flow);
  releaseCtx(m_encoder);
  releaseCtx(m_embedder);
  releaseDevice();
  releaseBackend();

  if (m_sysLib) {
    ::dlclose(m_sysLib);
    m_sysLib = nullptr;
  }
  if (m_lib) {
    ::dlclose(m_lib);
    m_lib = nullptr;
  }

  if (m_hasDevice && m_perf.destroyPowerConfigId) {
    m_perf.destroyPowerConfigId(m_perfId);
    s_perfIds.erase(m_perfId);
  }
}

// ─── setup ────────────────────────────────────────────────────────────────

bool InferenceRunner::setup(const ModelPaths & paths)
{
  if (!openLibraries(paths.backendLib, paths.systemLib))
    return false;
  if (!initBackend())
    return false;
  if (!initPerf())
    return false;

  if (!loadGraph(paths.bertPath, m_embedder)) {
#ifdef DEBUG_TTS
    std::cout << "[Engine] Failed to load bert\n";
#endif
    return false;
  }
  if (!loadGraph(paths.encoderPath, m_encoder)) {
#ifdef DEBUG_TTS
    std::cout << "[Engine] Failed to load encoder\n";
#endif
    return false;
  }
  if (!loadGraph(paths.flowPath, m_flow)) {
#ifdef DEBUG_TTS
    std::cout << "[Engine] Failed to load flow\n";
#endif
    return false;
  }
  if (!loadGraph(paths.decoderPath, m_decoder)) {
#ifdef DEBUG_TTS
    std::cout << "[Engine] Failed to load decoder\n";
#endif
    return false;
  }

#ifdef DEBUG_TTS
  std::cout << "[Engine] Initialised (bert/encoder/flow/decoder loaded)" << std::endl;
#endif
  return true;
}

// ─── openLibraries ────────────────────────────────────────────────────────

bool InferenceRunner::openLibraries(const std::string & backLib, const std::string & sysLib)
{
  m_lib = dlopen(backLib.c_str(), RTLD_NOW | RTLD_GLOBAL);
  if (!m_lib) {
#ifdef DEBUG_TTS
    std::cout << "[Engine] dlopen backend failed: " << dlerror() << std::endl;
#endif
    return false;
  }

  auto getProviders = loadQnnSym<QnnIfaceGetProvidersFn *>(m_lib, "QnnInterface_getProviders");
  if (!getProviders) {
#ifdef DEBUG_TTS
    std::cout << "[Engine] QnnInterface_getProviders not found\n";
#endif
    return false;
  }

  QnnInterface_t ** providers = nullptr;
  if (QNN_SUCCESS != getProviders((const QnnInterface_t ***)&providers, &m_nProviders) ||
      !providers || m_nProviders == 0) {
#ifdef DEBUG_TTS
    std::cout << "[Engine] Failed to get interface providers\n";
#endif
    return false;
  }

  bool foundIface = false;
  for (size_t i = 0; i < m_nProviders; ++i) {
    if (QNN_API_VERSION_MAJOR == providers[i]->apiVersion.coreApiVersion.major &&
        QNN_API_VERSION_MINOR <= providers[i]->apiVersion.coreApiVersion.minor) {
      m_iface = providers[i]->QNN_INTERFACE_VER_NAME;
      foundIface = true;
      break;
    }
  }
  if (!foundIface) {
#ifdef DEBUG_TTS
    std::cout << "[Engine] No compatible QNN interface version\n";
#endif
    return false;
  }

  m_sysLib = dlopen(sysLib.c_str(), RTLD_NOW | RTLD_GLOBAL);
  if (!m_sysLib) {
#ifdef DEBUG_TTS
    std::cout << "[Engine] dlopen system lib failed: " << dlerror() << std::endl;
#endif
    return false;
  }

  auto getSysProviders =
      loadQnnSym<QnnSysIfaceGetProvidersFn *>(m_sysLib, "QnnSystemInterface_getProviders");
  if (!getSysProviders) {
#ifdef DEBUG_TTS
    std::cout << "[Engine] QnnSystemInterface_getProviders not found\n";
#endif
    return false;
  }

  QnnSystemInterface_t ** sysProviders = nullptr;
  uint32_t numSys = 0;
  if (QNN_SUCCESS != getSysProviders((const QnnSystemInterface_t ***)&sysProviders, &numSys) ||
      !sysProviders || numSys == 0) {
#ifdef DEBUG_TTS
    std::cout << "[Engine] Failed to get system interface providers\n";
#endif
    return false;
  }
  m_sysIface = sysProviders[0]->QNN_SYSTEM_INTERFACE_VER_NAME;
  return true;
}

// ─── initBackend ──────────────────────────────────────────────────────────

bool InferenceRunner::initBackend()
{
  const QnnBackend_Config_t * beCfg[] = { nullptr };
  if (QNN_BACKEND_NO_ERROR != m_iface.backendCreate(nullptr, beCfg, &m_backend)) {
#ifdef DEBUG_TTS
    std::cout << "[Engine] backendCreate failed\n";
#endif
    return false;
  }

  if (m_iface.propertyHasCapability) {
    auto cap = m_iface.propertyHasCapability(QNN_PROPERTY_GROUP_DEVICE);
    if (QNN_PROPERTY_NOT_SUPPORTED == cap || QNN_PROPERTY_ERROR_UNKNOWN_KEY == cap) {
#ifdef DEBUG_TTS
      std::cout << "[Engine] Device property not supported\n";
#endif
      return false;
    }
  }
  const QnnDevice_Config_t * devCfg[] = { nullptr };
  if (m_iface.deviceCreate) {
    auto st = m_iface.deviceCreate(nullptr, devCfg, &m_device);
    if (st != QNN_SUCCESS && st != QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE) {
#ifdef DEBUG_TTS
      std::cout << "[Engine] deviceCreate failed\n";
#endif
      return false;
    }
  }
  m_hasDevice = true;
  return true;
}

// ─── initPerf ─────────────────────────────────────────────────────────────

bool InferenceRunner::initPerf()
{
  QnnDevice_Infrastructure_t devInfra = nullptr;
  if (QNN_SUCCESS != m_iface.deviceGetInfrastructure(&devInfra)) {
#ifdef DEBUG_TTS
    std::cout << "[Engine] deviceGetInfrastructure failed\n";
#endif
    return false;
  }
  auto * htpInfra = static_cast<QnnHtpDevice_Infrastructure_t *>(devInfra);
  m_perf = htpInfra->perfInfra;

  uint32_t devId = 0, coreId = 0;
  if (QNN_SUCCESS != m_perf.createPowerConfigId(devId, coreId, &m_perfId)) {
#ifdef DEBUG_TTS
    std::cout << "[Engine] createPowerConfigId failed\n";
#endif
    return false;
  }
  s_perfIds.insert(m_perfId);
  return true;
}

// ─── loadGraph ────────────────────────────────────────────────────────────

bool InferenceRunner::loadGraph(const std::string & path, GraphCtx & ctx)
{
  FILE * f = fopen(path.c_str(), "rb");
  if (!f) {
#ifdef DEBUG_TTS
    std::cout << "[Engine] Cannot open: " << path << std::endl;
#endif
    return false;
  }
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  rewind(f);
  if (sz <= 0) {
    fclose(f);
    return false;
  }
  std::vector<char> fileData(static_cast<size_t>(sz));
  if (fread(fileData.data(), 1, fileData.size(), f) != fileData.size()) {
    fclose(f);
    return false;
  }
  fclose(f);

  const void * buf = fileData.data();
  uint32_t size = static_cast<uint32_t>(fileData.size());

  QnnSystemContext_Handle_t sysCtx = nullptr;
  if (QNN_SUCCESS != m_sysIface.systemContextCreate(&sysCtx)) {
#ifdef DEBUG_TTS
    std::cout << "[Engine] systemContextCreate failed\n";
#endif
    return false;
  }

  const QnnSystemContext_BinaryInfo_t * binaryInfo = nullptr;
  Qnn_ContextBinarySize_t infoSize = 0;
  if (QNN_SUCCESS != m_sysIface.systemContextGetBinaryInfo(
                         sysCtx, const_cast<void *>(buf), size, &binaryInfo, &infoSize)) {
#ifdef DEBUG_TTS
    std::cout << "[Engine] systemContextGetBinaryInfo failed\n";
#endif
    m_sysIface.systemContextFree(sysCtx);
    return false;
  }

  if (!copyGraphMetadata(binaryInfo, ctx.graphs, ctx.graphCount)) {
    m_sysIface.systemContextFree(sysCtx);
    return false;
  }
  m_sysIface.systemContextFree(sysCtx);

  if (QNN_SUCCESS != m_iface.contextCreateFromBinary(m_backend, m_device, nullptr,
                         const_cast<void *>(buf), size, &ctx.ctx, nullptr)) {
#ifdef DEBUG_TTS
    std::cout << "[Engine] contextCreateFromBinary failed\n";
#endif
    return false;
  }

  for (uint32_t g = 0; g < ctx.graphCount; ++g) {
    if (QNN_SUCCESS !=
        m_iface.graphRetrieve(ctx.ctx, (*ctx.graphs)[g].graphName, &(*ctx.graphs)[g].graph)) {
#ifdef DEBUG_TTS
      std::cout << "[Engine] graphRetrieve failed\n";
#endif
      return false;
    }
  }

  if (!m_io.setupInputAndOutputTensors(&ctx.inputs, &ctx.outputs, *(*ctx.graphs))) {
#ifdef DEBUG_TTS
    std::cout << "[Engine] setupInputAndOutputTensors failed\n";
#endif
    return false;
  }
  return true;
}

// ─── runGraph ─────────────────────────────────────────────────────────────

bool InferenceRunner::runGraph(GraphCtx & ctx)
{
  if (ctx.graphCount == 0 || !ctx.inputs || !ctx.outputs)
    return false;
  GraphInfo_t & gi = *(*ctx.graphs);
  activateHighPerf();
  auto st = m_iface.graphExecute(
      gi.graph, ctx.inputs, gi.numInputTensors, ctx.outputs, gi.numOutputTensors, nullptr, nullptr);
  deactivateHighPerf();
  if (QNN_SUCCESS != st) {
#ifdef DEBUG_TTS
    std::cout << "[Engine] graphExecute failed (" << st << ")\n";
#endif
    return false;
  }
  return true;
}

// ─── lookupTensor / resolveTensor ─────────────────────────────────────────

Qnn_Tensor_t * InferenceRunner::lookupTensor(Qnn_Tensor_t * arr, uint32_t n, const char * name)
{
  for (uint32_t i = 0; i < n; ++i) {
    const char * tname = QNN_TENSOR_GET_NAME(&arr[i]);
    if (tname && strcmp(tname, name) == 0)
      return &arr[i];
  }
  return nullptr;
}

Qnn_Tensor_t *
InferenceRunner::resolveTensor(Qnn_Tensor_t * arr, uint32_t n, const char * name, uint32_t fallback)
{
  Qnn_Tensor_t * t = lookupTensor(arr, n, name);
  if (!t && fallback < n)
    return &arr[fallback];
  return t;
}

// ─── FP16 conversion ──────────────────────────────────────────────────────

uint16_t InferenceRunner::toFp16(float f)
{
  uint32_t x;
  memcpy(&x, &f, sizeof(x));
  uint16_t sign = static_cast<uint16_t>((x >> 16) & 0x8000u);
  int32_t exp = static_cast<int32_t>((x >> 23) & 0xFFu) - 127 + 15;
  uint32_t mantissa = (x >> 13) & 0x3FFu;
  if (exp <= 0)
    return sign;
  if (exp >= 31)
    return static_cast<uint16_t>(sign | 0x7C00u);
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | mantissa);
}

float InferenceRunner::fromFp16(uint16_t h)
{
  uint32_t sign = static_cast<uint32_t>(h >> 15) << 31;
  uint32_t exp = static_cast<uint32_t>((h >> 10) & 0x1Fu);
  uint32_t mantissa = static_cast<uint32_t>(h & 0x3FFu);
  uint32_t x;
  if (exp == 0)
    x = sign;
  else if (exp == 31)
    x = sign | 0x7F800000u | (mantissa << 13);
  else
    x = sign | ((exp + 112u) << 23) | (mantissa << 13);
  float result;
  memcpy(&result, &x, sizeof(result));
  return result;
}

// ─── Tensor element helpers ───────────────────────────────────────────────

void InferenceRunner::setFloat(Qnn_Tensor_t * t, size_t idx, float val)
{
  void * buf = QNN_TENSOR_GET_CLIENT_BUF(t).data;
  Qnn_DataType_t dt = QNN_TENSOR_GET_DATA_TYPE(t);
  switch (dt) {
    case QNN_DATATYPE_FLOAT_16:
      static_cast<uint16_t *>(buf)[idx] = toFp16(val);
      break;
    case QNN_DATATYPE_FLOAT_32:
      static_cast<float *>(buf)[idx] = val;
      break;
    case QNN_DATATYPE_INT_32:
      static_cast<int32_t *>(buf)[idx] = static_cast<int32_t>(val);
      break;
    case QNN_DATATYPE_UFIXED_POINT_8: {
      float scale = QNN_TENSOR_GET_QUANT_PARAMS(t).scaleOffsetEncoding.scale;
      int32_t off = QNN_TENSOR_GET_QUANT_PARAMS(t).scaleOffsetEncoding.offset;
      int q = static_cast<int>(std::round(val / scale)) - off;
      static_cast<uint8_t *>(buf)[idx] = static_cast<uint8_t>(std::max(0, std::min(255, q)));
      break;
    }
    case QNN_DATATYPE_UFIXED_POINT_16: {
      float scale = QNN_TENSOR_GET_QUANT_PARAMS(t).scaleOffsetEncoding.scale;
      int32_t off = QNN_TENSOR_GET_QUANT_PARAMS(t).scaleOffsetEncoding.offset;
      int q = static_cast<int>(std::round(val / scale)) - off;
      static_cast<uint16_t *>(buf)[idx] = static_cast<uint16_t>(std::max(0, std::min(65535, q)));
      break;
    }
    case QNN_DATATYPE_SFIXED_POINT_8: {
      float scale = QNN_TENSOR_GET_QUANT_PARAMS(t).scaleOffsetEncoding.scale;
      int32_t off = QNN_TENSOR_GET_QUANT_PARAMS(t).scaleOffsetEncoding.offset;
      int q = static_cast<int>(std::round(val / scale)) - off;
      static_cast<int8_t *>(buf)[idx] = static_cast<int8_t>(std::max(-128, std::min(127, q)));
      break;
    }
    case QNN_DATATYPE_SFIXED_POINT_16: {
      float scale = QNN_TENSOR_GET_QUANT_PARAMS(t).scaleOffsetEncoding.scale;
      int32_t off = QNN_TENSOR_GET_QUANT_PARAMS(t).scaleOffsetEncoding.offset;
      int q = static_cast<int>(std::round(val / scale)) - off;
      static_cast<int16_t *>(buf)[idx] = static_cast<int16_t>(std::max(-32768, std::min(32767, q)));
      break;
    }
    default:
      break;
  }
}

void InferenceRunner::setInt32(Qnn_Tensor_t * t, size_t idx, int32_t val)
{
  void * buf = QNN_TENSOR_GET_CLIENT_BUF(t).data;
  switch (QNN_TENSOR_GET_DATA_TYPE(t)) {
    case QNN_DATATYPE_INT_32:
      static_cast<int32_t *>(buf)[idx] = val;
      break;
    case QNN_DATATYPE_FLOAT_16:
      static_cast<uint16_t *>(buf)[idx] = toFp16(static_cast<float>(val));
      break;
    case QNN_DATATYPE_FLOAT_32:
      static_cast<float *>(buf)[idx] = static_cast<float>(val);
      break;
    default:
      break;
  }
}

float InferenceRunner::getFloat(const Qnn_Tensor_t * t, size_t idx)
{
  const void * buf = QNN_TENSOR_GET_CLIENT_BUF(t).data;
  Qnn_DataType_t dt = QNN_TENSOR_GET_DATA_TYPE(t);
  switch (dt) {
    case QNN_DATATYPE_FLOAT_16:
      return fromFp16(static_cast<const uint16_t *>(buf)[idx]);
    case QNN_DATATYPE_FLOAT_32:
      return static_cast<const float *>(buf)[idx];
    case QNN_DATATYPE_INT_32:
      return static_cast<float>(static_cast<const int32_t *>(buf)[idx]);
    case QNN_DATATYPE_UFIXED_POINT_8:
    case QNN_DATATYPE_UFIXED_POINT_16:
    case QNN_DATATYPE_UFIXED_POINT_32:
    case QNN_DATATYPE_SFIXED_POINT_8:
    case QNN_DATATYPE_SFIXED_POINT_16:
    case QNN_DATATYPE_SFIXED_POINT_32: {
      float scale = QNN_TENSOR_GET_QUANT_PARAMS(t).scaleOffsetEncoding.scale;
      int32_t off = QNN_TENSOR_GET_QUANT_PARAMS(t).scaleOffsetEncoding.offset;
      int64_t q;
      if (dt == QNN_DATATYPE_UFIXED_POINT_8)
        q = static_cast<int64_t>(static_cast<const uint8_t *>(buf)[idx]);
      else if (dt == QNN_DATATYPE_UFIXED_POINT_16)
        q = static_cast<int64_t>(static_cast<const uint16_t *>(buf)[idx]);
      else if (dt == QNN_DATATYPE_UFIXED_POINT_32)
        q = static_cast<int64_t>(static_cast<const uint32_t *>(buf)[idx]);
      else if (dt == QNN_DATATYPE_SFIXED_POINT_8)
        q = static_cast<int64_t>(static_cast<const int8_t *>(buf)[idx]);
      else if (dt == QNN_DATATYPE_SFIXED_POINT_16)
        q = static_cast<int64_t>(static_cast<const int16_t *>(buf)[idx]);
      else
        q = static_cast<int64_t>(static_cast<const int32_t *>(buf)[idx]);
      return static_cast<float>((q + off) * scale);
    }
    default:
      return 0.0f;
  }
}

void InferenceRunner::fillFloat(Qnn_Tensor_t * t, const float * src, size_t n)
{
  void * buf = QNN_TENSOR_GET_CLIENT_BUF(t).data;
  size_t cap = QNN_TENSOR_GET_CLIENT_BUF(t).dataSize;
  Qnn_DataType_t dt = QNN_TENSOR_GET_DATA_TYPE(t);
  if (dt == QNN_DATATYPE_FLOAT_16) {
    size_t cnt = std::min(n, cap / sizeof(uint16_t));
    for (size_t i = 0; i < cnt; ++i)
      static_cast<uint16_t *>(buf)[i] = toFp16(src[i]);
  } else if (dt == QNN_DATATYPE_FLOAT_32) {
    size_t bytes = std::min(n * sizeof(float), cap);
    memcpy(buf, src, bytes);
  } else if (dt == QNN_DATATYPE_INT_32) {
    size_t cnt = std::min(n, cap / sizeof(int32_t));
    for (size_t i = 0; i < cnt; ++i)
      static_cast<int32_t *>(buf)[i] = static_cast<int32_t>(src[i]);
  } else if (dt == QNN_DATATYPE_UFIXED_POINT_8) {
    float scale = QNN_TENSOR_GET_QUANT_PARAMS(t).scaleOffsetEncoding.scale;
    int32_t off = QNN_TENSOR_GET_QUANT_PARAMS(t).scaleOffsetEncoding.offset;
    size_t cnt = std::min(n, cap);
    for (size_t i = 0; i < cnt; ++i) {
      int q = static_cast<int>(std::round(src[i] / scale)) - off;
      static_cast<uint8_t *>(buf)[i] = static_cast<uint8_t>(std::max(0, std::min(255, q)));
    }
  } else if (dt == QNN_DATATYPE_UFIXED_POINT_16) {
    float scale = QNN_TENSOR_GET_QUANT_PARAMS(t).scaleOffsetEncoding.scale;
    int32_t off = QNN_TENSOR_GET_QUANT_PARAMS(t).scaleOffsetEncoding.offset;
    size_t cnt = std::min(n, cap / 2);
    for (size_t i = 0; i < cnt; ++i) {
      int q = static_cast<int>(std::round(src[i] / scale)) - off;
      static_cast<uint16_t *>(buf)[i] = static_cast<uint16_t>(std::max(0, std::min(65535, q)));
    }
  } else if (dt == QNN_DATATYPE_SFIXED_POINT_8) {
    float scale = QNN_TENSOR_GET_QUANT_PARAMS(t).scaleOffsetEncoding.scale;
    int32_t off = QNN_TENSOR_GET_QUANT_PARAMS(t).scaleOffsetEncoding.offset;
    size_t cnt = std::min(n, cap);
    for (size_t i = 0; i < cnt; ++i) {
      int q = static_cast<int>(std::round(src[i] / scale)) - off;
      static_cast<int8_t *>(buf)[i] = static_cast<int8_t>(std::max(-128, std::min(127, q)));
    }
  } else if (dt == QNN_DATATYPE_SFIXED_POINT_16) {
    float scale = QNN_TENSOR_GET_QUANT_PARAMS(t).scaleOffsetEncoding.scale;
    int32_t off = QNN_TENSOR_GET_QUANT_PARAMS(t).scaleOffsetEncoding.offset;
    size_t cnt = std::min(n, cap / 2);
    for (size_t i = 0; i < cnt; ++i) {
      int q = static_cast<int>(std::round(src[i] / scale)) - off;
      static_cast<int16_t *>(buf)[i] = static_cast<int16_t>(std::max(-32768, std::min(32767, q)));
    }
  }
}

void InferenceRunner::fillInt32(Qnn_Tensor_t * t, const int32_t * src, size_t n)
{
  void * buf = QNN_TENSOR_GET_CLIENT_BUF(t).data;
  size_t cap = QNN_TENSOR_GET_CLIENT_BUF(t).dataSize;
  Qnn_DataType_t dt = QNN_TENSOR_GET_DATA_TYPE(t);
  if (dt == QNN_DATATYPE_INT_32) {
    size_t bytes = std::min(n * sizeof(int32_t), cap);
    memcpy(buf, src, bytes);
  } else if (dt == QNN_DATATYPE_FLOAT_16) {
    size_t cnt = std::min(n, cap / sizeof(uint16_t));
    for (size_t i = 0; i < cnt; ++i)
      static_cast<uint16_t *>(buf)[i] = toFp16(static_cast<float>(src[i]));
  } else if (dt == QNN_DATATYPE_FLOAT_32) {
    size_t cnt = std::min(n, cap / sizeof(float));
    for (size_t i = 0; i < cnt; ++i)
      static_cast<float *>(buf)[i] = static_cast<float>(src[i]);
  }
}

void InferenceRunner::drainFloat(const Qnn_Tensor_t * t, float * dst, size_t n)
{
  const void * buf = QNN_TENSOR_GET_CLIENT_BUF(t).data;
  size_t cap = QNN_TENSOR_GET_CLIENT_BUF(t).dataSize;
  Qnn_DataType_t dt = QNN_TENSOR_GET_DATA_TYPE(t);
  if (dt == QNN_DATATYPE_FLOAT_16) {
    size_t cnt = std::min(n, cap / sizeof(uint16_t));
    for (size_t i = 0; i < cnt; ++i)
      dst[i] = fromFp16(static_cast<const uint16_t *>(buf)[i]);
  } else if (dt == QNN_DATATYPE_FLOAT_32) {
    size_t cnt = std::min(n, cap / sizeof(float));
    memcpy(dst, buf, cnt * sizeof(float));
  } else if (dt == QNN_DATATYPE_INT_32) {
    size_t cnt = std::min(n, cap / sizeof(int32_t));
    for (size_t i = 0; i < cnt; ++i)
      dst[i] = static_cast<float>(static_cast<const int32_t *>(buf)[i]);
  } else {
    float scale = QNN_TENSOR_GET_QUANT_PARAMS(t).scaleOffsetEncoding.scale;
    int32_t off = QNN_TENSOR_GET_QUANT_PARAMS(t).scaleOffsetEncoding.offset;
    switch (dt) {
      case QNN_DATATYPE_UFIXED_POINT_8: {
        size_t cnt = std::min(n, cap);
        for (size_t i = 0; i < cnt; ++i)
          dst[i] = (static_cast<int32_t>(static_cast<const uint8_t *>(buf)[i]) + off) * scale;
        break;
      }
      case QNN_DATATYPE_UFIXED_POINT_16: {
        size_t cnt = std::min(n, cap / 2);
        for (size_t i = 0; i < cnt; ++i)
          dst[i] = (static_cast<int32_t>(static_cast<const uint16_t *>(buf)[i]) + off) * scale;
        break;
      }
      case QNN_DATATYPE_UFIXED_POINT_32: {
        size_t cnt = std::min(n, cap / 4);
        for (size_t i = 0; i < cnt; ++i)
          dst[i] = static_cast<float>(
              (static_cast<int64_t>(static_cast<const uint32_t *>(buf)[i]) + off) * scale);
        break;
      }
      case QNN_DATATYPE_SFIXED_POINT_8: {
        size_t cnt = std::min(n, cap);
        for (size_t i = 0; i < cnt; ++i)
          dst[i] = (static_cast<int32_t>(static_cast<const int8_t *>(buf)[i]) + off) * scale;
        break;
      }
      case QNN_DATATYPE_SFIXED_POINT_16: {
        size_t cnt = std::min(n, cap / 2);
        for (size_t i = 0; i < cnt; ++i)
          dst[i] = (static_cast<int32_t>(static_cast<const int16_t *>(buf)[i]) + off) * scale;
        break;
      }
      case QNN_DATATYPE_SFIXED_POINT_32: {
        size_t cnt = std::min(n, cap / 4);
        for (size_t i = 0; i < cnt; ++i)
          dst[i] = static_cast<float>(
              (static_cast<int64_t>(static_cast<const int32_t *>(buf)[i]) + off) * scale);
        break;
      }
      default:
        break;
    }
  }
}

// ─── executeEmbedder ──────────────────────────────────────────────────────
// bert_wrapper.bin: [input_ids, attention_mask, token_type_ids] → hidden_states
// Input shapes: [1,200] int32;  Output: [200, 768] fp16

bool InferenceRunner::executeEmbedder(const int32_t * tokenIds,
    const int32_t * attentionMask,
    const int32_t * tokenTypes,
    float * hiddenStatesOut)
{
  if (m_embedder.graphCount == 0)
    return false;
  GraphInfo_t & gi = *(*m_embedder.graphs);

  constexpr size_t kBertLen = 200;

  Qnn_Tensor_t * t_ids = resolveTensor(m_embedder.inputs, gi.numInputTensors, "input_ids", 0);
  Qnn_Tensor_t * t_mask = resolveTensor(m_embedder.inputs, gi.numInputTensors, "attention_mask", 1);
  Qnn_Tensor_t * t_type = resolveTensor(m_embedder.inputs, gi.numInputTensors, "token_type_ids", 2);
  Qnn_Tensor_t * t_out = resolveTensor(m_embedder.outputs, gi.numOutputTensors, "hidden_states", 0);

  if (!t_ids || !t_mask || !t_type || !t_out) {
#ifdef DEBUG_TTS
    std::cout << "[Engine] BERT tensor not found\n";
#endif
    return false;
  }

  fillInt32(t_ids, tokenIds, kBertLen);
  fillInt32(t_mask, attentionMask, kBertLen);
  fillInt32(t_type, tokenTypes, kBertLen);

  if (!runGraph(m_embedder))
    return false;

  drainFloat(t_out, hiddenStatesOut, kBertLen * kBertDim);
  return true;
}

// ─── executeEncoder ───────────────────────────────────────────────────────
// encoder.bin: bert(0), ja_bert(1), language(2), length_scale(3),
//   noise_scale_w(4), sdp_ratio(5), sid(6), tone(7), x_lengths(8), x(9)
// Outputs: g(0), x_mask(1), m_p(2), logs_p(3), w_ceil(4), y_lengths(5)

bool InferenceRunner::executeEncoder(const int32_t * phonemes,
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
    float * yLenOut)
{
  if (m_encoder.graphCount == 0)
    return false;
  GraphInfo_t & gi = *(*m_encoder.graphs);

  Qnn_Tensor_t * t_bert = resolveTensor(m_encoder.inputs, gi.numInputTensors, "bert", 0);
  Qnn_Tensor_t * t_jabert = resolveTensor(m_encoder.inputs, gi.numInputTensors, "ja_bert", 1);
  Qnn_Tensor_t * t_lang = resolveTensor(m_encoder.inputs, gi.numInputTensors, "language", 2);
  Qnn_Tensor_t * t_lscale = resolveTensor(m_encoder.inputs, gi.numInputTensors, "length_scale", 3);
  Qnn_Tensor_t * t_nscalew =
      resolveTensor(m_encoder.inputs, gi.numInputTensors, "noise_scale_w", 4);
  Qnn_Tensor_t * t_sdp = resolveTensor(m_encoder.inputs, gi.numInputTensors, "sdp_ratio", 5);
  Qnn_Tensor_t * t_sid = resolveTensor(m_encoder.inputs, gi.numInputTensors, "sid", 6);
  Qnn_Tensor_t * t_tones = resolveTensor(m_encoder.inputs, gi.numInputTensors, "tone", 7);
  Qnn_Tensor_t * t_xlen = resolveTensor(m_encoder.inputs, gi.numInputTensors, "x_lengths", 8);
  Qnn_Tensor_t * t_x = resolveTensor(m_encoder.inputs, gi.numInputTensors, "x", 9);

  Qnn_Tensor_t * t_g = resolveTensor(m_encoder.outputs, gi.numOutputTensors, "g", 0);
  Qnn_Tensor_t * t_xmask = resolveTensor(m_encoder.outputs, gi.numOutputTensors, "x_mask", 1);
  Qnn_Tensor_t * t_mp = resolveTensor(m_encoder.outputs, gi.numOutputTensors, "m_p", 2);
  Qnn_Tensor_t * t_logsp = resolveTensor(m_encoder.outputs, gi.numOutputTensors, "logs_p", 3);
  Qnn_Tensor_t * t_wceil = resolveTensor(m_encoder.outputs, gi.numOutputTensors, "w_ceil", 4);
  Qnn_Tensor_t * t_ylen = resolveTensor(m_encoder.outputs, gi.numOutputTensors, "y_lengths", 5);

  if (!t_bert || !t_jabert || !t_lang || !t_lscale || !t_nscalew || !t_sdp || !t_sid || !t_tones ||
      !t_xlen || !t_x || !t_g || !t_xmask || !t_mp || !t_logsp || !t_wceil || !t_ylen) {
#ifdef DEBUG_TTS
    std::cout << "[Engine] Encoder tensor not found\n";
#endif
    return false;
  }

  int32_t language[kMaxPhoneSeq] = {};
  for (size_t i = 0; i < kMaxPhoneSeq; ++i)
    language[i] = (phonemes[i] == 0) ? 0 : langId;

  memset(QNN_TENSOR_GET_CLIENT_BUF(t_bert).data, 0, QNN_TENSOR_GET_CLIENT_BUF(t_bert).dataSize);
  fillFloat(t_jabert, bertFeats, kBertDim * kMaxPhoneSeq);
  fillInt32(t_lang, language, kMaxPhoneSeq);
  setFloat(t_lscale, 0, lengthScale);
  setFloat(t_nscalew, 0, noiseScaleW);
  setFloat(t_sdp, 0, sdpRatio);
  setInt32(t_sid, 0, speakerId);
  fillInt32(t_tones, tones, kMaxPhoneSeq);
  setInt32(t_xlen, 0, phoneCount);
  fillInt32(t_x, phonemes, kMaxPhoneSeq);

  if (!runGraph(m_encoder))
    return false;

  drainFloat(t_g, gOut, kSpeakerVecDim);
  drainFloat(t_xmask, xMaskOut, kMaxPhoneSeq);
  drainFloat(t_mp, mPOut, kMaxPhoneSeq * kLatentChannels);
  drainFloat(t_logsp, logsPOut, kMaxPhoneSeq * kLatentChannels);
  drainFloat(t_wceil, wCeilOut, kMaxPhoneSeq);
  *yLenOut = getFloat(t_ylen, 0);

#ifdef DEBUG_TTS
  std::printf("[enc_out] dtype: g=%d m_p=%d logs_p=%d w_ceil=%d\n",
      (int)QNN_TENSOR_GET_DATA_TYPE(t_g), (int)QNN_TENSOR_GET_DATA_TYPE(t_mp),
      (int)QNN_TENSOR_GET_DATA_TYPE(t_logsp), (int)QNN_TENSOR_GET_DATA_TYPE(t_wceil));
  std::printf("[enc_out] g[0..3]:     ");
  for (int i = 0; i < 4; i++)
    std::printf(" %.5f", gOut[i]);
  std::printf("\n[enc_out] m_p[0..3]:   ");
  for (int i = 0; i < 4; i++)
    std::printf(" %.5f", mPOut[i]);
  std::printf("\n[enc_out] logs_p[0..3]:");
  for (int i = 0; i < 4; i++)
    std::printf(" %.5f", logsPOut[i]);
  std::printf("\n[enc_out] w_ceil[0..3]:");
  for (int i = 0; i < 4; i++)
    std::printf(" %.5f", wCeilOut[i]);
  std::printf("\n");
#endif

  return true;
}

// ─── executeFlow ──────────────────────────────────────────────────────────
// flow.bin: [m_p, logs_p, y_mask, attn_squeezed, g, noise_scale] → z

bool InferenceRunner::executeFlow(const float * mP,
    const float * logsP,
    const float * yMask,
    const float * attnPath,
    const float * gVec,
    float noiseScale,
    float * zOut)
{
  if (m_flow.graphCount == 0)
    return false;
  GraphInfo_t & gi = *(*m_flow.graphs);

  Qnn_Tensor_t * t_mp = resolveTensor(m_flow.inputs, gi.numInputTensors, "m_p", 0);
  Qnn_Tensor_t * t_logsp = resolveTensor(m_flow.inputs, gi.numInputTensors, "logs_p", 1);
  Qnn_Tensor_t * t_ymask = resolveTensor(m_flow.inputs, gi.numInputTensors, "y_mask", 2);
  Qnn_Tensor_t * t_attn = resolveTensor(m_flow.inputs, gi.numInputTensors, "attn_squeezed", 3);
  Qnn_Tensor_t * t_g = resolveTensor(m_flow.inputs, gi.numInputTensors, "g", 4);
  Qnn_Tensor_t * t_nsc = resolveTensor(m_flow.inputs, gi.numInputTensors, "noise_scale", 5);
  Qnn_Tensor_t * t_z = resolveTensor(m_flow.outputs, gi.numOutputTensors, "z", 0);

  if (!t_mp || !t_logsp || !t_ymask || !t_attn || !t_g || !t_nsc || !t_z) {
#ifdef DEBUG_TTS
    std::cout << "[Engine] Flow tensor not found\n";
#endif
    return false;
  }

  fillFloat(t_mp, mP, kMaxPhoneSeq * kLatentChannels);
  fillFloat(t_logsp, logsP, kMaxPhoneSeq * kLatentChannels);
  fillFloat(t_ymask, yMask, kBatchSize * kMaxFrames);
  fillFloat(t_attn, attnPath, kMaxPhoneSeq * kMaxFrames);
  fillFloat(t_g, gVec, kSpeakerVecDim);
  setFloat(t_nsc, 0, noiseScale);

  if (!runGraph(m_flow))
    return false;

  drainFloat(t_z, zOut, kLatentChannels * kMaxFrames);

#ifdef DEBUG_TTS
  std::printf("[flow_out] z dtype=%d  z[0..3]:", (int)QNN_TENSOR_GET_DATA_TYPE(t_z));
  for (int i = 0; i < 4; ++i)
    std::printf(" %.5f", zOut[i]);
  std::printf("\n");
#endif

  return true;
}

// ─── executeDecoder ───────────────────────────────────────────────────────
// decoder.bin: [z_chunk, g] → audio
// z_chunk: [kLatentChannels, kDecoderWindow + 2*kDecoderOverlap] = [192, 64]
// audio:   [(kDecoderWindow + 2*kDecoderOverlap) * kHopSamples] = [32768]

bool InferenceRunner::executeDecoder(const float * zWindow, const float * gVec, float * audioOut)
{
  if (m_decoder.graphCount == 0)
    return false;
  GraphInfo_t & gi = *(*m_decoder.graphs);

  Qnn_Tensor_t * t_z = resolveTensor(m_decoder.inputs, gi.numInputTensors, "z", 0);
  Qnn_Tensor_t * t_g = resolveTensor(m_decoder.inputs, gi.numInputTensors, "g", 1);
  Qnn_Tensor_t * t_audio = resolveTensor(m_decoder.outputs, gi.numOutputTensors, "audio", 0);

  if (!t_z || !t_g || !t_audio) {
#ifdef DEBUG_TTS
    std::cout << "[Engine] Decoder tensor not found\n";
#endif
    return false;
  }

  constexpr size_t kZElems = kLatentChannels * (kDecoderWindow + 2 * kDecoderOverlap);
  constexpr size_t kAudioElems = (kDecoderWindow + 2 * kDecoderOverlap) * kHopSamples;

  fillFloat(t_z, zWindow, kZElems);
  fillFloat(t_g, gVec, kSpeakerVecDim);

#ifdef DEBUG_TTS
  std::printf("[dec_in] z[0..3]:");
  for (int i = 0; i < 4; ++i)
    std::printf(" %.5f", zWindow[i]);
  std::printf("  g[0..3]:");
  for (int i = 0; i < 4; ++i)
    std::printf(" %.5f", gVec[i]);
  std::printf("\n");
#endif

  if (!runGraph(m_decoder))
    return false;

  drainFloat(t_audio, audioOut, kAudioElems);

#ifdef DEBUG_TTS
  std::printf("[dec_out] audio dtype=%d  audio[0..3]:", (int)QNN_TENSOR_GET_DATA_TYPE(t_audio));
  for (int i = 0; i < 4; ++i)
    std::printf(" %.5f", audioOut[i]);
  std::printf("\n");
#endif

  return true;
}

// ─── Performance helpers ──────────────────────────────────────────────────

void InferenceRunner::activateHighPerf()
{
  QnnHtpPerfInfrastructure_PowerConfig_t cfg;
  memset(&cfg, 0, sizeof(cfg));
  cfg.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
  cfg.dcvsV3Config.dcvsEnable = 1;
  cfg.dcvsV3Config.setDcvsEnable = 1;
  cfg.dcvsV3Config.contextId = getPerfConfigId();
  cfg.dcvsV3Config.powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
  cfg.dcvsV3Config.setSleepLatency = 1;
  cfg.dcvsV3Config.setBusParams = 1;
  cfg.dcvsV3Config.setCoreParams = 1;
  cfg.dcvsV3Config.sleepDisable = 1;
  cfg.dcvsV3Config.setSleepDisable = 1;
  cfg.dcvsV3Config.sleepLatency = kPerfLatencyLow;
  cfg.dcvsV3Config.busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
  cfg.dcvsV3Config.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
  cfg.dcvsV3Config.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
  cfg.dcvsV3Config.coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
  cfg.dcvsV3Config.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
  cfg.dcvsV3Config.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
  const QnnHtpPerfInfrastructure_PowerConfig_t * cfgs[] = { &cfg, nullptr };

  QnnHtpPerfInfrastructure_PowerConfig_t dcvsCfg;
  memset(&dcvsCfg, 0, sizeof(dcvsCfg));
  dcvsCfg.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
  dcvsCfg.dcvsV3Config.dcvsEnable = 0;
  dcvsCfg.dcvsV3Config.setDcvsEnable = 1;
  dcvsCfg.dcvsV3Config.powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_ADJUST_UP_DOWN;
  dcvsCfg.dcvsV3Config.contextId = getPerfConfigId();
  const QnnHtpPerfInfrastructure_PowerConfig_t * dcvsCfgs[] = { &dcvsCfg, nullptr };

  m_perf.setPowerConfig(getPerfConfigId(), cfgs);
  m_perf.setPowerConfig(getPerfConfigId(), dcvsCfgs);
}

void InferenceRunner::deactivateHighPerf()
{
  QnnHtpPerfInfrastructure_PowerConfig_t cfg;
  memset(&cfg, 0, sizeof(cfg));
  cfg.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
  cfg.dcvsV3Config.dcvsEnable = 1;
  cfg.dcvsV3Config.setDcvsEnable = 1;
  cfg.dcvsV3Config.contextId = getPerfConfigId();
  cfg.dcvsV3Config.sleepLatency = kPerfLatencyHigh;
  cfg.dcvsV3Config.setSleepLatency = 1;
  cfg.dcvsV3Config.sleepDisable = 0;
  cfg.dcvsV3Config.setSleepDisable = 0;
  cfg.dcvsV3Config.powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_POWER_SAVER_MODE;
  cfg.dcvsV3Config.busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_MIN_VOLTAGE_CORNER;
  cfg.dcvsV3Config.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MIN_VOLTAGE_CORNER;
  cfg.dcvsV3Config.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MIN_VOLTAGE_CORNER;
  cfg.dcvsV3Config.setBusParams = 1;
  cfg.dcvsV3Config.coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_MIN_VOLTAGE_CORNER;
  cfg.dcvsV3Config.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MIN_VOLTAGE_CORNER;
  cfg.dcvsV3Config.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MIN_VOLTAGE_CORNER;
  cfg.dcvsV3Config.setCoreParams = 1;
  const QnnHtpPerfInfrastructure_PowerConfig_t * cfgs[] = { &cfg, nullptr };

  QnnHtpPerfInfrastructure_PowerConfig_t dcvsCfg;
  memset(&dcvsCfg, 0, sizeof(dcvsCfg));
  dcvsCfg.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
  dcvsCfg.dcvsV3Config.dcvsEnable = 1;
  dcvsCfg.dcvsV3Config.setDcvsEnable = 1;
  dcvsCfg.dcvsV3Config.powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_ADJUST_UP_DOWN;
  dcvsCfg.dcvsV3Config.contextId = getPerfConfigId();
  const QnnHtpPerfInfrastructure_PowerConfig_t * dcvsCfgs[] = { &dcvsCfg, nullptr };

  m_perf.setPowerConfig(getPerfConfigId(), cfgs);
  m_perf.setPowerConfig(getPerfConfigId(), dcvsCfgs);
}

// ─── Teardown ──────────────────────────────────────────────────────────────

void InferenceRunner::releaseCtx(GraphCtx & ctx)
{
  if (ctx.inputs && ctx.graphCount > 0)
    m_io.tearDownInputAndOutputTensors(
        ctx.inputs, ctx.outputs, (*ctx.graphs)->numInputTensors, (*ctx.graphs)->numOutputTensors);
  ctx.inputs = nullptr;
  ctx.outputs = nullptr;

  if (ctx.ctx && m_iface.contextFree)
    m_iface.contextFree(ctx.ctx, nullptr);
  ctx.ctx = nullptr;
  ctx.graphs = nullptr;
  ctx.graphCount = 0;
}

void InferenceRunner::releaseDevice()
{
  if (m_hasDevice && m_device && m_iface.deviceFree)
    m_iface.deviceFree(m_device);
  m_device = nullptr;
}

void InferenceRunner::releaseBackend()
{
  if (m_backend && m_iface.backendFree)
    m_iface.backendFree(m_backend);
  m_backend = nullptr;
}

// ─── Metadata copy helpers ────────────────────────────────────────────────

bool InferenceRunner::copyTensorArray(const Qnn_Tensor_t * src, Qnn_Tensor_t *& dst, uint32_t n)
{
  dst = static_cast<Qnn_Tensor_t *>(calloc(n, sizeof(Qnn_Tensor_t)));
  if (!dst)
    return false;
  for (uint32_t i = 0; i < n; ++i) {
    dst[i] = QNN_TENSOR_INIT;
    if (!m_io.deepCopyQnnTensorInfo(&dst[i], &src[i]))
      return false;
  }
  return true;
}

bool InferenceRunner::copyGraphV1(const QnnSystemContext_GraphInfoV1_t * src, GraphInfo_t * dst)
{
  dst->graphName =
      src->graphName ? datautil::StringOpStrndup(src->graphName, strlen(src->graphName)) : nullptr;
  dst->inputTensors = nullptr;
  dst->numInputTensors = 0;
  dst->outputTensors = nullptr;
  dst->numOutputTensors = 0;
  if (src->graphInputs) {
    if (!copyTensorArray(src->graphInputs, dst->inputTensors, src->numGraphInputs))
      return false;
    dst->numInputTensors = src->numGraphInputs;
  }
  if (src->graphOutputs) {
    if (!copyTensorArray(src->graphOutputs, dst->outputTensors, src->numGraphOutputs))
      return false;
    dst->numOutputTensors = src->numGraphOutputs;
  }
  return true;
}

bool InferenceRunner::copyGraphV3(const QnnSystemContext_GraphInfoV3_t * src, GraphInfo_t * dst)
{
  dst->graphName =
      src->graphName ? datautil::StringOpStrndup(src->graphName, strlen(src->graphName)) : nullptr;
  dst->inputTensors = nullptr;
  dst->numInputTensors = 0;
  dst->outputTensors = nullptr;
  dst->numOutputTensors = 0;
  if (src->graphInputs) {
    if (!copyTensorArray(src->graphInputs, dst->inputTensors, src->numGraphInputs))
      return false;
    dst->numInputTensors = src->numGraphInputs;
  }
  if (src->graphOutputs) {
    if (!copyTensorArray(src->graphOutputs, dst->outputTensors, src->numGraphOutputs))
      return false;
    dst->numOutputTensors = src->numGraphOutputs;
  }
  return true;
}

bool InferenceRunner::copyGraphList(const QnnSystemContext_GraphInfo_t * src,
    uint32_t n,
    GraphInfo_t **& dst)
{
  if (!src)
    return false;
  dst = static_cast<GraphInfo_t **>(calloc(n, sizeof(GraphInfo_t *)));
  GraphInfo_t * arr = static_cast<GraphInfo_t *>(calloc(n, sizeof(GraphInfo_t)));
  if (!dst || !arr)
    return false;
  for (uint32_t i = 0; i < n; ++i) {
    if (src[i].version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1)
      copyGraphV1(&src[i].graphInfoV1, &arr[i]);
    else if (src[i].version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3)
      copyGraphV3(&src[i].graphInfoV3, &arr[i]);
    dst[i] = arr + i;
  }
  return true;
}

bool InferenceRunner::copyGraphMetadata(const QnnSystemContext_BinaryInfo_t * info,
    GraphInfo_t **& graphs,
    uint32_t & count)
{
  if (!info)
    return false;
  count = 0;
  if (info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1 &&
      info->contextBinaryInfoV1.graphs) {
    if (!copyGraphList(
            info->contextBinaryInfoV1.graphs, info->contextBinaryInfoV1.numGraphs, graphs))
      return false;
    count = info->contextBinaryInfoV1.numGraphs;
    return true;
  } else if (info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2 &&
             info->contextBinaryInfoV2.graphs) {
    if (!copyGraphList(
            info->contextBinaryInfoV2.graphs, info->contextBinaryInfoV2.numGraphs, graphs))
      return false;
    count = info->contextBinaryInfoV2.numGraphs;
    return true;
  } else if (info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3 &&
             info->contextBinaryInfoV3.graphs) {
    if (!copyGraphList(
            info->contextBinaryInfoV3.graphs, info->contextBinaryInfoV3.numGraphs, graphs))
      return false;
    count = info->contextBinaryInfoV3.numGraphs;
    return true;
  }
#ifdef DEBUG_TTS
  std::cout << "[Engine] Unrecognised binary info version\n";
#endif
  return false;
}

}  // namespace audio
