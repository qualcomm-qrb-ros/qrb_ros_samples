// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear

#pragma once

#include "QnnTypes.h"

inline uint32_t getQnnTensorId(const Qnn_Tensor_t & tensor)
{
  return tensor.v1.id;
}

inline uint32_t getQnnTensorId(const Qnn_Tensor_t * tensor)
{
  return getQnnTensorId(*tensor);
}

inline const char * getQnnTensorName(const Qnn_Tensor_t & tensor)
{
  return tensor.v1.name;
}
inline const char * getQnnTensorName(const Qnn_Tensor_t * tensor)
{
  return getQnnTensorName(*tensor);
}

inline Qnn_TensorType_t getQnnTensorType(const Qnn_Tensor_t & tensor)
{
  return tensor.v1.type;
}

inline Qnn_TensorType_t getQnnTensorType(const Qnn_Tensor_t * tensor)
{
  return getQnnTensorType(*tensor);
}

inline Qnn_TensorDataFormat_t getQnnTensorDataFormat(const Qnn_Tensor_t & tensor)
{
  return tensor.v1.dataFormat;
}

inline Qnn_TensorDataFormat_t getQnnTensorDataFormat(const Qnn_Tensor_t * tensor)
{
  return getQnnTensorDataFormat(*tensor);
}

inline Qnn_DataType_t getQnnTensorDataType(const Qnn_Tensor_t & tensor)
{
  return tensor.v1.dataType;
}

inline Qnn_DataType_t getQnnTensorDataType(const Qnn_Tensor_t * tensor)
{
  return getQnnTensorDataType(*tensor);
}

inline Qnn_QuantizeParams_t getQnnTensorQuantParams(const Qnn_Tensor_t & tensor)
{
  return tensor.v1.quantizeParams;
}

inline Qnn_QuantizeParams_t getQnnTensorQuantParams(const Qnn_Tensor_t * const tensor)
{
  if (tensor != nullptr) {
    return getQnnTensorQuantParams(*tensor);
  }
  return QNN_QUANTIZE_PARAMS_INIT;
}

inline uint32_t getQnnTensorRank(const Qnn_Tensor_t & tensor)
{
  return tensor.v1.rank;
}

inline uint32_t getQnnTensorRank(const Qnn_Tensor_t * const tensor)
{
  if (tensor != nullptr) {
    return getQnnTensorRank(*tensor);
  }
  return 0u;
}

inline uint32_t * getQnnTensorDimensions(const Qnn_Tensor_t & tensor)
{
  return tensor.v1.dimensions;
}

inline uint32_t * getQnnTensorDimensions(const Qnn_Tensor_t * tensor)
{
  return getQnnTensorDimensions(*tensor);
}

inline uint8_t * getQnnTensorIsDynamicDimensions(const Qnn_Tensor_t & tensor)
{
  if (tensor.version == QNN_TENSOR_VERSION_2) {
    return tensor.v2.isDynamicDimensions;
  }
  return NULL;
}

inline uint8_t * getQnnTensorIsDynamicDimensions(const Qnn_Tensor_t * tensor)
{
  return getQnnTensorIsDynamicDimensions(*tensor);
}

inline Qnn_SparseParams_t getQnnTensorSparseParams(const Qnn_Tensor_t & tensor)
{
  if (tensor.version == QNN_TENSOR_VERSION_2) {
    return tensor.v2.sparseParams;
  }
  return QNN_SPARSE_PARAMS_INIT;
}

inline Qnn_SparseParams_t getQnnTensorSparseParams(const Qnn_Tensor_t * tensor)
{
  return getQnnTensorSparseParams(*tensor);
}

inline Qnn_ClientBuffer_t getQnnTensorClientBuf(const Qnn_Tensor_t & tensor)
{
  return tensor.v1.clientBuf;
}

inline Qnn_ClientBuffer_t getQnnTensorClientBuf(const Qnn_Tensor_t * tensor)
{
  return getQnnTensorClientBuf(*tensor);
}

inline void setQnnTensorId(Qnn_Tensor_t & tensor, const uint32_t id)
{
  tensor.v1.id = id;
}

inline void setQnnTensorId(Qnn_Tensor_t * tensor, uint32_t id)
{
  setQnnTensorId(*tensor, id);
}

inline void setQnnTensorName(Qnn_Tensor_t & tensor, const char * const name)
{
  tensor.v1.name = name;
}

inline void setQnnTensorName(Qnn_Tensor_t * tensor, const char * name)
{
  setQnnTensorName(*tensor, name);
}

inline void setQnnTensorType(Qnn_Tensor_t & tensor, Qnn_TensorType_t type)
{
  tensor.v1.type = type;
}

inline void setQnnTensorType(Qnn_Tensor_t * tensor, Qnn_TensorType_t type)
{
  setQnnTensorType(*tensor, type);
}

inline void setQnnTensorDataFormat(Qnn_Tensor_t & tensor, const Qnn_TensorDataFormat_t dataFormat)
{
  tensor.v1.dataFormat = dataFormat;
}

inline void setQnnTensorDataFormat(Qnn_Tensor_t * tensor, Qnn_TensorDataFormat_t format)
{
  setQnnTensorDataFormat(*tensor, format);
}

inline void setQnnTensorDataType(Qnn_Tensor_t & tensor, const Qnn_DataType_t dataType)
{
  tensor.v1.dataType = dataType;
}

inline void setQnnTensorDataType(Qnn_Tensor_t * tensor, Qnn_DataType_t dataType)
{
  setQnnTensorDataType(*tensor, dataType);
}

inline void setQnnTensorQuantParams(Qnn_Tensor_t & tensor,
    const Qnn_QuantizeParams_t quantizeParams)
{
  tensor.v1.quantizeParams = quantizeParams;
}

inline void setQnnTensorQuantParams(Qnn_Tensor_t * tensor, Qnn_QuantizeParams_t params)
{
  setQnnTensorQuantParams(*tensor, params);
}

inline void setQnnTensorRank(Qnn_Tensor_t & tensor, const uint32_t rank)
{
  tensor.v1.rank = rank;
}

inline void setQnnTensorRank(Qnn_Tensor_t * tensor, uint32_t rank)
{
  setQnnTensorRank(*tensor, rank);
}

inline void setQnnTensorDimensions(Qnn_Tensor_t & tensor, uint32_t * const dimensions)
{
  tensor.v1.dimensions = dimensions;
}

inline void setQnnTensorDimensions(Qnn_Tensor_t * tensor, uint32_t * dims)
{
  setQnnTensorDimensions(*tensor, dims);
}

inline void setQnnTensorIsDynamicDimensions(Qnn_Tensor_t & tensor, uint8_t * isDynamic)
{
  if (tensor.version == QNN_TENSOR_VERSION_2) {
    tensor.v2.isDynamicDimensions = isDynamic;
  }
}

inline void setQnnTensorIsDynamicDimensions(Qnn_Tensor_t * tensor, uint8_t * isDynamic)
{
  setQnnTensorIsDynamicDimensions(*tensor, isDynamic);
}

inline void setQnnTensorSparseParams(Qnn_Tensor_t & tensor, Qnn_SparseParams_t sparseParams)
{
  if (tensor.version == QNN_TENSOR_VERSION_2) {
    tensor.v2.sparseParams = sparseParams;
  }
}

inline void setQnnTensorSparseParams(Qnn_Tensor_t * tensor, Qnn_SparseParams_t sparseParams)
{
  setQnnTensorSparseParams(*tensor, sparseParams);
}

inline void setQnnTensorMemType(Qnn_Tensor_t & tensor, const Qnn_TensorMemType_t memType)
{
  tensor.v1.memType = memType;
}

inline void setQnnTensorMemType(Qnn_Tensor_t * tensor, Qnn_TensorMemType_t memType)
{
  setQnnTensorMemType(*tensor, memType);
}

inline void setQnnTensorClientBuf(Qnn_Tensor_t & tensor, const Qnn_ClientBuffer_t clientBuf)
{
  tensor.v1.clientBuf = clientBuf;
}

inline void setQnnTensorClientBuf(Qnn_Tensor_t * tensor, Qnn_ClientBuffer_t clientBuf)
{
  setQnnTensorClientBuf(*tensor, clientBuf);
}

// Accessors for QNN Tensor
#define QNN_TENSOR_GET_ID(tensor) getQnnTensorId(tensor)
#define QNN_TENSOR_GET_NAME(tensor) getQnnTensorName(tensor)
#define QNN_TENSOR_GET_TYPE(tensor) getQnnTensorType(tensor)
#define QNN_TENSOR_GET_DATA_FORMAT(tensor) getQnnTensorDataFormat(tensor)
#define QNN_TENSOR_GET_DATA_TYPE(tensor) getQnnTensorDataType(tensor)
#define QNN_TENSOR_GET_QUANT_PARAMS(tensor) getQnnTensorQuantParams(tensor)
#define QNN_TENSOR_GET_RANK(tensor) getQnnTensorRank(tensor)
#define QNN_TENSOR_GET_DIMENSIONS(tensor) getQnnTensorDimensions(tensor)
#define QNN_TENSOR_GET_IS_DYNAMIC_DIMENSIONS(tensor) getQnnTensorIsDynamicDimensions(tensor)
#define QNN_TENSOR_GET_SPARSE_PARAMS(tensor) getQnnTensorSparseParams(tensor)
#define QNN_TENSOR_GET_CLIENT_BUF(tensor) getQnnTensorClientBuf(tensor)

// Modifiers for QNN Tensor
#define QNN_TENSOR_SET_ID(tensor, value) setQnnTensorId(tensor, value)
#define QNN_TENSOR_SET_NAME(tensor, value) setQnnTensorName(tensor, value)
#define QNN_TENSOR_SET_TYPE(tensor, value) setQnnTensorType(tensor, value)
#define QNN_TENSOR_SET_DATA_FORMAT(tensor, value) setQnnTensorDataFormat(tensor, value)
#define QNN_TENSOR_SET_DATA_TYPE(tensor, value) setQnnTensorDataType(tensor, value)
#define QNN_TENSOR_SET_QUANT_PARAMS(tensor, value) setQnnTensorQuantParams(tensor, value)
#define QNN_TENSOR_SET_RANK(tensor, value) setQnnTensorRank(tensor, value)
#define QNN_TENSOR_SET_DIMENSIONS(tensor, value) setQnnTensorDimensions(tensor, value)
#define QNN_TENSOR_SET_IS_DYNAMIC_DIMENSIONS(tensor, value)                                        \
  setQnnTensorIsDynamicDimensions(tensor, value)
#define QNN_TENSOR_SET_SPARSE_PARAMS(tensor, value) setQnnTensorSparseParams(tensor, value)
#define QNN_TENSOR_SET_MEM_TYPE(tensor, value) setQnnTensorMemType(tensor, value)
#define QNN_TENSOR_SET_CLIENT_BUF(tensor, value) setQnnTensorClientBuf(tensor, value)
