// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear

#include "IOTensor.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>

#include "DataUtil.hpp"
// #include "Logger.hpp"
// #ifndef __hexagon__
// #include "PAL/Directory.hpp"
// #include "PAL/FileOp.hpp"
// #include "PAL/Path.hpp"
// #endif
// #include "PAL/StringOp.hpp"
// #include "QnnTypeMacros.hpp"

// using namespace qnn;
// using namespace qnn::tools;

// Helper method to read data from files to a buffer.
// iotensor::PopulateInputTensorsRetType_t
// iotensor::IOTensor::readDataAndAllocateBuffer(
//     const std::vector<std::string> &filePaths,
//     const size_t filePathsIndexOffset,
//     const bool loopBackToStart,
//     std::vector<size_t> dims,
//     Qnn_DataType_t dataType,
//     uint8_t **bufferToCopy)
// {
//     StatusCode returnStatus = StatusCode::SUCCESS;
//     *bufferToCopy = nullptr;
//     returnStatus = allocateBuffer(bufferToCopy, dims, dataType);
//     size_t numFilesPopulated = 0;
//     size_t batchSize = 0;
//     datautil::StatusCode status;
//     std::tie(status, numFilesPopulated, batchSize) =
//         datautil::readBatchData(filePaths,
//                                 filePathsIndexOffset,
//                                 loopBackToStart,
//                                 dims,
//                                 dataType,
//                                 reinterpret_cast<uint8_t *>(*bufferToCopy));
//     if (datautil::StatusCode::SUCCESS != status)
//     {
//         QNN_ERROR("Failure in datautil::readBatchData");
//         returnStatus = StatusCode::FAILURE;
//     }
//     if (StatusCode::SUCCESS != returnStatus)
//     {
//         if (nullptr != *bufferToCopy)
//         {
//             free(*bufferToCopy);
//             *bufferToCopy = nullptr;
//         }
//     }
//     return {returnStatus, numFilesPopulated, batchSize};
// }

// Helper method to copy a float buffer, quantize it, and copy
// it to a tensor (Qnn_Tensor_t) buffer.
bool IOTensor::copyFromFloatToNative(float * floatBuffer, Qnn_Tensor_t * tensor)
{
  if (nullptr == floatBuffer || nullptr == tensor) {
    std::cout << "copyFromFloatToNative(): received a nullptr" << std::endl;
    return false;
  }

  bool returnStatus = true;
  std::vector<size_t> dims;
  fillDims(dims, QNN_TENSOR_GET_DIMENSIONS(tensor), QNN_TENSOR_GET_RANK(tensor));

  switch (QNN_TENSOR_GET_DATA_TYPE(tensor)) {
    case QNN_DATATYPE_UFIXED_POINT_8:
      datautil::floatToTfN<uint8_t>(static_cast<uint8_t *>(QNN_TENSOR_GET_CLIENT_BUF(tensor).data),
          floatBuffer, QNN_TENSOR_GET_QUANT_PARAMS(tensor).scaleOffsetEncoding.offset,
          QNN_TENSOR_GET_QUANT_PARAMS(tensor).scaleOffsetEncoding.scale,
          datautil::calculateElementCount(dims));
      break;

    case QNN_DATATYPE_UFIXED_POINT_16:
      datautil::floatToTfN<uint16_t>(
          static_cast<uint16_t *>(QNN_TENSOR_GET_CLIENT_BUF(tensor).data), floatBuffer,
          QNN_TENSOR_GET_QUANT_PARAMS(tensor).scaleOffsetEncoding.offset,
          QNN_TENSOR_GET_QUANT_PARAMS(tensor).scaleOffsetEncoding.scale,
          datautil::calculateElementCount(dims));
      break;

    case QNN_DATATYPE_UINT_8:
      if (true != datautil::castFromFloat<uint8_t>(
                      static_cast<uint8_t *>(QNN_TENSOR_GET_CLIENT_BUF(tensor).data), floatBuffer,
                      datautil::calculateElementCount(dims))) {
        std::cout << "failure in castFromFloat<uint8_t>" << std::endl;
        returnStatus = false;
      }
      break;

    case QNN_DATATYPE_UINT_16:
      if (true != datautil::castFromFloat<uint16_t>(
                      static_cast<uint16_t *>(QNN_TENSOR_GET_CLIENT_BUF(tensor).data), floatBuffer,
                      datautil::calculateElementCount(dims))) {
        std::cout << "failure in castFromFloat<uint16_t>" << std::endl;
        returnStatus = false;
      }
      break;

    case QNN_DATATYPE_UINT_32:
      if (true != datautil::castFromFloat<uint32_t>(
                      static_cast<uint32_t *>(QNN_TENSOR_GET_CLIENT_BUF(tensor).data), floatBuffer,
                      datautil::calculateElementCount(dims))) {
        std::cout << ("failure in castFromFloat<uint32_t>") << std::endl;
        returnStatus = false;
      }
      break;

    case QNN_DATATYPE_UINT_64:
      if (true != datautil::castFromFloat<uint64_t>(
                      static_cast<uint64_t *>(QNN_TENSOR_GET_CLIENT_BUF(tensor).data), floatBuffer,
                      datautil::calculateElementCount(dims))) {
        std::cout << ("failure in castFromFloat<uint64_t>") << std::endl;
        returnStatus = false;
      }
      break;

    case QNN_DATATYPE_INT_8:
      if (true != datautil::castFromFloat<int8_t>(
                      static_cast<int8_t *>(QNN_TENSOR_GET_CLIENT_BUF(tensor).data), floatBuffer,
                      datautil::calculateElementCount(dims))) {
        std::cout << ("failure in castFromFloat<int8_t>") << std::endl;
        returnStatus = false;
      }
      break;

    case QNN_DATATYPE_INT_16:
      if (true != datautil::castFromFloat<int16_t>(
                      static_cast<int16_t *>(QNN_TENSOR_GET_CLIENT_BUF(tensor).data), floatBuffer,
                      datautil::calculateElementCount(dims))) {
        std::cout << ("failure in castFromFloat<int16_t>") << std::endl;
        returnStatus = false;
      }
      break;

    case QNN_DATATYPE_INT_32:
      if (true != datautil::castFromFloat<int32_t>(
                      static_cast<int32_t *>(QNN_TENSOR_GET_CLIENT_BUF(tensor).data), floatBuffer,
                      datautil::calculateElementCount(dims))) {
        std::cout << ("failure in castFromFloat<int32_t>") << std::endl;
        returnStatus = false;
      }
      break;

    case QNN_DATATYPE_INT_64:
      if (true != datautil::castFromFloat<int64_t>(
                      static_cast<int64_t *>(QNN_TENSOR_GET_CLIENT_BUF(tensor).data), floatBuffer,
                      datautil::calculateElementCount(dims))) {
        std::cout << ("failure in castFromFloat<int64_t>") << std::endl;
        returnStatus = false;
      }
      break;

    case QNN_DATATYPE_BOOL_8:
      if (true != datautil::castFromFloat<uint8_t>(
                      static_cast<uint8_t *>(QNN_TENSOR_GET_CLIENT_BUF(tensor).data), floatBuffer,
                      datautil::calculateElementCount(dims))) {
        std::cout << ("failure in castFromFloat<bool>") << std::endl;
        returnStatus = false;
      }
      break;

    default:
      std::cout << ("Datatype not supported yet!") << std::endl;
      returnStatus = false;
      break;
  }
  return returnStatus;
}

// Setup details for Qnn_Tensor_t for execution
// based on information in Qnn_TensorWrapper_t provided by model.so.
bool IOTensor::setupTensors(Qnn_Tensor_t ** tensors,
    uint32_t tensorCount,
    Qnn_Tensor_t * tensorWrappers)
{
  if (nullptr == tensorWrappers) {
    std::cout << "tensorWrappers is nullptr" << std::endl;
    return false;
  }
  if (0 == tensorCount) {
    std::cout << "tensor count is 0. Nothing to setup." << std::endl;
    return false;
  }
  auto returnStatus = true;
  *tensors = (Qnn_Tensor_t *)calloc(1, tensorCount * sizeof(Qnn_Tensor_t));
  if (nullptr == *tensors) {
    std::cout << "mem alloc failed for *tensors" << std::endl;
    return false;
  }
  for (size_t tensorIdx = 0; tensorIdx < tensorCount; tensorIdx++) {
    Qnn_Tensor_t wrapperTensor = tensorWrappers[tensorIdx];
    std::vector<size_t> dims;
    fillDims(dims, QNN_TENSOR_GET_DIMENSIONS(wrapperTensor), QNN_TENSOR_GET_RANK(wrapperTensor));
    (*tensors)[tensorIdx] = QNN_TENSOR_INIT;
    if (deepCopyQnnTensorInfo(((*tensors) + tensorIdx), &wrapperTensor) != true) {
      return false;
    } else {
      std::cout << "deepCopyQnnTensorInfo successful" << std::endl;
      QNN_TENSOR_SET_MEM_TYPE(((*tensors) + tensorIdx), QNN_TENSORMEMTYPE_RAW);
    }

    Qnn_ClientBuffer_t clientBuffer = QNN_CLIENT_BUFFER_INIT;
    returnStatus = allocateBuffer(reinterpret_cast<uint8_t **>(&clientBuffer.data), dims,
        QNN_TENSOR_GET_DATA_TYPE((*tensors) + tensorIdx));
    size_t length{ 0 };
    length = datautil::calculateLength(dims, QNN_TENSOR_GET_DATA_TYPE((*tensors) + tensorIdx));
    if (length == 0)
      returnStatus = false;
    clientBuffer.dataSize = length;
    QNN_TENSOR_SET_CLIENT_BUF(((*tensors) + tensorIdx), clientBuffer);
    if (true != returnStatus) {
      std::cout << ("Failure in setupTensors, cleaning up resources") << std::endl;
      if (nullptr != (QNN_TENSOR_GET_CLIENT_BUF((*tensors) + tensorIdx)).data) {
        free(QNN_TENSOR_GET_CLIENT_BUF((*tensors) + tensorIdx).data);
      }
      tearDownTensors(*tensors, tensorIdx);
      *tensors = nullptr;
      std::cout << ("Failure in setupTensors, done cleaning up resources") << std::endl;
      return returnStatus;
    }
  }
  return returnStatus;
}

// Setup details for all input and output tensors for graph execution.
bool IOTensor::setupInputAndOutputTensors(Qnn_Tensor_t ** inputs,
    Qnn_Tensor_t ** outputs,
    GraphInfo_t graphInfo)
{
  if (true != setupTensors(inputs, graphInfo.numInputTensors, (graphInfo.inputTensors))) {
    std::cout << "Failure in setting up input tensors" << std::endl;
    return false;
  }
  if (true != setupTensors(outputs, graphInfo.numOutputTensors, (graphInfo.outputTensors))) {
    std::cout << "Failure in setting up output tensors" << std::endl;
    return false;
  }

  return true;
}

// Clean up all tensors related data after execution.
bool IOTensor::tearDownTensors(Qnn_Tensor_t * tensors, uint32_t tensorCount)
{
  for (size_t tensorIdx = 0; tensorIdx < tensorCount; tensorIdx++) {
    printf("freeing resources for tensor: %d", tensorIdx);
    if (nullptr != QNN_TENSOR_GET_DIMENSIONS(tensors[tensorIdx])) {
      std::cout << "freeing dimensions" << std::endl;
      free(QNN_TENSOR_GET_DIMENSIONS(tensors[tensorIdx]));
    } else {
      return false;
    }
    if (nullptr != QNN_TENSOR_GET_CLIENT_BUF(tensors[tensorIdx]).data) {
      std::cout << "freeing clientBuf.data" << std::endl;
      free(QNN_TENSOR_GET_CLIENT_BUF(tensors[tensorIdx]).data);
    } else {
      return false;
    }
  }
  free(tensors);
  return true;
}

// Clean up all input and output tensors after execution.
bool IOTensor::tearDownInputAndOutputTensors(Qnn_Tensor_t * inputs,
    Qnn_Tensor_t * outputs,
    size_t numInputTensors,
    size_t numOutputTensors)
{
  if (nullptr != inputs) {
    std::cout << "cleaning up resources for input tensors" << std::endl;
    tearDownTensors(inputs, numInputTensors);
    inputs = nullptr;
  } else {
    std::cout << "inputs isnullptr" << std::endl;
    return false;
  }
  if (nullptr != outputs) {
    std::cout << "cleaning up resources for output tensors" << std::endl;
    tearDownTensors(outputs, numOutputTensors);
    outputs = nullptr;
  } else {
    std::cout << "outputs is nullptr" << std::endl;
    return false;
  }

  return true;
}

// Helper method to allocate a buffer.
bool IOTensor::allocateBuffer(uint8_t ** buffer, std::vector<size_t> dims, Qnn_DataType_t dataType)
{
  size_t elementCount = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<size_t>());
  auto returnStatus = true;
  switch (dataType) {
    case QNN_DATATYPE_FLOAT_32:
      std::cout << "allocating float buffer" << std::endl;
      returnStatus = allocateBuffer<float>(reinterpret_cast<float **>(buffer), elementCount);
      break;

    case QNN_DATATYPE_UINT_8:
    case QNN_DATATYPE_UFIXED_POINT_8:
      std::cout << "allocating uint8_t buffer" << std::endl;
      returnStatus = allocateBuffer<uint8_t>(reinterpret_cast<uint8_t **>(buffer), elementCount);
      break;

    case QNN_DATATYPE_UINT_16:
    case QNN_DATATYPE_UFIXED_POINT_16:
      std::cout << "allocating uint16_t buffer" << std::endl;
      returnStatus = allocateBuffer<uint16_t>(reinterpret_cast<uint16_t **>(buffer), elementCount);
      break;

    case QNN_DATATYPE_UINT_32:
      std::cout << "allocating uint32_t buffer" << std::endl;
      returnStatus = allocateBuffer<uint32_t>(reinterpret_cast<uint32_t **>(buffer), elementCount);
      break;

    case QNN_DATATYPE_UINT_64:
      std::cout << "allocating uint64_t buffer" << std::endl;
      returnStatus = allocateBuffer<uint64_t>(reinterpret_cast<uint64_t **>(buffer), elementCount);
      break;

    case QNN_DATATYPE_INT_8:
      std::cout << "allocating int8_t buffer" << std::endl;
      returnStatus = allocateBuffer<int8_t>(reinterpret_cast<int8_t **>(buffer), elementCount);
      break;

    case QNN_DATATYPE_INT_16:
      std::cout << "allocating int16_t buffer" << std::endl;
      returnStatus = allocateBuffer<int16_t>(reinterpret_cast<int16_t **>(buffer), elementCount);
      break;

    case QNN_DATATYPE_INT_32:
      std::cout << "allocating int32_t buffer" << std::endl;
      returnStatus = allocateBuffer<int32_t>(reinterpret_cast<int32_t **>(buffer), elementCount);
      break;

    case QNN_DATATYPE_INT_64:
      std::cout << "allocating int64_t buffer" << std::endl;
      returnStatus = allocateBuffer<int64_t>(reinterpret_cast<int64_t **>(buffer), elementCount);
      break;

    case QNN_DATATYPE_BOOL_8:
      std::cout << "allocating bool buffer" << std::endl;
      returnStatus = allocateBuffer<uint8_t>(reinterpret_cast<uint8_t **>(buffer), elementCount);
      break;

    default:
      std::cout << "Datatype not supported yet!" << std::endl;
      returnStatus = false;
      break;
  }
  return returnStatus;
}

// Helper method to allocate a buffer.
template <typename T>
bool IOTensor::allocateBuffer(T ** buffer, size_t & elementCount)
{
  /*
  printf("ElementCount: %d, sizeof(T): %d, total size: %d",
         elementCount,
         sizeof(T),
         elementCount * sizeof(T));
  */
  *buffer = (T *)malloc(elementCount * sizeof(T));
  if (nullptr == *buffer) {
    std::cout << "mem alloc failed for *buffer" << std::endl;
    return false;
  }
  return true;
}

bool IOTensor::deepCopyQnnTensorInfo(Qnn_Tensor_t * dst, const Qnn_Tensor_t * src)
{
  if (nullptr == dst || nullptr == src) {
    std::cout << "Received nullptr" << std::endl;
    return false;
  }
  // set tensor.version before using QNN_TENSOR_SET macros, as they require the
  // version to be set to correctly assign values
  dst->version = src->version;
  const char * tensorName = QNN_TENSOR_GET_NAME(src);
  if (!tensorName) {
    QNN_TENSOR_SET_NAME(dst, nullptr);
  } else {
    QNN_TENSOR_SET_NAME(dst, datautil::StringOpStrndup(tensorName, strlen(tensorName)));
  }
  QNN_TENSOR_SET_ID(dst, QNN_TENSOR_GET_ID(src));
  QNN_TENSOR_SET_TYPE(dst, QNN_TENSOR_GET_TYPE(src));
  QNN_TENSOR_SET_DATA_FORMAT(dst, QNN_TENSOR_GET_DATA_FORMAT(src));
  QNN_TENSOR_SET_DATA_TYPE(dst, QNN_TENSOR_GET_DATA_TYPE(src));
  Qnn_QuantizeParams_t qParams = QNN_QUANTIZE_PARAMS_INIT;
  qParams.encodingDefinition = QNN_TENSOR_GET_QUANT_PARAMS(src).encodingDefinition;
  qParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
  if (QNN_TENSOR_GET_QUANT_PARAMS(src).quantizationEncoding ==
      QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
    qParams.quantizationEncoding = QNN_TENSOR_GET_QUANT_PARAMS(src).quantizationEncoding;
    qParams.scaleOffsetEncoding = QNN_TENSOR_GET_QUANT_PARAMS(src).scaleOffsetEncoding;
  } else if (QNN_TENSOR_GET_QUANT_PARAMS(src).quantizationEncoding ==
             QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET) {
    qParams.quantizationEncoding = QNN_TENSOR_GET_QUANT_PARAMS(src).quantizationEncoding;
    qParams.axisScaleOffsetEncoding.axis =
        QNN_TENSOR_GET_QUANT_PARAMS(src).axisScaleOffsetEncoding.axis;
    qParams.axisScaleOffsetEncoding.numScaleOffsets =
        QNN_TENSOR_GET_QUANT_PARAMS(src).axisScaleOffsetEncoding.numScaleOffsets;
    if (QNN_TENSOR_GET_QUANT_PARAMS(src).axisScaleOffsetEncoding.numScaleOffsets > 0) {
      qParams.axisScaleOffsetEncoding.scaleOffset = (Qnn_ScaleOffset_t *)malloc(
          QNN_TENSOR_GET_QUANT_PARAMS(src).axisScaleOffsetEncoding.numScaleOffsets *
          sizeof(Qnn_ScaleOffset_t));
      if (qParams.axisScaleOffsetEncoding.scaleOffset) {
        for (size_t idx = 0;
             idx < QNN_TENSOR_GET_QUANT_PARAMS(src).axisScaleOffsetEncoding.numScaleOffsets;
             idx++) {
          qParams.axisScaleOffsetEncoding.scaleOffset[idx].scale =
              QNN_TENSOR_GET_QUANT_PARAMS(src).axisScaleOffsetEncoding.scaleOffset[idx].scale;
          qParams.axisScaleOffsetEncoding.scaleOffset[idx].offset =
              QNN_TENSOR_GET_QUANT_PARAMS(src).axisScaleOffsetEncoding.scaleOffset[idx].offset;
        }
      }
    }
  }
  QNN_TENSOR_SET_QUANT_PARAMS(dst, qParams);
  QNN_TENSOR_SET_RANK(dst, QNN_TENSOR_GET_RANK(src));
  QNN_TENSOR_SET_DIMENSIONS(dst, nullptr);
  if (QNN_TENSOR_GET_RANK(src) > 0) {
    QNN_TENSOR_SET_DIMENSIONS(dst, (uint32_t *)malloc(QNN_TENSOR_GET_RANK(src) * sizeof(uint32_t)));
    if (QNN_TENSOR_GET_DIMENSIONS(dst)) {
      datautil::StringOpMemscpy(QNN_TENSOR_GET_DIMENSIONS(dst),
          QNN_TENSOR_GET_RANK(src) * sizeof(uint32_t), QNN_TENSOR_GET_DIMENSIONS(src),
          QNN_TENSOR_GET_RANK(src) * sizeof(uint32_t));
    }
    if (QNN_TENSOR_GET_IS_DYNAMIC_DIMENSIONS(src)) {
      QNN_TENSOR_SET_IS_DYNAMIC_DIMENSIONS(
          dst, (uint8_t *)malloc(QNN_TENSOR_GET_RANK(src) * sizeof(uint8_t)));
      datautil::StringOpMemscpy(QNN_TENSOR_GET_IS_DYNAMIC_DIMENSIONS(dst),
          QNN_TENSOR_GET_RANK(src) * sizeof(uint8_t), QNN_TENSOR_GET_IS_DYNAMIC_DIMENSIONS(src),
          QNN_TENSOR_GET_RANK(src) * sizeof(uint8_t));
    }
  }
  QNN_TENSOR_SET_SPARSE_PARAMS(dst, QNN_TENSOR_GET_SPARSE_PARAMS(src));
  return true;
}

// Convert data to float or de-quantization. This is used when
// user requests for float output and the model produces
// non-float output.

// iotensor::StatusCode iotensor::IOTensor::convertToFloat(float **out,
// Qnn_Tensor_t *tensor)
// {
//     if (nullptr == tensor)
//     {
//         QNN_ERROR("tensors is nullptr");
//         return StatusCode::FAILURE;
//     }
//     std::vector<size_t> dims;
//     fillDims(dims, QNN_TENSOR_GET_DIMENSIONS(tensor),
//     QNN_TENSOR_GET_RANK(tensor)); auto returnStatus = StatusCode::SUCCESS;
//     size_t elementCount = datautil::calculateElementCount(dims);
//     returnStatus = allocateBuffer<float>(out, elementCount);
//     if (StatusCode::SUCCESS != returnStatus)
//     {
//         QNN_ERROR("failure in allocateBuffer<float>");
//         return returnStatus;
//     }
//     switch (QNN_TENSOR_GET_DATA_TYPE(tensor))
//     {
//     case QNN_DATATYPE_UFIXED_POINT_8:
//         if (datautil::StatusCode::SUCCESS !=
//             datautil::tfNToFloat<uint8_t>(
//                 *out,
//                 reinterpret_cast<uint8_t
//                 *>(QNN_TENSOR_GET_CLIENT_BUF(tensor).data),
//                 QNN_TENSOR_GET_QUANT_PARAMS(tensor).scaleOffsetEncoding.offset,
//                 QNN_TENSOR_GET_QUANT_PARAMS(tensor).scaleOffsetEncoding.scale,
//                 elementCount))
//         {
//             QNN_ERROR("failure in tfNToFloat<uint8_t>");
//             returnStatus = StatusCode::FAILURE;
//         }
//         break;

//     case QNN_DATATYPE_UFIXED_POINT_16:
//         if (datautil::StatusCode::SUCCESS !=
//             datautil::tfNToFloat<uint16_t>(
//                 *out,
//                 reinterpret_cast<uint16_t
//                 *>(QNN_TENSOR_GET_CLIENT_BUF(tensor).data),
//                 QNN_TENSOR_GET_QUANT_PARAMS(tensor).scaleOffsetEncoding.offset,
//                 QNN_TENSOR_GET_QUANT_PARAMS(tensor).scaleOffsetEncoding.scale,
//                 elementCount))
//         {
//             QNN_ERROR("failure in tfNToFloat<uint8_t>");
//             returnStatus = StatusCode::FAILURE;
//         }
//         break;

//     case QNN_DATATYPE_UINT_8:
//         if (datautil::StatusCode::SUCCESS !=
//             datautil::castToFloat<uint8_t>(
//                 *out,
//                 reinterpret_cast<uint8_t
//                 *>(QNN_TENSOR_GET_CLIENT_BUF(tensor).data), elementCount))
//         {
//             QNN_ERROR("failure in castToFloat<uint8_t>");
//             returnStatus = StatusCode::FAILURE;
//         }
//         break;

//     case QNN_DATATYPE_UINT_16:
//         if (datautil::StatusCode::SUCCESS !=
//             datautil::castToFloat<uint16_t>(
//                 *out,
//                 reinterpret_cast<uint16_t
//                 *>(QNN_TENSOR_GET_CLIENT_BUF(tensor).data), elementCount))
//         {
//             QNN_ERROR("failure in castToFloat<uint16_t>");
//             returnStatus = StatusCode::FAILURE;
//         }
//         break;

//     case QNN_DATATYPE_UINT_32:
//         if (datautil::StatusCode::SUCCESS !=
//             datautil::castToFloat<uint32_t>(
//                 *out,
//                 reinterpret_cast<uint32_t
//                 *>(QNN_TENSOR_GET_CLIENT_BUF(tensor).data), elementCount))
//         {
//             QNN_ERROR("failure in castToFloat<uint32_t>");
//             returnStatus = StatusCode::FAILURE;
//         }
//         break;

//     case QNN_DATATYPE_UINT_64:
//         if (datautil::StatusCode::SUCCESS !=
//             datautil::castToFloat<uint64_t>(
//                 *out,
//                 reinterpret_cast<uint64_t
//                 *>(QNN_TENSOR_GET_CLIENT_BUF(tensor).data), elementCount))
//         {
//             QNN_ERROR("failure in castToFloat<uint64_t>");
//             returnStatus = StatusCode::FAILURE;
//         }
//         break;

//     case QNN_DATATYPE_INT_8:
//         if (datautil::StatusCode::SUCCESS !=
//             datautil::castToFloat<int8_t>(
//                 *out,
//                 reinterpret_cast<int8_t
//                 *>(QNN_TENSOR_GET_CLIENT_BUF(tensor).data), elementCount))
//         {
//             QNN_ERROR("failure in castToFloat<int8_t>");
//             returnStatus = StatusCode::FAILURE;
//         }
//         break;

//     case QNN_DATATYPE_INT_16:
//         if (datautil::StatusCode::SUCCESS !=
//             datautil::castToFloat<int16_t>(
//                 *out,
//                 reinterpret_cast<int16_t
//                 *>(QNN_TENSOR_GET_CLIENT_BUF(tensor).data), elementCount))
//         {
//             QNN_ERROR("failure in castToFloat<int16_t>");
//             returnStatus = StatusCode::FAILURE;
//         }
//         break;

//     case QNN_DATATYPE_INT_32:
//         if (datautil::StatusCode::SUCCESS !=
//             datautil::castToFloat<int32_t>(
//                 *out,
//                 reinterpret_cast<int32_t
//                 *>(QNN_TENSOR_GET_CLIENT_BUF(tensor).data), elementCount))
//         {
//             QNN_ERROR("failure in castToFloat<int32_t>");
//             returnStatus = StatusCode::FAILURE;
//         }
//         break;

//     case QNN_DATATYPE_INT_64:
//         if (datautil::StatusCode::SUCCESS !=
//             datautil::castToFloat<int64_t>(
//                 *out,
//                 reinterpret_cast<int64_t
//                 *>(QNN_TENSOR_GET_CLIENT_BUF(tensor).data), elementCount))
//         {
//             QNN_ERROR("failure in castToFloat<int64_t>");
//             returnStatus = StatusCode::FAILURE;
//         }
//         break;

//     case QNN_DATATYPE_BOOL_8:
//         if (datautil::StatusCode::SUCCESS !=
//             datautil::castToFloat<uint8_t>(
//                 *out,
//                 reinterpret_cast<uint8_t
//                 *>(QNN_TENSOR_GET_CLIENT_BUF(tensor).data), elementCount))
//         {
//             QNN_ERROR("failure in castToFloat<bool>");
//             returnStatus = StatusCode::FAILURE;
//         }
//         break;

//     default:
//         QNN_ERROR("Datatype not supported yet!");
//         returnStatus = StatusCode::FAILURE;
//         break;
//     }
//     if (StatusCode::SUCCESS != returnStatus)
//     {
//         QNN_DEBUG("freeing *out");
//         if (*out != nullptr)
//         {
//             free(*out);
//             *out = nullptr;
//         }
//     }
//     return returnStatus;
// }

// Helper method to convert Output tensors to float and write them
// out to files.
// iotensor::StatusCode iotensor::IOTensor::convertAndWriteOutputTensorInFloat(
//     Qnn_Tensor_t *output,
//     std::vector<std::string> outputPaths,
//     std::string fileName,
//     size_t outputBatchSize)
// {
//     if (nullptr == output)
//     {
//         QNN_ERROR("output is nullptr");
//         return StatusCode::FAILURE;
//     }

//     auto returnStatus = StatusCode::SUCCESS;
//     std::vector<size_t> dims;
//     fillDims(dims, QNN_TENSOR_GET_DIMENSIONS(output),
//     QNN_TENSOR_GET_RANK(output)); float *floatBuffer = nullptr; returnStatus
//     = convertToFloat(&floatBuffer, output); if (StatusCode::SUCCESS !=
//     returnStatus)
//     {
//         QNN_ERROR("failure in convertToFloat");
//         return StatusCode::FAILURE;
//     }
//     uint8_t *bufferToWrite = reinterpret_cast<uint8_t *>(floatBuffer);
//     if (datautil::StatusCode::SUCCESS !=
//         datautil::writeBatchDataToFile(
//             outputPaths, fileName, dims, QNN_DATATYPE_FLOAT_32,
//             bufferToWrite, outputBatchSize))
//     {
//         QNN_ERROR("failure in writeBatchDataToFile");
//         returnStatus = StatusCode::FAILURE;
//     }
//     if (nullptr != floatBuffer)
//     {
//         QNN_DEBUG("freeing floatBuffer");
//         free(floatBuffer);
//         floatBuffer = nullptr;
//     }
//     return returnStatus;
// }

// Helper method to write out output. There is no de-quantization here.
// Just write output as is to files.
bool IOTensor::writeOutputTensor(Qnn_Tensor_t * output,
    std::vector<std::string> outputPaths,
    std::string fileName,
    size_t outputBatchSize)
{
  if (nullptr == output) {
    std::cout << "output is nullptr" << std::endl;
    return false;
  }
  auto returnStatus = true;
  std::vector<size_t> dims;
  fillDims(dims, QNN_TENSOR_GET_DIMENSIONS(output), QNN_TENSOR_GET_RANK(output));
  uint8_t * bufferToWrite = reinterpret_cast<uint8_t *>(QNN_TENSOR_GET_CLIENT_BUF(output).data);
  // if (datautil::StatusCode::SUCCESS !=
  //     datautil::writeBatchDataToFile(outputPaths,
  //                                    fileName,
  //                                    dims,
  //                                    QNN_TENSOR_GET_DATA_TYPE(output),
  //                                    bufferToWrite,
  //                                    outputBatchSize))
  // {
  //     QNN_ERROR("failure in writeBatchDataToFile");
  //     returnStatus = StatusCode::FAILURE;
  // }
  return returnStatus;
}

// Write out all output tensors to files. If output_data_type is float,
// then all outputs will be raw floats regardless of what the model outputs.
// If the output_data_type is native, then output is written as produced by the
// model. Also, for native option, a json with quantization parameters is
// written out. If output_data_type is float_and_native, both above are done. If
// the output in the graph is float, then output_data_type has no effect.
// iotensor::StatusCode iotensor::IOTensor::writeOutputTensors(uint32_t
// graphIdx,
//                                                             size_t startIdx,
//                                                             char *graphName,
//                                                             Qnn_Tensor_t
//                                                             *outputs,
//                                                             uint32_t
//                                                             numOutputs,
//                                                             iotensor::OutputDataType
//                                                             outputDatatype,
//                                                             uint32_t
//                                                             graphsCount,
//                                                             std::string
//                                                             outputPath,
//                                                             size_t
//                                                             numInputFilesPopulated,
//                                                             size_t
//                                                             outputBatchSize)
// {
//     if (nullptr == outputs)
//     {
//         QNN_ERROR("Received nullptr");
//         return StatusCode::FAILURE;
//     }
//     if (graphsCount > 1)
//     {
//         if (nullptr != graphName && strlen(graphName) > 0)
//         {
//             outputPath += (pal::Path::getSeparator() +
//             std::string(graphName));
//         }
//         else
//         {
//             outputPath += (pal::Path::getSeparator() + std::string("Graph_")
//             + std::to_string(graphIdx));
//         }
//     }
//     auto returnStatus = StatusCode::SUCCESS;
//     std::vector<std::string> outputPaths;
//     for (size_t idx = 0; idx < numInputFilesPopulated; idx++)
//     {
//         std::string output = outputPath + (pal::Path::getSeparator() +
//         std::string("Result_") +
//                                            std::to_string(startIdx + idx));
//         outputPaths.push_back(output);
//     }
//     for (size_t outputIdx = 0; outputIdx < numOutputs; outputIdx++)
//     {
//         QNN_DEBUG("Writing output for outputIdx: %d", outputIdx);
//         std::string outputFilePrefix;
//         if (nullptr != QNN_TENSOR_GET_NAME(outputs[outputIdx]) &&
//             strlen(QNN_TENSOR_GET_NAME(outputs[outputIdx])) > 0)
//         {
//             outputFilePrefix =
//             std::string(QNN_TENSOR_GET_NAME(outputs[outputIdx]));
//         }
//         else
//         {
//             outputFilePrefix = std::string("Output_") +
//             std::to_string(outputIdx);
//         }
//         auto outputFile = outputFilePrefix + std::string(".raw");
//         auto outputFileNative = outputFilePrefix +
//         std::string("_native.raw"); if
//         (QNN_TENSOR_GET_DATA_TYPE(outputs[outputIdx]) ==
//         QNN_DATATYPE_FLOAT_32)
//         {
//             QNN_DEBUG("Writing in output->dataType ==
//             QNN_DATATYPE_FLOAT_32"); returnStatus =
//                 writeOutputTensor(&(outputs[outputIdx]), outputPaths,
//                 outputFile, outputBatchSize);
//         }
//         else if (outputDatatype == OutputDataType::FLOAT_ONLY)
//         {
//             QNN_DEBUG("Writing in output->dataType ==
//             OutputDataType::FLOAT_ONLY"); returnStatus =
//             convertAndWriteOutputTensorInFloat(
//                 &(outputs[outputIdx]), outputPaths, outputFile,
//                 outputBatchSize);
//         }
//         else if (outputDatatype == OutputDataType::NATIVE_ONLY)
//         {
//             QNN_DEBUG("Writing in output->dataType ==
//             OutputDataType::NATIVE_ONLY"); returnStatus =
//                 writeOutputTensor(&(outputs[outputIdx]), outputPaths,
//                 outputFileNative, outputBatchSize);
//         }
//         else if (outputDatatype == OutputDataType::FLOAT_AND_NATIVE)
//         {
//             QNN_DEBUG("Writing in output->dataType ==
//             OutputDataType::FLOAT_AND_NATIVE"); returnStatus =
//             convertAndWriteOutputTensorInFloat(
//                 &(outputs[outputIdx]), outputPaths, outputFile,
//                 outputBatchSize);
//             if (StatusCode::SUCCESS == returnStatus)
//             {
//                 returnStatus = writeOutputTensor(
//                     &(outputs[outputIdx]), outputPaths, outputFileNative,
//                     outputBatchSize);
//             }
//         }
//     }
//     return returnStatus;
// }
// #endif

// Helper method to allocate a buffer and copy data to it.
bool IOTensor::allocateAndCopyBuffer(uint8_t ** buffer, Qnn_Tensor_t * tensor)
{
  if (nullptr == tensor) {
    return false;
  }
  std::vector<size_t> dims;
  fillDims(dims, QNN_TENSOR_GET_DIMENSIONS(tensor), QNN_TENSOR_GET_RANK(tensor));
  size_t length;
  length = datautil::calculateLength(dims, QNN_TENSOR_GET_DATA_TYPE(tensor));
  if (length == 0)
    return false;
  if (true != allocateBuffer(buffer, dims, QNN_TENSOR_GET_DATA_TYPE(tensor))) {
    std::cout << "failure in allocateBuffer" << std::endl;
    return false;
  }
  datautil::StringOpMemscpy(*buffer, length * sizeof(uint8_t),
      QNN_TENSOR_GET_CLIENT_BUF(tensor).data, length * sizeof(uint8_t));
  return true;
}

void IOTensor::fillDims(std::vector<size_t> & dims, uint32_t * inDimensions, uint32_t rank)
{
  if (nullptr == inDimensions) {
    std::cout << "input dimensions is nullptr" << std::endl;
    return;
  }
  for (size_t r = 0; r < rank; r++) {
    dims.push_back(inDimensions[r]);
  }
}

// iotensor::OutputDataType iotensor::parseOutputDataType(std::string
// dataTypeString)
// {
//     std::transform(dataTypeString.begin(), dataTypeString.end(),
//     dataTypeString.begin(), ::tolower); OutputDataType parsedDataType =
//     OutputDataType::INVALID; if (dataTypeString == "float_only")
//     {
//         parsedDataType = OutputDataType::FLOAT_ONLY;
//     }
//     else if (dataTypeString == "native_only")
//     {
//         parsedDataType = OutputDataType::NATIVE_ONLY;
//     }
//     else if (dataTypeString == "float_and_native")
//     {
//         parsedDataType = OutputDataType::FLOAT_AND_NATIVE;
//     }
//     return parsedDataType;
// }

// iotensor::InputDataType iotensor::parseInputDataType(std::string
// dataTypeString)
// {
//     std::transform(dataTypeString.begin(), dataTypeString.end(),
//     dataTypeString.begin(), ::tolower); InputDataType parsedDataType =
//     InputDataType::INVALID; if (dataTypeString == "float")
//     {
//         parsedDataType = InputDataType::FLOAT;
//     }
//     else if (dataTypeString == "native")
//     {
//         parsedDataType = InputDataType::NATIVE;
//     }
//     return parsedDataType;
// }