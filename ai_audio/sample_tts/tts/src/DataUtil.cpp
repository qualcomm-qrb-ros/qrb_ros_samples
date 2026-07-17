// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
#include "DataUtil.hpp"

#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <queue>

using namespace datautil;

size_t datautil::getDataTypeSizeInBytes(Qnn_DataType_t dataType)
{
  if (g_dataTypeToSize.find(dataType) == g_dataTypeToSize.end()) {
    std::cout << "Invalid qnn data type provided" << std::endl;
    return 0;
  }
  return g_dataTypeToSize.find(dataType)->second;
}

size_t datautil::calculateElementCount(std::vector<size_t> dims)
{
  if (dims.size() == 0) {
    return 0;
  }
  return std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<size_t>());
}

size_t datautil::calculateLength(std::vector<size_t> dims, Qnn_DataType_t dataType)
{
  if (dims.size() == 0) {
    std::cout << "dims.size() is zero" << std::endl;
    return 0;
  }
  size_t length = getDataTypeSizeInBytes(dataType);
  length *= calculateElementCount(dims);
  return length;
}

size_t datautil::StringOpMemscpy(void * dst, size_t dstSize, const void * src, size_t copySize)
{
  if (!dst || !src || !dstSize || !copySize)
    return 0;
  size_t minSize = dstSize < copySize ? dstSize : copySize;
  memcpy(dst, src, minSize);
  return minSize;
}

char * datautil::StringOpStrndup(const char * source, size_t maxlen)
{
  size_t length = strnlen(source, maxlen);
  char * destination = (char *)malloc((length + 1) * sizeof(char));
  if (destination == nullptr)
    return nullptr;
  strncpy(destination, source, length);
  destination[length] = '\0';
  return destination;
}

size_t datautil::getFileSize(std::string file_path)
{
  std::ifstream in(file_path, std::ifstream::binary);
  if (!in) {
    printf("Failed to open input file: %s", file_path.c_str());
    return 0;
  }
  in.seekg(0, in.end);
  const size_t length = in.tellg();
  in.seekg(0, in.beg);
  return length;
}

bool datautil::readBinaryFromFile(std::string filePath, uint8_t * buffer, size_t bufferSize)
{
  if (nullptr == buffer) {
    std::cout << "buffer is nullptr" << std::endl;
    return false;
  }
  std::ifstream in(filePath, std::ifstream::binary);
  if (!in) {
    printf("Failed to open input file: %s", filePath.c_str());
    return false;
  }
  if (!in.read(reinterpret_cast<char *>(buffer), bufferSize)) {
    printf("Failed to read the contents of: %s", filePath.c_str());
    return false;
  }
  return true;
}

template <typename T_QuantType>
bool datautil::floatToTfN(T_QuantType * out,
    float * in,
    int32_t offset,
    float scale,
    size_t numElements)
{
  static_assert(std::is_unsigned<T_QuantType>::value, "floatToTfN supports unsigned only!");
  if (nullptr == out || nullptr == in) {
    std::cout << "Received a nullptr" << std::endl;
    return false;
  }
  size_t dataTypeSizeInBytes = sizeof(T_QuantType);
  size_t bitWidth = dataTypeSizeInBytes * g_bitsPerByte;
  double trueBitWidthMax = pow(2, bitWidth) - 1;
  double encodingMin = offset * scale;
  double encodingMax = (trueBitWidthMax + offset) * scale;
  double encodingRange = encodingMax - encodingMin;
  for (size_t i = 0; i < numElements; ++i) {
    int quantizedValue = round(trueBitWidthMax * (in[i] - encodingMin) / encodingRange);
    if (quantizedValue < 0)
      quantizedValue = 0;
    else if (quantizedValue > (int)trueBitWidthMax)
      quantizedValue = (int)trueBitWidthMax;
    out[i] = static_cast<T_QuantType>(quantizedValue);
  }
  return true;
}

template bool datautil::floatToTfN<uint8_t>(uint8_t * out,
    float * in,
    int32_t offset,
    float scale,
    size_t numElements);
template bool datautil::floatToTfN<uint16_t>(uint16_t * out,
    float * in,
    int32_t offset,
    float scale,
    size_t numElements);

template <typename T_QuantType>
bool datautil::tfNToFloat(float * out,
    T_QuantType * in,
    int32_t offset,
    float scale,
    size_t numElements)
{
  static_assert(std::is_unsigned<T_QuantType>::value, "tfNToFloat supports unsigned only!");
  if (nullptr == out || nullptr == in) {
    std::cout << "Received a nullptr" << std::endl;
    return false;
  }
  for (size_t i = 0; i < numElements; i++) {
    double quantizedValue = static_cast<double>(in[i]);
    double offsetDouble = static_cast<double>(offset);
    out[i] = static_cast<double>((quantizedValue + offsetDouble) * scale);
  }
  return true;
}

template bool datautil::tfNToFloat<uint8_t>(float * out,
    uint8_t * in,
    int32_t offset,
    float scale,
    size_t numElements);
template bool datautil::tfNToFloat<uint16_t>(float * out,
    uint16_t * in,
    int32_t offset,
    float scale,
    size_t numElements);

template <typename T_QuantType>
bool datautil::castToFloat(float * out, T_QuantType * in, size_t numElements)
{
  if (nullptr == out || nullptr == in) {
    std::cout << "Received a nullptr" << std::endl;
    return false;
  }
  for (size_t i = 0; i < numElements; i++)
    out[i] = static_cast<float>(in[i]);
  return true;
}

template bool datautil::castToFloat<uint8_t>(float * out, uint8_t * in, size_t numElements);
template bool datautil::castToFloat<uint16_t>(float * out, uint16_t * in, size_t numElements);
template bool datautil::castToFloat<uint32_t>(float * out, uint32_t * in, size_t numElements);
template bool datautil::castToFloat<uint64_t>(float * out, uint64_t * in, size_t numElements);
template bool datautil::castToFloat<int8_t>(float * out, int8_t * in, size_t numElements);
template bool datautil::castToFloat<int16_t>(float * out, int16_t * in, size_t numElements);
template bool datautil::castToFloat<int32_t>(float * out, int32_t * in, size_t numElements);
template bool datautil::castToFloat<int64_t>(float * out, int64_t * in, size_t numElements);

template <typename T_QuantType>
bool datautil::castFromFloat(T_QuantType * out, float * in, size_t numElements)
{
  if (nullptr == out || nullptr == in) {
    std::cout << "Received a nullptr" << std::endl;
    return false;
  }
  for (size_t i = 0; i < numElements; i++)
    out[i] = static_cast<T_QuantType>(in[i]);
  return true;
}

template bool datautil::castFromFloat<uint8_t>(uint8_t * out, float * in, size_t numElements);
template bool datautil::castFromFloat<uint16_t>(uint16_t * out, float * in, size_t numElements);
template bool datautil::castFromFloat<uint32_t>(uint32_t * out, float * in, size_t numElements);
template bool datautil::castFromFloat<uint64_t>(uint64_t * out, float * in, size_t numElements);
template bool datautil::castFromFloat<int8_t>(int8_t * out, float * in, size_t numElements);
template bool datautil::castFromFloat<int16_t>(int16_t * out, float * in, size_t numElements);
template bool datautil::castFromFloat<int32_t>(int32_t * out, float * in, size_t numElements);
template bool datautil::castFromFloat<int64_t>(int64_t * out, float * in, size_t numElements);
