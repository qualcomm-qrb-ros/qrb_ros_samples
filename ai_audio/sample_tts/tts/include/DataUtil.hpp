// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear

#pragma once

#include <string.h>

#include <map>
#include <queue>
#include <string>
#include <vector>

#include "QnnTypes.h"

namespace datautil
{

const size_t g_bitsPerByte = 8;

size_t getDataTypeSizeInBytes(Qnn_DataType_t dataType);

size_t calculateLength(std::vector<size_t> dims, Qnn_DataType_t dataType);

size_t calculateElementCount(std::vector<size_t> dims);

size_t getFileSize(std::string file_path);

size_t StringOpMemscpy(void * dst, size_t dstSize, const void * src, size_t copySize);

char * StringOpStrndup(const char * source, size_t maxlen);

bool readBinaryFromFile(std::string filePath, uint8_t * buffer, size_t bufferSize);

template <typename T_QuantType>
bool floatToTfN(T_QuantType * out, float * in, int32_t offset, float scale, size_t numElements);

template <typename T_QuantType>
bool tfNToFloat(float * out, T_QuantType * in, int32_t offset, float scale, size_t numElements);

template <typename T_QuantType>
bool castToFloat(float * out, T_QuantType * in, size_t numElements);

template <typename T_QuantType>
bool castFromFloat(T_QuantType * out, float * in, size_t numElements);

const std::map<Qnn_DataType_t, size_t> g_dataTypeToSize = {
  { QNN_DATATYPE_INT_8, 1 },
  { QNN_DATATYPE_INT_16, 2 },
  { QNN_DATATYPE_INT_32, 4 },
  { QNN_DATATYPE_INT_64, 8 },
  { QNN_DATATYPE_UINT_8, 1 },
  { QNN_DATATYPE_UINT_16, 2 },
  { QNN_DATATYPE_UINT_32, 4 },
  { QNN_DATATYPE_UINT_64, 8 },
  { QNN_DATATYPE_FLOAT_16, 2 },
  { QNN_DATATYPE_FLOAT_32, 4 },
  { QNN_DATATYPE_FLOAT_64, 8 },
  { QNN_DATATYPE_SFIXED_POINT_8, 1 },
  { QNN_DATATYPE_SFIXED_POINT_16, 2 },
  { QNN_DATATYPE_SFIXED_POINT_32, 4 },
  { QNN_DATATYPE_UFIXED_POINT_8, 1 },
  { QNN_DATATYPE_UFIXED_POINT_16, 2 },
  { QNN_DATATYPE_UFIXED_POINT_32, 4 },
  { QNN_DATATYPE_BOOL_8, 1 },
};
}  // namespace datautil
