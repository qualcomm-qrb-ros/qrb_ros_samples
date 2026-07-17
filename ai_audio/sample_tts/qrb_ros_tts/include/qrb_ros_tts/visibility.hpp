// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear

#ifndef QRB_ROS_TTS__VISIBILITY_HPP_
#define QRB_ROS_TTS__VISIBILITY_HPP_

#define QRB_ROS_TTS_EXPORT __attribute__((visibility("default")))
#define QRB_ROS_TTS_IMPORT __attribute__((visibility("default")))

#if defined(QRB_ROS_TTS_BUILDING_DLL)
#define QRB_ROS_TTS_PUBLIC QRB_ROS_TTS_EXPORT
#else
#define QRB_ROS_TTS_PUBLIC QRB_ROS_TTS_IMPORT
#endif

#define QRB_ROS_TTS_LOCAL __attribute__((visibility("hidden")))

#endif  // QRB_ROS_TTS__VISIBILITY_HPP_
