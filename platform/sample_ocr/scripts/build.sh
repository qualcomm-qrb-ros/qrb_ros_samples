# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# This project is currently under active development
#!/bin/bash
THIS_SCRIPT=$(readlink -f ${BASH_SOURCE[0]})
scriptdir_ocr="$(dirname "${THIS_SCRIPT}")"

cd $scriptdir_ocr/..

export AMENT_PREFIX_PATH="${OECORE_TARGET_SYSROOT}/usr;${OECORE_NATIVE_SYSROOT}/usr"
export PYTHONPATH=${PYTHONPATH}:${OECORE_TARGET_SYSROOT}/usr/lib/python3.12/site-packages
export Python3_NumPy_INCLUDE_DIR=${OECORE_TARGET_SYSROOT}/usr/lib/python3.12/site-packages/numpy/core/include

colcon build --merge-install --cmake-args \
 -DPython3_NumPy_INCLUDE_DIR=${Python3_NumPy_INCLUDE_DIR} \
 -DPYTHON_SOABI=cpython-312-aarch64-linux-gnu \
 -DCMAKE_STAGING_PREFIX=$(pwd)/install \
 -DCMAKE_PREFIX_PATH=$(pwd)/install/share \
 -DBUILD_TESTING=OFF
