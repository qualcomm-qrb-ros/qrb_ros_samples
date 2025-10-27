#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import numpy as np
import tensorflow as tf

encoder_interpreter = None

def load_tf_whisper_encoder(tf_model):
    global encoder_interpreter
    encoder_interpreter = tf.lite.Interpreter(model_path=tf_model)
    encoder_interpreter.allocate_tensors()
    # print("load mode:" + tf_model)

def run_tf_whisper_encoder(mel_input: np.ndarray) -> np.ndarray:

    input_details = encoder_interpreter.get_input_details()
    encoder_interpreter.set_tensor(input_details[0]["index"], mel_input)
    # Run Inference
    encoder_interpreter.invoke()

    # Get the Indices of Input and Output Tensors
    output_details = encoder_interpreter.get_output_details()

    output_data = [
        encoder_interpreter.get_tensor(output_details[i]["index"])
        for i in range(len(output_details))
    ]

    return output_data

decoder_interpreter = None

def load_tf_whisper_decoder(tf_model):
    global decoder_interpreter
    decoder_interpreter = tf.lite.Interpreter(model_path=tf_model)
    decoder_interpreter.allocate_tensors()
    # print("load mode:" + tf_model)

def run_tf_whisper_decoder(*args) -> np.ndarray:

    # Get the Indices of Input and Output Tensors
    input_details = decoder_interpreter.get_input_details()
    i = 0
    for arg in args:
        new_arg = arg
        decoder_interpreter.set_tensor(input_details[i]["index"], new_arg)
        i += 1

    # Run Inference
    decoder_interpreter.invoke()

    # Get the Indices of Input and Output Tensors
    output_details = decoder_interpreter.get_output_details()
    output_data = [
        decoder_interpreter.get_tensor(output_details[i]["index"])
        for i in range(len(output_details))
    ]

    # print("load mode:" + tf_model);
    return output_data
