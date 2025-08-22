#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from typing import List, Tuple
import numpy as np
import samplerate
import torch
import whisper
from scipy import special as scipy_special
import time
import os
from qai_hub_models.models._shared.whisper.model import (
    CHUNK_LENGTH,
    HOP_LENGTH,
    MEL_FILTER_PATH,
    N_FFT,
    N_MELS,
    SAMPLE_RATE,
    MEAN_DECODE_LEN,
)
from qai_hub_models.models._shared.whisper.app import (
    TOKEN_SOT,
    TOKEN_EOT,
    TOKEN_BLANK,
    TOKEN_NO_TIMESTAMP,
    TOKEN_TIMESTAMP_BEGIN,
    TOKEN_NO_SPEECH,
    NO_SPEECH_THR,
    NON_SPEECH_TOKENS,
    SAMPLE_BEGIN,
    apply_timestamp_rules,
    log_mel_spectrogram,
    chunk_and_resample_audio,
)

from tflite_load_model import (
    load_tf_whisper_encoder,
    load_tf_whisper_decoder,
    run_tf_whisper_encoder,
    run_tf_whisper_decoder,
)

class CustomWhisperApp:
    """
    WhisperApp runs Whisper encoder and decoder to transcribe audio
    represented as mel spectrogram. It support all model variants of
    OpenAI Whisper.
    """

    def __init__(self):
        start_time = time.time()

        # set env
        whisper_model_path = os.getenv("WHISPER_MODEL_PATH")

        file_name = "whisper_tiny_en-whisperencoder.tflite"
        # full_path = os.path.join(whisper_model_path, file_name)
        full_path = f"{whisper_model_path}/{file_name}"
        load_tf_whisper_encoder(full_path)

        file_name = "whisper_tiny_en-whisperdecoder.tflite"
        # full_path = os.path.join(whisper_model_path, file_name)
        full_path = f"{whisper_model_path}/{file_name}"
        load_tf_whisper_decoder(full_path)

        file_name = "mel_filters.npz"
        # full_path = os.path.join(whisper_model_path, file_name)
        full_path = f"{whisper_model_path}/{file_name}"
        data = np.load(full_path)
        self.mel_filter = data["mel_80"]  # .npz

        end_time = time.time()
        elapsed_time = end_time - start_time
        start_time = end_time
        # print("[load TF models time:", elapsed_time, "seconds]")

        # encoder = WhisperEncoderInf(whisper_model)
        # decoder = WhisperDecoderInf(whisper_model.decoder)
        self.num_decoder_blocks = 4  # len(decoder.blocks)
        self.attention_dim = 384  # decoder.attention_dim
        self.num_decoder_heads = 6  # decoder.num_heads
        self.mean_decode_len = MEAN_DECODE_LEN

        self.hop_length = HOP_LENGTH
        self.sample_rate = SAMPLE_RATE
        self.max_audio_seconds = CHUNK_LENGTH
        self.n_fft = N_FFT
        self.max_audio_samples = self.max_audio_seconds * self.sample_rate

    def transcript_encoding(
        self, audio: np.ndarray | str, audio_sample_rate: int | None = None
    ) -> str:
        start_time = time.time()

        if isinstance(audio, str):
            # requires ffmpeg to be installed on host machine.
            import audio2numpy as a2n
            audio, audio_sample_rate = a2n.audio_from_file(audio)
        else:
            assert audio_sample_rate is not None

        audio_chunk = chunk_and_resample_audio(audio, audio_sample_rate)[0]

        mel_input = log_mel_spectrogram(
            self.mel_filter,
            audio_chunk,
            self.max_audio_samples,
            self.n_fft,
            self.hop_length,
        )

        end_time = time.time()
        elapsed_time = end_time - start_time
        start_time = end_time
        print("[Mel time:", elapsed_time, "seconds]")

        k_cache_cross, v_cache_cross = run_tf_whisper_encoder(mel_input)

        end_time = time.time()
        elapsed_time = end_time - start_time
        start_time = end_time
        print("[Encoding time:", elapsed_time, "seconds]")

        transcription = self.transcript_decoding(k_cache_cross, v_cache_cross)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print("[Decoding time:", elapsed_time, "seconds]")
        return transcription

    def transcript_decoding(self, k_cache_cross, v_cache_cross) -> str:
        # Start decoding
        # coreml only takes float tensors
        x = np.array([[TOKEN_SOT]])
        decoded_tokens = [TOKEN_SOT]
        sample_len = self.mean_decode_len  # mean # of tokens to sample
        k_cache_self = np.zeros(
            (
                self.num_decoder_blocks,
                self.num_decoder_heads,
                self.attention_dim // self.num_decoder_heads,
                sample_len,
            )
        ).astype(np.float32)
        v_cache_self = np.zeros(
            (
                self.num_decoder_blocks,
                self.num_decoder_heads,
                sample_len,
                self.attention_dim // self.num_decoder_heads,
            )
        ).astype(np.float32)

        sum_logprobs = 0

        for i in range(sample_len):
            index = torch.zeros([1, 1], dtype=torch.int32)
            index[0, 0] = i
            decoder_out = run_tf_whisper_decoder(
                x.astype(np.int32),
                index,
                k_cache_cross,
                v_cache_cross,
                k_cache_self,
                v_cache_self,
            )
            # logit has shape (1, decoded_len, 51864)
            logits = decoder_out[0]
            k_cache_self = decoder_out[1]
            v_cache_self = decoder_out[2]
            # logit has shape (51864,)
            logits = logits[0, -1]  # consider only the last token

            # Filters
            # SuppressBlank
            if i == 0:
                logits[[TOKEN_EOT, TOKEN_BLANK]] = -np.inf
            # SuppressTokens
            logits[NON_SPEECH_TOKENS] = -np.inf

            logits, logprobs = apply_timestamp_rules(logits, decoded_tokens)

            if i == 0:
                # detect no_speech
                no_speech_prob = np.exp(logprobs[TOKEN_NO_SPEECH])
                if no_speech_prob > NO_SPEECH_THR:
                    break

            # temperature = 0
            next_token = np.argmax(logits)
            if next_token == TOKEN_EOT:
                break

            sum_logprobs += logprobs[next_token]
            x = np.array([[next_token]])
            decoded_tokens.append(int(next_token))

        tokenizer = whisper.decoding.get_tokenizer(
            multilingual=False, language="en", task="transcribe"
        )

        text = tokenizer.decode(decoded_tokens[1:])  # remove TOKEN_SOT
        return text.strip()
