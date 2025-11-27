# MIT License

# Copyright (c) 2022 OpenAI

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------


import argparse
import numpy as np
from typing import List, Tuple

#from __future__ import annotations
import samplerate
#from scipy.signal import hann 
import torch

#audio parameters
# 20ms sample rate
SAMPLE_RATE = 16000
# Audio chunk length in seconds
CHUNK_LENGTH = 30

# Adopted from https://github.com/openai/whisper/blob/main/whisper/audio.py
def log_mel_spectrogram(
    mel_filter: np.ndarray,
    audio_np: np.ndarray,
    pad_to_length: int,
    n_fft: int,
    hop_length: int,
) -> np.ndarray:
    """
    Compute the log-Mel spectrogram of

    Parameters
    ----------
    audio_np: np.ndarray, shape = (*)

    pad_to_length: int
        Add zero samples to the right till this length. No op if
        len(audio) >= pad_to_length

    Returns
    -------
    np.ndarray, shape = (1, 80, n_frames)
        A Tensor that contains the Mel spectrogram. n_frames = 3000 for whisper
    """
    audio = torch.from_numpy(audio_np)
    assert isinstance(audio, torch.Tensor)

    if pad_to_length is not None:
        padding = pad_to_length - len(audio)
        if padding > 0:
            audio = torch.nn.functional.pad(audio, (0, padding))
    window = torch.hann_window(n_fft)
    stft = torch.stft(audio, n_fft, hop_length, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    mel_spec = torch.from_numpy(mel_filter) @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec.unsqueeze(0).detach().float().numpy()
    
    
def chunk_and_resample_audio(
    audio: np.ndarray,
    audio_sample_rate: int,
    model_sample_rate=SAMPLE_RATE,
    model_chunk_seconds=CHUNK_LENGTH,
) -> List[np.ndarray]:
    """
    Parameters
    ----------
    audio: str
        Raw audio numpy array of shape [# of samples]

    audio_sample_rate: int
        Sample rate of audio array, in samples / sec.

    model_sample_rate: int
        Sample rate (samples / sec) required to run Whisper. The audio file
        will be resampled to use this rate.

    model_chunk_seconds: int
        Split the audio in to N sequences of this many seconds.
        The final split may be shorter than this many seconds.

    Returns
    -------
    List of audio arrays, chunked into N arrays of model_chunk_seconds seconds.
    """
    if audio_sample_rate != model_sample_rate:
        audio = samplerate.resample(audio, model_sample_rate / audio_sample_rate)
        audio_sample_rate = model_sample_rate

    number_of_full_length_audio_chunks = (
        audio.shape[0] // audio_sample_rate // model_chunk_seconds
    )
    last_sample_in_full_length_audio_chunks = (
        audio_sample_rate * number_of_full_length_audio_chunks * model_chunk_seconds
    )

    if number_of_full_length_audio_chunks == 0:
        return [audio]

    return [
        *np.array_split(
            audio[:last_sample_in_full_length_audio_chunks],
            number_of_full_length_audio_chunks,
        ),
        audio[last_sample_in_full_length_audio_chunks:],
    ]



 

          
    
    


