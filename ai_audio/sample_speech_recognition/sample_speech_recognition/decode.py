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

import numpy as np
import subprocess
import struct
from typing import List, Tuple
from scipy import special as scipy_special  # type: ignore


#tokenizer 
import base64 
import tiktoken
import os

workdir = "/home/ubuntu/qirp-sdk/whisper/"

TOKEN_SOT = 50257  # Start of transcript
TOKEN_EOT = 50256  # end of transcript
TOKEN_BLANK = 220  # " "
TOKEN_NO_TIMESTAMP = 50362
TOKEN_TIMESTAMP_BEGIN = 50363
TOKEN_NO_SPEECH = 50361

# Above this prob we deem there's no speech in the audio
NO_SPEECH_THR = 0.6

SAMPLE_BEGIN = 1  # first token is TOKEN_SOT

# https://github.com/openai/whisper/blob/v20230314/whisper/decoding.py#L600
NON_SPEECH_TOKENS = [
    1,
    2,
    7,
    8,
    9,
    10,
    14,
    25,
    26,
    27,
    28,
    29,
    31,
    58,
    59,
    60,
    61,
    62,
    63,
    90,
    91,
    92,
    93,
    357,
    366,
    438,
    532,
    685,
    705,
    796,
    930,
    1058,
    1220,
    1267,
    1279,
    1303,
    1343,
    1377,
    1391,
    1635,
    1782,
    1875,
    2162,
    2361,
    2488,
    3467,
    4008,
    4211,
    4600,
    4808,
    5299,
    5855,
    6329,
    7203,
    9609,
    9959,
    10563,
    10786,
    11420,
    11709,
    11907,
    13163,
    13697,
    13700,
    14808,
    15306,
    16410,
    16791,
    17992,
    19203,
    19510,
    20724,
    22305,
    22935,
    27007,
    30109,
    30420,
    33409,
    34949,
    40283,
    40493,
    40549,
    47282,
    49146,
    50257,
    50357,
    50358,
    50359,
    50360,
    50361,
]

# https://github.com/openai/whisper/blob/v20230314/whisper/decoding.py#L545
precision = 0.02  # in second
max_initial_timestamp = 1.0  # in second
max_initial_timestamp_index = int(max_initial_timestamp / precision)


def get_tokenizer(name):
    vocab_path = name
    #print(f"vocab_path is {vocab_path}")
    ranks = {
        base64.b64decode(token): int(rank)
        for token, rank in (line.split() for line in open(vocab_path) if line)
    }
    n_vocab = len(ranks)
    special_tokens = {}

    specials = [
        "<|endoftext|>",
        "<|startoftranscript|>",
         "<|translate|>",
        "<|transcribe|>",
        "<|startoflm|>",
        "<|startofprev|>",
        "<|nospeech|>",
        "<|notimestamps|>",
        *[f"<|{i * 0.02:.2f}|>" for i in range(1501)],
    ]

    for token in specials:
        special_tokens[token] = n_vocab
        n_vocab += 1

    return tiktoken.Encoding(
        name=os.path.basename(vocab_path),        
        explicit_n_vocab=n_vocab,
        pat_str=r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
        special_tokens=special_tokens,
        mergeable_ranks=ranks,
    )
    #    special_tokens=special_tokens,
    
def apply_timestamp_rules(
    logits: np.ndarray, tokens: List[int]
) -> Tuple[np.ndarray, float]:
    """
    When predicting timestamps, there are a few post processing rules /
    heuristics to ensure well-formed timestamps. See in-line comments for details

    Args:
    - logits: of shape (51864,)

    Returns:

    - modified logits
    - log probability of modified logits (log(softmax(logits)))
    """
    # Require producing timestamp
    logits[TOKEN_NO_TIMESTAMP] = -np.inf

    # timestamps have to appear in pairs, except directly before EOT
    seq = tokens[SAMPLE_BEGIN:]
    last_was_timestamp = len(seq) >= 1 and seq[-1] >= TOKEN_TIMESTAMP_BEGIN
    penultimate_was_timestamp = len(seq) < 2 or seq[-2] >= TOKEN_TIMESTAMP_BEGIN
    if last_was_timestamp:
        if penultimate_was_timestamp:  # has to be non-timestamp
            logits[TOKEN_TIMESTAMP_BEGIN:] = -np.inf
        else:  # cannot be normal text tokens
            logits[:TOKEN_EOT] = -np.inf

    timestamps = [t for t in tokens if t >= TOKEN_TIMESTAMP_BEGIN]
    if len(timestamps) > 0:
        # timestamps shouldn't decrease; forbid timestamp tokens smaller than the last
        # also force each segment to have a nonzero length, to   prevent infinite looping
        if last_was_timestamp and not penultimate_was_timestamp:
            timestamp_last = timestamps[-1]
        else:
            timestamp_last = timestamps[-1] + 1
        logits[TOKEN_TIMESTAMP_BEGIN:timestamp_last] = -np.inf

    if len(tokens) == SAMPLE_BEGIN:
        # suppress generating non-timestamp tokens at the beginning
        logits[:TOKEN_TIMESTAMP_BEGIN] = -np.inf

        # apply the `max_initial_timestamp` option
        last_allowed = TOKEN_TIMESTAMP_BEGIN + max_initial_timestamp_index
        logits[(last_allowed + 1) :] = -np.inf

    # if sum of probability over timestamps is above any other token, sample timestamp
    logprobs = scipy_special.log_softmax(logits)
    timestamp_logprob = scipy_special.logsumexp(logprobs[TOKEN_TIMESTAMP_BEGIN:])
    max_text_token_logprob = logprobs[:TOKEN_TIMESTAMP_BEGIN].max()
    if timestamp_logprob > max_text_token_logprob:
        # Mask out all but timestamp tokens
        logits[:TOKEN_TIMESTAMP_BEGIN] = -np.inf

    return logits, logprobs
    
    
def main(args=None):
	# read output file
    #K_cache_cross = open("output/Result_0/k_cache.raw", 'rb')
    #V_cache_cross = open("output/Result_0/v_cache.raw", 'rb')
    
    #print(raw_data[:425])
    #max_index = np.argmax(raw_data)
    
    #Start decoding
    #coreml only takes float tensors
    x = np.array([[TOKEN_SOT]])
    decoded_tokens = [TOKEN_SOT]
    sample_len = 224  # mean # of tokens to sample
    num_decoder_blocks = 4
    num_decoder_heads = 6
    attention_dim = 384
    k_cache_self = np.zeros(
        (
            num_decoder_blocks,
            num_decoder_heads,
            attention_dim // num_decoder_heads,
            sample_len,
        )
    ).astype(np.float32)
    v_cache_self = np.zeros(
        (
            num_decoder_blocks,
            num_decoder_heads,
            sample_len,
            attention_dim // num_decoder_heads,
        )
    ).astype(np.float32)

    sum_logprobs = 0   
    
    j=0
    
    for i in range(sample_len):
        # Using i to index inside the decoder model hurts the
        # the model performance.
        # index - used to get positional embedding correctly.
                
        with open(workdir+'input/x.raw', 'wb') as f: f.write(struct.pack('i', x[0,0]))
        with open(workdir+'input/index.raw', 'wb') as f: f.write(struct.pack('i', i))
        k_cache_self.tofile(workdir+"input/k_cache_self.raw")
        v_cache_self.tofile(workdir+"input/v_cache_self.raw")
    
        command=f"qtld-net-run --model {workdir}../model/whisper_tiny_en-whisperdecoder.tflite --input {workdir}/input/decode_input.txt --output {workdir}outputs"
        subprocess.run(command, stdout=subprocess.DEVNULL, check=True, shell=True)
        #subprocess.run(["qtld-net-run --model workdir/], stdout=subprocess.DEVNULL, check=True, shell=True)
        
        logits_file = open(workdir+"outputs/Result_0/logits.raw", 'rb')
        logits = np.fromfile(logits_file, dtype=np.float32)
        
        k_cache_self_file = open(workdir+"outputs/Result_0/k_cache.raw", 'rb')
        k_cache_self = np.fromfile(k_cache_self_file, dtype=np.float32)
        
        v_cache_self_file = open(workdir+"outputs/Result_0/v_cache.raw", 'rb')
        v_cache_self = np.fromfile(v_cache_self_file, dtype=np.float32)
    
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
        #print(f"next decode token is {next_token}")
        
    tokenizer = get_tokenizer(workdir+"/../model/gpt2.tiktoken")
    
    try:
        text = tokenizer.decode(decoded_tokens[1:])  # remove TOKEN_SOT
    except Exception as e:
        #print(f"An error occurred: {e}")
        text = "No sound detected"  # 或者你想要的其他默认值
    return text.strip()
    
    #text = tokenizer.decode(decoded_tokens[1:])  # remove TOKEN_SOT
    # print(f"decode text is {text}")         
