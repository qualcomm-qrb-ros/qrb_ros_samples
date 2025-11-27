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

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import numpy as np
import subprocess
import struct
import argparse
from typing import List, Tuple
from scipy import special as scipy_special  # type: ignore
from threading import Timer
#tokenizer 
import base64 
import tiktoken
import os
import decode
import audio_process
import shutil
import time
#for ros pkgs info
from ament_index_python.packages import get_package_share_directory

#mel-filter link: https://github.com/openai/whisper/blob/main/whisper/assets/mel_filters.npz

# The number of Mel features per audio context
N_MELS = 80
mel_filter: np.ndarray | None = None
#audio parameters
# 20ms sample rate
SAMPLE_RATE = 16000
# Audio chunk length in seconds
CHUNK_LENGTH = 30
# Length of the Hann window signal used when applying a FFT to the audio.
N_FFT = 400
# Number of audio samples between adjacent STFT columns when applying FFT to the audio.
HOP_LENGTH = 160

workdir = "/home/ubuntu/qirp-sdk/whisper/"
audio_data_file="/opt/qirp_ws/rec.wav"
audio_data_file_lock="/opt/qirp_ws/rec.wav.lock"
read_audio_time=2.0


class whisper_node(Node):
    def __init__(self):
        super().__init__('qrb_ros_whisper')
        self.publisher_ = self.create_publisher(String, 'whisper_result', 100)
        timer_period = read_audio_time  
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.count = 0
        self.recog_flag = False

    def timer_callback(self):
        msg = String()
        audio_sample_rate = None
        if os.path.exists(audio_data_file_lock) and os.path.exists(audio_data_file) and os.path.getsize(audio_data_file) and not self.recog_flag > 0:
            #print(f"starting speech_recognize audio {audio_data_file}")
            self.recog_flag = True
            result = start_speech_recognize(audio_data_file, audio_sample_rate)
            msg.data = result
            self.get_logger().info('publish asr data: "%s"' % msg.data)
            self.publisher_.publish(msg)
            os.remove(audio_data_file_lock)
            os.remove(audio_data_file)
            print(f"unlock audio input, starting recording...")
            self.recog_flag = False
        else:
            msg.data = "no audio detect "
        #self.get_logger().info('time update: "%s"' % msg.data)

def start_speech_recognize(audio: str, audio_sample_rate: int | None = None) -> str:  
    if isinstance(audio, str):
        import audio2numpy as a2n  # import here, as this requires ffmpeg to be installed on host machine
        audio, audio_sample_rate = a2n.audio_from_file(audio)
    else:
        assert audio_sample_rate is not None
    #print(f"audio is {audio}, audio_sample_rate is {audio_sample_rate}")    

    result = " ".join(
        transcribe_single_chunk(x)
        for x in audio_process.chunk_and_resample_audio(audio, audio_sample_rate)
    )
    return result
    
def transcribe_single_chunk(audio):
    """
    Transcribe an audio chunk to text.

    Parameters:

    audio: numpy array
        A numpy array of audio of shape (number of samples).
        The sample rate of this audio must be self.sample_rate.
        The maximum length of this audio must be self.max_audio_samples.

    Returns:

    - transcribed texts
    """
    #preprocess audio file to log-mel.raw
    #logmel value define      
    max_audio_samples  = CHUNK_LENGTH * SAMPLE_RATE
    mel_input = audio_process.log_mel_spectrogram(
        mel_filter, audio, max_audio_samples, N_FFT, HOP_LENGTH )        
    mel_input.tofile(workdir+"./input/mel_input.raw")
    
    # whisper encode audio file
    command = f"qtld-net-run --model {workdir}../model/whisper_tiny_en-whisperencoder.tflite --input {workdir}/input/encode_input.txt --output {workdir}/output_encode"
    subprocess.run(command, stdout=subprocess.DEVNULL, check=True, shell=True)
    
    #copy encode output to decode input
    #output_encode/Result_0/k_cache.raw -> input/k_cache_cross.raw 
    #output_encode/Result_0/v_cache.raw input/v_cache_cross.raw
    shutil.copy(workdir+"output_encode/Result_0/k_cache.raw", workdir+"input/k_cache_cross.raw")
    shutil.copy(workdir+"output_encode/Result_0/v_cache.raw", workdir+"input/v_cache_cross.raw")
    
    # whisper decode to text
    decode_text = decode.main();
    return decode_text.strip()

def read_audio_file():
    audio_sample_rate = None
    print(f"{read_audio_time}s timer update.")
    if os.path.exists(audio_data_file) and  os.path.getsize(audio_data_file) > 0:
        result = start_speech_recognize(audio_data_file, audio_sample_rate)
        #print(f"audio result is {result}")
        #os.remove(audio_data_file)
    # restart timer
    Timer(read_audio_time, read_audio_file).start()


def main(args=None):
    # make temp dir in /home/ubuntu/qirp-sdk/whisper/
    package_share_directory = get_package_share_directory('sample_speech_recognition')
    
    if not os.path.exists(workdir+"input" ):
        os.makedirs(workdir+"input" )
        shutil.copy(package_share_directory+"/decode_input.txt", workdir+"input/decode_input.txt")
        shutil.copy(package_share_directory+"/encode_input.txt"+"", workdir+"input/encode_input.txt")
    if not os.path.exists(workdir+"output_encode" ):
        os.makedirs(workdir+"output_encode" )
        os.chmod(workdir+"output_encode", 0o777)
    if not os.path.exists(workdir+"output_decode" ):
        os.makedirs(workdir+"output_decode" )
        os.chmod(workdir+"output_decode", 0o777)
    if not os.path.exists(workdir+"outputs" ):
        os.makedirs(workdir+"outputs" )
        os.chmod(workdir+"outputs", 0o777)
    
    #load mel filter 
    data = np.load(workdir+"../model/mel_filters.npz", allow_pickle=True)
    global mel_filter 
    mel_filter = data[f"mel_{N_MELS}"]
    print("***loading mel filter file***")

    # start timer
    #Timer(read_audio_time, read_audio_file).start()
    
    rclpy.init(args=args)
    node = whisper_node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    

if __name__ == '__main__':   
    main()
    
     
