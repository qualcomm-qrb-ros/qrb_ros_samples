#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import time
import numpy as np
import sounddevice as sd
from timeit import default_timer as timer
import threading
import requests
from collections import deque
import wave
import queue

from whisper_transcription import CustomWhisperApp
from qai_hub_models.models._shared.whisper.app import WhisperApp


class AudioProcessor:
    def __init__(
        self,
        server_url,
        audio_energy_threshold,
        short_term_window,
        moving_average_window,
        available_window,
        localtiny,
        channels=1,
    ):
        self.server_url = server_url
        self.sample_rate = 16000
        self.channels = channels

        self.enable_asr = False
        self.status_record = 0

        self.buffer_audio = np.array([], dtype=np.float32)
        self.buffer_audio_energy = deque(maxlen=moving_average_window)
        self.queue_audio = queue.Queue()
        self.audio_event = threading.Event()

        # Store Parameters as Instance Variables
        self.audio_energy_threshold = audio_energy_threshold
        self.short_term_window = short_term_window
        self.moving_average_window = int(moving_average_window)
        self.available_window = available_window
        self.localtiny = localtiny

        if self.localtiny:
            print("\nLoading local model.")
            self.app = CustomWhisperApp()

        device_index = self.search_usb_audio()

        # Clear Buffer Energy Queue
        self.buffer_audio_energy.clear()

        audio_thread = threading.Thread(
            target=self.audio_stream_thread, args=(device_index,), daemon=True
        )
        audio_thread.start()

    def set_parameter_enable_asr(self, data):
        self.enable_asr = data

    def get_parameter_enable_asr(self):
        return self.enable_asr

    def calculate_short_term_energy(self, audio_data, short_term_window):
        window_size = int(self.sample_rate * short_term_window)
        start = max(len(audio_data) - window_size, 0)
        end = len(audio_data)
        window_samples = audio_data[start:end]
        energy_now = np.sum(window_samples**2) / len(window_samples)
        self.buffer_audio_energy.append(energy_now)
        moving_average_energy = np.mean(self.buffer_audio_energy)
        return moving_average_energy

    def record_audio_callback(self, indata, frames, time, status):
        if self.enable_asr:
            self.buffer_audio = np.append(self.buffer_audio, indata[:, 0])
            if len(self.buffer_audio) >= self.sample_rate * self.short_term_window:
                short_term_energies = self.calculate_short_term_energy(
                    self.buffer_audio, self.short_term_window
                )
                energy_now = np.max(short_term_energies) * 10000

                if self.status_record == 0:
                    if energy_now >= self.audio_energy_threshold:
                        self.status_record = 1
                    else:
                        self.buffer_audio = np.array([], dtype=np.float32)
                        return

                if self.status_record == 1:
                    if energy_now < self.audio_energy_threshold:
                        if (
                            len(self.buffer_audio)
                            > self.sample_rate * self.available_window
                        ):
                            print(f"EnableAsr: {self.get_parameter_enable_asr()}")
                            self.queue_audio.put(self.buffer_audio.copy())
                            self.audio_event.set()
                        self.buffer_audio = np.array([], dtype=np.float32)
                        self.buffer_audio_energy.clear()
                        self.status_record = 0
                    else:
                        return
        else:
            self.buffer_audio = np.array([], dtype=np.float32)
            self.buffer_audio_energy.clear()
            self.status_record = 0
            return

    def audio_stream_thread(self, device_index):
        with sd.InputStream(
            device=device_index,
            channels=self.channels,
            samplerate=self.sample_rate,
            callback=self.record_audio_callback,
        ):
            print(f"Starting real-time detection and recognition...")
            try:
                while True:
                    sd.sleep(1000)
            except KeyboardInterrupt:
                print("\nReal-time speech detection ended.")

    def search_usb_audio(self):
        devices = sd.query_devices()
        device_index = 0
        for i, device in enumerate(devices):
            if "2K USB Camera: Audio" in device["name"]:
                device_index = i
                print(f"{i}: {device['name']}")
                break
        return device_index

    def send_audio_to_server(self, audio_data):
        try:
            files = {"audio": audio_data.tobytes()}
            response = requests.post(self.server_url, files=files)
            if response.status_code == 200:
                result = response.json()
            else:
                print(f"Request failed: {response.status_code}")
                result = {"text": ""}
        except Exception as e:
            print(f"An error occurred while sending the request: {e}")
            result = {"text": ""}

        return result["text"]

    def transcription_whisper(self):
        if not self.queue_audio.empty():
            audio_data = np.frombuffer(self.queue_audio.get(), dtype=np.float32)
            start_time = time.time()
            print(f"Performing speech recognition")
            try:
                if self.localtiny:
                    print("local tiny module")
                    transcription = self.app.transcript_encoding(
                        audio_data, self.sample_rate
                    )
                else:
                    print("service module")
                    transcription = self.send_audio_to_server(audio_data)
            except Exception as e:
                print(f"An error occurred during speech recognition: {e}")
                transcription = ""
            elapsed_time = time.time() - start_time
            print("[Total elapsed time:", elapsed_time, "seconds]")
            return transcription
        else:
            return None
