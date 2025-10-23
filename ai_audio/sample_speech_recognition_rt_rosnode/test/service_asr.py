#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

# this is a example, I use whisper turbo model in service.
model_file = 'module/large-v3-turbo.pt'
 
# load Whisper
model = whisper.load_model(model_file, device="cuda")
 
# creat Flask application
app = Flask(__name__)
 
@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
 
    audio_file = request.files['audio']
    audio_data = np.frombuffer(audio_file.read(), dtype=np.float32)
 
    print('运行推理.')
    inference_start = timer()
 
    # 进行语音转录，只识别英文
    result = model.transcribe(audio_data, language='en')
 
    inference_end = timer() - inference_start
    print(f'Total elapsed time {inference_end:.3}s.')
 
    return jsonify({'text': result["text"], 'inference_time': inference_end}) # 将转录结果和推理时间以JSON格式返回。
 
if __name__ == '__main__':
    app.run(host='10.92.128.242', port=5000)
