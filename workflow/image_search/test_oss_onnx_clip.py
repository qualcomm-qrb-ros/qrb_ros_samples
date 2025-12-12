# Fetch ONNX CLIP model
# wget https://huggingface.co/sayantan47/clip-vit-b32-onnx/resolve/main/onnx/model_q4.onnx
# https://huggingface.co/sayantan47/clip-vit-b32-onnx

# Fetch example pics
# wget https://cdn.pixabay.com/photo/2017/02/20/18/03/lion-2083492_1280.jpg 1.jpg -O cat.jpg


import onnxruntime as ort
from transformers import CLIPProcessor
from PIL import Image
import numpy as np

# Need to levearge CLIP Preprocess Code, don't want to rebuild the wheel
repo_id = "sayantan47/clip-vit-b32-onnx"
processor = CLIPProcessor.from_pretrained(repo_id)

# Load ONNX Runtime session
session = ort.InferenceSession("model_q4.onnx", providers=["CPUExecutionProvider"])

# Example input
image = Image.open("img01.jpg")
texts = ["a dog", "a cat", "a dragon", "a human", "a bat"]

# Preprocess
inputs = processor(text=texts, images=image, return_tensors="np", padding=True)

new_inputs = {}
for k, v in inputs.items():
    if hasattr(v, 'dtype') and v.dtype == np.int32:
        new_inputs[k] = v.astype(np.int64)
    else:
        new_inputs[k] = v
inputs = new_inputs

# == Model Outputs Info ==
# 0 logits_per_image tensor(float) ['image_batch_size', 'txt_batch_size']    # used for easier checking logits
# 1 logits_per_text tensor(float) ['text_batch_size', 'image_batch_size']    # used for easier checking logits
# 2 text_embeds tensor(float) ['text_batch_size', 512]
# 3 image_embeds tensor(float) ['image_batch_size', 512]

# Run inference
outputs = session.run(None, inputs)

img_embeds = outputs[3] / np.linalg.norm(outputs[3], axis=1, keepdims=True)
txt_embeds = outputs[2] / np.linalg.norm(outputs[2], axis=1, keepdims=True)

logits_per_image = outputs[0]
probs = np.exp(logits_per_image) / np.exp(logits_per_image).sum(-1, keepdims=True)
print("Probabilities:", probs)

