# Fetch ONNX CLIP model
# wget https://huggingface.co/sayantan47/clip-vit-b32-onnx/resolve/main/onnx/model_q4.onnx
# infer from : https://huggingface.co/sayantan47/clip-vit-b32-onnx

# Fetch example pics with local path
# bash ./download_test_pic.sh


import onnxruntime as ort
from transformers import CLIPProcessor
from PIL import Image
import numpy as np

def build_example_image_embedding_dataset():

    # Need to levearge CLIP Preprocess Code, don't want to rebuild the wheel
    repo_id = "sayantan47/clip-vit-b32-onnx"
    processor = CLIPProcessor.from_pretrained(repo_id)

    # Load ONNX Runtime session
    session = ort.InferenceSession("model_q4.onnx", providers=["CPUExecutionProvider"])

    # Example input, use download_test_pic.sh first
    images = []
    for i in range(1, 11): 
        filename = f"img{i:02d}.jpg"  
        img = Image.open(filename)
        images.append(img)

    texts = ["query"]

    # Preprocess
    inputs = processor(text=texts, images=images, return_tensors="np", padding=True)

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
    print(img_embeds)
    return img_embeds