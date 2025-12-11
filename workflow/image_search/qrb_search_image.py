#!/usr/bin/env python
# coding: utf-8
# Fetch ONNX CLIP model
# wget https://huggingface.co/sayantan47/clip-vit-b32-onnx/resolve/main/onnx/model_q4.onnx
# infer from : https://huggingface.co/sayantan47/clip-vit-b32-onnx

# Fetch example pics with local path
# bash ./download_test_pic.sh


import onnxruntime as ort
from transformers import CLIPProcessor
from PIL import Image
import numpy as np


# Need to levearge CLIP Preprocess Code, don't want to rebuild the wheel
repo_id = "sayantan47/clip-vit-b32-onnx"
processor = CLIPProcessor.from_pretrained(repo_id)

# Load ONNX Runtime session
session = ort.InferenceSession("model_q4.onnx", providers=["CPUExecutionProvider"])
# session = ort.InferenceSession("model_q4.onnx", providers=["CUDAExecutionProvider"])

# == Model Outputs Info ==
# 0 logits_per_image tensor(float) ['image_batch_size', 'txt_batch_size']    # used for easier checking logits
# 1 logits_per_text tensor(float) ['text_batch_size', 'image_batch_size']    # used for easier checking logits
# 2 text_embeds tensor(float) ['text_batch_size', 512]
# 3 image_embeds tensor(float) ['image_batch_size', 512]

def encode_query_text_embedding(q):
    t = []
    t.append(q)
    dummy_pic = Image.open("img01.jpg")
    inputs = processor(text=t, images=dummy_pic, return_tensors="np", padding=True)
    new_inputs = {}
    for k, v in inputs.items():
        if hasattr(v, 'dtype') and v.dtype == np.int32:
            new_inputs[k] = v.astype(np.int64)
        else:
            new_inputs[k] = v
    inputs = new_inputs
    
    outputs = session.run(None, inputs)
    text_embeds = outputs[2]
    return text_embeds

def build_example_image_embedding_dataset():

    # Example input, use download_test_pic.sh first
    images = []
    for i in range(1, 11): 
        filename = f"img{i:02d}.jpg"  
        img = Image.open(filename)
        images.append(img)

    texts = ["query"]

    # Preprocess
    inputs = processor(text=texts, images=images, return_tensors="np", padding=True)
    
    # Rearrange Inputs as format
    new_inputs = {}
    for k, v in inputs.items():
        if hasattr(v, 'dtype') and v.dtype == np.int32:
            new_inputs[k] = v.astype(np.int64)
        else:
            new_inputs[k] = v
    inputs = new_inputs

    # Run inference
    outputs = session.run(None, inputs)

    img_embeds = outputs[3]
    
    img_names = [f"img{i:02d}.jpg" for i in range(1, 11)]

    img_dataset_list = list(zip(img_names, img_embeds))

    return img_dataset_list

def _search_image(q, img_dataset_list, k):
    eps = 1e-12
    q = ( q / (np.linalg.norm(q) + eps)).flatten()
    
    # add dot product score for sorting
    triplets = []
    scores = []
    for name, emb in img_dataset_list:
        e = emb / (np.linalg.norm(emb) + eps)
        s = float(np.dot(e, q))
        scores.append(s)
        triplets.append((name, emb, s))
    
    # change scores into probs, use Temperature to control 
    scores = np.array(scores)
    probs = np.exp(scores / T) / np.sum(np.exp(scores / T))
    
    # update triplets
    triplets = [(name, emb, s, prob) for (name, emb, s), prob in zip(triplets, probs)]

    # rank with probs
    triplets.sort(key=lambda x: x[3], reverse=True)
    
    k = max(1, min(k, len(triplets)))
    return [(triplets[i][0], triplets[i][3]) for i in range(k)]


def search_image(q, img_dataset_list, k):
    q = q.flatten()
    
    # add dot product score for sorting
    triplets = []
    scores = []
    for name, emb in img_dataset_list:
        e = emb
        s = float(np.dot(e, q))
        scores.append(s)
        triplets.append((name, emb, s))

    # change scores into probs
    scores = np.array(scores)
    probs = np.exp(scores) / np.sum(np.exp(scores)) 
    
    # update triplets
    triplets = [(name, emb, s, prob) for (name, emb, s), prob in zip(triplets, probs)]

    # rank with probs
    triplets.sort(key=lambda x: x[3], reverse=True)
    
    k = max(1, min(k, len(triplets)))
    return [(triplets[i][0], triplets[i][3]) for i in range(k)]


def main():
    img_dataset_list = build_example_image_embedding_dataset()
    query = "a black dog"
    query_emb = encode_query_text_embedding(query)
    top3 = search_image(query_emb, img_dataset_list, 3)
    print(top3)

main()