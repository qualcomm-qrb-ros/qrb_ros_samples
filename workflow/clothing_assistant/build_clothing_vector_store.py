import requests
import json
import sys
import os
import time
import argparse
import numpy as np
from PIL import Image, ImageOps
import onnxruntime as ort



# Fetch ONNX CLIP model
# https://github.com/Lednik7/CLIP-ONNX/tree/main
# wget https://clip-as-service.s3.us-east-2.amazonaws.com/models/onnx/ViT-B-32/visual.onnx
# wget https://clip-as-service.s3.us-east-2.amazonaws.com/models/onnx/ViT-B-32/textual.onnx


# normalized input image as CLIP needed
# https://github.com/openai/CLIP/blob/dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1/clip/clip.py#L85
# need to keep format as float32 to adapt model input requests
mean = np.array([0.48145466, 0.4578275, 0.40821073]).astype(np.float32)
std = np.array([0.26862954, 0.26130258, 0.27577711]).astype(np.float32)

def prepare_clip_text_tokenizer():

    # Import the downloaded script
    # from : https://github.com/openai/CLIP/blob/main/clip/clip.py#L14
    # from : https://github.com/openai/CLIP/blob/main/clip/clip.py#L28
    # from : https://github.com/openai/CLIP/blob/main/clip/clip.py#L228
    # from : https://github.com/openai/CLIP/blob/main/clip/clip.py#L229
    # from : https://github.com/openai/CLIP/blob/main/clip/clip.py#L230
    
    # URL of the script
    url = "https://raw.githubusercontent.com/openai/CLIP/dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1/clip/simple_tokenizer.py"

    # Download the script
    response = requests.get(url)
    
    with open("simple_tokenizer.py", "wb") as file:
        file.write(response.content)

    # URL of the bpe dictionary
    url = "https://github.com/openai/CLIP/raw/refs/heads/main/clip/bpe_simple_vocab_16e6.txt.gz"

    # Download bpe dictionary
    response = requests.get(url)
    
    with open("bpe_simple_vocab_16e6.txt.gz", "wb") as file:
        file.write(response.content)



# Load ONNX Runtime session
image_sess = ort.InferenceSession("visual.onnx", providers=["CPUExecutionProvider"])
text_sess = ort.InferenceSession("textual.onnx", providers=["CPUExecutionProvider"])

print()
print(" ======== CLIP as service Textual ONNX ======== ")
for inp in text_sess.get_inputs():
    print("name :", inp.name)
    print("shape:", inp.shape)
    print("dtype:", inp.type)
print()
print(" ======== CLIP as service Visual ONNX ======== ")
for inp in image_sess.get_inputs():
    print("name :", inp.name)
    print("shape:", inp.shape)
    print("dtype:", inp.type)
print()

def letterbox_to_square(img, target=224, fill=0):
    """
    Letterbox to target x target:
    - Resize so the longer side becomes `target` (keep aspect ratio)
    - Pad evenly to make a square
    - fill: padding color (0=black, 128=gray, 255=white for L mode; for RGB use tuple)
    """
    w, h = img.size

    # Resize: long side -> target
    scale = target / max(w, h)
    nw = int(round(w * scale))
    nh = int(round(h * scale))
    img_resized = img.resize((nw, nh), resample=Image.BICUBIC)

    # Pad to target x target (centered)
    pw = target - nw
    ph = target - nh
    left = pw // 2
    top = ph // 2
    right = pw - left
    bottom = ph - top
    img_padded = ImageOps.expand(img_resized, border=(left, top, right, bottom), fill=fill)

    debug = {
        "input_size": (w, h),
        "target": target,
        "scale": scale,
        "resized_size": (nw, nh),
        "padding": (left, top, right, bottom),
        "output_size": img_padded.size,
    }
    
    return img_padded

def images_norm_transpose(jpg):
    # currently only support 1 jpg file
    with Image.open(jpg) as img:
        # normalized input image as CLIP needed
        # https://github.com/openai/CLIP/blob/dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1/clip/clip.py#L85
        # Use Letterbox Resize instead of direct resize
        img = img.resize((224,224))
        img_array = np.array(img).astype(np.float32)
        img_array /= 255.0
        img_array = (img_array - mean) / std
    r = np.transpose(img_array, (2, 0, 1)).astype(np.float32)
    return r

def extract_image_feature(chw_jpg_array_add_batch):
    input_dict = {
        "input" : chw_jpg_array_add_batch
    }
    outputs = image_sess.run(None, input_dict)
    return np.array(outputs)

def extract_text_feature(txt):
    # currently only support 1,77 encode
    # since the scene is search pictures with text
    padded_all_tokens = tokenize_and_padding(txt)
    
    # padded_all_tokens.shape = (batch_size, 77)
    t_array = np.array(padded_all_tokens)
    batch_size = t_array.shape[0]
    input_dict = {
        "input" : t_array
    }
    outputs = text_sess.run(None, input_dict)
    # outputs is extracted text features
    # np.array(outputs).shape = (1, batch_size, 512)

    return np.array(outputs)

def tokenize_and_padding(txt_list):
    """
    this function will use CLIP preprocess critical steps to transform texts into tokens.
    max support text length is 77    
    
    Refering below source code: 
        https://github.com/openai/CLIP/blob/main/clip/clip.py#L14    
        https://github.com/openai/CLIP/blob/main/clip/clip.py#L28    
        https://github.com/openai/CLIP/blob/main/clip/clip.py#L228    
        https://github.com/openai/CLIP/blob/main/clip/clip.py#L229    
        https://github.com/openai/CLIP/blob/main/clip/clip.py#L230   
    """
    from simple_tokenizer import SimpleTokenizer as _Tokenizer    
        
    _tokenizer = _Tokenizer()    
    
    sot_token = _tokenizer.encoder["<|startoftext|>"]    
    eot_token = _tokenizer.encoder["<|endoftext|>"]    
    
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in txt_list]
    
    print("======================================================")
    print(f"Clip Tokenizer for '{txt_list}' are : {all_tokens}")
    print("======================================================")
    
    max_length = 77
    
    if len(all_tokens) <= max_length:
        # add Zeros to token array    
        # e.g. [[1,2,3,0,0],[4,5,7,8,0]]
        padded_all_tokens = [tokens + [0] * (max_length - len(tokens)) for tokens in all_tokens]    
    else:
        # for each tokens in padded_all_tokens , snap contents whithin max_length by tokens[:max_length]    
        padded_all_tokens = [tokens[:max_length] for tokens in padded_all_tokens]    
        
    return padded_all_tokens    

def text_to_images_search_1N512(img_vecs, txt_vec, tau=0.02, topk=None, eps=1e-12):
    """
    img_vecs: (1, N, 512)
    txt_vec : (1, 1, 512)

    returns:
      scores : (N,) cosine similarity for ranking
      percent: (N,) softmax percent over N images (display-only)
      order  : (N,) indices sorted by scores desc
      topk_idx: (K,) if topk is not None
    """
    I = np.asarray(img_vecs, dtype=np.float32)  # (1,N,512)
    t = np.asarray(txt_vec,  dtype=np.float32)  # (1,1,512)

    # 1) L2 normalize over feature axis=2
    I = I / (np.linalg.norm(I, axis=2, keepdims=True) + eps)  # (1,N,512)
    t = t / (np.linalg.norm(t, axis=2, keepdims=True) + eps)  # (1,1,512)

    # 2) cosine scores: broadcast multiply then sum over feature axis
    # (1,N,512) * (1,1,512) -> (1,N,512) -> sum(axis=2) -> (1,N)
    scores = np.sum(I * t, axis=2).reshape(-1)                # (N,)

    # 3) softmax over N images (temperature tau) -> percent (display)
    # x = scores / float(tau)
    # x = x - x.max()  # stability
    # p = np.exp(x)
    # percent = (p / p.sum()) * 100.0                           # (N,)
    
    # 3.1) cosine similarity
    percent = (scores + 1.0) * 50.0               # (-1~1) -> (0~100)

    # 4) ranking by cosine
    order = np.argsort(-scores)

    if topk is not None:
        k = int(topk)
        topk_idx = order[:k]
        return scores, percent, order, topk_idx

    return scores, percent, order


def main(args=None):

    
    # use picture to search pictures
    with open("file_name_and_metadata.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    test_pic_name = "./test/0b48b477aa4075c5489c5bfb0df35017.jpg"
    r = images_norm_transpose(test_pic_name)
    img_search_q = extract_image_feature(np.array([r]))  
    
    # use metadata to search pictures
    # OVER 77 WORDS !!!!
    
    # search_query = [data["./test/00a325fd4dc22079095ab46330abf7be.jpg"]["metadata"]]
    # print(search_query)
    # prepare_clip_text_tokenizer()
    # txt_features = extract_text_feature(search_query)
    # print(f"main : txt_features extracted : shape is {txt_features.shape} , value is {txt_features}")
    # exit()

    i = 1
    file_name_list = []
    img_array_list = []
    for jpg, _ in data.items():
        r = images_norm_transpose(jpg)
        img_array = np.array([r])
        images_features = extract_image_feature(img_array)  
        data.setdefault(jpg, {})["images_features"] = images_features
        print(f" Vector #{i}  : images_features extracted : shape is {images_features.shape}")
        file_name_list.append(jpg)
        img_array_list.append(images_features)
        i = i+1
    
    print(np.array(img_array_list).shape)
    
    img_array_list = np.concatenate(img_array_list, axis=1)
     
    print(np.array(img_array_list).shape)
    
    
    # if topk is not None: will return below 
    #     return scores, percent, order, topk_idx
    # else
    #     return scores, percent, order
    
    s,p,o,k = text_to_images_search_1N512(img_array_list, img_search_q, topk=5)
    
    # print top5 percent , k are the top5 index
    print(np.array(p)[k])
    
    # print top5 file name
    print(np.array(file_name_list)[k])

if __name__ == '__main__':
    main()