

#!/usr/bin/env python3
# ============================================================
# VIBE CODING WARNING
#
# This file is a minimal, idea-driven prototype.
# It prioritizes clarity and iteration speed over completeness.
#
# Thanks to the open source community for making vector search,
# numerical computing, and ONNX-based deployment practical:
# NumPy, SQLite, OpenCV, ONNX Runtime, and CLIP-related projects.
# ============================================================
#
# pip install numpy opencv-python onnxruntime
#


import requests
import json
import sys
import os
import time
import argparse
import numpy as np
from PIL import Image, ImageOps
import onnxruntime as ort

import sqlite3
import time
from typing import Optional, List, Tuple

# ============================================================
# Minimal ONNX runtime
# ============================================================
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


# ============================================================
# Minimal function name printer
# ============================================================
import inspect
def here():
    return inspect.currentframe().f_back.f_code.co_name


# ============================================================
# Manual Clip PreProcess Part
# ============================================================
# Fetch ONNX CLIP model
# https://github.com/Lednik7/CLIP-ONNX/tree/main
# wget https://clip-as-service.s3.us-east-2.amazonaws.com/models/onnx/ViT-B-32/visual.onnx
# wget https://clip-as-service.s3.us-east-2.amazonaws.com/models/onnx/ViT-B-32/textual.onnx


# normalized input image as CLIP needed
# https://github.com/openai/CLIP/blob/dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1/clip/clip.py#L85
# need to keep format as float32 to adapt model input requests


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


def images_norm_transpose(jpg):
    # currently only support 1 jpg file
    # with Image.open(jpg) as img:
    #     # normalized input image as CLIP needed
    #     # https://github.com/openai/CLIP/blob/dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1/clip/clip.py#L85
    #     # Use Letterbox Resize instead of direct resize
    #     img = img.resize((224,224), Image.BICUBIC)
    #     img_array = np.array(img).astype(np.float32)
    #     img_array /= 255.0
    #     img_array = (img_array - mean) / std
    # r = np.transpose(img_array, (2, 0, 1)).astype(np.float32)
    
    # Refering : # https://github.com/openai/CLIP/blob/dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1/clip/clip.py#L85
    
    img = Image.open(jpg).convert("RGB")    
    
    # resize shortest side
    w, h = img.size
    scale = 224 / min(w, h)
    img = img.resize((round(w * scale), round(h * scale)), Image.BICUBIC)

    # center crop
    w, h = img.size
    left = (w - 224) // 2
    top  = (h - 224) // 2
    img = img.crop((left, top, left + 224, top + 224))

    x = np.array(img).astype(np.float32) / 255.0

    mean = np.array([0.48145466, 0.4578275, 0.40821073])
    std  = np.array([0.26862954, 0.26130258, 0.27577711])
    img_array = (x - mean[None,None,:]) / std[None,None,:]

    r = np.transpose(img_array, (2, 0, 1)).astype(np.float32)
    return r

def extract_image_feature(chw_jpg_array_add_batch):
    input_dict = {
        "input" : chw_jpg_array_add_batch
    }
    outputs = image_sess.run(None, input_dict)
    v = outputs[0]
    v = v / np.linalg.norm(v)
    print(f"{here()} : v shape:", v.shape)
    print(f"{here()} : L2 norm after:", np.linalg.norm(v))
    print(f"{here()} : v@v:", v.squeeze()@v.squeeze())
    return v

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


# ============================================================
# Minimal timing helpers
# ============================================================
_t0 = None
def tic():
    global _t0
    _t0 = time.perf_counter()

def toc(name: str):
    dt = time.perf_counter() - _t0
    print(f"[TIME] {name}: {dt:.2f}s")


# ============================================================
# Global runtime state (explicit, no hidden magic)
# ============================================================
DIM = 512 # CLIP specific
MEM_MATRIX = np.empty((0, DIM), dtype=np.float32)
MEM_LAST_ID = 0


# ============================================================
# SQLite helpers (source of truth)
# ============================================================
def init_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS visual_store (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        path TEXT NOT NULL UNIQUE,
        metadata TEXT,
        vector BLOB NOT NULL
    );
    """)
    conn.commit()
    return conn


def get_id_by_path(conn: sqlite3.Connection, path: str) -> Optional[int]:
    cur = conn.execute(
        "SELECT id FROM visual_store WHERE path = ?", (path,)
    )
    row = cur.fetchone()
    return row[0] if row else None


def insert_vector(
    conn: sqlite3.Connection,
    path: str,
    vector: np.ndarray,
    metadata: str = ""
) -> int:
    cur = conn.execute(
        "INSERT INTO visual_store(path, metadata, vector) VALUES (?, ?, ?)",
        (path, metadata, vector)
    )
    conn.commit()
    return cur.lastrowid


def fetch_vectors_after(conn: sqlite3.Connection, last_id: int):
    cur = conn.execute(
        "SELECT id, vector FROM visual_store "
        "WHERE id > ? ORDER BY id ASC",
        (last_id,)
    )
    return cur.fetchall()


# ============================================================
# In-memory matrix sync (need to confirm matrix in memory is aligned with DB)
# ============================================================
def sync_mem_matrix(conn: sqlite3.Connection):
    global MEM_MATRIX, MEM_LAST_ID

    tic()
    rows = fetch_vectors_after(conn, MEM_LAST_ID)
    if not rows:
        return

    new_vecs = []
    for id_, blob in rows:
        new_vecs.append(np.frombuffer(blob, dtype=np.float32))
        MEM_LAST_ID = id_

    MEM_MATRIX = np.vstack([MEM_MATRIX, np.stack(new_vecs)])
    toc("DB → RAM sync")


# ============================================================
# Search (dot product)
# ============================================================
def search_db(
    query_vec: np.ndarray,
    conn: sqlite3.Connection,
    thresh: float = 0.85
) -> Tuple[List[int], np.ndarray]:
    sync_mem_matrix(conn)

    if MEM_MATRIX.shape[0] == 0:
        return [], np.array([])

    tic()
    print(f"{here()} : MEM_MATRIX shape is {MEM_MATRIX.shape}")
    print(f"{here()} : query_vec.squeeze shape is {query_vec.squeeze().shape}")
    sims = MEM_MATRIX @ query_vec.squeeze()
    toc("dot product")

    # DEBUG : search exsited jpg should be score 1
    # 
    # best = np.argmax(sims)
    # print( [int(best)] )
    # print( np.array([sims[best]]) )
    
    idx = np.where(sims >= thresh)[0]
    order = np.argsort(-sims[idx])
    idx = idx[order]
    return idx.tolist(), sims[idx]


# ============================================================
# match_level (consider results more than 0.85 cosine similarity)
# ============================================================
def match_level(cos):
    if cos >= 0.98:
        return "SAME"
    elif cos >= 0.88:
        return "SIMILAR"
    elif cos >= 0.85:
        return "RELATED"
    else:
        return None


# ============================================================
# print_matches (search in sql)
# ============================================================
def print_matches(conn, idx, score):
    output = ""
    
    if not idx:
        print("match idx: []")
        return

    print(f"match idx: {idx}")

    # MEM index -> db id
    db_ids = [i + 1 for i in idx]
    placeholders = ",".join("?" * len(db_ids))

    rows = conn.execute(
        f"""
        SELECT id, path, metadata
        FROM visual_store
        WHERE id IN ({placeholders})
        """,
        db_ids,
    ).fetchall()

    row_map = {row[0]: row for row in rows}

    for i, cos in zip(idx, score):
        level = match_level(cos)
        if not level:
            continue

        row = row_map.get(i + 1)
        if row:
            output = output + f"{level} {cos:.3f} {row}\n"

    return output


# ============================================================
# render_html (vide coding to output to html for easier look)
# ============================================================
import re
def render_html(result_str, out_path="./output.html"):
    items = []
    for line in result_str.strip().splitlines():
        m = re.match(r"(SAME|SIMILAR)\s+([\d.]+)\s+\((\d+),\s+'([^']+)',\s+'(.+)'\)", line)
        if m:
            level, score, id_, img, desc = m.groups()
            items.append((level, score, id_, img, desc))

    def card(i):
        return f"""
        <div class="card {'same' if i[0]=='SAME' else ''}">
            <img src="{i[3]}" />
            <div class="meta">
                <b>{i[0]}</b> · {i[1]} · ID {i[2]}
                <div>{i[4]}</div>
            </div>
        </div>
        """

    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>Image Search Result</title>
<style>
body{{font-family:sans-serif;background:#f5f5f5;padding:20px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:16px}}
.card{{background:#fff;border-radius:10px;box-shadow:0 2px 6px rgba(0,0,0,.1);overflow:hidden}}
.card.same{{border:3px solid #2ecc71}}
img{{width:100%;height:auto;max-height:340px;object-fit:contain;background:#eee}}
.meta{{padding:10px;font-size:12px;line-height:1.4}}
</style>
</head>
<body>

<h2>SAME</h2>
<div class="grid">
{''.join(card(i) for i in items if i[0]=='SAME')}
</div>

<h2>SIMILAR</h2>
<div class="grid">
{''.join(card(i) for i in items if i[0]=='SIMILAR')}
</div>

</body>
</html>"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


# ============================================================
# main (proceed sample picture search)
# ============================================================
def main(args=None):

    # self check of vectorize
    # v1 = encode_image(img)
    # v2 = encode_image(img)
    # 
    # print("self similarity:", v1 @ v2) , should be one

    test_pic_name = "./test/3b8e5457c9419df3a67ed590cbb46b56.jpg"
    r = images_norm_transpose(test_pic_name)
    img_search_q = extract_image_feature([r])
    
    # ====  USE TEXT TO SEARCH ====
    # use metadata to search pictures
    # OVER 77 WORDS !!!!
    # search_query = [data["./test/00a325fd4dc22079095ab46330abf7be.jpg"]["metadata"]]
    # print(search_query)
    # prepare_clip_text_tokenizer()
    # txt_features = extract_text_feature(search_query)
    # print(f"main : txt_features extracted : shape is {txt_features.shape} , value is {txt_features}")
    # ====  END ====
    
    # use picture to search pictures
    with open("file_name_and_metadata.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    # init DB
    DB_PATH = "visual.db"
    conn = init_db(DB_PATH)
    
    tic()
    for jpg, _ in data.items(): 
        if get_id_by_path(conn, jpg) is None:
            print(f" the jpg file not exsited in DB : {jpg}")
            r = images_norm_transpose(jpg)
            img_array = [r]
            images_features = extract_image_feature(img_array)
            insert_vector(conn, jpg, images_features, data[jpg]["metadata"])            
        else:
            print(f"[SKIP] image {jpg} already exists")
    toc("Vector Store Built")

    idx, score = search_db(img_search_q, conn)
    print("match idx:", idx)
    print("score:", score)
    o = print_matches(conn, idx, score)
    render_html(o)

if __name__ == '__main__':
    main()