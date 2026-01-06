

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


def extract_text_feature(txt_list):
    padded = tokenize_and_padding(txt_list)
    t_array = np.array(padded)   # ONNX 常见需要 int64
    outputs = text_sess.run(None, {"input": t_array})
    v = outputs[0]
    v = v / np.linalg.norm(v)
    print(f"{here()} : v shape:", v.shape)
    print(f"{here()} : L2 norm after:", np.linalg.norm(v))
    print(f"{here()} : v@v:", v.squeeze()@v.squeeze())
    return v

def tokenize_and_padding(txt_list):
    from simple_tokenizer import SimpleTokenizer as _Tokenizer
    _tokenizer = _Tokenizer()

    sot = _tokenizer.encoder["<|startoftext|>"]
    eot = _tokenizer.encoder["<|endoftext|>"]

    all_tokens = [[sot] + _tokenizer.encode(text) + [eot] for text in txt_list]

    max_length = 77
    padded_all_tokens = [
        (tokens + [0] * (max_length - len(tokens))) if len(tokens) <= max_length
        else tokens[:max_length]
        for tokens in all_tokens
    ]
    return padded_all_tokens


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


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-12)


# ============================================================
# Search (dot product)
# ============================================================
def search_db(
    query_vec: np.ndarray,
    conn: sqlite3.Connection,
    logit_scale: float = 50.0,
) -> Tuple[List[int], np.ndarray]:
    sync_mem_matrix(conn)

    if MEM_MATRIX.shape[0] == 0:
        return [], np.array([])

    tic()
    print(f"{here()} : MEM_MATRIX shape is {MEM_MATRIX.shape}")
    print(f"{here()} : query_vec.squeeze shape is {query_vec.squeeze().shape}")
    sims = MEM_MATRIX @ query_vec.squeeze()
    logits = float(logit_scale) * sims
    
    # 取 topk
    k = int(np.clip(10, 1, 10))
    idx = np.argsort(-logits)[:k]
    top_logits = logits[idx]
    probs = softmax(top_logits)
    print(probs)

    toc("dot product")
    return sims

# ============================================================
# match_level (consider results more than 0.85 cosine similarity)
# ============================================================
# def match_level(cos):
#     if cos >= 0.98:
#         return "SAME"
#     elif cos >= 0.90:
#         return "SIMILAR"
#     elif cos >= 0.88:
#         return "RELATED"
#     else:
#         return None


# ============================================================
# print_matches (search in sql)
# ============================================================
def print_matches(conn, idx, score):
    output = ""
    
    if not idx:
        print("match idx: []")
        return

    # print(f"match idx: {idx}")

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

    lines = []
    for i, cos in zip(idx, score):
        level = match_level(cos)
        if not level:
            continue

        row = row_map.get(i + 1)
        if row:
            lines.append(f"{level} {cos:.3f} {row}")

    output = "\n".join(lines)

    return output


# ============================================================
# print_topk (rank in cosine similarity)
# ============================================================
def print_topk(conn, scores, k=10):
    
    k = int(k)
    if k <= 0:
        return ""
    k = min(k, scores.size)
    
    scores = ((scores + 1.0) / 2.0)*100
    print(f"score to percent is : {scores}")
    topk_idx = np.argsort(-scores)[:k]   # 0-based indices
    topk_scores = scores[topk_idx]
    print(topk_scores)

    # Map MEM index -> db id (1-based)
    db_ids = [int(i) + 1 for i in topk_idx]
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

    lines = []
    for i, cos in zip(topk_idx, topk_scores):
        row = row_map.get(int(i) + 1)
        if row:
            lines.append(f"{cos:.3f} {row}")

    return "\n".join(lines)


# ============================================================
# render_html (vide coding to output to html for easier look)
# ============================================================

import ast
from html import escape

def render_topk_to_html(topk, k=10, out_path="./output.html", title="CLIP TopK Results"):
    """
    topk: list[str] 或 str(多行)
      每行格式示例:
        "64.955 (689, './test/xxx.jpg', 'neckline, shoe, ...')"
      其中第一个数字是 percent（你已换算好的分数）

    k: 1<=k<=10，会自动 clamp；解析条目少于 k 则显示全部
    """

    # ---- normalize input lines ----
    if topk is None:
        lines = []
    elif isinstance(topk, str):
        lines = [ln.strip() for ln in topk.strip().splitlines() if ln.strip()]
    else:
        lines = [str(ln).strip() for ln in topk if str(ln).strip()]

    # ---- clamp k to [1, 10] ----
    try:
        k = int(k)
    except Exception:
        k = 10
    k = 1 if k < 1 else (10 if k > 10 else k)

    # ---- parse lines robustly ----
    items = []
    for ln in lines:
        # split once: score + tuple_str
        parts = ln.split(None, 1)
        if len(parts) != 2:
            continue
        score_str, tuple_str = parts[0], parts[1]

        try:
            score = float(score_str)
            tpl = ast.literal_eval(tuple_str)  # (id, path, metadata)
            if not (isinstance(tpl, tuple) and len(tpl) >= 3):
                continue
            id_, path, meta = tpl[0], tpl[1], tpl[2]
            items.append({
                "score": score,
                "id": id_,
                "path": path,
                "meta": meta
            })
        except Exception:
            # line format not compatible -> skip
            continue

    # ---- sort & take topk ----
    items.sort(key=lambda x: x["score"], reverse=True)
    items = items[:min(k, len(items))]

    # ---- empty fallback ----
    if not items:
        html = f"""<!doctype html>
<html><head><meta charset="utf-8"/><title>{escape(title)}</title></head>
<body style="font-family:system-ui;padding:20px;background:#f6f7fb;">
<h2>{escape(title)}</h2>
<p>No valid items parsed.</p>
</body></html>"""
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"==== HTML Rendered: {out_path} ====")
        return out_path

    # ---- render cards ----
    def card(rank, it):
        score = it["score"]
        id_ = escape(str(it["id"]))
        path = escape(str(it["path"]))
        meta = escape(str(it["meta"]))

        # 折叠长文本，提高可读性
        if len(meta) > 120:
            meta_html = f"""
            <details>
              <summary>metadata</summary>
              <div class="desc">{meta}</div>
            </details>
            """
        else:
            meta_html = f"""<div class="desc">{meta}</div>"""

        return f"""
        <div class="card">
          <div class="thumb">
            <img src="{path}" alt="{id_}"/>
          </div>
          <div class="meta">
            <div class="row">
              <div class="rank">#{rank}</div>
              <div class="score">{score:.3f}%</div>
            </div>
            <div class="id">ID: {id_}</div>
            <div class="path">{path}</div>
            {meta_html}
          </div>
        </div>
        """

    cards_html = "\n".join(card(i + 1, it) for i, it in enumerate(items))

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{escape(title)}</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; background:#f6f7fb; padding:20px; }}
    h2 {{ margin: 0 0 10px 0; }}
    .hint {{ color:#666; font-size:12px; margin-bottom:14px; }}
    .grid {{ display:grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 14px; }}
    .card {{ background:#fff; border-radius:12px; box-shadow:0 2px 10px rgba(0,0,0,.08); overflow:hidden; border:1px solid #eee; }}
    .thumb {{ background:#eee; }}
    img {{ width:100%; height:300px; object-fit:contain; display:block; background:#eee; }}
    .meta {{ padding:10px 12px; font-size:12px; line-height:1.35; }}
    .row {{ display:flex; align-items:center; gap:10px; margin-bottom:6px; }}
    .rank {{ font-weight:700; color:#111; }}
    .score {{ font-weight:700; color:#2563eb; }}
    .id {{ color:#111; margin:4px 0; }}
    .path {{ color:#555; word-break: break-all; margin:4px 0; }}
    .desc {{ color:#333; white-space:pre-wrap; word-break: break-word; margin-top:6px; }}
    details summary {{ cursor:pointer; color:#111; font-weight:600; }}
  </style>
</head>
<body>
  <h2>{escape(title)}</h2>
  <div class="hint">Showing Top-{len(items)} (k={k}, clamped to 1..10), sorted by score desc.</div>
  <div class="grid">
    {cards_html}
  </div>
</body>
</html>"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"==== HTML Rendered: {out_path} ====")
    return out_path

# ============================================================
# main (proceed sample picture search)
# ============================================================
def main(args=None):

    # ==== Search With Single Picture ====
    # 
    # 1. verify vector similarity is working
    # self check of vectorize
    # v1 = encode_image(img)
    # v2 = encode_image(img)
    # print("self similarity:", v1 @ v2) , should be one
    #
    # 2. check with exsited picture
    # test_pic_name = "./test/3b8e5457c9419df3a67ed590cbb46b56.jpg"
    # r = images_norm_transpose(test_pic_name)
    # query_vector = extract_image_feature([r])
    
    # ====  Search With Text ( <= 77 tokens) ====
    # use metadata to search pictures
    # OVER 77 WORDS !!!!
    q_txt = ["a photo of red desk"]
    print(q_txt)
    prepare_clip_text_tokenizer()
    query_vector = extract_text_feature(q_txt)
    print(f"main : txt_features extracted : shape is {query_vector.shape} , value is {query_vector}")
    # ====  END ====
    
    # use picture to search pictures
    with open("file_name_and_metadata.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    # init DB
    DB_PATH = "visual.db"
    conn = init_db(DB_PATH)
    
    # assemble DB
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

    # search in DB
    scores = search_db(query_vector, conn)
    o = print_topk(conn, scores)
    render_topk_to_html(o)

if __name__ == '__main__':
    main()