# source page : https://www.modelscope.ai/models/Qwen/Qwen3-Embedding-0.6B
from sentence_transformers import SentenceTransformer

# Load the model
model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

# We recommend enabling flash_attention_2 for better acceleration and memory saving,
# together with setting `padding_side` to "left":
# model = SentenceTransformer(
#     "Qwen/Qwen3-Embedding-0.6B",
#     model_kwargs={"attn_implementation": "flash_attention_2", "device_map": "auto"},
#     tokenizer_kwargs={"padding_side": "left"},
# )

# The queries and documents to embed
queries = [
    "What is the capital of China?",
    "Explain gravity",
]
documents = [
    "The capital of China is Beijing.",
    "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
]

# Encode the queries and documents. Note that queries benefit from using a prompt
# Here we use the prompt called "query" stored under `model.prompts`, but you can
# also pass your own prompt via the `prompt` argument
query_embeddings = model.encode(queries, prompt_name="query")
document_embeddings = model.encode(documents)

# Compute the (cosine) similarity between the query and document embeddings
similarity = model.similarity(query_embeddings, document_embeddings)
print(similarity)
# tensor([[0.7646, 0.1414],
#         [0.1355, 0.6000]])


import os
import requests

def download_file(url:str):
    response = requests.get(url)
    response.raise_for_status()

    with open("test.txt", "wb") as f:
        f.write(response.content)

    return True

CHUNK_SIZE = 2048
OVERLAP = 128

def chunk_text_file(file_path: str):
    """
    load ./test.txt and cut into chunks
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"file not existed: {file_path}")

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    # change \r into \n
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    
    # keep some overlap of chunks
    step = CHUNK_SIZE - OVERLAP
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        if not chunk:
            break
        chunks.append(chunk)
        start += step
        
    return chunks

if __name__ == "__main__":
    
    # solute to https://github.com/The-Pocket/PocketFlow-Tutorial-Video-Generator and its author
    url = "https://raw.githubusercontent.com/The-Pocket/PocketFlow-Tutorial-Video-Generator/refs/heads/main/docs/llm/transformer.md"
    if download_file(url):
        chunk_list = chunk_text_file("test.txt")
        # print(chunk_list[0][:512])
        print("chunk list generated")
    else:
        print("test txt file download failed.")

    print(f"chunked txt num : {len(chunk_list)} ")

    # get embedding of each chunk
    vector_list = model.encode(chunk_list)

    # build vector store
    vector_store = [(chunk, None) for chunk in chunk_list]   
    for i, (chunk, _) in enumerate(vector_store):
        # store into another list
        print(f"vector_list[{i}]: {vector_list[i]}")
        vector_store[i] = (chunk, vector_list[i])

    print(f"vector_store[0][1]: {vector_store[0][1]}")
    exit()

    # test vector store
    query = "attention is all you need"
    query_embeddings = model.encode(query, prompt_name="query")
    
    for item in vector_store:
        k_txt, document_embeddings = item
        similarity = model.similarity(query_embeddings, document_embeddings)
        print(similarity)