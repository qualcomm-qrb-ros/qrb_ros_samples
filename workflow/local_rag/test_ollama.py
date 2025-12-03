import numpy as np
import requests

url = "http://xxxxxxx:22222/api/embeddings"
model = "qwen3-embedding:0.6b"

def get_embedding_ollama(text: str, timeout: int = 10) -> np.ndarray:
    payload = {"model": model, "prompt": text}  # Ollama 用 prompt
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return np.array(data.get("embedding", []), dtype=float)

print(get_embedding_ollama("attention is all you need"))