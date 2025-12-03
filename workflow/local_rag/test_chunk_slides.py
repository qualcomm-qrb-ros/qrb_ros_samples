import os
import requests

def download_file(url:str):
    response = requests.get(url)
    response.raise_for_status()

    with open("test.txt", "wb") as f:
        f.write(response.content)

    return True

CHUNK_SIZE = 1024
OVERLAP = 128

def chunk_text_file(file_path: str):
    """
    load local path test.txt and cut into chunks
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
        print(f"chunked txt num : {len(chunk_list)} ")
        print(chunk_list[0][:512])
    else:
        print("test text file download failed.")