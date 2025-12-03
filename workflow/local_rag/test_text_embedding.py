import numpy as np

# your embedding model

texts = "xxxxxxxxxxxxxx"

res = get_embeddings(texts)
embedding_np_array_k = np.array(res.data[0].embedding)

query = "xxxxxxxxxxxxxxx"
res = get_embeddings(query)
embedding_np_array_q = np.array(res.data[0].embedding)

similarities = np.dot(embedding_np_array_k, embedding_np_array_q)

print(similarities)