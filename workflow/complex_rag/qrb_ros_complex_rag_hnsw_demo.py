
# =========================================================
# ⚠️ VIBE CODING WARNING
#
# This code includes parts generated with the help of AI.
# It is for learning, demo, and experiments only.
#
# The logic is simplified and not production‑ready.
# Do NOT use this code in real products or commercial systems.
#
# Use at your own risk.
# =========================================================


# =========================================================
# Simple Hybrid Vector Search for Teaching (All-in-One)
# What this code does:
# - Read a long text file: fanren.txt
# - Split the text into small pieces
# - Use layered HNSW for frequently used data
# - Use simple IVF-PQ for less-used data
# - Single file, no external libraries
# =========================================================

import random


# ---------------- Basic Tools ----------------

def l2_distance(a, b):
    # Calculate the straight-line (Euclidean) distance between two vectors
    s = 0.0
    for i in range(len(a)):
        d = a[i] - b[i]
        s += d * d
    return s ** 0.5


def text_to_vector(text):
    """
    Very simple text-to-vector method for learning:
    - Turn each character into a number
    - Average them into a 3D vector
    """
    v = [0.0, 0.0, 0.0]
    for i, ch in enumerate(text):
        v[i % 3] += ord(ch)
    for i in range(3):
        v[i] = v[i] / len(text)
    return v


def split_text(text, chunk_size):
    """
    Split long text into fixed-length chunks
    """
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i + chunk_size])
        i += chunk_size
    return chunks


def load_text_file(path):
    """
    Try several common encodings to read a Chinese text file
    """
    encodings = ["utf-8", "gbk", "gb2312"]
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc) as f:
                text = f.read()
                print(f"File read successfully using {enc} encoding")
                return text
        except UnicodeDecodeError:
            pass
    raise RuntimeError("Cannot read file encoding, please convert to UTF-8")


# ---------------- Layered HNSW ----------------

class LayeredHNSW:
    """
    Teaching version of layered HNSW:
    - Multiple layers
    - Search from top layer to bottom layer
    """

    def __init__(self, M=16, efConstruction=128, max_level=10):
        self.M = M
        self.efConstruction = efConstruction
        self.max_level = max_level

        self.vectors = []     # All stored vectors
        self.levels = []      # Top level for each vector
        self.graph = []       # graph[level][node] -> neighbors
        self.entry_point = None

        for _ in range(max_level + 1):
            self.graph.append({})

    def _random_level(self):
        """
        Randomly choose a level for a new vector
        """
        return random.randint(0, self.max_level)

    def add(self, vector):
        idx = len(self.vectors)
        self.vectors.append(vector)
        print(f"\n✅Adding vector with index {idx}")

        level = self._random_level()
        self.levels.append(level)
        print(f"Assigned level: {level}")

        for l in range(level + 1):
            self.graph[l][idx] = []
            print(f"Create empty neighbor list at level {l}")

        if self.entry_point is None:
            self.entry_point = idx
            print("This is the first vector, set as entry point")
            return

        current = self.entry_point
        print(f"Start search from entry point {self.entry_point} of level {level}")

        for l in range(self.max_level, -1, -1):
            print(f"Searching at level {l}")

            current, path = self._greedy_search_with_path(vector, current, l)
            print(f"====> GREEDY : Closest node at this level is NODE #{current}")

            if l <= level:
                print("====> Now connect neighbors at this level")
                neighbors = self._select_neighbors_from_path(vector, path, l)
                print(f"Chosen neighbors: {neighbors}")
                for nb in neighbors:
                    print(f"====> Connect node {idx} and node {nb} at level {l}")
                    self._connect(idx, nb, l)
            else:
                print("====> Skip connecting at this level")

    def _greedy_search_with_path(self, query_vec, entry, level):
        current = entry
        current_dist = l2_distance(query_vec, self.vectors[current])
        print(f"Greedy search at level {level}, start from {entry}")

        path = [current]
        neighbors = self.graph[level].get(current, [])
        print(f"Current neighbors: {neighbors}")

        improved = True
        while improved:
            improved = False
            for nb in self.graph[level].get(current, []):
                d = l2_distance(query_vec, self.vectors[nb])
                print(f"Check neighbor {nb}, Q & {nb} is distance {d}, Q & Current {current} distance is {current_dist} ")
                if d < current_dist:
                    current = nb
                    current_dist = d
                    path.append(nb)
                    improved = True
                    print(f"Move to closer node {nb}")

        print("No closer neighbor, go to next level")
        return current, path

    def _select_neighbors_from_path(self, query_vec, path, level):
        candidates = []
        for node in path:
            if node in self.graph[level]:
                d = l2_distance(query_vec, self.vectors[node])
                candidates.append((d, node))

        candidates.sort(key=lambda x: x[0])
        return [node for (_, node) in candidates[:self.M]]

    def _connect(self, a, b, level):
        if a not in self.graph[level] or b not in self.graph[level]:
            return

        if b not in self.graph[level][a]:
            self.graph[level][a].append(b)
        if a not in self.graph[level][b]:
            self.graph[level][b].append(a)

        for node in (a, b):
            if len(self.graph[level][node]) > self.M:
                self._trim_neighbors(node, level)

    def _trim_neighbors(self, node, level):
        neighbors = self.graph[level][node]
        far = None
        far_dist = -1
        for nb in neighbors:
            d = l2_distance(self.vectors[node], self.vectors[nb])
            if d > far_dist:
                far_dist = d
                far = nb
        neighbors.remove(far)

    def search(self, query):
        if self.entry_point is None:
            return None

        current = self.entry_point
        for l in range(self.max_level, -1, -1):
            if current in self.graph[l]:
                current, _ = self._greedy_search_with_path(query, current, l)
        return self.vectors[current]


# ---------------- IVF-PQ ----------------

class SimpleIVFPQ:
    def __init__(self, centers, parts=3):
        self.centers = centers
        self.parts = parts
        self.data = []

    def _assign_center(self, vector):
        best = 0
        best_dist = l2_distance(vector, self.centers[0])
        for i in range(1, len(self.centers)):
            d = l2_distance(vector, self.centers[i])
            if d < best_dist:
                best = i
                best_dist = d
        return best

    def _pq_encode(self, vector):
        size = len(vector) // self.parts
        code = []
        for i in range(self.parts):
            seg = vector[i * size:(i + 1) * size]
            code.append(sum(seg) / len(seg))
        return code

    def add(self, vector):
        self._assign_center(vector)
        code = self._pq_encode(vector)
        self.data.append((code, vector))

    def search(self, query):
        q_code = self._pq_encode(query)
        best = None
        best_dist = float("inf")
        for code, vec in self.data:
            d = l2_distance(code, q_code)
            if d < best_dist:
                best = vec
                best_dist = d
        return best


# ---------------- Hybrid System ----------------

class HybridVectorSearch:
    def __init__(self, centers):
        self.hot = LayeredHNSW(M=16, efConstruction=128, max_level=10)
        self.cold = SimpleIVFPQ(centers)

    def add_hot(self, vector):
        self.hot.add(vector)

    def add_cold(self, vector):
        self.cold.add(vector)

    def search(self, query):
        candidates = []
        h = self.hot.search(query)
        c = self.cold.search(query)
        if h:
            candidates.append(h)
        if c:
            candidates.append(c)

        best = None
        best_dist = float("inf")
        for v in candidates:
            d = l2_distance(query, v)
            if d < best_dist:
                best = v
                best_dist = d
        return best


# ================= Main Program =================

if __name__ == "__main__":

    long_text = load_text_file("fanren.txt")
    print("Total number of characters:", len(long_text))

    chunk_size = 100
    chunks = split_text(long_text, chunk_size)
    print("Number of text chunks:", len(chunks))

    centers = [
        [3000.0, 3000.0, 3000.0],
        [12000.0, 12000.0, 12000.0]
    ]
    system = HybridVectorSearch(centers)

    hot_limit = 20
    max_chunks = 100

    for i, chunk in enumerate(chunks):
        vec = text_to_vector(chunk)
        if i < hot_limit:
            system.add_hot(vec)
        else:
            system.add_cold(vec)

        if i >= max_chunks:
            break

    print("Index building finished")
    print("Hot data:", hot_limit, "Cold data:", max_chunks - hot_limit)

    query_text = "修炼功法提升境界"
    query_vec = text_to_vector(query_text)

    result = system.search(query_vec)

    print("Query text:", query_text)
    print("Result found:", result is not None)