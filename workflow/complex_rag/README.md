# Prototype Complex RAG Scene

This implementation is based directly on the tutorial: [LLM Agents are simply Graph — Tutorial For Dummies](https://zacharyhuang.substack.com/p/llm-agent-internal-as-a-graph-tutorial).

---
This specific project is inspired by : https://www.anthropic.com/engineering/advanced-tool-use

**Modern RAG is usually a tough task since the Brute computation of Vectors' Cosine Similarity is not affordable in Data Exploration Era.**

**One of the common method to handle Vector Search Scene is using Hierarchical Navigable Small Worlds (HNSW).**

Check out explaination Video in 

**Vector Database Search - Hierarchical Navigable Small Worlds (HNSW) Explained** : https://www.youtube.com/watch?v=77QH0Y2PYKg

**How HNSW, IVF, & PQ Algorithms Power AI Search** : https://www.youtube.com/watch?v=xMCyq4pJcrQ

## Features (planing)

- Use pre-downloaded 7M words of large text file
- Cut into chunks and build vector database with Hierarchical Navigable Small Worlds (HNSW) 
- Search Vector with HNSW instead of Brute Computation

## Example Outputs

## Complex RAG Technical Breakdown

### 0. Core Premise: What RAG Really Is

Retrieval-Augmented Generation (RAG) is fundamentally an engineering problem of:

**Embedding + Vector Similarity Search**.

Given a query vector `q`, efficiently retrieve the Top‑K nearest vectors from a large set `V = {v1…vN}`.

```
q ──► embedding ──► query vector
               │
               ▼
        vector similarity computation
               │
               ▼
        Top‑K document vectors
```

As the scale grows, naive computation becomes infeasible.

---

### 1. Brute Force (Baseline)

#### Step‑by‑step

1. Convert all documents into vectors
2. At query time, compute distance to every vector
3. Sort all results and return Top‑K

```
q ──► dist(v1)
   ──► dist(v2)
   ──► ...
   ──► dist(vN)
```

**Time complexity**: `O(N · d)`  
**Properties**: exact, no index  
**Effective scale**: ≤ 10⁴ vectors

---

### 2. HNSW (Hierarchical Navigable Small World)

#### Core idea

Navigate a **multi‑layer small‑world graph** instead of scanning everything.

```
Level 2:    o────o
             │
Level 1:   o──o──o──o
             │      │
Level 0: o──o──o──o──o
```

### Search process

1. Start from a top‑layer entry point
2. Greedily move toward closer nodes
3. Stall → descend one layer
4. Bottom layer returns Top‑K

**Properties**: high recall, very low latency  
**Trade‑off**: higher memory usage due to graph edges  
**Effective scale**: 10⁵ – 10⁷ vectors

---

### 3. IVF (Inverted File Index)

#### Core idea

Perform **coarse space partitioning first**, then search locally.

```
Vector Space
┌────C1────┬────C2────┬────C3────┐
│          │          │          │
└──────────┴──────────┴──────────┘
```

### Index construction

1. Train `nlist` coarse centroids using KMeans
2. Assign each vector to its nearest centroid
3. Each centroid owns an inverted list

### Query execution

```
Query q
  │
  ├─► compute distance to all centroids
  │
  ├─► select nprobe closest centroids
  │
  └─► search corresponding inverted lists
```

**Effect**: drastically reduces candidate count  
**Sensitivity**: depends on `nlist` and `nprobe`  
**Effective scale**: 10⁶ – 10⁸ vectors

---

### 4. PQ (Product Quantization)

#### Core idea

Replace float distance computation with **table lookups**.

#### Encoding

```
Vector v
[x1 x2 | x3 x4 | x5 x6]
   ↓      ↓      ↓
 code1  code2  code3
```

Each sub‑vector is quantized independently and stored as an integer code.

#### Distance computation (ADC)

```
LUT1[code1] + LUT2[code2] + LUT3[code3]
```

**Properties**: extreme compression, very fast distance computation  
**Trade‑off**: small, controlled accuracy loss  
**Effective scale**: 10⁷ – 10⁹+ vectors

---

### 5. HNSW + IVF + PQ (Production‑Grade Architecture)

#### Design principle

Each component solves one scaling bottleneck:

```
Routing      Pruning        Compression
  HNSW   →     IVF     →        PQ
```

---

#### Index construction pipeline

```
Raw Vectors
  ├─► Train IVF centroids (KMeans)
  ├─► Build HNSW on centroids
  ├─► Train PQ codebooks
  └─► Encode vectors → inverted lists
```

---

#### Query execution path

```
Query
  ├─► HNSW routing (cluster selection)
  ├─► IVF pruning (candidate reduction)
  ├─► PQ LUT construction
  ├─► ADC distance computation
  └─► Top‑K results
```

---

### 6. Scale vs Technique Cheat Sheet

```
10^4          → Brute Force
10^5 – 10^7   → HNSW
10^6 – 10^8   → IVF
10^7 – 10^9+  → IVF + PQ
10^8 – 10^9+  → HNSW + IVF + PQ
```

## Getting Started

TBD
## How It Works?

TBD