# Complex RAG Technical Breakdown

## 0. 核心前提：RAG 的本质

RAG（Retrieval-Augmented Generation）的工程本质是：

**Embedding + 向量相似度搜索**。

给定查询向量 `q`，在大规模向量集合 `V={v1...vN}` 中高效找到 Top-K 相似向量。

```
q ──► embedding ──► 查询向量
               │
               ▼
        向量相似度计算
               │
               ▼
        Top‑K 文档向量
```

---

## 1. Brute Force（全量枚举）

### 原理步骤

1. 文档全部向量化
2. 查询向量与所有向量逐个计算距离
3. 全量排序取 Top‑K

```
q ──► dist(v1)
   ──► dist(v2)
   ──► ...
   ──► dist(vN)
```

**复杂度**：`O(N · d)`

**适用规模**：1e4 级以下

---

## 2. HNSW（分层图导航）

### 核心思想

构建多层小世界图，实现 "少看点，多跳转"。

```
Level2:    o────o
           │
Level1:   o──o──o──o
           │      │
Level0: o──o──o──o──o
```

### 搜索流程

1. 从最高层随机入口
2. 贪心靠近目标
3. 无法前进则下降一层
4. 底层返回 Top‑K

**特点**：高召回、低延迟、内存占用高

---

## 3. IVF（倒排文件索引）

### 核心思想

先做**空间粗分区**，再做局部精搜。

```
Vector Space
┌────C1────┬────C2────┬────C3────┐
│          │          │          │
└──────────┴──────────┴──────────┘
```

### 查询流程

```
Query q
  │
  ├─► 所有 Centroid
  │
  ├─► 选择 nprobe 个
  │
  └─► 搜索对应倒排表
```

**效果**：显著缩小候选集

---

## 4. PQ（乘积量化）

### 核心思想

用查表近似距离，减少内存与计算。

```
Vector v
[x1 x2 | x3 x4 | x5 x6]
   ↓      ↓      ↓
 code1  code2  code3
```

### ADC 距离

```
LUT1[c1] + LUT2[c2] + LUT3[c3]
```

**特点**：极高压缩率，小幅精度损失

---

## 5. HNSW + IVF + PQ（工业级组合）

### 索引构建

```
Raw Vectors
  ├─► IVF KMeans
  ├─► HNSW(centroid)
  ├─► PQ codebook
  └─► Encode → inverted lists
```

### 查询路径

```
Query
  ├─► HNSW 路由
  ├─► IVF 剪枝
  ├─► PQ 查表
  └─► Top‑K
```

---

## 6. 规模适配速查

```
10^4        → Brute Force
10^5–10^7   → HNSW
10^6–10^8   → IVF
10^7–10^9+  → IVF + PQ
10^8–10^9+  → HNSW + IVF + PQ
```

---

**总结**：
> 大规模 RAG = 路由（HNSW）+ 剪枝（IVF）+ 压缩（PQ）
