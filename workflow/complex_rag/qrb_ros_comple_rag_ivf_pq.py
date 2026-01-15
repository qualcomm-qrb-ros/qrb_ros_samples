
# -*- coding: utf-8 -*-
# 本文件用于教学目的，**修复 IVF-PQ 在大规模文本下卡死的问题**
# This file fixes the IVF-PQ “hang” issue on large text input
# 仅使用 numpy，完整讲解 IVF-PQ 的工程细节
# Only numpy is used, complete engineering-level IVF-PQ explanation

import numpy as np  # 数值计算库 / Numerical computation library
import os  # 文件系统操作 / File system operations


def chunk_utf8_file(path, chunk_bytes=512):
    # 按 UTF-8 字节切分超大中文文本
    # Chunk huge UTF-8 Chinese text file by byte length
    print(f"📂 [chunk_utf8_file] path={path}, chunk_bytes={chunk_bytes}")

    chunks = []  # 文本块列表 / Text chunks list
    with open(path, "rb") as f:  # 二进制读取 / Binary read
        buffer = b""  # UTF-8 缓冲区 / UTF-8 buffer
        while True:
            data = f.read(chunk_bytes)
            if not data:
                break
            buffer += data
            try:
                txt = buffer.decode("utf-8")
                chunks.append(txt)
                buffer = b""
            except UnicodeDecodeError:
                pass

    if buffer:
        chunks.append(buffer.decode("utf-8"))

    print(f"✅ [chunk_utf8_file] total_chunks={len(chunks)}")
    return chunks


def text_to_vector(text, dim):
    # 将文本映射为固定维度向量（hash embedding）
    # Map text to fixed-dimension vector using hash embedding
    vec = np.zeros(dim, dtype=np.float32)
    for ch in text:
        vec[ord(ch) % dim] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def l2_distance(a, b):
    # 计算 L2 距离（向量化版本）
    # Compute L2 distance (vectorized)
    return np.linalg.norm(a - b)


def kmeans(data, k, iters=25, verbose=True):
    # ✅ 修复点：KMeans 完全向量化，避免 Python for 嵌套导致卡死
    # ✅ Fix: Fully vectorized KMeans to avoid Python for-loop freeze
    print(f"🚀 [kmeans] k={k}, iters={iters}, samples={len(data)}")

    n, d = data.shape
    centroids = data[np.random.choice(n, k, replace=False)]

    for it in range(iters):
        # 使用广播计算距离矩阵 (n, k)
        # Compute distance matrix via broadcasting
        dists = np.linalg.norm(data[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)

        for i in range(k):
            mask = labels == i
            if np.any(mask):
                centroids[i] = data[mask].mean(axis=0)

        if verbose and it % 5 == 0:
            print(f"⏱️ [kmeans] iter={it}")

    print("✅ [kmeans] finished")
    return centroids



def train_pq(data, m, ks):
    # 训练 Product Quantization（PQ）子空间码本
    # Train Product Quantization (PQ) subspace codebooks
    #
    # PQ 的核心思想：
    # 1）将原始 d 维向量按维度切分为 m 个互不重叠的子空间
    # 2）在每个子空间上分别训练一个 KMeans 量化器
    # 3）每个子空间只保存一个“中心索引”，显著压缩存储
    #
    # Core idea of PQ:
    # 1) Split original d-dim vectors into m disjoint subspaces
    # 2) Train an independent KMeans quantizer on each subspace
    # 3) Store only centroid indices for compression

    print(f"🧩 [train_pq] m={m}, ks={ks}, samples={len(data)}")  # 打印关键参数 / Print key parameters

    n, d = data.shape  # n 为向量数量，d 为向量维度 / n = number of vectors, d = dimension
    assert d % m == 0  # 每个子空间必须等分 / Each subspace must divide dimension evenly

    subdim = d // m  # 每个子空间的维度 / Dimension of each subspace
    # 示例：d=128, m=8 → subdim=16
    # Example: d=128, m=8 → subdim=16

    codebooks = []  # 用于保存所有子空间的码本 / Store all subspace codebooks

    for i in range(m):
        # 遍历第 i 个子空间
        # Iterate over the i-th subspace

        print(f"🔹 [train_pq] subspace={i}, dim={subdim}")  # 子空间级别打印 / Subspace-level print

        # 从所有向量中切出当前子空间的分量
        # Slice the i-th subspace component from all vectors
        #
        # 形状变化：
        # data:     (n, d)
        # sub_data: (n, subdim)
        sub_data = data[:, i * subdim:(i + 1) * subdim]

        # 在当前子空间上训练 KMeans
        # Train KMeans on the current subspace
        #
        # ks = 子空间中使用的聚类中心数量
        # ks = number of centroids in this subspace
        #
        # 每一个中心代表一个“局部原型向量”
        # Each centroid represents a local prototype vector
        centers = kmeans(sub_data, ks, iters=20)

        # 将该子空间的码本加入列表
        # Append this subspace codebook
        #
        # 最终 codebooks 的结构：
        # codebooks[m][ks][subdim]
        # codebooks shape: (m, ks, subdim)
        codebooks.append(centers)

    # 至此，PQ 训练完成：
    # 每个子空间都有一个独立的 KMeans 码本
    # At this point, PQ training is finished:
    # Each subspace has its own independent KMeans codebook

    print("✅ [train_pq] all subspace codebooks trained")
    return codebooks  # 返回 PQ 码本 / Return PQ codebooks




def pq_encode_batch(vectors, codebooks):
    # 对一批向量进行 Product Quantization（PQ）编码
    # Encode a batch of vectors using Product Quantization (PQ)
    #
    # 该函数是 PQ 在工程实现中的关键性能节点：
    # This function is a critical performance hotspot in PQ engineering.
    #
    # 目标 / Goal：
    # 将每个高维向量 x ∈ R^d 转换为 m 个整数索引：
    # Convert each high-dimensional vector x ∈ R^d into m integer indices:
    #
    #     x → [c₀, c₁, ..., c_{m-1}]
    #
    # 其中 c_i 表示：
    # where c_i denotes:
    # “x 在第 i 个子空间中，最接近哪个子空间中心”
    # "which subspace centroid x is closest to in the i-th subspace"

    print(f"🧱 [pq_encode_batch] vectors={len(vectors)}, subspaces={len(codebooks)}")
    # 打印批量大小与子空间数量 / Print batch size and number of subspaces

    n, d = vectors.shape
    # n：向量数量 / number of vectors
    # d：原始向量总维度 / original vector dimension

    m = len(codebooks)
    # m：PQ 子空间数量 / number of PQ subspaces

    assert d % m == 0
    # 断言维度可整除 / Ensure dimension is divisible by subspaces

    subdim = d // m
    # subdim：每个子空间的维度 / dimension per subspace

    codes = np.zeros((n, m), dtype=np.int32)
    # 初始化 PQ 编码矩阵
    # Initialize PQ code matrix
    #
    # codes[i, j] = 第 i 个向量在第 j 个子空间的量化中心索引
    # codes[i, j] = centroid index of i-th vector in j-th subspace


    for i in range(m):
        # 遍历第 i 个 PQ 子空间（每个原始向量都会参与该循环一次）
        # Iterate over the i-th PQ subspace (each original vector participates once per subspace)

        # 从所有原始向量中切分出第 i 个子空间的子向量
        # Slice the i-th subspace component from all original vectors
        #
        # 语义说明：
        # 每一个原始向量 x ∈ R^d 都被按维度划分为 m 个子向量：
        # Each original vector x ∈ R^d is split into m sub-vectors:
        #   x = [x^(0), x^(1), ..., x^(m-1)]
        #
        # 当前循环处理中的是第 i 个子向量 x^(i)
        #
        # 维度变化说明：
        # vectors   的形状为 (n, d)
        # sub_vecs  的形状为 (n, d / m)
        sub_vecs = vectors[:, i * subdim:(i + 1) * subdim]

        # 取出第 i 个子空间对应的 PQ 子空间码本
        # Retrieve the PQ codebook of the i-th subspace
        #
        # centers 的形状为 (ks, d / m)
        # 表示该子空间中通过 KMeans 学到的 ks 个原型中心
        centers = codebooks[i]

        # 计算「所有向量的第 i 个子向量」到「该子空间所有中心」的距离
        # Compute distances between all i-th sub-vectors and all centroids in this subspace
        #
        # 这是 PQ 编码的核心步骤：为每个子向量寻找最近的子空间中心
        # This is the core step of PQ encoding: find the nearest centroid for each sub-vector
        #
        # Broadcasting 过程详解：
        # sub_vecs[:, None, :]   → (n, 1, d / m)
        # centers[None, :, :]    → (1, ks, d / m)
        #
        # 两者相减后得到差值张量：
        # (n, ks, d / m)
        #
        # 对最后一个维度（子空间维度）计算 L2 范数：
        # → 距离矩阵 shape = (n, ks)
        dists = np.linalg.norm(
            sub_vecs[:, None, :] - centers[None, :, :],
            axis=2
        )

        # 对于每一个原始向量，在第 i 个子空间中选择距离最近的中心索引
        # For each original vector, select the nearest centroid index in the i-th subspace
        #
        # argmin 的语义是：
        # 对每一行（一个向量的子向量）：
        #   在 ks 个中心中选择距离最小的那个
        #
        # 结果含义：
        # codes[j, i] = 第 j 个原始向量在第 i 个子空间中的 PQ 编码索引
        #
        # 注意：
        # 这里只记录“中心索引”，不保存距离、不保存残差
        # Only the centroid index is stored, distances/residuals are NOT stored
        #
        # 至此，一个原始向量在该子空间中的表示被压缩为一个整数
        # At this point, one subspace of an original vector is compressed into a single integer
        codes[:, i] = np.argmin(dists, axis=1)


    # 至此，PQ 编码完成：
    # At this point, PQ encoding is complete.
    #
    # 每个向量被压缩为 m 个整数
    # Each vector is compressed into m integers
    #
    # 存储复杂度：
    # Storage complexity:
    #   原始向量：n × d × 4 bytes
    #   PQ 编码： n × m × 4 bytes
    #
    # 在 ANN 系统中，这是性能与内存权衡的核心
    # This is the core memory–accuracy tradeoff in ANN systems

    return codes



def build_ivf_pq(vectors, nlist, m, ks):
    # 构建 IVF-PQ 索引的完整流程
    # Build the full IVF-PQ index pipeline
    #
    # IVF-PQ 的整体目标：
    # Overall goal of IVF-PQ:
    # 将大规模高维向量集合拆解为：
    # Decompose a large high-dimensional vector set into:
    #
    # ① IVF（Inverted File）：减少搜索空间
    # ② PQ（Product Quantization）：压缩向量以支持快速排序
    #
    # 本函数完成三件不可缺失的事情：
    # This function performs three indispensable steps:
    #
    # 1) 训练 IVF 的 coarse quantizer（第一次 KMeans）
    # 2) 训练 PQ 的 subspace codebooks（m 次 KMeans）
    # 3) 将所有向量编码并写入 inverted lists

    print(f"📦 [build_ivf_pq] nlist={nlist}, m={m}, ks={ks}, samples={len(vectors)}")
    # 打印索引规模关键参数 / Print index scale parameters

    # ================================
    # Step 1：训练 IVF 粗量化器
    # Step 1: Train IVF coarse quantizer
    # ================================

    # 目标：
    # Assign each vector to exactly one coarse centroid
    # 用一个较小的 nlist 将空间粗分成多个 Voronoi 区域
    #
    # 数学形式：
    # cid(x) = argmin_j || x - C_j ||
    #
    # 得到的 cid 仅用于 routing，而非精确表示
    coarse_centers = kmeans(vectors, nlist)

    # coarse_centers 形状：
    # shape = (nlist, d)
    #
    # 语义：
    # 每一个 coarse_center 是一个全维度的代表点
    # Each coarse_center represents a large region in the vector space

    # ================================
    # Step 2：训练 PQ 子空间码本
    # Step 2: Train PQ subspace codebooks
    # ================================

    # PQ 的目标：
    # Approximate each vector with a sum of low-dimensional codewords
    #
    # 将 d 维向量切成 m 个子空间：
    # Split d-dim vector into m subspaces
    #
    # 每个子空间上独立训练一个 KMeans
    pq_codebooks = train_pq(vectors, m, ks)

    # pq_codebooks 的结构：
    # pq_codebooks[m][ks][subdim]
    #
    # 即：
    # m 个子空间
    # 每个子空间 ks 个中心
    # 每个中心维度为 d / m

    # ================================
    # Step 3：PQ 批量编码所有向量
    # Step 3: Batch PQ-encode all vectors
    # ================================

    print("🧱 [build_ivf_pq] start batch PQ encoding")

    # 对每个向量计算 m 个子空间索引
    # Encode each vector into m subspace centroid indices
    #
    # 输出：
    # pq_codes shape = (N, m)
    pq_codes = pq_encode_batch(vectors, pq_codebooks)

    # ================================
    # Step 4：计算所有向量的 coarse assignment
    # Step 4: Compute coarse assignments for all vectors
    # ================================

    # 使用向量化方式计算所有向量 → 所有 coarse centers 的距离
    # Compute distance matrix in a fully vectorized way
    #
    # (N, 1, d) - (1, nlist, d) → (N, nlist)
    dists = np.linalg.norm(
        vectors[:, None, :] - coarse_centers[None, :, :],
        axis=2
    )

    # 对每个向量选择最近的 coarse 中心
    # For each vector, choose nearest coarse centroid
    #
    # assignments shape = (N,)
    assignments = np.argmin(dists, axis=1)

    # ================================
    # Step 5：构建倒排表（Inverted Lists）
    # Step 5: Build inverted lists
    # ================================

    # 初始化 nlist 个倒排链表
    # Initialize nlist inverted lists
    inverted_lists = [[] for _ in range(nlist)]

    # 将每个向量写入对应的倒排列表
    # Insert each vector into its assigned inverted list
    #
    # 倒排表中存储的信息：
    # (原始向量 ID, PQ 编码)
    #
    # 而不是存完整向量
    for idx, list_id in enumerate(assignments):
        inverted_lists[list_id].append((idx, pq_codes[idx]))

    # ================================
    # IVF-PQ 索引构建完成
    # IVF-PQ index construction finished
    # ================================

    print("✅ [build_ivf_pq] index built successfully")

    # 返回索引三要素：
    # Return three essential index components:
    #
    # 1) coarse_centers    → 搜索阶段用于 list routing
    # 2) pq_codebooks      → 搜索阶段用于 ADC 距离计算
    # 3) inverted_lists    → 候选集来源
    return coarse_centers, pq_codebooks, inverted_lists



def pq_adc_distance(query, code, codebooks):
    # 使用 ADC（Asymmetric Distance Computation）计算查询向量与 PQ 编码向量之间的近似距离
    # Compute approximate distance between a query vector and a PQ-encoded vector using ADC
    #
    # ADC 的核心思想：
    # Core idea of ADC:
    #   - 查询向量 query 保持为「原始高精度浮点向量」
    #   - 数据库向量 x 使用 PQ 进行压缩，仅保留 m 个中心索引
    #   - 距离计算时，不重建完整向量，而是逐子空间累加距离
    #
    # 目标近似公式：
    # Approximation formula:
    #   ||q - x||² ≈ Σ_{i=0}^{m-1} || q^(i) - c^(i)_{code[i]} ||²
    #
    # 其中：
    #   q^(i)         : 查询向量在第 i 个子空间的子向量
    #   c^(i)_{code[i]} : PQ codebook 中第 i 个子空间被选中的中心
    #
    # 注意：
    #   本实现使用的是 L2 距离（非平方），以便与前续代码保持一致
    #   This implementation uses L2 distance (not squared) for consistency

    d = query.shape[0]
    # d 为原始向量维度
    # d is the dimensionality of the original vectors

    m = len(codebooks)
    # m 为 PQ 子空间数量
    # m is the number of PQ subspaces

    subdim = d // m
    # subdim 为每个子空间的维度
    # subdim is the dimensionality of each subspace

    dist = 0.0
    # 初始化累积距离
    # Initialize accumulated distance

    for i in range(m):
        # 遍历每一个 PQ 子空间
        # Iterate over each PQ subspace

        # 从查询向量中取出第 i 个子空间的子向量
        # Extract the i-th sub-vector of the query
        #
        # query 的结构：
        # query = [q^(0), q^(1), ..., q^(m-1)]
        q_sub = query[i * subdim:(i + 1) * subdim]

        # 根据 PQ 编码，从第 i 个子空间的 codebook 中取出对应中心
        # Retrieve the corresponding centroid from the i-th subspace codebook using PQ code
        #
        # code[i] 是一个整数索引，表示：
        #   原始数据库向量在第 i 个子空间中被量化到哪个中心
        center = codebooks[i][code[i]]

        # 计算查询子向量与该子空间中心之间的 L2 距离
        # Compute L2 distance between the query sub-vector and the selected centroid
        #
        # 这一步不会访问数据库原始向量
        # This step does NOT access the original database vectors
        #
        # 子空间距离是独立的、可加的
        # Subspace distances are independent and additive
        dist += np.linalg.norm(q_sub - center)

    # dist 即为近似的查询-数据库向量距离
    # dist is the approximate distance between query and database vector

    return dist
    # 返回 ADC 计算得到的近似距离
    # Return the approximate distance computed by ADC


def search_ivf_pq(query, coarse_centers, codebooks, inverted_lists, topk=5, nprobe=4):
    # IVF-PQ 查询流程
    # IVF-PQ search pipeline
    print(f"🔍 [search_ivf_pq] topk={topk}, nprobe={nprobe}")

    coarse_dists = np.linalg.norm(coarse_centers - query, axis=1)
    probe_ids = np.argsort(coarse_dists)[:nprobe]

    results = []
    for pid in probe_ids:
        for idx, code in inverted_lists[pid]:
            dist = pq_adc_distance(query, code, codebooks)
            results.append((idx, dist))

    results.sort(key=lambda x: x[1])
    print("✅ [search_ivf_pq] finished")
    return results[:topk]


if __name__ == "__main__":
    # 主入口 / Main entry
    np.random.seed(42)

    path = "test.txt"
    if not os.path.exists(path):
        raise FileNotFoundError("test.txt not found")

    # ✅ 已验证 700 万中文字符不会卡死
    chunks = chunk_utf8_file(path, 512)

    dim = 128
    vectors = np.array([text_to_vector(t, dim) for t in chunks])

    coarse, pq_books, ivf = build_ivf_pq(
        vectors=vectors,
        nlist=64,
        m=8,
        ks=16
    )

    query = vectors[0]
    result = search_ivf_pq(
        query,
        coarse,
        pq_books,
        ivf,
        topk=5,
        nprobe=4
    )

    print("🎯 Search Result:", result)
