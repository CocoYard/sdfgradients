"""
sphere_exposed_pybind — Python 端用法示例
==========================================

模块共4个函数：
  1. compute_exposed_single  — 单个球的 exposed 区域计算
  2. compute_exposed_batch   — 批量球（CSR 邻居输入）
  3. query_inside            — 判断点是否在 exposed 区域内
  4. query_closest_on_arcs   — 查询点到 exposed 边界的最近点
  5. sample_arcs             — 在 exposed 边界上采样点

编译后 import：
  import sphere_exposed_pybind as sep
"""

import numpy as np
import sphere_exposed_pybind as sep   # 改成你的模块名


# ══════════════════════════════════════════════════════════════════
# 1. compute_exposed_single
# ══════════════════════════════════════════════════════════════════
#
# 输入：
#   center        (3,)   float64   主球球心
#   radius        float            主球半径
#   other_centers (N,3)  float64   邻居球球心
#   other_radii   (N,)   float64   邻居球半径
#   tol           float  =1e-8     数值容差（平行 cap 判断、区间跳过）
#   degen_tol     float  =1e-6     退化点检测阈值（total_arc < degen_tol 时触发）
#   merge_tol     float  =1e-12    区间合并容差
#
# 输出：dict，key 如下：
#   arcs_by_cap      dict  {int: [(start,end),...]}
#                          key = compacted cap 编号 0..K-1
#                          value = 该 cap 上的 exposed 弧段列表（角度，单位 rad）
#   exposed_points   list  [array(3,), ...]
#                          退化 exposed 点（total_arc≈0 时才有）
#   total_arc        float 所有弧段角度之和（衡量 exposed 区域大小）
#   n_caps           int   有弧段的 cap 数（compacted）
#   --- 以下供 query 函数使用 ---
#   cap_normals      (K,3) float64   compacted cap 法向量
#   cap_d            (K,)  float64   compacted cap 平面偏移 d（dot(n,x)=d 为边界）
#   cap_centers      (K,3) float64   cap 边界圆心
#   cap_radii        (K,)  float64   cap 边界圆半径
#   cap_u            (K,3) float64   边界圆局部 u 轴
#   cap_v            (K,3) float64   边界圆局部 v 轴
#   all_cap_normals  (M,3) float64   所有 cap 法向量（含无弧段的，用于 query_inside）
#   all_cap_d        (M,)  float64   所有 cap 偏移
#   arc_cap_idx      (A,)  int32     每条弧属于哪个 compacted cap
#   arc_start        (A,)  float64   弧起始角（rad, in [0, 2π]）
#   arc_end          (A,)  float64   弧终止角（rad）

def example_single():
    print("=" * 60)
    print("1. compute_exposed_single")
    print("=" * 60)

    # 主球
    center = np.array([0.0, 0.0, 0.0])
    radius = 1.0

    # 6 个邻居球，从 ±x ±y ±z 方向各贴一个
    other_centers = np.array([
        [ 1.5,  0.0,  0.0],
        [-1.5,  0.0,  0.0],
        [ 0.0,  1.5,  0.0],
        [ 0.0, -1.5,  0.0],
        [ 0.0,  0.0,  1.5],
        [ 0.0,  0.0, -1.5],
    ], dtype=np.float64)
    other_radii = np.full(6, 0.8, dtype=np.float64)

    result = sep.compute_exposed_single(
        center, radius,
        other_centers, other_radii,
        tol=1e-8, degen_tol=1e-6, merge_tol=1e-12
    )

    print(f"n_caps     : {result['n_caps']}")
    print(f"total_arc  : {result['total_arc']:.6f} rad")
    print(f"arcs_by_cap:")
    for cap_id, arcs in result['arcs_by_cap'].items():
        for s, e in arcs:
            print(f"  cap {cap_id}: [{s:.4f}, {e:.4f}]  len={e-s:.4f}")
    print(f"exposed_points: {len(result['exposed_points'])} pts")

    print(f"cap_normals shape : {result['cap_normals'].shape}")   # (K,3)
    print(f"arc_cap_idx shape : {result['arc_cap_idx'].shape}")   # (A,)
    print(f"arc_start   shape : {result['arc_start'].shape}")     # (A,)

    return result


# ══════════════════════════════════════════════════════════════════
# 2. compute_exposed_batch
# ══════════════════════════════════════════════════════════════════
#
# 输入：
#   centers      (N,3)  float64   所有球球心
#   radii        (N,)   float64   所有球半径
#   nbr_indices  (E,)   int64     CSR 邻居索引（所有球的邻居展平）
#   nbr_offsets  (N+1,) int64     CSR 偏移，第 i 球的邻居在 nbr_indices[off[i]:off[i+1]]
#   tol / degen_tol / merge_tol   同 single
#
# 输出：dict，key 如下：
#   n_caps           (N,)  int32   每球的 compacted cap 数
#   n_arcs           (N,)  int32   每球的弧段数
#   n_points         (N,)  int32   每球的退化点数
#   total_arc        (N,)  float64 每球的 total_arc
#   arc_sphere_idx   (A,)  int32   每条弧属于哪个球
#   arc_cap_idx      (A,)  int32   每条弧属于哪个 compacted cap（per-sphere）
#   arc_start        (A,)  float64
#   arc_end          (A,)  float64
#   point_sphere_idx (P,)  int32   退化点属于哪个球
#   point_positions  (P,3) float64 退化点坐标
#   cap_sphere_idx   (C,)  int32   每条 cap 记录属于哪个球
#   cap_id           (C,)  int32   per-sphere compacted cap 编号
#   cap_normals      (C,3) float64
#   cap_d            (C,)  float64
#   cap_centers      (C,3) float64
#   cap_radii        (C,)  float64
#   cap_u            (C,3) float64
#   cap_v            (C,3) float64

def example_batch():
    print("\n" + "=" * 60)
    print("2. compute_exposed_batch")
    print("=" * 60)

    rng = np.random.default_rng(42)
    N = 20
    centers = rng.uniform(-3, 3, (N, 3)).astype(np.float64)
    radii   = rng.uniform(0.3, 1.0, N).astype(np.float64)

    # 构造 CSR 邻居：简单用距离阈值找邻居
    threshold = 2.5
    nbr_lists = []
    for i in range(N):
        nbrs = []
        for j in range(N):
            if i != j:
                dist = np.linalg.norm(centers[i] - centers[j])
                if dist < radii[i] + radii[j] + threshold:
                    nbrs.append(j)
        nbr_lists.append(nbrs)

    nbr_offsets = np.zeros(N + 1, dtype=np.int64)
    for i, nbrs in enumerate(nbr_lists):
        nbr_offsets[i+1] = nbr_offsets[i] + len(nbrs)
    nbr_indices = np.concatenate([np.array(nb, dtype=np.int64) for nb in nbr_lists]
                                  if any(nbr_lists) else [np.empty(0, dtype=np.int64)])

    result = sep.compute_exposed_batch(
        centers, radii, nbr_indices, nbr_offsets,
        tol=1e-8, degen_tol=1e-6, merge_tol=1e-12
    )

    print(f"total arcs   : {len(result['arc_start'])}")
    print(f"total caps   : {len(result['cap_d'])}")
    print(f"total degen  : {len(result['point_positions'])}")
    print(f"n_arcs  (per sphere): {result['n_arcs']}")
    print(f"total_arc (per sphere): {np.round(result['total_arc'], 3)}")

    # 如何按球拆分弧段（利用 arc_sphere_idx）
    arc_si = result['arc_sphere_idx']    # (A,) 告诉你每条弧属于哪个球
    arc_s  = result['arc_start']
    arc_e  = result['arc_end']
    for i in [0, 1, 2]:
        mask = arc_si == i
        print(f"  sphere {i}: {mask.sum()} arcs, "
              f"total_arc={result['total_arc'][i]:.4f}")

    return result


# ══════════════════════════════════════════════════════════════════
# 3. query_inside
# ══════════════════════════════════════════════════════════════════
#
# 输入：
#   points           (N,3)  float64   待查询点（应在球面上或附近）
#   all_cap_normals  (M,3)  float64   来自 compute_exposed_single 的 all_cap_normals
#   all_cap_d        (M,)   float64   来自 compute_exposed_single 的 all_cap_d
#
# 输出：
#   inside           (N,)   bool
#                    True  = 该点在 exposed 区域内（不被任何 cap 覆盖）
#                    False = 该点被至少一个 cap 覆盖（非 exposed）

def example_query_inside(result_single, center, radius):
    print("\n" + "=" * 60)
    print("3. query_inside")
    print("=" * 60)

    # 在球面上均匀采样一些点
    rng = np.random.default_rng(0)
    pts_raw = rng.standard_normal((200, 3))
    pts = center + radius * pts_raw / np.linalg.norm(pts_raw, axis=1, keepdims=True)

    inside = sep.query_inside(
        pts.astype(np.float64),
        result_single['all_cap_normals'],
        result_single['all_cap_d']
    )

    print(f"查询 {len(pts)} 个点")
    print(f"exposed 区域内: {inside.sum()} 个")
    print(f"被 cap 覆盖  : {(~inside).sum()} 个")
    return pts, inside


# ══════════════════════════════════════════════════════════════════
# 4. query_closest_on_arcs
# ══════════════════════════════════════════════════════════════════
#
# 输入：
#   points        (N,3)  float64
#   sphere_center (3,)   float64   主球球心
#   sphere_radius float             主球半径
#   cap_centers   (K,3)  float64   来自 compute_exposed_single 的 cap_centers
#   cap_radii     (K,)   float64
#   cap_u         (K,3)  float64
#   cap_v         (K,3)  float64
#   arc_cap_idx   (A,)   int32     来自 compute_exposed_single 的 arc_cap_idx
#   arc_start     (A,)   float64
#   arc_end       (A,)   float64
#
# 输出：tuple(closest, distances, arc_indices)
#   closest       (N,3)  float64   每个查询点在 exposed 边界弧上的最近点
#   distances     (N,)   float64   欧式距离
#   arc_indices   (N,)   int32     最近点在哪条弧上（索引到 arc_cap_idx）

def example_query_closest(result_single, center, radius):
    print("\n" + "=" * 60)
    print("4. query_closest_on_arcs")
    print("=" * 60)

    if result_single['n_caps'] == 0:
        print("  (no caps, skip)")
        return

    # 取一些在球面上的点
    rng = np.random.default_rng(1)
    pts_raw = rng.standard_normal((50, 3))
    pts = center + radius * pts_raw / np.linalg.norm(pts_raw, axis=1, keepdims=True)

    closest, distances, arc_idx = sep.query_closest_on_arcs(
        pts.astype(np.float64),
        np.asarray(center, dtype=np.float64),
        float(radius),
        result_single['cap_centers'],
        result_single['cap_radii'],
        result_single['cap_u'],
        result_single['cap_v'],
        result_single['arc_cap_idx'],
        result_single['arc_start'],
        result_single['arc_end']
    )

    print(f"查询 {len(pts)} 个点")
    print(f"最近点距离 min={distances.min():.6f}  max={distances.max():.6f}  "
          f"mean={distances.mean():.6f}")
    print(f"arc_indices 范围: [{arc_idx.min()}, {arc_idx.max()}]")
    # 验证最近点在球面附近（它们在 cap 的边界圆上，不一定在主球面上）
    print(f"closest shape: {closest.shape}")


# ══════════════════════════════════════════════════════════════════
# 5. sample_arcs
# ══════════════════════════════════════════════════════════════════
#
# 输入：
#   sphere_center    (3,)  float64
#   sphere_radius    float
#   cap_centers      (K,3) float64   来自 compute_exposed_single
#   cap_radii        (K,)  float64
#   cap_u            (K,3) float64
#   cap_v            (K,3) float64
#   arc_cap_idx      (A,)  int32
#   arc_start        (A,)  float64
#   arc_end          (A,)  float64
#   n_total_samples  int              总采样数（≥ n_arcs，每条弧至少1个点）
#
# 输出：tuple(points, arc_indices)
#   points       (S,3) float64   采样点坐标（在各 cap 边界圆上）
#   arc_indices  (S,)  int32     每个采样点属于哪条弧

def example_sample_arcs(result_single, center, radius):
    print("\n" + "=" * 60)
    print("5. sample_arcs")
    print("=" * 60)

    if result_single['n_caps'] == 0:
        print("  (no caps, skip)")
        return

    points, arc_idx = sep.sample_arcs(
        np.asarray(center, dtype=np.float64),
        float(radius),
        result_single['cap_centers'],
        result_single['cap_radii'],
        result_single['cap_u'],
        result_single['cap_v'],
        result_single['arc_cap_idx'],
        result_single['arc_start'],
        result_single['arc_end'],
        200   # n_total_samples
    )

    print(f"采样点数: {len(points)}")
    print(f"points shape : {points.shape}")          # (S, 3)
    print(f"arc_indices  : {np.unique(arc_idx)}")    # 每条弧都有点
    # 采样点分布（每条弧分配到的点数）
    for a in np.unique(arc_idx):
        n = (arc_idx == a).sum()
        arc_len = result_single['arc_end'][a] - result_single['arc_start'][a]
        print(f"  arc {a}: {n} pts, arc_len={arc_len:.4f} rad")


# ══════════════════════════════════════════════════════════════════
# 典型 pipeline：batch 计算 → 按球重建 single 结构 → query
# ══════════════════════════════════════════════════════════════════

def example_batch_then_query():
    """
    batch 输出是展平的，这里演示如何针对某一个球
    重建出 single-style 的 dict，再调用 query 函数。
    """
    print("\n" + "=" * 60)
    print("Pipeline: batch → pick sphere i → query")
    print("=" * 60)

    rng = np.random.default_rng(2)
    N = 100
    centers = rng.uniform(-2, 2, (N, 3)).astype(np.float64)
    radii   = rng.uniform(0.4, 0.9, N).astype(np.float64)

    # CSR 邻居（全连接，简单示例）
    nbr_lists = [[j for j in range(N) if j != i] for i in range(N)]
    nbr_offsets = np.zeros(N+1, dtype=np.int64)
    for i, nb in enumerate(nbr_lists):
        nbr_offsets[i+1] = nbr_offsets[i] + len(nb)
    nbr_indices = np.array([j for nb in nbr_lists for j in nb], dtype=np.int64)

    batch = sep.compute_exposed_batch(
        centers, radii, nbr_indices, nbr_offsets)

    # ── 针对球 i=3，重建 single-style 结构 ──────────────────────
    i = 3
    arc_mask = batch['arc_sphere_idx'] == i
    cap_mask  = batch['cap_sphere_idx'] == i

    single_like = {
        'n_caps'         : int(batch['n_caps'][i]),
        'total_arc'      : float(batch['total_arc'][i]),
        'arc_cap_idx'    : batch['arc_cap_idx'][arc_mask],
        'arc_start'      : batch['arc_start'][arc_mask],
        'arc_end'        : batch['arc_end'][arc_mask],
        'cap_centers'    : batch['cap_centers'][cap_mask],
        'cap_radii'      : batch['cap_radii'][cap_mask],
        'cap_u'          : batch['cap_u'][cap_mask],
        'cap_v'          : batch['cap_v'][cap_mask],
        # batch 没有 all_cap_normals/d，如需 query_inside 要再调用 single
    }

    print(f"sphere {i}: n_caps={single_like['n_caps']}, "
          f"total_arc={single_like['total_arc']:.4f}, "
          f"n_arcs={arc_mask.sum()}")

    if single_like['n_caps'] > 0 and arc_mask.sum() > 0:
        pts_raw = rng.standard_normal((30, 3))
        pts = centers[i] + radii[i] * pts_raw / np.linalg.norm(pts_raw, axis=1, keepdims=True)

        pts_s, idx_s = sep.sample_arcs(
            centers[i], radii[i],
            single_like['cap_centers'],
            single_like['cap_radii'],
            single_like['cap_u'],
            single_like['cap_v'],
            single_like['arc_cap_idx'],
            single_like['arc_start'],
            single_like['arc_end'],
            100
        )
        print(f"  sampled {len(pts_s)} points on exposed boundary of sphere {i}")


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    center = np.array([0.0, 0.0, 0.0])
    radius = 1.0

    r_single = example_single()
    # example_batch()
    # example_query_inside(r_single, center, radius)
    # example_query_closest(r_single, center, radius)
    # example_sample_arcs(r_single, center, radius)
    example_batch_then_query()