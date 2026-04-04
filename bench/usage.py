"""
query 详细演示：batch → 取单球 → 3种query函数逐一展示
"""

import numpy as np
import sphere_exposed_pybind as sep


def get_sphere_data(batch, i):
    """从 batch 结果里切出第 i 个球的数据"""
    arc_mask = batch['arc_sphere_idx'] == i
    cap_mask = batch['cap_sphere_idx'] == i
    pt_mask  = batch['point_sphere_idx'] == i
    return {
        'n_caps'        : int(batch['n_caps'][i]),
        'total_arc'     : float(batch['total_arc'][i]),
        'arc_cap_idx'   : batch['arc_cap_idx'][arc_mask],
        'arc_start'     : batch['arc_start'][arc_mask],
        'arc_end'       : batch['arc_end'][arc_mask],
        'cap_normals'   : batch['cap_normals'][cap_mask],   # (K,3) 法向量
        'cap_d'         : batch['cap_d'][cap_mask],         # (K,)  平面偏移
        'cap_centers'   : batch['cap_centers'][cap_mask],   # (K,3) 边界圆心
        'cap_radii'     : batch['cap_radii'][cap_mask],     # (K,)  边界圆半径
        'cap_u'         : batch['cap_u'][cap_mask],         # (K,3) 边界圆 u 轴
        'cap_v'         : batch['cap_v'][cap_mask],         # (K,3) 边界圆 v 轴
        'exposed_points': batch['point_positions'][pt_mask],# (P,3) 退化点
    }


def demo():
    rng = np.random.default_rng(42)
    N = 80
    centers = rng.uniform(-2, 2, (N, 3)).astype(np.float64)
    radii   = rng.uniform(0.4, 0.9, N).astype(np.float64)

    # 构造 CSR 邻居
    nbr_lists   = [[j for j in range(N) if j != i] for i in range(N)]
    nbr_offsets = np.zeros(N + 1, dtype=np.int64)
    for i, nb in enumerate(nbr_lists):
        nbr_offsets[i + 1] = nbr_offsets[i] + len(nb)
    nbr_indices = np.array([j for nb in nbr_lists for j in nb], dtype=np.int64)

    batch = sep.compute_exposed_batch(centers, radii, nbr_indices, nbr_offsets)

    # 找一个有弧段的球来演示
    target = -1
    for i in range(N):
        if batch['n_caps'][i] > 0 and batch['n_arcs'][i] > 0:
            target = i
            break
    if target < 0:
        print("没有找到有弧段的球，换个随机种子试试")
        return

    sp = get_sphere_data(batch, target)
    c  = centers[target]   # 球心 (3,)
    r  = radii[target]     # 半径 float

    print(f"球 {target}:  球心={np.round(c,3)},  半径={r:.4f}")
    print(f"  n_caps    = {sp['n_caps']}   (有弧段的 cap 数)")
    print(f"  n_arcs    = {len(sp['arc_cap_idx'])}   (弧段总数)")
    print(f"  total_arc = {sp['total_arc']:.4f} rad = {np.degrees(sp['total_arc']):.2f}°")

    # ── cap_u / cap_v 是什么 ─────────────────────────────────────
    print(f"\n── cap 几何 ──")
    print(f"  cap_normals  shape={sp['cap_normals'].shape}  "
          f"  每行是 cap 平面法向量（单位向量，指向邻居球方向）")
    print(f"  cap_centers  shape={sp['cap_centers'].shape}  "
          f"  每行是 cap 边界圆的圆心（3D坐标）")
    print(f"  cap_radii    shape={sp['cap_radii'].shape}  "
          f"  每个值是边界圆半径")
    print(f"  cap_u        shape={sp['cap_u'].shape}  "
          f"  每行是边界圆的 u 轴（单位向量，在圆平面内）")
    print(f"  cap_v        shape={sp['cap_v'].shape}  "
          f"  每行是边界圆的 v 轴（单位向量，= normal × u）")
    print()

    # 验证 u/v 的正交性
    K = sp['n_caps']
    dot_un = np.einsum('ki,ki->k', sp['cap_u'], sp['cap_normals'])  # u·n 应≈0
    dot_vn = np.einsum('ki,ki->k', sp['cap_v'], sp['cap_normals'])  # v·n 应≈0
    dot_uv = np.einsum('ki,ki->k', sp['cap_u'], sp['cap_v'])        # u·v 应≈0
    norm_u = np.linalg.norm(sp['cap_u'], axis=1)                    # |u| 应≈1
    norm_v = np.linalg.norm(sp['cap_v'], axis=1)                    # |v| 应≈1
    print(f"  正交性验证（应全≈0）:")
    print(f"    max|u·n| = {np.abs(dot_un).max():.2e}")
    print(f"    max|v·n| = {np.abs(dot_vn).max():.2e}")
    print(f"    max|u·v| = {np.abs(dot_uv).max():.2e}")
    print(f"    max||u|-1| = {np.abs(norm_u - 1).max():.2e}")
    print(f"    max||v|-1| = {np.abs(norm_v - 1).max():.2e}")

    # 展示第一个 cap 的弧段，手动重建边界圆上的点
    print(f"\n── 弧段详情 ──")
    for a in range(len(sp['arc_cap_idx'])):
        ci  = sp['arc_cap_idx'][a]    # 属于哪个 compacted cap
        t_s = sp['arc_start'][a]
        t_e = sp['arc_end'][a]
        print(f"  arc[{a}]: cap={ci},  "
              f"t=[{t_s:.4f}, {t_e:.4f}]  "
              f"长度={t_e-t_s:.4f} rad = {np.degrees(t_e-t_s):.2f}°")

    # 手动验证 arc[0] 的端点确实在球面上
    a = 0
    ci  = sp['arc_cap_idx'][a]
    t_s = sp['arc_start'][a]
    t_e = sp['arc_end'][a]
    cc  = sp['cap_centers'][ci]
    cr  = sp['cap_radii'][ci]
    cu  = sp['cap_u'][ci]
    cv  = sp['cap_v'][ci]

    # 边界圆参数方程：p(t) = cc + cr*(cos(t)*u + sin(t)*v)
    pt_start = cc + cr * (np.cos(t_s) * cu + np.sin(t_s) * cv)
    pt_end   = cc + cr * (np.cos(t_e) * cu + np.sin(t_e) * cv)
    dist_s   = np.linalg.norm(pt_start - c)
    dist_e   = np.linalg.norm(pt_end   - c)
    print(f"\n  arc[0] 端点到球心距离（应≈球半径 {r:.4f}）:")
    print(f"    start端: {dist_s:.6f}")
    print(f"    end端:   {dist_e:.6f}")

    # ════════════════════════════════════════════════════════════
    # 1. sample_arcs：在 exposed 边界上均匀采样
    # ════════════════════════════════════════════════════════════
    print(f"\n{'═'*55}")
    print("1. sample_arcs")
    print(f"{'═'*55}")

    sample_pts, sample_arc_idx = sep.sample_arcs(
        sp['cap_centers'], sp['cap_radii'],
        sp['cap_u'],       sp['cap_v'],
        sp['arc_cap_idx'], sp['arc_start'], sp['arc_end'],
        n_total_samples=60
    )
    print(f"  采样点数: {len(sample_pts)}")
    print(f"  shape: {sample_pts.shape}")

    # 验证采样点到球心的距离（应≈球半径，因为边界圆在球面上）
    dists_to_center = np.linalg.norm(sample_pts - c, axis=1)
    print(f"  采样点到球心距离: "
          f"min={dists_to_center.min():.4f}  "
          f"max={dists_to_center.max():.4f}  "
          f"(球半径={r:.4f})")

    # 每条弧分配到的采样点数
    print(f"  每条弧的采样点数:")
    for a in range(len(sp['arc_cap_idx'])):
        n_a = int((sample_arc_idx == a).sum())
        arc_len = sp['arc_end'][a] - sp['arc_start'][a]
        print(f"    arc[{a}]: {n_a} pts  (弧长={arc_len:.3f} rad)")

    # ════════════════════════════════════════════════════════════
    # 2. query_inside：判断球面上的点是否在 exposed 区域
    # ════════════════════════════════════════════════════════════
    print(f"\n{'═'*55}")
    print("2. query_inside")
    print(f"{'═'*55}")
    print("  注意: query_inside 需要 all_cap_normals/all_cap_d，")
    print("  这两个字段只有 compute_exposed_single 才输出，")
    print("  batch 版不含（因为 batch 只保留有弧段的 cap）。")
    print("  所以对指定球单独调用一次 single 来获取：")

    # 取出该球的邻居
    nbr = nbr_lists[target]
    nbr_centers = centers[np.array(nbr)]
    nbr_radii   = radii[np.array(nbr)]

    single = sep.compute_exposed_single(
        c, r, nbr_centers, nbr_radii
    )
    # single 含 all_cap_normals (M,3) / all_cap_d (M,) —— M 包含所有cap，含无弧段的

    # 在球面上均匀采样测试点
    test_raw = rng.standard_normal((200, 3))
    test_pts = c + r * test_raw / np.linalg.norm(test_raw, axis=1, keepdims=True)

    inside = sep.query_inside(
        test_pts,
        single['all_cap_normals'],
        single['all_cap_d']
    )
    # inside[i] = True  → 该点不被任何 cap 覆盖 → 在 exposed 区域内
    # inside[i] = False → 被至少一个 cap 覆盖   → 非 exposed

    print(f"  测试 {len(test_pts)} 个球面点:")
    print(f"    exposed (True) : {inside.sum()}")
    print(f"    covered (False): {(~inside).sum()}")
    print(f"    exposed 比例   : {inside.mean()*100:.1f}%")

    # 验证：sample_arcs 采的点应该全在 exposed 区域内
    inside_samples = sep.query_inside(
        sample_pts,
        single['all_cap_normals'],
        single['all_cap_d']
    )
    print(f"\n  验证 sample_arcs 的点是否都在 exposed 内:")
    print(f"    {inside_samples.sum()}/{len(inside_samples)} 个点判为 exposed")
    print(f"    （边界圆上的点恰好在 cap 边界，数值上可能有极少数判为 covered）")

    # ════════════════════════════════════════════════════════════
    # 3. query_closest_on_arcs：查询点到 exposed 边界的最近点
    # ════════════════════════════════════════════════════════════
    print(f"\n{'═'*55}")
    print("3. query_closest_on_arcs")
    print(f"{'═'*55}")

    # 用 covered 的点来查询最近 exposed 边界（最有意义的用法）
    covered_pts = test_pts[~inside]
    if len(covered_pts) == 0:
        print("  (没有 covered 点，用全部测试点)")
        covered_pts = test_pts

    closest, distances, arc_idx = sep.query_closest_on_arcs(
        covered_pts,
        sp['cap_centers'], sp['cap_radii'],
        sp['cap_u'],       sp['cap_v'],
        sp['arc_cap_idx'], sp['arc_start'], sp['arc_end']
    )
    # closest[i]  : covered_pts[i] 在所有 exposed 弧上的最近点 (x,y,z)
    # distances[i]: 到最近点的欧式距离
    # arc_idx[i]  : 最近点在哪条弧上（索引到 arc_cap_idx/arc_start/arc_end）

    print(f"  查询 {len(covered_pts)} 个 covered 点到 exposed 边界的距离:")
    print(f"    min  = {distances.min():.6f}")
    print(f"    mean = {distances.mean():.6f}")
    print(f"    max  = {distances.max():.6f}")
    print(f"  最近点落在哪条弧上（arc index 分布）:")
    unique_arcs, counts = np.unique(arc_idx, return_counts=True)
    for a, cnt in zip(unique_arcs, counts):
        print(f"    arc[{a}]: {cnt} 个查询点")

    # 验证 closest 点确实在边界圆上
    # 对每个结果点，验证它到对应 cap 边界圆心的距离 ≈ cap_radius
    ci_arr = sp['arc_cap_idx'][arc_idx]   # 每个结果对应哪个 cap
    cc_arr = sp['cap_centers'][ci_arr]    # 对应 cap 的圆心
    cr_arr = sp['cap_radii'][ci_arr]      # 对应 cap 的圆半径
    dist_to_circle_center = np.linalg.norm(closest - cc_arr, axis=1)
    err = np.abs(dist_to_circle_center - cr_arr)
    print(f"\n  验证 closest 点在边界圆上（dist_to_cc ≈ cap_radius）:")
    print(f"    max误差 = {err.max():.2e}  mean误差 = {err.mean():.2e}")


if __name__ == '__main__':
    demo()