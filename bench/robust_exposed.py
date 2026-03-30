"""
Robust computation of exposed regions on a sphere arrangement,
including degenerate cases where the exposed region is an isolated point.

Approach:
  Phase 1: Standard arc computation (existing code).
  Phase 2: If total arc length ≈ 0, find candidate exposed points by
           intersecting all pairs of cap boundary circles and testing
           each intersection point against all other caps.
  Phase 3: Deduplicate candidate points.
  Phase 4: Fallback — linear program to find a feasible point on the
           sphere satisfying all cap constraints.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set

try:
    from scipy.optimize import linprog as _scipy_linprog
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ═══════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Sphere:
    center: np.ndarray
    radius: float
    def __post_init__(self):
        self.center = np.asarray(self.center, dtype=float)

@dataclass
class Cap:
    normal: np.ndarray
    d: float
    circle_center: np.ndarray
    circle_radius: float
    local_u: np.ndarray
    local_v: np.ndarray
    phi: float
    sphere_idx: int

@dataclass
class ExposedPoint:
    """An isolated exposed point on the sphere surface."""
    position: np.ndarray
    on_cap_boundaries: List[int] = field(default_factory=list)

@dataclass
class OrientedLine:
    a: float; b: float; c: float; source_cap_idx: int


# ═══════════════════════════════════════════════════════════════════
# Geometry helpers
# ═══════════════════════════════════════════════════════════════════

def perpendicular_unit(n):
    if abs(n[0]) < 0.9:
        v = np.cross(n, np.array([1.0, 0.0, 0.0]))
    else:
        v = np.cross(n, np.array([0.0, 1.0, 0.0]))
    return v / np.linalg.norm(v)

def compute_cap(main, other, sphere_idx):
    diff = other.center - main.center
    dist = np.linalg.norm(diff)
    if dist < 1e-14:
        if other.radius >= main.radius:
            n = np.array([1.0, 0.0, 0.0])
            return Cap(n, np.dot(n, main.center) - main.radius - 1,
                       main.center, 0.0, np.zeros(3), np.zeros(3), np.pi, sphere_idx)
        return None
    if dist >= main.radius + other.radius:
        return None
    if dist + other.radius <= main.radius:
        return None
    if dist + main.radius <= other.radius:
        n = diff / dist
        return Cap(n, np.dot(n, main.center) - main.radius - 1,
                   main.center, 0.0, np.zeros(3), np.zeros(3), np.pi, sphere_idx)
    n = diff / dist
    h = (main.radius**2 - other.radius**2 + dist**2) / (2 * dist)
    d_plane = np.dot(n, main.center) + h
    circle_center = main.center + h * n
    circle_r = np.sqrt(max(0.0, main.radius**2 - h**2))
    phi = np.arccos(np.clip(h / main.radius, -1.0, 1.0))
    u = perpendicular_unit(n)
    v = np.cross(n, u)
    return Cap(n, d_plane, circle_center, circle_r, u, v, phi, sphere_idx)

def compute_all_caps(main, others):
    caps = []
    for i, other in enumerate(others):
        cap = compute_cap(main, other, sphere_idx=i)
        if cap is not None:
            duplicate = False
            for e_idx in range(len(caps) - 1, -1, -1):
                existing = caps[e_idx]
                dot = np.dot(cap.normal, existing.normal)
                if dot > 1 - 1e-8:
                    if abs(cap.d - existing.d) < 1e-8:
                        duplicate = True; break
                    elif cap.d > existing.d:
                        duplicate = True; break
                    else:
                        caps.pop(e_idx); break
            if not duplicate:
                caps.append(cap)
    return caps

def intersect_plane_with_cap_plane(host_cap, other_cap, other_cap_idx):
    a = -np.dot(other_cap.normal, host_cap.local_u)
    b = -np.dot(other_cap.normal, host_cap.local_v)
    c = -(other_cap.d - np.dot(other_cap.normal, host_cap.circle_center))
    if abs(a) < 1e-14 and abs(b) < 1e-14:
        return None
    return OrientedLine(a=a, b=b, c=c, source_cap_idx=other_cap_idx)

def compute_oriented_lines_for_cap(host_idx, caps):
    host = caps[host_idx]
    lines = []
    for j, other in enumerate(caps):
        if j == host_idx:
            continue
        line = intersect_plane_with_cap_plane(host, other, other_cap_idx=j)
        if line is None:
            c_orig = np.dot(other.normal, host.circle_center) - other.d
            if c_orig > 1e-10:
                return [], False
            continue
        lines.append(line)
    return lines, True

def _intersect_interval_lists(list_a, list_b):
    result = []
    for a_s, a_e in list_a:
        for b_s, b_e in list_b:
            lo = max(a_s, b_s)
            hi = min(a_e, b_e)
            if hi - lo > 1e-12:
                result.append((lo, hi))
    result.sort()
    merged = []
    for s, e in result:
        if merged and s <= merged[-1][1] + 1e-12:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged

def find_arc_in_half_planes(circle_radius, lines):
    R = circle_radius
    if R < 1e-14:
        return []
    intervals = [(0.0, 2 * np.pi)]
    for line in lines:
        a, b, c = line.a, line.b, line.c
        A_line = np.sqrt(a**2 + b**2)
        if A_line < 1e-14:
            if c > 1e-10:
                return []
            continue
        ratio = c / (R * A_line)
        if ratio <= -1.0 - 1e-10:
            continue
        if ratio >= 1.0 + 1e-10:
            return []
        ratio = np.clip(ratio, -1.0, 1.0)
        alpha = np.arctan2(b, a)
        delta = np.arccos(ratio)
        arc_start = (alpha - delta) % (2 * np.pi)
        arc_end = (alpha + delta) % (2 * np.pi)
        if arc_start < arc_end:
            constraint_intervals = [(arc_start, arc_end)]
        else:
            constraint_intervals = [(arc_start, 2 * np.pi), (0.0, arc_end)]
        intervals = _intersect_interval_lists(intervals, constraint_intervals)
        if not intervals:
            return []
    return intervals


# ═══════════════════════════════════════════════════════════════════
# Phase 2: Degenerate point detection
# ═══════════════════════════════════════════════════════════════════

def _find_cap_circle_intersections(main, caps):
    """Find all points where pairs of cap boundary circles intersect on the sphere."""
    results = []
    R = main.radius
    c0 = main.center

    for i in range(len(caps)):
        if caps[i].circle_radius < 1e-14:
            continue
        for j in range(i + 1, len(caps)):
            if caps[j].circle_radius < 1e-14:
                continue

            ni, di = caps[i].normal, caps[i].d
            nj, dj = caps[j].normal, caps[j].d

            line_dir = np.cross(ni, nj)
            ld_norm = np.linalg.norm(line_dir)
            if ld_norm < 1e-12:
                continue
            line_dir /= ld_norm

            A = np.array([ni, nj, line_dir])
            b_vec = np.array([di, dj, 0.0])
            try:
                p0 = np.linalg.solve(A, b_vec)
            except np.linalg.LinAlgError:
                continue

            delta = p0 - c0
            b_coeff = 2.0 * np.dot(delta, line_dir)
            c_coeff = np.dot(delta, delta) - R * R
            disc = b_coeff**2 - 4 * c_coeff

            if disc < -1e-10:
                continue
            disc = max(disc, 0.0)
            sqrt_disc = np.sqrt(disc)

            for sign in [-1, 1]:
                t = (-b_coeff + sign * sqrt_disc) / 2.0
                pt = p0 + t * line_dir
                if abs(np.linalg.norm(pt - c0) - R) < 1e-6:
                    results.append((pt, i, j))

    return results


def _is_point_exposed(pt, caps, exclude, atol):
    """Check if pt is not covered by any cap outside the exclude set."""
    for k, cap in enumerate(caps):
        if k in exclude:
            continue
        if np.dot(cap.normal, pt) - cap.d > atol:
            return False
    return True


def _deduplicate_points(points, merge_tol=1e-6):
    if not points:
        return []
    unique = [points[0].copy()]
    for pt in points[1:]:
        if all(np.linalg.norm(pt - u) >= merge_tol for u in unique):
            unique.append(pt.copy())
    return unique


def _find_exposed_point_lp(main, caps, atol=1e-8):
    """
    Find an exposed point via LP relaxation.

    We want p = c0 + R*d on the sphere with n_k · p < d_k for all k.
    Substituting: R*(n_k · d) < d_k - n_k · c0.
    Relax ||d||=1 to ||d||<=1 (box bounds), solve LP to maximize
    the minimum slack, then project back onto the unit sphere.
    """
    if not _HAS_SCIPY:
        return _find_exposed_point_subgradient(main, caps, atol)

    R = main.radius
    c0 = main.center
    if len(caps) == 0:
        return c0 + np.array([R, 0.0, 0.0])

    # x = [d_x, d_y, d_z, s], maximize s subject to:
    #   R * n_k · d + s <= d_k - n_k · c0
    A_ub = np.zeros((len(caps), 4))
    b_ub = np.zeros(len(caps))
    for k, cap in enumerate(caps):
        A_ub[k, :3] = R * cap.normal
        A_ub[k, 3] = 1.0
        b_ub[k] = cap.d - np.dot(cap.normal, c0)

    c_obj = np.array([0.0, 0.0, 0.0, -1.0])
    bounds = [(-1, 1), (-1, 1), (-1, 1), (None, None)]
    result = _scipy_linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

    if result.success and result.x[3] > -atol:
        d = result.x[:3]
        d_norm = np.linalg.norm(d)
        if d_norm < 1e-14:
            return None
        d /= d_norm
        pt = c0 + R * d
        if all(np.dot(cap.normal, pt) - cap.d < atol for cap in caps):
            return pt
    return None


def _find_exposed_point_subgradient(main, caps, atol=1e-8):
    """Subgradient descent fallback."""
    R = main.radius
    c0 = main.center
    if len(caps) == 0:
        return c0 + np.array([R, 0.0, 0.0])

    mean_n = np.mean([cap.normal for cap in caps], axis=0)
    mn = np.linalg.norm(mean_n)
    direction = -mean_n / mn if mn > 1e-14 else np.array([1.0, 0.0, 0.0])
    p = c0 + R * direction
    best_p, best_val = p.copy(), float('inf')

    for it in range(500):
        violations = [np.dot(cap.normal, p) - cap.d for cap in caps]
        max_val = max(violations)
        if max_val < best_val:
            best_val = max_val; best_p = p.copy()
        if max_val < -atol:
            return p

        k = violations.index(max_val)
        grad = caps[k].normal
        n_p = (p - c0) / R
        gt = grad - np.dot(grad, n_p) * n_p
        gtn = np.linalg.norm(gt)
        if gtn < 1e-14:
            perturb = np.random.randn(3)
            perturb -= np.dot(perturb, n_p) * n_p
            pn = np.linalg.norm(perturb)
            if pn > 1e-14:
                gt = perturb / pn; gtn = 1.0
            else:
                break
        step = R * 0.3 / (1 + it * 0.05)
        p = p - step * (gt / gtn)
        p = c0 + R * (p - c0) / np.linalg.norm(p - c0)

    return best_p if best_val < atol else None


# ═══════════════════════════════════════════════════════════════════
# Main robust function
# ═══════════════════════════════════════════════════════════════════

def compute_exposed_arcs_robust(
    main: Sphere,
    caps: List[Cap],
    atol: float = 1e-8,
    point_merge_tol: float = 1e-6,
) -> Tuple[Dict[int, List[Tuple[float, float]]], List[ExposedPoint]]:
    """
    Compute exposed arcs + isolated exposed points.

    Returns
    -------
    arcs_by_cap : dict  {cap_idx: [(t_start, t_end), ...]}
    exposed_points : list of ExposedPoint
    """
    # Phase 1: standard arcs
    arcs_by_cap = {}
    for i, cap in enumerate(caps):
        if cap.phi >= np.pi - 1e-10 or cap.circle_radius < 1e-14:
            arcs_by_cap[i] = []
            continue
        lines, feasible = compute_oriented_lines_for_cap(i, caps)
        if not feasible:
            arcs_by_cap[i] = []
            continue
        arcs_by_cap[i] = find_arc_in_half_planes(cap.circle_radius, lines)

    total_arc = sum(sum(te - ts for ts, te in a) for a in arcs_by_cap.values())
    exposed_points = []
    if np.abs(main.center[0] - 0.46666667) < 1e-6 and np.abs(main.center[1] - 0.37918178) < 1e-6 and np.abs(main.center[2] - 0.46325978) < 1e-6:
        print(f"Debug: Total arc length = {total_arc:.2e}, caps={len(caps)}")
    # Phase 2: degenerate point detection
    # Skip if no caps at all — entire sphere is exposed, nothing degenerate.
    if total_arc < 1e-6 and len(caps) > 0:
        # Quick feasibility check: is there ANY exposed point on the sphere?

        candidates = _find_cap_circle_intersections(main, caps)
        raw_exposed = []
        for pt, ci, cj in candidates:
            if _is_point_exposed(pt, caps, exclude={ci, cj}, atol=atol):
                raw_exposed.append(pt)

        unique_pts = _deduplicate_points(raw_exposed, merge_tol=point_merge_tol)

        for pt in unique_pts:
            on_caps = [k for k, cap in enumerate(caps)
                        if abs(np.dot(cap.normal, pt) - cap.d) < point_merge_tol]
            exposed_points.append(ExposedPoint(position=pt, on_cap_boundaries=on_caps))

        # Phase 3: LP already confirmed feasibility, so if circle
        # intersections missed it, find the point via LP projection
        if not exposed_points:
            pt_opt = _find_exposed_point_lp(main, caps, atol=atol)
            if pt_opt is not None:
                exposed_points.append(ExposedPoint(position=pt_opt, on_cap_boundaries=[]))
    return arcs_by_cap, exposed_points


# ═══════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════

def test_four_spheres_two_points():
    """4 spheres covering all of main sphere except (±1,0,0)."""
    print("=" * 60)
    print("Test 1: 4 spheres → 2 exposed points at (±1,0,0)")
    print("=" * 60)

    main = Sphere([0, 0, 0], 1.0)
    others = [
        Sphere([0, 0, 1], np.sqrt(2)),
        Sphere([0, 0, -1], np.sqrt(2)),
        Sphere([0, -1, 0], np.sqrt(2)),
        Sphere([0, 1, 0], np.sqrt(2)),
    ]
    caps = compute_all_caps(main, others)
    arcs, points = compute_exposed_arcs_robust(main, caps)

    total_arc = sum(sum(te - ts for ts, te in a) for a in arcs.values())
    print(f"  Total arc: {total_arc:.2e} rad")
    print(f"  Exposed points: {len(points)}")
    for p in points:
        print(f"    pos={np.round(p.position, 8)}, on boundaries={p.on_cap_boundaries}")

    assert len(points) == 2, f"Expected 2, got {len(points)}"
    pts_sorted = sorted(points, key=lambda p: p.position[0])
    np.testing.assert_allclose(pts_sorted[0].position, [-1, 0, 0], atol=1e-6)
    np.testing.assert_allclose(pts_sorted[1].position, [1, 0, 0], atol=1e-6)
    print("  PASSED ✓\n")


def test_five_spheres_one_point():
    """5 spheres leaving only (1,0,0) exposed."""
    print("=" * 60)
    print("Test 2: 5 spheres → 1 exposed point at (1,0,0)")
    print("=" * 60)

    main = Sphere([0, 0, 0], 1.0)
    others = [
        Sphere([0, 0, 1], np.sqrt(2)),
        Sphere([0, 0, -1], np.sqrt(2)),
        Sphere([0, -1, 0], np.sqrt(2)),
        Sphere([0, 1, 0], np.sqrt(2)),
        Sphere([-1, 0, 0], np.sqrt(2)),
    ]
    caps = compute_all_caps(main, others)
    arcs, points = compute_exposed_arcs_robust(main, caps)

    total_arc = sum(sum(te - ts for ts, te in a) for a in arcs.values())
    print(f"  Total arc: {total_arc:.2e} rad")
    print(f"  Exposed points: {len(points)}")
    for p in points:
        print(f"    pos={np.round(p.position, 8)}, on boundaries={p.on_cap_boundaries}")

    assert len(points) == 1, f"Expected 1, got {len(points)}"
    np.testing.assert_allclose(points[0].position, [1, 0, 0], atol=1e-6)
    print("  PASSED ✓\n")


def test_normal_arcs():
    """Normal case: arcs exist, no degenerate points."""
    print("=" * 60)
    print("Test 3: 2 spheres → normal arcs, 0 degenerate points")
    print("=" * 60)

    main = Sphere([0, 0, 0], 1.0)
    others = [Sphere([1.5, 0, 0], 1.0), Sphere([-1.5, 0, 0], 1.0)]
    caps = compute_all_caps(main, others)
    arcs, points = compute_exposed_arcs_robust(main, caps)

    total_arc = sum(sum(te - ts for ts, te in a) for a in arcs.values())
    print(f"  Total arc: {total_arc:.4f} rad")
    print(f"  Exposed points: {len(points)}")
    assert total_arc > 0.1
    assert len(points) == 0
    print("  PASSED ✓\n")


def test_offset_center():
    """Same as test 1 but with non-origin center."""
    print("=" * 60)
    print("Test 4: Non-origin center, 4 spheres → 2 points")
    print("=" * 60)

    c = np.array([3.0, -2.0, 7.0])
    main = Sphere(c, 1.0)
    others = [
        Sphere(c + [0, 0, 1], np.sqrt(2)),
        Sphere(c + [0, 0, -1], np.sqrt(2)),
        Sphere(c + [0, -1, 0], np.sqrt(2)),
        Sphere(c + [0, 1, 0], np.sqrt(2)),
    ]
    caps = compute_all_caps(main, others)
    arcs, points = compute_exposed_arcs_robust(main, caps)

    print(f"  Exposed points: {len(points)}")
    for p in points:
        print(f"    pos={np.round(p.position, 8)}")

    assert len(points) == 2
    pts_sorted = sorted(points, key=lambda p: p.position[0])
    np.testing.assert_allclose(pts_sorted[0].position, c + [-1, 0, 0], atol=1e-6)
    np.testing.assert_allclose(pts_sorted[1].position, c + [1, 0, 0], atol=1e-6)
    print("  PASSED ✓\n")


def test_fully_covered():
    """6 spheres fully covering the main sphere → no exposed region at all."""
    print("=" * 60)
    print("Test 5: Fully covered sphere → nothing exposed")
    print("=" * 60)

    main = Sphere([0, 0, 0], 1.0)
    others = [
        Sphere([0, 0, 1], np.sqrt(2)),
        Sphere([0, 0, -1], np.sqrt(2)),
        Sphere([0, -1, 0], np.sqrt(2)),
        Sphere([0, 1, 0], np.sqrt(2)),
        Sphere([1, 0, 0], np.sqrt(2)),
        Sphere([-1, 0, 0], np.sqrt(2)),
    ]
    caps = compute_all_caps(main, others)
    arcs, points = compute_exposed_arcs_robust(main, caps)

    total_arc = sum(sum(te - ts for ts, te in a) for a in arcs.values())
    print(f"  Total arc: {total_arc:.2e}")
    print(f"  Exposed points: {len(points)}")
    assert len(points) == 0
    assert total_arc < 1e-10
    print("  PASSED ✓\n")


if __name__ == "__main__":
    test_four_spheres_two_points()
    test_five_spheres_one_point()
    test_normal_arcs()
    test_offset_center()
    test_fully_covered()