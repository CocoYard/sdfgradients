"""
Incremental clipping approach for computing exposed regions on a sphere.

For each new cap added:
  1. Clip existing boundary arcs (remove parts covered by the new cap)
  2. Compute the new cap's own exposed arcs (not covered by any prior cap)
     and add them to the boundary.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict

# Data structures

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
    containment_gap: float = np.inf  # only set when the other sphere fully contains main (dist + main.r <= other.r).
                                     # value = other.r - dist - main.r = gap between the two sphere surfaces at the closest point.
                                     # 0 means internally tangent; inf means not a containment cap.

@dataclass
class BoundaryArc:
    """
    Represents an arc on the circle of a cap that forms part of the exposed region boundary.
    t_start and t_end are angles in radians, parameterizing the arc on the cap's circle.
    t_start < t_end is guaranteed
    """
    cap_idx: int
    t_start: float
    t_end: float

    def length(self):
        return self.t_end - self.t_start

    def midpoint_angle(self):
        return (self.t_start + self.t_end) / 2.0

    def point_at(self, t, caps):
        cap = caps[self.cap_idx]
        return (cap.circle_center
                + cap.circle_radius * (np.cos(t) * cap.local_u + np.sin(t) * cap.local_v))

    def start_point(self, caps):
        return self.point_at(self.t_start, caps)

    def end_point(self, caps):
        return self.point_at(self.t_end, caps)


# Geometry helpers

def perpendicular_unit(n):
    n = np.asarray(n, dtype=float)
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
        gap = other.radius - dist - main.radius
        return Cap(n, np.dot(n, main.center) - main.radius - 1,
                   main.center, 0.0, np.zeros(3), np.zeros(3), np.pi, sphere_idx,
                   containment_gap=gap)
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


def point_on_cap_circle(cap, t):
    return (cap.circle_center
            + cap.circle_radius * (np.cos(t) * cap.local_u + np.sin(t) * cap.local_v))


def is_inside_cap(pt, cap):
    return np.dot(cap.normal, pt) - cap.d > 1e-12


def _angle_in_arc(t, t_start, t_end):
    t = t % (2 * np.pi)
    dt = (t - t_start) % (2 * np.pi)
    arc_len = t_end - t_start
    return dt < arc_len - 1e-14


def intersect_circle_with_plane(circle_cap, cutting_cap):
    n = cutting_cap.normal
    R = circle_cap.circle_radius
    if R < 1e-14:
        return []
    a = np.dot(n, circle_cap.local_u)
    b = np.dot(n, circle_cap.local_v)
    c_val = (cutting_cap.d - np.dot(n, circle_cap.circle_center)) / R
    A_amp = np.sqrt(a*a + b*b)
    if A_amp < 1e-14:
        return []
    ratio = c_val / A_amp
    if abs(ratio) > 1.0 + 1e-10:
        return []
    ratio = np.clip(ratio, -1.0, 1.0)
    alpha = np.arctan2(b, a)
    delta = np.arccos(ratio)
    if delta < 1e-12:
        return []
    t1 = (alpha - delta) % (2 * np.pi)
    t2 = (alpha + delta) % (2 * np.pi)
    return sorted([t1, t2])


# Clip a single arc by a cap

def clip_arc_by_cap(arc, cutting_cap, caps):
    host_cap = caps[arc.cap_idx]
    hits = intersect_circle_with_plane(host_cap, cutting_cap)
    hits_in = [t for t in hits if _angle_in_arc(t, arc.t_start, arc.t_end)]

    if not hits_in:
        mid_pt = arc.point_at(arc.midpoint_angle(), caps)
        if is_inside_cap(mid_pt, cutting_cap):
            return []
        return [arc]

    hits_in.sort(key=lambda t: (t - arc.t_start) % (2 * np.pi))
    boundaries = [arc.t_start] + hits_in + [arc.t_end]
    kept = []
    for k in range(len(boundaries) - 1):
        t_s, t_e = boundaries[k], boundaries[k + 1]
        if t_e < t_s - 1e-14:
            t_e += 2 * np.pi
        if t_e - t_s < 1e-15:
            continue
        mid_pt = point_on_cap_circle(host_cap, (t_s + t_e) / 2)
        if not is_inside_cap(mid_pt, cutting_cap):
            kept.append(BoundaryArc(cap_idx=arc.cap_idx, t_start=t_s, t_end=t_e))
    return kept


# Compute exposed arcs on a cap's circle (half-plane intersection)

def _intersect_interval_lists(list_a, list_b, skip_tol=1e-8, merge_tol=1e-12):
    result = []
    for a_s, a_e in list_a:
        for b_s, b_e in list_b:
            lo = max(a_s, b_s)
            hi = min(a_e, b_e)
            if hi - lo > -skip_tol:
                if hi < lo:
                    mid = (lo + hi) / 2
                    result.append((mid - 1e-15, mid + 1e-15))
                else:
                    result.append((lo, hi))
    result.sort()
    merged = []
    for s, e in result:
        if merged and s <= merged[-1][1] + merge_tol:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def compute_exposed_arcs_on_circle(cap_idx, caps, active_cap_indices, skip_tol=1e-8):
    host = caps[cap_idx]
    R = host.circle_radius
    if R < 1e-14:
        return []

    intervals = [(0.0, 2 * np.pi)]

    for j in active_cap_indices:
        if j == cap_idx:
            continue
        other = caps[j]

        a = -np.dot(other.normal, host.local_u)
        b = -np.dot(other.normal, host.local_v)
        c = -(other.d - np.dot(other.normal, host.circle_center))

        A_line = np.sqrt(a*a + b*b)
        if A_line < 1e-14:
            if c > skip_tol: # host cap is higher than other cap, so it is eaten by the other cap
                return []
            # other cap is effectively parallel and outside the host's circle plane, so it doesn't cut anything
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

        if arc_start < arc_end: # cutting arc does not wrap around 0
            constraint = [(arc_start, arc_end)]
        else:
            constraint = [(arc_start, 2 * np.pi), (0.0, arc_end)]
        intervals = _intersect_interval_lists(intervals, constraint, skip_tol=skip_tol)
        if not intervals:
            return []

    expanded = []
    for s, e in intervals:
        expanded.append((min(s, e), max(s, e)))

    return expanded


# Main function

def compute_exposed_region(main, caps, tol=1e-8, degen_tol=1e-6, merge_tol=1e-12, tangent_tol=1e-8):
    """
    Compute the exposed region boundary incrementally.

    Returns
    -------
    arcs_by_cap : dict {cap_idx: [BoundaryArc, ...]}
    degenerate_points : list of np.ndarray (deduplicated)
    """
    if len(caps) == 0:
        return {}, []

    valid_caps = []
    for i, cap in enumerate(caps):
        if cap.phi >= np.pi - 1e-10:
            if cap.containment_gap <= tangent_tol:
                # Near-tangent internal containment: treat tangent point as degenerate.
                # Use the point opposite to the containing sphere (away from it), which is the exposed side.
                tangent_pt = main.center - main.radius * cap.normal
                return {}, [tangent_pt]
            return {}, []
        if cap.circle_radius < 1e-14:
            continue
        valid_caps.append(i)

    if not valid_caps:
        return {}, []

    all_arcs = []
    active_caps = []

    for cap_idx in valid_caps:
        # Clip all existing arcs against the current cap, discarding portions covered by it
        new_all_arcs = []
        for arc in all_arcs:
            surviving = clip_arc_by_cap(arc, caps[cap_idx], caps)
            new_all_arcs.extend(surviving)
        all_arcs = new_all_arcs

        # Add the current cap to the active set, then compute exposed arc intervals on its circle
        new_intervals = compute_exposed_arcs_on_circle(
            cap_idx, caps, active_caps, skip_tol=tol
        )
        active_caps.append(cap_idx)

        # Append the newly exposed arcs to the global arc list
        for t_s, t_e in new_intervals:
            all_arcs.append(BoundaryArc(cap_idx=cap_idx, t_start=t_s, t_end=t_e))

    # Group arcs by their cap index for easy per-cap access
    arcs_by_cap = {}
    for arc in all_arcs:
        if arc.cap_idx not in arcs_by_cap:
            arcs_by_cap[arc.cap_idx] = []
        arcs_by_cap[arc.cap_idx].append(arc)

    # Collect midpoints of degenerate arcs (near-zero length) as vertex candidates
    degen_raw = []
    _total_arc_length = 0.0
    for arc in all_arcs:
        length = arc.length()
        _total_arc_length += length
    if _total_arc_length < degen_tol:
        for arc in all_arcs:
            mid_pt = arc.point_at(arc.midpoint_angle(), caps)
            degen_raw.append(mid_pt)
    # Deduplicate degenerate points
    unique = []
    for pt in degen_raw:
        if all(np.linalg.norm(pt - u) >= merge_tol for u in unique):
            unique.append(pt.copy())
    return arcs_by_cap, unique


# Tests

def _total_arc_length(arcs_by_cap):
    return sum(sum(arc.length() for arc in arcs) for arcs in arcs_by_cap.values())


def test_no_caps():
    print("=" * 60)
    print("Test: No caps")
    print("=" * 60)
    main = Sphere([0, 0, 0], 1.0)
    arcs_by_cap, degen = compute_exposed_region(main, [])
    assert len(arcs_by_cap) == 0 and len(degen) == 0
    print("  PASSED\n")


def test_one_cap():
    print("=" * 60)
    print("Test: 1 cap -> full circle")
    print("=" * 60)
    main = Sphere([0, 0, 0], 1.0)
    caps = compute_all_caps(main, [Sphere([1.5, 0, 0], 1.0)])
    arcs_by_cap, degen = compute_exposed_region(main, caps)
    total = _total_arc_length(arcs_by_cap)
    print(f"  Total arc: {total:.4f} (expect {2*np.pi:.4f})")
    assert abs(total - 2 * np.pi) < 1e-4 and len(degen) == 0
    print("  PASSED\n")


def test_two_caps():
    print("=" * 60)
    print("Test: 2 caps +/-x -> arcs on both caps")
    print("=" * 60)
    main = Sphere([0, 0, 0], 1.0)
    caps = compute_all_caps(main, [Sphere([1.5, 0, 0], 1.0), Sphere([-1.5, 0, 0], 1.0)])
    arcs_by_cap, degen = compute_exposed_region(main, caps)
    print(f"  Caps with arcs: {list(arcs_by_cap.keys())}")
    for ci, arcs in arcs_by_cap.items():
        print(f"    cap {ci}: {len(arcs)} arc(s), total={sum(a.length() for a in arcs):.4f}")
    assert len(arcs_by_cap) == 2 and _total_arc_length(arcs_by_cap) > 0.1 and len(degen) == 0
    print("  PASSED\n")


def test_four_caps_degenerate():
    print("=" * 60)
    print("Test: 4 caps -> degenerate at (+/-1,0,0)")
    print("=" * 60)
    main = Sphere([0, 0, 0], 1.0)
    others = [
        Sphere([0, 0, 1], np.sqrt(2)), Sphere([0, 0, -1], np.sqrt(2)),
        Sphere([0, -1, 0], np.sqrt(2)), Sphere([0, 1, 0], np.sqrt(2)),
    ]
    caps = compute_all_caps(main, others)
    arcs_by_cap, degen = compute_exposed_region(main, caps, merge_tol=1e-6, degen_tol=1e-5)
    print(f"  Degenerate points: {len(degen)}")
    for pt in degen:
        print(f"    ({pt[0]:+.6f}, {pt[1]:+.6f}, {pt[2]:+.6f})")
    assert len(degen) == 2
    pts_sorted = sorted(degen, key=lambda p: p[0])
    np.testing.assert_allclose(pts_sorted[0], [-1, 0, 0], atol=1e-4)
    np.testing.assert_allclose(pts_sorted[1], [1, 0, 0], atol=1e-4)
    print("  PASSED\n")


def test_five_caps_one_point():
    print("=" * 60)
    print("Test: 5 caps -> degenerate at (1,0,0)")
    print("=" * 60)
    main = Sphere([0, 0, 0], 1.0)
    others = [
        Sphere([0, 0, 1], np.sqrt(2)), Sphere([0, 0, -1], np.sqrt(2)),
        Sphere([0, -1, 0], np.sqrt(2)), Sphere([0, 1, 0], np.sqrt(2)),
        Sphere([-1, 0, 0], np.sqrt(2)),
    ]
    caps = compute_all_caps(main, others)
    arcs_by_cap, degen = compute_exposed_region(main, caps, merge_tol=1e-6, degen_tol=1e-5)
    print(f"  Degenerate points: {len(degen)}")
    for pt in degen:
        print(f"    ({pt[0]:+.6f}, {pt[1]:+.6f}, {pt[2]:+.6f})")
    assert len(degen) == 1
    np.testing.assert_allclose(degen[0], [1, 0, 0], atol=1e-4)
    print("  PASSED\n")


def test_fully_covered():
    print("=" * 60)
    print("Test: 6 caps -> fully covered")
    print("=" * 60)
    main = Sphere([0, 0, 0], 1.0)
    others = [
        Sphere([0, 0, 1], np.sqrt(2)), Sphere([0, 0, -1], np.sqrt(2)),
        Sphere([0, -1, 0], np.sqrt(2)), Sphere([0, 1, 0], np.sqrt(2)),
        Sphere([1, 0, 0], np.sqrt(2)), Sphere([-1, 0, 0], np.sqrt(2)),
    ]
    caps = compute_all_caps(main, others)
    arcs_by_cap, degen = compute_exposed_region(main, caps)
    assert len(arcs_by_cap) == 0 and len(degen) == 0
    print("  PASSED\n")


def test_near_tangent_containment():
    print("=" * 60)
    print("Test: near-tangent containment -> degenerate point at tangent")
    print("=" * 60)
    # small sphere at y=-0.537, r=0.0368; large sphere at y=-0.600, r=0.1000
    # dist=0.0632, gap = 0.1000 - 0.0632 - 0.0368 = 0.0 (exact tangency)
    main = Sphere([0.0, -0.537, 0.0], 0.0368)
    others = [Sphere([0.0, -0.600, 0.0], 0.1000)]
    caps = compute_all_caps(main, others)
    arcs_by_cap, degen = compute_exposed_region(main, caps, tangent_tol=1e-3)
    print(f"  arcs: {len(arcs_by_cap)}, degen points: {len(degen)}")
    for pt in degen:
        print(f"    ({pt[0]:+.6f}, {pt[1]:+.6f}, {pt[2]:+.6f})")
    assert len(degen) == 1
    # tangent point should be main.center - r * normal(toward large sphere) = away from large sphere
    # normal = (large.center - main.center) / dist = (0,-1,0)
    expected = np.array([0.0, -0.537 + 0.0368, 0.0])
    np.testing.assert_allclose(degen[0], expected, atol=1e-6)
    print("  PASSED\n")


def test_offset_center():
    print("=" * 60)
    print("Test: Offset center, 4 caps -> 2 degenerate points")
    print("=" * 60)
    c = np.array([3.0, -2.0, 7.0])
    main = Sphere(c, 1.0)
    others = [
        Sphere(c + [0, 0, 1], np.sqrt(2)), Sphere(c + [0, 0, -1], np.sqrt(2)),
        Sphere(c + [0, -1, 0], np.sqrt(2)), Sphere(c + [0, 1, 0], np.sqrt(2)),
    ]
    caps = compute_all_caps(main, others)
    arcs_by_cap, degen = compute_exposed_region(main, caps, merge_tol=1e-6, degen_tol=1e-5)
    print(f"  Degenerate points: {len(degen)}")
    for pt in degen:
        print(f"    ({pt[0]:+.6f}, {pt[1]:+.6f}, {pt[2]:+.6f})")
    assert len(degen) == 2
    pts_sorted = sorted(degen, key=lambda p: p[0])
    np.testing.assert_allclose(pts_sorted[0], c + [-1, 0, 0], atol=1e-4)
    np.testing.assert_allclose(pts_sorted[1], c + [1, 0, 0], atol=1e-4)
    print("  PASSED\n")


def test_random_compare():
    print("=" * 60)
    print("Test: Random spheres, compare with old method")
    print("=" * 60)
    np.random.seed(42)
    main = Sphere([0, 0, 0], 1.0)
    others = [Sphere(np.random.randn(3) * 1.2, 0.6 + np.random.rand() * 0.5) for _ in range(8)]
    caps = compute_all_caps(main, others)
    arcs_by_cap, degen = compute_exposed_region(main, caps)
    total_new = _total_arc_length(arcs_by_cap)
    total_old = 0.0
    all_cap_indices = list(range(len(caps)))
    for i in range(len(caps)):
        if caps[i].circle_radius < 1e-14:
            continue
        ivs = compute_exposed_arcs_on_circle(i, caps, all_cap_indices, merge_tol=0)
        total_old += sum(e - s for s, e in ivs)
    print(f"  Old total: {total_old:.6f}")
    print(f"  New total: {total_new:.6f}")
    print(f"  Diff: {abs(total_new - total_old):.2e}")
    assert abs(total_new - total_old) < 1e-4
    print("  PASSED\n")


if __name__ == "__main__":
    # test_no_caps()
    # test_one_cap()
    # test_two_caps()
    # test_four_caps_degenerate()
    # test_five_caps_one_point()
    # test_fully_covered()
    test_near_tangent_containment()
    # test_offset_center()
    # test_random_compare()
    # print("=" * 60)
    # print("All tests passed!")
    # print("=" * 60)