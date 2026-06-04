"""
Post-process SDF_DUMP_DEGEN_PTS output to find the degen point farthest from
the gt surface, and print its sphere index.

Usage:
    python find_worst_degen.py <dump_path> <gt_mesh.obj> [grid_len]

If grid_len is given, only that block is considered; otherwise all blocks are
pooled. Prints the worst sphere idx per block (or overall) plus its distance.
"""
import sys
from pathlib import Path
import numpy as np
import trimesh
import igl


def load_blocks(path):
    """Parse the dump: returns list of (grid_len, list_of_(idx, x, y, z))."""
    blocks = []
    cur_label = None
    cur = []
    for raw in Path(path).read_text().splitlines():
        s = raw.strip()
        if not s:
            continue
        if s.startswith("#"):
            if cur_label is not None:
                blocks.append((cur_label, cur))
            cur_label = s
            cur = []
            continue
        parts = s.split()
        idx = int(parts[0])
        xyz = tuple(float(v) for v in parts[1:4])
        cur.append((idx, *xyz))
    if cur_label is not None:
        blocks.append((cur_label, cur))
    return blocks


def worst_in_block(rows, V, F):
    if not rows:
        return None
    idxs = np.array([r[0] for r in rows], dtype=np.int32)
    pts = np.array([(r[1], r[2], r[3]) for r in rows], dtype=np.float64)
    sqrD, _, _ = igl.point_mesh_squared_distance(pts, V, F)
    d = np.sqrt(sqrD)
    k = int(np.argmax(d))
    return idxs[k], d[k], pts[k], d


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    dump_path = sys.argv[1]
    mesh_path = sys.argv[2]
    target_gl = int(sys.argv[3]) if len(sys.argv) > 3 else None

    mesh = trimesh.load(mesh_path, force="mesh")
    V = np.asarray(mesh.vertices, dtype=np.float64)
    # Match SDF_to_surface_3D.py's normalization: center + scale by max extent.
    mn, mx = V.min(0), V.max(0)
    V = (V - (mn + mx) / 2) / float((mx - mn).max())
    F = np.asarray(mesh.faces, dtype=np.int32)
    print(f"loaded gt mesh: {V.shape[0]} verts {F.shape[0]} faces  "
          f"(normalized bbox: {V.min(0)} .. {V.max(0)})")

    blocks = load_blocks(dump_path)
    print(f"found {len(blocks)} blocks in {dump_path}")

    for label, rows in blocks:
        if target_gl is not None and f"grid_len={target_gl}" not in label:
            continue
        res = worst_in_block(rows, V, F)
        if res is None:
            print(f"  {label}: empty")
            continue
        idx, dmax, pt, d_all = res
        print(f"  {label}: {len(rows)} pts; "
              f"worst sphere_idx={idx} dist={dmax:.6g} at ({pt[0]:.6g},{pt[1]:.6g},{pt[2]:.6g})")
        print(f"    dist stats: min={d_all.min():.3g} mean={d_all.mean():.3g} "
              f"p50={np.median(d_all):.3g} p95={np.percentile(d_all, 95):.3g} max={d_all.max():.3g}")


if __name__ == "__main__":
    main()
