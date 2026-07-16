"""
Visualize a 3D SDF field as nested iso-surface shells, cut by a plane so the
cross-section reveals the concentric layers. Exports one combined .obj.

Each shell is the iso-surface f(x)=level for a set of SDF levels (inward negative
+ the 0 surface + outward positive offsets). Every shell is clipped to one side
of a cutting plane, so opening the obj shows the onion-ring cross-section — a
direct read on whether the field is a clean, monotone distance function or has
spurious pockets/floaters between layers.

Usage:
    python debug/sdf_contour_shells.py --source igr --name rings
    python debug/sdf_contour_shells.py --source hotspot --name rings --axis x
"""
import os
import sys
import argparse
import numpy as np
import trimesh
from skimage.measure import marching_cubes

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _field_fn(source, name):
    """Return a callable sdf(points)->values for the chosen field source."""
    if source == 'igr':
        import igr_adapter as A
        net = A.load_igr_network(name)
        return lambda P: A.igr_sdf_values(net, P)
    else:  # a neural_sdf.py model (gt / hotspot / point-cloud)
        import neural_sdf as N
        if source == 'pointcloud':
            model, _, _ = N.train_neural_sdf_from_points(  # loads cache
                np.zeros((1, 3)), name)  # centroid unused when cached
        else:
            model, _ = N.train_neural_sdf(f'examples/{name}.obj', name, mode=source)
        return lambda P: N.neural_sdf_values(model, P)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default='igr', choices=['igr', 'hotspot', 'gt', 'pointcloud'])
    ap.add_argument('--name', default='rings')
    ap.add_argument('--res', type=int, default=256)
    ap.add_argument('--pad', type=float, default=0.1)
    ap.add_argument('--axis', default='y', choices=['x', 'y', 'z'], help='cut-plane normal')
    ap.add_argument('--cut', type=float, default=0.0, help='cut plane position along --axis')
    ap.add_argument('--levels', type=str, default=None,
                    help='comma list of SDF levels; default = auto from field range')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    sdf = _field_fn(args.source, args.name)

    lo, hi = -0.5 - args.pad, 0.5 + args.pad
    xs = np.linspace(lo, hi, args.res)
    G = np.stack(np.meshgrid(xs, xs, xs, indexing='ij'), axis=-1).reshape(-1, 3)
    V = sdf(G).reshape(args.res, args.res, args.res)
    spacing = (xs[1] - xs[0],) * 3
    print(f"field range: [{V.min():.4f}, {V.max():.4f}]")

    if args.levels:
        levels = [float(s) for s in args.levels.split(',')]
    else:
        # a few inward shells + the surface + several outward offset shells
        inner = float(V.min())
        levels = sorted(set(
            [inner * 0.5, 0.0] +
            list(np.round(np.linspace(0.02, min(0.3, V.max() * 0.9), 7), 4))))

    axis = {'x': 0, 'y': 1, 'z': 2}[args.axis]

    # diverging colour map: 0 -> white, positive -> red, negative -> blue, with
    # saturation growing to the farthest shell. Positive and negative use their
    # own range so an asymmetric field (shallow inside, deep outside) still shows
    # saturated colour on both sides.
    pos_scale = max(1e-6, max(levels))
    neg_scale = max(1e-6, -min(levels))

    def shade(lv):
        if lv >= 0:
            t = min(1.0, lv / pos_scale)           # white -> red
            return [255, int(round(255 * (1 - t))), int(round(255 * (1 - t))), 255]
        t = min(1.0, -lv / neg_scale)              # white -> blue
        return [int(round(255 * (1 - t))), int(round(255 * (1 - t))), 255, 255]

    shells = []
    for lv in levels:
        if not (V.min() < lv < V.max()):
            continue
        verts, faces, _, _ = marching_cubes(V, level=lv, spacing=spacing)
        verts += lo
        m = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        # clip to one side of the plane by keeping faces whose centroid is on it
        cent = m.vertices[m.faces].mean(axis=1)
        keep = (cent[:, axis] <= args.cut)
        if not keep.any():
            continue
        m.update_faces(keep)
        m.remove_unreferenced_vertices()
        m.visual.vertex_colors = np.tile(shade(lv), (len(m.vertices), 1))
        shells.append(m)
        print(f"  level {lv:+.4f}: {len(m.vertices)} verts after cut  colour {shade(lv)[0]}")

    combined = trimesh.util.concatenate(shells)
    base = args.out or f'neural_sdf/{args.name}_{args.source}_shells_{args.axis}{args.cut}'
    base = os.path.splitext(base)[0]
    os.makedirs(os.path.dirname(base) or '.', exist_ok=True)
    # OBJ can't carry colour reliably; export coloured GLB + PLY, plain OBJ for geometry.
    for ext in ('glb', 'ply', 'obj'):
        combined.export(f'{base}.{ext}')
    print(f"Saved {len(shells)} coloured shells -> {base}.glb / .ply (+ plain .obj)  "
          f"({len(combined.vertices)} verts)")


if __name__ == '__main__':
    main()
