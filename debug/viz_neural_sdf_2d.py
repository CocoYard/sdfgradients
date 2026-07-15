"""
2D slice visualization of a trained neural SDF field.

The neural SDF is a 3D field f(x,y,z). To "see it in 2D" we cut a plane
(default z=0), sample a dense grid on that plane, evaluate the network, and plot:
  - the signed distance as a diverging heatmap (blue inside / red outside),
  - the zero level set (the object's cross-section) as a contour,
  - a few iso-distance contours, and
  - the in-plane gradient direction as a quiver field.

The exact-mesh SDF zero contour is overlaid (dashed) so you can eyeball how well
the network fits the true surface on this slice.

Usage:
    python debug/viz_neural_sdf_2d.py                 # bunny, z=0 slice
    python debug/viz_neural_sdf_2d.py --axis y --offset 0.1
    python debug/viz_neural_sdf_2d.py --name armadillo --res 400
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from neural_sdf import train_neural_sdf, neural_sdf_values, neural_sdf_gradients, mesh_sdf


def slice_grid(bbox_min, bbox_max, axis, offset, res):
    """Build (res*res, 3) points on the plane `axis == offset`, plus the two
    in-plane axis indices and the 2D coordinate arrays for plotting."""
    plane_axes = [i for i in range(3) if i != axis]
    a, b = plane_axes
    ua = np.linspace(bbox_min[a], bbox_max[a], res)
    ub = np.linspace(bbox_min[b], bbox_max[b], res)
    UA, UB = np.meshgrid(ua, ub)
    pts = np.empty((UA.size, 3))
    pts[:, a] = UA.ravel()
    pts[:, b] = UB.ravel()
    pts[:, axis] = offset
    return pts, plane_axes, UA, UB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default='bunny')
    ap.add_argument('--mesh', default=None, help='path to mesh (default examples/<name>.obj)')
    ap.add_argument('--axis', default='z', choices=['x', 'y', 'z'], help='slice-normal axis')
    ap.add_argument('--offset', type=float, default=0.0, help='plane position along --axis')
    ap.add_argument('--res', type=int, default=300, help='grid resolution per side')
    ap.add_argument('--pad', type=float, default=0.1, help='bbox padding')
    ap.add_argument('--quiver', type=int, default=25, help='gradient arrows per side (0=off)')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    mesh_path = args.mesh or f'examples/{args.name}.obj'
    axis = {'x': 0, 'y': 1, 'z': 2}[args.axis]
    axis_names = ['x', 'y', 'z']

    model, mesh = train_neural_sdf(mesh_path, args.name, verbose=True)

    bbox_min = mesh.vertices.min(axis=0) - args.pad
    bbox_max = mesh.vertices.max(axis=0) + args.pad

    pts, (a, b), UA, UB = slice_grid(bbox_min, bbox_max, axis, args.offset, args.res)

    vals = neural_sdf_values(model, pts).reshape(args.res, args.res)
    gt = mesh_sdf(mesh, pts).reshape(args.res, args.res)

    vmax = np.percentile(np.abs(vals), 99)

    fig, ax = plt.subplots(figsize=(8, 7))
    extent = [bbox_min[a], bbox_max[a], bbox_min[b], bbox_max[b]]
    im = ax.imshow(vals, extent=extent, origin='lower', cmap='RdBu',
                   vmin=-vmax, vmax=vmax, aspect='equal')
    fig.colorbar(im, ax=ax, label='signed distance')

    # iso-distance contours + zero level set of the network
    levels = np.linspace(-vmax, vmax, 15)
    ax.contour(UA, UB, vals, levels=levels, colors='k', linewidths=0.3, alpha=0.4)
    ax.contour(UA, UB, vals, levels=[0.0], colors='k', linewidths=2.0)

    # exact-mesh zero contour (dashed) for comparison
    ax.contour(UA, UB, gt, levels=[0.0], colors='lime', linewidths=1.5, linestyles='--')

    if args.quiver > 0:
        step = max(1, args.res // args.quiver)
        gpts = pts.reshape(args.res, args.res, 3)[::step, ::step].reshape(-1, 3)
        grads = neural_sdf_gradients(model, gpts, normalize=True).reshape(-1, 3)
        qa = gpts[:, a]
        qb = gpts[:, b]
        ax.quiver(qa, qb, grads[:, a], grads[:, b], color='k', alpha=0.5,
                  scale=40, width=0.002)

    ax.set_xlabel(axis_names[a])
    ax.set_ylabel(axis_names[b])
    ax.set_title(f'Neural SDF slice: {args.name}  {axis_names[axis]}={args.offset}\n'
                 f'black=network 0-level, green dashed=exact mesh')

    out = args.out or f'debug/neural_sdf_2d_{args.name}_{args.axis}{args.offset}.png'
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f'Saved {out}')


if __name__ == '__main__':
    main()
