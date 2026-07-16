"""
Neural SDF test-data generator.

Trains a small neural field (MLP with positional encoding) to fit the SDF of a
given mesh, then samples SDF points/values from the *network* instead of the
exact mesh distance. This mimics the situation where the input SDF samples come
from a neural SDF (learned, slightly inexact, but smooth), so downstream
reconstruction can be tested against neural-field inputs.

Usage (drop-in replacement for SDF_to_surface_3D.generate_test_mesh_data):

    from neural_sdf import generate_neural_sdf_data
    mesh, points, distances, gradients = generate_neural_sdf_data(
        'examples/bunny.obj', 'bunny', grid_len=20)

`distances` are the network's predictions at the sample points and `gradients`
are the (normalized) autograd gradients of the network — both may deviate from
the exact mesh SDF, which is the point of the test.

Trained weights are cached in out/neural_sdf/<outbase>.pt and reused unless
retrain=True.
"""
import os
# torch and sdf_cpp each bundle their own libomp; without this the second one to
# initialize aborts the process (OMP Error #15). Must be set before importing torch.
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import time
import numpy as np
import trimesh
import igl
import torch
import torch.nn as nn

# Module-level seed, same convention as SDF_to_surface_3D (None = nondeterministic).
seed = None

_CACHE_DIR = 'out/neural_sdf'


def _device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def load_normalized_mesh(path_to_mesh):
    """Load and normalize a mesh exactly like generate_test_mesh_data:
    centered at the bbox center and scaled by the max bbox extent."""
    mesh = trimesh.load(path_to_mesh, force='mesh')
    vmin = np.min(mesh.vertices, axis=0)
    vmax = np.max(mesh.vertices, axis=0)
    mesh.vertices -= (vmin + vmax) / 2
    mesh.vertices /= np.max(vmax - vmin)
    return mesh


def mesh_sdf(mesh, points):
    """Exact signed distance to the mesh (positive outside), same construction
    as generate_test_mesh_data: unsigned distance + winding-number sign."""
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int32)
    sq_dists, _, _ = igl.point_mesh_squared_distance(points, V, F)
    distances = np.sqrt(sq_dists)
    W = igl.winding_number(V, F, points)
    distances[W > 0.5] *= -1.0
    return distances


class NeuralSDF(nn.Module):
    """Simple neural field: positional encoding + MLP with one skip connection."""

    def __init__(self, hidden=256, n_layers=6, n_freqs=6):
        super().__init__()
        self.n_freqs = n_freqs
        self.in_dim = in_dim = 3 + 3 * 2 * n_freqs  # xyz + sin/cos encodings
        self.skip_at = n_layers // 2
        layers = []
        dim = in_dim
        for i in range(n_layers):
            out = hidden
            if i == self.skip_at:
                dim += in_dim  # skip connection re-injects the encoded input
            layers.append(nn.Linear(dim, out))
            dim = out
        self.layers = nn.ModuleList(layers)
        self.out = nn.Linear(hidden, 1)
        self.act = nn.Softplus(beta=100)

    def encode(self, x):
        if self.n_freqs == 0:
            return x
        freqs = 2.0 ** torch.arange(self.n_freqs, device=x.device, dtype=x.dtype) * np.pi
        xf = x.unsqueeze(-1) * freqs  # (N, 3, F)
        enc = torch.cat([torch.sin(xf), torch.cos(xf)], dim=-1).flatten(1)
        return torch.cat([x, enc], dim=-1)

    def forward(self, x):
        e = self.encode(x)
        h = e
        for i, layer in enumerate(self.layers):
            if i == self.skip_at:
                h = torch.cat([h, e], dim=-1)
            h = self.act(layer(h))
        return self.out(h).squeeze(-1)

    @torch.no_grad()
    def geometric_init(self, radius=0.5, center=(0.0, 0.0, 0.0)):
        """SAL/IGR geometric initialization: f(x) ≈ ||x-center|| - radius at init
        (an approximate sphere SDF). Required for the HotSpot / Eikonal mode,
        whose losses are symmetric in sign(u) — the geometric init is what fixes
        the interior as negative, so the sphere must be placed on the shape (pass
        the mesh centroid). Fourier-encoding columns are zeroed so the initial
        field depends only on the raw coordinates (smooth)."""
        c = torch.tensor(center, dtype=self.layers[0].weight.dtype,
                         device=self.layers[0].weight.device)
        for i, lin in enumerate(self.layers):
            out_dim = lin.weight.shape[0]
            nn.init.constant_(lin.bias, 0.0)
            nn.init.normal_(lin.weight, 0.0, np.sqrt(2.0 / out_dim))
            if i == 0 and self.in_dim > 3:
                lin.weight[:, 3:] = 0.0            # zero encoded input, keep raw xyz
            if i == 0:
                lin.bias.copy_(-(lin.weight[:, :3] @ c))  # center: W(x-center)
        if self.in_dim > 3:
            self.layers[self.skip_at].weight[:, -(self.in_dim - 3):] = 0.0
        nn.init.normal_(self.out.weight,
                        mean=np.sqrt(np.pi) / np.sqrt(self.out.weight.shape[1]),
                        std=1e-4)
        nn.init.constant_(self.out.bias, -radius)


def _sample_training_points(mesh, n_surface=150_000, n_uniform=50_000, rng=None,
                            pad=0.1, near_sigmas=(0.005, 0.02, 0.08)):
    """Near-surface points (surface samples + Gaussian offsets at several scales)
    plus uniform points in the padded bbox. Returns (points, gt_sdf)."""
    rng = rng or np.random.default_rng()
    surf, _ = trimesh.sample.sample_surface(mesh, n_surface)
    surf = np.asarray(surf, dtype=np.float64)
    parts = []
    per = n_surface // len(near_sigmas)
    for k, s in enumerate(near_sigmas):
        chunk = surf[k * per:(k + 1) * per]
        parts.append(chunk + rng.normal(0, s, chunk.shape))
    bbox_min = mesh.vertices.min(axis=0) - pad
    bbox_max = mesh.vertices.max(axis=0) + pad
    parts.append(rng.uniform(bbox_min, bbox_max, (n_uniform, 3)))
    points = np.vstack(parts)
    return points, mesh_sdf(mesh, points)


def _grad(model, x):
    """f(x) and ∇f(x) with create_graph=True so the Eikonal term is trainable."""
    y = model(x)
    g, = torch.autograd.grad(y.sum(), x, create_graph=True)
    return y, g


def _train_hotspot(model, mesh, device, rng, n_steps, batch_size, lr, verbose,
                   pad=0.1, near_sigmas=(0.01, 0.03, 0.08), lam0=5.0, lam1=50.0,
                   w_b=20.0, w_e=0.5, w_h=0.5, p=1):
    """HotSpot training (Wang et al., arXiv:2411.14628): boundary + Eikonal +
    heat loss, using ONLY the mesh surface (the zero level set) — no ground-truth
    signed-distance values anywhere off the surface.

    The heat term ½·E[e^{-2λ|u|}(‖∇u‖²+1)] is an asymptotically sufficient
    condition for a true SDF (as λ→∞), which the plain Eikonal term lacks; that
    sufficiency is what suppresses the spurious ghost interiors that Eikonal-only
    training produces. No normals are used — the sign of the field comes from the
    geometric initialization (sphere centered on the mesh centroid). λ is annealed
    up and the heat integral is importance-sampled toward the surface."""
    surf_np, _ = trimesh.sample.sample_surface(mesh, 200_000)
    surf_np = np.asarray(surf_np, dtype=np.float64)
    surf = torch.tensor(surf_np, dtype=torch.float32, device=device)

    # off-surface pool: near-surface Gaussian shells (importance sampling toward
    # the surface, where the heat weight has its mass) + uniform in the padded bbox
    bbox_min = mesh.vertices.min(axis=0) - pad
    bbox_max = mesh.vertices.max(axis=0) + pad
    per = 80_000
    pool = [surf_np[rng.integers(0, len(surf_np), per)] + rng.normal(0, s, (per, 3))
            for s in near_sigmas]
    pool.append(rng.uniform(bbox_min, bbox_max, (80_000, 3)))
    off = torch.tensor(np.vstack(pool), dtype=torch.float32, device=device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps)
    nb = batch_size // 2
    model.train()
    for step in range(n_steps):
        lam = lam0 + (lam1 - lam0) * step / max(1, n_steps - 1)
        si = torch.randint(0, len(surf), (nb,), device=device)
        oi = torch.randint(0, len(off), (nb,), device=device)
        xs = surf[si].clone().requires_grad_(True)
        xo = off[oi].clone().requires_grad_(True)
        us, gs = _grad(model, xs)
        uo, go = _grad(model, xo)

        boundary = us.abs().pow(p).mean()                        # u=0 on surface
        g_all = torch.cat([gs, go], dim=0)
        eikonal = (g_all.norm(dim=1) - 1.0).abs().pow(p).mean()  # |∇u|=1
        heat = 0.5 * (torch.exp(-2.0 * lam * uo.abs())
                      * (go.norm(dim=1) ** 2 + 1.0)).mean()       # HotSpot Eq.7
        loss = w_b * boundary + w_e * eikonal + w_h * heat
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if verbose and (step % 1000 == 0 or step == n_steps - 1):
            print(f"    step {step:>5d}  λ {lam:5.1f}  boundary {boundary.item():.5f}  "
                  f"eik {eikonal.item():.4f}  heat {heat.item():.4f}")


def train_neural_sdf(path_to_mesh, outbase, mode='gt', retrain=False, n_steps=3000,
                     batch_size=16384, lr=1e-3, hidden=256, n_layers=6, n_freqs=6,
                     verbose=True):
    """
    Train (or load from cache) a NeuralSDF fitting the mesh at path_to_mesh.

    mode:
      'gt'      - regress the exact mesh SDF sampled in space (DeepSDF-style;
                  the network sees ground-truth signed-distance values).
      'hotspot' - HotSpot loss (boundary + Eikonal + heat, arXiv:2411.14628):
                  trains from surface points only, with NO ground-truth SDF and
                  NO normals. Fourier encoding is disabled and a geometric init
                  is used, as required by the PDE-based losses.

    Returns (model, mesh) where mesh is the normalized ground-truth mesh.
    """
    mesh = load_normalized_mesh(path_to_mesh)
    device = _device()
    # The PDE-based HotSpot loss fights Fourier features (high frequencies spawn
    # ghost regions), so it uses plain coordinates like canonical IGR/HotSpot.
    if mode == 'hotspot':
        n_freqs = 0
    model = NeuralSDF(hidden=hidden, n_layers=n_layers, n_freqs=n_freqs).to(device)

    os.makedirs(_CACHE_DIR, exist_ok=True)
    suffix = '' if mode == 'gt' else f'_{mode}'  # keep the gt cache name unchanged
    ckpt_path = f'{_CACHE_DIR}/{outbase}{suffix}_h{hidden}_l{n_layers}_f{n_freqs}.pt'
    if os.path.exists(ckpt_path) and not retrain:
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        if verbose:
            print(f"Loaded cached neural SDF ({mode}) from {ckpt_path}")
        return model, mesh

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed if seed is not None else np.random.SeedSequence().entropy % 2**31)
    timer = time.perf_counter()

    if mode == 'hotspot':
        centroid = np.asarray(mesh.vertices).mean(axis=0)
        model.geometric_init(radius=0.3, center=tuple(centroid))
        _train_hotspot(model, mesh, device, rng, max(n_steps, 10000),
                       batch_size, lr, verbose)
    else:  # 'gt'
        pts_np, sdf_np = _sample_training_points(mesh, rng=rng)
        if verbose:
            print(f"  ⏱  {'GT SDF for training set':<30} {time.perf_counter() - timer:>7.2f} s "
                  f"({len(pts_np)} points)")
        pts = torch.tensor(pts_np, dtype=torch.float32, device=device)
        sdf = torch.tensor(sdf_np, dtype=torch.float32, device=device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps)
        timer = time.perf_counter()
        model.train()
        for step in range(n_steps):
            idx = torch.randint(0, len(pts), (batch_size,), device=device)
            pred = model(pts[idx])
            loss = torch.mean(torch.abs(pred - sdf[idx]))  # L1 on signed distance
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            if verbose and (step % 500 == 0 or step == n_steps - 1):
                print(f"    step {step:>5d}  L1 loss {loss.item():.6f}")

    model.eval()
    if verbose:
        print(f"  ⏱  {'Neural SDF training (' + mode + ')':<30} "
              f"{time.perf_counter() - timer:>7.2f} s")

    torch.save(model.state_dict(), ckpt_path)
    if verbose:
        print(f"Saved neural SDF ({mode}) to {ckpt_path}")
    return model, mesh


@torch.no_grad()
def neural_sdf_values(model, points, chunk=65536):
    """Evaluate the neural SDF at (N,3) numpy points. Returns (N,) numpy values."""
    device = next(model.parameters()).device
    out = []
    for i in range(0, len(points), chunk):
        x = torch.tensor(points[i:i + chunk], dtype=torch.float32, device=device)
        out.append(model(x).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def neural_sdf_gradients(model, points, chunk=16384, normalize=True):
    """Autograd gradients of the neural SDF at (N,3) numpy points. Returns (N,3) numpy."""
    device = next(model.parameters()).device
    out = []
    for i in range(0, len(points), chunk):
        x = torch.tensor(points[i:i + chunk], dtype=torch.float32, device=device,
                         requires_grad=True)
        y = model(x)
        g, = torch.autograd.grad(y.sum(), x)
        out.append(g.detach().cpu().numpy())
    grads = np.concatenate(out).astype(np.float64)
    if normalize:
        norm = np.linalg.norm(grads, axis=1, keepdims=True)
        norm[np.abs(norm) <= 1e-8] = 1.0
        grads /= norm
    return grads


def generate_neural_sdf_data(path_to_mesh, outbase, grid_len=10, save=False, noise=0.0,
                             bound=1.0, scatter=False, retrain=False, verbose=True,
                             **train_kwargs):
    """
    Drop-in replacement for SDF_to_surface_3D.generate_test_mesh_data, except the
    SDF values (and gradients) come from a neural field trained on the mesh
    instead of the exact mesh distance.

    Returns:
    mesh: the normalized ground-truth mesh (for error evaluation)
    points: (N, 3) sample coordinates
    distances: (N,) neural SDF values at the samples
    gradients: (N, 3) normalized autograd gradients of the neural SDF
    """
    model, mesh = train_neural_sdf(path_to_mesh, outbase, retrain=retrain,
                                   verbose=verbose, **train_kwargs)
    rng = np.random.default_rng(seed)

    # Same sample layout as generate_test_mesh_data
    bbox_min = np.min(mesh.vertices, axis=0) - 0.1
    bbox_max = np.max(mesh.vertices, axis=0) + 0.1
    x = np.linspace(bbox_min[0], bbox_max[0], grid_len)
    y = np.linspace(bbox_min[1], bbox_max[1], grid_len)
    z = np.linspace(bbox_min[2], bbox_max[2], grid_len)
    X, Y, Z = np.meshgrid(x, y, z)
    points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    if scatter:
        points = rng.uniform(bbox_min, bbox_max, (grid_len**3, 3))

    distances = neural_sdf_values(model, points)
    gradients = neural_sdf_gradients(model, points)

    if noise > 0:
        distances += rng.normal(0, noise, distances.shape)

    mask = np.abs(distances) > 1e-8
    points, distances, gradients = points[mask], distances[mask], gradients[mask]

    if bound < 1.0:
        mask = np.abs(distances) <= bound
        points, distances, gradients = points[mask], distances[mask], gradients[mask]

    if verbose:
        gt = mesh_sdf(mesh, points)
        err = np.abs(distances - gt)
        print(f"Neural SDF vs exact SDF on {len(points)} samples: "
              f"MAE {err.mean():.5f}  max {err.max():.5f}")

    if save:
        os.makedirs('normalized_examples', exist_ok=True)
        mesh.export(f'normalized_examples/{outbase}.obj')
        print(f"Saved normalized mesh to normalized_examples/{outbase}.obj")

    return mesh, points, distances, gradients


def export_neural_marching_cubes(model, outpath, resolution=128, pad=0.1, level=0.0):
    """Marching cubes directly on the neural field, for a visual sanity check of
    how well the network fits the mesh."""
    from skimage.measure import marching_cubes
    lo, hi = -0.5 - pad, 0.5 + pad
    xs = np.linspace(lo, hi, resolution)
    G = np.stack(np.meshgrid(xs, xs, xs, indexing='ij'), axis=-1).reshape(-1, 3)
    vals = neural_sdf_values(model, G).reshape(resolution, resolution, resolution)
    sp = (xs[1] - xs[0],) * 3
    verts, faces, _, _ = marching_cubes(vals, level=level, spacing=sp)
    verts += lo
    os.makedirs(os.path.dirname(outpath) or '.', exist_ok=True)
    trimesh.Trimesh(vertices=verts, faces=faces).export(outpath)
    print(f"Exported neural-field marching cubes to {outpath}")


if __name__ == "__main__":
    # Train (or refresh) the cached neural field for one mesh. The reconstruction
    # pipeline loads these weights in-process, so this is only for (re)training and
    # an optional marching-cubes sanity check of the raw field.
    import argparse
    ap = argparse.ArgumentParser(description="Train/cache a neural SDF for a mesh.")
    ap.add_argument('--name', default='bunny', help='examples/<name>.obj')
    ap.add_argument('--mode', default='hotspot', choices=['gt', 'hotspot'])
    ap.add_argument('--retrain', action='store_true', help='retrain even if cached')
    ap.add_argument('--mc', type=int, default=128,
                    help='marching-cubes resolution for a sanity export (0 = skip)')
    args = ap.parse_args()

    seed = 1
    t0 = time.perf_counter()
    model, mesh = train_neural_sdf(f'examples/{args.name}.obj', args.name,
                                   mode=args.mode, retrain=args.retrain)
    if args.mc:
        export_neural_marching_cubes(
            model, f'{_CACHE_DIR}/{args.name}_{args.mode}_mc{args.mc}.obj',
            resolution=args.mc)
    print(f"  ⏱  {'Total':<30} {time.perf_counter() - t0:>7.2f} s")
