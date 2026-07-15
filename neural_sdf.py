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
        in_dim = 3 + 3 * 2 * n_freqs  # xyz + sin/cos encodings
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


def train_neural_sdf(path_to_mesh, outbase, retrain=False, n_steps=3000, batch_size=16384,
                     lr=1e-3, hidden=256, n_layers=6, n_freqs=6, verbose=True):
    """
    Train (or load from cache) a NeuralSDF fitting the mesh at path_to_mesh.
    Returns (model, mesh) where mesh is the normalized ground-truth mesh.
    """
    mesh = load_normalized_mesh(path_to_mesh)
    device = _device()
    model = NeuralSDF(hidden=hidden, n_layers=n_layers, n_freqs=n_freqs).to(device)

    os.makedirs(_CACHE_DIR, exist_ok=True)
    ckpt_path = f'{_CACHE_DIR}/{outbase}_h{hidden}_l{n_layers}_f{n_freqs}.pt'
    if os.path.exists(ckpt_path) and not retrain:
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        if verbose:
            print(f"Loaded cached neural SDF from {ckpt_path}")
        return model, mesh

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed if seed is not None else np.random.SeedSequence().entropy % 2**31)

    timer = time.perf_counter()
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
        print(f"  ⏱  {'Neural SDF training':<30} {time.perf_counter() - timer:>7.2f} s")

    torch.save(model.state_dict(), ckpt_path)
    if verbose:
        print(f"Saved neural SDF to {ckpt_path}")
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


def save_neural_sdf_npz(path_to_mesh, outbase, grid_len=10, noise=0.0, bound=1.0,
                        scatter=False, retrain=False, verbose=True, **train_kwargs):
    """
    Generate neural SDF samples and save them in the npz format that the
    reconstruction pipeline already reads via options.path_to_sdf
    (keys: points, sdf_values; gradients included as extra).

    IMPORTANT: torch and sdf_cpp bundle conflicting libomp runtimes and crash when
    loaded in the same process, so use this two-step flow: run this (torch process)
    to write the npz, then run the pipeline (sdf_cpp process) with
    options.path_to_sdf pointing at it. Also saves the normalized GT mesh to
    normalized_examples/<outbase>.obj for error evaluation.
    """
    mesh, points, distances, gradients = generate_neural_sdf_data(
        path_to_mesh, outbase, grid_len=grid_len, save=True, noise=noise, bound=bound,
        scatter=scatter, retrain=retrain, verbose=verbose, **train_kwargs)
    os.makedirs('out', exist_ok=True)
    out_path = f'out/{outbase}_neural_sdf_{len(points)}.npz'
    np.savez(out_path, points=points, sdf_values=distances, gradients=gradients)
    print(f"Saved neural SDF samples to {out_path}")
    return out_path


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
    seed = 1
    name = 'bunny'
    t0 = time.perf_counter()
    mesh, points, distances, gradients = generate_neural_sdf_data(
        f'examples/{name}.obj', name, grid_len=20, retrain=False)
    print(f"points {points.shape}, distances {distances.shape}, gradients {gradients.shape}")

    # save npz for the reconstruction pipeline (separate process, see save_neural_sdf_npz)
    save_neural_sdf_npz(f'examples/{name}.obj', name, grid_len=20, verbose=False)

    # sanity check: marching cubes on the raw neural field
    model, _ = train_neural_sdf(f'examples/{name}.obj', name, verbose=False)
    export_neural_marching_cubes(model, f'out/neural_sdf/{name}_mc128.obj', resolution=128)

    # compare gradient direction with GT (closest-point direction)
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int32)
    _, _, closest = igl.point_mesh_squared_distance(points, V, F)
    gt_dir = points - closest
    gt_dir /= np.maximum(np.linalg.norm(gt_dir, axis=1, keepdims=True), 1e-12)
    gt_dir *= np.sign(mesh_sdf(mesh, points))[:, None]
    cos = np.abs(np.sum(gradients * gt_dir, axis=1))
    print(f"gradient alignment |cos|: mean {cos.mean():.4f}  min {cos.min():.4f}")
    print(f"  ⏱  {'Total':<30} {time.perf_counter() - t0:>7.2f} s")
