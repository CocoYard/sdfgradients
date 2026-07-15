"""
2D neural SDF: train a small neural field directly on a 2D image contour.

This is the 2D analogue of neural_sdf.py. Instead of a mesh we take an image
(examples/eiffel.png), extract its contours exactly like
SDF_to_surface.generate_2D_mesh (find_contours @0.5, normalize to [0,1], flip y,
even-odd rule for inside/outside), and use that as the ground-truth SDF to
supervise a positional-encoding MLP f(x,y) -> signed distance.

Because everything is 2D you can look at the *whole* field directly (no slicing):
the network's zero level set is the reconstructed shape.

Usage:
    python neural_sdf_2d.py                        # train on eiffel.png, save viz
    python neural_sdf_2d.py --image examples/horse.png --name horse
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import time
import argparse
import numpy as np
import torch
import torch.nn as nn

seed = None
_CACHE_DIR = 'out/neural_sdf'


def _device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def extract_contours(path_to_image):
    """Contours from an image in normalized, y-flipped [0,1] coords.

    Mirrors SDF_to_surface.generate_2D_mesh so the coordinate frame matches:
    grayscale, find_contours @0.5, swap to (x,y), divide by a global max, flip y.
    Returns a list of (Mi, 2) arrays (outer boundary + holes)."""
    from skimage import io, color, measure
    from skimage.util import img_as_float
    import warnings
    image = io.imread(path_to_image)
    image = img_as_float(image[:, :, :3])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        gray = color.rgb2gray(image)
    contours = measure.find_contours(gray, level=0.5)
    all_xy = [c[:, ::-1] for c in contours]  # (row,col) -> (x,y)
    global_max = max(np.max(c) for c in all_xy) * 1.01
    return [np.column_stack([c[:, 0] / global_max, 1.0 - c[:, 1] / global_max])
            for c in all_xy]


def make_sdf_fn(contours):
    """Vectorized ground-truth signed distance for the extracted contours.

    Distance = nearest distance to any contour vertex (contours are pixel-dense,
    so vertex distance is a fine approximation to true boundary distance).
    Sign via the even-odd rule: a point inside an odd number of contours is
    inside the shape (negative), handling holes correctly."""
    from scipy.spatial import cKDTree
    from matplotlib.path import Path
    verts = np.vstack(contours)
    tree = cKDTree(verts)
    paths = [Path(c) for c in contours]

    def sdf(pts):
        pts = np.asarray(pts, dtype=np.float64)
        dist, _ = tree.query(pts)
        inside = np.zeros(len(pts), dtype=np.int32)
        for p in paths:
            inside += p.contains_points(pts).astype(np.int32)
        sign = np.where(inside % 2 == 1, -1.0, 1.0)
        return dist * sign

    return sdf


class NeuralSDF2D(nn.Module):
    """positional encoding + MLP with one skip connection (2D input)."""

    def __init__(self, hidden=256, n_layers=6, n_freqs=6):
        super().__init__()
        self.n_freqs = n_freqs
        self.in_dim = in_dim = 2 + 2 * 2 * n_freqs  # xy + sin/cos encodings
        self.skip_at = n_layers // 2
        layers = []
        dim = in_dim
        for i in range(n_layers):
            if i == self.skip_at:
                dim += in_dim
            layers.append(nn.Linear(dim, hidden))
            dim = hidden
        self.layers = nn.ModuleList(layers)
        self.out = nn.Linear(hidden, 1)
        self.act = nn.Softplus(beta=100)

    def encode(self, x):
        if self.n_freqs == 0:
            return x
        freqs = 2.0 ** torch.arange(self.n_freqs, device=x.device, dtype=x.dtype) * np.pi
        xf = x.unsqueeze(-1) * freqs
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
    def geometric_init(self, radius=0.5, center=(0.0, 0.0)):
        """SAL/IGR geometric initialization: make f(x) ≈ ||x-center|| - radius at
        init (an approximate circle SDF). Crucial for PDE-based training (Eikonal
        / HotSpot) because those losses are symmetric in the sign of u — the
        geometric init is what fixes the interior as negative. The circle must be
        placed on the shape (pass the contour centroid as center), otherwise the
        misplaced init lets training collapse to a degenerate ramp. The
        positional-encoding columns are zeroed so the initial field depends only
        on the raw coordinates (smooth)."""
        c = torch.tensor(center, dtype=self.layers[0].weight.dtype,
                         device=self.layers[0].weight.device)
        for i, lin in enumerate(self.layers):
            out_dim = lin.weight.shape[0]
            nn.init.constant_(lin.bias, 0.0)
            nn.init.normal_(lin.weight, 0.0, np.sqrt(2.0 / out_dim))
            if i == 0 and self.in_dim > 2:
                lin.weight[:, 2:] = 0.0            # zero encoded input, keep raw xy
            if i == 0:
                # bias = -(W_raw @ center) so the first layer computes W(x-center),
                # centering the initial circle on the shape.
                lin.bias.copy_(-(lin.weight[:, :2] @ c))
        # zero the re-injected encoded input at the skip layer (keep raw xy);
        # the skip input is [hidden activations | e], and within e the first 2
        # columns are the raw coords, the rest are the encoding. (No-op when
        # n_freqs==0, i.e. plain-coordinate IGR with no encoding to zero.)
        if self.in_dim > 2:
            self.layers[self.skip_at].weight[:, -(self.in_dim - 2):] = 0.0
        nn.init.normal_(self.out.weight,
                        mean=np.sqrt(np.pi) / np.sqrt(self.out.weight.shape[1]),
                        std=1e-4)
        nn.init.constant_(self.out.bias, -radius)


def _sample_training_points(sdf_fn, n_boundary=100_000, n_uniform=50_000, rng=None,
                            near_sigmas=(0.003, 0.01, 0.04), lo=-0.1, hi=1.1,
                            contours=None):
    """Near-contour points (contour vertices + Gaussian offsets at several scales)
    plus uniform points in the padded [lo,hi]^2 box. Returns (points, gt_sdf)."""
    rng = rng or np.random.default_rng()
    verts = np.vstack(contours)
    parts = []
    per = n_boundary // len(near_sigmas)
    for s in near_sigmas:
        idx = rng.integers(0, len(verts), per)
        parts.append(verts[idx] + rng.normal(0, s, (per, 2)))
    parts.append(rng.uniform(lo, hi, (n_uniform, 2)))
    pts = np.vstack(parts)
    return pts, sdf_fn(pts)


def surface_normals(sdf_fn, verts, eps=1e-4):
    """Outward unit normals at surface points via finite-difference of the GT
    signed distance (∇sdf points outward). In IGR-style training the surface
    normals are a *given input* (they come with the point cloud), so using the
    exact contour geometry for them is fair — what mode='eikonal' withholds is
    the off-surface *distance* values, not the surface normals."""
    n = np.zeros_like(verts)
    for d in range(2):
        e = np.zeros((1, 2))
        e[0, d] = eps
        n[:, d] = (sdf_fn(verts + e) - sdf_fn(verts - e)) / (2 * eps)
    nn = np.linalg.norm(n, axis=1, keepdims=True)
    nn[nn < 1e-9] = 1.0
    return n / nn


def _grad(model, x):
    """f(x) and ∇f(x) with create_graph=True so the Eikonal term is trainable."""
    y = model(x)
    g, = torch.autograd.grad(y.sum(), x, create_graph=True)
    return y, g


def train_neural_sdf_2d(path_to_image, outbase, mode='gt', retrain=False,
                        n_steps=3000, batch_size=16384, lr=1e-3, hidden=256,
                        n_layers=6, n_freqs=6, verbose=True):
    """Train (or load from cache) a NeuralSDF2D on the image contour.

    mode:
      'gt'      - regress full-space GT SDF values (DeepSDF-style supervision).
      'eikonal' - surface-only: f=0 + normal alignment on the contour, plus a
                  self-supervised |∇f|=1 Eikonal term off-surface. Uses NO
                  off-surface distance truth (IGR-style; the realistic no-GT case).
      'hybrid'  - GT regression + Eikonal term (accurate but not a realistic
                  deployment: if you had full-space GT you'd never need Eikonal).

    Returns (model, contours, sdf_fn)."""
    contours = extract_contours(path_to_image)
    sdf_fn = make_sdf_fn(contours)
    device = _device()
    # Canonical IGR / HotSpot use plain coordinates (no Fourier features): the
    # encoding's high frequencies fight the smooth PDE solution and spawn ghosts.
    if mode in ('eikonal', 'hotspot'):
        n_freqs = 0
    model = NeuralSDF2D(hidden=hidden, n_layers=n_layers, n_freqs=n_freqs).to(device)

    os.makedirs(_CACHE_DIR, exist_ok=True)
    ckpt = f'{_CACHE_DIR}/{outbase}_2d_{mode}_h{hidden}_l{n_layers}_f{n_freqs}.pt'
    if os.path.exists(ckpt) and not retrain:
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
        if verbose:
            print(f"Loaded cached 2D neural SDF ({mode}) from {ckpt}")
        return model, contours, sdf_fn

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed if seed is not None else np.random.SeedSequence().entropy % 2**31)

    timer = time.perf_counter()
    model.train()

    if mode in ('eikonal', 'hotspot'):
        centroid = np.vstack(contours).mean(axis=0)
        model.geometric_init(radius=0.3, center=tuple(centroid))
        # PDE-based training needs more iterations to propagate the surface
        # constraint outward; GT regression converges much faster.
        pde_steps = max(n_steps, 12000)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=pde_steps)
        trainer = _train_eikonal if mode == 'eikonal' else _train_hotspot
        trainer(model, opt, sched, contours, sdf_fn, device, rng,
                pde_steps, batch_size, verbose)
    else:  # 'gt' or 'hybrid'
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps)
        pts_np, sdf_np = _sample_training_points(sdf_fn, rng=rng, contours=contours)
        pts = torch.tensor(pts_np, dtype=torch.float32, device=device)
        sdf = torch.tensor(sdf_np, dtype=torch.float32, device=device)
        eik_w = 0.1 if mode == 'hybrid' else 0.0
        for step in range(n_steps):
            idx = torch.randint(0, len(pts), (batch_size,), device=device)
            x = pts[idx]
            if eik_w > 0:
                x = x.clone().requires_grad_(True)
                pred, g = _grad(model, x)
                loss = torch.mean(torch.abs(pred - sdf[idx]))
                loss = loss + eik_w * torch.mean((g.norm(dim=1) - 1.0) ** 2)
            else:
                loss = torch.mean(torch.abs(model(x) - sdf[idx]))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            if verbose and (step % 500 == 0 or step == n_steps - 1):
                print(f"    step {step:>5d}  loss {loss.item():.6f}")

    model.eval()
    if verbose:
        print(f"  training ({mode}): {time.perf_counter() - timer:.2f} s")

    torch.save(model.state_dict(), ckpt)
    if verbose:
        print(f"Saved 2D neural SDF ({mode}) to {ckpt}")
    return model, contours, sdf_fn


def _train_eikonal(model, opt, sched, contours, sdf_fn, device, rng, n_steps,
                   batch_size, verbose, lo=-0.1, hi=1.1,
                   near_sigmas=(0.01, 0.05)):
    """IGR-style loss: no off-surface distance truth, only surface constraints
    (f=0, ∇f=normal) and a self-supervised Eikonal term |∇f|=1 off-surface."""
    surf_np = np.vstack(contours)
    nrm_np = surface_normals(sdf_fn, surf_np)
    surf = torch.tensor(surf_np, dtype=torch.float32, device=device)
    nrm = torch.tensor(nrm_np, dtype=torch.float32, device=device)

    # off-surface sampling pool: near-surface (Gaussian) + uniform in the box
    per = 60_000
    pool = [surf_np[rng.integers(0, len(surf_np), per)] + rng.normal(0, s, (per, 2))
            for s in near_sigmas]
    pool.append(rng.uniform(lo, hi, (60_000, 2)))
    off_np = np.vstack(pool)
    off = torch.tensor(off_np, dtype=torch.float32, device=device)

    nb = batch_size // 2
    for step in range(n_steps):
        si = torch.randint(0, len(surf), (nb,), device=device)
        oi = torch.randint(0, len(off), (nb,), device=device)
        xs = surf[si].clone().requires_grad_(True)
        xo = off[oi].clone().requires_grad_(True)
        ys, gs = _grad(model, xs)
        _, go = _grad(model, xo)

        manifold = ys.abs().mean()                              # f=0 on surface
        normal = (gs - nrm[si]).norm(dim=1).mean()              # ∇f aligns normal
        g_all = torch.cat([gs, go], dim=0)
        eikonal = ((g_all.norm(dim=1) - 1.0) ** 2).mean()       # |∇f|=1 everywhere
        loss = manifold * 5.0 + normal * 1.0 + eikonal * 0.1
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if verbose and (step % 500 == 0 or step == n_steps - 1):
            print(f"    step {step:>5d}  manifold {manifold.item():.5f}  "
                  f"normal {normal.item():.4f}  eik {eikonal.item():.4f}")


def _train_hotspot(model, opt, sched, contours, sdf_fn, device, rng, n_steps,
                   batch_size, verbose, lo=-0.1, hi=1.1, near_sigmas=(0.01, 0.05),
                   lam0=5.0, lam1=50.0, w_b=20.0, w_e=0.5, w_h=0.5, p=1):
    """HotSpot loss (Wang et al., arXiv:2411.14628): boundary + Eikonal + heat.

    The heat term ½·E[e^{-2λ|u|}(‖∇u‖²+1)] is an *asymptotically sufficient*
    condition for a true SDF (as λ→∞), which the plain Eikonal term lacks —
    that sufficiency is what suppresses the spurious ghost interiors that
    Eikonal-only training produces. No normals are used; the sign of the field
    is set by the geometric initialization. λ is annealed up over training and
    the heat integral is importance-sampled toward the surface (our near-surface
    Gaussian pool), where the e^{-2λ|u|} weight has its mass."""
    surf_np = np.vstack(contours)
    surf = torch.tensor(surf_np, dtype=torch.float32, device=device)

    per = 60_000
    pool = [surf_np[rng.integers(0, len(surf_np), per)] + rng.normal(0, s, (per, 2))
            for s in near_sigmas]
    pool.append(rng.uniform(lo, hi, (60_000, 2)))
    off = torch.tensor(np.vstack(pool), dtype=torch.float32, device=device)

    nb = batch_size // 2
    for step in range(n_steps):
        lam = lam0 + (lam1 - lam0) * step / max(1, n_steps - 1)
        si = torch.randint(0, len(surf), (nb,), device=device)
        oi = torch.randint(0, len(off), (nb,), device=device)
        xs = surf[si].clone().requires_grad_(True)
        xo = off[oi].clone().requires_grad_(True)
        us, gs = _grad(model, xs)
        uo, go = _grad(model, xo)

        boundary = us.abs().pow(p).mean()                       # u=0 on surface
        g_all = torch.cat([gs, go], dim=0)
        eikonal = (g_all.norm(dim=1) - 1.0).abs().pow(p).mean()  # |∇u|=1
        heat = 0.5 * (torch.exp(-2.0 * lam * uo.abs())
                      * (go.norm(dim=1) ** 2 + 1.0)).mean()      # Eq.7
        loss = w_b * boundary + w_e * eikonal + w_h * heat
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if verbose and (step % 500 == 0 or step == n_steps - 1):
            print(f"    step {step:>5d}  λ {lam:5.1f}  boundary {boundary.item():.5f}  "
                  f"eik {eikonal.item():.4f}  heat {heat.item():.4f}")


@torch.no_grad()
def neural_sdf_values(model, points, chunk=65536):
    device = next(model.parameters()).device
    out = []
    for i in range(0, len(points), chunk):
        x = torch.tensor(points[i:i + chunk], dtype=torch.float32, device=device)
        out.append(model(x).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def neural_sdf_gradients(model, points, chunk=16384, normalize=True):
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
        norm[norm <= 1e-8] = 1.0
        grads /= norm
    return grads


def visualize(model, contours, sdf_fn, name, mode='gt', res=400, quiver=30,
              lo=-0.1, hi=1.1, out=None):
    import matplotlib.pyplot as plt
    xs = np.linspace(lo, hi, res)
    X, Y = np.meshgrid(xs, xs)
    grid = np.column_stack([X.ravel(), Y.ravel()])
    vals = neural_sdf_values(model, grid).reshape(res, res)
    gt = sdf_fn(grid).reshape(res, res)
    gnorm = np.linalg.norm(neural_sdf_gradients(model, grid, normalize=False), axis=1)

    vmax = np.percentile(np.abs(vals), 99)
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(vals, extent=[lo, hi, lo, hi], origin='lower', cmap='RdBu',
                   vmin=-vmax, vmax=vmax, aspect='equal')
    fig.colorbar(im, ax=ax, label='signed distance')

    levels = np.linspace(-vmax, vmax, 17)
    ax.contour(X, Y, vals, levels=levels, colors='k', linewidths=0.3, alpha=0.4)
    ax.contour(X, Y, vals, levels=[0.0], colors='k', linewidths=2.0)
    # exact contour(s) from the image (dashed) for comparison
    for c in contours:
        ax.plot(c[:, 0], c[:, 1], '--', color='lime', linewidth=1.2)

    if quiver > 0:
        step = max(1, res // quiver)
        gp = grid.reshape(res, res, 2)[::step, ::step].reshape(-1, 2)
        g = neural_sdf_gradients(model, gp)
        ax.quiver(gp[:, 0], gp[:, 1], g[:, 0], g[:, 1], color='k', alpha=0.5,
                  scale=40, width=0.002)

    err = np.abs(vals - gt)
    ax.set_title(f'2D Neural SDF [{mode}]: {name}\n'
                 f'MAE vs exact={err.mean():.4f}   mean|∇f|={gnorm.mean():.3f} '
                 f'(SDF wants 1.0)\nblack=network 0-level, green dashed=image contour')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    out = out or f'debug/neural_sdf_2d_{name}_{mode}.png'
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f'Saved {out}  (MAE {err.mean():.5f}, mean|grad| {gnorm.mean():.3f})')
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', default='examples/eiffel.png')
    ap.add_argument('--name', default='eiffel')
    ap.add_argument('--mode', default='all',
                    choices=['gt', 'eikonal', 'hotspot', 'hybrid', 'all'])
    ap.add_argument('--retrain', action='store_true')
    ap.add_argument('--res', type=int, default=400)
    ap.add_argument('--quiver', type=int, default=30)
    args = ap.parse_args()

    modes = ['gt', 'eikonal', 'hotspot', 'hybrid'] if args.mode == 'all' else [args.mode]
    t0 = time.perf_counter()
    for m in modes:
        print(f"\n=== mode: {m} ===")
        model, contours, sdf_fn = train_neural_sdf_2d(args.image, args.name, mode=m,
                                                      retrain=args.retrain)
        visualize(model, contours, sdf_fn, args.name, mode=m, res=args.res,
                  quiver=args.quiver)
    print(f"\n  total {time.perf_counter() - t0:.2f} s")
