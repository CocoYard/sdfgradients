"""
Adapter to use a trained IGR (Gropp et al., ICML 2020) network as the SDF source
for the reconstruction pipeline.

IGR is vendored under external/IGR. Training is run with its own script
(external/IGR/code/reconstruction/run.py); this module loads the resulting
checkpoint and evaluates the learned SDF at arbitrary points, so the field can be
sampled on a grid and fed to the RBF reconstruction exactly like the other
neural-SDF sources.
"""
import os
import sys
import glob
import numpy as np
import torch

_IGR_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'external', 'IGR')
_IGR_CODE = os.path.join(_IGR_ROOT, 'code')
_IGR_MODELS = os.path.join(_IGR_ROOT, 'models')  # committed clean weights, per shape
_CONF = os.path.join(_IGR_CODE, 'reconstruction', 'setup.conf')


def _device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def _latest_checkpoint(expname, checkpoint='latest'):
    """Newest timestamp dir under exps/<expname>, then ModelParameters/<checkpoint>.pth."""
    exp_dir = os.path.join(_IGR_ROOT, 'exps', expname)
    stamps = sorted(glob.glob(os.path.join(exp_dir, '*')))
    if not stamps:
        raise FileNotFoundError(f"No IGR runs under {exp_dir}")
    path = os.path.join(stamps[-1], 'checkpoints', 'ModelParameters', f'{checkpoint}.pth')
    if not os.path.exists(path):
        raise FileNotFoundError(f"No IGR checkpoint at {path}")
    return path


def _model_path(expname):
    """Weights to load: prefer the committed clean file models/<name>.pth (travels
    with the repo, runs on CUDA-less machines); fall back to the newest exps/
    checkpoint (present only on the machine that just trained)."""
    clean = os.path.join(_IGR_MODELS, f'{expname}.pth')
    return clean if os.path.exists(clean) else _latest_checkpoint(expname)


def has_igr_model(expname):
    """True if trained weights exist (committed models/ file OR a local exps/ run)."""
    if os.path.exists(os.path.join(_IGR_MODELS, f'{expname}.pth')):
        return True
    try:
        _latest_checkpoint(expname)
        return True
    except FileNotFoundError:
        return False


def export_model(expname):
    """Copy the freshly trained exps/ weights into the tracked models/ dir so they
    travel with the repo (7 MB vs the ~100 MB exps/ tree)."""
    import shutil
    os.makedirs(_IGR_MODELS, exist_ok=True)
    dst = os.path.join(_IGR_MODELS, f'{expname}.pth')
    shutil.copyfile(_latest_checkpoint(expname), dst)
    print(f"Exported IGR weights -> {dst}")
    return dst


def prepare_igr_input(mesh_name, n_points=100000, out_path=None, seed=1):
    """Sample a (N,6) point cloud [xyz, normal] from examples/<mesh_name>.obj in
    the normalized frame and save it as the .npy IGR trains on. Returns the path.
    Then point external/IGR/code/reconstruction/setup.conf's input_path at it and
    run reconstruction/run.py --expname <mesh_name>."""
    import trimesh
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from neural_sdf import load_normalized_mesh
    mesh = load_normalized_mesh(f'examples/{mesh_name}.obj')
    rng = np.random.default_rng(seed)
    pts, fidx = trimesh.sample.sample_surface(mesh, n_points, seed=int(rng.integers(1 << 30)))
    data = np.concatenate([np.asarray(pts), mesh.face_normals[fidx]], axis=1).astype(np.float32)
    out_path = out_path or os.path.join(_IGR_ROOT, 'data', f'{mesh_name}.npy')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, data)
    print(f"Saved IGR input {data.shape} -> {out_path}")
    return out_path


def load_igr_network(expname, checkpoint='latest', conf_path=_CONF, device=None):
    """Instantiate ImplicitNet from the conf and load the trained weights."""
    device = device or _device()
    if _IGR_CODE not in sys.path:
        sys.path.insert(0, _IGR_CODE)
    from pyhocon import ConfigFactory
    from model.network import ImplicitNet
    conf = ConfigFactory.parse_file(conf_path)
    net = ImplicitNet(d_in=conf.get_int('train.d_in'),
                      **conf.get_config('network.inputs'))
    ckpt = torch.load(_model_path(expname), map_location=device)
    net.load_state_dict(ckpt['model_state_dict'])
    net.to(device).eval()
    return net


@torch.no_grad()
def igr_sdf_values(net, points, chunk=65536, device=None):
    """Evaluate the IGR SDF at (N,3) numpy points. Returns (N,) numpy values."""
    device = device or _device()
    out = []
    for i in range(0, len(points), chunk):
        x = torch.tensor(points[i:i + chunk], dtype=torch.float32, device=device)
        out.append(net(x).squeeze(-1).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def igr_sdf_gradients(net, points, chunk=16384, normalize=True, device=None):
    """Autograd gradients of the IGR SDF at (N,3) numpy points."""
    device = device or _device()
    out = []
    for i in range(0, len(points), chunk):
        x = torch.tensor(points[i:i + chunk], dtype=torch.float32, device=device,
                         requires_grad=True)
        y = net(x).sum()
        g, = torch.autograd.grad(y, x)
        out.append(g.detach().cpu().numpy())
    grads = np.concatenate(out).astype(np.float64)
    if normalize:
        norm = np.linalg.norm(grads, axis=1, keepdims=True)
        norm[norm <= 1e-8] = 1.0
        grads /= norm
    return grads


def igr_sdf_grid(expname, bbox_min, bbox_max, grid_len, scatter=False,
                 checkpoint='latest', rng=None):
    """Sample the trained IGR field on a grid over [bbox_min, bbox_max]. Returns
    (points, distances, gradients) ready for the reconstruction pipeline."""
    net = load_igr_network(expname, checkpoint=checkpoint)
    rng = rng or np.random.default_rng()
    x = np.linspace(bbox_min[0], bbox_max[0], grid_len)
    y = np.linspace(bbox_min[1], bbox_max[1], grid_len)
    z = np.linspace(bbox_min[2], bbox_max[2], grid_len)
    X, Y, Z = np.meshgrid(x, y, z)
    grid = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    if scatter:
        grid = rng.uniform(bbox_min, bbox_max, (grid_len**3, 3))
    distances = igr_sdf_values(net, grid)
    gradients = igr_sdf_gradients(net, grid)
    mask = np.abs(distances) > 1e-8
    return grid[mask], distances[mask], gradients[mask]
