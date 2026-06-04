"""Interactive 3D plot of sphere 330898's local geometry.

Outputs sphere_330898.html that you can open in a browser to rotate/zoom.

Shows:
  - main sphere (translucent)
  - 3 neighbor spheres that produced the kept caps (translucent, distinct colors)
  - the 3 cap circles on the main sphere
  - the 2 degenerate points (large markers)
  - the gt mesh segment near these points
"""
import numpy as np
import plotly.graph_objects as go
import trimesh
from pathlib import Path


# ── Data from sphere_330898.txt ──────────────────────────────────────
MAIN_CENTER = np.array([-0.094815129008615967, -0.19999999999999996, 0.11395679322264736])
MAIN_RADIUS = 0.12742232875677365

# (label, center, radius, color) — neighbor spheres for caps 0, 1, 2
NEIGHBORS = [
    ("cap 0 (nbr 330099)", np.array([-0.11309274423919254, -0.19999999999999996, 0.11630641782517616]), 0.14150019744908687, "tomato"),
    ("cap 1 (nbr 330097)", np.array([-0.11309274423919254, -0.19999999999999996, 0.11160716862011853]), 0.13821917008942819, "royalblue"),
    ("cap 2 (nbr 330194)", np.array([-0.11080804233537046, -0.19999999999999996, 0.10455829481253211]), 0.13175347870909099, "mediumseagreen"),
]

# (label, normal, d, circle_center, circle_radius, color) — kept caps
CAPS = [
    ("cap 0", np.array([-0.99183820822582835, 0, 0.12750281841347058]),
     0.015064998217897155,
     np.array([-0.0020722244136341722, -0.19999999999999996, 0.10203450425894231]),
     0.086562477661809828, "tomato"),
    ("cap 1", np.array([-0.99183820822582813, 0, -0.12750281841347205]),
     0.010906769756021895,
     np.array([-0.026770380567874553, -0.19999999999999996, 0.1227040839929328]),
     0.10737712502956857, "royalblue"),
    ("cap 2", np.array([-0.86214846851445071, 0, -0.5066557196343362]),
     0.003026149536598223,
     np.array([-0.076725781888989569, -0.19999999999999996, 0.12458729541260283]),
     0.12568300525438719, "mediumseagreen"),
]

DEGEN_PTS = [
    ("deg 0 (y=-0.2047)", np.array([-0.01309274423919137, -0.20472290992204922, 0.016306417825183422]), "deeppink"),
    ("deg 1 (y=-0.1953) PICKED", np.array([-0.013092744239191373, -0.19527709007795074, 0.016306417825183415]), "gold"),
]

MESH_PATH = "examples/43665.obj"


# ── Helpers ──────────────────────────────────────────────────────────
def sphere_surface(center, radius, n=40):
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


def circle_points(center, radius, normal, n=200):
    """Sample a circle of given center/radius in the plane perpendicular to normal."""
    n_hat = normal / np.linalg.norm(normal)
    # pick any vector not parallel to normal
    tmp = np.array([0.0, 0.0, 1.0]) if abs(n_hat[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(n_hat, tmp); u /= np.linalg.norm(u)
    v = np.cross(n_hat, u)
    t = np.linspace(0, 2 * np.pi, n)
    pts = center[None, :] + radius * (np.cos(t)[:, None] * u[None, :] + np.sin(t)[:, None] * v[None, :])
    return pts


def normalized_mesh(path):
    m = trimesh.load(path, force="mesh")
    V = np.asarray(m.vertices, dtype=np.float64)
    mn, mx = V.min(0), V.max(0)
    V = (V - (mn + mx) / 2) / float((mx - mn).max())
    F = np.asarray(m.faces, dtype=np.int32)
    return V, F


# ── Build figure ─────────────────────────────────────────────────────
fig = go.Figure()

# Main sphere
x, y, z = sphere_surface(MAIN_CENTER, MAIN_RADIUS)
fig.add_trace(go.Surface(
    x=x, y=y, z=z,
    colorscale=[[0, "lightgray"], [1, "lightgray"]],
    opacity=0.20, showscale=False, name="main sphere",
    hoverinfo="name",
))

# Neighbor spheres (translucent, colored)
for label, c, r, color in NEIGHBORS:
    xn, yn, zn = sphere_surface(c, r)
    fig.add_trace(go.Surface(
        x=xn, y=yn, z=zn,
        colorscale=[[0, color], [1, color]],
        opacity=0.10, showscale=False, name=label,
        hoverinfo="name",
    ))

# Cap circles on main sphere
for label, normal, d, ccenter, crad, color in CAPS:
    pts = circle_points(ccenter, crad, normal)
    fig.add_trace(go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        mode="lines",
        line=dict(color=color, width=6),
        name=f"{label} circle",
    ))

# Degenerate points
for label, p, color in DEGEN_PTS:
    fig.add_trace(go.Scatter3d(
        x=[p[0]], y=[p[1]], z=[p[2]],
        mode="markers+text",
        marker=dict(size=8, color=color, symbol="diamond"),
        text=[label], textposition="top right",
        name=label,
    ))

# Main sphere center marker
fig.add_trace(go.Scatter3d(
    x=[MAIN_CENTER[0]], y=[MAIN_CENTER[1]], z=[MAIN_CENTER[2]],
    mode="markers", marker=dict(size=4, color="black"),
    name="main center",
))

# gt mesh (normalized) — clipped to a y-window around the sphere
if Path(MESH_PATH).exists():
    V, F = normalized_mesh(MESH_PATH)
    y_min, y_max = MAIN_CENTER[1] - 2 * MAIN_RADIUS, MAIN_CENTER[1] + 2 * MAIN_RADIUS
    face_y = V[F].mean(axis=1)[:, 1]
    keep = (face_y >= y_min) & (face_y <= y_max)
    Fk = F[keep]
    if len(Fk):
        fig.add_trace(go.Mesh3d(
            x=V[:, 0], y=V[:, 1], z=V[:, 2],
            i=Fk[:, 0], j=Fk[:, 1], k=Fk[:, 2],
            color="purple", opacity=0.35, name="gt mesh (local)",
            hoverinfo="name",
        ))

fig.update_layout(
    title="sphere 330898: main + 3 neighbors + cap circles + degen pts + gt",
    scene=dict(
        xaxis_title="x", yaxis_title="y", zaxis_title="z",
        aspectmode="data",
    ),
    legend=dict(orientation="v", x=1.02, y=1),
)

out = Path(__file__).parent / "sphere_330898.html"
fig.write_html(out)
print(f"saved {out}")
