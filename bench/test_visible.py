"""
Matplotlib 3D visualization for sphere arrangement.
Works with the incremental clipping approach (exposed_region_clip).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from exposed_region_clip import (
    Sphere, Cap, BoundaryArc,
    compute_all_caps, compute_exposed_region,
    point_on_cap_circle,
)


# ═══════════════════════════════════════════════════════════════════
# Drawing helpers
# ═══════════════════════════════════════════════════════════════════

def draw_sphere_wireframe(ax, sphere, color='blue', alpha=0.1):
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 25)
    x = sphere.center[0] + sphere.radius * np.outer(np.cos(u), np.sin(v))
    y = sphere.center[1] + sphere.radius * np.outer(np.sin(u), np.sin(v))
    z = sphere.center[2] + sphere.radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, alpha=alpha, color=color, edgecolor='none')


def draw_circle(ax, cap, color='green', linewidth=1.5, linestyle='-'):
    t = np.linspace(0, 2 * np.pi, 200)
    pts = (cap.circle_center[:, None]
           + cap.circle_radius * (np.cos(t) * cap.local_u[:, None]
                                  + np.sin(t) * cap.local_v[:, None]))
    ax.plot(pts[0], pts[1], pts[2], color=color, lw=linewidth, ls=linestyle)


def draw_arc(ax, cap, t_start, t_end, color='red', linewidth=3):
    n_pts = max(int((t_end - t_start) / 0.02), 20)
    t = np.linspace(t_start, t_end, n_pts)
    pts = (cap.circle_center[:, None]
           + cap.circle_radius * (np.cos(t) * cap.local_u[:, None]
                                  + np.sin(t) * cap.local_v[:, None]))
    ax.plot(pts[0], pts[1], pts[2], color=color, lw=linewidth,
            solid_capstyle='round')


def draw_degenerate_point(ax, pt, color='orangered', size=120):
    ax.scatter(*pt, color=color, s=size, zorder=10,
               edgecolors='black', linewidths=0.8, marker='*')
    ax.scatter(*pt, color='none', s=size * 3, zorder=9,
               edgecolors=color, linewidths=1.0, alpha=0.5)


def deduplicate_points(points, tol=1e-4):
    """Merge 3D points that are within tol of each other."""
    if not points:
        return []
    unique = [points[0].copy()]
    for pt in points[1:]:
        if all(np.linalg.norm(pt - u) >= tol for u in unique):
            unique.append(pt.copy())
    return unique


# ═══════════════════════════════════════════════════════════════════
# Main visualization
# ═══════════════════════════════════════════════════════════════════

def visualize_robust(main, others, title_extra='', tol=1e-6):
    # ── Compute ──
    caps = compute_all_caps(main, others)
    arcs_by_cap, degenerate_points = compute_exposed_region(main, caps, tol=tol)

    total_arc = sum(
        sum(arc.length() for arc in arcs)
        for arcs in arcs_by_cap.values()
    )
    total_arc_deg = np.degrees(total_arc)

    # ── Figure ──
    fig = plt.figure(figsize=(14, 11))
    ax = fig.add_subplot(111, projection='3d')

    cmap = plt.colormaps['tab10']
    n_others = len(others)
    sphere_colors = [cmap(i / max(n_others, 1)) for i in range(n_others)]
    n_caps = len(caps)
    cap_colors = [cmap(i / max(n_caps, 1)) for i in range(n_caps)]

    draw_sphere_wireframe(ax, main, color='cornflowerblue', alpha=0.08)
    for i, o in enumerate(others):
        draw_sphere_wireframe(ax, o, color=sphere_colors[i], alpha=0.05)
    for idx, cap in enumerate(caps):
        if cap.circle_radius < 1e-14:
            continue
        draw_circle(ax, cap, color=cap_colors[idx], linewidth=1.2)

    # Draw all arcs (including degenerate — they just appear as dots)
    n_arc_total = 0
    for arcs in arcs_by_cap.values():
        n_arc_total += len(arcs)
        for arc in arcs:
            draw_arc(ax, caps[arc.cap_idx], arc.t_start, arc.t_end,
                        color='red', linewidth=3.5)

    # Draw degenerate point markers on top
    for pt in degenerate_points:
        draw_degenerate_point(ax, pt, color='orangered', size=120)

    # ── Axis limits ──
    mr = max(main.radius, max(
        (np.linalg.norm(np.asarray(o.center) - np.asarray(main.center)) + o.radius
         for o in others),
        default=main.radius))
    lim = mr * 1.3
    mc = np.asarray(main.center)
    ax.set_xlim(mc[0] - lim, mc[0] + lim)
    ax.set_ylim(mc[1] - lim, mc[1] + lim)
    ax.set_zlim(mc[2] - lim, mc[2] + lim)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

    ax.set_title(
        f'Sphere Arrangement  {title_extra}\n'
        f'Caps: {n_caps}  |  Arcs: {n_arc_total} ({total_arc_deg:.4f}°)  |  '
        f'Degenerate points: {len(degenerate_points)}',
        fontsize=10)

    legend_handles = [
        Line2D([0], [0], color='cornflowerblue', lw=6, alpha=0.3, label='Main sphere'),
        Line2D([0], [0], color='red', lw=3, label='Exposed arcs'),
        Line2D([0], [0], marker='*', color='orangered', lw=0, ms=12,
               markeredgecolor='black', label='Degenerate points'),
    ]
    for idx, cap in enumerate(caps):
        legend_handles.append(
            Line2D([0], [0], color=cap_colors[idx], lw=1.5,
                   label=f'Cap {cap.sphere_idx} (ϕ={np.degrees(cap.phi):.1f}°)')
        )
    ax.legend(handles=legend_handles, loc='upper left', fontsize=8)
    ax.set_box_aspect([1, 1, 1])
    plt.tight_layout()

    return fig, {
        'caps': caps,
        'arcs_by_cap': arcs_by_cap,
        'degenerate_points': degenerate_points,
        'total_arc_rad': total_arc,
    }


# ═══════════════════════════════════════════════════════════════════
# Demo
# ═══════════════════════════════════════════════════════════════════

def main():
    test_cases = [
        {
            'name': '4 spheres → 2 exposed points at (±1,0,0)',
            'main': Sphere([0, 0, 0], 1.0 - 1e-6),
            'others': [
                Sphere([0, 0, 1], np.sqrt(2)),
                Sphere([0, 0, -1], np.sqrt(2)),
                Sphere([0, -1, 0], np.sqrt(2)),
                Sphere([0, 1, 0], np.sqrt(2)),
                # Sphere([0, .5, .5], np.sqrt(1)),
            ],
        },
        {
            'name': '5 spheres → 1 exposed point at (1,0,0)',
            'main': Sphere([0, 0, 0], 1.0-1e-7),
            'others': [
                Sphere([0, 0, 1], np.sqrt(2)),
                Sphere([0, 0, -1], np.sqrt(2)),
                Sphere([0, -1, 0], np.sqrt(2)),
                Sphere([0, 1, 0], np.sqrt(2)),
                Sphere([-1, 0, 0], np.sqrt(2)),
            ],
        },
        # {
        #     'name': '2 spheres → normal arcs',
        #     'main': Sphere([0, 0, 0], 1.0),
        #     'others': [
        #         Sphere([1.5, 0, 0], 1.0),
        #         Sphere([-1.5, 0, 0], 1.0),
        #     ],
        # },
        # {
        #     'name': '6 spheres → fully covered',
        #     'main': Sphere([0, 0, 0], 1.85),
        #     'others': [
        #         Sphere([0, 0, 1], np.sqrt(2)),
        #         Sphere([0, 0, -1], np.sqrt(2)),
        #         Sphere([0, -1, 0], np.sqrt(2)),
        #         Sphere([0, 1, 0], np.sqrt(2)),
        #         Sphere([1, 0, 0], np.sqrt(2)),
        #         Sphere([-1, 0, 0], np.sqrt(2)),
        #     ],
        # },
    ]

    for i, tc in enumerate(test_cases):
        print("=" * 65)
        print(f"Test {i + 1}: {tc['name']}")
        print("=" * 65)

        fig, info = visualize_robust(tc['main'], tc['others'],
                                     title_extra=f"— {tc['name']}")

        # Print caps
        caps = info['caps']
        print(f"\n  {len(caps)} cap(s):")
        for ci, cap in enumerate(caps):
            print(f"    Cap {ci}: sphere {cap.sphere_idx}, "
                  f"ϕ={np.degrees(cap.phi):.1f}°, "
                  f"circle_r={cap.circle_radius:.4f}")
            
        # print arcs by cap
        arcs_by_cap = info['arcs_by_cap']
        print(f"\n  Exposed arcs by cap {arcs_by_cap.keys()}:")
        for ci, arcs in arcs_by_cap.items():
            print(f"    Cap {ci}: {len(arcs)} arc(s)")
            for ai, arc in enumerate(arcs):
                print(f"      Arc {ai}: t_start={arc.t_start:.4f}, "
                      f"t_end={arc.t_end:.4f}, length={arc.length():.4f} rad")

        # Print degenerate points (deduplicated)
        dpts = info['degenerate_points']
        if dpts:
            print(f"\n  Degenerate points (deduplicated): {len(dpts)}")
            for ei, pt in enumerate(dpts):
                print(f"    point {ei}: ({pt[0]:+.6f}, {pt[1]:+.6f}, {pt[2]:+.6f})")

        print(f"\n  Total arc: {np.degrees(info['total_arc_rad']):.6f}°")
        print()

        plt.show()
        plt.close(fig)


if __name__ == '__main__':
    main()