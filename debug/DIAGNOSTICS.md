# Diagnostic instrumentation (off-tree)

Stash for the env-var-gated dump code used while investigating the degen-point
filter (the spheres where `find_degen_pts` fires + `filter_degenerate_pts`
picks a candidate via RBF). The blocks are not compiled into the source tree;
paste them back in when you need to repeat the workflow.

## Workflow that uses these blocks

1. Build, then run with `SDF_DUMP_DEGEN_PTS=degen_dump.txt python SDF_to_surface_3D.py`
   to dump post-filter degen points (sphere idx + xyz) per grid_len.
2. `python debug/find_worst_degen.py degen_dump.txt examples/<name>.obj`
   loads the gt mesh (normalised the same way as the SDF pipeline), finds the
   sphere whose degen point lies farthest from gt, prints its idx.
3. Re-run with `SDF_DEBUG_SPHERE_IDX=<idx> SDF_DEBUG_SPHERE_OUT=sphere_<idx>.txt`
   to get the full geometry dump (neighbors / kept caps / arcs / degen pts)
   for that single sphere. Same env var also makes Step 2 of
   `filter_degenerate_pts` print the candidates + RBF preds on stderr.
4. `python debug/plot_sphere_330898.py` (after editing the data block to point
   at the new sphere) builds an interactive plotly HTML view.

## Block A — `cpp/src/main_algorithm.cpp`

Both blocks need `#include <cstdio>` and `#include <cstdlib>` near the top of
the file (currently transitively pulled in, but add explicitly if you ever see
build errors on those names).

### A.1 RBF candidate dump inside `filter_degenerate_pts` Step 2

Inside `if (!ambig_idx.empty()) { ... }` block, right after the
`Eigen::VectorXd preds = interpolator.predict(P);` line and before the
`int off = 0;` loop:

```cpp
        // Diagnostic: when SDF_DEBUG_SPHERE_IDX matches an ambiguous sphere,
        // dump the candidates and their |predicted SDF| so we can see whether
        // RBF picked the geometrically-closest-to-surface candidate.
        int dbg_idx = -1;
        if (const char* dbg_env = std::getenv("SDF_DEBUG_SPHERE_IDX"))
            dbg_idx = std::atoi(dbg_env);
```

Then inside the `for (size_t i = 0; i < ambig_idx.size(); i++)` loop, after
the inner `for (k = 1; ...)` that computes `best`, insert before
`degenerate_pts[ambig_idx[i]] = { c[best] };`:

```cpp
            if ((int)ambig_idx[i] == dbg_idx) {
                std::fprintf(stderr,
                    "[FILTER_DBG] sphere %d: %zu candidates, RBF pred (signed):\n",
                    dbg_idx, c.size());
                for (int k = 0; k < (int)c.size(); k++) {
                    std::fprintf(stderr,
                        "[FILTER_DBG]   cand %d: (%.17g, %.17g, %.17g)  pred=%.6g%s\n",
                        k, c[k].x(), c[k].y(), c[k].z(),
                        preds(off + k),
                        k == best ? "   <-- PICKED" : "");
                }
                std::fflush(stderr);
            }
```

### A.2 Post-filter degen point dump

Right after the `filter_degenerate_pts(...)` call (and the `if (false) {
... degen_stats.txt ... }` block) in `init_gradients_by_degenerate_pts`:

```cpp
    // Diagnostic: dump post-filter degen points "<sphere_idx> x y z" to
    // SDF_DUMP_DEGEN_PTS path (append mode) so we can post-process with the
    // gt mesh in Python to find the sphere whose degen pt is farthest from
    // the true surface.
    if (const char* dump_path = std::getenv("SDF_DUMP_DEGEN_PTS")) {
        if (FILE* fp = std::fopen(dump_path, "a")) {
            std::fprintf(fp, "# grid_len=%d count=%zu\n",
                         options.grid_len, degenerate_pts.size());
            for (const auto& [idx, pts] : degenerate_pts) {
                if (pts.empty()) continue;
                const auto& p = pts[0];
                std::fprintf(fp, "%d %.17g %.17g %.17g\n",
                             idx, p.x(), p.y(), p.z());
            }
            std::fclose(fp);
        }
    }
```

## Block B — `cpp/src/existing_modules_adapter.cpp`

Inside `compute_exposed_batch`'s parallel `for (int i = 0; i < n; i++)` loop,
right after the `[SLOW SPHERE]` block (after the `std::fflush(stderr);`):

```cpp
        // Diagnostic: if SDF_DEBUG_SPHERE_IDX matches this sphere, dump its
        // arcs, kept caps, and degen points to SDF_DEBUG_SPHERE_OUT (default
        // "debug_sphere.txt"). Used to inspect why a specific sphere produced
        // a degen pt far from the gt surface. Writes once per matching call,
        // overwrite mode so re-runs replace the previous dump.
        if (const char* dbg_env = std::getenv("SDF_DEBUG_SPHERE_IDX")) {
            int dbg_idx = std::atoi(dbg_env);
            if (dbg_idx == i) {
                const char* out_path = std::getenv("SDF_DEBUG_SPHERE_OUT");
                if (!out_path) out_path = "debug_sphere.txt";
                if (FILE* fp = std::fopen(out_path, "w")) {
                    std::fprintf(fp,
                        "# sphere_idx=%d\n"
                        "# center=(%.17g,%.17g,%.17g) radius=%.17g\n"
                        "# n_nbrs=%d n_use=%d ncaps=%d narcs=%d npts=%d total_arc=%.17g dt=%.3fs\n",
                        i, centers[i*3], centers[i*3+1], centers[i*3+2], radii[i],
                        n_nbrs, n_use, ncaps, narcs, npts, total_arc, _dt_co);

                    // Neighbor list (global sphere ids).
                    std::fprintf(fp, "# --- neighbors (global ids) ---\n");
                    for (int k = 0; k < n_use; k++) {
                        int gj = nb[k];
                        std::fprintf(fp, "nbr %d %d %.17g %.17g %.17g %.17g\n",
                                     k, gj,
                                     centers[gj*3], centers[gj*3+1], centers[gj*3+2],
                                     radii[gj]);
                    }

                    // Kept caps (in remap order; _caps[c].sphere_idx is the
                    // LOCAL index into nb[]).
                    std::fprintf(fp, "# --- kept caps (kept_idx local_nbr global_nbr nx ny nz d phi circle_r ccx ccy ccz) ---\n");
                    for (int orig = 0; orig < n_pcaps; orig++) {
                        int kept_idx = remap[orig];
                        if (kept_idx < 0) continue;
                        const auto& C = _caps[orig];
                        int gnb = (C.sphere_idx >= 0 && C.sphere_idx < n_use)
                                  ? nb[C.sphere_idx] : -1;
                        std::fprintf(fp,
                            "cap %d %d %d %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g\n",
                            kept_idx, C.sphere_idx, gnb,
                            C.normal.x, C.normal.y, C.normal.z, C.d, C.phi,
                            C.circle_radius,
                            C.circle_center.x, C.circle_center.y, C.circle_center.z);
                    }

                    // Surviving arcs (cap_idx is the kept-index after remap).
                    std::fprintf(fp, "# --- arcs (a kept_cap t_start t_end angular_extent arc_length) ---\n");
                    for (int a = 0; a < narcs; a++) {
                        // Back-map kept→orig to fetch host circle_radius for
                        // the arc length.
                        int kept = _ac[a];
                        int orig = -1;
                        for (int k = 0; k < n_pcaps; k++) {
                            if (remap[k] == kept) { orig = k; break; }
                        }
                        double R = (orig >= 0) ? _caps[orig].circle_radius : 0.0;
                        double dt = _ae[a] - _as[a];
                        std::fprintf(fp,
                            "arc %d %d %.17g %.17g %.17g %.17g\n",
                            a, _ac[a], _as[a], _ae[a], dt, R * dt);
                    }

                    // Degen points emitted by this sphere.
                    std::fprintf(fp, "# --- degen pts (p x y z) ---\n");
                    for (int p = 0; p < npts; p++) {
                        std::fprintf(fp, "deg %d %.17g %.17g %.17g\n",
                                     p, _dp[p].x, _dp[p].y, _dp[p].z);
                    }
                    std::fclose(fp);
                    std::fprintf(stderr, "[DEBUG_SPHERE] wrote %s for i=%d\n",
                                 out_path, i);
                }
            }
        }
```
