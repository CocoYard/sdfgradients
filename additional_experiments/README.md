# Additional experiments

Three experiments on how reconstruction behaves when the SDF samples are not a
clean regular grid, one on how fast the iteration converges, and one on how
sensitive the method is to its short-arc threshold. Each is one script with
every parameter written at the top of the file, so reproducing a table means
running the script and nothing else:

```bash
python additional_experiments/scattered.py     # samples at random positions
python additional_experiments/truncation.py    # samples only near the surface
python additional_experiments/noise.py         # samples with noisy distances
python additional_experiments/convergence.py   # iteration count 0..15
python additional_experiments/degen_tol.py     # short-arc threshold 1e-4..1e-8
```

Run them from anywhere; they locate the repo themselves. `_common.py` holds the
shared plumbing and is not meant to be run directly.

## What each one sweeps

| script | models | samples | swept parameter |
| --- | --- | --- | --- |
| `scattered.py` | 10 | 50³ | random positions vs. the regular-grid control |
| `truncation.py` | rossignol | 50³ | band half-width 1.0 (control), 0.1, 0.05, 0.005, 0.002 |
| `noise.py` | 10 | 50³ | Gaussian sigma 0 (control), 0.001, 0.005, 0.01|
| `convergence.py` | `MESHES` in the script, one resolution each | — | `max_iters` 0…15 with short arcs, plus `max_iters=0` without them |
| `degen_tol.py` | eiffel | 20³ and 50³ | `degen_tol` 1e-4, 1e-5 (default), 1e-6, 1e-7, 1e-8 |

Distances are in the units of the mesh after normalization to the unit cube,
which is what `generate_test_mesh_data` does to every model.

Compared methods are ours, RFTA and MES, plus marching cubes where it applies.
All of them consume the *same* samples in a given cell — `ours` generates them
and passes them to the baselines — so a difference in the output is a
difference in the method, not in its input. MC is absent from the scattered
cells because it needs a grid to run on, and from the noisy cells because the
experiment uses it as the clean-field reference.

Truncation is the one place where MC does not get the identical array, because
it cannot read one: it needs a dense grid, and the truncated *point set* leaves
it without a sign for every cell that was dropped. It receives the same
degradation as a clamped TSDF instead — `|d| > bound` saturated to `±bound`,
grid intact, values inside the band untouched — which is how range-scan TSDFs
actually store it. Clamping keeps every sign, so MC barely moves with the band
while the methods reading the point set do. `_mc_samples` in `_common.py` is
where this happens, and it is inert at `bound = 1.0`.

The method configuration is not restated here: every run takes it from the
defaults in `Options` (`SDF_to_surface_3D.py`), which are the single definition
of the main config. The scripts pass only the model, the sample count and the
degradation knob. `convergence.py` is the exception, and only because the knobs
it sweeps *are* part of that config: it passes `max_iters` and
`turn_off_short_arcs`, and nothing else.

`convergence.py` also stands apart mechanically — it borrows `_common.py` but
does not go through `sweep()`, because it needs neither the per-cell
directories nor a summary. Only `ours` is swept, the baselines having no
iteration count; they run once per mesh as a flat reference line under the
curve. Its `max_iters=0` runs are the plain RBF interpolant of the samples: the
last fit before the iteration loop uses values only, so no projection has been
applied yet. Turning short arcs off there also drops the zero-valued
degenerate-arc points, which makes that run the bare interpolant of the input
alone. Each mesh runs at one resolution, so each curve is read against its own
iteration count and not against another mesh.

## Output

```
results/<experiment>/<param>_gl<N>/<mesh>/out/<mesh>/*.obj   reconstructions
results/<experiment>/metrics.csv                            one row per reconstruction
results/<experiment>/summary.csv                            mean / std / median over meshes
```

Only the two CSVs are committed; the reconstructions are large and regenerable.
In `summary.csv` the std is the spread across models -- how much the model you
picked matters at that setting -- not an uncertainty on any single number.

`convergence` has a layout of its own, because `ours` puts every knob it sweeps
into the name it exports and so its runs cannot collide:

```
results/convergence/out/<mesh>/ours_<grid_len>_<max_iters>_<shortArcs|noShortArcs>_*.obj
results/convergence/out/<mesh>/{rfta,mes,sample_points}_<grid_len>.obj
results/convergence/metrics.csv    mesh, grid_len, algo, max_iters, short_arcs, Hausdorff, Chamfer, F1
```

There is no `summary.csv` for it: with one mesh per resolution, every group
`summarize()` could build would be a single mesh averaged with itself.

`degen_tol` has a layout of its own for the same reason as the degraded-input
experiments (the `.obj` name does not encode the threshold), plus a CSV that
carries two kinds of measurement:

```
results/degen_tol/eps<tol>_gl<N>/eiffel/out/eiffel/ours_*.obj   reconstruction
results/degen_tol/eps<tol>_gl<N>/eiffel/candidates.npz          short-arc candidates
results/degen_tol/metrics.csv                                   one row per cell
```

`degen_tol` is the total exposed-arc length -- a length in mesh units, not an
angle -- below which a sphere's exposed region is collapsed to a tangent point
and the midpoint of its short arc becomes a surface candidate: a zero-valued
point handed to the RBF fit. The CSV reports the candidates (how many the
filter saw, how many it kept, and how far the kept ones lie from the true
surface, as a mean and as the fraction within 1e-8 / 1e-7 / 1e-6 / 1e-5) next to the
reconstruction the run produced from them, which is the point of the
experiment: the candidate population moves with the threshold whether or not
the output does.

The candidates cannot be recovered from a reconstruction, so unlike every other
experiment here this one writes something besides the `.obj`: `candidates.npz`
holds them, and a cell counts as done only when both files are there. Running
the script prints the LaTeX body rows for the table as well as the CSV.

To recompute both CSVs from whatever is already on disk, without re-running or
regenerating anything:

```bash
python additional_experiments/collect.py          # every experiment
python additional_experiments/collect.py noise    # just one
```

Cells are discovered from the directory tree, so a sweep run in several batches
reports everything that finished so far. `sweep()` calls this at the end of a
run, so the CSVs mean the same thing either way. `convergence` and `degen_tol`
are not driven by `sweep()`, so `collect.py` recomputes them through their own
`report()` instead; those walk the output tree too, and report every cell
sitting there whether or not it is still in the script's lists.

Do not redirect a run's stdout into `metrics.csv` -- the shell holds that path
open while the sweep writes its own CSV to it, and the two interleave. Redirect
to a log file instead; the CSVs are written regardless.

Every cell gets its own directory because the runners name their outputs
relative to the working directory and do not always encode the degradation in
the filename (`test_mes` writes `mes_50.obj` at every noise level).
`convergence` is the exception: every algorithm it runs encodes what varies
across its runs, so they all share one directory per mesh.

Reruns skip any cell whose `.obj` files already exist, so an interrupted sweep
resumes and adding a parameter value only runs the new cells. Metrics are
recomputed from disk every time, so the CSV is complete either way. Random
draws — sample positions, noise, and the surface sampling behind Chamfer and
F1 — are seeded from `SEED` in `_common.py`, so a rerun reproduces the CSV
exactly.
