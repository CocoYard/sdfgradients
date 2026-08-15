# Degraded-input experiments

Three experiments on how reconstruction behaves when the SDF samples are not a
clean regular grid. Each is one script with every parameter written at the top
of the file, so reproducing a table means running the script and nothing else:

```bash
python additional_experiments/scattered.py     # samples at random positions
python additional_experiments/truncation.py    # samples only near the surface
python additional_experiments/noise.py         # samples with noisy distances
```

Run them from anywhere; they locate the repo themselves. `_common.py` holds the
shared plumbing and is not meant to be run directly.

## What each one sweeps

| script | models | samples | swept parameter |
| --- | --- | --- | --- |
| `scattered.py` | 10 | 50³ | random positions vs. the regular-grid control |
| `truncation.py` | rossignol | 50³ | band half-width 1.0 (control), 0.1, 0.05, 0.005, 0.002 |
| `noise.py` | 10 | 50³ | Gaussian sigma 0 (control), 0.001, 0.005, 0.01|

Distances are in the units of the mesh after normalization to the unit cube,
which is what `generate_test_mesh_data` does to every model.

Compared methods are ours, RFTA and MES, plus marching cubes where it applies.
All of them consume the *same* samples in a given cell — `ours` generates them
and passes them to the baselines — so a difference in the output is a
difference in the method, not in its input. MC is absent from the scattered
cells because it needs a grid to run on, and from the noisy cells because the
experiment uses it as the clean-field reference.

The method configuration is not restated here: every run takes it from the
defaults in `Options` (`SDF_to_surface_3D.py`), which are the single definition
of the main config. The scripts pass only the model, the sample count and the
degradation knob.

## Output

```
results/<experiment>/<param>_gl<N>/<mesh>/out/<mesh>/*.obj   reconstructions
results/<experiment>/metrics.csv                            one row per reconstruction
results/<experiment>/summary.csv                            mean / std / median over meshes
```

Only the two CSVs are committed; the reconstructions are large and regenerable.
In `summary.csv` the std is the spread across models -- how much the model you
picked matters at that setting -- not an uncertainty on any single number.

To recompute both CSVs from whatever is already on disk, without re-running or
regenerating anything:

```bash
python additional_experiments/collect.py          # every experiment
python additional_experiments/collect.py noise    # just one
```

Cells are discovered from the directory tree, so a sweep run in several batches
reports everything that finished so far. `sweep()` calls this at the end of a
run, so the CSVs mean the same thing either way.

Do not redirect a run's stdout into `metrics.csv` -- the shell holds that path
open while the sweep writes its own CSV to it, and the two interleave. Redirect
to a log file instead; the CSVs are written regardless.

Every cell gets its own directory because the runners name their outputs
relative to the working directory and do not always encode the degradation in
the filename (`test_mes` writes `mes_50.obj` at every noise level).

Reruns skip any cell whose `.obj` files already exist, so an interrupted sweep
resumes and adding a parameter value only runs the new cells. Metrics are
recomputed from disk every time, so the CSV is complete either way. Random
draws — sample positions, noise, and the surface sampling behind Chamfer and
F1 — are seeded from `SEED` in `_common.py`, so a rerun reproduces the CSV
exactly.
