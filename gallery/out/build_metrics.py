"""Parse accuracy_new.txt and write metrics.json for the gallery.

This script is self-contained inside `gallery/out/`. Its CWD-independent paths
mean the whole `gallery/` directory can be copied elsewhere and still work.

What it writes to metrics.json:
  <model>.<algo>.<N>  =  {"hausdorff": float, "chamfer": float, "f1": float?}
                         (parsed from `accuracy_new.txt` log)
  _samples[model]     =  sorted list of N's that have at least one algo file
  _paths[model][algo][N]   =  "out/<model>/<filename>.obj"  (filesystem scan)
  _interp_paths[model][N]  =  legacy alias of _paths[model]["interpolation"][N]
  _examples[model]    =  "examples/<model>.obj" (or first match found recursively)

Models without a ground-truth obj under ../examples/ are dropped from
`_samples` and `_paths` so the gallery dropdown only lists comparable models.

Algo file-name conventions recognised (most-specific first):
  interpolation:  ours_<N>.obj  |  interpolant_<N>_*.obj
  mes:            mes_sw10_<N>.obj  |  mes_<N>.obj
  rfta:           rfta_sw10_<N>.obj  |  rfta_<N>.obj
  mc:             mc_<N>.obj  |  sample_points_<N>.obj

When several files match a single (model, algo, N), the **shortest filename**
wins (lex order as tiebreak) — deterministic and tends to pick the "canonical"
variant over experimental-suffix ones.
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "accuracy_new.txt")
OUT = os.path.join(HERE, "metrics.json")

RE_EXPORT = re.compile(r"Exported:\s*out/([^/]+)/")


def _model_header_re() -> re.Pattern:
    """Bare model header: a line whose only content is a subdir name under out/."""
    names = sorted(
        d for d in os.listdir(HERE)
        if os.path.isdir(os.path.join(HERE, d))
    )
    if not names:
        return re.compile(r"(?!x)x")  # matches nothing
    return re.compile(r"^\s*(" + "|".join(re.escape(n) for n in names) + r")\s*$")


RE_MODEL_HEADER = _model_header_re()
RE_METRIC = re.compile(
    r"^(\S+\.obj)\s+against ground truth.*?Hausdorff:\s*([0-9.eE+-]+)\s+Chamfer:\s*([0-9.eE+-]+)"
    r"(?:\s+F1:\s*([0-9.eE+-]+))?"
)

# (regex, algo_key). Order matters — first match wins, so list the more
# specific patterns before their bare-N siblings.
ALGO_PATTERNS = [
    (re.compile(r"^ours_(\d+)\.obj$"),          "interpolation"),
    (re.compile(r"^interpolant_(\d+)_"),        "interpolation"),
    (re.compile(r"^mes_sw10_(\d+)\.obj$"),      "mes"),
    (re.compile(r"^mes_(\d+)\.obj$"),           "mes"),
    (re.compile(r"^rfta_sw10_(\d+)\.obj$"),     "rfta"),
    (re.compile(r"^rfta_(\d+)\.obj$"),          "rfta"),
    (re.compile(r"^mc_(\d+)\.obj$"),            "mc"),
    (re.compile(r"^sample_points_(\d+)\.obj$"), "mc"),
]


def classify(fname: str):
    for rx, algo in ALGO_PATTERNS:
        m = rx.match(fname)
        if m:
            return algo, int(m.group(1))
    return None, None


def main() -> None:
    # Start from existing metrics so re-runs with a missing log don't wipe data.
    data: dict = {}
    if os.path.exists(OUT):
        with open(OUT, "r") as f:
            try:
                data = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
            except Exception:
                data = {}

    if os.path.exists(LOG):
        with open(LOG, "r", encoding="utf-8", errors="replace") as f:
            lines = list(f)
    else:
        print(f"note: {LOG} not found — keeping existing metrics, only refreshing _samples / _paths")
        lines = []

    current = None
    for line in lines:
        me = RE_EXPORT.search(line)
        if me:
            current = me.group(1)
            continue
        mh = RE_MODEL_HEADER.match(line)
        if mh:
            current = mh.group(1)
            continue
        mm = RE_METRIC.match(line)
        if mm and current:
            fname = mm.group(1)
            h = float(mm.group(2)); c = float(mm.group(3))
            f1 = float(mm.group(4)) if mm.group(4) else None
            algo, n = classify(fname)
            if algo is None:
                continue
            entry = {"hausdorff": h, "chamfer": c}
            if f1 is not None:
                entry["f1"] = f1
            data.setdefault(current, {}).setdefault(algo, {})[str(n)] = entry

    # ---- Filesystem scan: per-model {algo: {N: path}} + union of N's ----
    samples_by_model: dict = {}
    paths_by_model:   dict = {}
    for model_dir in sorted(os.listdir(HERE)):
        full = os.path.join(HERE, model_dir)
        if not os.path.isdir(full):
            continue
        best: dict = {}  # (algo, N) -> filename
        for f in os.listdir(full):
            if not f.endswith(".obj"):
                continue
            algo, n = classify(f)
            if algo is None:
                continue
            key = (algo, n)
            prev = best.get(key)
            if prev is None or (len(f), f) < (len(prev), prev):
                best[key] = f
        if not best:
            continue
        algo_paths: dict = {}
        ns_union: set = set()
        for (algo, n), fname in best.items():
            algo_paths.setdefault(algo, {})[str(n)] = f"out/{model_dir}/{fname}"
            ns_union.add(n)
        samples_by_model[model_dir] = sorted(ns_union)
        paths_by_model[model_dir]   = algo_paths

    # ---- Ground truth map from ../examples/ (relative to gallery/) ----
    project_root = os.path.dirname(HERE)
    examples_dir = os.path.join(project_root, "examples")
    examples_map: dict = {}
    if os.path.isdir(examples_dir):
        for dirpath, _, files in os.walk(examples_dir):
            for fname in sorted(files):
                if not fname.endswith(".obj"):
                    continue
                model = fname[:-4]
                full = os.path.join(dirpath, fname)
                rel = os.path.relpath(full, project_root)
                existing = examples_map.get(model)
                # shallowest path wins so a file directly under examples/ beats nested copies
                if existing is None or rel.count(os.sep) < existing.count(os.sep):
                    examples_map[model] = rel

    # Drop models with no GT so the dropdown only lists comparable ones.
    samples = {m: ns for m, ns in samples_by_model.items() if m in examples_map}
    paths   = {m: p  for m, p  in paths_by_model.items()   if m in examples_map}
    skipped = sorted(set(samples_by_model) - set(samples))

    data["_samples"]      = samples
    data["_paths"]        = paths
    data["_examples"]     = examples_map
    # Legacy alias kept so older gallery.html still works without touching it.
    data["_interp_paths"] = {m: p.get("interpolation", {}) for m, p in paths.items()}

    with open(OUT, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    print("wrote", OUT)
    for m, algos in sorted(data.items()):
        if m.startswith("_"):
            continue
        if m not in samples:
            continue
        ns = samples.get(m, [])
        print(f"  {m}: " + ", ".join(f"{a}={len(v)}" for a, v in algos.items())
              + f"  samples={ns}")
    if skipped:
        print("skipped (no GT in examples/):", ", ".join(skipped))


if __name__ == "__main__":
    main()
