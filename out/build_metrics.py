"""Parse experiments_autoMES.txt and write out/metrics.json.

Tracks the currently-being-exported model from "Exported: out/<model>/..." lines,
then attributes subsequent "<file>.obj against ground truth... Hausdorff: H Chamfer: C"
lines to that model.

Output schema:
  { "<model>": { "<algo>": { "<N>": { "hausdorff": float, "chamfer": float } } } }
Algo keys: "interpolation", "mes", "rfta", "mc".
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

RE_INTERP = re.compile(r"interpolant_(\d+)_")
RE_MES = re.compile(r"^mes_(\d+)\.obj$")
RE_RFTA = re.compile(r"^rfta_(\d+)\.obj$")
RE_MC = re.compile(r"^sample_points_(\d+)\.obj$")


def classify(fname: str):
    for rx, algo in [(RE_INTERP, "interpolation"), (RE_MES, "mes"), (RE_RFTA, "rfta"), (RE_MC, "mc")]:
        m = rx.search(fname)
        if m:
            return algo, int(m.group(1))
    return None, None


def main() -> None:
    data: dict = {}
    current = None
    if os.path.exists(OUT):
        with open(OUT, "r") as f:
            try:
                data = {k: v for k, v in json.load(f).items() if k != "_samples"}
            except Exception:
                data = {}
    if os.path.exists(LOG):
        with open(LOG, "r", encoding="utf-8", errors="replace") as f:
            lines = list(f)
    else:
        print(f"note: {LOG} not found — keeping existing metrics, only refreshing _samples")
        lines = []
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

    # Scan out/<model>/ for interpolant_<N>_*.obj so the gallery can show a row
    # per N even when only "ours" has an output at that resolution.
    samples: dict = {}
    for model_dir in sorted(os.listdir(HERE)):
        full = os.path.join(HERE, model_dir)
        if not os.path.isdir(full):
            continue
        ns = set()
        for f in os.listdir(full):
            if not f.endswith(".obj"):
                continue
            m = RE_INTERP.match(f)
            if m:
                ns.add(int(m.group(1)))
        if ns:
            samples[model_dir] = sorted(ns)
    data["_samples"] = samples

    # Map model name → relative path to its ground-truth obj. Prefer a file
    # directly under examples/; else fall back to the first match found
    # recursively (sorted for determinism).
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
                # Prefer the shallowest path (files directly under examples/ win).
                if existing is None or rel.count(os.sep) < existing.count(os.sep):
                    examples_map[model] = rel
    data["_examples"] = examples_map

    with open(OUT, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    print("wrote", OUT)
    for m, algos in data.items():
        if m.startswith("_"):
            continue
        ns = samples.get(m, [])
        print(f"  {m}: " + ", ".join(f"{a}={len(v)}" for a, v in algos.items())
              + f"  samples={ns}")


if __name__ == "__main__":
    main()
