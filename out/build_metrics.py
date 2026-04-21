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
LOG = os.path.join(os.path.dirname(HERE), "accuracy_new.txt")
OUT = os.path.join(HERE, "metrics.json")

RE_EXPORT = re.compile(r"Exported:\s*out/([^/]+)/")
# Bare model header, e.g. a line containing only "eiffel" / "horse" / "loewe".
RE_MODEL_HEADER = re.compile(r"^\s*(eiffel|horse|loewe)\s*$")
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
    with open(LOG, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
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

    with open(OUT, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    print("wrote", OUT)
    for m, algos in data.items():
        print(f"  {m}: " + ", ".join(f"{a}={len(v)}" for a, v in algos.items()))


if __name__ == "__main__":
    main()
