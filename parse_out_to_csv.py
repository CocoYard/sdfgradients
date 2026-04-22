import csv
import re
from pathlib import Path

SRC = Path(__file__).parent / "out.txt"
DST = Path(__file__).parent / "out.csv"

metric_re = re.compile(
    r"^(?P<fn>\S+\.obj)\s+against ground truth\.\.\.\s+"
    r"Hausdorff:\s*(?P<h>[0-9.eE+-]+)\s+"
    r"Chamfer:\s*(?P<c>[0-9.eE+-]+)"
)
interp_re = re.compile(
    r"^interpolant_(?P<grid>\d+)_13_"
    r"(?P<sa>no)?[Ss]hortArcs_"
    r"(?:no)?clamp_"
    r"(?P<mes>no)?MES_"
    r"(?P<post>no)?post_"
    r"dc_PU_reg0"
    r"(?P<nt>_notrunc)?\.obj$"
)
name_re = re.compile(r"^([A-Za-z][A-Za-z0-9_\-]*)\s*$")

current_name = ""
rows = []

for line in SRC.read_text().splitlines():
    stripped = line.rstrip()
    nm = name_re.match(stripped)
    if nm and "Hausdorff" not in stripped:
        current_name = nm.group(1)
        continue
    m = metric_re.match(stripped)
    if not m:
        continue
    im = interp_re.match(m.group("fn"))
    if not im:
        continue
    rows.append({
        "name": current_name,
        "grid_len": int(im.group("grid")),
        "short_arc": 0 if im.group("sa") else 1,
        "MES": 0 if im.group("mes") else 1,
        "post": 0 if im.group("post") else 1,
        "truncate": 0 if im.group("nt") else 1,
        "Hausdorff": m.group("h"),
        "Chamfer": m.group("c"),
    })

with DST.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=[
        "name", "grid_len", "short_arc", "MES", "post", "truncate",
        "Hausdorff", "Chamfer",
    ])
    w.writeheader()
    w.writerows(rows)

print(f"wrote {len(rows)} rows to {DST}")
