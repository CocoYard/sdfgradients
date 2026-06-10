import matplotlib.pyplot as plt
from pathlib import Path

src = Path(__file__).parent.parent / "logs" / "degen_stats.txt"
groups = {}
label = None
for raw in src.read_text().splitlines():
    s = raw.strip()
    if not s or s.startswith("n "):
        continue
    parts = s.split()
    if not parts[0].lstrip("-").isdigit():
        label = s
        groups[label] = {"n": [], "vals": []}
        continue
    n = int(parts[0])
    vals = [int(x) for x in parts[1:]]
    groups[label]["n"].append(n)
    groups[label]["vals"].append(vals)

fig, ax = plt.subplots(figsize=(8, 5))
for label, g in groups.items():
    if not g["n"]:
        continue
    is_hidden = "hidden" in label.lower()
    # degen-point series use the first value column; hidden uses its single col
    y = [v[0] for v in g["vals"]]
    if is_hidden:
        ax.plot(g["n"], y, marker="x", linestyle="--", color="black",
                linewidth=2, label=label + " (reference)", zorder=5)
    else:
        ax.plot(g["n"], y, marker="o", label=label)

ax.set_xlabel("n (grid_len)")
ax.set_ylabel("count")
ax.set_title("Remaining degenerate points vs grid_len")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
out = Path(__file__).parent / "degen_stats.png"
fig.savefig(out, dpi=500)
print(f"saved {out}")
