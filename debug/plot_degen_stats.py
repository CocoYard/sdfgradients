import matplotlib.pyplot as plt
from pathlib import Path

src = Path(__file__).parent / "degen_stats.txt"
groups = {}
label = None
for raw in src.read_text().splitlines():
    s = raw.strip()
    if not s or s.startswith("n "):
        continue
    parts = s.split()
    if not parts[0].lstrip("-").isdigit():
        label = s
        groups[label] = {"n": [], "degen": [], "covered": []}
        continue
    n, d, c = int(parts[0]), int(parts[1]), int(parts[2])
    groups[label]["n"].append(n)
    groups[label]["degen"].append(d)
    groups[label]["covered"].append(c)

fig, ax = plt.subplots(figsize=(8, 5))
for label, g in groups.items():
    ax.plot(g["n"], g["degen"], marker="o", label=label)

ax.set_xlabel("n (grid_len)")
ax.set_ylabel("degenerate points (remaining)")
ax.set_title("Remaining degenerate points vs grid_len")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
out = Path(__file__).parent / "degen_stats.png"
fig.savefig(out, dpi=500)
print(f"saved {out}")
