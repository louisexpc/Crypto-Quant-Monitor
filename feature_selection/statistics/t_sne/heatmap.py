import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

csv_path = Path("feature_selection/results/t_sne_grid/summary.csv")
out_png = csv_path.with_name("summary_heatmap.png")

data = {}
m_set, n_set = set(), set()
with csv_path.open() as f:
    reader = csv.DictReader(f)
    for row in reader:
        m, n, s = row.get("m"), row.get("n"), row.get("silhouette_msm")
        if not m or not n or not s:
            continue
        try:
            m = int(float(m))
            n = int(float(n))
            s = float(s)
        except ValueError:
            continue
        data[(m, n)] = s
        m_set.add(m)
        n_set.add(n)

m_list = sorted(m_set)
n_list = sorted(n_set)
grid = np.full((len(n_list), len(m_list)), np.nan)
for i, n in enumerate(n_list):
    for j, m in enumerate(m_list):
        if (m, n) in data:
            grid[i, j] = data[(m, n)]

mask = np.isnan(grid)
grid_masked = np.ma.array(grid, mask=mask)

fig, ax = plt.subplots(figsize=(len(m_list) * 0.7 + 2, len(n_list) * 0.6 + 2))
im = ax.imshow(grid_masked, origin="lower", aspect="auto")
ax.set_xticks(range(len(m_list)))
ax.set_xticklabels(m_list, rotation=45)
ax.set_xlabel("m (h-corr k)")
ax.set_yticks(range(len(n_list)))
ax.set_yticklabels(n_list)
ax.set_ylabel("n (K_means)")
ax.set_title("silhouette_msm heatmap")

cbar = fig.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label("silhouette_msm")

# optional: 在格子內標數值
if np.isfinite(grid).any():
    vmin, vmax = np.nanmin(grid), np.nanmax(grid)
    mid = (vmin + vmax) / 2 if np.isfinite(vmin) and np.isfinite(vmax) else None
    for i, n in enumerate(n_list):
        for j, m in enumerate(m_list):
            if not mask[i, j]:
                color = "white" if mid is not None and grid[i, j] < mid else "black"
                ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center", fontsize=7, color=color)

fig.tight_layout()
fig.savefig(out_png, dpi=160)
print(f"[OK] saved {out_png}")

