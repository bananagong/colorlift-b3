"""Development-only previews, with no final geometry access."""
import json
import numpy as np
from PIL import Image
from .spec import RUN, CATEGORIES, REGIONS


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    pools = {c: json.loads((RUN / "development" / c / "pool_completed.json").read_text())["valid_layouts"] for c in CATEGORIES}
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), constrained_layout=True)
    for ax, (category, rows) in zip(axes.ravel(), pools.items()):
        xy = np.array([r["xy"] for r in rows]).reshape(-1, 2)
        ax.scatter(xy[:, 0], xy[:, 1], s=19, alpha=.6, color="#377ca7")
        for x0, x1, y0, y1 in REGIONS: ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, facecolor="#ddd", hatch="///", edgecolor="#777", alpha=.6))
        ax.set(title=f"{category}: 64 valid development geometries", xlim=(-.155, .155), ylim=(-.155, .155), xlabel="World X (m)", ylabel="World Y (m)", aspect="equal")
        ax.grid(alpha=.2)
    fig.suptitle("Measured development coverage / all 3 slots, no object-color labels\nShaded areas reserved for Spatial-OOD; final layouts not shown")
    fig.savefig(RUN / "development_position_coverage.png", dpi=150); plt.close(fig)
    fig, axes = plt.subplots(4, 4, figsize=(10, 11), constrained_layout=True)
    for i, category in enumerate(CATEGORIES):
        for j, row in enumerate(pools[category][:4]):
            image = Image.open(RUN / "development" / category / "pool" / row["geometry_id"] / "initial_two_views.png")
            ax = axes[i, j]; ax.imshow(image.crop((0, 0, image.height, image.height)))
            ax.set_title(f'{category}\nseed {row["seed"]}', fontsize=9); ax.axis("off")
    fig.suptitle("16 different physical development layouts / front camera only\nOne panel = one independent scene; no final-test images")
    fig.savefig(RUN / "16_development_layouts_preview.png", dpi=145); plt.close(fig)


if __name__ == "__main__": main()
