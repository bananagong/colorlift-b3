"""Compact legacy provenance/schema check without listing thousands of files."""
from collections import Counter
import json
import numpy as np
from .spec import ROOT, RUN, freeze


def main():
    freeze(); out = {}
    for name in ("colorlift_conditional_v1", "colorlift_recovery_v1"):
        root = ROOT / "experiments" / name
        manifest = json.loads((root / "data/manifest.json").read_text())
        episodes = manifest["episodes"]
        row = next(r for r in episodes if r["split"] == "train" and r.get("eligible", False))
        path = root / row["file"] if "file" in row else root / row["directory"] / "trajectory.npz"
        with np.load(path, allow_pickle=False) as item:
            shapes = {k: list(item[k].shape) for k in item.files if k != "metadata"}
            meta = json.loads(str(item["metadata"])) if "metadata" in item else {}
        out[name] = {"counts": dict(Counter(r["split"] for r in episodes if r.get("eligible", False))),
                     "first_row": row, "shapes": shapes, "metadata_keys": list(meta),
                     "initial_positions": meta.get("initial_positions"), "initial_layout": meta.get("layout")}
    (RUN / "legacy_schema.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__": main()
