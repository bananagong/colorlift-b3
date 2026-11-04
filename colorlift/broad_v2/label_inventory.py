import csv
import json
import numpy as np
from .spec import RUN


def main():
    total = 0; target_max = 0.; delta_max = 0.; raw_formula_max = 0.
    for marker in (RUN / "data_candidates").glob("*/*/admission.json"):
        group = json.loads(marker.read_text())
        for relative in group["episodes"]:
            directory = RUN / relative
            with (directory / "steps.csv").open(encoding="utf-8-sig", newline="") as stream: rows = list(csv.DictReader(stream))
            target = np.array([[float(r["target_before_"+a]) for a in "xyz"] for r in rows])
            eef = np.array([[float(r["eef_before_"+a]) for a in "xyz"] for r in rows])
            with np.load(directory / "trajectory.npz", allow_pickle=False) as data:
                target_max = max(target_max, float(np.abs(target-data["target_position_label"]).max()))
                delta_max = max(delta_max, float(np.abs(data["goal_relative"]-(target-eef)).max()))
                raw_formula_max = max(raw_formula_max, float(np.abs(data["goal_relative"]-(data["target_position_label"]-data["states"][:, 9:12])).max()))
            total += 1
    result = {"completed_episodes": total, "raw_target_vs_physics_target_max_m": target_max,
              "raw_relative_vs_physics_relative_max_m": delta_max, "raw_relative_input_consistency_max_m": raw_formula_max}
    (RUN / "raw_label_inventory.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__": main()
