import csv
import json
import numpy as np
from .spec import RUN


def main():
    path = next((RUN / "data_candidates").glob("*/*/admission.json"))
    row = json.loads(path.read_text()); directory = RUN / row["episodes"][0]
    with (directory / "steps.csv").open(encoding="utf-8-sig", newline="") as stream: steps = list(csv.DictReader(stream))
    with np.load(directory / "trajectory.npz", allow_pickle=False) as data:
        eef = np.array([[float(r["eef_before_"+a]) for a in "xyz"] for r in steps])
        target = np.array([[float(r["target_before_"+a]) for a in "xyz"] for r in steps])
        err = data["goal_relative"]-(target-eef)
        index = int(np.abs(err).max(axis=1).argmax())
        result = {"directory": str(directory), "max_delta_error_m": float(np.abs(err).max()),
                  "max_input_EEF_error_m": float(np.abs(data["states"][:, 9:12]-eef).max()),
                  "max_target_label_error_m": float(np.abs(data["target_position_label"]-target).max()),
                  "max_input_based_delta_error_m": float(np.abs(data["goal_relative"]-(data["target_position_label"]-data["states"][:, 9:12])).max()),
                  "worst_index": index, "raw_goal_relative": data["goal_relative"][index].tolist(), "physical_delta": (target-eef)[index].tolist(),
                  "eef_input": data["states"][index, 9:12].tolist(), "eef_snapshot": eef[index].tolist(),
                  "first_indices": {"EEF_input": data["states"][:4, 9:12].tolist(), "EEF_snapshot": eef[:4].tolist()}}
    print(json.dumps(result, indent=2), flush=True)
    (RUN / "alignment_diagnosis.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__": main()
