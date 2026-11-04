"""Early label check on finished candidate groups; no final layout access."""
import csv
import json
import numpy as np
from .spec import RUN
from .data_contract import SupervisedData, schema


def main():
    groups = []
    for path in (RUN / "data_candidates").glob("*/*/admission.json"):
        row = json.loads(path.read_text())
        if row["eligible_for_training"]: groups.append(row)
    maximum = 0.; checked = 0
    for row in groups:
        directory = RUN / row["episodes"][0]
        with (directory / "steps.csv").open(encoding="utf-8-sig", newline="") as stream: steps = list(csv.DictReader(stream))
        with np.load(directory / "trajectory.npz", allow_pickle=False) as data:
            expected = np.array([[float(r["target_before_"+a])-float(r["eef_before_"+a]) for a in "xyz"] for r in steps])
            maximum = max(maximum, float(np.abs(expected-data["goal_relative"]).max()))
            assert maximum < 1e-10
        contract = SupervisedData.__new__(SupervisedData); contract.directories = [directory]
        inputs, labels = contract.step(0, 0)
        assert set(inputs) == {"rgb", "proprioception_and_phase", "target_color_index"}
        assert inputs["rgb"].shape == (2, 128, 128, 3) and inputs["proprioception_and_phase"].shape == (16,)
        checked += 1
    (RUN / "input_label_schema.json").write_text(json.dumps(schema(), indent=2))
    print(json.dumps({"finished_groups_checked": checked, "goal_relative_max_error_m": maximum, "input_label_separation": True}, indent=2))


if __name__ == "__main__": main()
