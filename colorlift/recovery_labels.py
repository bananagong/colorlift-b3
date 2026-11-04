"""Export unambiguous ACT macro targets beside untouched raw IK/OSC trajectories."""
import csv
import json
import numpy as np
from .config import ROOT

RUN=ROOT / "experiments/colorlift_recovery_v1"


def main():
    manifest=json.loads((RUN / "data/manifest.json").read_text())
    valid=0
    for trial in manifest["episodes"]:
        directory=RUN / trial["directory"]
        with np.load(directory / "trajectory.npz", allow_pickle=False) as data:
            actions=data["actions"].copy(); phase=data["states"][:, 12:16].argmax(1)
            supervised=data["supervised"].copy()
        with (directory / "steps.csv").open(encoding="utf-8-sig") as stream: trace=list(csv.DictReader(stream))
        target=np.array([[float(row["target_before_"+axis]) for axis in "xy"] for row in trace])
        actions[phase<2, :2]=target[phase<2]
        actions[phase==0, 2]=.865; actions[phase==1, 2]=.818
        np.savez_compressed(directory / "training_macro_labels.npz", actions=actions, supervised=supervised,
                            semantics="ACT task macro target; temporary recovery waypoints remain in raw trajectory.npz and steps.csv")
        valid+=int(supervised.sum())
    (RUN / "label_semantics.json").write_text(json.dumps({"episodes": len(manifest["episodes"]), "supervised_steps": valid,
         "raw_actions": "trajectory.npz/actions: temporary expert IK waypoint XYZ + phase; osc_actions are real 7D OSC commands",
         "training_actions": "training_macro_labels.npz/actions: ACT task macro target, body XY before the control interval for phase0/1; nominal heights .865/.818",
         "disturbance_labels_masked": True, "future_chunks_cannot_cross_disturbances": True,
         "student_input": "RGB + 16D robot/phase state + target color ID, no body pose or perturbation flag"}, indent=2))
    print("EXPORTED_MACRO_LABELS", len(manifest["episodes"]), valid)


if __name__ == "__main__": main()
