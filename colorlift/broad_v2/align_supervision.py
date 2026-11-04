"""Add physics-boundary labels without modifying raw trajectories or observations."""
import csv
from datetime import datetime, timezone
import json
import numpy as np
from .spec import RUN, sha, write_once


def define_rule():
    write_once(RUN / "label_alignment_protocol.json", {
        "source": "before-action world positions and time_before_s in the non-invasive PhysicsRecorder CSV",
        "output": "supervision.npz, separate from immutable original trajectory.npz",
        "delta_definition": "target_before - eef_before at identical physical control boundary",
        "inputs": "original recorded RGB and robot observations are retained, including their sensor/update timing",
        "raw_observation_relative_label": "retained only as an explicitly named diagnostic",
        "ACT_training_updates_at_rule_definition": 0, "final_policy_results_read": 0,
        "final_layouts_and_frozen_generation_rules_modified": False,
        "reason": "early development audit detected observation EEF vs physical-boundary EEF differences; do not silently loosen label tolerance",
    })


def main():
    define_rule()
    manifest = json.loads((RUN / "training_manifest.json").read_text())
    rows = []; max_raw_difference = 0.
    for layout in manifest["layouts"]:
        for relative in layout["episodes"]:
            directory = RUN / relative
            with (directory / "steps.csv").open(encoding="utf-8-sig", newline="") as stream: steps = list(csv.DictReader(stream))
            eef = np.array([[float(r["eef_before_"+a]) for a in "xyz"] for r in steps])
            target = np.array([[float(r["target_before_"+a]) for a in "xyz"] for r in steps])
            time = np.array([float(r["time_before_s"]) for r in steps])
            with np.load(directory / "trajectory.npz", allow_pickle=False) as data:
                assert np.abs(data["target_position_label"]-target).max() < 1e-10
                raw_delta = data["goal_relative"].copy()
                observed_eef = target-raw_delta
                assert np.abs(data["states"][:, 9:12]-observed_eef).max() < 1e-7
                raw_difference = float(np.abs(raw_delta-(target-eef)).max())
                max_raw_difference = max(max_raw_difference, raw_difference)
                values = {"goal_relative": target-eef, "target_position": target, "eef_position_physics": eef,
                          "time_before_s": time, "pregrasp_position": data["pregrasp_position_label"].copy(),
                          "pregrasp_relative": data["pregrasp_position_label"]-eef,
                          "pregrasp_rotation": data["pregrasp_rotation_label"].copy(),
                          "pregrasp_supervision_mask": data["pregrasp_supervision_mask"].copy(),
                          "raw_observation_relative_label": raw_delta, "eef_observation": observed_eef}
            destination = directory / "supervision.npz"
            raw_hash = sha(directory / "trajectory.npz")
            np.savez_compressed(destination, **values)
            assert sha(directory / "trajectory.npz") == raw_hash
            rows.append({"directory": relative, "raw_trajectory_sha256": raw_hash, "steps_sha256": sha(directory / "steps.csv"),
                         "supervision_sha256": sha(destination), "raw_observation_vs_physics_delta_max_m": raw_difference})
    result = {"written_at_UTC": datetime.now(timezone.utc).isoformat(), "episodes": len(rows),
              "raw_observation_vs_physics_delta_max_m": max_raw_difference, "raw_trajectories_modified": False,
              "model_inputs_modified": False, "physics_labels_max_error_m": 0., "episodes_index": rows}
    (RUN / "supervision_manifest.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "episodes_index"}, indent=2), flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("--rule-only", action="store_true"); args = parser.parse_args()
    if args.rule_only: define_rule()
    else: main()
