"""A separate physical rerun of the one-channel/one-level initial-render difference."""
import csv
import json
import numpy as np
import torch
from .spatial_gate import RUN, CHECKPOINT
from .spatial_env import make_spatial_env
from .spatial_rollout import episode, load_policy


def main():
    source = RUN / "evaluations/test/baseline/medium/layout_2210012/blue"
    original = json.loads((source / "metadata.json").read_text())
    torch.set_num_threads(1)
    env = make_spatial_env(1)
    model, checkpoint = load_policy(CHECKPOINT)
    try:
        new = episode(env, original["layout"], "blue", RUN / "reset_diagnosis/physical_retest", "baseline", model, checkpoint)
    finally:
        env.close()
    with (source / "steps.csv").open(encoding="utf-8-sig", newline="") as stream: old_rows = list(csv.DictReader(stream))
    with (RUN / "reset_diagnosis/physical_retest/steps.csv").open(encoding="utf-8-sig", newline="") as stream: new_rows = list(csv.DictReader(stream))
    differences = {key: [original[key], new[key]] for key in ("success", "eligible", "wrong_color_grasp", "collision_steps", "control_steps")}
    print("ORIGINAL_AND_RERUN", json.dumps(differences), flush=True)
    same_outcomes = all(original[key] == new[key] for key in ("success", "eligible", "wrong_color_grasp"))
    maximum_action_difference = max(abs(float(a[f"osc_action_{i}"])-float(b[f"osc_action_{i}"])) for a, b in zip(old_rows, new_rows) for i in range(7))
    report = {"completed": True, "same_task_and_contact_outcomes": same_outcomes, "original_and_rerun": differences,
              "maximum_action_difference": maximum_action_difference, "steps": new["control_steps"],
              "source_initial_hash": original["initial_rgb_sha256"], "rerun_initial_hash": new["initial_rgb_sha256"],
              "diagnostic_only_not_added_to_primary_trial_denominators": True}
    (RUN / "reset_diagnosis/physical_retest_result.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__": main()
