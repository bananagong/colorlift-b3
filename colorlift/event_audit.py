"""Independent audit of real traces, paired resets, contact gates and admitted data."""
import csv
import hashlib
import json
from collections import defaultdict
import numpy as np
from .event_efficiency import RUN
from .event_launch import fingerprint


def truth(value):
    return str(value).lower() in ("1", "true")


def read_rows(directory):
    with (directory / "steps.csv").open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    freeze = json.loads((RUN / "formal_freeze.json").read_text())
    current = fingerprint()
    current["protocol"] = hashlib.sha256((RUN / "protocol.json").read_bytes()).hexdigest()
    assert freeze == current
    reports = []
    reset_groups = defaultdict(list)
    scene_groups = defaultdict(list)
    goal_choices = []
    for path in sorted((RUN / "test").glob("*/*/scene_*/*/metadata.json")):
        meta = json.loads(path.read_text())
        rows = read_rows(path.parent)
        assert len(rows) == meta["control_steps"]
        assert all(abs(float(row["time_after_s"])-float(row["time_before_s"])-.05) < 1e-8 for row in rows)
        assert all(int(row["physics_samples"]) == 25 for row in rows)
        assert meta["physical_eef_path_m"]+1e-6 >= meta["control_chord_path_m"]
        for previous, following in zip(rows, rows[1:]):
            assert max(abs(float(previous["eef_after_"+a])-float(following["eef_before_"+a])) for a in "xyz") < 1e-10
        if meta["success"]:
            final = rows[-40:]
            assert [int(row["hold_streak"]) for row in final] == list(range(1, 41))
            assert min(float(row["target_lift_m"]) for row in final) > .04
            assert max(float(row["target_linear_speed"]) for row in final) <= .025
            assert max(float(row["target_angular_speed"]) for row in final) <= .4
            anchor = np.array([float(final[0]["target_after_"+axis]) for axis in "xyz"])
            assert max(np.linalg.norm(np.array([float(row["target_after_"+axis]) for axis in "xyz"])-anchor) for row in final) <= .005+1e-10
        first_lift = next((row for row in rows if int(row["phase"]) == 3), None)
        if meta["scheme"] != "baseline" and first_lift:
            assert truth(first_lift["grip_confirmed_before"])
            assert min(float(first_lift[side+"_pad_force_n_before"]) for side in ("left", "right")) >= .2
            assert float(first_lift["finger_speed_m_s_before"]) <= .02
        reset_groups[meta["seed"], meta["target_color"]].append(meta)
        scene_groups[meta["seed"]].append(meta)
        if meta["actor"] == "act":
            initial_goal = np.array([float(rows[0]["goal_"+axis]) for axis in "xy"])
            chosen = min(meta["initial_positions"], key=lambda color: np.linalg.norm(initial_goal-np.array(meta["initial_positions"][color][:2])))
            goal_choices.append({"seed": meta["seed"], "scheme": meta["scheme"], "commanded_color": meta["target_color"],
                                 "initial_goal_nearest_color": chosen, "first_goal_correct": chosen == meta["target_color"],
                                 "success": meta["success"], "initial_rgb_sha256": meta["initial_rgb_sha256"]})
        reports.append({"directory": meta["directory"], "success": meta["success"]})
    assert len(reports) == 900, len(reports)
    for key, group in reset_groups.items():
        assert len(group) == 6, (key, len(group))
        assert len({meta["initial_rgb_sha256"] for meta in group}) == 1, key
        assert max(np.max(np.abs(np.array(meta["initial_robot_state"])-np.array(group[0]["initial_robot_state"]))) for meta in group) < 1e-10
        assert all(meta["initial_positions"] == group[0]["initial_positions"] for meta in group)
    assert len(scene_groups) == 50
    for seed, group in scene_groups.items():
        assert len(group) == 18
        assert len({meta["initial_rgb_sha256"] for meta in group}) == 1, seed
        assert all(meta["initial_positions"] == group[0]["initial_positions"] for meta in group)
    with (RUN / "test/color_switching.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(goal_choices[0]))
        writer.writeheader()
        writer.writerows(goal_choices)
    verifications = [json.loads(path.read_text()) for path in sorted((RUN / "replays").glob("*_*_*/verification.json"))]
    assert len(verifications) == 210, len(verifications)
    admitted = [v for v in verifications if v["admitted"]]
    maximum_target_position_error = 0.
    for verification in admitted:
        source = RUN / verification["candidate"] / "trajectory.npz"
        data = RUN / verification["data_file"]
        assert hashlib.sha256(data.read_bytes()).hexdigest() == verification["source_npz_sha256"]
        assert hashlib.sha256(source.read_bytes()).hexdigest() == verification["source_npz_sha256"]
        assert verification["replay_success"] and verification["replay_contact_qualified"] and verification["identical_actions"]
        assert verification["maximum_state_error"] <= 2e-5
        assert verification["initial_rgb_matches"]
        original_rows = read_rows(RUN / verification["candidate"])
        replay_rows = read_rows(RUN / "replays" / (RUN / verification["candidate"]).name)
        assert len(original_rows) == len(replay_rows)
        target_error = max(abs(float(before["target_after_"+axis])-float(after["target_after_"+axis]))
                           for before, after in zip(original_rows, replay_rows) for axis in "xyz")
        maximum_target_position_error = max(maximum_target_position_error, target_error)
        assert target_error <= 2e-5
    manifest = {"episodes": verifications, "admitted": len(admitted), "rejected": len(verifications)-len(admitted),
                "train": sum(v["split"] == "train" for v in admitted), "val": sum(v["split"] == "val" for v in admitted),
                "colors": {color: sum(v["target_color"] == color for v in admitted) for color in ("red", "yellow", "blue")},
                "physical_replay_required": True, "raw_step_and_RGB_data": True}
    (RUN / "data/manifest.json").write_text(json.dumps(manifest, indent=2))
    result = {"passed": True, "formal_trials": len(reports), "paired_initial_scenes": len(reset_groups),
              "physics_path_sample_hz": 500, "mandatory_hold_steps": 40, "checkpoints_and_controller_hashes_unchanged": True,
              "identical_scene_color_command_switches": len(scene_groups),
              "first_goal_correct_by_scheme": {scheme: sum(row["first_goal_correct"] for row in goal_choices if row["scheme"] == scheme)
                                               for scheme in ("baseline", "single_step", "hybrid", "efficiency")},
              "candidate_episodes": len(verifications), "admitted_episodes": len(admitted), "rejected_episodes": len(verifications)-len(admitted),
              "maximum_replay_state_error": max(v.get("maximum_state_error", 0.) for v in verifications)}
    result["maximum_replay_target_position_error_m"] = maximum_target_position_error
    (RUN / "audit.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
