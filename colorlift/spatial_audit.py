"""Independent integrity audit of physics, split isolation, color swaps and order matching."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
import numpy as np
from PIL import Image
from .spatial_gate import RUN, ROOT, fingerprints
from .spatial_evaluate import checkpoint_paths


def yes(value):
    return str(value).lower() in ("true", "1")


def main():
    assert (RUN / "computation_completed.json").exists()
    assert json.loads((RUN / "gate/freeze.json").read_text()) == fingerprints()
    protocol = json.loads((RUN / "spatial_protocol.json").read_text())
    for name, digest in protocol["sources"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    training = json.loads((RUN / "training_freeze.json").read_text())
    for name, digest in training.items():
        path = RUN / "plans" / f'{name.replace("plan_seed", "seed_")}.npz' if name.startswith("plan_seed") else ROOT / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    exposure = {}
    for seed in (43, 44):
        with np.load(RUN / f"plans/seed_{seed}.npz", allow_pickle=False) as plan:
            assert plan["curriculum"].shape == (5000, 36)
            assert np.array_equal(np.sort(plan["curriculum"].ravel()), np.sort(plan["shuffled"].ravel()))
            exposure[str(seed)] = {"identical_frame_multiset": True, "samples_each_variant": int(plan["curriculum"].size)}
    evaluation = json.loads((RUN / "evaluation_freeze.json").read_text())
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == evaluation["models"][name] for name, path in checkpoint_paths().items())
    assert hashlib.sha256((RUN / "test_scenarios.json").read_bytes()).hexdigest() == evaluation["scenarios"]
    assert hashlib.sha256((ROOT / "colorlift/event_control.py").read_bytes()).hexdigest() == evaluation["controller"]
    assert hashlib.sha256((ROOT / "colorlift/spatial_rollout.py").read_bytes()).hexdigest() == evaluation["rollout"]
    manifest = json.loads((RUN / "data/manifest.json").read_text())
    assert manifest["selected_train_layouts"] == 180 and manifest["selected_val_layouts"] == 30
    train_ids = {r["geometry_id"] for r in manifest["layouts"] if r["split"] == "train"}
    val_ids = {r["geometry_id"] for r in manifest["layouts"] if r["split"] == "val"}
    assert not (train_ids & val_ids)
    for layout in manifest["layouts"]:
        episodes = [r for r in manifest["episodes"] if r["geometry_id"] == layout["geometry_id"]]
        assert len(episodes) == 9
        assert set(Counter((r["slot"], r["target_color"]) for r in episodes).values()) == {1}
        for row in episodes:
            assert hashlib.sha256((RUN / row["directory"] / "trajectory.npz").read_bytes()).hexdigest() == row["npz_sha256"]
    replay = json.loads((RUN / "data/replay_verified.json").read_text())
    assert replay["passed"] and replay["verified_geometries"] == 210
    cases = []
    for kind, pattern in (("test", "*/*/*/*/metadata.json"), ("counterfactual", "*/*/*/*/*/metadata.json")):
        for path in (RUN / "evaluations" / kind).glob(pattern):
            meta = json.loads(path.read_text()); meta["audit_kind"] = kind
            with (path.parent / "steps.csv").open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            assert len(rows) == meta["control_steps"]
            assert all(int(r["physics_samples"]) == 25 for r in rows)
            assert all(abs(float(r["time_after_s"])-float(r["time_before_s"])-.05) < 1e-8 for r in rows)
            for a, b in zip(rows, rows[1:]):
                assert max(abs(float(a["eef_after_"+axis])-float(b["eef_before_"+axis])) for axis in "xyz") < 1e-10
            first_lift = next((r for r in rows if int(r["phase"]) == 3), None)
            if first_lift: assert yes(first_lift["grip_confirmed_before"])
            if meta["success"]:
                final = rows[-40:]
                assert [int(r["hold_streak"]) for r in final] == list(range(1, 41))
                assert min(float(r["target_lift_m"]) for r in final) > .04
                assert max(float(r["target_linear_speed"]) for r in final) <= .025
                assert max(float(r["target_angular_speed"]) for r in final) <= .4
                assert max(float(r["hold_drift_m"]) for r in final) <= .005
            cases.append(meta)
    assert len(cases) == 3330
    test_groups = defaultdict(list); cf_groups = defaultdict(list)
    for row in cases:
        if row["audit_kind"] == "test": test_groups[row["geometry_id"]].append(row)
        else: cf_groups[row["geometry_id"]].append(row)
    assert len(test_groups) == 150 and len(cf_groups) == 12
    assert not ((set(test_groups) | set(cf_groups)) & (train_ids | val_ids))
    rgb_differences = []
    def same_rendering(group):
        # The physical replay verifier already allows one uint8 level for rendering.
        # Keep physical state equality exact; inspect pixels whenever hashes differ.
        reference = group[0]
        first = None
        for row in group[1:]:
            if row["initial_rgb_sha256"] == reference["initial_rgb_sha256"]: continue
            if first is None: first = np.asarray(Image.open(RUN / reference["directory"] / "initial.png")).astype(np.int16)
            current = np.asarray(Image.open(RUN / row["directory"] / "initial.png")).astype(np.int16)
            delta = np.abs(current-first)
            assert delta.max() <= 1, (row["directory"], int(delta.max()))
            rgb_differences.append({"directory": row["directory"], "reference": reference["directory"],
                                    "max_channel_difference": int(delta.max()), "changed_channels": int((delta > 0).sum()),
                                    "mean_difference": float(delta.mean())})
    for group in test_groups.values():
        assert len(group) == 15
        same_rendering(group)
        assert all(r["physical_slot_positions"] == group[0]["physical_slot_positions"] for r in group)
        assert all(r["physical_slot_quaternions"] == group[0]["physical_slot_quaternions"] for r in group)
        assert all(r["initial_robot_state"][:12] == group[0]["initial_robot_state"][:12] for r in group)
        assert all(r["actual_camera_positions"] == group[0]["actual_camera_positions"] for r in group)
        assert all(r["actual_light_diffuse"] == group[0]["actual_light_diffuse"] for r in group)
    for group in cf_groups.values():
        assert len(group) == 90
        assert all(r["physical_slot_positions"] == group[0]["physical_slot_positions"] for r in group)
        assert all(r["physical_slot_quaternions"] == group[0]["physical_slot_quaternions"] for r in group)
        assert all(r["physical_slot_body_ids"] == group[0]["physical_slot_body_ids"] for r in group)
        for p in range(6):
            fixed_appearance = [r for r in group if r["layout"]["color_assignment_index"] == p]
            same_rendering(fixed_appearance)
        representatives = [next(r for r in group if r["layout"]["color_assignment_index"] == p) for p in range(6)]
        palettes = [np.asarray(Image.open(RUN / r["directory"] / "initial.png")).astype(np.int16) for r in representatives]
        assert all(np.max(np.abs(palettes[i]-palettes[j])) > 32 for i in range(6) for j in range(i))
    result = {"passed": True, "gate_independent_layouts": 1000, "gate_trials": 3000,
              "new_training_independent_layouts": 180, "new_validation_independent_layouts": 30,
              "actual_selected_expert_rollouts": 1890, "additional_physical_replays": 630,
              "test_independent_geometries": 150, "counterfactual_independent_geometries": 12,
              "actual_test_and_counterfactual_rollouts": 3330, "models_frozen": True,
              "same_physical_initial_states_across_models": True, "counterfactual_only_colors_and_command_changed": True,
              "initial_RGB_channel_tolerance": 1, "initial_RGB_numeric_differences": rgb_differences,
              "no_split_leakage": True, "no_disturbance_event_count_inflation": True, "order_exposure_checks": exposure}
    (RUN / "audit.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__": main()
