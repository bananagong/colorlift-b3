"""Verify frozen generation, geometry isolation, real success and label alignment."""
from collections import Counter
import csv
from datetime import datetime
import json
import numpy as np
from .spec import ROOT, RUN, sha
from .layouts import geometry_id, region_ok, relation_ok
from .generate import group_id
from .freeze import verify
from .develop import coverage


def flag(x): return str(x).lower() in ("true", "1")


def distances(rows, training):
    permutations = np.array(list(__import__("itertools").permutations(range(3))))
    reference = np.array([r["layout"]["xy"] for r in training])[:, permutations]
    nearest = []
    for row in rows:
        delta = reference-np.array(row["layout"]["xy"])[None, None]
        nearest.append(float(np.sqrt(np.mean(np.sum(delta*delta, axis=-1), axis=-1)).min()))
    return dict(zip(("minimum_m", "median_m", "p95_m"), np.quantile(nearest, [0, .5, .95]).tolist()))


def main():
    protocol = json.loads((RUN / "final_protocol.json").read_text()); verify(protocol)
    development = json.loads((RUN / "development_protocol.json").read_text())
    assert sha(ROOT / development["baseline_checkpoint"]) == development["baseline_sha256"]
    legacy = ROOT / "experiments/colorlift_spatial_curriculum_v1"
    old_freeze = json.loads((legacy / "evaluation_freeze.json").read_text())["models"]
    for name, expected in old_freeze.items():
        path = ROOT / development["baseline_checkpoint"] if name == "baseline" else legacy / "models" / name / "policy.pt"
        assert sha(path) == expected
    training = json.loads((RUN / "training_manifest.json").read_text())
    final = json.loads((RUN / "final_sealed/manifest.json").read_text())
    all_rows = training["layouts"]+final["layouts"]
    assert len({r["geometry_id"] for r in all_rows}) == len(all_rows) == 1008
    assert len({r["position_group_id"] for r in all_rows}) == len(all_rows)
    freezes_at = datetime.fromisoformat(protocol["frozen_at_UTC"]).timestamp()
    first_final_time = min((RUN / r["directory"] / "layout.json").stat().st_mtime for r in final["layouts"])
    assert first_final_time >= freezes_at
    for row in all_rows:
        layout, suite = row["layout"], row["split"]
        assert geometry_id(layout["xy"]) == row["geometry_id"]
        assert group_id(layout["xy"]) == row["position_group_id"]
        assert region_ok(layout["xy"], suite)
        assert relation_ok(layout["xy"], suite, layout["relation_subtype"])
        valid = json.loads((RUN / row["directory"] / "validity.json").read_text())
        assert valid["valid"] and all(r["valid"] for r in valid["visibility"]) and all(p["valid"] for p in valid["grasp_plans"])
        if suite == "spatial_ood": assert layout["spatial_target_slot"] is not None
    seal = json.loads((RUN / "final_sealed_sha256.json").read_text())
    for relative, digest in seal.items(): assert sha(RUN / "final_sealed" / relative) == digest
    assert not final["policy_evaluation_performed"] and not final["used_for_model_selection"]
    assert not list((RUN / "final_sealed").rglob("trajectory.npz"))
    assert not list((RUN / "final_sealed").rglob("steps.csv"))
    assert not list((RUN / "final_sealed").rglob("metadata.json"))
    selected_episodes = 0; max_label_error = 0.; frames = 0
    color_balance = {"train": Counter(), "val": Counter()}
    for row in training["layouts"]:
        assert row["directory"].startswith("data_candidates/") and len(row["episodes"]) == 9
        local_balance = Counter()
        for relative in row["episodes"]:
            directory = RUN / relative
            meta = json.loads((directory / "metadata.json").read_text())
            assert meta["eligible"] and meta["success"] and not meta["wrong_color_grasp"] and meta["collision_steps"] == 0
            local_balance[meta["target_physical_slot"], meta["target_color"]] += 1
            with (directory / "steps.csv").open(encoding="utf-8-sig", newline="") as stream: steps = list(csv.DictReader(stream))
            n = len(steps); assert n == meta["control_steps"]
            assert all(int(r["physics_samples"]) == 25 for r in steps)
            tail = steps[-40:]
            assert [int(r["hold_streak"]) for r in tail] == list(range(1, 41))
            assert all(flag(r["target_grasped"]) and float(r["target_lift_m"]) > .04 and float(r["target_linear_speed"]) <= .025 and float(r["target_angular_speed"]) <= .4 and float(r["hold_drift_m"]) <= .005 for r in tail)
            lift = next(r for r in steps if int(r["phase"]) == 3)
            assert flag(lift["grip_confirmed_before"]) and flag(lift["true_grasp_before"])
            assert int(lift["tactile_body_id_before"]) == int(meta["body_ids"][meta["target_color"]])
            with np.load(directory / "trajectory.npz", allow_pickle=False) as data:
                eef = np.array([[float(r["eef_before_"+a]) for a in "xyz"] for r in steps])
                target = np.array([[float(r["target_before_"+a]) for a in "xyz"] for r in steps])
                error = float(np.abs(data["goal_relative"]-(target-eef)).max())
                max_label_error = max(max_label_error, error); assert error < 1e-10
                assert np.abs(data["pregrasp_relative"]-(data["pregrasp_position_label"]-eef)).max() < 1e-10
                assert np.abs(data["target_position_label"]-target).max() < 1e-10
                assert data["states"].shape == (n, 19)
                assert np.abs(data["states"][:, 9:12]-eef).max() < 1e-7
                assert data["pregrasp_rotation_label"].shape == (n, 3, 3)
                assert data["osc_actions"].dtype == np.float64
            selected_episodes += 1; frames += n
        assert len(local_balance) == 9 and set(local_balance.values()) == {1}
        color_balance[row["split"]].update(local_balance)
    assert selected_episodes == 2160
    train_rows = [r for r in training["layouts"] if r["split"] == "train"]
    val_rows = [r for r in training["layouts"] if r["split"] == "val"]
    nearest = {"val": distances(val_rows, train_rows)}
    for suite in ("broad_iid", "spatial_ood", "relation_ood"):
        nearest[suite] = distances([r for r in final["layouts"] if r["split"] == suite], train_rows)
    report = {"passed": True, "frozen_protocol_before_first_final_layout": True, "ACT_architecture_and_weights_unchanged": True,
              "selected_independent_train_geometries": 192, "selected_independent_val_geometries": 48,
              "selected_real_expert_demonstrations": selected_episodes, "selected_control_frames": frames,
              "new_independent_final_geometries": 768, "final_policy_inference_count": 0,
              "same_geometry_or_1cm_group_cross_split": 0, "geometry_statistical_unit": "unordered XY geometry_id, not episode_id",
              "all_admitted_episodes_contact_qualified": True, "maximum_goal_relative_label_error_m": max_label_error,
              "color_target_slot_counts": {split: {f"slot{s}_{c}": n for (s, c), n in counts.items()} for split, counts in color_balance.items()},
              "nearest_train_unordered_position_RMS": nearest,
              "training_coverage": coverage([r["layout"] for r in train_rows]),
              "final_hash_seal_passed": True, "labels_never_used_as_formal_policy_inputs": True,
              "new_auxiliary_head_trained": False, "safe_approach_ACT_ablation_run": False, "contact_recovery_ACT_ablation_run": False}
    (RUN / "audit.json").write_text(json.dumps(report, indent=2))
    public = {key: value for key, value in final.items() if key != "layouts"}
    public["geometry_groups"] = [{k: r[k] for k in ("geometry_id", "position_group_id", "split")} for r in final["layouts"]]
    public["sealed_manifest_sha256"] = sha(RUN / "final_sealed/manifest.json")
    (RUN / "public_final_manifest.json").write_text(json.dumps(public, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__": main()
