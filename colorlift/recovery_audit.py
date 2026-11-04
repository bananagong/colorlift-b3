"""Audit actual perturbations, masked labels and strict recovery evaluation traces."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from .config import ROOT

RUN=ROOT / "experiments/colorlift_recovery_v1"


def load(path): return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open(encoding="utf-8-sig") as stream: return list(csv.DictReader(stream))


def matrix(data, keys): return np.array([[float(row[key]) for key in keys] for row in data])


def data_audit():
    manifest=load(RUN / "data/manifest.json")
    total=valid=recovery=0; amplitudes=[]; requested=[]; residuals=[]; successes=0
    for trial in manifest["episodes"]:
        directory=RUN / trial["directory"]
        trace=rows(directory / "steps.csv")
        before=matrix(trace, ["eef_before_"+a for a in "xyz"])
        after=matrix(trace, ["eef_after_"+a for a in "xyz"])
        assert np.allclose(before[1:], after[:-1], rtol=0, atol=1e-10)
        with np.load(directory / "trajectory.npz", allow_pickle=False) as data:
            n=len(trace); total+=n
            assert len(data["states"])==n and data["images"].shape==(n, 2, 128, 128, 3)
            assert np.allclose(data["states"][:, 9:12], before, atol=1e-7)
            supervision=data["supervised"].astype(bool)
            disturbance=data["category"]=="injected_offset"
            assert np.array_equal(~supervision, disturbance)
            assert np.all(supervision[data["recovery"]])
            assert data["ik_qpos"].shape==(n, 7) and np.isfinite(data["ik_qpos"]).all()
            valid+=int(supervision.sum()); recovery+=int(data["recovery"].sum())
            for event in trial["noise_events"]:
                start, end=event["start_step"]-1, event["end_step"]-1
                assert not supervision[start:end+1].any()
                assert np.allclose(after[end]-before[start], event["realized_offset"], atol=1e-8)
                requested.append(float(np.linalg.norm(event["requested_offset"])))
                amplitudes.append(event["realized_offset_norm_m"])
                assert abs(requested[-1]-.03)<1e-12
        residuals.append(trial["maximum_ik_residual_m"])
        successes+=trial["success"]
    report={"passed": True, "attempted_episodes": len(manifest["episodes"]), "successful_episodes": successes, "eligible_counts": manifest["eligible_counts"],
            "total_control_steps": total, "supervised_corrective_steps": valid, "excluded_injection_steps": total-valid, "recovery_steps": recovery,
            "requested_offset_m": .03, "actual_offset_mean_m": float(np.mean(amplitudes)), "actual_offset_min_m": float(np.min(amplitudes)),
            "actual_offset_max_m": float(np.max(amplitudes)), "events": len(amplitudes), "max_IK_position_residual_m": max(residuals),
            "no_robot_state_teleportation": True, "all_injection_actions_masked_from_labels": True, "recovery_actions_supervised": True}
    (RUN / "data_audit.json").write_text(json.dumps(report, indent=2))
    print("RECOVERY DATA AUDIT", json.dumps(report))


def test_audit():
    paths=list((RUN / "evaluations/test").glob("*/*/scene_*/*/metadata.json"))
    assert len(paths)==720, len(paths)
    groups={}; checked=successes=0
    for path in paths:
        trial=load(path); trace=rows(path.parent / "steps.csv")
        n=trial["control_steps"]
        assert len(trace)==n and [int(row["step"]) for row in trace]==list(range(1, n+1))
        before=matrix(trace, ["eef_before_"+a for a in "xyz"]); after=matrix(trace, ["eef_after_"+a for a in "xyz"])
        assert np.allclose(before[1:], after[:-1], rtol=0, atol=1e-10)
        assert np.allclose(matrix(trace, [f"osc_action_{i}" for i in range(3)])*.05, matrix(trace, ["command_"+a for a in "xyz"]), atol=1e-9)
        assert all(int(row["target_body_id"])==trial["body_ids"][trial["target_color"]] for row in trace)
        group=(trial["disturbed"], trial["seed"])
        groups.setdefault(group, []).append(trial)
        if trial["success"]:
            successes+=1; last=trace[-40:]
            assert [int(row["hold_streak"]) for row in last]==list(range(1, 41))
            assert min(float(row["target_lift_m"]) for row in last)>.04
            assert max(float(row["target_linear_speed"]) for row in last)<=.025
            assert max(float(row["target_angular_speed"]) for row in last)<=.4
            position=matrix(last, ["target_"+a for a in "xyz"])
            assert np.linalg.norm(position-position[0], axis=1).max()<=.005+1e-10
            assert trial["first_picked_color"]==trial["target_color"] and not trial["wrong_color_grasp"]
        checked+=1
    for group in groups.values():
        assert len(set(row["initial_rgb_sha256"] for row in group))==1
        assert all(np.array_equal(row["initial_robot_state"], group[0]["initial_robot_state"]) for row in group)
    report={"passed": True, "test_episodes": checked, "strict_successes_verified": successes,
            "initial_RGB_and_robot_state_matched_between_variants": True, "no_state_teleportation": True,
            "OSC_trace_matches_executed_actions": True, "success_requires_40_valid_samples_above_4cm": True}
    (RUN / "test_audit.json").write_text(json.dumps(report, indent=2))
    print("RECOVERY TEST AUDIT", json.dumps(report))


if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--test", action="store_true"); args=p.parse_args()
    test_audit() if args.test else data_audit()
