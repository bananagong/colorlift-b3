"""Independently audit saved ACT test traces and same-scene intervention evidence."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .conditional import RUN, COLORS


def load(path): return json.loads(Path(path).read_text())


def matrix(rows, keys): return np.array([[float(row[key]) for key in keys] for row in rows])


def main():
    metrics=load(RUN / "eval_test_selected/metrics.json")
    assert metrics["scenes"] == 50 and metrics["episodes"] == 150
    train=load(RUN / "data/manifest.json")
    training_seeds={row["seed"] for row in train["episodes"]}
    assert training_seeds.isdisjoint({row["seed"] for row in metrics["trials"]})
    successes=0
    image_checks={}
    for trial in metrics["trials"]:
        directory=RUN / trial["directory"]
        with (directory / "steps.csv").open(encoding="utf-8-sig") as f: rows=list(csv.DictReader(f))
        n=trial["control_steps"]
        assert len(rows) == n and [int(row["step"]) for row in rows] == list(range(1, n+1))
        before=matrix(rows, ["eef_before_"+axis for axis in "xyz"])
        after=matrix(rows, ["eef_after_"+axis for axis in "xyz"])
        assert np.allclose(before[1:], after[:-1], rtol=0, atol=1e-10)
        actions=matrix(rows, [f"osc_action_{i}" for i in range(7)])
        assert np.isfinite(actions).all()
        assert np.allclose(actions[:, :3]*.05, matrix(rows, ["command_"+axis for axis in "xyz"]), atol=1e-9)
        assert all(int(row["target_body_id"]) == trial["body_ids"][trial["target_color"]] for row in rows)
        phases=matrix(rows, ["phase"])[:, 0]
        assert np.all(np.diff(phases) >= 0) and np.all(np.diff(phases) <= 1)
        assert trial["outcome"] != "success" or trial["success"]
        image=np.array(Image.open(directory / "initial_rgb.png"))
        # Stored PNG joins the two RGB views horizontally; reconstruct their original order.
        reconstructed=np.stack(np.split(image, 2, axis=1))
        digest=hashlib.sha256(reconstructed.tobytes()).hexdigest()
        assert digest == trial["initial_rgb_sha256"]
        image_checks.setdefault(trial["seed"], []).append(digest)
        confirmed={color: 0 for color in COLORS}
        first=None
        for row in rows:
            for color in COLORS:
                grasp=row["grasp_"+color].lower() == "true"
                confirmed[color]=confirmed[color]+1 if grasp else 0
                if confirmed[color] >= 3 and first is None: first=color
        assert (first or "none") == trial["first_picked_color"]
        if trial["success"]:
            successes+=1
            assert first == trial["target_color"] and trial["wrong_color_grasp"] is False
            last=rows[-40:]
            assert [int(row["hold_streak"]) for row in last] == list(range(1, 41))
            assert min(float(row["target_lift_m"]) for row in last) > .04
            assert max(float(row["target_linear_speed"]) for row in last) <= .025
            assert max(float(row["target_angular_speed"]) for row in last) <= .4
            positions=matrix(last, ["target_"+axis for axis in "xyz"])
            assert np.linalg.norm(positions-positions[0], axis=1).max() <= .005+1e-10
    assert successes == metrics["successes"]
    assert all(len(hashes) == 3 and len(set(hashes)) == 1 for hashes in image_checks.values())
    executions=[]
    for gpu in (2, 4):
        for line in (RUN / f"eval_test_gpu{gpu}.log").read_text().splitlines():
            if line.startswith("EXECUTE_EPISODE "):
                fields=line.split()
                executions.append((int(fields[1]), fields[2]))
    assert len(executions) == len(set(executions)) == 150
    outcome={"passed": True, "test_episodes": 150, "test_scenes": 50, "verified_strict_successes": successes,
             "training_test_scene_disjoint": True, "saved_initial_RGB_identical_across_color_commands": True,
             "actual_grasp_classification_recomputed_from_step_contacts": True, "single_monotone_grasp_attempt": True,
             "EEF_temporal_continuity": True, "OSC_actions_match_saved_commands": True, "exactly_150_unique_actual_executions": True,
             "success_height_velocity_drift_and_40_consecutive_hold_samples": True}
    (RUN / "audit.json").write_text(json.dumps(outcome, indent=2))
    print("AUDIT PASSED", json.dumps(outcome))


if __name__ == "__main__": main()
