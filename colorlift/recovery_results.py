"""Aggregate complete paired clean/disturbed evaluations without dropping failed attempts."""
import argparse
import json
from pathlib import Path
import numpy as np
from .config import ROOT, COLORS
from .step_recording import save_rows
from lift_act.evaluate import wilson

RUN=ROOT / "experiments/colorlift_recovery_v1"


def load(path): return json.loads(Path(path).read_text())


def summarize(trials):
    good=[row for row in trials if row["success"]]
    closing=[row["closing_true_xy_error_m"] for row in trials if row["closing_true_xy_error_m"] is not None]
    return {"episodes": len(trials), "successes": len(good), "success_rate": len(good)/len(trials), "wilson_95_ci": wilson(len(good), len(trials)),
            "collision_free_successes": sum(row["success"] and not row["collision_steps"] for row in trials),
            "collision_episodes": sum(row["collision_steps"]>0 for row in trials), "wrong_color_episodes": sum(row["wrong_color_grasp"] for row in trials),
            "hold_failure_episodes": sum(row["outcome"]=="hold_failure" for row in trials),
            "mean_success_steps": float(np.mean([row["control_steps"] for row in good])) if good else None,
            "mean_closing_xy_error_m": float(np.mean(closing)) if closing else None,
            "two_noise_event_episodes": sum(len(row["noise_events"])==2 for row in trials),
            "per_color": {color: {"episodes": sum(row["target_color"]==color for row in trials),
                                  "successes": sum(row["target_color"]==color and row["success"] for row in trials)} for color in COLORS}}


def main():
    p=argparse.ArgumentParser(); p.add_argument("--split", choices=("dev", "test"), default="dev"); args=p.parse_args()
    out=RUN / "evaluations" / args.split
    for shard in (0, 1): assert (out / f"shard_{shard}_completed.json").exists()
    paths=sorted(out.glob("*/*/scene_*/*/metadata.json"))
    expected=54 if args.split=="dev" else 720
    assert len(paths)==expected, (len(paths), expected)
    groups={}; trials=[]
    for path in paths:
        row=load(path); trials.append(row)
        groups.setdefault(("disturbed" if row["disturbed"] else "clean", row["variant"]), []).append(row)
    summaries={condition: {} for condition in ("clean", "disturbed")}
    rows=[]; switching=[]
    for (condition, variant), values in groups.items():
        stats=summarize(values); summaries[condition][variant]=stats
        per_scene={}
        for row in values: per_scene.setdefault(row["seed"], []).append(row)
        counts=0
        for seed, scene in per_scene.items():
            assert len(scene)==3 and len(set(row["initial_rgb_sha256"] for row in scene))==1
            success=all(row["success"] for row in scene)
            counts+=success
            switching.append({"condition": condition, "variant": variant, "seed": seed, "all_three_successful": success,
                              **{row["target_color"]+"_picked": row["first_picked_color"] for row in scene}})
        stats["all_three_successful_scenes"]=counts
        stats["scenes"]=len(per_scene)
        rows.append({"condition": condition, "variant": variant, **{key: stats[key] for key in ("episodes", "successes", "success_rate", "collision_free_successes", "collision_episodes", "wrong_color_episodes", "mean_success_steps", "mean_closing_xy_error_m", "two_noise_event_episodes", "all_three_successful_scenes")},
                     **{color+"_successes": stats["per_color"][color]["successes"] for color in COLORS}})
    paired={}
    for condition in ("clean", "disturbed"):
        new=groups[condition, "new_recovery"]
        for baseline in ("old_original", "old_long", "old_short"):
            if (condition, baseline) not in groups: continue
            old={(row["seed"], row["target_color"]): row for row in groups[condition, baseline]}
            rescued=regressed=0; changes={}; dose_matched=[]
            for row in new:
                reference=old[row["seed"], row["target_color"]]
                assert row["initial_rgb_sha256"]==reference["initial_rgb_sha256"]
                assert np.array_equal(row["initial_robot_state"], reference["initial_robot_state"])
                rescued+=row["success"] and not reference["success"]
                regressed+=reference["success"] and not row["success"]
                changes.setdefault(row["seed"], []).append(int(row["success"])-int(reference["success"]))
                if condition=="disturbed" and len(row["noise_events"])==len(reference["noise_events"])==2:
                    assert np.array_equal([e["requested_offset"] for e in row["noise_events"]], [e["requested_offset"] for e in reference["noise_events"]])
                    dose_matched.append((reference["success"], row["success"]))
            values=np.array([np.mean(value) for value in changes.values()])
            rng=np.random.default_rng(2031)
            boot=values[rng.integers(0, len(values), (5000, len(values)))].mean(1)
            paired[condition+"_vs_"+baseline]={"rescued": int(rescued), "regressed": int(regressed), "success_delta": float(values.mean()),
                                              "scene_bootstrap_95_ci": np.quantile(boot, [.025, .975]).tolist(),
                                              "both_received_two_offsets": len(dose_matched),
                                              "two_offset_matched_old_successes": sum(old for old, new in dose_matched),
                                              "two_offset_matched_new_successes": sum(new for old, new in dose_matched)}
    save_rows(out / "comparison.csv", rows)
    save_rows(out / "switching.csv", switching)
    save_rows(out / "episodes.csv", [{key: row[key] for key in ("seed", "target_color", "variant", "disturbed", "success", "outcome", "control_steps", "collision_steps", "first_picked_color", "closing_true_xy_error_m", "directory")} for row in trials])
    result={"split": args.split, "summary": summaries, "paired_comparisons": paired, "episodes": len(trials), "trials": trials}
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    if args.split=="test":
        regression=[load(path) for path in (RUN / "regression_old_failures").glob("scene_*/*/metadata.json")]
        assert len(regression)==8
        (RUN / "old_failures_replay.json").write_text(json.dumps({"rescued": sum(row["success"] for row in regression), "episodes": 8, "trials": regression}, indent=2))
    print(json.dumps({"summary": summaries, "paired_comparisons": paired}, indent=2))


if __name__ == "__main__": main()
