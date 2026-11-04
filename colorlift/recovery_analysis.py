"""Matched-scene execution-length analysis and a frozen protocol for recovery training."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from .conditional import RUN as ORIGINAL
from .config import ROOT, COLORS
from .step_recording import save_rows

RUN=ROOT / "experiments/colorlift_recovery_v1"


def load(path): return json.loads(Path(path).read_text())


def baseline_trials():
    results=load(ORIGINAL / "eval_test_selected/metrics.json")["trials"]
    for row in results:
        with (ORIGINAL / row["directory"] / "steps.csv").open(encoding="utf-8-sig") as f:
            closing=next((r for r in csv.DictReader(f) if int(r["phase"])==2), None)
        row["closing_true_xy_error_m"]=None if closing is None else float(np.linalg.norm([float(closing["eef_before_"+a])-float(closing["target_before_"+a]) for a in "xy"]))
    return results


def summarize(trials):
    good=[row for row in trials if row["success"]]
    return {"episodes": len(trials), "successes": len(good), "success_rate": len(good)/len(trials),
            "collision_free_successes": sum(row["success"] and not row["collision_steps"] for row in trials),
            "collision_episodes": sum(row["collision_steps"]>0 for row in trials),
            "wrong_color_episodes": sum(row["wrong_color_grasp"] for row in trials),
            "mean_success_steps": float(np.mean([row["control_steps"] for row in good])) if good else None,
            **{color+"_successes": sum(row["target_color"]==color and row["success"] for row in trials) for color in COLORS}}


def main():
    p=argparse.ArgumentParser(); p.add_argument("--partial", action="store_true"); args=p.parse_args()
    out=RUN / "execution_sweep"; out.mkdir(parents=True, exist_ok=True)
    original=baseline_trials()
    groups={"h3_original": original}
    for name in ("h1", "h2", "h6", "h12", "adaptive"):
        paths=list((out / name).glob("scene_*/*/metadata.json"))
        if len(paths)==150: groups[name]=[load(path) for path in sorted(paths)]
        elif not args.partial: raise RuntimeError(f"{name} incomplete: {len(paths)}")
    summaries=[]; details={}
    baseline={(row["seed"], row["target_color"]): row for row in original}
    for name, trials in groups.items():
        rescued=[]; regressed=[]; scene_changes={seed: [] for seed in range(75000, 75050)}
        for row in trials:
            old=baseline[row["seed"], row["target_color"]]
            assert row["initial_rgb_sha256"] == old["initial_rgb_sha256"]
            for color in COLORS: assert np.array_equal(row["initial_positions"][color], old["initial_positions"][color])
            if row["success"] and not old["success"]: rescued.append({"seed": row["seed"], "color": row["target_color"]})
            if old["success"] and not row["success"]: regressed.append({"seed": row["seed"], "color": row["target_color"]})
            scene_changes[row["seed"]].append(int(row["success"])-int(old["success"]))
        changes=np.array([np.mean(scene_changes[seed]) for seed in sorted(scene_changes)])
        rng=np.random.default_rng(2029)
        bootstrap=changes[rng.integers(0, len(changes), (5000, len(changes)))].mean(1)
        changed=len(rescued)+len(regressed)
        pvalue=min(1., 2*sum(math.comb(changed, k) for k in range(min(len(rescued), len(regressed))+1))/2**changed) if changed else 1.
        summaries.append({"mode": name, **summarize(trials), "rescued_baseline_failures": len(rescued), "regressed_baseline_successes": len(regressed),
                          "paired_success_delta": float(changes.mean()), "scene_bootstrap_95_low": float(np.quantile(bootstrap, .025)),
                          "scene_bootstrap_95_high": float(np.quantile(bootstrap, .975)), "paired_discordance_exact_p": pvalue})
        details[name]={"rescued": rescued, "regressed": regressed}
    save_rows(out / "comparison.csv", summaries)
    (out / "comparison.json").write_text(json.dumps({"summary": summaries, "paired_changes": details}, indent=2))
    closing={}
    for good in (True, False):
        values=[row["closing_true_xy_error_m"] for row in original if row["success"]==good and row["closing_true_xy_error_m"] is not None]
        closing["success" if good else "failure"]={"with_closing_stage": len(values), "mean_xy_error_m": float(np.mean(values)) if values else None,
                                                     "median_xy_error_m": float(np.median(values)) if values else None}
    (out / "original_closing_errors.json").write_text(json.dumps(closing, indent=2))
    if not args.partial:
        candidates=[row for row in summaries if row["mode"] in ("h1", "h2", "adaptive")]
        selected=max(candidates, key=lambda row: (row["successes"], row["collision_free_successes"], row["mode"]=="adaptive", -row["mean_success_steps"]))
        schedules={"h1": [1, 1, 1, 1], "h2": [2, 2, 2, 2], "adaptive": [4, 1, 1, 3]}
        protocol={"old_checkpoint": str(ORIGINAL / "model/policy.pt"), "old_model_sha256": hashlib.sha256((ORIGINAL / "model/policy.pt").read_bytes()).hexdigest(),
                  "execution_sweep_scenes": [75000, 75049], "sweep_controller": "unchanged FastExecutor", "sweep_success_criteria": "unchanged >4cm and 40 stable samples",
                  "selected_short_execution_mode": selected["mode"], "selected_query_schedule": schedules[selected["mode"]],
                  "training_max_chunk": 4, "training_contact_recovery_prefix": 2,
                  "training_sampling_probabilities": {"approach": .05, "precise_align": .30, "closing_contact": .30, "recovery": .30, "lift_hold": .05},
                  "noisy_train_scene_seeds": [78000, 78079], "noisy_validation_scene_seeds": [79000, 79014],
                  "new_development_scenes": [80000, 80011], "fresh_clean_test_scenes": [81000, 81049], "fresh_disturbed_test_scenes": [82000, 82029],
                  "disturbance_amplitude_m": .03, "disturbance_events_per_episode": 2, "disturbance_actions_supervised": False,
                  "expert_IK": "damped 6D pose IK in private MjData; execute FK-validated waypoints through OSC, no teleportation",
                  "teacher_target": "simulator pose used only for scripted expert labels; student receives RGB/proprio/color only",
                  "new_adapter": "RecoveryExecutor uses ACT target XY and proprioceptive error for generic clearance recovery; old/new models share it. Macro target labels are separate from temporary IK waypoints.",
                  "fresh_comparison_variants": ["old_long", "old_short", "new_recovery"],
                  "fresh_clean_cases_per_variant": 150, "fresh_disturbed_cases_per_variant": 90,
                  "max_GPUs": 2, "physical_GPUs": [1, 6]}
        (RUN / "protocol.json").write_text(json.dumps(protocol, indent=2))
    print(json.dumps({"summary": summaries, "closing_errors": closing}, indent=2))


if __name__ == "__main__": main()
