"""Efficiency summaries use actual 20Hz steps and 500Hz physical path measurements."""
import argparse
from collections import defaultdict
import csv
import json
import numpy as np
from .event_efficiency import RUN
from .event_control import SCHEMES


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        out = csv.DictWriter(f, fieldnames=list(rows[0]))
        out.writeheader()
        out.writerows(rows)


def summarize(split):
    out = RUN / split
    episodes = [json.loads(path.read_text()) for path in sorted(out.glob("*/*/scene_*/*/metadata.json"))]
    assert episodes
    groups = defaultdict(list)
    for episode in episodes:
        groups[episode["actor"], episode["scheme"]].append(episode)
    rows = []
    for (actor, scheme), cases in groups.items():
        successful = [e for e in cases if e["success"]]
        mean = lambda key: float(np.mean([e[key] for e in successful])) if successful else None
        rows.append(dict(actor=actor, scheme=scheme, trials=len(cases), successes=len(successful),
                         success_rate=len(successful)/len(cases),
                         contact_qualified_successes=sum(e["contact_qualified_collision_free_success"] for e in cases),
                         collision_free_successes=sum(e["collision_free_success"] for e in cases),
                         wrong_color_failures=sum(e["wrong_color_grasp"] for e in cases),
                         collision_episodes=sum(e["collision_steps"] > 0 for e in cases),
                         lift_without_confirmed_contact=sum(e["first_lift_step"] is not None and not e["lift_started_with_confirmed_contact"] for e in cases),
                         mean_steps_success=mean("control_steps"),
                         median_steps_success=float(np.median([e["control_steps"] for e in successful])) if successful else None,
                         p90_steps_success=float(np.quantile([e["control_steps"] for e in successful], .9)) if successful else None,
                         mean_steps_before_final_hold=mean("steps_before_final_hold"),
                         mean_physical_path_m=mean("physical_eef_path_m"),
                         mean_path_before_final_hold_m=mean("physical_eef_path_before_final_hold_m"),
                         mean_queries=mean("policy_queries"),
                         mean_motion_reversals=mean("motion_reversals_within_phase"),
                         mean_command_reversals=mean("command_reversals_within_phase"),
                         mean_post_close_idle=mean("post_close_idle_steps"),
                         mean_empty_gripper_wait=mean("empty_gripper_wait_steps"),
                         mean_no_progress=mean("no_progress_steps"),
                         red_successes=sum(e["success"] for e in cases if e["target_color"] == "red"),
                         yellow_successes=sum(e["success"] for e in cases if e["target_color"] == "yellow"),
                         blue_successes=sum(e["success"] for e in cases if e["target_color"] == "blue")))
    write_csv(out / "comparison.csv", rows)
    paired = []
    index = {(e["actor"], e["scheme"], e["seed"], e["target_color"]): e for e in episodes}
    for actor in ("expert", "act"):
        schemes = [scheme for a, scheme in groups if a == actor]
        keys = sorted({(e["seed"], e["target_color"]) for e in episodes if e["actor"] == actor})
        common = [(seed, color) for seed, color in keys if all(index[actor, scheme, seed, color]["success"] for scheme in schemes)]
        for scheme in schemes:
            cases = [index[actor, scheme, *key] for key in common]
            baseline = [index[actor, "baseline", *key] for key in common]
            if not cases:
                continue
            reduction = np.array([b["control_steps"]-e["control_steps"] for b, e in zip(baseline, cases)])
            by_scene = defaultdict(list)
            for (seed, _), delta in zip(common, reduction):
                by_scene[seed].append(delta)
            scene_sums = np.array([sum(v) for v in by_scene.values()])
            scene_counts = np.array([len(v) for v in by_scene.values()])
            rng = np.random.default_rng(42)
            cluster_ids = rng.integers(len(scene_sums), size=(5000, len(scene_sums)))
            boot = scene_sums[cluster_ids].sum(1)/scene_counts[cluster_ids].sum(1)
            paired.append(dict(actor=actor, scheme=scheme, common_success_cases=len(cases),
                               mean_steps=float(np.mean([e["control_steps"] for e in cases])),
                               mean_baseline_steps=float(np.mean([e["control_steps"] for e in baseline])),
                               mean_steps_saved=float(reduction.mean()),
                               saved_ci95_low=float(np.quantile(boot, .025)), saved_ci95_high=float(np.quantile(boot, .975)),
                               cases_faster=int((reduction > 0).sum()), cases_tied=int((reduction == 0).sum()), cases_slower=int((reduction < 0).sum()),
                               mean_physical_path_m=float(np.mean([e["physical_eef_path_m"] for e in cases]))))
    write_csv(out / "paired_common_success.csv", paired)
    phases = []
    for (actor, scheme), cases in groups.items():
        cases = [e for e in cases if e["success"]]
        if not cases:
            continue
        phases.append(dict(actor=actor, scheme=scheme, successes=len(cases),
                           approach=float(np.mean([e["phase_control_steps"]["approach"] for e in cases])),
                           align=float(np.mean([e["phase_control_steps"]["align"] for e in cases])),
                           close=float(np.mean([e["phase_control_steps"]["close"] for e in cases])),
                           lift_and_settle=float(np.mean([e["phase_control_steps"]["lift_hold"]-40 for e in cases])),
                           mandatory_hold=40.))
    write_csv(out / "phase_breakdown.csv", phases)
    selected_keys = ("actor", "scheme", "seed", "target_color", "success", "collision_free_success", "contact_qualified_collision_free_success",
                     "control_steps", "steps_before_final_hold", "physical_eef_path_m", "physical_eef_path_before_final_hold_m",
                     "policy_queries", "command_reversals_within_phase", "motion_reversals_within_phase", "post_close_idle_steps",
                     "empty_gripper_wait_steps", "no_progress_steps", "collision_steps", "wrong_color_grasp", "outcome", "directory")
    write_csv(out / "episodes.csv", [{key: e.get(key) for key in selected_keys} for e in episodes])
    write_csv(out / "first70_breakdown.csv", [{"actor": e["actor"], "scheme": e["scheme"], "seed": e["seed"],
               "target_color": e["target_color"], "observed_steps": e["first70"]["observed_steps"],
               **e["first70"]["composition"], "hold_streak_at_70": e["first70"]["hold_streak_at_end"]} for e in episodes])
    metrics = {"episodes": len(episodes), "comparison": rows, "paired_common_success": paired, "phases": phases}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    if split == "dev":
        experts = {row["scheme"]: row for row in rows if row["actor"] == "expert"}
        old, new = experts["baseline"], experts["efficiency"]
        acceptable = (new["successes"] >= old["successes"] and new["contact_qualified_successes"] >= old["contact_qualified_successes"]
                      and new["mean_steps_success"] < old["mean_steps_success"])
        (RUN / "selected_expert.json").write_text(json.dumps({"scheme": "efficiency" if acceptable else "hybrid", "dev_decision_only": True,
                                                              "efficiency_passed_development_gate": acceptable}, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=("dev", "test"), required=True)
    args = p.parse_args()
    summarize(args.split)


if __name__ == "__main__":
    main()
