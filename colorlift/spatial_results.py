"""Layout-aware analysis; repeated colors, permutations and training seeds are not layouts."""
from collections import defaultdict
import csv
import hashlib
import json
import numpy as np
from .spatial_gate import RUN, ROOT, wilson, fingerprints
from .spatial_evaluate import MODELS, LEVELS, checkpoint_paths


def csv_file(path, rows):
    if not rows: return
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def summarize_group(rows):
    passed = [r for r in rows if r["success"]]
    n = len(rows)
    by_color = {}
    for color in ("red", "yellow", "blue"):
        cases = [r for r in rows if r["target_color"] == color]
        successes = sum(r["success"] for r in cases)
        strict = sum(r["eligible"] for r in cases)
        by_color[color] = {"n": len(cases), "successes": successes, "success_wilson95": wilson(successes, len(cases)),
                           "contact_successes": strict, "contact_wilson95": wilson(strict, len(cases))}
    return {"trials": n, "independent_geometries": len({r["geometry_id"] for r in rows}),
            "successes": len(passed), "success_rate": len(passed)/n, "contact_successes": sum(r["eligible"] for r in rows),
            "contact_success_rate": sum(r["eligible"] for r in rows)/n, "wrong_colors": sum(r["wrong_color_grasp"] for r in rows),
            "collision_episodes": sum(r["collision_steps"] > 0 for r in rows),
            "mean_success_steps": float(np.mean([r["control_steps"] for r in passed])) if passed else None,
            "mean_steps_all_attempts": float(np.mean([r["control_steps"] for r in rows])),
            "p90_success_steps": float(np.quantile([r["control_steps"] for r in passed], .9)) if passed else None,
            "mean_steps_before_final_hold_success": float(np.mean([r["steps_before_final_hold"] for r in passed])) if passed else None,
            "mean_success_physical_path_m": float(np.mean([r["physical_eef_path_m"] for r in passed])) if passed else None,
            "mean_post_close_idle": float(np.mean([r["post_close_idle_steps"] for r in passed])) if passed else None,
            "mean_no_progress": float(np.mean([r["no_progress_steps"] for r in passed])) if passed else None,
            "first_goal_correct": sum(r["first_goal_nearest_slot"] == r["target_physical_slot"] for r in rows), "by_color": by_color}


def main():
    assert (RUN / "computation_completed.json").exists()
    out = RUN / "results"; out.mkdir(exist_ok=True)
    trials = [json.loads(p.read_text()) for p in sorted((RUN / "evaluations/test").glob("*/*/*/*/metadata.json"))]
    cf = [json.loads(p.read_text()) for p in sorted((RUN / "evaluations/counterfactual").glob("*/*/*/*/*/metadata.json"))]
    validation = [json.loads(p.read_text()) for p in sorted((RUN / "evaluations/validation").glob("*/*/*/*/*/metadata.json"))]
    assert len(trials) == 2250 and len(cf) == 1080 and len(validation) == 216
    comparison = []
    for model in MODELS:
        for level in LEVELS:
            rows = [r for r in trials if r["actor"] == model and r["layout"]["level"] == level]
            comparison.append({"model": model, "level": level, **summarize_group(rows)})
    csv_file(out / "comparison.csv", [{k: v for k, v in row.items() if k != "by_color"} for row in comparison])
    cf_summary = []
    for model in MODELS:
        rows = [r for r in cf if r["actor"] == model]
        summary = {"model": model, **summarize_group(rows)}
        geometries = defaultdict(list)
        for row in rows: geometries[row["geometry_id"]].append(row)
        summary.update(first_pick_correct=sum(r["first_picked_slot"] == r["target_physical_slot"] for r in rows),
                       no_confirmed_pick=sum(r["first_picked_slot"] is None for r in rows),
                       all18_success_geometries=sum(all(r["success"] for r in group) for group in geometries.values()),
                       fixed_red_all6_success=sum(all(r["success"] for r in group if r["target_color"] == "red") for group in geometries.values()),
                       recolor_update_command_slot0_all6_success=sum(all(r["success"] for r in group if r["target_physical_slot"] == 0) for group in geometries.values()),
                       fixed_appearance_three_commands_success=sum(all(r["success"] for r in group if r["layout"]["color_assignment_index"] == 0) for group in geometries.values()))
        cf_summary.append(summary)
    csv_file(out / "counterfactual.csv", [{k: v for k, v in row.items() if k != "by_color"} for row in cf_summary])
    families = []
    for family, names in (("baseline", ["baseline"]), ("curriculum", ["curriculum_seed43", "curriculum_seed44"]), ("shuffled", ["shuffled_seed43", "shuffled_seed44"])):
        for level in LEVELS:
            selected = [row for row in comparison if row["model"] in names and row["level"] == level]
            families.append({"family": family, "level": level, "training_seeds": len(names), "independent_geometries": selected[0]["independent_geometries"],
                             "mean_success_rate": float(np.mean([r["success_rate"] for r in selected])),
                             "seed_min_success_rate": min(r["success_rate"] for r in selected), "seed_max_success_rate": max(r["success_rate"] for r in selected),
                             "mean_contact_rate": float(np.mean([r["contact_success_rate"] for r in selected])),
                             "mean_success_steps": float(np.mean([r["mean_success_steps"] for r in selected if r["mean_success_steps"] is not None])),
                             "wrong_color_total": sum(r["wrong_colors"] for r in selected)})
    csv_file(out / "family_comparison.csv", families)
    pair_deltas = []
    for level in LEVELS:
        selected = [r for r in trials if r["layout"]["level"] == level]
        ids = sorted({r["geometry_id"] for r in selected})
        differences = []
        for geometry in ids:
            current = [r for r in selected if r["geometry_id"] == geometry]
            rates = {}
            for family in ("curriculum", "shuffled"):
                family_rows = [r for r in current if r["actor"].startswith(family)]
                assert len(family_rows) == 6
                rates[family] = np.array([np.mean([r["success"] for r in family_rows]), np.mean([r["eligible"] for r in family_rows])])
            differences.append(rates["curriculum"]-rates["shuffled"])
        differences = np.array(differences)
        rng = np.random.default_rng(20261030)
        boot = np.array([differences[rng.integers(len(ids), size=len(ids))].mean(0) for _ in range(5000)])
        pair_deltas.append({"level": level, "independent_geometries": len(ids), "curriculum_minus_shuffled_success_pp": float(100*differences[:, 0].mean()),
                            "success_pp_cluster95": (100*np.quantile(boot[:, 0], [.025, .975])).tolist(),
                            "curriculum_minus_shuffled_contact_pp": float(100*differences[:, 1].mean()),
                            "contact_pp_cluster95": (100*np.quantile(boot[:, 1], [.025, .975])).tolist(),
                            "training_seed_uncertainty": "two paired seeds only; this interval resamples layouts"})
    csv_file(out / "paired_order_effects.csv", pair_deltas)
    validation_rows = []
    for model in MODELS[1:]:
        for step in (1500, 3000, 5000):
            for level in LEVELS[:3]:
                rows = [r for r in validation if r["actor"] == model and f"step_{step}/" in r["directory"] and r["layout"]["level"] == level]
                assert len(rows) == 6
                validation_rows.append({"model": model, "step": step, "level": level, **summarize_group(rows)})
    csv_file(out / "stage_validation.csv", [{k: v for k, v in row.items() if k != "by_color"} for row in validation_rows])
    fields = ("actor", "seed", "target_color", "target_physical_slot", "first_goal_nearest_slot", "first_picked_slot", "success", "eligible", "wrong_color_grasp", "collision_steps", "control_steps", "steps_before_final_hold", "outcome", "directory")
    csv_file(out / "all_test_episodes.csv", [{key: row.get(key) for key in fields} for row in trials+cf])
    result = {"comparison": comparison, "families": families, "counterfactual": cf_summary, "paired_order_effects": pair_deltas,
              "stage_validation": validation_rows, "test_unique_geometries": 150, "counterfactual_unique_geometries": 12,
              "test_rollouts": len(trials), "counterfactual_rollouts": len(cf), "stage_validation_rollouts": len(validation),
              "repeats_not_independent_layouts": True}
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    failures = [r for r in trials+cf if not r["success"]]
    (out / "failure_cases.json").write_text(json.dumps(failures, indent=2))
    print(json.dumps({"comparisons": [{k: v for k, v in r.items() if k != "by_color"} for r in comparison],
                      "counterfactual": [{k: v for k, v in r.items() if k != "by_color"} for r in cf_summary]}, indent=2), flush=True)


if __name__ == "__main__": main()
