"""Read-only development failure analysis and mixture-resample stability."""
from collections import Counter
import csv
import json
import numpy as np
from .spec import RUN, CATEGORIES, MIXES
from .develop import coverage


def main():
    dev = json.loads((RUN / "development_results.json").read_text())
    rows = []
    for path in (RUN / "development").glob("*/episodes/*/act/*/metadata.json"):
        meta = json.loads(path.read_text())
        with (path.parent / "steps.csv").open(encoding="utf-8-sig", newline="") as stream: steps = list(csv.DictReader(stream))
        meta["low_progress_steps"] = sum(str(r["no_progress"]).lower() == "true" for r in steps)
        meta["post_close_idle_steps"] = sum(str(r["post_close_idle"]).lower() == "true" for r in steps)
        rows.append(meta)
    grouped = {}
    for success in (True, False):
        selected = [r for r in rows if r["success"] == success]
        errors = [r["first_close_target_XY_error_m"] for r in selected if r["first_close_target_XY_error_m"] is not None]
        grouped[str(success)] = {"episodes": len(selected), "independent_geometries": len({r["geometry_id"] for r in selected}),
                                "close_XY_error_mean_m": float(np.mean(errors)), "close_XY_error_median_m": float(np.median(errors)),
                                "close_XY_error_p90_m": float(np.quantile(errors, .9)),
                                "no_close_event": sum(r["first_close_target_XY_error_m"] is None for r in selected),
                                "collision_episodes": sum(r["collision_steps"] > 0 for r in selected),
                                "mean_low_progress_steps": float(np.mean([r["low_progress_steps"] for r in selected])),
                                "failure_categories": dict(Counter(r["failure_category"] for r in selected))}
    pools = {category: json.loads((RUN / "development" / category / "pool_completed.json").read_text())["valid_layouts"] for category in CATEGORIES}
    scores = {}
    for name, weights in MIXES.items():
        quota = dev["mixture_candidates"][name]["quota_128"]
        other = .25*dev["mixture_candidates"][name]["expected_safe_contact_success"]+.10*dev["mixture_candidates"][name]["harmonic_proposal_acceptance"]
        values = []
        for seed in range(100):
            rng = np.random.default_rng(4100+seed)
            layouts = [row for c, n in zip(CATEGORIES, quota) for row in [pools[c][i] for i in rng.choice(64, int(n), replace=False)]]
            result = coverage(layouts)
            values.append(.45*result["XY_grid_occupancy"]+.20*result["relation_histogram_entropy"]+other)
        assert abs(np.mean(values)-dev["mixture_candidates"][name]["score"]) < 1e-12
        scores[name] = np.array(values)
    stability = {name: {"mean": float(values.mean()), "std": float(values.std()), "resample_2.5_97.5_percentiles": np.quantile(values, [.025, .975]).tolist()}
                 for name, values in scores.items()}
    chosen = dev["selected_mixture"]
    paired = {name: {"mean_difference": float((scores[chosen]-values).mean()), "resample_percentiles": np.quantile(scores[chosen]-values, [.025, .975]).tolist()}
              for name, values in scores.items() if name != chosen}
    result = {"development_ACT_success_vs_failure": grouped, "mixture_resample_stability": stability,
              "chosen_minus_others_paired_resampling": paired,
              "resample_intervals_are_not_ACT_success_confidence_intervals": True,
              "alignment_association_is_not_proof_auxiliary_head_will_help": True,
              "final_layouts_or_scores_read": False}
    (RUN / "development_diagnosis.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__": main()
