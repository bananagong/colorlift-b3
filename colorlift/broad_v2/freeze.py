"""Freeze all final generation rules before creating a single final layout."""
from datetime import datetime, timezone
import json
import numpy as np
from .spec import ROOT, RUN, CATEGORIES, VALID, REGIONS, sha, write_once


def quotas(total, weights):
    exact = np.array(weights)*total
    counts = np.floor(exact).astype(int)
    for i in np.argsort(-(exact-counts))[:total-counts.sum()]: counts[i] += 1
    return dict(zip(CATEGORIES, counts.tolist()))


def main():
    path = RUN / "final_protocol.json"
    if path.exists():
        protocol = json.loads(path.read_text()); verify(protocol); return protocol
    assert not (RUN / "final_sealed").exists(), "Freeze before final generation"
    dev_protocol = json.loads((RUN / "development_protocol.json").read_text())
    for relative, digest in dev_protocol["source_snapshot"].items(): assert sha(ROOT / relative) == digest, relative
    assert sha(ROOT / dev_protocol["baseline_checkpoint"]) == dev_protocol["baseline_sha256"]
    result = json.loads((RUN / "development_results.json").read_text())
    donor = {}
    for category in CATEGORIES:
        old = result["paired_actor_results"][category+"/current_expert"]
        safe = result["paired_actor_results"][category+"/safe_expert"]
        donor[category] = "safe_expert" if safe["contact_qualified_successes"] > old["contact_qualified_successes"]+2 and safe["wrong_colors"] == 0 else "current_expert"
    protocol = {"version": 2, "frozen_at_UTC": datetime.now(timezone.utc).isoformat(),
                "old_results_status": "development evidence only, never final evidence",
                "development_protocol_sha256": sha(RUN / "development_protocol.json"),
                "development_results_sha256": sha(RUN / "development_results.json"),
                "selected_accepted_layout_mixture": result["selected_mixture"], "mixture_weights": result["selected_weights"],
                "mixture_not_claimed_ACT_optimal": True,
                "workspace_xy_m": [-.145, .145], "valid": VALID,
                "fixed_camera_and_light_for_all_three_suites": True,
                "reserved_regions_xmin_xmax_ymin_ymax": REGIONS,
                "Broad_IID": {"layouts": 256, "proposal_mixture": "same as training", "spatial_support": "outside reserved rectangles expanded by8mm", "relation_support": "min pair>=80mm and normalized area>=0.080", "designation": "IID conditional on policy-independent validity and preregistered duplicate rejection"},
                "Spatial_OOD": {"layouts": 256, "proposal_mixture": "same joint proposal mixture; conditional rejection without coordinate overwrite", "target_support": "exactly one slot inside a reserved rectangle by>=3mm", "distractor_support": "outside expanded reserved regions", "relation_support": "same as training", "colors_commands": "3 cyclic recolorings; command the reserved physical slot's actual color"},
                "Relation_OOD": {"layouts": 256, "subtypes": {"near_gap": 128, "aligned": 128}, "spatial_support": "same as training; no reserved region", "near_gap": "minimum center distance62-74mm, area ratio>=.080", "aligned": "minimum center distance>=80mm, area ratio<=.025", "colors_commands": "3 cyclic recolorings x3 colors"},
                "train_accepted_successful_geometries": quotas(192, result["selected_weights"]),
                "val_accepted_successful_geometries": quotas(48, result["selected_weights"]),
                "training_expert_by_category": donor,
                "expert_choice_rule": "prefer current for efficiency; switch category only if safe gives >2 additional qualified development successes out of48 and zero wrong color; no significance claim",
                "training_admission": "complete9 real color/command episodes all task-successful and strict contact-qualified; preserve every rejected candidate",
                "final_admission": "policy-independent Valid only; no expert/policy success filtering or resampling after policy failure",
                "geometry_id": "unordered three XY centers rounded1mm; independent of yaw, color, target command and robot seed",
                "position_group_id": "unordered three XY centers rounded1cm; all variants/groups remain in one split",
                "split_exclusions": "legacy layouts, development/preflight, train, val and final suites all reserved in atomic geometry/coarse-group registry",
                "geometry_statistics": "one independent geometry per cluster; average color/permutation outcomes within geometry; layout-cluster bootstrap confidence intervals; report seed spread separately",
                "coverage": ["XY10x10 occupancy", "joint6D centered discrepancy of accepted layouts", "pair distances and area", "unordered XY RMS nearest train layout", "proposal/rejection counts and category quotas"],
                "Sobol": "scipy.stats.qmc.Sobol(d=6,scramble=True).random_base2(16); rejection loses formal balance guarantee",
                "final_seed_streams": {"broad_iid": 6100, "spatial_ood": 6200, "relation_near_gap": 6300, "relation_aligned": 6400},
                "train_seed_base": 5100, "val_seed_base": 5200,
                "final_policy_evaluation_permitted_now": False,
                "final_model_selection_permitted": False,
                "future_unseal_requirement": "lock all candidate checkpoint hashes and ablation/hyperparameter/checkpoint-selection rules before first final inference; final results never select checkpoint",
                "ACT_change_in_this_phase": False,
                "future_ablation_order": ["same-data ACT vs goal-relative L1 head, then add pregrasp pose head; labels only during training", "matched layouts/budget original vs safe-approach demonstrations", "matched contact/slip recovery demonstrations"],
                "future_auxiliary_loss": "L_ACT + lambda_p*L1(predicted target-minus-eef, label); lambda selected on development/validation only",
                "privileged_labels_not_formal_policy_inputs": ["target_position_label", "goal_relative", "pregrasp_position_label", "pregrasp_relative", "pregrasp_rotation_label", "body IDs", "layout category"],
                "data_field_semantics": {"actions": "canonical world task goal XYZ plus phase; phase0 labels the validated safe pregrasp height .925m, not intermediate route points",
                                         "servo_waypoints": "actual expert route waypoint; current expert may use a lower .865m pregrasp",
                                         "osc_actions": "actual float64 7D OSC commands executed in MuJoCo",
                                         "pregrasp_position_label": "planner-validated safety pose, may be above the current expert route; label for future auxiliary head only"},
                "success": "original target lift>4cm + stable2s; strict contact score and wrong-color rule unchanged",
                "max_physical_GPUs": [1, 2],
                "source_snapshot": {str(p.relative_to(ROOT)): sha(p) for p in sorted(__import__("pathlib").Path(__file__).parent.glob("*.py"))}}
    write_once(path, protocol)
    (RUN / "final_protocol.sha256").write_text(sha(path)+"\n")
    print(json.dumps({"frozen": True, "sha256": sha(path), "mixture": result["selected_mixture"], "donor": donor}, indent=2), flush=True)
    return protocol


def verify(protocol):
    assert sha(RUN / "final_protocol.json") == (RUN / "final_protocol.sha256").read_text().strip()
    for relative, digest in protocol["source_snapshot"].items(): assert sha(ROOT / relative) == digest, relative


if __name__ == "__main__": main()
