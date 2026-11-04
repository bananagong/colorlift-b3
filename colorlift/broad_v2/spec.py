from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "experiments/colorlift_broad_protocol_v2"
CATEGORIES = ("uniform", "sobol", "near", "edge_adverse")
COLORS = ("red", "yellow", "blue")
WORKSPACE = .145
DEV_PER_CATEGORY = 64
EXPERT_DEV_PER_CATEGORY = 16
MIXES = {"balanced": [.25, .25, .25, .25], "coverage": [.35, .40, .15, .10],
         "contact": [.20, .20, .40, .20], "edge": [.20, .20, .20, .40]}
REGIONS = [[-.145, -.100, .050, .145], [.100, .145, -.145, -.050]]
VALID = {"footprint_gap_m": .003, "settle_seconds": .6, "settle_max_drift_m": .0015,
         "settle_max_height_error_m": .0015, "settle_max_tilt_rad": .04,
         "settle_max_linear_speed_m_s": .01, "settle_max_angular_speed_rad_s": .10,
         "IK_position_error_m": .001, "IK_rotation_error_rad": .025, "joint_limit_margin_rad": .005,
         "visible_pixels_per_camera": 6, "visible_fraction_best_camera": .45,
         "visible_fraction_other_camera": .12, "image_border_px": 2,
         "unwanted_static_penetration_m": .0005, "joint_sweep_samples": 8,
         "region_guard_m": .008, "region_inside_margin_m": .003,
         "train_min_center_distance_m": .080, "train_min_area_ratio": .080,
         "OOD_near_distance_m": [.062, .074], "OOD_aligned_max_area_ratio": .025,
         "pregrasp_z_m": .925, "safe_transport_z_m": .98, "grasp_z_m": .818,
         "lift_z_m": .94}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_once(path, value):
    value = json.loads(json.dumps(value))
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert json.loads(path.read_text()) == value, f"Frozen artifact differs: {path}"
    else:
        path.write_text(json.dumps(value, indent=2))


def development_protocol():
    value = {"version": 2, "purpose": "data distribution, validity and expert development; no ACT architecture/weight updates",
             "categories": CATEGORIES, "workspace_half_extent_m": WORKSPACE, "valid": VALID,
             "held_out_rectangles_xyxy": REGIONS, "development_valid_layouts_per_category": DEV_PER_CATEGORY,
             "paired_expert_layouts_per_category": EXPERT_DEV_PER_CATEGORY, "color_commands_per_layout": 3,
             "candidate_accepted_layout_mixtures": MIXES, "mixture_resamples": 100, "mixture_layout_budget": 128,
             "selection_score": {"XY_grid_occupancy": .45, "relation_histogram_entropy": .20,
                                 "safe_expert_contact_success": .25, "proposal_valid_acceptance": .10},
             "selection_limitation": "development data-quality proxy, not a demonstrated ACT-optimal mixture",
             "source_snapshot": {str(p.relative_to(ROOT)): sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
             "baseline_checkpoint": "experiments/colorlift_recovery_v1/model/policy.pt",
             "baseline_sha256": sha(ROOT / "experiments/colorlift_recovery_v1/model/policy.pt"),
             "training_policy": "do not train until data/protocol foundation is complete; geometry auxiliary learning then safe-approach then contact recovery in separate ablations",
             "final_outcomes_must_not_inform_development": True}
    write_once(RUN / "development_protocol.json", value)
    return value
