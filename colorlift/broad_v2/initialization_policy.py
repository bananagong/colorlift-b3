"""Keep task-pretraining history from invalidating the future Spatial-OOD claim."""
from datetime import datetime, timezone
import json
from .spec import RUN, sha, write_once


def main():
    path = RUN / "training_initialization_policy.json"
    if path.exists(): return
    value = {"frozen_at_UTC": datetime.now(timezone.utc).isoformat(),
             "geometry_protocol_sha256": sha(RUN / "final_protocol.json"),
             "reason": "legacy task sampling +/-0.11m intersects reserved Spatial-OOD rectangles; novel geometry IDs alone do not exclude historical regional exposure",
             "new_formal_candidates": "same fresh ACT initialization within each paired seed, or initialization with verified absence of reserved-region task exposure",
             "architecture": "retain shared RGB-ACT architecture; no architecture changes in this data phase",
             "generic_visual_pretraining": "allowed only if unrelated to these task-specific MuJoCo layouts; record source and hash equally for all candidates",
             "legacy_task_ACT_weights": "development/reference only; not a clean unseen-region initialization for formal Spatial-OOD ablations",
             "legacy_reference_scores_must_not_select_model": True,
             "checkpoint_lock_required_before_final_inference": ["checkpoint_sha256", "initialization_lineage", "training_geometry_whitelist", "validation_selection_rule", "all_hyperparameters"],
             "new_ACT_training_updates_at_freeze": 0, "final_policy_inference_count_at_freeze": 0,
             "sealed_final_geometries_modified": False,
             "timing": "additional training-lineage restriction frozen before new model training; original generation rules were frozen before final layouts"}
    write_once(path, value)
    print(json.dumps({"initialization_policy_frozen": True, "sha256": sha(path), "final_policy_inference_count": 0}, indent=2), flush=True)


if __name__ == "__main__": main()
