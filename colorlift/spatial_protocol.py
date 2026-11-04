"""Declare spatial data, order comparison and held-out tests before new training."""
import hashlib
import json
from .spatial_gate import RUN, ROOT
from .spatial_collect import require_gate


def freeze():
    require_gate()
    files = ["colorlift/spatial_layouts.py", "colorlift/spatial_env.py", "colorlift/spatial_rollout.py", "colorlift/spatial_collect.py",
             "colorlift/event_control.py", "colorlift/env.py"]
    source = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files}
    protocol = {"gate_required_and_passed": True, "data": {"candidate_geometries_per_level": {"train": 80, "val": 15},
                 "selected_geometries_per_level": {"train": 60, "val": 10}, "levels": ["easy", "medium", "hard"],
                 "physical_rollouts_per_geometry": 9, "color_permutations_per_training_geometry": 3,
                 "each_color_at_each_physical_slot": True, "layout_count_ignores_colors_commands_and_perturbations": True,
                 "Easy": "three central well-separated anchors; independent +/-0.02m vertex jitter; centers>=0.108m",
                 "Medium": "cluster-center +/-0.04m jitter about an anchor, random orientation, centers~0.078-0.092m",
                 "Hard": "all slots at reachable workspace edge|coordinate|0.125-0.145m; yaw +/-0.30rad",
                 "teacher": "new MuJoCo simulations for each new geometry/slot/color; privileged body pose labels only for expert",
                 "raw_actual_source_episodes": 1890, "additional_replays": 630},
                "training": {"variants": ["curriculum", "shuffled"], "seeds": [43, 44], "steps": 5000, "batch_size": 36,
                  "same_initial_checkpoint": True, "same_exact_frame_exposures": True, "same_optimizer_and_LR_schedule": True,
                  "curriculum": [[1, 1500, [1, 0, 0]], [1501, 3000, [.3, .7, 0]], [3001, 5000, [.1, .3, .6]]],
                  "shuffled": "randomly reorder exactly the same batches", "curriculum_is_fixed_schedule": True,
                  "selection": "best fixed balanced val score at3500/4000/4500/5000; never test-based"},
                "held_out_evaluation": {"levels": {"easy": 30, "medium": 30, "hard": 30, "ood_position": 20, "ood_light": 20, "ood_camera": 20},
                  "commands_per_geometry": 3, "variants": 5, "trials_per_variant": 450,
                  "counterfactual": {"geometries_per_Easy_Medium_Hard": 4, "color_permutations": 6, "commands_per_permutation": 3,
                                     "physical_positions_and_orientations_fixed": True, "trials_per_variant": 216},
                  "OOD_is_never_training_or_validation": True},
                "sources": source}
    path = RUN / "spatial_protocol.json"
    if path.exists():
        assert json.loads(path.read_text()) == protocol
    else:
        path.write_text(json.dumps(protocol, indent=2))
    print("SPATIAL PROTOCOL FROZEN", flush=True)


if __name__ == "__main__": freeze()
