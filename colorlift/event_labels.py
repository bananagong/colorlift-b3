"""Keep physical servo commands separate from canonical ACT task-goal labels."""
import json
import numpy as np
from .conditional import configuration
from .event_efficiency import RUN


def write_labels(manifest):
    cfg, expert = configuration()
    total, changes = 0, 0
    for verification in manifest["episodes"]:
        if not verification["admitted"]:
            continue
        directory = (RUN / verification["data_file"]).parent
        meta = json.loads((directory / "metadata.json").read_text())
        with np.load(directory / "trajectory.npz", allow_pickle=False) as data:
            original = data["actions"].copy()
        labels = original.copy()
        phases = labels[:, 3:].argmax(1)
        estimate = np.asarray(meta["visual_estimate"])
        for phase, z in ((0, cfg.table_height+cfg.cube_size/2+expert.pregrasp_clearance), (1, cfg.grasp_height)):
            labels[phases == phase, :2] = estimate[:2]
            labels[phases == phase, 2] = z
        changed = np.linalg.norm(labels[:, :3]-original[:, :3], axis=1) > 1e-6
        changes += int(changed.sum())
        total += len(labels)
        np.savez_compressed(directory / "training_macro_labels.npz", actions=labels,
                            supervised=np.ones(len(labels), dtype=bool), phase=phases.astype(np.int8))
    semantics = {"raw_trace": "trajectory.npz retains unchanged actual OSC actions and actual stage servo waypoints",
                 "ACT_training": "training_macro_labels.npz actions: canonical task XY, stage Z and phase onehot",
                 "phase0_1_XY_source": "initial two-camera HSV estimate; not simulator target pose",
                 "reason": "a temporary clearance/retreat waypoint must not become the intended object location",
                 "phase2_3": "retain physically locked grasp XY and stage goal",
                 "episodes": manifest["admitted"], "frames": total, "changed_servo_waypoint_labels": changes}
    (RUN / "data/label_semantics.json").write_text(json.dumps(semantics, indent=2))
    return semantics
