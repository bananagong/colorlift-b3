"""Explicit RGB/proprioception/color inputs; privileged geometry is label-only."""
import json
from pathlib import Path
import numpy as np
from .spec import RUN, COLORS


class SupervisedData:
    def __init__(self, split):
        assert split in ("train", "val"), "Final layouts are not supervised data"
        manifest = json.loads((RUN / "training_manifest.json").read_text())
        self.directories = [RUN / relative for row in manifest["layouts"] if row["split"] == split for relative in row["episodes"]]
        root = (RUN / "data_candidates").resolve()
        assert all(root in path.resolve().parents for path in self.directories)

    def step(self, episode_index, step):
        directory = self.directories[episode_index]
        meta = json.loads((directory / "metadata.json").read_text())
        with np.load(directory / "trajectory.npz", allow_pickle=False) as data:
            state = data["states"][step].astype(np.float32)
            color = COLORS.index(meta["target_color"])
            assert state.shape == (19,) and int(state[16:].argmax()) == color
            inputs = {"rgb": data["images"][step].copy(), "proprioception_and_phase": state[:16].copy(), "target_color_index": color}
            canonical_action = data["actions"][step].copy()
        with np.load(directory / "supervision.npz", allow_pickle=False) as supervision:
            labels = {"canonical_ACT_target": canonical_action, "goal_relative": supervision["goal_relative"][step].copy(),
                      "safe_pregrasp_relative": supervision["pregrasp_relative"][step].copy(),
                      "safe_pregrasp_rotation": supervision["pregrasp_rotation"][step].copy(),
                      "pregrasp_supervision_mask": bool(supervision["pregrasp_supervision_mask"][step])}
        return inputs, labels


def schema():
    return {"inputs": {"rgb": [2, 128, 128, 3], "proprioception_and_phase": [16], "target_color_index": "integer0..2"},
            "training_only_labels": {"canonical_ACT_target": [7], "goal_relative": [3], "safe_pregrasp_relative": [3],
                                     "safe_pregrasp_rotation": [3, 3], "pregrasp_supervision_mask": "bool"},
            "denied_inputs": ["target_position", "body_ID", "geometry_id", "layout_category", "safe_pregrasp_ground_truth"],
            "allowed_supervised_splits": ["train", "val"], "final_is_not_a_supervised_split": True,
            "ACT_architecture_changes": 0}


if __name__ == "__main__":
    (RUN / "input_label_schema.json").write_text(json.dumps(schema(), indent=2))
    for split in ("train", "val"):
        data = SupervisedData(split)
        inputs, labels = data.step(0, 0)
        assert set(inputs) == {"rgb", "proprioception_and_phase", "target_color_index"}
        assert inputs["rgb"].shape == (2, 128, 128, 3)
        print(split, len(data.directories), list(inputs), list(labels), flush=True)
