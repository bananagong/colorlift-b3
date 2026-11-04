"""Shared mmap data and exactly exposure-matched curriculum/shuffled sampling plans."""
import argparse
import hashlib
import json
import numpy as np
from .spatial_gate import RUN
from .spatial_collect import require_gate

PHASE_PROBABILITIES = np.array([.10, .35, .35, .20])
STEPS = 5000
BATCH = 36


def prepare():
    require_gate()
    assert (RUN / "data/replay_verified.json").exists(), "Physically verify demonstrations before preparing training data"
    manifest = json.loads((RUN / "data/manifest.json").read_text())
    cache = RUN / "cache"; cache.mkdir(exist_ok=True)
    if (cache / "completed.json").exists():
        return
    summaries = {}
    for split in ("train", "val"):
        entries = [row for row in manifest["episodes"] if row["split"] == split]
        count = sum(row["control_steps"] for row in entries)
        shapes = {"states": ((count, 16), np.float32), "actions": ((count, 7), np.float32),
                  "images": ((count, 2, 128, 128, 3), np.uint8), "classes": ((count, 2, 128, 128), np.uint8),
                  "colors": ((count,), np.uint8), "phases": ((count,), np.uint8), "levels": ((count,), np.uint8),
                  "ends": ((count,), np.int64)}
        arrays = {key: np.lib.format.open_memmap(cache / f"{split}_{key}.npy", mode="w+", dtype=dtype, shape=shape)
                  for key, (shape, dtype) in shapes.items()}
        offset = 0
        for row in entries:
            path = RUN / row["directory"] / "trajectory.npz"
            assert hashlib.sha256(path.read_bytes()).hexdigest() == row["npz_sha256"]
            with np.load(path, allow_pickle=False) as source:
                n = len(source["states"]); target = slice(offset, offset+n)
                arrays["states"][target] = source["states"][:, :16]
                arrays["actions"][target] = source["actions"]
                arrays["images"][target] = source["images"]
                masks = source["masks"]
                arrays["classes"][target] = np.where(masks.max(-1) > 0, masks.argmax(-1)+1, 0)
                arrays["colors"][target] = ("red", "yellow", "blue").index(row["target_color"])
                arrays["phases"][target] = source["states"][:, 12:16].argmax(1)
                arrays["levels"][target] = ("easy", "medium", "hard").index(row["level"])
                arrays["ends"][target] = offset+n-1
                offset += n
        assert offset == count
        for array in arrays.values(): array.flush()
        summaries[split] = {"frames": count, "episodes": len(entries), "geometries": len({r["geometry_id"] for r in entries})}
        print("CACHE", split, summaries[split], flush=True)
    (cache / "completed.json").write_text(json.dumps({"complete": True, "splits": summaries}, indent=2))


class SpatialDataset:
    def __init__(self, split, stats):
        self.stats = stats
        cache = RUN / "cache"
        self.arrays = {key: np.load(cache / f"{split}_{key}.npy", mmap_mode="r")
                       for key in ("states", "actions", "images", "classes", "colors", "phases", "levels", "ends")}
        self.pools = {}
        for level in range(3):
            for color in range(3):
                for phase in range(4):
                    ids = np.flatnonzero((self.arrays["levels"] == level) & (self.arrays["colors"] == color) & (self.arrays["phases"] == phase))
                    assert len(ids), (split, level, color, phase)
                    self.pools[level, color, phase] = ids

    def indices(self, rng, level_probabilities, count):
        colors = np.arange(count)%3; rng.shuffle(colors)
        phases = rng.choice(4, count, p=PHASE_PROBABILITIES)
        levels = rng.choice(3, count, p=level_probabilities)
        return np.array([rng.choice(self.pools[level, color, phase]) for level, color, phase in zip(levels, colors, phases)], dtype=np.int64)

    def batch(self, ids):
        import torch
        a = self.arrays
        future = ids[:, None]+np.arange(4)
        lengths = np.where(np.isin(a["phases"][ids], (1, 2)), 2, 4)
        pad = (future > a["ends"][ids, None]) | (np.arange(4)[None] >= lengths[:, None])
        future = np.minimum(future, a["ends"][ids, None])
        states = ((a["states"][ids]-self.stats["state_mean"])/self.stats["state_std"]).astype(np.float32)
        actions = np.array(a["actions"][future])
        actions[..., :3] = (actions[..., :3]-self.stats["xyz_mean"])/self.stats["xyz_std"]
        classes = np.array(a["classes"][ids])
        masks = (classes[..., None] == np.arange(1, 4)).transpose(0, 1, 4, 2, 3).copy()
        convert = lambda value: torch.as_tensor(value, device="cuda")
        return (convert(states), convert(a["images"][ids].transpose(0, 1, 4, 2, 3).copy()).float()/255,
                convert(np.array(a["colors"][ids], dtype=np.int64)), convert(actions), convert(pad), convert(masks).float())


def sampling_plan(seed, dataset):
    out = RUN / "plans"; out.mkdir(exist_ok=True)
    path = out / f"seed_{seed}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as source:
            return source["curriculum"].copy(), source["shuffled"].copy()
    rng = np.random.default_rng(seed)
    sequence = []
    for step in range(STEPS):
        probabilities = [1., 0., 0.] if step < 1500 else [.3, .7, 0.] if step < 3000 else [.1, .3, .6]
        sequence.append(dataset.indices(rng, probabilities, BATCH))
    curriculum = np.array(sequence)
    permutation = rng.permutation(STEPS)
    shuffled = curriculum[permutation].copy()
    assert np.array_equal(np.sort(curriculum.ravel()), np.sort(shuffled.ravel()))
    np.savez_compressed(path, curriculum=curriculum, shuffled=shuffled, batch_permutation=permutation)
    info = {"seed": seed, "steps": STEPS, "batch_size": BATCH, "exact_same_frame_multiset": True,
            "curriculum_stages": [{"steps": [1, 1500], "level_probabilities": [1, 0, 0]},
                                  {"steps": [1501, 3000], "level_probabilities": [.3, .7, 0]},
                                  {"steps": [3001, 5000], "level_probabilities": [.1, .3, .6]}],
            "shuffled_control": "same 5000 batches, randomly reordered; not a different dataset or exposure budget",
            "phase_sampling": PHASE_PROBABILITIES.tolist(), "plan_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (out / f"seed_{seed}.json").write_text(json.dumps(info, indent=2))
    return curriculum, shuffled


if __name__ == "__main__":
    prepare()
    import torch
    from .spatial_gate import CHECKPOINT
    initial = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    dataset = SpatialDataset("train", initial["stats"])
    for seed in (43, 44): sampling_plan(seed, dataset)
