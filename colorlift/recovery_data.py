"""Old clean + new perturbed expert data, with disturbance gaps and phase-biased sampling."""
import json
import csv
import numpy as np
import torch
from .conditional import RUN as ORIGINAL
from .config import ROOT, COLORS

RUN=ROOT / "experiments/colorlift_recovery_v1"
CATEGORIES=("approach", "precise_align", "closing_contact", "recovery", "lift_hold")
PROBABILITIES=np.array([.05, .30, .30, .30, .05])


def merge_data():
    episodes=[]
    for shard in range(2):
        assert (RUN / f"data/shard_{shard}_completed.json").exists()
        episodes.extend(json.loads((RUN / f"data/manifest_shard_{shard}.json").read_text())["episodes"])
    for row in episodes: row["eligible"]=bool(row["success"] and len(row["noise_events"])==2 and row["recovery_steps"]>0)
    counts={split: {color: sum(row["split"]==split and row["target_color"]==color and row["eligible"] for row in episodes) for color in COLORS} for split in ("train", "val")}
    assert min(counts["train"].values())>=60 and min(counts["val"].values())>=10, counts
    manifest={"episodes": episodes, "eligible_counts": counts, "selection": "successful recoveries with two actual perturbations; all failures retained", "disturbance_actions_supervised": False}
    (RUN / "data/manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


class RecoveryDataset:
    def __init__(self, split, stats, include_noisy=True):
        blocks={key: [] for key in ("states", "images", "masks", "actions", "colors", "categories", "source", "valid", "ends")}
        offset=0
        original=json.loads((ORIGINAL / "data/manifest.json").read_text())["episodes"]
        sources=[(ORIGINAL, [row for row in original if row["split"]==split and row["eligible"]], 0)]
        if include_noisy:
            new=json.loads((RUN / "data/manifest.json").read_text())["episodes"]
            sources.append((RUN, [row for row in new if row["split"]==split and row["eligible"]], 1))
        for root, entries, source in sources:
            for row in entries:
                path=root / (row["directory"]+"/trajectory.npz" if source else row["file"])
                with np.load(path, allow_pickle=False) as item:
                    n=len(item["states"])
                    phase=item["states"][:, 12:16].argmax(1)
                    categories=np.choose(phase, [0, 1, 2, 4]) if not source else np.array([CATEGORIES.index(str(category)) if str(category) in CATEGORIES else -1 for category in item["category"]])
                    valid=np.ones(n, dtype=bool) if not source else item["supervised"].astype(bool)
                    ends=np.empty(n, dtype=int)
                    end=n-1
                    for t in range(n-1, -1, -1):
                        if not valid[t]: end=t-1
                        ends[t]=max(t, end)
                    for key in ("images", "masks", "actions"): blocks[key].append(item[key].copy())
                    if source:
                        # ACT predicts the task macro target, not the temporary IK retreat point.
                        # Raw servo waypoints and OSC actions remain untouched in trajectory.npz/CSV.
                        with (path.parent / "steps.csv").open(encoding="utf-8-sig") as stream:
                            trace=list(csv.DictReader(stream))
                        target_xy=np.array([[float(step["target_before_"+a]) for a in "xy"] for step in trace])
                        macro=blocks["actions"][-1]
                        reaching=phase<2
                        macro[reaching, :2]=target_xy[reaching]
                        macro[phase==0, 2]=.865
                        macro[phase==1, 2]=.818
                    blocks["states"].append(item["states"][:, :16].copy())
                    blocks["colors"].append(np.full(n, COLORS.index(row["target_color"])))
                    blocks["categories"].append(categories)
                    blocks["source"].append(np.full(n, source))
                    blocks["valid"].append(valid)
                    blocks["ends"].append(ends+offset)
                    offset+=n
        for key, values in blocks.items(): setattr(self, key, np.concatenate(values))
        self.stats=stats
        self.states=((self.states-stats["state_mean"])/stats["state_std"]).astype(np.float32)
        self.actions[:, :3]=(self.actions[:, :3]-stats["xyz_mean"])/stats["xyz_std"]
        self.pool={}
        for c in range(3):
            for category in range(5):
                for source in (0, 1):
                    self.pool[c, category, source]=np.flatnonzero(self.valid & (self.colors==c) & (self.categories==category) & (self.source==source))
        self.samples=np.zeros(5, dtype=int)

    def sample_indices(self, rng, batch):
        colors=np.arange(batch)%3; rng.shuffle(colors)
        categories=rng.choice(5, batch, p=PROBABILITIES)
        result=[]
        for c, category in zip(colors, categories):
            source=1 if category==3 else int(rng.random()<.5)
            pool=self.pool[c, category, source]
            if not len(pool): pool=self.pool[c, category, 1-source]
            if not len(pool): raise RuntimeError(f"Missing category {c},{category}")
            result.append(rng.choice(pool)); self.samples[category]+=1
        return np.array(result)

    def batch(self, ids):
        future=ids[:, None]+np.arange(4)
        lengths=np.where(np.isin(self.categories[ids], [1, 2, 3]), 2, 4)
        pad=(future>self.ends[ids, None]) | (np.arange(4)[None]>=lengths[:, None])
        future=np.minimum(future, self.ends[ids, None])
        assert np.all(self.valid[ids]) and not np.any(~self.valid[future] & ~pad)
        convert=lambda x: torch.as_tensor(x, device="cuda")
        return (convert(self.states[ids]), convert(self.images[ids].transpose(0, 1, 4, 2, 3).copy()).float()/255,
                convert(self.colors[ids]), convert(self.actions[future]), convert(pad),
                convert(self.masks[ids].transpose(0, 1, 4, 2, 3).copy()).float()/255)
