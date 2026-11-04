"""Audited mixed-source mmap cache and reproducible random/balanced curricula."""
import argparse
from collections import Counter, defaultdict
import csv
import json
import numpy as np
import torch
from .spec import ROOT, DATA, RUN, COLORS, SOURCE_WEIGHTS, PHASE_WEIGHTS, STEPS, BATCH, freeze, sha
from ..broad_v2.layouts import region_ok, relation_ok


def numeric_key(xy, decimals=3):
    return tuple(sorted(tuple(float(v) for v in np.round(p, decimals)) for p in xy))


def spatial(meta):
    points = np.array([meta["initial_positions"][c][:2] for c in COLORS])
    target = np.array(meta["initial_positions"][meta["target_color"]][:2])
    distance = min(float(np.linalg.norm(target-p)) for c, p in zip(COLORS, points) if c != meta["target_color"])
    edge = float(np.max(np.abs(target)))
    cell = np.digitize(target, [-.05, .05]).astype(int)
    difficulty = .55*np.clip((.15-distance)/.07, 0, 1)+.45*np.clip(edge/.145, 0, 1)
    return int(cell[0]*3+cell[1]), int(distance < .095 or edge > .115), float(difficulty)


def entries():
    new = json.loads((DATA / "training_manifest.json").read_text())
    result = {"train": [], "val": []}
    val_cells = {numeric_key(row["layout"]["xy"], 2) for row in new["layouts"] if row["split"] == "val"}
    rejected = Counter()
    for group in new["layouts"]:
        for relative in group["episodes"]:
            path = DATA / relative
            meta = json.loads((path / "metadata.json").read_text())
            result[group["split"]].append({"root": str(DATA), "path": relative+"/trajectory.npz", "supervision": relative+"/supervision.npz",
                "trace": relative+"/steps.csv", "source": 0, "category": ("uniform", "sobol", "near", "edge_adverse").index(group["layout"]["category"]),
                "geometry_id": group["geometry_id"], "target_color": meta["target_color"], "steps": meta["control_steps"],
                "spatial": spatial(meta), "auxiliary_labels": True})
    for source, name in ((1, "colorlift_conditional_v1"), (2, "colorlift_recovery_v1")):
        root = ROOT / "experiments" / name
        original = json.loads((root / "data/manifest.json").read_text())["episodes"]
        groups = defaultdict(list)
        for row in original:
            if row["split"] == "train": groups[numeric_key([p[:2] for p in row["initial_positions"].values()])].append(row)
        for key, rows in groups.items():
            xy = [p[:2] for p in rows[0]["initial_positions"].values()]
            reason = None
            if not all(r.get("eligible", False) and r["success"] and not r["collision_steps"] and not r["wrong_color_grasp"] for r in rows): reason = "legacy_quality"
            elif not region_ok(xy, "train") or not relation_ok(xy, "train"): reason = "held_out_region_or_relation_support"
            elif numeric_key(xy, 2) in val_cells: reason = "new_validation_coarse_group"
            if reason: rejected[(name, reason)] += len(rows); continue
            for row in rows:
                path = row["file"] if source == 1 else row["directory"]+"/trajectory.npz"
                trace = str(__import__("pathlib").Path(path).with_suffix(".csv")) if source == 1 else row["directory"]+"/steps.csv"
                gid = __import__("hashlib").sha256(json.dumps(key).encode()).hexdigest()
                result["train"].append({"root": str(root), "path": path, "trace": trace, "source": source,
                     "category": 3+source, "geometry_id": gid, "target_color": row["target_color"], "steps": row["control_steps"],
                     "spatial": spatial(row), "auxiliary_labels": False})
    return result, {name+":"+reason: n for (name, reason), n in rejected.items()}


SHAPES = {"states": ((16,), np.float32), "actions": ((7,), np.float32), "images": ((2,128,128,3), np.uint8),
          "classes": ((2,128,128), np.uint8), "colors": ((), np.uint8), "phases": ((), np.uint8), "source": ((), np.uint8),
          "category": ((), np.uint8), "cell": ((), np.uint8), "difficulty_bin": ((), np.uint8), "difficulty": ((), np.float32),
          "geometry": ((), np.int32), "valid": ((), np.bool_), "aux_valid": ((), np.bool_), "ends": ((), np.int64),
          "relative": ((3,), np.float32), "pre_relative": ((3,), np.float32), "pre_rotation": ((6,), np.float32), "recovery": ((), np.bool_)}


def prepare():
    protocol = freeze()
    cache = RUN / "cache"; cache.mkdir(exist_ok=True)
    if (cache / "completed.json").exists(): return
    all_entries, rejected = entries()
    summary = {}
    manifest_rows = []
    for split in ("train", "val"):
        rows = all_entries[split]; count = sum(r["steps"] for r in rows)
        arrays = {key: np.lib.format.open_memmap(cache / f"{split}_{key}.npy", mode="w+", dtype=dtype, shape=(count,)+shape) for key, (shape, dtype) in SHAPES.items()}
        ids = {gid: i for i, gid in enumerate(sorted({r["geometry_id"] for r in rows}))}
        offset = 0
        for index, row in enumerate(rows):
            path = __import__("pathlib").Path(row["root"]) / row["path"]
            with np.load(path, allow_pickle=False) as item:
                n = len(item["states"]); assert n == row["steps"]
                block = slice(offset, offset+n)
                states = item["states"][:, :16].copy()
                phases = states[:, 12:16].argmax(1)
                actions = item["actions"].copy()
                valid = item["supervised"].astype(bool) if row["source"] == 2 else np.ones(n, dtype=bool)
                if row["source"] == 2:
                    with (path.parent / "steps.csv").open(encoding="utf-8-sig", newline="") as stream: trace = list(csv.DictReader(stream))
                    actions[phases < 2, :2] = np.array([[float(r["target_before_"+a]) for a in "xy"] for r in trace])[phases < 2]
                # Match the unchanged executor; safety pose is an auxiliary label.
                actions[phases == 0, 2] = .865
                ends = np.empty(n, dtype=np.int64); end = n-1
                for t in range(n-1, -1, -1):
                    if not valid[t]: end = t-1
                    ends[t] = max(t, end)+offset
                values = {"states": states, "actions": actions, "images": item["images"],
                          "classes": np.where(item["masks"].max(-1)>0, item["masks"].argmax(-1)+1, 0),
                          "colors": COLORS.index(row["target_color"]), "phases": phases, "source": row["source"], "category": row["category"],
                          "cell": row["spatial"][0], "difficulty_bin": row["spatial"][1], "difficulty": row["spatial"][2],
                          "geometry": ids[row["geometry_id"]], "valid": valid, "ends": ends,
                          "recovery": item["recovery"].astype(bool) if "recovery" in item else np.zeros(n, bool),
                          "relative": np.zeros((n,3), np.float32), "pre_relative": np.zeros((n,3), np.float32),
                          "pre_rotation": np.zeros((n,6), np.float32), "aux_valid": np.zeros(n, bool)}
                if row["source"] == 0:
                    with np.load(__import__("pathlib").Path(row["root"]) / row["supervision"], allow_pickle=False) as labels:
                        values.update(relative=labels["goal_relative"], pre_relative=labels["pregrasp_relative"],
                                      pre_rotation=labels["pregrasp_rotation"][:, :, :2].reshape(n,6), aux_valid=valid)
                for key in arrays: arrays[key][block] = values[key]
            manifest_rows.append(row | {"split": split, "offset": offset, "npz_sha256": sha(path)})
            offset += n
            if (index+1)%150 == 0: print("CACHE", split, index+1, len(rows), offset, flush=True)
        for array in arrays.values(): array.flush()
        summary[split] = {"frames": count, "episodes": len(rows), "geometries": len(ids),
                          "episodes_by_source": dict(Counter(r["source"] for r in rows)), "valid_frames": int(arrays["valid"].sum())}
        assert all(np.any(arrays["valid"] & (arrays["source"] == source)) for source in ((0,1,2) if split == "train" else (0,)))
    value = {"splits": summary, "legacy_rejections": rejected, "rows": manifest_rows,
             "new_validation_never_used_for_training": True, "legacy_validation_not_imported": True,
             "old_task_statistics_preserved_for_explicit_warm_start": True, "injected_controls_masked": True}
    (RUN / "data_manifest.json").write_text(json.dumps(value, indent=2))
    (cache / "completed.json").write_text(json.dumps({"complete": True, "summary": summary}, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


class Dataset:
    def __init__(self, split, stats):
        self.split, self.stats = split, stats
        self.arrays = {key: np.load(RUN / "cache" / f"{split}_{key}.npy", mmap_mode="r") for key in SHAPES}
        self.pools = {}; self.balanced = {}
        a = self.arrays
        for source in np.unique(a["source"]):
            for color in range(3):
                for phase in range(4):
                    mask = a["valid"] & (a["source"]==source) & (a["colors"]==color) & (a["phases"]==phase)
                    ids = np.flatnonzero(mask); self.pools[int(source),color,phase] = ids
                    if not len(ids): raise RuntimeError((split, source, color, phase, "empty pool"))
                    if source == 0:
                        bins = defaultdict(dict)
                        for category, cell, difficulty, geometry in np.unique(np.c_[a["category"][ids], a["cell"][ids], a["difficulty_bin"][ids], a["geometry"][ids]], axis=0):
                            selected = ids[(a["category"][ids]==category)&(a["cell"][ids]==cell)&(a["difficulty_bin"][ids]==difficulty)&(a["geometry"][ids]==geometry)]
                            bins[int(category),int(cell),int(difficulty)][int(geometry)] = selected
                        categories = defaultdict(list)
                        for key, value in bins.items(): categories[key[0]].append(value)
                        self.balanced[color,phase] = dict(categories)

    def plan(self, seed, mode):
        out = RUN / "plans"; out.mkdir(exist_ok=True)
        path = out / f"{mode}_{seed}.npz"
        if path.exists(): return np.load(path)["ids"]
        rng = np.random.default_rng(seed); total = STEPS*BATCH
        source = np.concatenate([np.full(int(total*w), i, dtype=np.uint8) for i,w in enumerate(SOURCE_WEIGHTS)])
        assert len(source) == total; rng.shuffle(source)
        colors = np.arange(total)%3; rng.shuffle(colors)
        phases = rng.choice(4, total, p=PHASE_WEIGHTS)
        ids = np.empty(total, np.int64)
        for i, (s,c,p) in enumerate(zip(source,colors,phases)):
            if mode in ("balanced", "curriculum") and s == 0:
                categories = self.balanced[int(c),int(p)]; category = int(rng.choice(list(categories)))
                bins = categories[category]
                group = bins[int(rng.integers(len(bins)))]; geoms = list(group)
                ids[i] = rng.choice(group[geoms[int(rng.integers(len(geoms)))]] )
            else: ids[i] = rng.choice(self.pools[int(s),int(c),int(p)])
        if mode == "curriculum":
            # This method's random draws above are identical to balanced.
            original = ids.copy()
            for s in range(3):
                for c in range(3):
                    for p in range(4):
                        locations = np.flatnonzero((source==s)&(colors==c)&(phases==p))
                        values = ids[locations]
                        ids[locations] = values[np.argsort(self.arrays["difficulty"][values], kind="stable")]
            assert np.array_equal(np.sort(original), np.sort(ids))
        np.savez_compressed(path, ids=ids.reshape(STEPS,BATCH))
        return ids.reshape(STEPS,BATCH)

    def batch(self, ids):
        a = self.arrays; ids = np.asarray(ids)
        states = (a["states"][ids]-self.stats["state_mean"])/self.stats["state_std"]
        future = ids[:,None]+np.arange(4)
        length = np.where(np.isin(a["phases"][ids], [1,2]) | a["recovery"][ids], 2, 4)
        pad = (future>a["ends"][ids,None]) | (np.arange(4)[None]>=length[:,None])
        future = np.minimum(future, a["ends"][ids,None])
        assert np.all(a["valid"][ids]) and not np.any(~a["valid"][future] & ~pad)
        actions = a["actions"][future].copy(); actions[:,:,:3] = (actions[:,:,:3]-self.stats["xyz_mean"])/self.stats["xyz_std"]
        convert = lambda x: torch.as_tensor(np.asarray(x).copy(), device="cuda")
        return {"states": convert(states).float(), "rgb": convert(a["images"][ids].transpose(0,1,4,2,3)).float()/255,
                "colors": convert(a["colors"][ids]).long(), "actions": convert(actions).float(), "pad": convert(pad).bool(),
                "classes": convert(a["classes"][ids]).long(), "relative": convert(a["relative"][ids]).float(),
                "pre_relative": convert(a["pre_relative"][ids]).float(), "pre_rotation": convert(a["pre_rotation"][ids]).float(),
                "aux_valid": convert(a["aux_valid"][ids]).bool()}


if __name__ == "__main__": prepare()
