"""Generate train/val success data and sealed final layouts under frozen rules."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from .spec import ROOT, RUN, CATEGORIES, COLORS, sha
from .layouts import geometry_id, recolor
from .conditional_proposals import ConditionalProposals
from .freeze import main as freeze, quotas, verify


def group_id(xy):
    return __import__("hashlib").sha256(json.dumps(sorted(tuple(np.round(p, 2)) for p in xy)).encode()).hexdigest()


def reserve(layout, split):
    registry = RUN / "registry"
    registry.mkdir(exist_ok=True)
    if (registry / ("geometry_"+layout["geometry_id"]+".json")).exists(): return False
    value = {"geometry_id": layout["geometry_id"], "position_group_id": group_id(layout["xy"]), "split": split, "xy": layout["xy"]}
    try:
        with (registry / ("group_"+value["position_group_id"]+".json")).open("x") as stream: json.dump(value, stream)
        with (registry / ("geometry_"+value["geometry_id"]+".json")).open("x") as stream: json.dump(value, stream)
    except FileExistsError: return False
    return True


def seed_registry():
    # Preserve old experiments as development evidence and keep their geometries
    # out of the newly generated final evidence.
    old = ROOT / "experiments/colorlift_spatial_curriculum_v1"
    layouts = []
    manifest = json.loads((old / "data/manifest.json").read_text())
    for row in manifest["layouts"]:
        meta = json.loads((old / row["episodes"][0] / "metadata.json").read_text())
        layouts.append(meta["layout"])
    scenarios = json.loads((old / "test_scenarios.json").read_text())
    layouts.extend(scenarios["layouts"]+scenarios["counterfactual_base_layouts"])
    for seed in range(1000000, 1001000):
        row = json.loads((old / f"gate/episodes/scene_{seed}/red/metadata.json").read_text())
        layouts.append({"xy": [p[:2] for p in row["initial_positions"].values()]})
    # Include rejected/unselected legacy candidates and earlier three-color
    # runs, not just the selected manifest rows listed above.
    for path in (ROOT / "experiments").rglob("metadata.json"):
        if RUN in path.parents: continue
        meta = json.loads(path.read_text())
        points = meta.get("physical_slot_positions")
        if points is None and isinstance(meta.get("initial_positions"), dict): points = list(meta["initial_positions"].values())
        if points is not None and len(points) == 3 and np.asarray(points).shape == (3, 3):
            layouts.append({"xy": np.asarray(points)[:, :2].tolist()})
    for path in (RUN / "development").glob("*/pool_completed.json"):
        layouts.extend(json.loads(path.read_text())["valid_layouts"])
    for path in (RUN / "preflight").glob("*.json"):
        value = json.loads(path.read_text())
        if path.name == "result.json": layouts.extend(r["example"]["layout"] for r in value.values())
        elif path.name == "support_and_negatives.json":
            layouts.extend(e["layout"] for r in value.values() if isinstance(r, dict) for e in r.get("examples", []))
    for layout in layouts:
        layout = layout | {"geometry_id": geometry_id(layout["xy"])}
        reserve(layout, "development_legacy")
    (RUN / "registry_seeded.json").write_text(json.dumps({"excluded_layout_rows": len(layouts)}))


def worker(kind, category, count, seed, gpu, subtype=None):
    from .validity import make_env, Validator
    from .rollout import episode
    from ..vision import observation_images
    from PIL import Image
    protocol = json.loads((RUN / "final_protocol.json").read_text()); verify(protocol)
    label = kind+"_"+(subtype or category)
    root = RUN / ("final_sealed" if kind in ("broad_iid", "spatial_ood", "relation_ood") else "data_candidates") / label
    root.mkdir(parents=True, exist_ok=True)
    marker = root / "completed.json"
    if marker.exists(): return
    stream = ConditionalProposals(seed, category, kind, subtype)
    env = make_env(gpu); validator = Validator(env)
    accepted, rejects, valid_candidates, attempted_episodes = [], Counter(), 0, 0
    try:
        with (root / "proposal_log.jsonl").open("w") as log:
            for _ in range(50000):
                layout, reason = stream.next()
                validity = None
                if reason is None:
                    validity = validator.check(layout)
                    if not validity["valid"]: reason = validity["reason"]
                if reason is None and not reserve(layout, kind): reason = "geometry_or_coarse_group_already_reserved"
                if reason:
                    rejects[reason] += 1
                    log.write(json.dumps({"index": stream.index-1, "reason": reason, "geometry_id": layout["geometry_id"]})+"\n"); continue
                valid_candidates += 1
                directory = root / layout["geometry_id"]
                directory.mkdir(exist_ok=True)
                (directory / "layout.json").write_text(json.dumps(layout, indent=2))
                (directory / "validity.json").write_text(json.dumps(validity, indent=2))
                Image.fromarray(np.concatenate(list(observation_images(env._get_observations())), axis=1)).save(directory / "initial_two_views.png")
                episodes = []
                qualified = True
                if kind in ("train", "val"):
                    actor = protocol["training_expert_by_category"][category]
                    for permutation in range(3):
                        colored = recolor(layout, permutation)
                        for color in COLORS:
                            meta = episode(env, colored, color, directory / f"p{permutation}_{color}", actor, save_data=True)
                            episodes.append(meta); attempted_episodes += 1
                    qualified = all(r["eligible"] for r in episodes)
                row = {"geometry_id": layout["geometry_id"], "position_group_id": group_id(layout["xy"]), "layout": layout,
                       "directory": str(directory.relative_to(RUN)), "eligible_for_training": qualified if kind in ("train", "val") else None,
                       "episodes": [r["directory"] for r in episodes]}
                (directory / "admission.json").write_text(json.dumps(row, indent=2))
                log.write(json.dumps({"index": stream.index-1, "valid": True, "training_qualified": qualified if episodes else None, "geometry_id": layout["geometry_id"]})+"\n")
                if qualified: accepted.append(row)
                else: rejects["expert_group_failed_despite_valid_layout"] += 1
                print("GENERATED", label, len(accepted), count, valid_candidates, stream.index, flush=True)
                if len(accepted) == count: break
            assert len(accepted) == count, (label, len(accepted), count, dict(rejects))
        marker.write_text(json.dumps({"kind": kind, "category": category, "subtype": subtype, "accepted": accepted,
                                      "proposal_count": stream.index, "valid_candidates": valid_candidates, "executed_expert_episodes": attempted_episodes,
                                      "rejections": dict(rejects), "protocol_sha256": sha(RUN / "final_protocol.json")}, indent=2))
    finally: env.close()


def jobs(protocol):
    result = []
    for split, seed in (("train", 5100), ("val", 5200)):
        for i, category in enumerate(CATEGORIES): result.append((split, category, protocol[split+"_accepted_successful_geometries"][category], seed+i, None))
    for suite, seed in (("broad_iid", 6100), ("spatial_ood", 6200)):
        for i, (category, count) in enumerate(quotas(256, protocol["mixture_weights"]).items()): result.append((suite, category, count, seed+i, None))
    result.extend(("relation_ood", "uniform", 128, seed, subtype) for seed, subtype in ((6300, "near_gap"), (6400, "aligned")))
    return result


def assemble(protocol):
    training, final = [], []
    statistics = []
    for kind, category, _, _, subtype in jobs(protocol):
        label = kind+"_"+(subtype or category)
        path = RUN / ("final_sealed" if kind not in ("train", "val") else "data_candidates") / label / "completed.json"
        value = json.loads(path.read_text()); statistics.append({k: v for k, v in value.items() if k != "accepted"})
        for row in value["accepted"]:
            record = row | {"split": kind}
            (training if kind in ("train", "val") else final).append(record)
    all_rows = training+final
    assert len({r["geometry_id"] for r in all_rows}) == len(all_rows)
    assert len({r["position_group_id"] for r in all_rows}) == len(all_rows)
    assert len(training) == 240 and len(final) == 768
    for row in final:
        assert not row["episodes"], "No final policy/expert inference permitted"
    train_manifest = {"independent_train_geometries": 192, "independent_val_geometries": 48, "real_selected_demonstrations": 2160,
                      "primary_group": "geometry_id", "coarse_group_also_isolated": True, "geometry_labels_are_training_only": True,
                      "final_directories_forbidden_to_training": True, "layouts": training, "protocol_sha256": sha(RUN / "final_protocol.json")}
    (RUN / "training_manifest.json").write_text(json.dumps(train_manifest, indent=2))
    final_manifest = {"independent_final_geometries": 768, "counts": {k: sum(r["split"] == k for r in final) for k in ("broad_iid", "spatial_ood", "relation_ood")},
                      "policy_evaluation_performed": False, "used_for_model_selection": False, "layouts": final,
                      "protocol_sha256": sha(RUN / "final_protocol.json"),
                      "future_policy_attempts_per_model": {"broad_iid": 2304, "spatial_ood": 768, "relation_ood": 2304}}
    (RUN / "final_sealed/manifest.json").write_text(json.dumps(final_manifest, indent=2))
    (RUN / "generation_statistics.json").write_text(json.dumps(statistics, indent=2))
    seal = {str(p.relative_to(RUN / "final_sealed")): sha(p) for p in sorted((RUN / "final_sealed").rglob("*")) if p.is_file()}
    (RUN / "final_sealed_sha256.json").write_text(json.dumps(seal, indent=2))
    (RUN / "generation_completed.json").write_text(json.dumps({"train": 192, "val": 48, "final": 768, "selected_expert_episodes": 2160,
                                                               "final_policy_attempts": 0, "ACT_updates": 0}, indent=2))


def launch():
    protocol = freeze()
    if not (RUN / "registry_seeded.json").exists(): seed_registry()
    handles = []; active = []
    # Four process workers share only the explicitly allowed two physical GPUs.
    for i, (kind, category, count, seed, subtype) in enumerate(jobs(protocol)):
        gpu = (1, 2)[i%2]
        if len(active) == 4:
            process = active.pop(0); assert process.wait() == 0
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), MUJOCO_EGL_DEVICE_ID=str(gpu), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        label = kind+"_"+(subtype or category)
        log = (RUN / ("generation_"+label+".log")).open("a"); handles.append(log)
        args = [sys.executable, "-u", "-m", "colorlift.broad_v2.generate", "--kind", kind, "--category", category,
                "--count", str(count), "--seed", str(seed), "--gpu", str(gpu)]
        if subtype: args += ["--subtype", subtype]
        active.append(subprocess.Popen(args, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT))
    codes = [p.wait() for p in active]
    for h in handles: h.close()
    assert all(code == 0 for code in codes), codes
    assemble(protocol)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--launch", action="store_true"); p.add_argument("--detach", action="store_true")
    p.add_argument("--kind"); p.add_argument("--category"); p.add_argument("--count", type=int); p.add_argument("--seed", type=int); p.add_argument("--gpu", type=int); p.add_argument("--subtype")
    args = p.parse_args()
    if args.detach:
        freeze()
        with (RUN / "generation.log").open("a") as log:
            child = subprocess.Popen([sys.executable, "-u", "-m", "colorlift.broad_v2.generate", "--launch"], cwd=ROOT,
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({"pid": child.pid, "physical_GPUs": [1, 2]}))
    elif args.launch: launch()
    else: worker(args.kind, args.category, args.count, args.seed, args.gpu, args.subtype)
