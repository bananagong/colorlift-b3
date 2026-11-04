"""Independent development-only proposal, paired-expert and mixture comparison."""
import argparse
from collections import Counter
import json
import os
import subprocess
import sys
import numpy as np
from scipy.stats import qmc
from .spec import ROOT, RUN, CATEGORIES, DEV_PER_CATEGORY, EXPERT_DEV_PER_CATEGORY, MIXES, development_protocol, sha, write_once
from .layouts import Proposals


def generate(category, gpu, count=DEV_PER_CATEGORY):
    from .validity import make_env, Validator
    from ..vision import observation_images
    from PIL import Image
    out = RUN / "development" / category
    out.mkdir(parents=True, exist_ok=True)
    if (out / "pool_completed.json").exists(): return
    stream = Proposals(3100+CATEGORIES.index(category), category)
    env = make_env(gpu); validator = Validator(env)
    accepted, seen, rejects = [], set(), Counter()
    try:
        with (out / "proposal_log.jsonl").open("w") as log:
            for _ in range(50000):
                layout, reason = stream.next()
                result = None
                if reason is None:
                    result = validator.check(layout)
                    if not result["valid"]: reason = result["reason"]
                if layout["geometry_id"] in seen: reason = "duplicate_geometry"
                log.write(json.dumps({"index": layout["proposal_index"], "geometry_id": layout["geometry_id"], "reason": reason,
                                      "validation": result if result is not None and not result["valid"] else None})+"\n")
                if reason: rejects[reason] += 1; continue
                seen.add(layout["geometry_id"]); accepted.append(layout)
                directory = out / "pool" / layout["geometry_id"]
                directory.mkdir(parents=True, exist_ok=True)
                (directory / "layout.json").write_text(json.dumps(layout, indent=2))
                (directory / "validity.json").write_text(json.dumps(result, indent=2))
                Image.fromarray(np.concatenate(list(observation_images(env._get_observations())), axis=1)).save(directory / "initial_two_views.png")
                if len(accepted)%16 == 0: print("VALID_POOL", category, len(accepted), stream.index, flush=True)
                if len(accepted) == count: break
            assert len(accepted) == count
        (out / "pool_completed.json").write_text(json.dumps({"category": category, "valid_layouts": accepted, "proposal_count": stream.index,
                                                            "accepted": count, "rejections": dict(rejects)}, indent=2))
    finally: env.close()


def paired(category, gpu):
    from .validity import make_env
    from .rollout import episode
    from ..spatial_rollout import load_policy
    import torch
    torch.set_num_threads(1)
    out = RUN / "development" / category
    if (out / "episodes_completed.json").exists(): return
    protocol = json.loads((RUN / "development_protocol.json").read_text())
    checkpoint_path = ROOT / protocol["baseline_checkpoint"]
    assert sha(checkpoint_path) == protocol["baseline_sha256"]
    model, checkpoint = load_policy(checkpoint_path)
    layouts = json.loads((out / "pool_completed.json").read_text())["valid_layouts"][:EXPERT_DEV_PER_CATEGORY]
    env = make_env(gpu); results = []
    try:
        for layout in layouts:
            for actor in ("act", "current_expert", "safe_expert"):
                for color in ("red", "yellow", "blue"):
                    directory = out / "episodes" / layout["geometry_id"] / actor / color
                    row = episode(env, layout, color, directory, actor, model, checkpoint)
                    results.append(row)
                    print("PAIRED", category, layout["proposal_index"], actor, color, row["success"], row["eligible"], row["control_steps"], flush=True)
        assert sha(checkpoint_path) == protocol["baseline_sha256"]
        (out / "episodes_completed.json").write_text(json.dumps({"results": results, "independent_layouts": len(layouts)}, indent=2))
    finally: env.close()


def coverage(layouts):
    xy = np.array([r["xy"] for r in layouts])
    cells = np.floor((xy.reshape(-1, 2)+.145)/.290*10).clip(0, 9).astype(int)
    occupancy = len(set(map(tuple, cells)))/100
    distance = [r["features"]["minimum_center_distance_m"] for r in layouts]
    area = [r["features"]["area_ratio"] for r in layouts]
    histogram, _, _ = np.histogram2d(distance, area, bins=([.08, .10, .12, .15, .20, .40], [.08, .15, .30, .5, .7, 1.]))
    p = histogram.ravel(); p = p[p > 0]/max(1, p.sum())
    entropy = float(-np.sum(p*np.log(p))/np.log(histogram.size))
    discrepancy = float(qmc.discrepancy((xy.reshape(-1, 6)+.145)/.290))
    return {"XY_grid_occupancy": occupancy, "relation_histogram_entropy": entropy, "six_dimensional_centered_discrepancy": discrepancy}


def summarize():
    pools, summaries, actor_results = {}, {}, {}
    for category in CATEGORIES:
        out = RUN / "development" / category
        pool = json.loads((out / "pool_completed.json").read_text())
        rows = json.loads((out / "episodes_completed.json").read_text())["results"]
        pools[category] = pool["valid_layouts"]
        summaries[category] = {"accepted": pool["accepted"], "proposals": pool["proposal_count"], "valid_acceptance": pool["accepted"]/pool["proposal_count"], "rejections": pool["rejections"]}
        for actor in ("act", "current_expert", "safe_expert"):
            selected = [r for r in rows if r["actor"] == actor]
            actor_results[category+"/"+actor] = {"episodes": len(selected), "independent_geometries": len({r["geometry_id"] for r in selected}),
                "successes": sum(r["success"] for r in selected), "contact_qualified_successes": sum(r["eligible"] for r in selected),
                "wrong_colors": sum(r["wrong_color_grasp"] for r in selected), "collision_episodes": sum(r["collision_steps"] > 0 for r in selected),
                "mean_success_steps": float(np.mean([r["control_steps"] for r in selected if r["success"]])) if any(r["success"] for r in selected) else None,
                "mean_close_XY_error_m": float(np.mean([r["first_close_target_XY_error_m"] for r in selected if r["first_close_target_XY_error_m"] is not None])),
                "failure_categories": dict(Counter(r["failure_category"] for r in selected if not r["success"]))}
    all_ids = [r["geometry_id"] for rows in pools.values() for r in rows]
    assert len(set(all_ids)) == len(all_ids)
    candidates = {}
    for name, weights in MIXES.items():
        trials = []
        quota = np.floor(np.array(weights)*128).astype(int)
        for index in np.argsort(-(np.array(weights)*128-quota))[:128-quota.sum()]: quota[index] += 1
        for seed in range(100):
            rng = np.random.default_rng(4100+seed)
            layouts = [r for category, n in zip(CATEGORIES, quota) for r in [pools[category][i] for i in rng.choice(DEV_PER_CATEGORY, int(n), replace=False)]]
            trials.append(coverage(layouts))
        average = {key: float(np.mean([r[key] for r in trials])) for key in trials[0]}
        qualified = sum(w*actor_results[c+"/safe_expert"]["contact_qualified_successes"]/actor_results[c+"/safe_expert"]["episodes"] for w, c in zip(weights, CATEGORIES))
        acceptance = 1/sum(w/summaries[c]["valid_acceptance"] for w, c in zip(weights, CATEGORIES))
        score = .45*average["XY_grid_occupancy"]+.20*average["relation_histogram_entropy"]+.25*qualified+.10*acceptance
        candidates[name] = {"weights": weights, "quota_128": quota.tolist(), **average,
                            "expected_safe_contact_success": qualified, "harmonic_proposal_acceptance": acceptance, "score": score}
    name = max(candidates, key=lambda k: (candidates[k]["score"], k))
    result = {"proposal_statistics": summaries, "paired_actor_results": actor_results, "mixture_candidates": candidates,
              "selected_mixture": name, "selected_weights": candidates[name]["weights"],
              "ACT_architecture_or_weights_changed": False, "selection_is_development_data_quality_proxy_not_ACT_optimum": True,
              "held_out_final_used": False}
    write_once(RUN / "development_results.json", result)
    print(json.dumps(result, indent=2), flush=True)


def launch():
    development_protocol()
    handles = []; workers = []
    for i, category in enumerate(CATEGORIES):
        gpu = (1, 2)[i%2]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), MUJOCO_EGL_DEVICE_ID=str(gpu), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        log = (RUN / f"dev_{category}.log").open("a"); handles.append(log)
        workers.append(subprocess.Popen([sys.executable, "-u", "-m", "colorlift.broad_v2.develop", "--category", category, "--gpu", str(gpu)], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT))
    codes = [w.wait() for w in workers]
    for h in handles: h.close()
    assert codes == [0]*len(workers), codes
    summarize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--launch", action="store_true"); parser.add_argument("--detach", action="store_true"); parser.add_argument("--category", choices=CATEGORIES); parser.add_argument("--gpu", type=int, default=1)
    args = parser.parse_args()
    if args.detach:
        development_protocol()
        with (RUN / "development.log").open("a") as log:
            child = subprocess.Popen([sys.executable, "-u", "-m", "colorlift.broad_v2.develop", "--launch"], cwd=ROOT,
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({"pid": child.pid, "physical_GPUs": [1, 2], "valid_layouts": 256, "paired_episodes": 576}))
    elif args.launch: launch()
    else: generate(args.category, args.gpu); paired(args.category, args.gpu)
