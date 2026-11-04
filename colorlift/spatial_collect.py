"""Collect new physical demonstrations, admitting only fully color-balanced geometries."""
import argparse
from collections import Counter
import hashlib
import json
import os
import shutil
import subprocess
import sys
import numpy as np
from .spatial_gate import ROOT, RUN, fingerprints

LEVELS = ("easy", "medium", "hard")
WORKERS = 8


def require_gate():
    assert (RUN / "gate/completed.json").exists(), "Complete all 3000 gate trials before collecting training data"
    gate = json.loads((RUN / "gate/metrics.json").read_text())
    assert gate["gate_passed"] and gate["total_trials"] == 3000, "Do not train or collect after a failed gate"
    assert json.loads((RUN / "gate/freeze.json").read_text()) == fingerprints()


def jobs():
    result = []
    for split, base, count in (("train", 2000000, 80), ("val", 2100000, 15)):
        for index, level in enumerate(LEVELS):
            for offset in range(count):
                result.append((split, level, base+index*10000+offset))
    return result


def worker(gpu, shard):
    require_gate()
    from .config import COLORS
    from .spatial_env import make_spatial_env
    from .spatial_layouts import descriptor
    from .spatial_rollout import episode
    from .recovery_expert import DampedIK
    env = make_spatial_env(gpu)
    ik = DampedIK(env)
    try:
        for index, (split, level, seed) in enumerate(jobs()):
            if index%WORKERS != shard:
                continue
            directory = RUN / "candidates" / f"{split}_{level}_{seed}"
            directory.mkdir(parents=True, exist_ok=True)
            summary = directory / "layout.json"
            if summary.exists():
                continue
            base = descriptor(seed, level)
            env.reset_layout(base, "red")
            residuals = [ik.solve(np.r_[xy, env.config.grasp_height])[1]["ik_position_residual_m"] for xy in base["xy"]]
            if max(residuals) > .001:
                result = {"split": split, "level": level, "seed": seed, "geometry_id": base["geometry_id"],
                          "eligible": False, "reason": "IK_unreachable_before_rollout", "IK_residuals": residuals, "episodes": []}
            else:
                episodes = []
                for permutation in range(3):
                    layout = descriptor(seed, level, permutation)
                    for color in COLORS:
                        case = episode(env, layout, color, directory / f"p{permutation}_{color}", "expert", save_data=True)
                        episodes.append(case)
                        print("SPATIAL_EXPERT", split, level, seed, permutation, color, case["success"], case["eligible"], case["control_steps"], flush=True)
                # Keep every color equally represented at every slot in each admitted layout.
                complete = all(case["eligible"] for case in episodes)
                balance = Counter((case["target_physical_slot"], case["target_color"]) for case in episodes)
                assert set(balance.values()) == {1} and len(balance) == 9
                assert len({json.dumps(case["physical_slot_positions"]) for case in episodes}) == 1
                result = {"split": split, "level": level, "seed": seed, "geometry_id": base["geometry_id"],
                          "eligible": complete, "reason": "all9_actual_success_and_contact" if complete else "at_least_one_case_failed_contact_or_task",
                          "IK_residuals": residuals, "episodes": [case["directory"] for case in episodes]}
            summary.write_text(json.dumps(result, indent=2))
            print("LAYOUT", split, level, seed, result["eligible"], result["reason"], flush=True)
    finally:
        env.close()
    (RUN / f"collection_shard_{shard}_completed.json").write_text(json.dumps({"complete": True}))


def manifest():
    require_gate()
    candidates = [json.loads(path.read_text()) for path in sorted((RUN / "candidates").glob("*/layout.json"))]
    assert len(candidates) == len(jobs()), (len(candidates), len(jobs()))
    selected_layouts, selected_episodes = [], []
    counts = {}
    missing = []
    for split, quota in (("train", 60), ("val", 10)):
        for level in LEVELS:
            pool = [row for row in candidates if row["split"] == split and row["level"] == level and row["eligible"]]
            counts[split+"_"+level] = {"candidates": sum(r["split"] == split and r["level"] == level for r in candidates),
                                      "fully_eligible_layouts": len(pool), "quota": quota}
            if len(pool) < quota:
                missing.append((split, level, len(pool), quota))
            for row in sorted(pool, key=lambda r: r["seed"])[:quota]:
                selected_layouts.append(row)
                for directory in row["episodes"]:
                    meta = json.loads((RUN / directory / "metadata.json").read_text())
                    selected_episodes.append({"split": split, "level": level, "geometry_id": row["geometry_id"], "directory": directory,
                                              "target_color": meta["target_color"], "slot": meta["target_physical_slot"],
                                              "permutation": meta["layout"]["color_assignment_index"], "control_steps": meta["control_steps"], "eligible": True,
                                              "npz_sha256": hashlib.sha256((RUN / directory / "trajectory.npz").read_bytes()).hexdigest()})
    out = RUN / "data"; out.mkdir(exist_ok=True)
    metadata = {"candidate_unique_geometries": len(candidates), "selected_unique_geometries": len(selected_layouts),
                "selected_train_layouts": sum(r["split"] == "train" for r in selected_layouts),
                "selected_val_layouts": sum(r["split"] == "val" for r in selected_layouts),
                "selected_executed_demonstrations": len(selected_episodes), "counts": counts,
                "selection": "first seed-ordered fully qualified geometries; all9 physical rollouts retained together for exact slot/color balance",
                "disturbance_events_counted_as_layouts": False, "image_only_randomization": False,
                "teacher_uses_target_truth": True, "student_uses_target_truth": False,
                "layouts": selected_layouts, "episodes": selected_episodes, "quota_shortfall": missing}
    (out / "manifest.json").write_text(json.dumps(metadata, indent=2))
    assert not missing, missing
    assert len({r["geometry_id"] for r in selected_layouts}) == len(selected_layouts)
    assert not ({r["geometry_id"] for r in selected_layouts if r["split"] == "train"}
                & {r["geometry_id"] for r in selected_layouts if r["split"] == "val"})
    print(json.dumps({k: v for k, v in metadata.items() if k not in ("layouts", "episodes")}, indent=2), flush=True)


def launch(detach):
    require_gate()
    if detach:
        with (RUN / "collection.log").open("a") as log:
            child = subprocess.Popen([sys.executable, "-u", "-m", "colorlift.spatial_collect", "--launch"], cwd=ROOT,
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({"pid": child.pid, "GPUs": [1, 2], "candidate_geometries": len(jobs())})); return
    workers, handles = [], []
    for shard in range(WORKERS):
        gpu = (1, 2)[shard%2]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), MUJOCO_EGL_DEVICE_ID=str(gpu), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        handle = (RUN / f"collection_{shard}.log").open("a"); handles.append(handle)
        workers.append(subprocess.Popen([sys.executable, "-u", "-m", "colorlift.spatial_collect", "--gpu", str(gpu), "--shard", str(shard)],
                                        cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT))
    codes = [worker.wait() for worker in workers]
    for handle in handles: handle.close()
    assert codes == [0]*WORKERS, codes
    manifest()
    (RUN / "collection_completed.json").write_text(json.dumps({"complete": True, "candidate_unique_geometries": len(jobs())}))


def main():
    p = argparse.ArgumentParser(); p.add_argument("--launch", action="store_true"); p.add_argument("--detach", action="store_true")
    p.add_argument("--gpu", type=int); p.add_argument("--shard", type=int); p.add_argument("--manifest", action="store_true")
    args = p.parse_args()
    if args.launch: launch(args.detach)
    elif args.manifest: manifest()
    else: worker(args.gpu, args.shard)


if __name__ == "__main__": main()
