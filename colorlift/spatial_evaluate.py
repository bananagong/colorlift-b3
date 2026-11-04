"""Paired held-out spatial/OOD and physical recoloring tests of frozen checkpoints."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import numpy as np
from .spatial_gate import ROOT, RUN, CHECKPOINT

WORKERS = 8
MODELS = ("baseline", "curriculum_seed43", "shuffled_seed43", "curriculum_seed44", "shuffled_seed44")
LEVELS = ("easy", "medium", "hard", "ood_position", "ood_light", "ood_camera")


def checkpoint_paths():
    return {name: CHECKPOINT if name == "baseline" else RUN / "models" / name / "policy.pt" for name in MODELS}


def prepare_scenarios():
    from .spatial_env import make_spatial_env
    from .spatial_layouts import descriptor, counterfactual_descriptor
    from .recovery_expert import DampedIK
    path = RUN / "test_scenarios.json"
    if path.exists(): return
    manifest = json.loads((RUN / "data/manifest.json").read_text())
    training_ids = {r["geometry_id"] for r in manifest["layouts"]}
    env = make_spatial_env(1); ik = DampedIK(env)
    layouts = []; cf = []; rejected = []
    try:
        for index, level in enumerate(LEVELS):
            count = 30 if index < 3 else 20
            for i in range(count):
                layout = descriptor(2200000+index*10000+i, level)
                env.reset_layout(layout, "red")
                errors = [ik.solve(np.r_[xy, env.config.grasp_height])[1]["ik_position_residual_m"] for xy in layout["xy"]]
                assert max(errors) < .001, (level, i, errors)
                layouts.append(layout)
        for index, level in enumerate(LEVELS[:3]):
            for i in range(4):
                layout = counterfactual_descriptor(2300000+index*10000+i, level, 0)
                env.reset_layout(layout, "red")
                errors = [ik.solve(np.r_[xy, env.config.grasp_height])[1]["ik_position_residual_m"] for xy in layout["xy"]]
                assert max(errors) < .001
                cf.append(layout)
    finally:
        env.close()
    all_ids = {d["geometry_id"] for d in layouts+cf}
    assert len(all_ids) == len(layouts)+len(cf) and not (all_ids & training_ids)
    payload = {"layouts": layouts, "counterfactual_base_layouts": cf, "independent_geometry_count": len(all_ids),
               "test_commands_per_model": len(layouts)*3, "counterfactual_commands_per_model": len(cf)*6*3,
               "physical_reachability_checked_before_policy_results": True, "disjoint_train_val_test_geometries": True}
    path.write_text(json.dumps(payload, indent=2))
    print("SCENARIOS", len(layouts), len(cf), flush=True)


def worker(gpu, shard, validation=False):
    import torch
    from .config import COLORS
    from .spatial_layouts import counterfactual_descriptor
    from .spatial_env import make_spatial_env
    from .spatial_rollout import load_policy, episode
    torch.set_num_threads(1)
    env = make_spatial_env(gpu)
    if validation:
        manifest = json.loads((RUN / "data/manifest.json").read_text())
        selected = []
        for level in LEVELS[:3]:
            rows = [r for r in manifest["layouts"] if r["split"] == "val" and r["level"] == level][:2]
            for row in rows:
                meta = json.loads((RUN / row["episodes"][0] / "metadata.json").read_text())
                selected.append(meta["layout"])
        jobs = [(name, step) for name in MODELS[1:] for step in (1500, 3000, 5000)]
        try:
            for index, (name, step) in enumerate(jobs):
                if index%WORKERS != shard: continue
                model, checkpoint = load_policy(RUN / "models" / name / f"stage_{step}.pt")
                for layout in selected:
                    for color in COLORS:
                        directory = RUN / "evaluations/validation" / name / f"step_{step}" / layout["level"] / f'layout_{layout["seed"]}' / color
                        row = episode(env, layout, color, directory, name, model, checkpoint)
                        print("STAGE_VALIDATION", name, step, layout["level"], row["success"], flush=True)
                del model
        finally:
            env.close()
        return
    freeze = json.loads((RUN / "evaluation_freeze.json").read_text())
    paths = checkpoint_paths()
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == freeze["models"][name] for name, path in paths.items())
    policies = {name: load_policy(path) for name, path in paths.items()}
    scenarios = json.loads((RUN / "test_scenarios.json").read_text())
    try:
        for index, layout in enumerate(scenarios["layouts"]):
            if index%WORKERS != shard: continue
            for name, (model, checkpoint) in policies.items():
                for color in COLORS:
                    directory = RUN / "evaluations/test" / name / layout["level"] / f'layout_{layout["seed"]}' / color
                    row = episode(env, layout, color, directory, name, model, checkpoint)
                    print("SPATIAL_TEST", name, layout["level"], layout["seed"], color, row["success"], row["eligible"], row["control_steps"], flush=True)
        for index, base in enumerate(scenarios["counterfactual_base_layouts"]):
            if index%WORKERS != shard: continue
            for permutation in range(6):
                layout = counterfactual_descriptor(base["seed"], base["level"], permutation)
                for name, (model, checkpoint) in policies.items():
                    for color in COLORS:
                        directory = RUN / "evaluations/counterfactual" / name / layout["level"] / f'layout_{layout["seed"]}' / f"p{permutation}" / color
                        row = episode(env, layout, color, directory, name, model, checkpoint)
                        print("COUNTERFACTUAL", name, layout["seed"], permutation, color, row["success"], row["first_goal_nearest_slot"], row["target_physical_slot"], flush=True)
    finally:
        env.close()
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == freeze["models"][name] for name, path in paths.items())
    (RUN / f"evaluation_shard_{shard}_completed.json").write_text(json.dumps({"complete": True}))


def launch(detach, validation=False):
    if detach:
        with (RUN / ("validation_evaluation.log" if validation else "evaluation.log")).open("a") as log:
            command = [sys.executable, "-u", "-m", "colorlift.spatial_evaluate", "--launch"]
            if validation: command += ["--validation"]
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({"pid": child.pid, "GPUs": [1, 2]})); return
    if not validation:
        prepare_scenarios()
        freeze = {"models": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in checkpoint_paths().items()},
                  "scenarios": hashlib.sha256((RUN / "test_scenarios.json").read_bytes()).hexdigest(),
                  "controller": hashlib.sha256((ROOT / "colorlift/event_control.py").read_bytes()).hexdigest(),
                  "rollout": hashlib.sha256((ROOT / "colorlift/spatial_rollout.py").read_bytes()).hexdigest()}
        freeze_path = RUN / "evaluation_freeze.json"
        if freeze_path.exists(): assert json.loads(freeze_path.read_text()) == freeze
        else: freeze_path.write_text(json.dumps(freeze, indent=2))
    workers, handles = [], []
    for shard in range(WORKERS):
        gpu = (1, 2)[shard%2]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), MUJOCO_EGL_DEVICE_ID=str(gpu), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        handle = (RUN / f'{"validation" if validation else "evaluation"}_{shard}.log').open("a"); handles.append(handle)
        command = [sys.executable, "-u", "-m", "colorlift.spatial_evaluate", "--gpu", str(gpu), "--shard", str(shard)]
        if validation: command += ["--validation"]
        workers.append(subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT))
    codes = [worker.wait() for worker in workers]
    for handle in handles: handle.close()
    assert codes == [0]*WORKERS, codes
    (RUN / ("validation_evaluation_completed.json" if validation else "evaluation_completed.json")).write_text(json.dumps({"complete": True}))


def main():
    p = argparse.ArgumentParser(); p.add_argument("--launch", action="store_true"); p.add_argument("--detach", action="store_true")
    p.add_argument("--gpu", type=int); p.add_argument("--shard", type=int); p.add_argument("--prepare", action="store_true"); p.add_argument("--validation", action="store_true")
    args = p.parse_args()
    if args.prepare: prepare_scenarios()
    elif args.launch: launch(args.detach, args.validation)
    else: worker(args.gpu, args.shard, args.validation)


if __name__ == "__main__": main()
