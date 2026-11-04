"""Predeclared 1000-layout/color qualification of the unchanged event-driven ACT."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "experiments/colorlift_spatial_curriculum_v1"
CHECKPOINT = ROOT / "experiments/colorlift_recovery_v1/model/policy.pt"
COUNT = 1000
BASE_SEED = 1000000
WORKERS = 8


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprints():
    files = ("colorlift/env.py", "colorlift/conditional.py", "colorlift/event_control.py", "colorlift/event_efficiency.py",
             "colorlift/primitives.py", "colorlift/step_recording.py", "colorlift/efficient_expert.py",
             "experiments/colorlift_recovery_v1/model/policy.pt")
    return {name: sha(ROOT / name) for name in files}


def wilson(k, n):
    z = 1.959963984540054
    p = k/n
    denominator = 1+z*z/n
    middle = (p+z*z/(2*n))/denominator
    radius = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denominator
    return [max(0., middle-radius), min(1., middle+radius)]


def protocol():
    return {"independent_layouts": COUNT, "commands_per_layout": 3, "total_trials": COUNT*3,
            "seed_range_inclusive": [BASE_SEED, BASE_SEED+COUNT-1], "scheme": "efficiency", "schedule": [4, 2, 1, 4],
            "checkpoint": str(CHECKPOINT), "checkpoint_sha256": sha(CHECKPOINT), "max_GPUs": 2, "physical_GPUs": [1, 2],
            "fixed_sample_count_no_optional_stopping": True, "no_external_disturbance": True,
            "distribution": "unchanged baseline ColorSampler: +/-0.11m, >=0.095m centers, yaw +/-0.2rad",
            "success": ">0.04m relative lift and original continuous stable2s scoring",
            "unintended_contact": "original 500Hz force-qualified collision categories; robot5N, object-object1N; normal target-pad grasp allowed",
            "contact_success_denominator": "all attempts, not only successful attempts",
            "gate": {"per_color_target_success_rate_min": .98, "per_color_contact_qualified_collision_free_success_min": .95,
                     "per_color_observed_wrong_color_count_max": 0},
            "uncertainty": "Wilson95 intervals per color; layout-cluster bootstrap pooled; intervals reported separately from requested empirical gates",
            "post_gate_training": "forbidden unless every color passes all three empirical thresholds"}


def worker(gpu, shard):
    import torch
    from .conditional import configuration
    from .config import COLORS
    from .env import make_env
    from . import event_efficiency as evaluation
    evaluation.RUN = RUN
    torch.set_num_threads(1)
    freeze = json.loads((RUN / "gate/freeze.json").read_text())
    assert freeze == fingerprints()
    model, checkpoint = evaluation.load_model()
    envs = {color: make_env(gpu, configuration(color)[0]) for color in COLORS}
    episodes = 0
    try:
        for seed in range(BASE_SEED+shard, BASE_SEED+COUNT, WORKERS):
            for color in COLORS:
                directory = RUN / "gate/episodes" / f"scene_{seed}" / color
                result = evaluation.run_episode(envs[color], seed, directory, "act", "efficiency", model, checkpoint)
                episodes += 1
                print("GATE", shard, seed, color, result["success"], result["contact_qualified_collision_free_success"],
                      result["wrong_color_grasp"], result["control_steps"], flush=True)
    finally:
        for env in envs.values():
            env.close()
    assert freeze == fingerprints()
    (RUN / f"gate/shard_{shard}_completed.json").write_text(json.dumps({"complete": True, "trials": episodes, "GPU": gpu}))


def summarize():
    from .config import COLORS
    out = RUN / "gate"
    episodes = [json.loads(p.read_text()) for p in sorted((out / "episodes").glob("scene_*/*/metadata.json"))]
    assert len(episodes) == COUNT*3, len(episodes)
    layout_groups = defaultdict(list)
    for row in episodes:
        layout_groups[row["seed"]].append(row)
    assert len(layout_groups) == COUNT
    geometry_ids = set()
    for seed, rows in layout_groups.items():
        assert len(rows) == 3 and set(r["target_color"] for r in rows) == set(COLORS)
        assert len({r["initial_rgb_sha256"] for r in rows}) == 1
        assert all(r["initial_positions"] == rows[0]["initial_positions"] for r in rows)
        points = sorted(tuple(round(v, 10) for v in p) for p in rows[0]["initial_positions"].values())
        geometry_ids.add(hashlib.sha256(json.dumps(points).encode()).hexdigest())
    assert len(geometry_ids) == COUNT, "A repeated geometry cannot count as an independent layout"
    comparison = []
    for color in COLORS:
        selected = [row for row in episodes if row["target_color"] == color]
        n = len(selected)
        succeeded = [row for row in selected if row["success"]]
        k = len(succeeded)
        contact = sum(row["contact_qualified_collision_free_success"] for row in selected)
        wrong = sum(row["wrong_color_grasp"] for row in selected)
        row = {"color": color, "trials": n, "successes": k, "target_success_rate": k/n,
               "success_wilson95": wilson(k, n), "contact_qualified_successes": contact, "contact_success_rate": contact/n,
               "contact_wilson95": wilson(contact, n), "wrong_color_count": wrong, "observed_wrong_color_rate": wrong/n,
               "wrong_color_wilson95": wilson(wrong, n),
               "collision_episodes": sum(r["collision_steps"] > 0 for r in selected),
               "mean_steps_success": float(np.mean([r["control_steps"] for r in succeeded])),
               "median_steps_success": float(np.median([r["control_steps"] for r in succeeded])),
               "p90_steps_success": float(np.quantile([r["control_steps"] for r in succeeded], .9)),
               "mean_steps_all_attempts": float(np.mean([r["control_steps"] for r in selected])),
               "passes": k/n >= .98 and contact/n >= .95 and wrong == 0}
        comparison.append(row)
    scene_values = np.array([[sum(r["success"] for r in layout_groups[s])/3,
                              sum(r["contact_qualified_collision_free_success"] for r in layout_groups[s])/3]
                             for s in sorted(layout_groups)])
    rng = np.random.default_rng(20261030)
    boot = np.array([scene_values[rng.integers(COUNT, size=COUNT)].mean(0) for _ in range(5000)])
    metrics = {"gate_passed": all(r["passes"] for r in comparison), "independent_layouts": COUNT,
               "unique_geometry_hashes": len(geometry_ids), "total_trials": len(episodes), "by_color": comparison,
               "pooled_target_success_rate": float(scene_values[:, 0].mean()),
               "pooled_contact_success_rate": float(scene_values[:, 1].mean()),
               "pooled_layout_cluster95": np.quantile(boot, [.025, .975], axis=0).T.tolist(),
               "wrong_color_count": sum(r["wrong_color_grasp"] for r in episodes),
               "observed_zero_is_not_proof_of_population_zero": True, "training_authorized_by_gate": all(r["passes"] for r in comparison)}
    assert json.loads((out / "freeze.json").read_text()) == fingerprints()
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    with (out / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison[0]))
        writer.writeheader(); writer.writerows(comparison)
    fields = ("seed", "target_color", "success", "contact_qualified_collision_free_success", "collision_steps", "wrong_color_grasp",
              "control_steps", "steps_before_final_hold", "physical_eef_path_m", "motion_reversals_within_phase", "no_progress_steps", "outcome", "directory")
    with (out / "episodes.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows({key: row.get(key) for key in fields} for row in episodes)
    print(json.dumps(metrics, indent=2), flush=True)


def launch(detach=False):
    out = RUN / "gate"
    out.mkdir(parents=True, exist_ok=True)
    if detach:
        with (out / "launch.log").open("a") as log:
            child = subprocess.Popen([sys.executable, "-u", "-m", "colorlift.spatial_gate", "--launch"], cwd=ROOT,
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({"pid": child.pid, "layouts": COUNT, "trials": COUNT*3, "GPUs": [1, 2]}))
        return
    assert not (out / "metrics.json").exists(), "Preserve the complete preregistered gate"
    freeze = fingerprints()
    protocol_path = out / "protocol.json"
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text()) == protocol()
        assert json.loads((out / "freeze.json").read_text()) == freeze
    else:
        protocol_path.write_text(json.dumps(protocol(), indent=2))
        (out / "freeze.json").write_text(json.dumps(freeze, indent=2))
        shutil.copy2(CHECKPOINT, out / "frozen_policy.pt")
    workers, handles = [], []
    for shard in range(WORKERS):
        gpu = (1, 2)[shard % 2]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), MUJOCO_EGL_DEVICE_ID=str(gpu), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        handle = (out / f"worker_{shard}.log").open("a"); handles.append(handle)
        workers.append(subprocess.Popen([sys.executable, "-u", "-m", "colorlift.spatial_gate", "--gpu", str(gpu), "--shard", str(shard)],
                                        cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT))
    codes = [p.wait() for p in workers]
    for handle in handles:
        handle.close()
    assert codes == [0]*WORKERS, codes
    assert freeze == fingerprints()
    summarize()
    (out / "completed.json").write_text(json.dumps({"complete": True, "trials": COUNT*3, "model_and_controller_frozen": True}))


def main():
    p = argparse.ArgumentParser(); p.add_argument("--launch", action="store_true"); p.add_argument("--detach", action="store_true")
    p.add_argument("--gpu", type=int); p.add_argument("--shard", type=int); p.add_argument("--summarize", action="store_true")
    args = p.parse_args()
    if args.launch: launch(args.detach)
    elif args.summarize: summarize()
    else: worker(args.gpu, args.shard)


if __name__ == "__main__": main()
