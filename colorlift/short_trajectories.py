"""Audit first-70-step composition and collect shorter successful visual expert demonstrations."""
import argparse
from dataclasses import asdict
import csv
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import cv2
import imageio.v2 as imageio
from .config import Config, ROOT
from .env import make_env
from .collect import rollout as baseline_rollout
from .efficient_expert import ExpertConfig, task_config, rollout as efficient_rollout
from .step_recording import StepRecorder, save_rows, first_steps_breakdown
from .vision import mask_preview

SOURCE = ROOT / "experiments" / "colorlift_v1"
OUTPUT = ROOT / "experiments" / "colorlift_short_expert_v1"


def analyze_existing():
    out = OUTPUT / "existing_first70"
    out.mkdir(parents=True, exist_ok=True)
    aggregate, details = [], []
    for variant in ("hsv", "rgb", "mask", "rgb_mask"):
        metrics = json.loads((SOURCE / variant / "eval_test" / "metrics.json").read_text())
        for trial in metrics["trials"]:
            path = SOURCE / variant / "eval_test" / trial["step_trace"]
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
            breakdown = first_steps_breakdown(rows)
            detail = {"variant": variant, "seed": trial["seed"], "success": trial["success"],
                      "total_steps": trial["control_steps"], "source": str(path), **breakdown}
            details.append(detail)
            aggregate.append({"variant": variant, "seed": trial["seed"], "success": trial["success"],
                              "total_steps": trial["control_steps"], "observed_first70_steps": breakdown["observed_steps"],
                              **breakdown["composition"], "post_close_idle_steps": breakdown["post_close_idle_steps"],
                              "no_progress_steps": breakdown["no_progress_steps"], "hold_streak_at_step70": breakdown["hold_streak_at_end"]})
    save_rows(out / "episodes.csv", aggregate)
    (out / "all_breakdowns.json").write_text(json.dumps(details, indent=2))


def write_case(env, output, kind, seed, expert, *, save_dataset=False, learned=None):
    output.mkdir(parents=True, exist_ok=True)
    with StepRecorder(env, capture_images=True) as recorder:
        if kind == "baseline":
            result, history, diagnostic = baseline_rollout(env, seed, record=save_dataset)
        elif kind == "efficient":
            result, history, diagnostic = efficient_rollout(env, seed, expert, record=save_dataset)
        else:
            from .evaluate import learned_rollout
            model, checkpoint = learned
            result, diagnostic = learned_rollout(env, model, checkpoint, seed, output, 0, "mask")
            history = {}
    if "localization" in result:
        estimate = np.r_[np.mean([item["xy"] for item in result["localization"]], axis=0), env.config.table_height + env.config.cube_size / 2]
    else:
        estimate = None
    rows = recorder.enrich(diagnostic, seed=seed, visual_xyz=estimate)
    schema = recorder.schema()
    result.update(controller=kind, task_config=asdict(env.config), trace_schema=schema,
                  first70=first_steps_breakdown(rows), full_trace="steps.csv", first70_trace="first70_steps.csv")
    save_rows(output / "steps.csv", rows)
    save_rows(output / "first70_steps.csv", rows[:70])
    (output / "metadata.json").write_text(json.dumps(result, indent=2))
    (output / "first70_breakdown.json").write_text(json.dumps(result["first70"], indent=2))
    (output / "schema.json").write_text(json.dumps(schema, indent=2))
    frames = [(recorder.initial_images, recorder.initial_masks, None)]
    frames.extend((record["images"], record["masks"], row) for record, row in zip(recorder.records, rows))
    with imageio.get_writer(str(output / "trajectory.mp4"), fps=env.config.hz, codec="libx264", quality=7) as writer:
        for rgb, masks, row in frames:
            frame = np.concatenate([np.concatenate(list(rgb), axis=1),
                                    np.concatenate([mask_preview(mask) for mask in masks], axis=1)], axis=0)
            if row:
                text = f'{kind} step={row["step"]} {row["phase_name"]} hold={row["hold_streak"]}/40'
                cv2.putText(frame, text, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, .32, (30, 30, 30), 1)
            writer.append_data(frame)
    if save_dataset:
        history["osc_actions"] = np.stack([record["action"] for record in recorder.records]).astype(np.float32)
        history["eef_before"] = np.stack([record["before"]["eef"] for record in recorder.records])
        history["eef_after"] = np.stack([record["after"]["eef"] for record in recorder.records])
        history["diagnostic_target_xyz"] = np.stack([record["after"]["target"] for record in recorder.records])
        history["gripper_qpos"] = np.stack([record["after"]["finger_qpos"] for record in recorder.records])
        history["gripper_qvel"] = np.stack([record["after"]["finger_qvel"] for record in recorder.records])
        history["actuator_controls"] = np.stack([record["after"]["actuator_ctrl"] for record in recorder.records])
        np.savez_compressed(output / "trajectory.npz", **history, metadata=json.dumps(result))
    assert len(rows) == result["control_steps"]
    if result["success"]:
        cfg = env.config
        last = rows[-cfg.hold_steps:]
        assert [int(row["hold_streak"]) for row in last] == list(range(1, cfg.hold_steps + 1))
        anchor = np.array([last[0]["target_" + axis] for axis in "xyz"])
        for row in last:
            assert row["target_lift_m"] > cfg.lift_threshold
            assert row["target_linear_speed"] <= cfg.max_linear_speed and row["target_angular_speed"] <= cfg.max_angular_speed
            assert np.linalg.norm(np.array([row["target_" + axis] for axis in "xyz"]) - anchor) <= cfg.max_hold_drift + 1e-10
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--demonstrations", type=int, default=50)
    parser.add_argument("--test-episodes", type=int, default=30)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(args.gpu)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if (OUTPUT / "completed.json").exists():
        raise RuntimeError("Study already complete; preserve its results")
    baseline = Config.load(SOURCE / "config.json")
    expert = ExpertConfig(**json.loads((OUTPUT / "expert_config.json").read_text()))
    efficient = task_config(baseline, expert)
    protocol = {"baseline_source": str(SOURCE), "target_color": "red", "first_steps_to_analyze": 70,
                "development_seeds": list(range(60000, 60005)), "evaluation_seeds": list(range(50000, 50000 + args.test_episodes)),
                "demonstration_seed_start": 60000, "requested_successful_demonstrations": args.demonstrations,
                "baseline_config": asdict(baseline), "efficient_task_config": asdict(efficient),
                "expert_config": asdict(expert), "physical_gpu": args.gpu,
                "unchanged_success_criteria": ["lift > 0.04m relative to reset", "40 consecutive stable samples at 20Hz", "correct red grasp"],
                "ground_truth_policy": "object truth recorded for analysis only; actions use HSV estimate, robot state, fixed geometry"}
    (OUTPUT / "protocol.json").write_text(json.dumps(protocol, indent=2))
    analyze_existing()
    started = time.time()
    summaries = []
    for kind, cfg in (("baseline", baseline), ("efficient", efficient)):
        env = make_env(args.gpu, cfg)
        try:
            result = write_case(env, OUTPUT / "representative" / kind, kind, 40000, expert)
            print("REPRESENTATIVE", kind, result["control_steps"], result["first70"]["composition"], flush=True)
            for index, seed in enumerate(protocol["evaluation_seeds"]):
                result = write_case(env, OUTPUT / "paired" / kind / f"seed_{seed}", kind, seed, expert)
                summaries.append(result)
                print("TEST", kind, seed, result["success"], result["control_steps"], result["collision_steps"], flush=True)
        finally:
            env.close()
    # Full-state replay of the best existing ACT structure at the same example seed.
    import torch
    from .model import ColorACT
    torch.set_num_threads(4)
    checkpoint = torch.load(SOURCE / "mask" / "policy.pt", map_location="cpu", weights_only=False)
    model = ColorACT("mask", checkpoint["config"]["chunk_size"], pretrained=False).cuda().eval()
    model.load_state_dict(checkpoint["model"])
    env = make_env(args.gpu, baseline)
    try:
        result = write_case(env, OUTPUT / "representative" / "mask_act", "mask_act", 40000, expert, learned=(model, checkpoint))
        print("REPRESENTATIVE mask_act", result["success"], result["control_steps"], flush=True)
    finally:
        env.close()
    del model, checkpoint
    torch.cuda.empty_cache()
    demonstrations, failures = [], []
    env = make_env(args.gpu, efficient)
    try:
        attempt = 0
        while len(demonstrations) < args.demonstrations:
            seed = 60000 + attempt
            attempt += 1
            directory = OUTPUT / "demonstrations" / f"attempt_{attempt:03d}_seed_{seed}"
            result = write_case(env, directory, "efficient", seed, expert, save_dataset=True)
            result["directory"] = str(directory.relative_to(OUTPUT))
            (demonstrations if result["success"] else failures).append(result)
            print("DEMO", len(demonstrations), attempt, seed, result["success"], result["control_steps"], flush=True)
            if attempt > args.demonstrations * 4:
                raise RuntimeError("Success rate too low; inspect failed attempts")
    finally:
        env.close()
    (OUTPUT / "paired_results.json").write_text(json.dumps(summaries, indent=2))
    (OUTPUT / "demonstration_manifest.json").write_text(json.dumps({"successful": demonstrations, "failed_attempts": failures,
                                                                   "success_count": len(demonstrations), "attempt_count": len(demonstrations) + len(failures)}, indent=2))
    status = {"completed": True, "seconds": time.time() - started, "successful_demonstrations": len(demonstrations),
              "paired_evaluation_episodes_per_controller": args.test_episodes, "all_saved_successes_verified": True}
    (OUTPUT / "completed.json").write_text(json.dumps(status, indent=2))
    print("SHORT TRAJECTORY STUDY COMPLETE", json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
