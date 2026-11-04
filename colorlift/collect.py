"""HSV-calibrated four-primitive expert and reproducible ColorLift demonstration collection."""
import argparse
import csv
import json
import time
import numpy as np
import imageio.v2 as imageio
from .config import Config, EXPERIMENT, PHASES
from .env import make_env
from .vision import observation_images, observation_masks, mask_preview, locate_on_table
from .primitives import PrimitiveExecutor, MotionDiagnostics, policy_state


def rollout(env, seed, *, expert_color=None, record=True, release_during_hold=False):
    obs = env.reset_seed(seed)
    cfg = env.config
    metadata = env.episode_metadata()
    metadata["seed"] = seed
    color = expert_color or env.target_color
    estimate, camera_details = locate_on_table(env, obs, color)
    metadata["expert_color"] = color
    metadata["localization"] = camera_details
    truth = env.sim.data.body_xpos[env.color_body_ids[color]]
    metadata["hsv_localization_error_m"] = float(np.linalg.norm(estimate[:2] - truth[:2]))
    executor = PrimitiveExecutor(cfg)
    diagnostics = MotionDiagnostics(cfg)
    history = {key: [] for key in ("states", "images", "masks", "actions")}
    info = {}
    done, aborted, released = False, False, False
    goals = [np.r_[estimate[:2], z] for z in (cfg.approach_height, cfg.grasp_height, cfg.grasp_height, cfg.lift_height)]
    budgets = [40, 35, cfg.minimum_close_steps + 2, cfg.horizon]
    for phase in range(4):
        for local_step in range(budgets[phase]):
            previous = obs
            input_state = policy_state(obs, executor.phase)
            action, control = executor.action(obs, phase, goals[phase], force_transition=True)
            if record:
                history["states"].append(input_state)
                history["images"].append(observation_images(obs))
                history["masks"].append(observation_masks(obs))
                history["actions"].append(np.r_[goals[phase], np.eye(4)[phase]].astype(np.float32))
            if release_during_hold and (released or info.get("hold_streak", 0) >= 10):
                action = np.r_[np.zeros(6), -1.0]
                released = True
            obs, _, done, info = env.step(action)
            diagnostics.update(env, previous, obs, control, info)
            if done:
                break
            if executor.should_abort_empty(obs) and not release_during_hold:
                aborted = True
                break
            if phase in (0, 1) and local_step >= 4 and np.linalg.norm(obs["robot0_eef_pos"] - control["goal_xyz"]) < cfg.goal_reached_tolerance:
                break
            if phase == 2 and local_step + 1 >= cfg.minimum_close_steps and np.linalg.norm(obs["robot0_gripper_qpos"] - previous["robot0_gripper_qpos"]) < 0.0002:
                break
        if done or aborted:
            break
    result = {**metadata, **info, **diagnostics.summary(),
              "macro_actions": executor.transitions + 1,
              "phase_requests_rejected": executor.rejected_phase_requests,
              "aborted_empty_grasp": aborted, "released_for_test": released}
    result["outcome"] = ("success" if info.get("success") else "wrong_color" if info.get("wrong_color_grasp") else
                         "hold_failure" if info.get("reached_height") else "empty_grasp" if aborted else "no_lift")
    if record:
        history = {key: np.asarray(value) for key, value in history.items()}
    return result, history, diagnostics.rows


def save_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--config")
    parser.add_argument("--pilot", type=int, default=0)
    parser.add_argument("--train-count", type=int, default=100)
    parser.add_argument("--val-count", type=int, default=20)
    args = parser.parse_args()
    cfg = Config.load(args.config)
    root = EXPERIMENT / "data"
    root.mkdir(parents=True, exist_ok=True)
    existing = EXPERIMENT / "config.json"
    if existing.exists() and Config.load(existing) != cfg and any(root.glob("train_*.npz")):
        raise RuntimeError("Task/controller config differs from saved data; set COLORLIFT_RUN_DIR to a new experiment folder")
    cfg.save(EXPERIMENT / "config.json")
    env = make_env(args.gpu, cfg)
    manifest, failures = [], []
    started = time.time()
    try:
        splits = [("pilot", args.pilot, 10000)] if args.pilot else [("train", args.train_count, 10000), ("val", args.val_count, 20000)]
        for split, count, seed_base in splits:
            index, attempt = 0, 0
            while index < count:
                path = root / f"{split}_{index:03d}.npz"
                if path.exists() and split != "pilot":
                    with np.load(path) as data:
                        result = json.loads(str(data["metadata"]))
                    manifest.append(result)
                    attempt = max(attempt, result["seed"] - seed_base + 1)
                    index += 1
                    continue
                result, history, rows = rollout(env, seed_base + attempt)
                attempt += 1
                result["split"] = split
                print(json.dumps({k: result.get(k) for k in ("seed", "success", "control_steps", "outcome", "hsv_localization_error_m", "max_hold_streak", "collision_steps", "command_reversals_within_phase", "post_close_idle_steps")}), flush=True)
                if result["success"] or split == "pilot":
                    np.savez_compressed(path, **history, metadata=json.dumps(result))
                    manifest.append(result)
                    if index < 2 or split == "pilot":
                        save_csv(root / f"{split}_{index:03d}_steps.csv", rows)
                        with imageio.get_writer(str(root / f"{split}_{index:03d}.mp4"), fps=cfg.hz, codec="libx264") as writer:
                            for rgb, masks in zip(history["images"], history["masks"]):
                                upper = np.concatenate(list(rgb), axis=1)
                                lower = np.concatenate([mask_preview(mask) for mask in masks], axis=1)
                                writer.append_data(np.concatenate([upper, lower], axis=0))
                    index += 1
                else:
                    failures.append(result)
                if attempt > count * 4:
                    raise RuntimeError("HSV expert failed too often; inspect localization and physical control")
        output = {"episodes": manifest, "failed_attempts": failures, "seconds": time.time() - started,
                  "expert": "HSV masks + camera calibration + four OSC primitives; no object-pose action input"}
        (root / ("pilot.json" if args.pilot else "manifest.json")).write_text(json.dumps(output, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
