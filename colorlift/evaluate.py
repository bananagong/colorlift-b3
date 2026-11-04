"""Fixed-scene closed-loop comparison with per-step and per-episode diagnostics."""
import argparse
import json
import time
import numpy as np
import torch
import cv2
import imageio.v2 as imageio
from PIL import Image
from .config import Config, EXPERIMENT, PHASES
from .env import make_env
from .model import ColorACT, VARIANTS
from .vision import observation_images, observation_masks, mask_preview
from .primitives import PrimitiveExecutor, MotionDiagnostics, policy_state
from .collect import rollout as expert_rollout, save_csv
from lift_act.evaluate import wilson


def learned_rollout(env, model, checkpoint, seed, out, index, variant):
    cfg = env.config
    obs = env.reset_seed(seed)
    metadata = env.episode_metadata()
    executor, diagnostics = PrimitiveExecutor(cfg), MotionDiagnostics(cfg)
    stats = checkpoint["stats"]
    queries, aborted = 0, False
    info = {}
    with imageio.get_writer(str(out / f"episode_{index:03d}.mp4"), fps=cfg.hz, codec="libx264", quality=7) as writer:
        for step in range(cfg.horizon):
            rgb, masks = observation_images(obs), observation_masks(obs)
            image_preview = np.concatenate([np.concatenate(list(rgb), axis=1),
                                            np.concatenate([mask_preview(m) for m in masks], axis=1)], axis=0)
            cv2.putText(image_preview, f'{variant} {PHASES[executor.phase]} t={step} hold={info.get("hold_streak",0)}/40', (3, 12), cv2.FONT_HERSHEY_SIMPLEX, .32, (30, 30, 30), 1)
            writer.append_data(image_preview)
            new_query = step % cfg.query_every == 0
            if new_query:
                visual = rgb if variant == "rgb" else masks if variant == "mask" else np.concatenate([rgb, masks], axis=-1)
                robot = (policy_state(obs, executor.phase) - stats["state_mean"]) / stats["state_std"]
                s = torch.as_tensor(robot[None], device="cuda")
                v = torch.as_tensor(visual.transpose(0, 3, 1, 2)[None].copy(), device="cuda").float() / 255
                with torch.inference_mode():
                    prediction = model(s, v)[0][0].cpu().numpy()
                prediction[:, :3] = prediction[:, :3] * stats["xyz_std"] + stats["xyz_mean"]
                queries += 1
            action_index = step % cfg.query_every
            xyz = prediction[action_index, :3]
            requested_phase = int(prediction[action_index, 3:].argmax())
            assert np.isfinite(xyz).all()
            action, control = executor.action(obs, requested_phase, xyz, new_query=new_query)
            previous = obs
            obs, _, done, info = env.step(action)
            diagnostics.update(env, previous, obs, control, info)
            if done:
                break
            if executor.should_abort_empty(obs):
                aborted = True
                break
    rgb = observation_images(obs)
    Image.fromarray(np.concatenate(list(rgb), axis=1)).save(out / f"episode_{index:03d}_final.png")
    result = {**metadata, **info, **diagnostics.summary(), "seed": seed,
              "macro_actions": executor.transitions + 1, "policy_queries": queries,
              "phase_requests_rejected": executor.rejected_phase_requests, "aborted_empty_grasp": aborted}
    return result, diagnostics.rows


def outcome(result):
    if result.get("success"):
        return "success"
    if result.get("wrong_color_grasp"):
        return "wrong_color"
    if result.get("reached_height"):
        return "hold_failure"
    if result.get("aborted_empty_grasp"):
        return "empty_grasp"
    return "no_lift" if result.get("target_grasped_ever") else "no_grasp"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["hsv", *VARIANTS], required=True)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--checkpoint", type=__import__("pathlib").Path)
    parser.add_argument("--output", type=__import__("pathlib").Path)
    parser.add_argument("--config", type=__import__("pathlib").Path)
    args = parser.parse_args()
    cfg = Config.load(args.config or EXPERIMENT / "config.json")
    out = args.output.resolve() if args.output else EXPERIMENT / args.variant / ("eval_" + args.split)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "metrics.json").exists():
        raise RuntimeError("This evaluation is already recorded; preserve it")
    torch.set_num_threads(4)
    model = checkpoint = None
    if args.variant != "hsv":
        checkpoint = torch.load(args.checkpoint or EXPERIMENT / args.variant / "best.pt", map_location="cpu", weights_only=False)
        model = ColorACT(args.variant, checkpoint["config"]["chunk_size"], pretrained=False).cuda().eval()
        model.load_state_dict(checkpoint["model"])
        if args.config is None and "task_config" in checkpoint:
            cfg = Config(**checkpoint["task_config"])
    env = make_env(args.gpu, cfg)
    results = []
    started = time.time()
    base = 40000 if args.split == "test" else 30000
    try:
        for index, seed in enumerate(range(base, base + args.episodes)):
            if model is None:
                result, history, rows = expert_rollout(env, seed, record=True)
                with imageio.get_writer(str(out / f"episode_{index:03d}.mp4"), fps=cfg.hz, codec="libx264", quality=7) as writer:
                    for rgb, masks in zip(history["images"], history["masks"]):
                        writer.append_data(np.concatenate([np.concatenate(list(rgb), axis=1),
                                                           np.concatenate([mask_preview(m) for m in masks], axis=1)], axis=0))
                Image.fromarray(np.concatenate(list(history["images"][-1]), axis=1)).save(out / f"episode_{index:03d}_final.png")
            else:
                result, rows = learned_rollout(env, model, checkpoint, seed, out, index, args.variant)
            result["outcome"] = outcome(result)
            result["video"] = f"episode_{index:03d}.mp4"
            result["step_trace"] = f"episode_{index:03d}_steps.csv"
            result["final_holding_failure"] = bool(not result["success"] and result["reached_height"])
            result["necessary_hold_steps"] = cfg.hold_steps if result["success"] else 0
            result["non_mandatory_control_steps"] = result["control_steps"] - result["necessary_hold_steps"]
            result["excess_valid_hold_steps"] = max(0, result["valid_hold_steps"] - result["necessary_hold_steps"])
            save_csv(out / result["step_trace"], rows)
            results.append(result)
            print(json.dumps({key: result.get(key) for key in ("seed", "success", "outcome", "control_steps", "macro_actions", "wrong_color_grasp", "collision_steps", "holding_failure_events", "command_reversals_within_phase", "post_close_idle_steps", "no_progress_steps")}), flush=True)
            (out / "episodes.partial.json").write_text(json.dumps(results, indent=2))
    finally:
        env.close()
    successes = [r for r in results if r["success"]]
    count = len(successes)
    mean = lambda key, items=results: float(np.mean([r.get(key, 0) for r in items])) if items else None
    summary = {
        "variant": args.variant, "episodes": len(results), "successes": count,
        "success_rate": count / len(results), "wilson_95_ci": wilson(count, len(results)),
        "collision_free_success_rate": sum(bool(r["success"] and r["collision_steps"] == 0) for r in results) / len(results),
        "mean_control_steps": mean("control_steps"), "mean_success_steps": mean("control_steps", successes),
        "mean_non_hold_success_steps": mean("non_hold_control_steps", successes),
        "mean_non_mandatory_success_steps": mean("non_mandatory_control_steps", successes),
        "mean_macro_actions": mean("macro_actions"),
        "wrong_color_rate": mean("wrong_color_grasp"), "collision_episode_rate": float(np.mean([r["collision_steps"] > 0 for r in results])),
        "hold_failure_rate": mean("final_holding_failure"),
        "hold_interruption_rate": float(np.mean([r["holding_failure_events"] > 0 for r in results])),
        "mean_command_reversals_within_phase": mean("command_reversals_within_phase"),
        "mean_motion_reversals_within_phase": mean("motion_reversals_within_phase"),
        "mean_post_close_idle_steps": mean("post_close_idle_steps"),
        "mean_empty_gripper_wait_steps": mean("empty_gripper_wait_steps"), "mean_no_progress_steps": mean("no_progress_steps"),
        "mean_collision_steps": mean("collision_steps"), "mean_valid_hold_steps": mean("valid_hold_steps"),
        "checkpoint_step": None if checkpoint is None else checkpoint["step"],
        "validation_score": None if checkpoint is None else checkpoint["val_score"],
        "training_config": None if checkpoint is None else checkpoint["config"],
        "wall_seconds": time.time() - started, "target_color": cfg.target_color,
        "collision_sample_hz": 500, "hold_sample_hz": cfg.hz,
        "task_config": __import__("dataclasses").asdict(cfg), "trials": results,
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    fields = ["seed", "outcome", "success", "control_steps", "macro_actions", "non_hold_control_steps",
              "wrong_color_grasp", "collision_steps", "collision_events", "holding_failure_events", "max_hold_streak",
              "command_reversals_all", "command_reversals_within_phase", "motion_reversals_within_phase",
              "post_close_idle_steps", "empty_gripper_wait_steps", "no_progress_steps", "max_lift_m",
              "phase_requests_rejected", "aborted_empty_grasp", "necessary_hold_steps", "non_mandatory_control_steps", "excess_valid_hold_steps"]
    save_csv(out / "episodes.csv", [{field: trial.get(field) for field in fields} for trial in results])
    print(json.dumps({key: value for key, value in summary.items() if key not in ("trials", "task_config")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
