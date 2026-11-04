"""Closed-loop ACT evaluation on fixed seeds; no expert corrections or object-pose input."""
import argparse
import json
import math
import time
import numpy as np
import torch
import imageio.v2 as imageio
from PIL import Image
from .env import EXPERIMENT, HORIZON, HZ, make_env, reset_seed, state, images, servo_action, success_tracker
from .model import ACT


def wilson(successes, trials):
    z = 1.959963984540054
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return [center - half, center + half]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demos", type=int, choices=[20, 50], required=True)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--query-every", type=int, default=5)
    parser.add_argument("--checkpoint", default="best.pt")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    root = EXPERIMENT / f"act_{args.demos}"
    checkpoint = torch.load(root / args.checkpoint, map_location="cpu", weights_only=False)
    model = ACT(checkpoint["config"]["chunk_size"], pretrained=False).cuda().eval()
    model.load_state_dict(checkpoint["model"])
    stats = checkpoint["stats"]
    out = root / f"eval_{args.split}_q{args.query_every}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "metrics.json").exists():
        raise RuntimeError("Evaluation already exists; preserve the recorded result")
    seeds = list(range(4000 if args.split == "dev" else 5000, (4000 if args.split == "dev" else 5000) + args.episodes))
    env = make_env(args.gpu)
    results = []
    started = time.time()
    try:
        for index, seed in enumerate(seeds):
            obs = reset_seed(env, seed)
            initial = obs["cube_pos"].copy()
            streak, success, peak, first_threshold_step = 0, False, 0.0, None
            heights, trajectory = [], []
            video = out / f"episode_{index:03d}.mp4"
            with imageio.get_writer(str(video), fps=HZ, codec="libx264", quality=7) as writer:
                for step in range(HORIZON):
                    frames = images(obs)
                    writer.append_data(np.concatenate(list(frames), axis=1))
                    if step % args.query_every == 0:
                        s = (state(obs) - stats["state_mean"]) / stats["state_std"]
                        tensor_s = torch.as_tensor(s[None], device="cuda")
                        tensor_i = torch.as_tensor(frames.transpose(0, 3, 1, 2)[None].copy(), device="cuda").float() / 255
                        with torch.inference_mode():
                            actions = model(tensor_s, tensor_i)[0][0].cpu().numpy()
                        actions = actions * stats["action_std"] + stats["action_mean"]
                    target = actions[step % args.query_every]
                    assert np.isfinite(target).all()
                    obs, _, _, _ = env.step(servo_action(obs, target))
                    streak, success, height = success_tracker(env, streak)
                    heights.append(height)
                    trajectory.append(np.r_[state(obs), target])
                    peak = max(peak, height)
                    if height >= 0.90 and first_threshold_step is None:
                        first_threshold_step = step + 1
                    if success:
                        writer.append_data(np.concatenate(list(images(obs)), axis=1))
                        break
            Image.fromarray(np.concatenate(list(images(obs)), axis=1)).save(out / f"episode_{index:03d}_final.png")
            np.savez_compressed(out / f"episode_{index:03d}_trajectory.npz", heights=heights, robot_state_and_target=trajectory)
            result = {"seed": seed, "success": bool(success), "steps": step + 1,
                      "seconds_simulated": (step + 1) / HZ, "initial_cube_pos": initial.tolist(),
                      "max_cube_height": peak, "first_threshold_step": first_threshold_step,
                      "final_hold_streak": streak, "video": video.name,
                      "failure_observation": None if success else ("height_reached_but_not_held" if first_threshold_step else "height_not_reached")}
            results.append(result)
            print(json.dumps(result), flush=True)
            (out / "episodes.partial.json").write_text(json.dumps(results, indent=2))
    finally:
        env.close()
    count = sum(x["success"] for x in results)
    success_steps = [x["steps"] for x in results if x["success"]]
    metrics = {
        "demos": args.demos, "split": args.split, "episodes": len(results), "successes": count,
        "success_rate": count / len(results), "wilson_95_ci": wilson(count, len(results)),
        "mean_success_steps": float(np.mean(success_steps)) if success_steps else None,
        "median_success_steps": float(np.median(success_steps)) if success_steps else None,
        "checkpoint": args.checkpoint, "checkpoint_training_step": checkpoint["step"],
        "validation_prior_l1": checkpoint["val_prior_l1"], "query_every_steps": args.query_every,
        "success_rule": "cube center z>=0.90m for 10 consecutive steps at 20Hz",
        "policy_access": "RGB and robot proprioception only; no expert fallback",
        "wall_seconds": time.time() - started, "trials": results,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: v for k, v in metrics.items() if k != "trials"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
