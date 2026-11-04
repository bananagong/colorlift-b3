"""Collect scripted successful grasp-and-lift demonstrations; expert access is explicit."""
import argparse
import json
import time
import numpy as np
import imageio.v2 as imageio
from .env import EXPERIMENT, HORIZON, HZ, make_env, reset_seed, images, state, servo_action, success_tracker


def rollout(env, seed, record=True):
    obs = reset_seed(env, seed)
    initial = obs["cube_pos"].copy()
    # Privileged cube position is used ONLY by this scripted demonstration expert.
    grasp = initial.copy()
    grasp[2] = 0.818
    waypoints = [(grasp + [0, 0, 0.13], -1, 40), (grasp, -1, 35),
                 (grasp, 1, 20), (grasp + [0, 0, 0.17], 1, 40),
                 (grasp + [0, 0, 0.17], 1, 25)]
    history = {"states": [], "images": [], "actions": [], "heights": []}
    phases = []
    streak, success, elapsed = 0, False, 0
    for phase, (goal, grip, max_steps) in enumerate(waypoints):
        for local_step in range(max_steps):
            target = np.r_[goal, grip].astype(np.float32)
            if record:
                history["states"].append(state(obs))
                history["images"].append(images(obs))
                history["actions"].append(target)
                phases.append(phase)
            obs, _, _, _ = env.step(servo_action(obs, target))
            elapsed += 1
            streak, success, z = success_tracker(env, streak)
            history["heights"].append(z)
            if success:
                # Keep five extra frames to demonstrate a stable lifted hold.
                if streak >= 15:
                    break
            if phase in (0, 1) and local_step >= 8 and np.linalg.norm(obs["robot0_eef_pos"] - goal) < 0.004:
                break
        if success and streak >= 15:
            break
    result = {"seed": seed, "success": bool(success), "steps": elapsed,
              "initial_cube_pos": initial.tolist(), "max_height": max(history["heights"]),
              "final_eef_pos": obs["robot0_eef_pos"].tolist()}
    if record:
        history = {key: np.asarray(value) for key, value in history.items()}
        history["phases"] = np.asarray(phases, dtype=np.int8)
    return result, history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--pilot", type=int, default=0)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--val-count", type=int, default=10)
    args = parser.parse_args()
    out = EXPERIMENT / "data"
    out.mkdir(parents=True, exist_ok=True)
    env = make_env(args.gpu, cameras=True)
    results = []
    started = time.time()
    try:
        for split, count, seed_base in (("pilot", args.pilot, 1000),) if args.pilot else (("train", args.count, 1000), ("val", args.val_count, 3000)):
            successes, attempt = 0, 0
            while successes < count:
                seed = seed_base + attempt
                attempt += 1
                path = out / f"{split}_{successes:03d}.npz"
                if path.exists() and split != "pilot":
                    with np.load(path) as cached:
                        results.append(json.loads(str(cached["metadata"])))
                    successes += 1
                    continue
                result, history = rollout(env, seed)
                result["split"] = split
                print(json.dumps(result), flush=True)
                if result["success"] or split == "pilot":
                    np.savez_compressed(path, **history, metadata=json.dumps(result))
                    results.append(result)
                    if successes == 0 or split == "pilot":
                        with imageio.get_writer(str(out / f"{split}_{successes:03d}.mp4"), fps=HZ, codec="libx264") as writer:
                            for frame in history["images"]:
                                writer.append_data(np.concatenate(list(frame), axis=1))
                    successes += 1
                if attempt > count * 3:
                    raise RuntimeError("Expert success too low; inspect collection before training")
        (out / ("pilot.json" if args.pilot else "manifest.json")).write_text(json.dumps({"episodes": results, "seconds": time.time() - started}, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
