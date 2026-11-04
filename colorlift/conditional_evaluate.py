"""Single-attempt three-color evaluation and same-RGB target-color interventions."""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
import cv2
import imageio.v2 as imageio
from PIL import Image
from .conditional import RUN, ConditionalACT, FastExecutor, recognized_masks, configuration
from .config import COLORS
from .env import make_env
from .vision import observation_images, observation_masks, mask_preview
from .primitives import MotionDiagnostics, policy_state
from .step_recording import StepRecorder, save_rows
from lift_act.evaluate import wilson


def tensor(x): return torch.as_tensor(x, device="cuda")


def run_episode(env, model, checkpoint, seed, directory, query_every=3, *, query_schedule=None, run_root=None):
    directory.mkdir(parents=True, exist_ok=True)
    cfg, expert = env.config, configuration()[1]
    obs = env.reset_seed(seed)
    initial_rgb = observation_images(obs)
    meta = env.episode_metadata()
    meta.update(seed=seed, initial_rgb_sha256=hashlib.sha256(initial_rgb.tobytes()).hexdigest(),
                initial_robot_state=policy_state(obs, 0)[:16].tolist())
    executor, diagnostics = FastExecutor(cfg, expert), MotionDiagnostics(cfg)
    statistics = checkpoint["stats"]
    color_id = COLORS.index(env.target_color)
    grasp_streak = {color: 0 for color in COLORS}
    first_picked = None
    all_grasped = set()
    info, aborted, queries = {}, False, 0
    first_prediction = first_uv = initial_iou = None
    last_query_step = -1
    with StepRecorder(env) as recorder, imageio.get_writer(str(directory / "rollout.mp4"), fps=cfg.hz, codec="libx264", quality=7) as writer:
        writer.append_data(np.concatenate(list(initial_rgb), axis=1))
        for step in range(cfg.horizon):
            rgb = observation_images(obs)
            interval = int(query_schedule[executor.phase]) if query_schedule is not None else query_every
            new_query = last_query_step < 0 or step-last_query_step >= interval
            if new_query:
                last_query_step = step
                robot = (policy_state(obs, executor.phase)[:16]-statistics["state_mean"])/statistics["state_std"]
                with torch.inference_mode():
                    output, logits, uv = model(tensor(robot[None]), tensor(rgb.transpose(0, 3, 1, 2)[None].copy()).float()/255,
                                               tensor(np.array([color_id], dtype=np.int64)))
                prediction = output[0][0].cpu().numpy()
                prediction[:, :3] = prediction[:, :3]*statistics["xyz_std"]+statistics["xyz_mean"]
                queries += 1
                if first_prediction is None:
                    first_prediction = prediction[0].tolist()
                    first_uv = uv[0].cpu().tolist()
                    learned_masks = recognized_masks(logits)[0].cpu().numpy().transpose(0, 2, 3, 1)
                    # HSV is an evaluation label here, computed AFTER the learned prediction.
                    labels = observation_masks(obs) > 0
                    intersection = (learned_masks & labels).sum((1, 2))
                    union = (learned_masks | labels).sum((1, 2)).clip(1)
                    initial_iou = (intersection/union).mean(0).tolist()
                    Image.fromarray(np.concatenate(list(rgb), axis=1)).save(directory / "initial_rgb.png")
                    preview = np.concatenate([mask_preview(mask.astype(np.uint8)*255) for mask in learned_masks], axis=1)
                    Image.fromarray(preview).save(directory / "initial_learned_masks.png")
            index = step-last_query_step
            xyz, phase = prediction[index, :3], int(prediction[index, 3:].argmax())
            assert np.isfinite(xyz).all()
            action, control = executor.action(obs, phase, xyz)
            before = obs
            obs, _, done, info = env.step(action)
            executor.after_step(obs, control)
            diagnostics.update(env, before, obs, control, info)
            row = diagnostics.rows[-1]
            row.update(policy_query_age=index, new_policy_query=new_query, execution_length=interval)
            for axis, value in zip("xyz", xyz): row["policy_goal_"+axis] = float(value)
            for color in COLORS:
                grasp = info["grasp_by_color"][color]
                grasp_streak[color] = grasp_streak[color]+1 if grasp else 0
                row[f"grasp_{color}"] = grasp
                xyz_true = env.sim.data.body_xpos[env.color_body_ids[color]]
                for axis, value in zip("xyz", xyz_true): row[f"body_{color}_{axis}"] = float(value)
                row[f"lift_{color}_m"] = float(xyz_true[2]-env.initial_heights[color])
                if grasp_streak[color] >= cfg.wrong_grasp_confirm_steps:
                    all_grasped.add(color)
                    if first_picked is None: first_picked=color
            row.update(first_picked_color=first_picked or "none", target_color=env.target_color)
            frame = np.concatenate(list(observation_images(obs)), axis=1)
            text = f'{env.target_color} ACT t={step+1} p={executor.phase} hold={info["hold_streak"]}/40'
            cv2.putText(frame, text, (2, 11), cv2.FONT_HERSHEY_SIMPLEX, .3, (20, 20, 20), 1)
            writer.append_data(frame)
            empty = executor.phase == 3 and executor.close_steps >= 12 and abs(obs["robot0_gripper_qpos"]).sum() < .006
            if done or empty:
                aborted = bool(empty and not done)
                break
    rows = recorder.enrich(diagnostics.rows, seed=seed)
    save_rows(directory / "steps.csv", rows)
    Image.fromarray(np.concatenate(list(observation_images(obs)), axis=1)).save(directory / "final_rgb.png")
    result = {**meta, **info, **diagnostics.summary(), "first_picked_color": first_picked or "none", "confirmed_grasped_colors": sorted(all_grasped),
              "correct_object_selected": first_picked == env.target_color, "aborted_empty_grasp": aborted,
              "initial_predicted_xyz": first_prediction[:3], "initial_predicted_phase": int(np.argmax(first_prediction[3:])),
              "initial_predicted_uv": first_uv, "initial_recognition_iou_by_color": dict(zip(COLORS, initial_iou)),
              "policy_queries": queries, "query_every": query_every, "action_chunk_maximum": checkpoint["config"]["max_chunk"],
              "executed_chunk_prefix": list(query_schedule) if query_schedule is not None else query_every, "checkpoint_step": checkpoint["step"],
              "outcome": "success" if info["success"] else "wrong_color" if info["wrong_color_grasp"] else "hold_failure" if info["reached_height"] else "empty_grasp" if aborted else "no_grasp",
              "directory": str(directory.relative_to(run_root or RUN))}
    closing = next((row for row in rows if row["phase"] == 2), None)
    result["closing_true_xy_error_m"] = None if closing is None else float(np.linalg.norm([closing["eef_before_"+axis]-closing["target_before_"+axis] for axis in "xy"]))
    true_xy = np.array([meta["initial_positions"][color][:2] for color in COLORS])
    result["initial_waypoint_nearest_color"] = COLORS[np.argmin(np.linalg.norm(true_xy-np.asarray(first_prediction[:2]), axis=1))]
    result["initial_xy_error_m"] = float(np.linalg.norm(np.asarray(first_prediction[:2])-true_xy[color_id]))
    if result["success"]:
        assert [int(row["hold_streak"]) for row in rows[-40:]] == list(range(1, 41))
        assert min(row["target_lift_m"] for row in rows[-40:]) > .04
        assert not info["wrong_color_grasp"]
    temporary = directory / "metadata.tmp.json"
    temporary.write_text(json.dumps(result, indent=2))
    temporary.replace(directory / "metadata.json")
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=4)
    p.add_argument("--split", choices=("dev", "test"), default="dev")
    p.add_argument("--scenes", type=int, default=12)
    p.add_argument("--checkpoint", type=Path, default=RUN / "model/policy.pt")
    p.add_argument("--name", default="selected")
    p.add_argument("--query-every", type=int, default=3)
    p.add_argument("--cooperate", action="store_true", help="Claim unique episodes while another GPU works in the same output")
    p.add_argument("--reverse", action="store_true")
    args=p.parse_args()
    out = RUN / f"eval_{args.split}_{args.name}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "metrics.json").exists(): raise RuntimeError("Evaluation already exists")
    torch.set_num_threads(4)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = ConditionalACT(12, pretrained=False).cuda().eval()
    model.load_state_dict(checkpoint["model"])
    base = 74000 if args.split == "dev" else 75000
    trials=[]
    envs={color: make_env(args.gpu, configuration(color)[0]) for color in COLORS}
    started=time.time()
    try:
        seeds=list(range(base, base+args.scenes))
        if args.reverse: seeds.reverse()
        for seed in seeds:
            for color in COLORS:
                directory=out / f"scene_{seed}" / color
                if (directory / "metadata.json").exists():
                    result=json.loads((directory / "metadata.json").read_text())
                else:
                    lock=directory / "running.lock"
                    directory.mkdir(parents=True, exist_ok=True)
                    if args.cooperate:
                        try:
                            with lock.open("x") as stream: stream.write(str(os.getpid()))
                        except FileExistsError:
                            continue
                    try:
                        if (directory / "metadata.json").exists():
                            result=json.loads((directory / "metadata.json").read_text())
                        else:
                            print("EXECUTE_EPISODE", seed, color, "worker", args.gpu, flush=True)
                            result=run_episode(envs[color], model, checkpoint, seed, directory, args.query_every)
                    finally:
                        if args.cooperate: lock.unlink(missing_ok=True)
                trials.append(result)
                print("EVAL", args.split, args.name, seed, color, result["success"], result["first_picked_color"], result["control_steps"], result["collision_steps"], flush=True)
                (out / (f"partial_gpu{args.gpu}.json" if args.cooperate else "partial.json")).write_text(json.dumps(trials, indent=2))
        if args.cooperate:
            expected=[out / f"scene_{seed}" / color for seed in range(base, base+args.scenes) for color in COLORS]
            deadline=time.time()+1800
            while not all((directory / "metadata.json").exists() for directory in expected):
                if time.time() > deadline: raise RuntimeError("Other evaluator did not finish; preserve partial results")
                time.sleep(3)
            trials=[json.loads((directory / "metadata.json").read_text()) for directory in expected]
    finally:
        for env in envs.values(): env.close()
    switching=[]
    for seed in range(base, base+args.scenes):
        group=[trial for trial in trials if trial["seed"] == seed]
        assert len(set(row["initial_rgb_sha256"] for row in group)) == 1
        assert all(np.array_equal(row["initial_robot_state"], group[0]["initial_robot_state"]) for row in group)
        for color in COLORS: assert all(np.array_equal(row["initial_positions"][color], group[0]["initial_positions"][color]) for row in group)
        picked=[row["first_picked_color"] for row in group]
        switching.append({"seed": seed, "same_RGB_and_robot_state": True,
                          "red_command_picked": picked[0], "yellow_command_picked": picked[1], "blue_command_picked": picked[2],
                          "all_three_selected_correctly": all(row["correct_object_selected"] for row in group),
                          "all_three_grasps_successful": all(row["success"] for row in group),
                          "three_distinct_objects_picked": len(set(picked)-{"none"}) == 3,
                          "initial_predicted_target_changes_correctly": all(row["initial_waypoint_nearest_color"] == row["target_color"] for row in group)})
    by_color={}
    confusion={command: {picked: 0 for picked in (*COLORS, "none")} for command in COLORS}
    for row in trials: confusion[row["target_color"]][row["first_picked_color"]]+=1
    for color in COLORS:
        group=[row for row in trials if row["target_color"] == color]
        successes=[row for row in group if row["success"]]
        by_color[color]={"episodes": len(group), "successes": len(successes), "success_rate": len(successes)/len(group),
                         "wilson_95_ci": wilson(len(successes), len(group)),
                         "correct_object_selections": sum(row["correct_object_selected"] for row in group),
                         "mean_success_steps": float(np.mean([row["control_steps"] for row in successes])) if successes else None,
                         "wrong_color_episodes": sum(row["wrong_color_grasp"] for row in group),
                         "collision_episodes": sum(row["collision_steps"] > 0 for row in group),
                         "hold_failure_episodes": sum(row["outcome"] == "hold_failure" for row in group),
                         "initial_xy_error_mean_m": float(np.mean([row["initial_xy_error_m"] for row in group])),
                         "recognition_iou_mean": float(np.mean([row["initial_recognition_iou_by_color"][color] for row in group]))}
    metric={"split": args.split, "checkpoint": str(args.checkpoint), "checkpoint_step": checkpoint["step"], "query_every": args.query_every,
            "scenes": args.scenes, "episodes": len(trials), "successes": sum(row["success"] for row in trials),
            "per_color": by_color, "command_vs_actual_grasp_confusion": confusion,
            "all_three_successful_scenes": sum(row["all_three_grasps_successful"] for row in switching),
            "all_three_correctly_selected_scenes": sum(row["all_three_selected_correctly"] for row in switching),
            "counterfactual_initial_prediction_correct_scenes": sum(row["initial_predicted_target_changes_correctly"] for row in switching),
            "all_scene_initial_images_identical_across_commands": True, "single_attempt_no_retry": True,
            "model_input": "RGB, proprioception, learned color ID embedding; no HSV/true object coordinates",
            "seconds": time.time()-started, "trials": trials, "switching": switching}
    temporary=out / f"metrics_{os.getpid()}.tmp.json"
    temporary.write_text(json.dumps(metric, indent=2))
    temporary.replace(out / "metrics.json")
    save_rows(out / "switching.csv", switching)
    save_rows(out / "episodes.csv", [{key: row[key] for key in ("seed", "target_color", "success", "first_picked_color", "control_steps", "outcome", "collision_steps", "wrong_color_grasp", "initial_xy_error_m", "initial_waypoint_nearest_color")} for row in trials])
    print("EVALUATION COMPLETE", json.dumps({key: value for key, value in metric.items() if key not in ("trials", "switching")}), flush=True)


if __name__ == "__main__": main()
