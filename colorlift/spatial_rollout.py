"""Real physics rollouts for new layouts; object truth is restricted to teacher/audit."""
import hashlib
import json
import numpy as np
import torch
import imageio.v2 as imageio
import cv2
from PIL import Image
from .conditional import ConditionalACT, configuration
from .config import COLORS
from .event_control import EventExecutor, FingerContactSensor, event_config
from .event_efficiency import PhysicsRecorder, eligibility
from .primitives import MotionDiagnostics, policy_state
from .step_recording import save_rows, first_steps_breakdown
from .vision import observation_images, observation_masks
from .spatial_gate import RUN


def load_policy(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = ConditionalACT(checkpoint["config"]["max_chunk"], pretrained=False).cuda().eval()
    model.load_state_dict(checkpoint["model"])
    return model, checkpoint


def episode(env, layout, target_color, directory, actor, model=None, checkpoint=None, save_data=False):
    if (directory / "metadata.json").exists():
        return json.loads((directory / "metadata.json").read_text())
    directory.mkdir(parents=True, exist_ok=True)
    obs = env.reset_layout(layout, target_color)
    cfg, expert = env.config, configuration()[1]
    initial_meta = env.episode_metadata()
    initial_images = observation_images(obs)
    initial_robot_state = policy_state(obs, 0).tolist()
    executor = EventExecutor(cfg, expert, event_config("efficiency"))
    sensor = FingerContactSensor(env, event_config("efficiency"))
    diagnostics = MotionDiagnostics(cfg)
    data = {key: [] for key in ("states", "images", "masks", "actions", "servo_waypoints", "osc_actions")} if save_data else None
    prediction = None; last_query = -1000; queries = 0; close_steps = 0
    grasp_streaks = {color: 0 for color in COLORS}
    first_picked_body = None
    first_goal = None
    info = {}
    with PhysicsRecorder(env, sensor) as recorder:
        for tick in range(cfg.horizon):
            old_phase = executor.phase
            new_query = False
            index = -1
            interval = (4, 2, 1, 4)[executor.phase]
            if actor == "expert":
                # Privileged teacher; these positions never enter the student's observations.
                truth = env.sim.data.body_xpos[env.color_body_ids[target_color]].copy()
                xyz, requested_phase = truth, executor.phase
            else:
                new_query = prediction is None or tick-last_query >= interval
                if new_query:
                    state = (policy_state(obs, executor.phase)[:16]-checkpoint["stats"]["state_mean"])/checkpoint["stats"]["state_std"]
                    rgb = observation_images(obs)
                    with torch.inference_mode():
                        output, _, _ = model(torch.as_tensor(state[None], device="cuda"),
                                             torch.as_tensor(rgb.transpose(0, 3, 1, 2)[None].copy(), device="cuda").float()/255,
                                             torch.tensor([COLORS.index(target_color)], device="cuda"))
                    prediction = output[0][0].cpu().numpy()
                    prediction[:, :3] = prediction[:, :3]*checkpoint["stats"]["xyz_std"]+checkpoint["stats"]["xyz_mean"]
                    last_query = tick; queries += 1
                index = tick-last_query
                assert 0 <= index < 4
                xyz, requested_phase = prediction[index, :3], int(prediction[index, 3:].argmax())
            if first_goal is None:
                first_goal = np.asarray(xyz[:2]).copy()
            action, control = executor.action(obs, requested_phase, xyz, sensor.read(obs))
            if save_data:
                macro = control["goal_xyz"].copy()
                if control["phase"] < 2:
                    macro[:2] = truth[:2]
                    macro[2] = cfg.table_height+cfg.cube_size/2+expert.pregrasp_clearance if control["phase"] == 0 else cfg.grasp_height
                data["states"].append(policy_state(obs, control["phase"]))
                data["images"].append(observation_images(obs))
                data["masks"].append(observation_masks(obs))
                data["actions"].append(np.r_[macro, np.eye(4)[control["phase"]]].astype(np.float32))
                data["servo_waypoints"].append(control["goal_xyz"].copy())
                data["osc_actions"].append(action.copy())
            before = obs
            obs, _, done, info = env.step(action)
            diagnostics.update(env, before, obs, control, info)
            diagnostics.rows[-1].update(transition_event=control["transition_event"], new_policy_query=new_query,
                                        policy_query_age=index, execution_length=interval, hold_drift_m=info["hold_drift_m"])
            executor.after_step(obs, control)
            if executor.phase != old_phase:
                prediction = None
            for color in COLORS:
                grasp_streaks[color] = grasp_streaks[color]+1 if info["grasp_by_color"][color] else 0
                if first_picked_body is None and grasp_streaks[color] >= 3:
                    first_picked_body = int(env.color_body_ids[color])
            empty = executor.phase >= 2 and executor.phase_steps >= 12 and np.abs(obs["robot0_gripper_qpos"]).sum() < .006
            if done or empty:
                break
    rows = recorder.enrich(diagnostics.rows, seed=layout["seed"])
    result = {**initial_meta, **info, **diagnostics.summary(), "actor": actor, "seed": layout["seed"], "policy_queries": queries,
              "initial_rgb_sha256": hashlib.sha256(initial_images.tobytes()).hexdigest(), "initial_robot_state": initial_robot_state,
              "first_predicted_goal_xy": first_goal.tolist(), "first_goal_nearest_slot": int(np.argmin(np.linalg.norm(np.array(layout["xy"])-first_goal, axis=1))),
              "first_picked_body_id": first_picked_body,
              "first_picked_slot": None if first_picked_body is None else initial_meta["physical_slot_body_ids"].index(first_picked_body),
              "physical_eef_path_m": sum(row["physics_path_m"] for row in rows),
              "steps_before_final_hold": len(rows)-40 if info["success"] else None,
              "first70": first_steps_breakdown(rows), "directory": str(directory.relative_to(RUN)),
              "outcome": "success" if info["success"] else "wrong_color" if info["wrong_color_grasp"] else "hold_failure" if info["reached_height"] else "no_grasp"}
    reasons = eligibility(result, rows, env.color_body_ids[target_color])
    result.update(eligible=not reasons, contact_qualified_collision_free_success=not reasons, rejection_reasons=reasons)
    if info["success"]:
        assert [row["hold_streak"] for row in rows[-40:]] == list(range(1, 41))
        assert min(row["target_lift_m"] for row in rows[-40:]) > .04
    save_rows(directory / "steps.csv", rows)
    Image.fromarray(np.concatenate(list(initial_images), axis=1)).save(directory / "initial.png")
    Image.fromarray(np.concatenate(list(observation_images(obs)), axis=1)).save(directory / "final.png")
    with imageio.get_writer(str(directory / "rollout.mp4"), fps=cfg.hz, codec="libx264", quality=7) as writer:
        writer.append_data(np.concatenate(list(initial_images), axis=1))
        for record, row in zip(recorder.records, rows):
            frame = np.concatenate(list(record["images"]), axis=1)
            cv2.putText(frame, f'{actor} {layout["level"]} {target_color} t={row["step"]} hold={row["hold_streak"]}',
                        (2, 11), cv2.FONT_HERSHEY_SIMPLEX, .26, (0, 0, 0), 1)
            writer.append_data(frame)
    if save_data:
        np.savez_compressed(directory / "trajectory.npz", **{key: np.asarray(value) for key, value in data.items()},
                            final_images=observation_images(obs), metadata=json.dumps(result))
    (directory / "metadata.json").write_text(json.dumps(result, indent=2))
    return result
