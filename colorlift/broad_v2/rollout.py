"""Frozen ACT / old expert / raised-transport expert; training-only geometry labels."""
from dataclasses import replace
import hashlib
import json
import numpy as np
import torch
import cv2
import imageio.v2 as imageio
from PIL import Image
from .spec import RUN, VALID as V, COLORS
from ..conditional import configuration
from ..event_control import EventExecutor, FingerContactSensor, event_config
from ..event_efficiency import PhysicsRecorder, eligibility
from ..primitives import MotionDiagnostics, policy_state
from ..step_recording import save_rows
from ..vision import observation_images, observation_masks


class SafeExecutor(EventExecutor):
    """Raise in place, translate high, descend vertically; gate lift on tactile grip."""
    def __init__(self, cfg, expert):
        super().__init__(cfg, expert, replace(event_config("efficiency"), early_descent=False))
        self.route_index = 0
        self.start_xy = None
        self.transport_z = None
        self.align_retreat = False

    def action(self, obs, requested_phase, xyz, tactile):
        eef = np.asarray(obs["robot0_eef_pos"])
        if self.start_xy is None:
            self.start_xy = eef[:2].copy(); self.transport_z = max(float(eef[2]), V["safe_transport_z_m"])
        xy = np.asarray(xyz)[:2]
        if self.phase == 0:
            goals = [np.r_[self.start_xy, self.transport_z], np.r_[xy, self.transport_z], np.r_[xy, V["pregrasp_z_m"]]]
            while self.route_index < 3 and np.linalg.norm(goals[self.route_index]-eef) <= .0045:
                self.route_index += 1
            if self.route_index < 3:
                goal = goals[self.route_index]
                delta = goal-eef
                if np.linalg.norm(delta) > .035: delta *= .035/np.linalg.norm(delta)
                self.phase_steps += 1; self.last_goal = goal.copy()
                control = {"phase": 0, "requested_phase": 0, "close_steps": 0, "goal_xyz": goal,
                           "command_translation_m": delta.copy(), "transition_event": f"safe_route_{self.route_index}",
                           "phase_changed": False, "predicted_xy_error_m": float(np.linalg.norm(xy-eef[:2]))}
                grip = -1. if np.abs(obs["robot0_gripper_qpos"]).sum() < self.expert.open_gap else 0.
                return np.r_[delta/.05, np.zeros(3), grip], control
            self.phase = 1; self.phase_steps = 0; self.transitions += 1
        action, control = super().action(obs, requested_phase, xyz, tactile)
        if self.phase == 1:
            # Never correct a sizeable XY error sideways at cube height.
            error = float(np.linalg.norm(xy-eef[:2]))
            self.align_retreat |= error > .006 and eef[2] < .89
            if self.align_retreat:
                goal = np.r_[eef[:2] if eef[2] < V["pregrasp_z_m"]-.006 else xy, V["pregrasp_z_m"]]
                if eef[2] >= V["pregrasp_z_m"]-.006 and error <= .004: self.align_retreat = False
            else: goal = np.r_[xy, self.cfg.grasp_height]
            delta = goal-eef
            if np.linalg.norm(delta) > .012: delta *= .012/np.linalg.norm(delta)
            action[:3] = delta/.05
            control.update(goal_xyz=goal, command_translation_m=delta.copy())
        return action, control


def episode(env, layout, color, directory, actor, model=None, checkpoint=None, save_data=False):
    marker = directory / "metadata.json"
    if marker.exists(): return json.loads(marker.read_text())
    directory.mkdir(parents=True, exist_ok=True)
    obs = env.reset_layout(layout, color)
    meta = env.episode_metadata()
    meta.update(actor=actor, seed=layout["seed"], directory=str(directory.relative_to(RUN)), settlement=env.settlement,
                geometry_labels_training_only=True, target_pose_is_policy_input=False)
    initial = observation_images(obs)
    initial_state = policy_state(obs, 0).tolist()
    executor = SafeExecutor(env.config, configuration()[1]) if actor == "safe_expert" else EventExecutor(env.config, configuration()[1], event_config("efficiency"))
    sensor = FingerContactSensor(env)
    diagnostics = MotionDiagnostics(env.config)
    prediction = None; last_query = -1000; queries = 0; first_close_error = None
    records = {key: [] for key in ("states", "images", "masks", "actions", "servo_waypoints", "osc_actions",
                                   "goal_relative", "target_position_label", "pregrasp_position_label", "pregrasp_relative", "pregrasp_rotation_label", "pregrasp_supervision_mask")} if save_data else None
    pre_rotation = env.sim.data.site_xmat[next(iter(env.robots[0].eef_site_id.values()))].reshape(3, 3).copy()
    info = {}
    with PhysicsRecorder(env, sensor) as recorder:
        for tick in range(env.config.horizon):
            old_phase = executor.phase
            truth = env.sim.data.body_xpos[env.color_body_ids[color]].copy()
            before_eef = np.asarray(obs["robot0_eef_pos"]).copy()
            new_query = False; interval = (4, 2, 1, 4)[executor.phase]
            if actor.endswith("expert"):
                xyz, requested = truth, executor.phase
            else:
                if prediction is None or tick-last_query >= interval:
                    state = (policy_state(obs, executor.phase)[:16]-checkpoint["stats"]["state_mean"])/checkpoint["stats"]["state_std"]
                    rgb = observation_images(obs)
                    with torch.inference_mode():
                        output, _, _ = model(torch.as_tensor(state[None], device="cuda"),
                                             torch.as_tensor(rgb.transpose(0, 3, 1, 2)[None].copy(), device="cuda").float()/255,
                                             torch.tensor([COLORS.index(color)], device="cuda"))
                    prediction = output[0][0].cpu().numpy()
                    prediction[:, :3] = prediction[:, :3]*checkpoint["stats"]["xyz_std"]+checkpoint["stats"]["xyz_mean"]
                    last_query = tick; queries += 1; new_query = True
                xyz = prediction[tick-last_query, :3]; requested = int(prediction[tick-last_query, 3:].argmax())
            action, control = executor.action(obs, requested, xyz, sensor.read(obs))
            if control["phase"] == 2 and first_close_error is None: first_close_error = float(np.linalg.norm(before_eef[:2]-truth[:2]))
            if save_data:
                pregrasp = np.r_[truth[:2], V["pregrasp_z_m"]]
                macro = control["goal_xyz"].copy()
                if control["phase"] < 2: macro = np.r_[truth[:2], V["pregrasp_z_m"] if control["phase"] == 0 else env.config.grasp_height]
                values = {"states": policy_state(obs, control["phase"]), "images": observation_images(obs), "masks": observation_masks(obs),
                          "actions": np.r_[macro, np.eye(4)[control["phase"]]].astype(np.float32), "servo_waypoints": control["goal_xyz"], "osc_actions": action.copy(),
                          "goal_relative": truth-before_eef, "target_position_label": truth, "pregrasp_position_label": pregrasp,
                          "pregrasp_relative": pregrasp-before_eef, "pregrasp_rotation_label": pre_rotation,
                          "pregrasp_supervision_mask": control["phase"] < 2}
                for key, value in values.items(): records[key].append(value)
            before = obs
            obs, _, done, info = env.step(action)
            diagnostics.update(env, before, obs, control, info)
            diagnostics.rows[-1].update(transition_event=control["transition_event"], new_policy_query=new_query,
                                        execution_length=interval, hold_drift_m=info["hold_drift_m"])
            executor.after_step(obs, control)
            if old_phase != executor.phase: prediction = None
            empty = executor.phase >= 2 and executor.phase_steps >= 12 and np.abs(obs["robot0_gripper_qpos"]).sum() < .006
            if done or empty: break
    rows = recorder.enrich(diagnostics.rows, seed=layout["seed"])
    result = {**meta, **info, "control_steps": len(rows), "policy_queries": queries,
              "first_close_target_XY_error_m": first_close_error,
              "physical_eef_path_m": sum(r["physics_path_m"] for r in rows),
              "first_frame_robot_state": initial_state, "initial_rgb_sha256": hashlib.sha256(initial.tobytes()).hexdigest()}
    reasons = eligibility(result, rows, env.color_body_ids[color])
    result.update(eligible=not reasons, contact_qualified_collision_free_success=not reasons, rejection_reasons=reasons,
                  failure_category="none" if info["success"] else "wrong_color" if info["wrong_color_grasp"] else "slip_or_hold" if info["reached_height"] else "initial_alignment_or_empty_grasp")
    save_rows(directory / "steps.csv", rows)
    Image.fromarray(np.concatenate(list(initial), axis=1)).save(directory / "initial_two_views.png")
    with imageio.get_writer(str(directory / "front.mp4"), fps=env.config.hz, codec="libx264", quality=7) as writer:
        for number, image in enumerate([initial[0]]+[r["images"][0] for r in recorder.records]):
            frame = np.full((592, 512, 3), 250, np.uint8)
            frame[80:] = cv2.resize(image, (512, 512), interpolation=cv2.INTER_NEAREST)
            for y, text in zip((20, 43, 66), (f'{actor} | target {color} | front camera only', f'{layout["category"]} | seed {layout["seed"]}', f'{number}/{len(rows)} | {"PASS" if result["success"] else "FAIL"} | contact {result["eligible"]}')):
                cv2.putText(frame, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, .44, (20, 20, 20), 1)
            writer.append_data(frame)
    if save_data:
        np.savez_compressed(directory / "trajectory.npz", **{k: np.asarray(v) for k, v in records.items()}, metadata=json.dumps(result))
    marker.write_text(json.dumps(result, indent=2))
    return result
