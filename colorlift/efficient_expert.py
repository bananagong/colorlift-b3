"""Visual efficient red-cube expert, preserving the original ColorLift scoring rules."""
from dataclasses import dataclass, asdict
import numpy as np
from .config import Config
from .vision import locate_on_table, observation_images, observation_masks
from .primitives import policy_state, MotionDiagnostics


@dataclass
class ExpertConfig:
    kp: float = 210.0
    pregrasp_clearance: float = 0.065
    lift_margin: float = 0.015
    open_gap: float = 0.058
    maximum_step: float = 0.05
    align_maximum_step: float = 0.045
    minimum_close_steps: int = 3
    stationary_finger_steps: int = 2
    finger_motion_tolerance: float = 0.0003
    xyz_tolerance: float = 0.005
    xy_tolerance: float = 0.0045
    deadband: float = 0.0007
    visual_refine: bool = False


def task_config(base, expert):
    from dataclasses import replace
    # Only controller gains change. Geometry, randomization and scoring stay identical.
    return replace(base, osc_kp=expert.kp)


def rollout(env, seed, expert=None, record=True):
    expert = expert or ExpertConfig()
    cfg = env.config
    obs = env.reset_seed(seed)
    metadata = env.episode_metadata()
    estimate, visual_details = locate_on_table(env, obs)
    initial_estimate = estimate.copy()
    initial_truth = env.sim.data.body_xpos[env.color_body_ids[env.target_color]].copy()
    metadata.update(seed=seed, expert_config=asdict(expert), localization=visual_details,
                    hsv_localization_error_m=float(np.linalg.norm(estimate[:2] - initial_truth[:2])))
    phase = 0
    phase_steps = 0
    close_steps = 0
    stationary_steps = 0
    closed_xy = None
    lift_lock = None
    diagnostics = MotionDiagnostics(cfg)
    history = {key: [] for key in ("states", "images", "masks", "actions", "osc_actions")}
    info, aborted = {}, False
    for _ in range(cfg.horizon):
        before = obs
        phase_before = phase
        eef = np.asarray(obs["robot0_eef_pos"])
        if phase == 0:
            goal = np.r_[estimate[:2], estimate[2] + expert.pregrasp_clearance]
        elif phase == 1:
            goal = np.r_[estimate[:2], cfg.grasp_height]
        elif phase == 2:
            goal = np.r_[closed_xy, cfg.grasp_height]
        else:
            goal = np.r_[closed_xy, initial_estimate[2] + cfg.lift_threshold + expert.lift_margin]
            if lift_lock is not None:
                goal = lift_lock.copy()
        delta = goal - eef
        delta[np.abs(delta) < expert.deadband] = 0
        maximum = expert.align_maximum_step if phase == 1 else expert.maximum_step
        norm = np.linalg.norm(delta)
        if norm > maximum:
            delta *= maximum / norm
        gap = float(np.abs(obs["robot0_gripper_qpos"]).sum())
        grip = (-1.0 if gap < expert.open_gap else 0.0) if phase < 2 else 1.0
        action = np.r_[delta / 0.05, np.zeros(3), grip]
        if record:
            history["states"].append(policy_state(obs, phase))
            history["images"].append(observation_images(obs))
            history["masks"].append(observation_masks(obs))
            history["actions"].append(np.r_[goal, np.eye(4)[phase]].astype(np.float32))
            history["osc_actions"].append(action.astype(np.float32))
        if phase >= 2:
            close_steps += 1
        phase_steps += 1
        obs, _, done, info = env.step(action)
        control = {"phase": phase, "requested_phase": phase, "close_steps": close_steps,
                   "goal_xyz": goal.copy(), "command_translation_m": delta.copy()}
        diagnostics.update(env, before, obs, control, info)
        row = diagnostics.rows[-1]
        for i, axis in enumerate("xyz"):
            row["visual_target_" + axis] = float(estimate[i])
        row["vision_source"] = "initial_two_camera_HSV"
        stationary = np.linalg.norm(obs["robot0_gripper_qpos"] - before["robot0_gripper_qpos"]) < expert.finger_motion_tolerance
        stationary_steps = stationary_steps + 1 if stationary else 0
        error = np.asarray(obs["robot0_eef_pos"]) - goal
        if phase == 0 and np.linalg.norm(error[:2]) <= expert.xy_tolerance and abs(error[2]) <= expert.xyz_tolerance:
            phase, phase_steps = 1, 0
        elif phase == 1 and np.linalg.norm(error[:2]) <= expert.xy_tolerance and abs(error[2]) <= expert.xyz_tolerance:
            phase, phase_steps = 2, 0
            closed_xy = np.asarray(obs["robot0_eef_pos"][:2]).copy()
            stationary_steps = 0
        elif phase == 2 and close_steps >= expert.minimum_close_steps and stationary_steps >= expert.stationary_finger_steps:
            phase, phase_steps = 3, 0
            closed_xy = np.asarray(obs["robot0_eef_pos"][:2]).copy()
        elif phase == 3 and lift_lock is None and np.linalg.norm(error) <= .003:
            lift_lock = goal.copy()
        if done:
            break
        if phase == 3 and close_steps >= 12 and float(np.abs(obs["robot0_gripper_qpos"]).sum()) < .006:
            aborted = True
            break
    outcome = "success" if info.get("success") else "wrong_color" if info.get("wrong_color_grasp") else "hold_failure" if info.get("reached_height") else "empty_grasp" if aborted else "no_lift"
    result = {**metadata, **info, **diagnostics.summary(), "outcome": outcome,
              "macro_actions": max(int(row["phase"]) for row in diagnostics.rows) + 1,
              "aborted_empty_grasp": aborted, "controller": "visual_efficient_expert"}
    if record:
        history = {key: np.asarray(value) for key, value in history.items()}
        # Include the true final observation in the video / observation sequence.
        history["final_images"] = observation_images(obs)
        history["final_masks"] = observation_masks(obs)
    return result, history, diagnostics.rows
