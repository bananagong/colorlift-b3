"""Four monotone OSC primitives and diagnostics that never guide actions with object truth."""
from collections import deque
import numpy as np
from .config import COLORS, PHASES


def policy_state(obs, phase):
    return np.r_[obs["robot0_joint_pos"], obs["robot0_gripper_qpos"], obs["robot0_eef_pos"],
                 np.eye(4)[phase], obs["target_color_onehot"]].astype(np.float32)


class PrimitiveExecutor:
    """Convert (phase, XYZ) to bounded OSC_POSE increments with explicit primitive semantics.

    Uses only robot proprioception and fixed workspace geometry. It does not read
    target body IDs, object positions, HSV centroids, or success scores for action selection.
    Physical control steps remain visible rather than being hidden inside four macro calls.
    """
    def __init__(self, config):
        self.config = config
        self.phase = 0
        self.phase_steps = 0
        self.close_steps = 0
        self.transitions = 0
        self.rejected_phase_requests = 0
        self._confirm = 0
        self._pending = 0
        self._filtered = None
        self._grasp_xy = None
        self._lift_lock = None
        self.empty_wait = 0

    def action(self, obs, requested_phase, xyz, *, new_query=True, force_transition=False):
        cfg = self.config
        requested_phase = int(np.clip(requested_phase, 0, 3))
        eef = np.asarray(obs["robot0_eef_pos"])
        if requested_phase < self.phase or requested_phase > self.phase + 1:
            self.rejected_phase_requests += 1
        if force_transition:
            next_phase = min(requested_phase, self.phase + 1)
        else:
            next_phase = self.phase
            if new_query:
                pending = min(requested_phase, self.phase + 1)
                self._confirm = self._confirm + 1 if pending == self._pending else 1
                self._pending = pending
                if pending > self.phase and self._confirm >= cfg.phase_confirm_queries:
                    next_phase = pending
        if self.phase == 2 and self.close_steps < cfg.minimum_close_steps:
            next_phase = 2
        if not force_transition and cfg.auto_advance_primitives:
            if self.phase in (0, 1) and self._filtered is not None and self.phase_steps >= cfg.minimum_reach_steps and np.linalg.norm(eef - self._filtered) < cfg.goal_reached_tolerance:
                next_phase = self.phase + 1
            if self.phase == 2 and self.close_steps >= cfg.minimum_close_steps:
                fingers = np.asarray(obs["robot0_gripper_qpos"])
                if hasattr(self, "_last_fingers") and np.linalg.norm(fingers - self._last_fingers) < 0.0002:
                    next_phase = 3
        self._last_fingers = np.asarray(obs["robot0_gripper_qpos"]).copy()
        if not force_transition and self.phase in (0, 1) and next_phase > self.phase:
            if self._filtered is None or np.linalg.norm(eef - self._filtered) > cfg.goal_reached_tolerance * 1.5 or self.phase_steps < 4:
                next_phase = self.phase
        if next_phase > self.phase:
            self.phase = next_phase
            self.transitions += 1
            self.phase_steps = 0
            self._filtered = None
            if self.phase == 2:
                self._grasp_xy = eef[:2].copy()
        goal = np.asarray(xyz, dtype=float).copy()
        goal[:2] = goal[:2].clip(-0.18, 0.18)
        if self.phase == 0:
            goal[2] = cfg.approach_height
            maximum = cfg.max_step_approach
        elif self.phase == 1:
            goal[2] = cfg.grasp_height
            maximum = cfg.max_step_align
        elif self.phase == 2:
            goal[:2] = self._grasp_xy
            goal[2] = cfg.grasp_height
            maximum = cfg.max_step_align
            self.close_steps += 1
        else:
            goal[:2] = self._grasp_xy
            goal[2] = float(np.clip(goal[2], cfg.table_height + cfg.lift_goal_min_offset, cfg.table_height + cfg.lift_goal_max_offset))
            maximum = cfg.max_step_lift
            self.close_steps += 1
        if self._filtered is None:
            self._filtered = goal.copy()
        else:
            self._filtered = cfg.target_smoothing * goal + (1 - cfg.target_smoothing) * self._filtered
        if self.phase == 3:
            if self._lift_lock is None and np.linalg.norm(eef - self._filtered) < cfg.goal_reached_tolerance:
                self._lift_lock = self._filtered.copy()
            if self._lift_lock is not None:
                self._filtered = self._lift_lock.copy()
        delta = self._filtered - eef
        delta[np.abs(delta) < cfg.position_deadband] = 0
        norm = np.linalg.norm(delta)
        if norm > maximum:
            delta *= maximum / norm
        grip = 1.0 if self.phase >= 2 else -1.0
        self.phase_steps += 1
        command = np.r_[delta / 0.05, np.zeros(3), grip]
        return command, {"phase": self.phase, "phase_name": PHASES[self.phase], "requested_phase": requested_phase,
                         "phase_steps": self.phase_steps, "close_steps": self.close_steps,
                         "goal_xyz": self._filtered.copy(), "command_translation_m": delta.copy()}

    def should_abort_empty(self, obs):
        gap = float(np.abs(obs["robot0_gripper_qpos"]).sum())
        empty = self.phase == 3 and self.close_steps >= self.config.minimum_close_steps and gap < self.config.empty_gripper_gap
        self.empty_wait = self.empty_wait + 1 if empty else 0
        return self.config.empty_grasp_abort_steps > 0 and self.empty_wait >= self.config.empty_grasp_abort_steps


class MotionDiagnostics:
    def __init__(self, config):
        self.config = config
        self.counts = {key: 0 for key in (
            "command_reversals_all", "command_reversals_within_phase", "motion_reversals_all",
            "motion_reversals_within_phase", "post_close_idle_steps", "empty_gripper_wait_steps",
            "no_progress_steps", "valid_hold_steps", "hold_attempt_steps",
        )}
        self.phase_counts = {name: 0 for name in PHASES}
        self._vectors = {}
        self._distances = deque(maxlen=config.progress_window + 1)
        self._last_phase = None
        self.rows = []

    def _direction(self, kind, vector, phase, tick):
        cfg = self.config
        norm = np.linalg.norm(vector)
        if norm < cfg.motion_epsilon:
            return
        if kind in self._vectors:
            previous, previous_phase, previous_tick = self._vectors[kind]
            cosine = np.dot(previous, vector) / (np.linalg.norm(previous) * norm)
            if tick - previous_tick <= cfg.reversal_max_gap_steps and cosine < np.cos(np.deg2rad(cfg.reversal_angle_deg)):
                self.counts[kind + "_reversals_all"] += 1
                self.counts[kind + "_reversals_within_phase"] += previous_phase == phase
        self._vectors[kind] = (vector.copy(), phase, tick)

    def update(self, env, previous_obs, obs, control, info):
        cfg = self.config
        phase, tick = control["phase"], info["control_steps"]
        displacement = obs["robot0_eef_pos"] - previous_obs["robot0_eef_pos"]
        self._direction("command", control["command_translation_m"], phase, tick)
        self._direction("motion", displacement, phase, tick)
        target = env.sim.data.body_xpos[env.color_body_ids[env.target_color]]
        distance = float(np.linalg.norm(obs["robot0_eef_pos"] - target))
        if phase != self._last_phase:
            self._distances.clear()
        self._last_phase = phase
        self._distances.append(distance)
        improvement = self._distances[0] - distance
        plateau = phase in (0, 1) and len(self._distances) == cfg.progress_window + 1 and distance > cfg.progress_distance_tolerance and improvement < cfg.progress_epsilon
        valid_hold = info["hold_streak"] > 0
        stationary_gripper = np.linalg.norm(obs["robot0_gripper_qpos"] - previous_obs["robot0_gripper_qpos"]) < 0.0002
        idle = phase >= 2 and control["close_steps"] >= cfg.minimum_close_steps and stationary_gripper and np.linalg.norm(displacement) < cfg.idle_motion_threshold and not valid_hold
        gap = float(np.abs(obs["robot0_gripper_qpos"]).sum())
        self.counts["post_close_idle_steps"] += bool(idle)
        self.counts["empty_gripper_wait_steps"] += bool(idle and gap < cfg.empty_gripper_gap)
        self.counts["no_progress_steps"] += bool(plateau)
        self.counts["valid_hold_steps"] += bool(valid_hold)
        self.counts["hold_attempt_steps"] += bool(phase == 3 and info["height_ok"] and info["recent_target_grasp"])
        self.phase_counts[PHASES[phase]] += 1
        row = {"step": tick, "phase": phase, "phase_name": PHASES[phase],
               "requested_phase": control["requested_phase"],
               "eef_target_distance_m": distance, "distance_improvement_window_m": float(improvement),
               "eef_move_m": float(np.linalg.norm(displacement)), "gripper_gap_m": gap,
               "target_lift_m": info["lift_m"], "target_linear_speed": info["linear_speed_m_s"],
               "target_angular_speed": info["angular_speed_rad_s"], "hold_streak": info["hold_streak"],
               "target_grasped": info["target_grasped"], "wrong_color_grasp": info["wrong_color_grasp"],
               "collision": bool(info["collisions"]), "post_close_idle": bool(idle), "no_progress": bool(plateau)}
        for i, axis in enumerate("xyz"):
            row["eef_" + axis] = float(obs["robot0_eef_pos"][i])
            row["target_" + axis] = float(target[i])
            row["goal_" + axis] = float(control["goal_xyz"][i])
            row["command_" + axis] = float(control["command_translation_m"][i])
        self.rows.append(row)

    def summary(self):
        return {**self.counts, "phase_control_steps": self.phase_counts,
                "non_hold_control_steps": len(self.rows) - self.counts["valid_hold_steps"]}


def step_primitive(env, executor, obs, primitive, goal_xyz, max_steps=40):
    """Public macro-action helper; returns all low-level transitions and true step count."""
    transitions = []
    for _ in range(max_steps):
        action, control = executor.action(obs, primitive, goal_xyz, force_transition=True)
        next_obs, reward, done, info = env.step(action)
        transitions.append((obs, next_obs, control, info))
        stationary_gripper = np.linalg.norm(next_obs["robot0_gripper_qpos"] - obs["robot0_gripper_qpos"]) < 0.0002
        obs = next_obs
        if done or (primitive < 2 and np.linalg.norm(obs["robot0_eef_pos"] - control["goal_xyz"]) < env.config.goal_reached_tolerance):
            break
        if primitive == 2 and executor.close_steps >= env.config.minimum_close_steps and stationary_gripper:
            break
        if executor.should_abort_empty(obs):
            info["aborted_empty_grasp"] = True
            break
    return obs, transitions
