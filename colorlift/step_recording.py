"""Non-invasive, explicit pre/action/post telemetry for any ColorLift controller."""
from pathlib import Path
import csv
import json
import numpy as np


class StepRecorder:
    """Wrap env.step without changing commands or simulation state.

    Target ground truth is diagnostic only. The controller supplies the action
    and keeps its own separate visual estimate / waypoint.
    """
    def __init__(self, env, capture_images=False):
        self.env = env
        self.capture_images = capture_images
        self.initial_images = None
        self.initial_masks = None
        self.records = []
        self.site_id = next(iter(env.robots[0].eef_site_id.values()))
        gripper = env.robots[0].gripper
        self.gripper = next(iter(gripper.values())) if isinstance(gripper, dict) else gripper

    def snapshot(self):
        env = self.env
        body = env.color_body_ids[env.target_color]
        return {
            "time": float(env.sim.data.time),
            "eef": env.sim.data.site_xpos[self.site_id].copy(),
            "target": env.sim.data.body_xpos[body].copy(),
            "finger_qpos": np.array([env.sim.data.get_joint_qpos(name) for name in self.gripper.joints]),
            "finger_qvel": np.array([env.sim.data.get_joint_qvel(name) for name in self.gripper.joints]),
            "actuator_ctrl": env.sim.data.ctrl.copy(),
            "joints": env.robots[0]._joint_positions.copy(),
        }

    def step(self, action):
        if self.capture_images and self.initial_images is None:
            from .vision import observation_images, observation_masks
            observation = self.env._get_observations()
            self.initial_images = observation_images(observation)
            self.initial_masks = observation_masks(observation)
        before = self.snapshot()
        action = np.asarray(action, dtype=float).copy()
        result = self.original_step(action)
        after = self.snapshot()
        record = {"before": before, "after": after, "action": action,
                  "gripper_integrator": self.gripper.current_action.copy()}
        self.last_observation = result[0]
        if self.capture_images:
            from .vision import observation_images, observation_masks
            record["images"] = observation_images(result[0])
            record["masks"] = observation_masks(result[0])
        self.records.append(record)
        return result

    def __enter__(self):
        self.original_step = self.env.step
        self.env.step = self.step
        return self

    def __exit__(self, *args):
        self.env.step = self.original_step

    def enrich(self, diagnostic_rows, *, seed, visual_xyz=None):
        assert len(diagnostic_rows) == len(self.records)
        rows = []
        dt = 1 / self.env.config.hz
        for diagnostic, record in zip(diagnostic_rows, self.records):
            before, after, action = record["before"], record["after"], record["action"]
            row = dict(diagnostic)
            row.update(seed=seed, time_before_s=before["time"], time_after_s=after["time"],
                       target_body_id=self.env.color_body_ids[self.env.target_color],
                       gripper_command=float(action[-1]),
                       gripper_mode="open" if action[-1] < 0 else "close" if action[-1] > 0 else "hold_aperture")
            for i, axis in enumerate("xyz"):
                row["eef_before_" + axis] = float(before["eef"][i])
                row["eef_after_" + axis] = float(after["eef"][i])
                row["eef_velocity_" + axis] = float((after["eef"][i] - before["eef"][i]) / dt)
                row["target_before_" + axis] = float(before["target"][i])
                row["target_after_" + axis] = float(after["target"][i])
                if visual_xyz is not None:
                    row.setdefault("visual_target_" + axis, float(visual_xyz[i]))
            for i in range(2):
                for key in ("finger_qpos", "finger_qvel"):
                    row[f"{key}_{i}_before"] = float(before[key][i])
                    row[f"{key}_{i}_after"] = float(after[key][i])
                row[f"gripper_integrator_{i}"] = float(record["gripper_integrator"][i])
            for i, value in enumerate(action):
                row[f"osc_action_{i}"] = float(value)
            for i, value in enumerate(after["actuator_ctrl"]):
                row[f"actuator_ctrl_{i}"] = float(value)
            for i, value in enumerate(after["joints"]):
                row[f"robot_joint_{i}"] = float(value)
            rows.append(row)
        return rows

    def schema(self):
        return {
            "sample_hz": self.env.config.hz,
            "positions": "world frame, meters; before/after are control-interval boundaries",
            "target": "actual red body position is for diagnostics only; visual estimate and commanded goal are separate",
            "osc_action_order": ["dx_normalized", "dy_normalized", "dz_normalized", "rx", "ry", "rz", "gripper"],
            "osc_translation_scale_m": 0.05,
            "gripper_command": "-1 open, 0 hold aperture, +1 close",
            "actuator_order": list(self.env.sim.model.actuator_names),
            "actuator_units": "7 Panda arm torque controls (Nm), then two finger position controls (m)",
            "gripper_joint_names": list(self.gripper.joints),
        }


def save_rows(path, rows):
    path = Path(path)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def primary_bucket(row):
    if int(row["hold_streak"]) > 0:
        return "stable_hold"
    return ("approach", "align", "close", "lift_and_settle")[int(row["phase"])]


def first_steps_breakdown(rows, limit=70):
    selected = rows[:limit]
    names = ["approach", "align", "close", "lift_and_settle", "stable_hold"]
    counts = {name: sum(primary_bucket(row) == name for row in selected) for name in names}
    segments = []
    for row in selected:
        bucket = primary_bucket(row)
        step = int(row["step"])
        if segments and segments[-1]["bucket"] == bucket:
            segments[-1]["end_step"] = step
            segments[-1]["steps"] += 1
        else:
            segments.append({"bucket": bucket, "start_step": step, "end_step": step, "steps": 1})
    flag = lambda value: str(value).lower() in ("true", "1")
    result = {"requested_steps": limit, "observed_steps": len(selected), "composition": counts,
              "segments": segments, "post_close_idle_steps": sum(flag(r["post_close_idle"]) for r in selected),
              "no_progress_steps": sum(flag(r["no_progress"]) for r in selected),
              "hold_streak_at_end": int(selected[-1]["hold_streak"]) if selected else 0}
    assert sum(counts.values()) == len(selected)
    return result
