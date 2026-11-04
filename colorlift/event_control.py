"""Event-driven OSC stages and color-blind, force-qualified finger contact sensing."""
from dataclasses import dataclass, asdict
import numpy as np
import mujoco


@dataclass
class EventConfig:
    xy_tolerance_m: float = .0045
    z_tolerance_m: float = .005
    contact_force_n: float = .2
    maximum_finger_speed_m_s: float = .02
    minimum_occupied_gap_m: float = .012
    maximum_occupied_gap_m: float = .075
    early_descent: bool = False
    early_descent_xy_m: float = .008
    early_descent_height_m: float = .890


class FingerContactSensor:
    """Simulated tactile sensor; never reads color IDs, target poses or task scores.

    Both pads must push the same external dynamic body. Body equality associates
    the two contacts; its identity is never interpreted as a color by control.
    Static table contacts and robot/self contacts cannot qualify as a grasp.
    """
    def __init__(self, env, config=None):
        self.env, self.cfg = env, config or EventConfig()
        gripper = env.robots[0].gripper
        self.gripper = next(iter(gripper.values())) if isinstance(gripper, dict) else gripper
        self.sides = [{env.sim.model.geom_name2id(name) for name in self.gripper.important_geoms[side]}
                      for side in ("left_finger", "right_finger")]

    def read(self, obs):
        env, cfg = self.env, self.cfg
        forces = [{}, {}]
        wrench = np.zeros(6)
        for index, contact in enumerate(env.sim.data.contact):
            a, b = int(contact.geom1), int(contact.geom2)
            for pad, other in ((a, b), (b, a)):
                if other in env.robot_geom_ids:
                    continue
                body = int(env.sim.model.geom_bodyid[other])
                if int(env.sim.model.body_dofnum[body]) <= 0:
                    continue
                for side, ids in enumerate(self.sides):
                    if pad not in ids:
                        continue
                    mujoco.mj_contactForce(env.sim.model._model, env.sim.data._data, index, wrench)
                    forces[side][body] = forces[side].get(body, 0.) + abs(float(wrench[0]))
        common = set(forces[0]) & set(forces[1])
        body = max(common, key=lambda b: min(forces[0][b], forces[1][b])) if common else -1
        left, right = (forces[0][body], forces[1][body]) if body >= 0 else (0., 0.)
        gap = float(np.abs(obs["robot0_gripper_qpos"]).sum())
        speed = max(abs(float(env.sim.data.get_joint_qvel(name))) for name in self.gripper.joints)
        bilateral = body >= 0 and min(left, right) >= cfg.contact_force_n
        occupied = cfg.minimum_occupied_gap_m <= gap <= cfg.maximum_occupied_gap_m
        settled = speed <= cfg.maximum_finger_speed_m_s
        return {"tactile_body_id": body, "left_pad_force_n": left, "right_pad_force_n": right,
                "tactile_bilateral": bool(bilateral), "occupied_gap": bool(occupied),
                "finger_speed_m_s": speed, "fingers_settled": bool(settled),
                "grip_confirmed": bool(bilateral and occupied and settled)}


class EventExecutor:
    """No minimum stage dwell: pose events advance reach; tactile events start lift.

    The ACT supplies XY only. Fixed table heights and the existing clearance
    recovery gate are retained. No target ground truth or HSV enters this class.
    """
    def __init__(self, config, expert, event=None):
        self.cfg, self.expert, self.event = config, expert, event or EventConfig()
        self.phase = self.phase_steps = self.close_steps = self.transitions = 0
        self.filtered_xy = self.grasp_xy = self.last_error = None
        self.recovering_alignment = False
        self.last_goal = None

    def action(self, obs, requested_phase, xyz, tactile):
        cfg, e, p = self.cfg, self.expert, self.event
        eef = np.asarray(obs["robot0_eef_pos"])
        fingers = np.asarray(obs["robot0_gripper_qpos"])
        xy = np.asarray(xyz[:2], dtype=float).clip(-.15, .15)
        self.filtered_xy = xy if self.filtered_xy is None else .6*xy + .4*self.filtered_xy
        xy_error = float(np.linalg.norm(eef[:2] - self.filtered_xy))
        phase_before = self.phase
        transition = "none"
        if self.phase == 0:
            pregrasp_z = cfg.table_height + cfg.cube_size/2 + e.pregrasp_clearance
            reached = xy_error <= p.xy_tolerance_m and abs(eef[2]-pregrasp_z) <= p.z_tolerance_m
            flowing = (p.early_descent and xy_error <= p.early_descent_xy_m
                       and cfg.grasp_height+.02 <= eef[2] <= p.early_descent_height_m)
            if reached or flowing:
                self.phase = 1
                transition = "horizontal_alignment_early_descent" if flowing and not reached else "pregrasp_pose_reached"
        elif self.phase == 1:
            reached = xy_error <= p.xy_tolerance_m and abs(eef[2]-cfg.grasp_height) <= p.z_tolerance_m
            if reached and not self.recovering_alignment:
                self.phase = 2
                self.grasp_xy = eef[:2].copy()
                transition = "grasp_pose_reached"
        elif self.phase == 2 and tactile["grip_confirmed"]:
            self.phase = 3
            transition = "bilateral_force_occupied_settled"
        if self.phase != phase_before:
            self.phase_steps = 0
            self.transitions += 1
        goal = np.r_[self.filtered_xy, cfg.grasp_height]
        if self.phase == 0:
            goal[2] = cfg.table_height + cfg.cube_size/2 + e.pregrasp_clearance
        elif self.phase == 1:
            if xy_error > .012:
                self.recovering_alignment = True
            if self.recovering_alignment:
                if eef[2] < .870:
                    goal[:2] = eef[:2]
                elif xy_error <= .004:
                    self.recovering_alignment = False
                goal[2] = .875 if self.recovering_alignment else cfg.grasp_height
        if self.phase >= 2:
            goal[:2] = self.grasp_xy
            self.close_steps += 1
        if self.phase == 3:
            goal[2] = cfg.table_height + cfg.cube_size/2 + cfg.lift_threshold + e.lift_margin
        delta = goal-eef
        delta[np.abs(delta) < e.deadband] = 0
        maximum = e.align_maximum_step if self.phase == 1 else e.maximum_step
        if np.linalg.norm(delta) > maximum:
            delta *= maximum/np.linalg.norm(delta)
        grip = (-1. if np.abs(fingers).sum() < e.open_gap else 0.) if self.phase < 2 else 1.
        self.phase_steps += 1
        self.last_goal = goal.copy()
        control = {"phase": self.phase, "requested_phase": int(requested_phase), "close_steps": self.close_steps,
                   "goal_xyz": goal, "command_translation_m": delta.copy(), "transition_event": transition,
                   "phase_changed": self.phase != phase_before, "predicted_xy_error_m": xy_error}
        return np.r_[delta/.05, np.zeros(3), grip], control

    def after_step(self, obs, control):
        self.last_error = np.asarray(obs["robot0_eef_pos"])-control["goal_xyz"]


SCHEMES = {
    "baseline": {"schedule": [2, 2, 2, 2], "event_driven": False},
    "single_step": {"schedule": [1, 1, 1, 1], "event_driven": True},
    "hybrid": {"schedule": [4, 1, 1, 4], "event_driven": True},
    "efficiency": {"schedule": [4, 2, 1, 4], "event_driven": True, "early_descent": True},
}


def event_config(scheme):
    return EventConfig(early_descent=SCHEMES[scheme].get("early_descent", False))


def protocol():
    return {"schemes": SCHEMES, "event_thresholds": asdict(EventConfig()),
            "ACT_checkpoint": "experiments/colorlift_recovery_v1/model/policy.pt",
            "model_capacity": 4, "teacher": "two-camera HSV localization; no object pose in control",
            "student": "learned RGB + color embedding + proprioception; no HSV or target pose in control",
            "tactile": "simulated color-blind bilateral dynamic-body contacts; only event variants use tactile gating",
            "control_hz": 20, "success": "original >0.04m + stable2s + correct grasp; unchanged",
            "dev_seeds": list(range(84000, 84008)), "test_seeds": list(range(85000, 85050)),
            "data_train_seeds": list(range(86000, 86060)), "data_val_seeds": list(range(87000, 87010)),
            "dataset_eligibility": "true original success, collision-free, force-qualified correct target before first lift; identical-action MuJoCo replay must also pass"}
