"""robosuite ColorLift: three randomized cubes, target-only grasp and stable hold scoring."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from collections import OrderedDict
import numpy as np
import mujoco
from robosuite.environments.manipulation.lift import Lift
from robosuite.controllers import load_composite_controller_config
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from catlab_assets import CubeObject
from lift_act.env import camera_quat
from .config import CAMERAS, COLORS, Config

RGBA = {"red": (0.88, 0.04, 0.04, 1), "yellow": (0.92, 0.78, 0.02, 1), "blue": (0.035, 0.12, 0.88, 1)}


class ColorSampler:
    """Rejection sampler with a conservative footprint separation and shuffled color order."""
    def __init__(self, objects, config, rng):
        self.objects, self.config, self.rng = objects, config, rng

    def sample(self):
        result, positions = {}, []
        cfg = self.config
        minimum = max(cfg.min_center_distance, np.sqrt(2) * cfg.cube_size + 0.005)
        for index in self.rng.permutation(len(self.objects)):
            obj = self.objects[index]
            for _ in range(2000):
                xy = self.rng.uniform(-cfg.xy_limit, cfg.xy_limit, size=2)
                if all(np.linalg.norm(xy - other) >= minimum for other in positions):
                    break
            else:
                raise RuntimeError("Could not place three non-overlapping cubes; increase workspace")
            positions.append(xy)
            yaw = self.rng.uniform(-cfg.yaw_limit, cfg.yaw_limit)
            pos = np.r_[xy, cfg.table_height + cfg.cube_size / 2]
            result[obj.name] = (pos, np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]), obj)
        return result

    def reset(self):
        self.objects = []

    def add_objects(self, objects):
        self.objects.extend(objects if isinstance(objects, (list, tuple)) else [objects])


class ColorLift(Lift):
    """RGB/proprioception observations; ground-truth object state stays in evaluation metadata.

    Standard step() accepts robosuite's OSC_POSE action. PrimitiveExecutor supplies
    the four-stage action interface. Target-only success also requires bilateral
    finger contact and a continuous stable hold, measured relative to reset height.
    """
    def __init__(self, robots="Panda", *, config=None, target_color=None, **kwargs):
        self.config = config or Config()
        if target_color is not None:
            from dataclasses import replace
            self.config = replace(self.config, target_color=target_color)
        self.target_color = self.config.target_color
        self._capture_contacts = False
        self._episode = {"success": False}
        kwargs.setdefault("use_object_obs", False)
        kwargs.setdefault("control_freq", self.config.hz)
        kwargs.setdefault("horizon", self.config.horizon)
        kwargs.setdefault("camera_names", list(CAMERAS))
        kwargs.setdefault("camera_heights", self.config.image_size)
        kwargs.setdefault("camera_widths", self.config.image_size)
        kwargs.setdefault("initialization_noise", None)
        kwargs.setdefault("table_friction", (0.8, 0.005, 0.0001))
        controller = load_composite_controller_config(robot="Panda")
        parts = controller["body_parts"]
        arm = parts.get("arms", parts)["right"]
        arm["kp"] = self.config.osc_kp
        arm["damping_ratio"] = self.config.osc_damping_ratio
        kwargs.setdefault("controller_configs", controller)
        super().__init__(robots=robots, **kwargs)

    def _load_model(self):
        self.table_offset[2] = self.config.table_height
        super()._load_model()
        arena = self.model.mujoco_arena
        for name, pos in zip(CAMERAS, [[0, -0.52, 1.23], [0.52, 0.02, 1.23]]):
            arena.set_camera(name, pos=pos, quat=camera_quat(pos, [0, 0, 0.85]), camera_attribs={"fovy": "44"})
        self.color_objects = {color: CubeObject(color + "_cube", size=self.config.cube_size,
                              mass=self.config.cube_mass, rgba=RGBA[color]) for color in COLORS}
        self.cube = self.color_objects[self.target_color]
        self.placement_initializer = ColorSampler(list(self.color_objects.values()), self.config, self.rng)
        self.model = ManipulationTask(arena, self.model.mujoco_robots, list(self.color_objects.values()))

    def _setup_references(self):
        super()._setup_references()
        self.color_body_ids = {color: self.sim.model.body_name2id(obj.root_body) for color, obj in self.color_objects.items()}
        self.color_geom_ids = {color: {self.sim.model.geom_name2id(name) for name in obj.contact_geoms}
                               for color, obj in self.color_objects.items()}
        self.color_visual_geom_ids = {color: {self.sim.model.geom_name2id(name) for name in obj.visual_geoms}
                                      for color, obj in self.color_objects.items()}
        self.robot_geom_ids = {self.sim.model.geom_name2id(name) for name in self.robots[0].robot_model.contact_geoms}
        grippers = self.robots[0].gripper
        grippers = list(grippers.values()) if isinstance(grippers, dict) else [grippers]
        self.finger_geom_ids = set()
        self.gripper_geom_ids = set()
        for gripper in grippers:
            self.gripper_geom_ids.update(self.sim.model.geom_name2id(name) for name in gripper.contact_geoms)
            for side in ("left_finger", "right_finger"):
                self.finger_geom_ids.update(self.sim.model.geom_name2id(name) for name in gripper.important_geoms[side])
        self.robot_geom_ids |= self.gripper_geom_ids
        self.table_geom_id = self.sim.model.geom_name2id("table_collision")

    def _setup_observables(self):
        observables = super()._setup_observables()
        @sensor(modality="task")
        def target_color_onehot(obs_cache):
            return np.eye(3, dtype=np.float32)[COLORS.index(self.target_color)]
        observables["target_color_onehot"] = Observable("target_color_onehot", target_color_onehot, sampling_rate=self.control_freq)
        return observables

    def _reset_internal(self):
        super()._reset_internal()
        self.sim.forward()
        self.initial_positions = {color: self.sim.data.body_xpos[bid].copy() for color, bid in self.color_body_ids.items()}
        self.initial_heights = {color: pos[2] for color, pos in self.initial_positions.items()}
        self.assert_no_overlap()
        self._last_grasp_step = {color: -1000 for color in COLORS}
        self._wrong_grasp_streak = {color: 0 for color in COLORS}
        self._hold_anchor = None
        self._episode = {
            "success": False, "wrong_color_grasp": False, "wrong_colors": [],
            "control_steps": 0, "hold_streak": 0, "max_hold_streak": 0,
            "holding_failure_events": 0, "reached_height": False,
            "collision_steps": 0, "collision_events": 0, "max_collision_force_n": 0.0,
            "collision_physics_steps": 0, "collision_types": [],
            "max_lift_m": 0.0, "target_grasped_ever": False,
        }
        self._last_collision = False

    def assert_no_overlap(self):
        minimum = np.sqrt(2) * self.config.cube_size
        for i, a in enumerate(COLORS):
            for b in COLORS[i + 1:]:
                distance = np.linalg.norm(self.sim.data.body_xpos[self.color_body_ids[a], :2] - self.sim.data.body_xpos[self.color_body_ids[b], :2])
                if distance <= minimum:
                    raise AssertionError(f"Overlapping footprint bounds: {a}, {b}, distance={distance}")

    def _contacts(self):
        """Count force-qualified unwanted contacts, excluding target-finger grasp and resting cubes."""
        categories, peak = set(), 0.0
        force = np.zeros(6)
        for index, contact in enumerate(self.sim.data.contact):
            a, b = int(contact.geom1), int(contact.geom2)
            category = None
            for robot_geom, other in ((a, b), (b, a)):
                if robot_geom not in self.robot_geom_ids:
                    continue
                if other == self.table_geom_id:
                    category = "robot_table"
                for color, geoms in self.color_geom_ids.items():
                    if other in geoms:
                        if color != self.target_color:
                            category = "robot_distractor"
                        elif robot_geom not in self.finger_geom_ids:
                            category = "nonfinger_target"
            object_a = next((c for c, ids in self.color_geom_ids.items() if a in ids), None)
            object_b = next((c for c, ids in self.color_geom_ids.items() if b in ids), None)
            if object_a and object_b and object_a != object_b:
                category = "object_object"
            if a in self.robot_geom_ids and b in self.robot_geom_ids and not (a in self.finger_geom_ids and b in self.finger_geom_ids):
                category = "robot_self"
            if category:
                mujoco.mj_contactForce(self.sim.model._model, self.sim.data._data, index, force)
                normal = abs(float(force[0]))
                threshold = self.config.object_collision_force_n if category == "object_object" else self.config.collision_force_n
                if normal >= threshold:
                    categories.add(category)
                    peak = max(peak, normal)
        return sorted(categories), peak

    def _update_observables(self, force=False):
        super()._update_observables(force=force)
        if self._capture_contacts and self.sim.data.time > self._contact_sample_time + self.model_timestep * 0.5:
            categories, peak = self._contacts()
            self._substep_categories.update(categories)
            self._substep_peak = max(self._substep_peak, peak)
            self._substep_collision_steps += bool(categories)
            self._contact_sample_time = self.sim.data.time

    def _score_step(self):
        cfg, episode = self.config, self._episode
        episode["control_steps"] += 1
        tick = episode["control_steps"]
        grasp = {color: bool(self._check_grasp(self.robots[0].gripper, obj)) for color, obj in self.color_objects.items()}
        for color in COLORS:
            if grasp[color]:
                self._last_grasp_step[color] = tick
            self._wrong_grasp_streak[color] = self._wrong_grasp_streak[color] + 1 if grasp[color] else 0
            if color != self.target_color and self._wrong_grasp_streak[color] >= cfg.wrong_grasp_confirm_steps:
                episode["wrong_color_grasp"] = True
                if color not in episode["wrong_colors"]:
                    episode["wrong_colors"].append(color)
        color = self.target_color
        bid = self.color_body_ids[color]
        position = self.sim.data.body_xpos[bid].copy()
        lift = float(position[2] - self.initial_heights[color])
        linear = float(np.linalg.norm(self.sim.data.get_body_xvelp(self.color_objects[color].root_body)))
        angular = float(np.linalg.norm(self.sim.data.get_body_xvelr(self.color_objects[color].root_body)))
        recent_grasp = tick - self._last_grasp_step[color] <= cfg.contact_grace_steps
        episode["target_grasped_ever"] |= grasp[color]
        episode["max_lift_m"] = max(episode["max_lift_m"], lift)
        height_ok = lift > cfg.lift_threshold
        episode["reached_height"] |= height_ok and recent_grasp
        stable = linear <= cfg.max_linear_speed and angular <= cfg.max_angular_speed
        drift = 0.0 if self._hold_anchor is None else float(np.linalg.norm(position - self._hold_anchor))
        valid = height_ok and recent_grasp and stable and drift <= cfg.max_hold_drift and not episode["wrong_color_grasp"]
        if valid:
            if episode["hold_streak"] == 0:
                self._hold_anchor = position.copy()
            episode["hold_streak"] += 1
        else:
            if episode["hold_streak"] > 0:
                episode["holding_failure_events"] += 1
            episode["hold_streak"] = 0
            self._hold_anchor = None
        episode["max_hold_streak"] = max(episode["max_hold_streak"], episode["hold_streak"])
        episode["success"] = episode["hold_streak"] >= cfg.hold_steps and not episode["wrong_color_grasp"]
        collisions, force = self._contacts()
        collisions = sorted(set(collisions) | self._substep_categories)
        force = max(force, self._substep_peak)
        episode["collision_physics_steps"] += self._substep_collision_steps
        episode["collision_types"] = sorted(set(episode["collision_types"]) | set(collisions))
        episode["collision_steps"] += bool(collisions)
        episode["collision_events"] += bool(collisions) and not self._last_collision
        episode["max_collision_force_n"] = max(episode["max_collision_force_n"], force)
        self._last_collision = bool(collisions)
        return {**episode, "target_color": color, "body_ids": dict(self.color_body_ids),
                "lift_m": lift, "linear_speed_m_s": linear, "angular_speed_rad_s": angular,
                "hold_drift_m": drift, "target_grasped": grasp[color], "grasp_by_color": grasp,
                "recent_target_grasp": recent_grasp, "height_ok": height_ok,
                "stable_now": stable, "collisions": collisions,
                "holding_failed": episode["reached_height"] and not episode["success"],
                "collision_sample_hz": round(1 / self.model_timestep), "hold_sample_hz": cfg.hz,
                "collision_free_success": episode["success"] and episode["collision_steps"] == 0}

    def step(self, action):
        self._substep_categories, self._substep_peak, self._substep_collision_steps = set(), 0.0, 0
        self._contact_sample_time = self.sim.data.time
        self._capture_contacts = True
        try:
            obs, _, done, info = super().step(action)
        finally:
            self._capture_contacts = False
        score = self._score_step()
        info.update(score)
        done = bool(done or score["success"] or score["wrong_color_grasp"])
        return obs, float(score["success"]), done, info

    def reward(self, action=None):
        return float(self._episode.get("success", False))

    def _check_success(self):
        return self._episode.get("success", False)

    def reset_seed(self, seed):
        self.rng = np.random.default_rng(seed)
        self.placement_initializer.rng = self.rng
        return self.reset()

    def episode_metadata(self):
        return {"target_color": self.target_color, "body_ids": dict(self.color_body_ids),
                "body_names": {c: o.root_body for c, o in self.color_objects.items()},
                "initial_positions": {c: p.tolist() for c, p in self.initial_positions.items()},
                "initial_heights": {c: float(z) for c, z in self.initial_heights.items()}}


def make_env(gpu=1, config=None, cameras=True, segmentations=False):
    cfg = config or Config()
    return ColorLift(config=cfg, has_renderer=False, has_offscreen_renderer=cameras,
                     use_camera_obs=cameras, camera_names=list(CAMERAS),
                     camera_heights=cfg.image_size, camera_widths=cfg.image_size,
                     camera_segmentations="element" if segmentations else None,
                     render_gpu_device_id=gpu, hard_reset=False, seed=0)
