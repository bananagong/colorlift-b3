"""Cube Lift protocol, shared by demonstration collection and policy evaluation."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
from scipy.spatial.transform import Rotation
from robosuite.environments.manipulation.lift import Lift
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler
from catlab_assets import CubeObject

CAMERAS = ["lift_front", "lift_side"]
IMAGE_SIZE = 128
HZ = 20
HORIZON = 160
SUCCESS_Z = 0.90
HOLD_STEPS = 10
ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "lift_act_v1"


def camera_quat(position, target):
    z = np.asarray(position, dtype=float) - np.asarray(target, dtype=float)
    z /= np.linalg.norm(z)
    x = np.cross([0, 0, 1], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    xyzw = Rotation.from_matrix(np.stack([x, y, z], axis=1)).as_quat()
    return xyzw[[3, 0, 1, 2]]


class CubeLift(Lift):
    def _load_model(self):
        super()._load_model()
        arena = self.model.mujoco_arena
        for name, pos in zip(CAMERAS, [[0, -0.47, 1.18], [0.47, 0.02, 1.18]]):
            arena.set_camera(name, pos=pos, quat=camera_quat(pos, [0, 0, 0.86]), camera_attribs={"fovy": "40"})
        self.cube = CubeObject("cube")
        self.placement_initializer = UniformRandomSampler(
            name="CubeSampler", mujoco_objects=self.cube,
            x_range=[-0.055, 0.055], y_range=[-0.055, 0.055], rotation=[-0.25, 0.25],
            ensure_object_boundary_in_range=False, ensure_valid_placement=True,
            reference_pos=self.table_offset, z_offset=0.002, rng=self.rng,
        )
        self.model = ManipulationTask(arena, self.model.mujoco_robots, [self.cube])

    def _check_success(self):
        return bool(self.sim.data.body_xpos[self.cube_body_id, 2] >= SUCCESS_Z)


def make_env(gpu=1, cameras=True, image_size=IMAGE_SIZE):
    return CubeLift(
        robots="Panda", initialization_noise=None, has_renderer=False,
        has_offscreen_renderer=cameras, use_camera_obs=cameras,
        camera_names=CAMERAS, camera_heights=image_size, camera_widths=image_size,
        render_gpu_device_id=gpu, control_freq=HZ, horizon=HORIZON,
        hard_reset=False, reward_shaping=False, seed=0,
    )


def reset_seed(env, seed):
    env.rng = np.random.default_rng(seed)
    env.placement_initializer.rng = env.rng
    return env.reset()


def state(obs):
    # Robot-only observations: no privileged object pose enters the learned policy.
    return np.concatenate([obs["robot0_joint_pos"], obs["robot0_gripper_qpos"], obs["robot0_eef_pos"]]).astype(np.float32)


def images(obs):
    # Convert OpenGL bottom-up images to conventional top-down RGB.
    return np.stack([obs[c + "_image"][::-1] for c in CAMERAS]).astype(np.uint8)


def servo_action(obs, target):
    # ACT predicts absolute Cartesian XYZ and gripper command. OSC_POSE tracks XYZ;
    # zero rotation commands retain the initial end-effector orientation.
    xyz = np.clip(np.asarray(target[:3]), [-0.25, -0.25, 0.795], [0.25, 0.25, 1.15])
    delta = np.clip((xyz - obs["robot0_eef_pos"]) / 0.05, -1, 1)
    grip = 1.0 if target[3] > 0 else -1.0
    return np.r_[delta, np.zeros(3), grip]


def success_tracker(env, streak):
    z = float(env.sim.data.body_xpos[env.cube_body_id, 2])
    streak = streak + 1 if z >= SUCCESS_Z else 0
    return streak, streak >= HOLD_STEPS, z
