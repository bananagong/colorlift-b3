"""Panda tabletop preview of TBlockObject; this is an asset demo, not a Push-T task."""

import numpy as np
from robosuite.environments.manipulation.lift import Lift
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler

from .t_block import TBlockObject


class TBlockPreview(Lift):
    """Reuse Lift's robot and observations to preview a custom object.

    The inherited cube_* observation keys refer to the T block in this preview.
    Reward is always zero and no success criterion is defined for the demo.
    """

    def _load_model(self):
        super()._load_model()
        arena = self.model.mujoco_arena
        arena.set_camera("tblock_topview", pos=[0, 0, 1.5], quat=[1, 0, 0, 0], camera_attribs={"fovy": "40"})
        self.cube = TBlockObject(name="t_block")
        self.placement_initializer = UniformRandomSampler(
            name="TBlockSampler", mujoco_objects=self.cube,
            x_range=[0, 0], y_range=[0, 0], rotation=0,
            ensure_object_boundary_in_range=False, ensure_valid_placement=True,
            reference_pos=self.table_offset, z_offset=0.002, rng=self.rng,
        )
        self.model = ManipulationTask(
            mujoco_arena=arena, mujoco_robots=self.model.mujoco_robots,
            mujoco_objects=self.cube,
        )

    def reward(self, action=None):
        return 0.0

    def _check_success(self):
        return False


def make_preview(gpu=2, image_size=512):
    return TBlockPreview(
        robots="Panda", has_renderer=False, has_offscreen_renderer=True,
        use_camera_obs=True, camera_names=["agentview", "tblock_topview"],
        camera_heights=image_size, camera_widths=image_size,
        render_gpu_device_id=gpu, initialization_noise=None,
        control_freq=20, horizon=300, seed=42,
    )


def main():
    import argparse
    from pathlib import Path
    from PIL import Image

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "outputs" / "t_block")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = make_preview(args.gpu)
    try:
        obs = env.reset()
        for _ in range(20):
            obs, _, _, _ = env.step(np.zeros(env.action_dim))
        for name in ("agentview", "tblock_topview"):
            Image.fromarray(obs[name + "_image"][::-1]).save(args.output_dir / ("panda_" + name + ".png"))
        print("Panda + T-block preview saved to " + str(args.output_dir))
    finally:
        env.close()


if __name__ == "__main__":
    main()
