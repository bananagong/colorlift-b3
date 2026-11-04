"""Verify custom cube geometry, camera dimensions, deterministic resets, and expert control."""
import json
import numpy as np
from PIL import Image
from .env import EXPERIMENT, make_env, reset_seed, images
from .collect import rollout


def main():
    out = EXPERIMENT / "environment"
    out.mkdir(parents=True, exist_ok=True)
    env = make_env(1)
    try:
        first = reset_seed(env, 9000)
        pos = first["cube_pos"].copy()
        joints = first["robot0_joint_pos"].copy()
        reset_seed(env, 9001)
        again = reset_seed(env, 9000)
        assert np.allclose(pos, again["cube_pos"], atol=1e-10)
        assert np.allclose(joints, again["robot0_joint_pos"], atol=1e-10)
        frames = images(again)
        assert frames.shape == (2, 128, 128, 3) and np.isfinite(frames).all()
        body = env.sim.model.body_name2id(env.cube.root_body)
        assert np.isclose(env.sim.model.body_mass[body], 0.064)
        assert np.allclose(env.cube.get_bounding_box_size(), [0.04] * 3)
        Image.fromarray(np.concatenate(list(frames), axis=1)).save(out / "cube_cameras.png")
        expert_results = []
        for seed in [9002, 9003, 9004]:
            result, _ = rollout(env, seed, record=False)
            assert result["success"], result
            expert_results.append(result)
        report = {"passed": True, "cube_edge_m": 0.04, "cube_mass_kg": 0.064,
                  "camera_shape": list(frames.shape), "deterministic_reset": True,
                  "expert_control_trials": expert_results, "gpu": 1}
        (out / "verification.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
