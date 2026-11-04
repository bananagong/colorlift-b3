"""Read-only sampler/reset/IK checks; never collect demonstrations or train a policy."""
import hashlib
import json
import numpy as np
from PIL import Image
from .spatial_gate import RUN
from .spatial_layouts import audit_descriptors, descriptor, counterfactual_descriptor
from .spatial_env import make_spatial_env
from .recovery_expert import DampedIK
from .vision import observation_images
from .env import RGBA


def main():
    out = RUN / "preflight"; out.mkdir(parents=True, exist_ok=True)
    report = audit_descriptors()
    env = make_spatial_env(1)
    rows = []
    poses = []
    rgb_hashes = []
    try:
        for index in range(6):
            layout = counterfactual_descriptor(1999000, "easy", index)
            obs = env.reset_layout(layout, "red")
            meta = env.episode_metadata()
            images = observation_images(obs)
            rgb_hashes.append(hashlib.sha256(images.tobytes()).hexdigest())
            poses.append(meta["physical_slot_positions"])
            for slot, color in enumerate(layout["colors_by_slot"]):
                assert env.color_body_ids[color] == meta["physical_slot_body_ids"][slot]
                for geom in env.color_visual_geom_ids[color]:
                    assert np.allclose(env.sim.model.geom_rgba[geom], RGBA[color])
            Image.fromarray(np.concatenate(list(images), axis=1)).save(out / f"color_permutation_{index}.png")
            rows.append({"permutation": index, "colors": layout["colors_by_slot"], "target_red_slot": meta["target_physical_slot"], "body_ids": meta["body_ids"]})
        assert np.max(np.abs(np.array(poses)-poses[0])) == 0
        assert len(set(rgb_hashes)) == 6
        ik = DampedIK(env)
        reachability = []
        for index, level in enumerate(("easy", "medium", "hard", "ood_position", "ood_light", "ood_camera")):
            layout = descriptor(1999100+index, level)
            obs = env.reset_layout(layout, "red")
            residuals = []
            for position in layout["xy"]:
                _, diagnostic = ik.solve(np.r_[position, env.config.grasp_height])
                residuals.append(diagnostic["ik_position_residual_m"])
            reachability.append({"level": level, "IK_residual_m": residuals})
            assert max(residuals) < .001, (level, residuals)
            Image.fromarray(np.concatenate(list(observation_images(obs)), axis=1)).save(out / f"{level}.png")
        report.update(counterfactual_positions_exactly_identical=True, counterfactual_rgb_all_different=True,
                      physical_color_reassignment=rows, reachability=reachability,
                      no_expert_rollouts=True, no_training=True)
    finally:
        env.close()
    (out / "checks.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__": main()
