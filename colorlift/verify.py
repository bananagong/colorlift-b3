"""Functional tests of randomization, HSV, correct/wrong grasp, and interrupted holding."""
import argparse
import json
from dataclasses import replace
import numpy as np
from PIL import Image
from .config import Config, EXPERIMENT, COLORS, CAMERAS
from .env import make_env
from .collect import rollout, save_csv
from .vision import observation_images, observation_masks, mask_preview


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=1)
    args = parser.parse_args()
    cfg = Config()
    out = EXPERIMENT / "environment"
    out.mkdir(parents=True, exist_ok=True)
    env = make_env(args.gpu, cfg, segmentations=True)
    report = {"checks": {}}
    try:
        positions = []
        for seed in range(90000, 90030):
            obs = env.reset_seed(seed)
            env.assert_no_overlap()
            assert len(set(env.color_body_ids.values())) == 3
            positions.append([env.initial_positions[color] for color in COLORS])
        assert np.asarray(positions).std(axis=0)[:, :2].min() > 0.025
        first = env.reset_seed(90000)
        p = env.episode_metadata()["initial_positions"]
        env.reset_seed(90001)
        obs = env.reset_seed(90000)
        assert p == env.episode_metadata()["initial_positions"]
        assert "robot0_joint_pos" in obs and "target_color_onehot" in obs
        assert "cube_pos" not in obs
        rgb, masks = observation_images(obs), observation_masks(obs)
        assert rgb.shape == masks.shape == (2, cfg.image_size, cfg.image_size, 3)
        ious = {}
        for camera_index, camera in enumerate(CAMERAS):
            segmentation = obs[camera + "_segmentation_element"][::-1, :, 0]
            for color_index, color in enumerate(COLORS):
                truth = np.isin(segmentation, list(env.color_visual_geom_ids[color]))
                prediction = masks[camera_index, ..., color_index] > 0
                union = np.logical_or(truth, prediction).sum()
                iou = float(np.logical_and(truth, prediction).sum() / max(1, union))
                ious[camera + "/" + color] = iou
                assert prediction.sum() > 5 and iou > 0.45, (camera, color, iou)
        Image.fromarray(np.concatenate([np.concatenate(list(rgb), axis=1),
                                        np.concatenate([mask_preview(m) for m in masks], axis=1)], axis=0)).save(out / "rgb_hsv.png")
        report["checks"]["randomization_body_ids_rgb_hsv"] = {"passed": True, "resets": 30, "body_ids": dict(env.color_body_ids), "HSV_IoU_against_visual_geom_ids": ious}
        yellow = env.color_objects["yellow"]
        red_qpos = env.sim.data.get_joint_qpos(env.color_objects["red"].joints[0]).copy()
        env.sim.data.set_joint_qpos(yellow.joints[0], red_qpos)
        env.sim.forward()
        try:
            env.assert_no_overlap()
        except AssertionError:
            overlap_rejected = True
        else:
            overlap_rejected = False
        assert overlap_rejected
        report["checks"]["overlap_negative_case"] = {"passed": True}
        result, _, rows = rollout(env, 90002, record=False)
        assert result["success"] and result["max_hold_streak"] >= 40 and result["max_lift_m"] > 0.04
        report["checks"]["correct_target_2s_hold"] = {"passed": True, "steps": result["control_steps"], "max_hold_streak": result["max_hold_streak"], "max_lift_m": result["max_lift_m"]}
        save_csv(out / "positive_steps.csv", rows)
        wrong, _, rows = rollout(env, 90003, expert_color="blue", record=False)
        assert wrong["wrong_color_grasp"] and not wrong["success"] and "blue" in wrong["wrong_colors"]
        report["checks"]["wrong_color_negative_case"] = {"passed": True, "wrong_colors": wrong["wrong_colors"], "collision_steps": wrong["collision_steps"], "max_collision_force_n": wrong["max_collision_force_n"]}
        save_csv(out / "wrong_color_steps.csv", rows)
        released, _, rows = rollout(env, 90004, record=False, release_during_hold=True)
        assert released["released_for_test"] and not released["success"] and released["max_hold_streak"] < 40 and released["holding_failure_events"] > 0
        report["checks"]["interrupted_hold_negative_case"] = {"passed": True, "max_hold_streak": released["max_hold_streak"], "holding_failure_events": released["holding_failure_events"]}
        save_csv(out / "interrupted_hold_steps.csv", rows)
        env.hard_reset = True
        env.reset_seed(90005)
        assert len(env.color_body_ids) == 3
        env.assert_no_overlap()
        report["checks"]["hard_reset_rebuilds_body_ids"] = {"passed": True, "body_ids": dict(env.color_body_ids)}
        report["passed"] = True
        cfg.save(EXPERIMENT / "config.json")
        (out / "verification.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
