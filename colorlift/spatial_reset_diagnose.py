"""Investigate an initial-image audit mismatch without changing trained data or policies."""
from collections import defaultdict
import inspect
import json
import numpy as np
from PIL import Image
from .spatial_gate import RUN
from .spatial_env import make_spatial_env
from .vision import observation_images


def main():
    groups = defaultdict(list)
    for path in (RUN / "evaluations/test").glob("*/*/*/*/metadata.json"):
        row = json.loads(path.read_text()); groups[row["geometry_id"]].append(row)
    bad = next(group for group in groups.values() if len({r["initial_rgb_sha256"] for r in group}) > 1)
    print("MISMATCH", bad[0]["layout"]["level"], bad[0]["seed"], [(r["actor"], r["target_color"], r["initial_rgb_sha256"][:10]) for r in bad], flush=True)
    mismatches = []
    for group in groups.values():
        if len({r["initial_rgb_sha256"] for r in group}) <= 1: continue
        reference = max(group, key=lambda r: sum(x["initial_rgb_sha256"] == r["initial_rgb_sha256"] for x in group))
        base = np.asarray(Image.open(RUN / reference["directory"] / "initial.png")).astype(int)
        for row in group:
            if row["initial_rgb_sha256"] == reference["initial_rgb_sha256"]: continue
            current = np.asarray(Image.open(RUN / row["directory"] / "initial.png")).astype(int)
            difference = np.abs(current-base)
            mismatches.append({"level": row["layout"]["level"], "seed": row["seed"], "actor": row["actor"], "color": row["target_color"],
                               "max_pixel_difference": int(difference.max()), "mean_pixel_difference": float(difference.mean()),
                               "changed_channels": int((difference > 0).sum()),
                               "physical_positions_equal": row["physical_slot_positions"] == reference["physical_slot_positions"],
                               "robot_state_equal": row["initial_robot_state"][:12] == reference["initial_robot_state"][:12],
                               "camera_equal": row["actual_camera_positions"] == reference["actual_camera_positions"],
                               "lights_equal": row["actual_light_diffuse"] == reference["actual_light_diffuse"]})
    print("ALL_NUMERIC_DIFFERENCES", json.dumps(mismatches, indent=2), flush=True)
    env = make_spatial_env(1)
    try:
        print(inspect.getsource(env._get_observations), flush=True)
        print(inspect.getsource(type(env).__mro__[3]._reset_internal), flush=True)
        obs = env.reset_layout(bad[0]["layout"], "red")
        old = observation_images(obs)
        time_before = float(env.sim.data.time); qpos_before = env.sim.data.qpos.copy()
        fresh = observation_images(env._get_observations(force_update=True))
        output = {"initial_image_difference_mean": float(np.abs(old.astype(float)-fresh.astype(float)).mean()),
                  "initial_image_difference_max": int(np.abs(old.astype(int)-fresh.astype(int)).max()),
                  "physics_time_change": float(env.sim.data.time)-time_before,
                  "qpos_change": float(np.abs(env.sim.data.qpos-qpos_before).max()),
                  "level": bad[0]["layout"]["level"], "seed": bad[0]["seed"]}
        print(json.dumps(output, indent=2), flush=True)
        out = RUN / "reset_diagnosis"; out.mkdir(exist_ok=True)
        Image.fromarray(np.concatenate([np.concatenate(list(old), axis=1), np.concatenate(list(fresh), axis=1)], axis=0)).save(out / "cached_vs_fresh.png")
        (out / "diagnosis.json").write_text(json.dumps(output, indent=2))
    finally:
        env.close()


if __name__ == "__main__": main()
