"""Verify the T asset's mass, notch collision geometry, pushing, and robosuite integration."""

import argparse
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "outputs" / "t_block")
    args = parser.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(args.gpu)

    import imageio.v2 as imageio
    import mujoco
    import numpy as np
    from PIL import Image
    from robosuite.models.arenas import TableArena
    from robosuite.models.tasks import ManipulationTask
    from robosuite.utils.placement_samplers import UniformRandomSampler
    from .t_block import TBlockObject

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"checks": {}, "gpu": args.gpu, "asset": "catlab_assets.TBlockObject"}

    def compile_objects(objects):
        arena = TableArena(table_friction=(0.5, 0.005, 0.0001))
        task = ManipulationTask(mujoco_arena=arena, mujoco_robots=[], mujoco_objects=objects)
        model = mujoco.MjModel.from_xml_string(task.get_xml())
        return model

    # Different dimensions and multiple instances must preserve requested mass and unique names.
    a = TBlockObject("first")
    b = TBlockObject("small", width=0.12, length=0.16, height=0.02, bar_depth=0.04, stem_width=0.035, mass=0.16)
    multi = compile_objects([a, b])
    for obj in (a, b):
        bid = mujoco.mj_name2id(multi, mujoco.mjtObj.mjOBJ_BODY, obj.root_body)
        assert bid >= 0 and np.isclose(multi.body_mass[bid], obj.mass, atol=1e-10)
        assert len(obj.contact_geoms) == 2 and len(obj.visual_geoms) == 2 and len(obj.joints) == 1
        assert np.all(multi.body_inertia[bid] > 0)
    sampler = UniformRandomSampler(
        name="verify_sampler", mujoco_objects=b, x_range=[0, 0], y_range=[0, 0], rotation=0,
        ensure_object_boundary_in_range=False, reference_pos=[0, 0, 0.8], z_offset=0.003,
    )
    sampled_pos, _, _ = sampler.sample()[b.name]
    assert np.isclose(sampled_pos[2], 0.8 + b.height / 2 + 0.003)
    assert np.allclose(b.get_bounding_box_size(), [0.12, 0.16, 0.02])
    assert np.isclose(a.horizontal_radius, np.hypot(0.1, 0.1))
    polygon = a.footprint_vertices
    area = 0.5 * abs(np.dot(polygon[:, 0], np.roll(polygon[:, 1], -1)) - np.dot(polygon[:, 1], np.roll(polygon[:, 0], -1)))
    assert np.isclose(area, 0.0175)
    report["checks"]["custom_dimensions_names_mass_sampler"] = {"passed": True, "default_mass_kg": 0.3, "footprint_area_m2": area}
    print("PASS: dimensions, independent instances, mass, inertia, and sampler", flush=True)

    block = TBlockObject("push_t")
    block.set_pos([0, 0, 0.85])
    arena = TableArena(table_friction=(0.5, 0.005, 0.0001))
    arena.set_camera("asset_topview", pos=[0, 0, 1.40], quat=[1, 0, 0, 0], camera_attribs={"fovy": "38"})
    task = ManipulationTask(mujoco_arena=arena, mujoco_robots=[], mujoco_objects=block)
    task.root.find("option").set("timestep", "0.002")
    global_visual = ET.SubElement(task.root.find("visual"), "global")
    global_visual.set("offwidth", "512")
    global_visual.set("offheight", "512")
    probe = ET.SubElement(task.worldbody, "body", name="probe", mocap="true", pos="2 2 2")
    ET.SubElement(probe, "geom", name="probe_geom", type="sphere", size="0.004", group="3", rgba="0 0 0 0")
    pusher = ET.SubElement(task.worldbody, "body", name="pusher", mocap="true", pos="-0.14 -0.05 0.820")
    ET.SubElement(pusher, "geom", name="pusher_collision", type="cylinder", size="0.012 0.019", group="0", rgba="0 0 0 0", friction="0.5 0.005 0.0001")
    ET.SubElement(pusher, "geom", name="pusher_visual", type="cylinder", size="0.012 0.019", group="1", rgba="0.95 0.4 0.08 1", contype="0", conaffinity="0", mass="0")
    xml = task.get_xml()
    (args.output_dir / "t_block_scene.xml").write_text(xml, encoding="utf-8")
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, block.root_body)
    block_gids = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in block.contact_geoms}
    probe_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "probe_geom")
    probe_mid = model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "probe")]
    pusher_mid = model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pusher")]
    for _ in range(2000):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    assert np.isfinite(data.qpos).all()
    assert abs(data.xpos[bid, 2] - 0.8125) < 0.002
    assert data.xmat[bid].reshape(3, 3)[2, 2] > 0.995
    report["checks"]["stable_table_contact"] = {"passed": True, "body_height_m": float(data.xpos[bid, 2]), "mass_kg": float(model.body_mass[bid]), "center_of_mass_local": model.body_ipos[bid].tolist()}

    # A sphere must fit in both notches, while the cap and stem have solid contacts.
    probe_results = {}
    for label, xy, expected in (
        ("left_notch", [-0.075, -0.05], False),
        ("right_notch", [0.075, -0.05], False),
        ("stem_solid", [0, -0.05], True),
        ("cap_solid", [0.075, 0.075], True),
    ):
        data.mocap_pos[probe_mid] = data.xpos[bid] + np.array([*xy, 0])
        mujoco.mj_forward(model, data)
        touching = any((c.geom1 == probe_gid and c.geom2 in block_gids) or (c.geom2 == probe_gid and c.geom1 in block_gids) for c in data.contact)
        assert touching == expected, (label, touching)
        probe_results[label] = {"contact": touching, "expected_contact": expected}
    data.mocap_pos[probe_mid] = [2, 2, 2]
    mujoco.mj_forward(model, data)
    report["checks"]["nonconvex_notches"] = {"passed": True, "probes": probe_results}
    print("PASS: table stability and true T-shaped notch collision boundaries", flush=True)

    initial_xy = data.xpos[bid, :2].copy()
    initial_yaw = float(np.arctan2(data.xmat[bid, 3], data.xmat[bid, 0]))
    contacts = 0
    min_z_axis = 1.0
    options = mujoco.MjvOption()
    options.geomgroup[0] = 0
    options.geomgroup[1] = 1
    with mujoco.Renderer(model, height=512, width=512) as renderer:
        renderer.update_scene(data, camera="asset_topview", scene_option=options)
        Image.fromarray(renderer.render()).save(args.output_dir / "t_block_topview.png")
        with imageio.get_writer(str(args.output_dir / "t_block_push.mp4"), fps=25, codec="libx264", quality=8) as writer:
            for step in range(1500):
                data.mocap_pos[pusher_mid, 0] = -0.14 + 0.22 * (step + 1) / 1500
                mujoco.mj_step(model, data)
                assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
                pusher_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "pusher_collision")
                contacts += sum((c.geom1 == pusher_gid and c.geom2 in block_gids) or (c.geom2 == pusher_gid and c.geom1 in block_gids) for c in data.contact)
                min_z_axis = min(min_z_axis, float(data.xmat[bid].reshape(3, 3)[2, 2]))
                if step % 20 == 0:
                    renderer.update_scene(data, camera="asset_topview", scene_option=options)
                    writer.append_data(renderer.render())
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera="asset_topview", scene_option=options)
            Image.fromarray(renderer.render()).save(args.output_dir / "t_block_after_push.png")
    displacement = float(np.linalg.norm(data.xpos[bid, :2] - initial_xy))
    yaw = float(np.arctan2(data.xmat[bid, 3], data.xmat[bid, 0]))
    yaw_change = float(np.arctan2(np.sin(yaw - initial_yaw), np.cos(yaw - initial_yaw)))
    assert contacts > 0 and displacement > 0.02 and abs(yaw_change) > 0.05
    assert min_z_axis > 0.95
    report["checks"]["contact_driven_push"] = {"passed": True, "steps": 1500, "contact_count": int(contacts), "displacement_m": displacement, "yaw_change_rad": yaw_change, "minimum_up_axis_z": min_z_axis}
    print("PASS: contact-driven sliding and rotation", report["checks"]["contact_driven_push"], flush=True)

    from .preview import make_preview

    env = make_preview(args.gpu)
    try:
        for reset_index in range(2):
            obs = env.reset()
            for _ in range(20):
                obs, reward, _, _ = env.step(np.zeros(env.action_dim))
                assert reward == 0 and np.isfinite(env.sim.data.qpos).all()
            assert len(env.cube.contact_geoms) == 2
            body_id = env.sim.model.body_name2id(env.cube.root_body)
            assert np.isclose(env.sim.model.body_mass[body_id], 0.3)
            assert np.isfinite(obs["cube_pos"]).all()
            for camera in ("agentview", "tblock_topview"):
                frame = obs[camera + "_image"]
                assert frame.shape == (512, 512, 3) and frame.std() > 1
                if reset_index == 1:
                    Image.fromarray(frame[::-1]).save(args.output_dir / ("panda_" + camera + ".png"))
        report["checks"]["robosuite_panda_integration"] = {"passed": True, "resets": 2, "steps_per_reset": 20, "root_body": env.cube.root_body, "joint": env.cube.joints[0], "cameras": ["agentview", "tblock_topview"]}
    finally:
        env.close()
    report["passed"] = True
    (args.output_dir / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print("ALL T-BLOCK CHECKS PASSED: " + str(args.output_dir), flush=True)


if __name__ == "__main__":
    main()
