"""Policy-independent geometry, stability, visibility, IK and pinch-corridor checks."""
import copy
from dataclasses import replace
import cv2
import mujoco
import numpy as np
from .spec import VALID as V, COLORS
from .layouts import footprint_gap, square
from ..config import CAMERAS
from ..conditional import configuration
from ..spatial_env import SpatialColorLift
from ..recovery_expert import DampedIK


class BroadColorLift(SpatialColorLift):
    def assert_no_overlap(self):
        if self._physical_objects is None:
            return super().assert_no_overlap()
        points = np.array([self.sim.data.body_xpos[b, :2] for b in self._physical_ids["body"]])
        yaw = [2*np.arctan2(self.sim.data.body_xquat[b, 3], self.sim.data.body_xquat[b, 0]) for b in self._physical_ids["body"]]
        assert footprint_gap(points, yaw) > 0, "Overlapping rotated cube footprints"

    def reset_layout(self, layout, target_color):
        obs = super().reset_layout(layout, target_color)
        fresh = copy.deepcopy(self._episode)
        before = np.array([self.sim.data.body_xpos[b].copy() for b in self._physical_ids["body"]])
        for _ in range(round(V["settle_seconds"]*self.config.hz)):
            obs, _, _, _ = self.step(np.r_[np.zeros(6), -1.])
        after = np.array([self.sim.data.body_xpos[b].copy() for b in self._physical_ids["body"]])
        tilt = [float(np.arccos(np.clip(self.sim.data.body_xmat[b].reshape(3, 3)[2, 2], -1, 1))) for b in self._physical_ids["body"]]
        velocities = [np.asarray(self.sim.data.get_joint_qvel(obj.joints[0])) for obj in self._physical_objects]
        self.settlement = {"max_drift_m": float(np.linalg.norm(after-before, axis=1).max()),
                           "max_height_error_m": float(np.abs(after[:, 2]-.82).max()), "max_tilt_rad": max(tilt),
                           "max_linear_speed_m_s": max(float(np.linalg.norm(v[:3])) for v in velocities),
                           "max_angular_speed_rad_s": max(float(np.linalg.norm(v[3:])) for v in velocities),
                           "unintended_contact_steps": self._episode["collision_steps"],
                           "positions_after_settle": after.tolist()}
        # Begin each task after the same actual settling procedure, without a
        # live qpos teleport. Scoring is reset independently of physical time.
        self._episode = fresh; self._hold_anchor = None; self._last_collision = False
        self._last_grasp_step = {c: -1000 for c in COLORS}; self._wrong_grasp_streak = {c: 0 for c in COLORS}
        self.initial_positions = {c: self.sim.data.body_xpos[b].copy() for c, b in self.color_body_ids.items()}
        self.initial_heights = {c: float(p[2]) for c, p in self.initial_positions.items()}
        self.timestep = 0
        return self._get_observations(force_update=True)


def make_env(gpu=1):
    cfg = replace(configuration()[0], horizon=220)
    return BroadColorLift(config=cfg, has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True,
                          camera_names=list(CAMERAS), camera_segmentations="element", render_gpu_device_id=gpu, hard_reset=False, seed=0)


class PoseIK(DampedIK):
    def solve_pose(self, position, start=None):
        m, d = self.model, self.data
        d.qpos[:] = self.env.sim.data.qpos; d.qvel[:] = 0
        if start is not None: d.qpos[self.qpos] = start
        desired = self.env.sim.data.site_xmat[self.site].reshape(3, 3).copy()
        jp = np.zeros((3, m.nv)); jr = np.zeros_like(jp)
        for iteration in range(65):
            mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
            actual = d.site_xmat[self.site].reshape(3, 3)
            ep = np.array(position)-d.site_xpos[self.site]
            er = .5*sum(np.cross(actual[:, i], desired[:, i]) for i in range(3))
            if np.linalg.norm(ep) < .00025 and np.linalg.norm(er) < .003: break
            mujoco.mj_jacSite(m, d, jp, jr, self.site)
            jac = np.vstack((jp[:, self.dofs], .4*jr[:, self.dofs]))
            change = jac.T @ np.linalg.solve(jac@jac.T+.0001*np.eye(6), np.r_[ep, .4*er])
            if np.linalg.norm(change) > .16: change *= .16/np.linalg.norm(change)
            d.qpos[self.qpos] = np.clip(d.qpos[self.qpos]+change, self.limits[:, 0]+.005, self.limits[:, 1]-.005)
        mujoco.mj_forward(m, d)
        rotation_error = float(np.arccos(np.clip((np.trace(desired.T@d.site_xmat[self.site].reshape(3, 3))-1)/2, -1, 1)))
        position_error = float(np.linalg.norm(d.site_xpos[self.site]-position))
        margin = float(np.minimum(d.qpos[self.qpos]-self.limits[:, 0], self.limits[:, 1]-d.qpos[self.qpos]).min())
        return d.qpos[self.qpos].copy(), {"position_error_m": position_error, "rotation_error_rad": rotation_error,
                                          "joint_margin_rad": margin, "iterations": iteration+1}


def visibility(env, obs):
    model, data = env.sim.model._model, env.sim.data._data
    records = []
    for slot, body in enumerate(env._physical_ids["body"]):
        corners = np.array(list(__import__("itertools").product([-.02, .02], repeat=3)))
        world = corners@data.xmat[body].reshape(3, 3).T+data.xpos[body]
        cameras = []
        for camera in CAMERAS:
            seg = obs[camera+"_segmentation_element"][::-1, :, 0]
            height, width = seg.shape
            cid = env.sim.model.camera_name2id(camera)
            view = (world-data.cam_xpos[cid])@data.cam_xmat[cid].reshape(3, 3)
            f = .5*height/np.tan(np.radians(model.cam_fovy[cid])/2)
            uv = np.c_[width/2+f*view[:, 0]/(-view[:, 2]), height/2-f*view[:, 1]/(-view[:, 2])]
            polygon = cv2.convexHull(uv.astype(np.float32))
            area = max(1., cv2.contourArea(polygon))
            count = int(np.isin(seg, list(env._physical_ids["visual"][slot])).sum())
            border = float(min(uv[:, 0].min(), uv[:, 1].min(), width-1-uv[:, 0].max(), height-1-uv[:, 1].max()))
            cameras.append({"camera": camera, "pixels": count, "projected_area_px": area,
                            "visible_fraction": min(1., count/area), "border_px": border, "in_front": bool((view[:, 2] < 0).all())})
        fractions = sorted(c["visible_fraction"] for c in cameras)
        valid = all(c["pixels"] >= V["visible_pixels_per_camera"] and c["border_px"] >= V["image_border_px"] and c["in_front"] for c in cameras)
        valid &= fractions[-1] >= V["visible_fraction_best_camera"] and fractions[0] >= V["visible_fraction_other_camera"]
        records.append({"slot": slot, "valid": bool(valid), "cameras": cameras})
    return records


class Validator:
    def __init__(self, env):
        self.env = env; self.ik = PoseIK(env)
        self.model = env.sim.model._model
        gripper = env.robots[0].gripper
        gripper = next(iter(gripper.values())) if isinstance(gripper, dict) else gripper
        self.finger_qpos = np.array([self.model.jnt_qposadr[self.model.joint(name).id] for name in gripper.joints])
        self.pad_sides = [[env.sim.model.geom_name2id(name) for name in gripper.important_geoms[side]] for side in ("left_finger", "right_finger")]

    def collisions(self, target_slot, allowed_target=False):
        env, data, model = self.env, self.ik.data, self.model
        target = set(env._physical_ids["contact"][target_slot])
        unexpected = []
        for contact in data.contact:
            a, b = int(contact.geom1), int(contact.geom2)
            if float(contact.dist) >= -V["unwanted_static_penetration_m"]: continue
            if not (a in env.robot_geom_ids or b in env.robot_geom_ids): continue
            if a in env.finger_geom_ids and b in env.finger_geom_ids: continue
            if allowed_target and ((a in env.finger_geom_ids and b in target) or (b in env.finger_geom_ids and a in target)): continue
            unexpected.append([a, b, float(contact.dist)])
        # Check positive clearance of fingers/palm to distractors, not merely an
        # absence of penetrating contacts. Arm collision uses the full model above.
        for slot in range(3):
            if slot == target_slot: continue
            for g in env.gripper_geom_ids:
                for other in env._physical_ids["contact"][slot]:
                    distance = float(mujoco.mj_geomDistance(model, data, g, other, .003, None))
                    if distance < .001: unexpected.append([int(g), int(other), distance])
        return unexpected

    def grasp_plan(self, slot):
        env, ik, model = self.env, self.ik, self.model
        xy = env.sim.data.body_xpos[env._physical_ids["body"][slot], :2].copy()
        home = env.sim.data.site_xpos[ik.site].copy()
        points = [np.r_[home[:2], max(home[2], V["safe_transport_z_m"])],
                  np.r_[xy, V["safe_transport_z_m"]], np.r_[xy, V["pregrasp_z_m"]],
                  np.r_[xy, V["grasp_z_m"]]]
        previous = env.sim.data.qpos[ik.qpos].copy()
        open_fingers = env.sim.data.qpos[self.finger_qpos].copy()
        diagnostics, joints = [], []
        for index, point in enumerate(points):
            q, diagnostic = ik.solve_pose(point, previous)
            diagnostics.append(diagnostic); joints.append(q.tolist())
            if diagnostic["position_error_m"] > V["IK_position_error_m"] or diagnostic["rotation_error_rad"] > V["IK_rotation_error_rad"] or diagnostic["joint_margin_rad"] < .0049:
                return {"valid": False, "reason": "IK_pose_or_joint_limits", "slot": slot, "IK": diagnostics}
            for alpha in np.linspace(0, 1, V["joint_sweep_samples"]):
                ik.data.qpos[ik.qpos] = (1-alpha)*previous+alpha*q
                ik.data.qpos[self.finger_qpos] = open_fingers
                mujoco.mj_forward(model, ik.data)
                hits = self.collisions(slot, allowed_target=index == 3)
                if hits: return {"valid": False, "reason": "open_approach_collision", "slot": slot, "segment": index, "contacts": hits[:4]}
            previous = q
        # Conservative projected cube width along the actual closing axis.
        ik.data.qpos[ik.qpos] = previous; mujoco.mj_forward(model, ik.data)
        rotation = ik.data.site_xmat[ik.site].reshape(3, 3)
        closing = ik.data.geom_xpos[self.pad_sides[1]].mean(0)-ik.data.geom_xpos[self.pad_sides[0]].mean(0)
        closing /= np.linalg.norm(closing)
        body_rotation = env.sim.data.body_xmat[env._physical_ids["body"][slot]].reshape(3, 3)
        width = .04*float(np.abs(body_rotation.T@closing).sum())
        if width+.004 > float(np.abs(open_fingers).sum()):
            return {"valid": False, "reason": "insufficient_open_gap", "slot": slot, "required_width_m": width+.004}
        closed = np.sign(open_fingers)*(width/2-.0003)
        for alpha in np.linspace(0, 1, 5):
            ik.data.qpos[self.finger_qpos] = (1-alpha)*open_fingers+alpha*closed
            mujoco.mj_forward(model, ik.data)
            hits = self.collisions(slot, allowed_target=True)
            if hits: return {"valid": False, "reason": "pinch_corridor_collision", "slot": slot, "contacts": hits[:4]}
        lift_q, diagnostic = ik.solve_pose(np.r_[xy, V["lift_z_m"]], previous)
        if diagnostic["position_error_m"] > .001 or diagnostic["rotation_error_rad"] > .025:
            return {"valid": False, "reason": "lift_pose_unreachable", "slot": slot, "IK": diagnostic}
        # The held object itself is not teleported during validity checks. Sweep
        # the robot lift and reserve a vertical object corridor analytically.
        for alpha in np.linspace(0, 1, 8):
            ik.data.qpos[ik.qpos] = (1-alpha)*previous+alpha*lift_q
            ik.data.qpos[self.finger_qpos] = closed; mujoco.mj_forward(model, ik.data)
            hits = self.collisions(slot, allowed_target=True)
            if hits: return {"valid": False, "reason": "lift_corridor_collision", "slot": slot, "contacts": hits[:4]}
        return {"valid": True, "slot": slot, "waypoints": [p.tolist() for p in points], "IK_qpos": joints,
                "IK": diagnostics, "lift_IK": diagnostic, "pregrasp_position": points[2].tolist(),
                "pregrasp_rotation_matrix": rotation.tolist(), "pinch_width_m": width}

    def check(self, layout):
        env = self.env
        obs = env.reset_layout(layout, "red")
        result = {"geometry_id": layout["geometry_id"], "settlement": env.settlement}
        checks = (("max_drift_m", "settle_max_drift_m"), ("max_height_error_m", "settle_max_height_error_m"),
                  ("max_tilt_rad", "settle_max_tilt_rad"), ("max_linear_speed_m_s", "settle_max_linear_speed_m_s"),
                  ("max_angular_speed_rad_s", "settle_max_angular_speed_rad_s"))
        if any(env.settlement[a] > V[b] for a, b in checks) or env.settlement["unintended_contact_steps"]:
            return result | {"valid": False, "reason": "unstable_initial_scene"}
        result["visibility"] = visibility(env, obs)
        if not all(r["valid"] for r in result["visibility"]): return result | {"valid": False, "reason": "camera_visibility"}
        result["grasp_plans"] = [self.grasp_plan(slot) for slot in range(3)]
        if not all(p["valid"] for p in result["grasp_plans"]):
            reason = next(p["reason"] for p in result["grasp_plans"] if not p["valid"])
            return result | {"valid": False, "reason": reason}
        return result | {"valid": True, "reason": "all_policy_independent_checks_passed", "actual_positions": env.settlement["positions_after_settle"]}
