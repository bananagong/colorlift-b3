"""Physical slot identities stay fixed when their visible colors are exchanged."""
from dataclasses import replace
import numpy as np
from .config import CAMERAS, COLORS
from .conditional import configuration
from .env import ColorLift, RGBA
from .spatial_layouts import descriptor
from lift_act.env import camera_quat


class FixedLayoutSampler:
    def __init__(self, objects, env):
        self.objects, self.env = list(objects), env
        self.rng = env.rng

    def sample(self):
        result = {}
        for slot, obj in enumerate(self.objects):
            angle = self.env.layout["yaw"][slot]
            position = np.r_[self.env.layout["xy"][slot], self.env.config.table_height+self.env.config.cube_size/2]
            result[obj.name] = (position, np.array([np.cos(angle/2), 0, 0, np.sin(angle/2)]), obj)
        return result

    def reset(self):
        pass

    def add_objects(self, objects):
        pass


class SpatialColorLift(ColorLift):
    def __init__(self, *args, **kwargs):
        self.layout = descriptor(1999999, "easy")
        self._physical_objects = self._physical_ids = self._rendering_base = None
        super().__init__(*args, **kwargs)

    def _load_model(self):
        super()._load_model()
        self.placement_initializer = FixedLayoutSampler([self.color_objects[c] for c in COLORS], self)

    def _reset_internal(self):
        if self._physical_objects is not None:
            self.color_objects = dict(zip(COLORS, self._physical_objects))
            self.color_body_ids = dict(zip(COLORS, self._physical_ids["body"]))
            self.color_geom_ids = dict(zip(COLORS, self._physical_ids["contact"]))
            self.color_visual_geom_ids = dict(zip(COLORS, self._physical_ids["visual"]))
            self.cube = self.color_objects[self.target_color]
        super()._reset_internal()
        if self._physical_objects is None:
            self._physical_objects = [self.color_objects[c] for c in COLORS]
            self._physical_ids = {"body": [self.color_body_ids[c] for c in COLORS],
                                  "contact": [self.color_geom_ids[c] for c in COLORS],
                                  "visual": [self.color_visual_geom_ids[c] for c in COLORS]}
        model = self.sim.model._model
        if self._rendering_base is None:
            self._rendering_base = {key: np.array(getattr(model, key)).copy() for key in ("cam_pos", "cam_quat", "cam_fovy", "light_ambient", "light_diffuse", "light_specular")}
        for key, values in self._rendering_base.items():
            getattr(model, key)[:] = values
        self.color_objects, self.color_body_ids, self.color_geom_ids, self.color_visual_geom_ids = {}, {}, {}, {}
        for slot, color in enumerate(self.layout["colors_by_slot"]):
            self.color_objects[color] = self._physical_objects[slot]
            self.color_body_ids[color] = self._physical_ids["body"][slot]
            self.color_geom_ids[color] = self._physical_ids["contact"][slot]
            self.color_visual_geom_ids[color] = self._physical_ids["visual"][slot]
            for geom in self._physical_ids["visual"][slot]:
                model.geom_rgba[geom] = RGBA[color]
        self.cube = self.color_objects[self.target_color]
        rng = np.random.default_rng(self.layout["rendering_seed"])
        if self.layout["level"] == "ood_light":
            scale = rng.uniform(.60, .78) if rng.random() < .5 else rng.uniform(1.20, 1.40)
            tint = rng.uniform(.85, 1.12, 3)
            model.light_diffuse[:] = np.clip(self._rendering_base["light_diffuse"]*scale*tint, 0, 1.5)
            model.light_ambient[:] = np.clip(self._rendering_base["light_ambient"]*scale, 0, 1.5)
        elif self.layout["level"] == "ood_camera":
            for camera in CAMERAS:
                index = self.sim.model.camera_name2id(camera)
                pos = self._rendering_base["cam_pos"][index]+rng.uniform(-.015, .015, 3)
                target = np.array([0., 0., .85])+rng.uniform(-.01, .01, 3)
                model.cam_pos[index] = pos
                model.cam_quat[index] = camera_quat(pos, target)
                model.cam_fovy[index] = self._rendering_base["cam_fovy"][index]+rng.uniform(-3, 3)
        self.sim.forward()
        self.initial_positions = {color: self.sim.data.body_xpos[bid].copy() for color, bid in self.color_body_ids.items()}
        self.initial_heights = {color: float(p[2]) for color, p in self.initial_positions.items()}
        self.assert_no_overlap()

    def reset_layout(self, layout, target_color):
        self.layout = layout
        self.target_color = target_color
        self.config = replace(self.config, target_color=target_color)
        return self.reset_seed(layout["seed"])

    def episode_metadata(self):
        metadata = super().episode_metadata()
        model = self.sim.model._model
        metadata.update(layout=self.layout, geometry_id=self.layout["geometry_id"],
                        physical_slot_body_ids=list(self._physical_ids["body"]),
                        physical_slot_positions=[self.sim.data.body_xpos[bid].tolist() for bid in self._physical_ids["body"]],
                        physical_slot_quaternions=[self.sim.data.body_xquat[bid].tolist() for bid in self._physical_ids["body"]],
                        target_physical_slot=self.layout["colors_by_slot"].index(self.target_color),
                        actual_camera_positions={camera: model.cam_pos[self.sim.model.camera_name2id(camera)].tolist() for camera in CAMERAS},
                        actual_light_diffuse=model.light_diffuse.tolist())
        return metadata


def make_spatial_env(gpu=1):
    return SpatialColorLift(config=configuration()[0], has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True,
                            camera_names=list(CAMERAS), render_gpu_device_id=gpu, hard_reset=False, seed=0)
