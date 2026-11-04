"""Reusable cube asset; size is the full edge length in meters."""
from pathlib import Path
import numpy as np
from robosuite.models.objects import MujocoXMLObject
from robosuite.utils.mjcf_utils import array_to_string


class CubeObject(MujocoXMLObject):
    def __init__(self, name="cube", *, size=0.04, mass=0.064,
                 friction=(0.8, 0.005, 0.0001), rgba=(0.85, 0.12, 0.12, 1), joints="default"):
        if not np.isfinite([size, mass]).all() or min(size, mass) <= 0:
            raise ValueError("size and mass must be finite and positive")
        friction, rgba = np.asarray(friction), np.asarray(rgba)
        if friction.shape != (3,) or not np.isfinite(friction).all() or np.any(friction < 0):
            raise ValueError("friction must be three finite nonnegative values")
        if rgba.shape != (4,) or not np.isfinite(rgba).all() or np.any((rgba < 0) | (rgba > 1)):
            raise ValueError("rgba must contain four values in [0, 1]")
        self.size, self.mass = float(size), float(mass)
        super().__init__(str(Path(__file__).parent / "assets" / "objects" / "cube.xml"),
                         name=name, joints=joints, obj_type="all", duplicate_collision_geoms=False)
        for geom in self.get_obj().iter("geom"):
            geom.set("size", array_to_string([size / 2] * 3))
            if geom.get("group") == "0":
                geom.set("mass", str(mass))
                geom.set("friction", array_to_string(friction))
            else:
                geom.set("rgba", array_to_string(rgba))
        for key, value in {"bottom_site": [0, 0, -size / 2], "top_site": [0, 0, size / 2],
                           "horizontal_radius_site": [np.sqrt(2) * size / 2, 0, 0]}.items():
            self.worldbody.find(f"./body/site[@name='{self.naming_prefix}{key}']").set("pos", array_to_string(value))

    def get_bounding_box_half_size(self):
        return np.full(3, self.size / 2)
