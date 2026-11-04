"""A single rigid T-shaped block with two convex collision primitives."""

from pathlib import Path

import numpy as np
from robosuite.models.objects import MujocoXMLObject
from robosuite.utils.mjcf_utils import array_to_string


class TBlockObject(MujocoXMLObject):
    """T block for tabletop pushing; all lengths are full dimensions in meters.

    Local +Y points towards the cap, +Z upwards. The body frame is at the center
    of the axis-aligned bounding box, not the center of mass. Uniform density is
    represented by distributing the requested mass over two non-overlapping
    boxes. By default a single free joint allows translation and rotation.

    Args:
        width: Cap width along X.
        length: Overall T length along Y, including the cap.
        height: Thickness along Z.
        bar_depth: Cap depth along Y.
        stem_width: Stem width along X.
        mass: Total mass in kilograms, independent of dimensions.
        friction: MuJoCo sliding, torsional, and rolling friction coefficients.
        rgba: Visual color, in [0, 1].
        joints: Standard MujocoXMLObject joint specification; None fixes the body.
    """

    def __init__(
        self, name="t_block", *, width=0.20, length=0.20, height=0.025,
        bar_depth=0.05, stem_width=0.05, mass=0.30,
        friction=(0.5, 0.005, 0.0001), rgba=(0.08, 0.48, 0.85, 1.0),
        joints="default",
    ):
        dimensions = np.asarray([width, length, height, bar_depth, stem_width, mass], dtype=float)
        if not np.isfinite(dimensions).all() or np.any(dimensions <= 0):
            raise ValueError("Dimensions and mass must be finite and positive")
        width, length, height, bar_depth, stem_width, mass = dimensions.tolist()
        if bar_depth >= length or stem_width >= width:
            raise ValueError("bar_depth must be less than length; stem_width must be less than width")
        friction = np.asarray(friction, dtype=float)
        rgba = np.asarray(rgba, dtype=float)
        if friction.shape != (3,) or not np.isfinite(friction).all() or np.any(friction < 0):
            raise ValueError("friction must contain three finite nonnegative values")
        if rgba.shape != (4,) or not np.isfinite(rgba).all() or np.any((rgba < 0) | (rgba > 1)):
            raise ValueError("rgba must contain four finite values between zero and one")
        self.width, self.length, self.height = width, length, height
        self.bar_depth, self.stem_width, self.mass = bar_depth, stem_width, mass
        super().__init__(
            str(Path(__file__).parent / "assets" / "objects" / "t_block.xml"),
            name=name, joints=joints, obj_type="all", duplicate_collision_geoms=False,
        )
        cap_area = width * bar_depth
        stem_area = stem_width * (length - bar_depth)
        area = cap_area + stem_area
        segments = {
            "cap": ((width / 2, bar_depth / 2, height / 2),
                    (0, (length - bar_depth) / 2, 0), mass * cap_area / area),
            "stem": ((stem_width / 2, (length - bar_depth) / 2, height / 2),
                     (0, -bar_depth / 2, 0), mass * stem_area / area),
        }
        for segment, (size, pos, segment_mass) in segments.items():
            for kind in ("collision", "visual"):
                geom = self.get_obj().find(f"./geom[@name='{self.naming_prefix}{segment}_{kind}']")
                geom.set("size", array_to_string(size))
                geom.set("pos", array_to_string(pos))
                if kind == "collision":
                    geom.set("mass", str(segment_mass))
                    geom.set("friction", array_to_string(friction))
                else:
                    geom.set("rgba", array_to_string(rgba))
            site = self.get_obj().find(f"./site[@name='{self.naming_prefix}{segment}_center']")
            site.set("pos", array_to_string(pos))
        self.get_obj().find(f"./site[@name='{self.naming_prefix}stem_tip']").set(
            "pos", array_to_string([0, -length / 2, 0]),
        )
        offsets = {
            "bottom_site": [0, 0, -height / 2],
            "top_site": [0, 0, height / 2],
            "horizontal_radius_site": [np.hypot(width / 2, length / 2), 0, 0],
        }
        for site_name, pos in offsets.items():
            self.worldbody.find(f"./body/site[@name='{self.naming_prefix}{site_name}']").set(
                "pos", array_to_string(pos),
            )

    def get_bounding_box_half_size(self):
        return np.array([self.width, self.length, self.height]) / 2

    @property
    def important_sites(self):
        sites = super().important_sites
        sites.update({key: self.naming_prefix + key for key in ("cap_center", "stem_center", "stem_tip")})
        return sites

    @property
    def footprint_vertices(self):
        """Counterclockwise local XY polygon, useful for future Push-T rewards."""
        x, y = self.width / 2, self.length / 2
        sx, shoulder = self.stem_width / 2, y - self.bar_depth
        return np.array([
            [-sx, -y], [sx, -y], [sx, shoulder], [x, shoulder],
            [x, y], [-x, y], [-x, shoulder], [-sx, shoulder],
        ])
