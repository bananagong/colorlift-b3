"""Final Spatial-OOD conditions the existing joint proposals without coordinate edits."""
import numpy as np
from .layouts import Proposals, region_ok, relation_ok, footprint_gap, in_regions
from .spec import VALID, WORKSPACE


class ConditionalProposals:
    def __init__(self, seed, category, suite, subtype=None):
        self.suite = suite
        self.base = Proposals(seed, category, "train" if suite == "spatial_ood" else suite, subtype)

    @property
    def index(self): return self.base.index

    def next(self):
        layout, reason = self.base.next()
        if self.suite == "spatial_ood":
            xy, yaw = np.array(layout["xy"]), layout["yaw"]
            if np.abs(xy).max() > WORKSPACE: reason = "workspace"
            elif not region_ok(xy, self.suite): reason = "reserved_spatial_support"
            elif not relation_ok(xy, self.suite): reason = "reserved_relation_support"
            elif footprint_gap(xy, yaw) < VALID["footprint_gap_m"]: reason = "footprint_gap"
            else: reason = None
            layout.update(suite=self.suite, level=self.suite,
                          spatial_target_slot=next((i for i, p in enumerate(xy) if in_regions(p, -VALID["region_inside_margin_m"])), None))
        return layout, reason
