"""Exchangeable six-coordinate proposals, separate colors, explicit support restrictions."""
import hashlib
import itertools
import json
import numpy as np
from scipy.stats import qmc
from .spec import WORKSPACE as W, REGIONS, VALID, COLORS

PERMS = list(itertools.permutations(COLORS))


def geometry_id(xy):
    # A different yaw/color/body ordering does not create an independent geometry.
    canonical = sorted(tuple(np.round(p, 3)) for p in xy)
    return hashlib.sha256(json.dumps(canonical).encode()).hexdigest()


def features(xy):
    xy = np.asarray(xy)
    distances = [float(np.linalg.norm(xy[i]-xy[j])) for i, j in ((0, 1), (0, 2), (1, 2))]
    area2 = abs(float(np.cross(xy[1]-xy[0], xy[2]-xy[0])))
    return {"distances_m": sorted(distances), "minimum_center_distance_m": min(distances),
            "area_ratio": area2/max(max(distances)**2, 1e-12), "edge_distance_m": float(W-np.abs(xy).max())}


def in_regions(point, expansion=0.):
    x, y = point
    return any(x0-expansion <= x <= x1+expansion and y0-expansion <= y <= y1+expansion
               for x0, x1, y0, y1 in REGIONS)


def region_ok(xy, suite):
    if suite == "spatial_ood":
        inside = [in_regions(p, -VALID["region_inside_margin_m"]) for p in xy]
        clear = [not in_regions(p, VALID["region_guard_m"]) for p in xy]
        return sum(inside) == 1 and all(a or b for a, b in zip(inside, clear))
    return not any(in_regions(p, VALID["region_guard_m"]) for p in xy)


def relation_ok(xy, suite, subtype=None):
    f = features(xy)
    if suite != "relation_ood":
        return f["minimum_center_distance_m"] >= .080 and f["area_ratio"] >= .080
    if subtype == "near_gap":
        return .062 <= f["minimum_center_distance_m"] <= .074 and f["area_ratio"] >= .080
    return f["minimum_center_distance_m"] >= .080 and f["area_ratio"] <= .025


def square(xy, yaw, half=.02):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[-half, -half], [half, -half], [half, half], [-half, half]]) @ np.array([[c, s], [-s, c]])+xy


def footprint_gap(xy, yaw):
    polygons = [square(p, y) for p, y in zip(xy, yaw)]
    worst = 100.
    for i, j in ((0, 1), (0, 2), (1, 2)):
        a, b = polygons[i], polygons[j]
        axes = []
        for poly in (a, b):
            for edge in (poly[1]-poly[0], poly[2]-poly[1]):
                axes.append(np.array([-edge[1], edge[0]])/np.linalg.norm(edge))
        gaps = [max(float((b@v).min()-(a@v).max()), float((a@v).min()-(b@v).max())) for v in axes]
        worst = min(worst, max(gaps))
    return worst


class Proposals:
    def __init__(self, seed, category, suite="train", subtype=None):
        self.seed, self.category, self.suite, self.subtype = seed, category, suite, subtype
        self.rng = np.random.default_rng(seed)
        # Draw an entire power-of-two block. Validity rejection subsequently breaks
        # the formal Sobol balance guarantee; accepted-set coverage is measured.
        self.sobol = qmc.Sobol(6, scramble=True, rng=seed).random_base2(16)
        self.index = 0

    def next(self):
        k = self.index; self.index += 1
        rng = self.rng
        xy = rng.uniform(-W, W, (3, 2))
        if self.category == "sobol": xy = -W+2*W*self.sobol[k].reshape(3, 2)
        if self.category == "near" or self.suite == "relation_ood" and self.subtype == "near_gap":
            a, b, c = rng.permutation(3)
            center = rng.uniform(-.105, .105, 2)
            angle = rng.uniform(-np.pi, np.pi)
            distance = rng.uniform(.062, .074) if self.suite == "relation_ood" else rng.uniform(.080, .100)
            delta = distance*np.array([np.cos(angle), np.sin(angle)])
            xy[a], xy[b] = center-delta/2, center+delta/2
        if self.category == "edge_adverse":
            a, b, c = rng.permutation(3); axis = int(rng.integers(2))
            xy[a, axis] = rng.choice([-1, 1])*rng.uniform(.115, W)
            # A nearby distractor with unrestricted bearing challenges the open
            # gripper corridor; invalid bearings are rejected by the robot check.
            angle = rng.uniform(-np.pi, np.pi)
            xy[b] = xy[a]+rng.uniform(.082, .115)*np.array([np.cos(angle), np.sin(angle)])
        if self.suite == "spatial_ood":
            # Force exactly one uniformly chosen physical slot into a reserved
            # continuous region, then reject any distractor in its guard band.
            slot = int(rng.integers(3)); rect = REGIONS[int(rng.integers(len(REGIONS)))]
            x0, x1, y0, y1 = rect; pad = VALID["region_inside_margin_m"]
            xy[slot] = [rng.uniform(x0+pad, x1-pad), rng.uniform(y0+pad, y1-pad)]
        if self.suite == "relation_ood" and self.subtype == "aligned":
            angle = rng.uniform(-np.pi, np.pi)
            along = np.array([np.cos(angle), np.sin(angle)])
            across = np.array([-along[1], along[0]])
            distance = rng.uniform(.084, .12)
            xy = rng.uniform(-.025, .025, 2)+np.arange(-1, 2)[:, None]*distance*along+rng.uniform(-.001, .001, (3, 1))*across
            xy = xy[rng.permutation(3)]
        yaw = rng.uniform(-.35, .35, 3)
        reason = None
        if np.abs(xy).max() > W: reason = "workspace"
        elif not region_ok(xy, self.suite): reason = "reserved_spatial_support"
        elif not relation_ok(xy, self.suite, self.subtype): reason = "reserved_relation_support"
        elif footprint_gap(xy, yaw) < VALID["footprint_gap_m"]: reason = "footprint_gap"
        gid = geometry_id(xy)
        colors = np.random.default_rng(self.seed*100003+k+710000000).permutation(COLORS).tolist()
        layout = {"seed": int(self.seed*100000+k), "stream_seed": self.seed, "proposal_index": k,
                  "level": self.suite, "suite": self.suite, "category": self.category, "relation_subtype": self.subtype,
                  "xy": xy.tolist(), "yaw": yaw.tolist(), "colors_by_slot": colors, "color_assignment_index": 0,
                  "geometry_id": gid, "rendering_seed": int(self.seed*100000+k+900000001),
                  "scenario_id": gid[:16]+"_"+self.suite, "features": features(xy),
                  "spatial_target_slot": next((i for i, p in enumerate(xy) if in_regions(p, -.003)), None) if self.suite == "spatial_ood" else None}
        return layout, reason


def recolor(layout, permutation):
    return layout | {"colors_by_slot": np.roll(np.array(layout["colors_by_slot"]), permutation).tolist(), "color_assignment_index": permutation}
