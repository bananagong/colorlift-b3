"""Color-free physical layouts; colors/permutations are a separate random variable."""
import hashlib
import itertools
import json
import numpy as np

COLORS = ("red", "yellow", "blue")
PERMUTATIONS = tuple(itertools.permutations(COLORS))
ANCHORS = np.array([[-.065, -.055], [.065, -.055], [0., .075]])


def geometry_hash(xy, yaw):
    points = sorted(tuple(np.round(np.r_[point, angle], 9)) for point, angle in zip(xy, yaw))
    return hashlib.sha256(json.dumps(points).encode()).hexdigest()


def geometry(seed, level):
    """No color or target instruction is supplied to geometry sampling."""
    rng = np.random.default_rng(seed)
    base_level = level.split("_")[0] if level.startswith("ood_") else level
    for attempt in range(2000):
        if base_level == "easy":
            xy = ANCHORS+rng.uniform(-.02, .02, (3, 2))
            yaw = rng.uniform(-.2, .2, 3)
        elif base_level == "medium":
            distance = rng.uniform(.078, .092)
            angle = rng.uniform(-np.pi, np.pi)
            directions = angle+np.arange(3)*2*np.pi/3
            cluster_anchor = ANCHORS[int(rng.integers(3))]*.45
            xy = cluster_anchor+rng.uniform(-.04, .04, 2)+distance/np.sqrt(3)*np.c_[np.cos(directions), np.sin(directions)]
            xy += rng.uniform(-.002, .002, (3, 2))
            yaw = rng.uniform(-.2, .2, 3)
        elif base_level == "hard":
            sides = rng.choice(4, 3, replace=False)
            xy = rng.uniform(-.09, .09, (3, 2))
            for i, side in enumerate(sides):
                xy[i, side//2] = (1 if side%2 else -1)*rng.uniform(.125, .145)
            yaw = rng.uniform(-.30, .30, 3)
        elif level == "ood_position":
            angle = rng.uniform(-np.pi, np.pi)
            along = np.array([np.cos(angle), np.sin(angle)])
            across = np.array([-along[1], along[0]])
            xy = rng.uniform(-.035, .035, 2)+np.arange(-1, 2)[:, None]*rng.uniform(.081, .095)*along
            xy += rng.uniform(-.002, .002, (3, 1))*across
            yaw = rng.uniform(-.2, .2, 3)
        elif level in ("ood_light", "ood_camera"):
            # New Easy/Medium/Hard geometries, plus a held-out rendering perturbation.
            return geometry(seed, ("easy", "medium", "hard")[seed%3]) | {"level": level, "geometry_source_level": ("easy", "medium", "hard")[seed%3]}
        else:
            raise ValueError(level)
        distances = np.linalg.norm(xy[:, None]-xy[None], axis=-1)+np.eye(3)*100
        minimum = float(distances.min())
        if np.max(np.abs(xy)) > .145 or minimum < .074:
            continue
        if level == "easy" and minimum < .108:
            continue
        if level == "medium" and np.max(distances[distances < 100]) > .099:
            continue
        area = abs(float(np.cross(xy[1]-xy[0], xy[2]-xy[0])))/2
        return {"seed": int(seed), "level": level, "xy": xy.tolist(), "yaw": yaw.tolist(),
                "geometry_id": geometry_hash(xy, yaw), "minimum_center_distance_m": minimum,
                "nearest_neighbor_by_slot_m": distances.min(1).tolist(), "triangle_area_m2": area,
                "max_absolute_coordinate_m": float(np.max(np.abs(xy))), "sampling_attempts": attempt+1,
                "randomization_recipe": "independent vertex jitter +/-0.02m" if level == "easy" else "cluster-center jitter +/-0.04m and random rotation" if level == "medium" else "reachable workspace edge band0.125-0.145m" if level == "hard" else "held-out near-collinear combination",
                "color_independent_geometry": True}
    raise RuntimeError((seed, level, "No feasible non-overlapping layout"))


def descriptor(seed, level, permutation=0):
    layout = geometry(seed, level)
    rng = np.random.default_rng(seed+700000001)
    order = np.array(PERMUTATIONS[int(rng.integers(6))])
    # Three cyclic assignments place every color at every slot for every training geometry.
    assignment = np.roll(order, permutation%3).tolist() if permutation < 3 else list(PERMUTATIONS[permutation%6])
    layout.update(colors_by_slot=assignment, color_assignment_index=permutation,
                  rendering_seed=int(seed+900000001), scenario_id=f'{layout["geometry_id"][:16]}_{level}_{permutation}')
    return layout


def counterfactual_descriptor(seed, level, permutation):
    layout = geometry(seed, level)
    layout.update(colors_by_slot=list(PERMUTATIONS[permutation]), color_assignment_index=permutation,
                  rendering_seed=int(seed+900000001), scenario_id=f'{layout["geometry_id"][:16]}_cf_{permutation}')
    return layout


def audit_descriptors():
    report = {}
    identifiers = set()
    for index, level in enumerate(("easy", "medium", "hard", "ood_position", "ood_light", "ood_camera")):
        generated = [descriptor(1900000+index*10000+i, level) for i in range(1000)]
        assert len({d["geometry_id"] for d in generated}) == 1000
        assert not (identifiers & {d["geometry_id"] for d in generated})
        identifiers.update(d["geometry_id"] for d in generated)
        for d in generated:
            assert d["minimum_center_distance_m"] >= .074
            for slot in range(3):
                assert {descriptor(d["seed"], level, p)["colors_by_slot"][slot] for p in range(3)} == set(COLORS)
        report[level] = {"layouts": len(generated), "min_center_distance_m": min(d["minimum_center_distance_m"] for d in generated),
                         "max_coordinate_m": max(d["max_absolute_coordinate_m"] for d in generated)}
    return {"passed": True, "independent_geometry_count": len(identifiers), "by_level": report,
            "color_assignment_exhaustively_balanced_per_slot": True}
