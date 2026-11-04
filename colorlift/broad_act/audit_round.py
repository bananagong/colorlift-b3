"""Freeze execution sources and verify the curriculum exposure control."""
from datetime import datetime, timezone
import json
import shutil
import numpy as np
from .spec import ROOT, RUN, SEEDS, sha


def freeze_sources():
    initial = json.loads((RUN / "stage1_source_freeze.json").read_text())
    for name, digest in initial.items():
        assert sha(ROOT / "colorlift/broad_act" / name) == digest, name
    names = sorted(set(initial) | {"stage1.py", "pipeline.py", "calibrate.py", "safety_collect.py",
                                   "safety_data.py", "select_safety.py", "audit_round.py"})
    snapshot = {name: sha(ROOT / "colorlift/broad_act" / name) for name in names}
    marker = RUN / "execution_source_freeze.json"
    if marker.exists():
        assert json.loads(marker.read_text())["sha256"] == snapshot
    else:
        target = RUN / "source_snapshot"; target.mkdir(exist_ok=True)
        for name in names:
            shutil.copy2(ROOT / "colorlift/broad_act" / name, target / name)
        marker.write_text(json.dumps({"frozen_at_UTC": datetime.now(timezone.utc).isoformat(),
                                      "B0_B1_sources_unchanged": True, "sha256": snapshot}, indent=2))


def exposure_audit():
    arrays = {key: np.load(RUN / "cache" / f"train_{key}.npy", mmap_mode="r")
              for key in ("source", "colors", "phases", "difficulty", "valid")}
    results = {}
    for seed in SEEDS:
        paths = {mode: RUN / "plans" / f"{mode}_{seed}.npz" for mode in ("balanced", "curriculum")}
        assert all(path.exists() for path in paths.values())
        plans = {mode: np.load(path)["ids"] for mode, path in paths.items()}
        a, b = plans["balanced"], plans["curriculum"]
        assert np.array_equal(np.sort(a.ravel()), np.sort(b.ravel()))
        assert all(arrays["valid"][ids].all() for ids in plans.values())
        # Sorting encoded source/color/phase tags in each minibatch checks their
        # joint distribution rather than just the three separate marginals.
        tags = {mode: arrays["source"][ids]*12+arrays["colors"][ids]*4+arrays["phases"][ids] for mode, ids in plans.items()}
        assert np.array_equal(np.sort(tags["balanced"], axis=1), np.sort(tags["curriculum"], axis=1))
        results[str(seed)] = {"exact_frame_multiset_equal": True, "per_batch_joint_source_color_phase_equal": True,
            "frame_exposures": int(a.size), "plan_sha256": {mode: sha(path) for mode, path in paths.items()},
            "difficulty_by_training_tenth": {mode: [float(np.mean(arrays["difficulty"][part]))
                for part in np.array_split(ids, 10)] for mode, ids in plans.items()}}
    (RUN / "curriculum_exposure_audit.json").write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    freeze_sources()
    print(json.dumps(exposure_audit(), indent=2), flush=True)
