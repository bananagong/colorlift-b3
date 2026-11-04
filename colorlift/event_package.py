"""Deliver full accepted expert data, all traces/videos and the frozen checkpoint."""
import hashlib
import importlib.metadata
import json
import shutil
import tarfile
from .event_efficiency import ROOT, RUN, CHECKPOINT


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            value.update(chunk)
    return value.hexdigest()


def main():
    assert json.loads((RUN / "completed.json").read_text())["complete"]
    source = RUN / "source/colorlift"
    source.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for path in (ROOT / "colorlift").glob("*.py"):
        shutil.copy2(path, source / path.name)
        hashes[str(path.relative_to(ROOT))] = digest(path)
    (RUN / "source_sha256.json").write_text(json.dumps(hashes, indent=2))
    frozen = RUN / "frozen_checkpoint"
    frozen.mkdir(exist_ok=True)
    shutil.copy2(CHECKPOINT, frozen / "policy.pt")
    assert digest(CHECKPOINT) == digest(frozen / "policy.pt")
    (frozen / "origin.json").write_text(json.dumps({"original": str(CHECKPOINT), "sha256": digest(CHECKPOINT),
                                                    "unchanged_weights": True, "retrained_this_experiment": False}, indent=2))
    (RUN / "data_sha256.json").write_text(json.dumps({str(p.relative_to(RUN)): digest(p) for p in (RUN / "data").glob("*/*.npz")}, indent=2))
    versions = {name: importlib.metadata.version(name) for name in ("torch", "torchvision", "mujoco", "robosuite", "numpy", "opencv-python", "imageio")}
    (RUN / "runtime_versions.json").write_text(json.dumps(versions, indent=2))
    inference = {"architecture": "ConditionalACT(max_chunk=4)", "checkpoint": "frozen_checkpoint/policy.pt", "colors": {"red": 0, "yellow": 1, "blue": 2},
                 "controller": "EventExecutor", "scheme": "efficiency", "execution_lengths": [4, 2, 1, 4], "tactile_sensor_required": True,
                 "example": "CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1 python -m colorlift.event_demo --gpu 1 --color red --seed 88000 --scheme efficiency --tag demo",
                 "teacher_raw_data_included": True, "note": "Fixed ACT weights; stage/contact rules changed. Inspect comparison before selecting deployment scheme."}
    (RUN / "inference_config.json").write_text(json.dumps(inference, indent=2))
    archive = ROOT / "experiments/colorlift_event_efficiency_v1_results.tar.gz"
    count = 0
    with tarfile.open(archive, "w:gz", compresslevel=4) as tar:
        for path in sorted(RUN.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(RUN)
            if "__pycache__" in relative.parts:
                continue
            # Candidates are duplicates; admitted data contains the byte-identical raw NPZ.
            if relative.parts[0] == "candidates" and path.name == "trajectory.npz":
                continue
            tar.add(path, arcname=str(path.relative_to(ROOT)), recursive=False)
            count += 1
    package = {"archive": str(archive), "bytes": archive.stat().st_size, "files": count, "sha256": digest(archive),
               "accepted_expert_raw_npz_included": True, "candidate_raw_npz_preserved_on_server": True}
    (RUN / "package.json").write_text(json.dumps(package, indent=2))
    print(json.dumps(package, indent=2))


if __name__ == "__main__":
    main()
