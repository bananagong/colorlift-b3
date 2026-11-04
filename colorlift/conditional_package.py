"""Package policy, all evaluations and source; leave raw training images on the server."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import tarfile
from .conditional import RUN, ROOT


def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""): h.update(chunk)
    return h.hexdigest()


def main():
    assert json.loads((RUN / "completed.json").read_text())["complete"]
    source=RUN / "source" / "colorlift"
    source.mkdir(parents=True, exist_ok=True)
    hashes={}
    for path in (ROOT / "colorlift").glob("*.py"):
        shutil.copy2(path, source / path.name)
        hashes[str(path.relative_to(ROOT))]=digest(path)
    (RUN / "source_sha256.json").write_text(json.dumps(hashes, indent=2))
    versions={"python": sys.version}
    for package in ("torch", "torchvision", "mujoco", "robosuite", "numpy", "opencv-python", "imageio"):
        try: versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: versions[package]="not available"
    (RUN / "runtime_versions.json").write_text(json.dumps(versions, indent=2))
    data_hashes={str(path.relative_to(RUN)): digest(path) for path in (RUN / "data").glob("*.npz")}
    (RUN / "data_sha256.json").write_text(json.dumps(data_hashes, indent=2))
    inference={"architecture": "colorlift.conditional.ConditionalACT", "checkpoint": "model/policy.pt", "colors": {"red": 0, "yellow": 1, "blue": 2},
               "RGB_cameras": ["color_front", "color_side"], "RGB_shape": [2, 128, 128, 3], "state_dim": 16, "learned_color_embedding_dim": 32,
               "OSC_kp": 260, "control_hz": 20, "approach_z": .865, "grasp_z": .818, "lift_z": .875, "query_every": 3,
               "action_label": "world XYZ + four-phase logits", "actual_OSC": "normalized xyz increments, zero rotation, gripper command",
               "usage": "CUDA_VISIBLE_DEVICES=4 MUJOCO_EGL_DEVICE_ID=4 python -m colorlift.conditional_demo --gpu 4 --color red --seed 76000 --tag demo"}
    (RUN / "inference_config.json").write_text(json.dumps(inference, indent=2))
    archive=ROOT / "experiments/colorlift_conditional_v1_results.tar.gz"
    count=0
    with tarfile.open(archive, "w:gz", compresslevel=4) as tar:
        for path in sorted(RUN.rglob("*")):
            if not path.is_file(): continue
            relative=path.relative_to(RUN)
            if any(part.startswith("model_pre_") or part == "__pycache__" for part in relative.parts): continue
            if "pre_attention_sharpen" in path.name or "pre_categorical" in path.name: continue
            if relative.parts[0] == "data" and path.suffix in (".npz", ".csv"): continue
            if relative.parts[0] == "model" and path.name not in ("policy.pt", "config.json", "metrics.jsonl", "completed.json"): continue
            if path.suffix == ".lock": raise RuntimeError("A test worker is still running")
            tar.add(path, arcname=str(path.relative_to(ROOT)), recursive=False)
            count+=1
    print(json.dumps({"archive": str(archive), "files": count, "bytes": archive.stat().st_size,
                      "sha256": digest(archive), "raw_demonstration_files_preserved_on_server": len(data_hashes)}))


if __name__ == "__main__": main()
