"""Package the recovery model, all evaluation traces/videos and expert labels."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import tarfile
from .config import ROOT

RUN=ROOT / "experiments/colorlift_recovery_v1"


def digest(path):
    value=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""): value.update(chunk)
    return value.hexdigest()


def main():
    assert json.loads((RUN / "completed.json").read_text())["complete"]
    source=RUN / "source/colorlift"; source.mkdir(parents=True, exist_ok=True)
    hashes={}
    for path in (ROOT / "colorlift").glob("*.py"):
        shutil.copy2(path, source / path.name)
        hashes[str(path.relative_to(ROOT))]=digest(path)
    (RUN / "source_sha256.json").write_text(json.dumps(hashes, indent=2))
    data_hashes={str(path.relative_to(RUN)): digest(path) for path in (RUN / "data").glob("*/*.npz")}
    (RUN / "data_sha256.json").write_text(json.dumps(data_hashes, indent=2))
    versions={}
    for package in ("torch", "torchvision", "mujoco", "robosuite", "numpy", "opencv-python", "imageio"):
        try: versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: versions[package]="not available"
    (RUN / "runtime_versions.json").write_text(json.dumps(versions, indent=2))
    config={"architecture": "colorlift.conditional.ConditionalACT(max_chunk=4, pretrained=False)", "checkpoint": "model/policy.pt",
            "color_IDs": {"red": 0, "yellow": 1, "blue": 2}, "input": "two RGB cameras +16 robot/phase values +color embedding",
            "horizon": "forecast capacity4, contact/recovery supervision2, execute2 before replanning", "controller": "RecoveryExecutor, target-independent clearance gate",
            "example": "CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1 python -m colorlift.recovery_demo --gpu 1 --color red --disturbed --seed 83000 --tag demo",
            "teacher_data_on_server": str(RUN / "data"), "delivery": "All videos/CSV/metadata and macro labels included. Raw RGB/mask trajectory NPZ files stay on server; six pilot NPZ examples included."}
    (RUN / "inference_config.json").write_text(json.dumps(config, indent=2))
    archive=ROOT / "experiments/colorlift_recovery_v1_results.tar.gz"
    files=0
    with tarfile.open(archive, "w:gz", compresslevel=4) as tar:
        for path in sorted(RUN.rglob("*")):
            if not path.is_file(): continue
            relative=path.relative_to(RUN)
            if any("pre_macro_labels" in part or part=="__pycache__" for part in relative.parts): continue
            if relative.parts[0]=="model" and path.name not in ("policy.pt", "config.json", "completed.json", "metrics.jsonl"): continue
            if relative.parts[0]=="data" and path.name=="trajectory.npz": continue
            if path.suffix==".lock": raise RuntimeError("A worker is still active")
            tar.add(path, arcname=str(path.relative_to(ROOT)), recursive=False); files+=1
    print(json.dumps({"archive": str(archive), "bytes": archive.stat().st_size, "files": files, "sha256": digest(archive),
                      "expert_raw_npz_files_preserved_on_server": 285}))


if __name__ == "__main__": main()
