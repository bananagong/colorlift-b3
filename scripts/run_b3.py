"""Run the frozen B3 policy with the original OSC executor on one recorded layout."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--color", choices=("red", "yellow", "blue"), default="red")
    parser.add_argument("--gpu", type=int, choices=(1, 2), default=1, help="Physical GPU; one GPU per invocation")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models/main_frozen_reference_seed44/policy.pt")
    parser.add_argument("--layout", type=Path, default=ROOT / "examples/development_layout.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.checkpoint.exists():
        parser.error("Download a checkpoint first: python scripts/download_artifacts.py --profile main")
    digest = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    manifest = json.loads((ROOT / "publication/release_manifest.json").read_text(encoding="utf-8"))
    if digest not in {item["sha256"] for item in manifest["models"]}:
        parser.error("Checkpoint is not one of the eight archived, verified B3 policies")
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), MUJOCO_EGL_DEVICE_ID=str(args.gpu),
                      MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    import torch
    from colorlift.broad_act import runtime
    from colorlift.broad_act.model import load_policy
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        parser.error("The frozen simulator runner requires a compatible NVIDIA CUDA/EGL environment")
    directory = (args.output or ROOT / "outputs" / ("b3_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))).resolve()
    if directory.exists():
        parser.error("Output already exists; choose a new directory to preserve previous episodes")
    layout = json.loads(args.layout.read_text(encoding="utf-8"))
    model, checkpoint = load_policy(args.checkpoint)
    runtime.RUN = directory.parent
    env = runtime.make_env(args.gpu)
    try:
        result = runtime.episode(env, layout, args.color, directory, "B3_frozen", model, checkpoint,
                                 controller="original", representative=True)
    finally:
        env.close()
    print(json.dumps({"output": str(directory), "checkpoint_sha256": digest, "success": result["success"],
                      "contact_qualified": result["eligible"], "wrong_color": result["wrong_color_grasp"],
                      "control_steps": result["control_steps"]}, indent=2))


if __name__ == "__main__":
    main()
