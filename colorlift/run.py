"""Run the three fair ACT comparisons using at most the explicit two-GPU allowlist."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import time
from .config import EXPERIMENT, ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--steps", type=int, default=8000)
    args = parser.parse_args()
    if len(args.gpus) > 2 or len(set(args.gpus)) != len(args.gpus):
        raise ValueError("Use one or two distinct physical GPUs only")
    manifest = json.loads((EXPERIMENT / "data" / "manifest.json").read_text())
    assert len([e for e in manifest["episodes"] if e["split"] == "train"]) == 100
    assert len([e for e in manifest["episodes"] if e["split"] == "val"]) == 20
    protocol = {"variants": ["rgb", "mask", "rgb_mask"], "target_color": "red", "train_demos": 100,
                "validation_demos": 20, "updates_per_variant": args.steps, "batch_size": 32,
                "training_seed": 42, "chunk_size": 16, "query_every": 3,
                "test_seeds": list(range(40000, 40030)), "development_seeds": list(range(30000, 30006)),
                "physical_gpu_allowlist": args.gpus, "max_simultaneous_gpus": 2,
                "checkpoint_selection": "minimum held-out prior XYZ L1 + 0.25 phase CE",
                "RGB_input": "2 cameras x RGB", "Mask_input": "2 cameras x red/yellow/blue binary HSV channels",
                "RGB_Mask_input": "2 cameras x 6-channel early fusion; extra conv weights initialized to zero",
                "shared_control": "four monotone OSC primitives; bounded increments, same safety/height/locking priors",
                "expert": "HSV masks + fixed camera calibration, no object-pose action input",
                "manifest_sha256": hashlib.sha256((EXPERIMENT / "data" / "manifest.json").read_bytes()).hexdigest()}
    (EXPERIMENT / "protocol.json").write_text(json.dumps(protocol, indent=2))
    tasks = queue.Queue()
    for name in protocol["variants"]:
        tasks.put(name)
    status = {"started": time.time(), "gpus": args.gpus, "steps": args.steps, "state": "training"}
    (EXPERIMENT / "runner_status.json").write_text(json.dumps(status, indent=2))

    def environment(gpu):
        return os.environ | {"CUDA_VISIBLE_DEVICES": str(gpu), "MUJOCO_EGL_DEVICE_ID": str(gpu),
                             "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "MPLBACKEND": "Agg"}

    def worker(gpu):
        while True:
            try:
                variant = tasks.get_nowait()
            except queue.Empty:
                return
            out = EXPERIMENT / variant
            out.mkdir(exist_ok=True)
            if not (out / "completed.json").exists():
                command = [sys.executable, "-u", "-m", "colorlift.train", "--variant", variant, "--steps", str(args.steps)]
                if (out / "last.pt").exists():
                    command.append("--resume")
                print(f"Training {variant} on physical GPU {gpu}", flush=True)
                with (out / "train.log").open("a") as log:
                    subprocess.run(command, env=environment(gpu), cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            print("Training finished: " + variant, flush=True)

    try:
        with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
            list(pool.map(worker, args.gpus))
        status["state"] = "evaluating"
        (EXPERIMENT / "runner_status.json").write_text(json.dumps(status, indent=2))
        gpu = args.gpus[-1]
        for variant in ["hsv", *protocol["variants"]]:
            out = EXPERIMENT / variant / "eval_test"
            if not (out / "metrics.json").exists():
                with (EXPERIMENT / ("eval_" + variant + ".log")).open("w") as log:
                    subprocess.run([sys.executable, "-u", "-m", "colorlift.evaluate", "--variant", variant,
                                    "--gpu", str(gpu), "--episodes", "30", "--split", "test"],
                                   env=environment(gpu), cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            print("Evaluation finished: " + variant, flush=True)
        subprocess.run([sys.executable, "-m", "colorlift.report"], cwd=ROOT, env=environment(gpu), check=True)
        status["state"] = "complete"
    except Exception as error:
        status["state"], status["error"] = "failed", str(error)
        raise
    finally:
        status["updated"] = time.time()
        (EXPERIMENT / "runner_status.json").write_text(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
