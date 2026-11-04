"""Run collection-dependent training and development evaluations on at most two GPUs."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "experiments/colorlift_conditional_v1"


def execute(module, arguments, gpu, log):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), MUJOCO_EGL_DEVICE_ID=str(gpu), OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
    with (RUN / log).open("w") as output:
        subprocess.run([sys.executable, "-u", "-m", module, *arguments], cwd=ROOT, env=env,
                       stdout=output, stderr=subprocess.STDOUT, check=True)


def main():
    while not all((RUN / name).exists() for name in ("data/shard_0_completed.json", "data/shard_1_completed.json", "verification/preflight.json")):
        print("WAITING_FOR_COLLECTION", time.time(), flush=True)
        time.sleep(30)
    print("TRAINING_START", flush=True)
    execute("colorlift.conditional_train", [], 2, "train.log")
    print("TRAINING_DONE", flush=True)
    # Development scenes are separate from train/validation and final test scenes.
    for chunk in (4, 8):
        execute("colorlift.conditional_evaluate", ["--gpu", "4", "--split", "dev", "--scenes", "12", "--name", f"chunk{chunk}",
                 "--checkpoint", str(RUN / f"model/stage_chunk_{chunk}.pt")], 4, f"eval_dev_chunk{chunk}.log")
    execute("colorlift.conditional_evaluate", ["--gpu", "4", "--split", "dev", "--scenes", "12", "--name", "selected"], 4, "eval_dev_selected.log")
    (RUN / "development_complete.json").write_text(json.dumps({"complete": True, "requires_final_evaluation": True}))
    print("DEVELOPMENT_DONE_READY_FOR_FINAL_TEST", flush=True)


if __name__ == "__main__": main()
