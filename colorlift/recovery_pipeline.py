"""Finish the frozen-weight study, then train only after the recovery datasets are complete."""
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT / "experiments/colorlift_recovery_v1"


def main():
    sweep=RUN / "execution_sweep"
    while not all((sweep / f"shard_{i}_completed.json").exists() for i in (0, 1)):
        print("WAITING_FOR_FIXED_CHECKPOINT_SWEEP", flush=True); time.sleep(15)
    with (RUN / "sweep_analysis.log").open("w") as log:
        subprocess.run([sys.executable, "-m", "colorlift.recovery_analysis"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    print("EXECUTION_LENGTH_STUDY_COMPLETE", flush=True)
    while not all((RUN / f"data/shard_{i}_completed.json").exists() for i in (0, 1)):
        print("WAITING_FOR_RECOVERY_DATA", flush=True); time.sleep(15)
    env=dict(os.environ, CUDA_VISIBLE_DEVICES="1", MUJOCO_EGL_DEVICE_ID="1", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
    print("START_RECOVERY_FINETUNING", flush=True)
    with (RUN / "train.log").open("w") as log:
        subprocess.run([sys.executable, "-u", "-m", "colorlift.recovery_train"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    print("RECOVERY_FINETUNING_COMPLETE_READY_FOR_DEVELOPMENT", flush=True)


if __name__ == "__main__": main()
