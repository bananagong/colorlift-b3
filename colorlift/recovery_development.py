"""Evaluate the recovery candidate only on development scenes before the fresh final tests."""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT / "experiments/colorlift_recovery_v1"


def main():
    while not (RUN / "model/completed.json").exists(): time.sleep(10)
    processes=[]; logs=[]
    for shard, gpu in enumerate((1, 6)):
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), MUJOCO_EGL_DEVICE_ID=str(gpu), OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
        output=(RUN / f"development_{shard}.log").open("w"); logs.append(output)
        processes.append(subprocess.Popen([sys.executable, "-u", "-m", "colorlift.recovery_evaluate", "--gpu", str(gpu), "--shard", str(shard), "--split", "dev"], cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT))
    codes=[process.wait() for process in processes]
    for log in logs: log.close()
    assert codes==[0, 0], codes
    subprocess.run([sys.executable, "-m", "colorlift.recovery_results", "--split", "dev"], cwd=ROOT, check=True)
    print("RECOVERY DEVELOPMENT COMPLETE", flush=True)


if __name__ == "__main__": main()
