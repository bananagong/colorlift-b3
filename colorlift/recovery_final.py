"""Freeze both checkpoints and run the pre-specified fresh-scene comparisons."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT / "experiments/colorlift_recovery_v1"
ORIGINAL=ROOT / "experiments/colorlift_conditional_v1"


def fingerprint():
    paths={"old_model": ORIGINAL / "model/policy.pt", "new_model": RUN / "model/policy.pt", "controller": ROOT / "colorlift/recovery_control.py", "protocol": RUN / "protocol.json"}
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}


def main():
    assert (RUN / "evaluations/dev/metrics.json").exists()
    if (RUN / "evaluations/test/metrics.json").exists(): raise RuntimeError("Preserve the completed final test")
    freeze=fingerprint()
    (RUN / "final_freeze.json").write_text(json.dumps(freeze, indent=2))
    workers=[]; outputs=[]
    for shard, gpu in enumerate((1, 6)):
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), MUJOCO_EGL_DEVICE_ID=str(gpu), OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
        log=(RUN / f"final_{shard}.log").open("w"); outputs.append(log)
        workers.append(subprocess.Popen([sys.executable, "-u", "-m", "colorlift.recovery_evaluate", "--gpu", str(gpu), "--shard", str(shard), "--split", "test"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT))
    codes=[worker.wait() for worker in workers]
    for output in outputs: output.close()
    assert codes==[0, 0], codes
    assert freeze==fingerprint()
    for module, arguments in (("colorlift.recovery_results", ["--split", "test"]), ("colorlift.recovery_audit", ["--test"])):
        subprocess.run([sys.executable, "-m", module, *arguments], cwd=ROOT, check=True)
    (RUN / "evaluation_completed.json").write_text(json.dumps({"complete": True, "formal_episodes": 720, "old_failure_replays": 8, "model_and_controller_hashes_unchanged": True}, indent=2))
    print("RECOVERY FINAL EVALUATION COMPLETE", flush=True)


if __name__ == "__main__": main()
