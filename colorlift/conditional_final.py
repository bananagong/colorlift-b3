"""Freeze the selected model, run 150 unique tests on two GPUs, audit and report."""
import hashlib
import json
import os
import subprocess
import sys
from .conditional import RUN, ROOT


def finalize(freeze):
    policy=RUN / "model/policy.pt"
    assert hashlib.sha256(policy.read_bytes()).hexdigest() == freeze["sha256_before_final_test"]
    for module in ("colorlift.conditional_audit", "colorlift.conditional_report"):
        subprocess.run([sys.executable, "-m", module], cwd=ROOT, check=True)
    (RUN / "completed.json").write_text(json.dumps({"complete": True, "trained_single_ACT": True, "formal_test_episodes": 150,
                                                    "model_hash_unchanged_during_test": True, "audit_passed": True}, indent=2))
    print("CONDITIONAL ACT STUDY COMPLETE", flush=True)


def main():
    assert (RUN / "development_complete.json").exists()
    if "--finalize-existing" in sys.argv:
        assert (RUN / "eval_test_selected/metrics.json").exists()
        finalize(json.loads((RUN / "final_model_freeze.json").read_text()))
        return
    if (RUN / "eval_test_selected/metrics.json").exists(): raise RuntimeError("Final test already exists")
    policy=RUN / "model/policy.pt"
    freeze={"checkpoint": str(policy), "sha256_before_final_test": hashlib.sha256(policy.read_bytes()).hexdigest(),
            "model_selection": "best held-out prior score in chunk12 stage", "test_seeds": [75000, 75049],
            "episodes_per_color": 50, "maximum_physical_GPUs": 2, "retry_failed_grasps": False}
    (RUN / "final_model_freeze.json").write_text(json.dumps(freeze, indent=2))
    workers=[]
    files=[]
    try:
        for gpu in (2, 4):
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), MUJOCO_EGL_DEVICE_ID=str(gpu), OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
            output=(RUN / f"eval_test_gpu{gpu}.log").open("w")
            files.append(output)
            args=[sys.executable, "-u", "-m", "colorlift.conditional_evaluate", "--gpu", str(gpu), "--split", "test", "--scenes", "50", "--name", "selected", "--cooperate"]
            if gpu == 4: args.append("--reverse")
            workers.append(subprocess.Popen(args, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT))
        codes=[worker.wait() for worker in workers]
        assert codes == [0, 0], codes
    finally:
        for output in files: output.close()
    finalize(freeze)


if __name__ == "__main__": main()
