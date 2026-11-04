"""Complete the conditional, two-GPU spatial learning experiment after collection."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from .spatial_gate import ROOT, RUN
from .spatial_collect import require_gate


def command(module, arguments=(), env=None):
    print("BEGIN", module, list(arguments), flush=True)
    subprocess.run([sys.executable, "-u", "-m", module, *arguments], cwd=ROOT, env=env, check=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--detach", action="store_true"); p.add_argument("--collection-pid", type=int, required=True)
    args = p.parse_args()
    require_gate()
    if args.detach:
        with (RUN / "pipeline.log").open("a") as log:
            child = subprocess.Popen([sys.executable, "-u", "-m", "colorlift.spatial_pipeline", "--collection-pid", str(args.collection_pid)],
                                     cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({"pid": child.pid, "gate_passed": True, "physical_GPUs": [1, 2]})); return
    while not (RUN / "collection_completed.json").exists():
        os.kill(args.collection_pid, 0)
        time.sleep(5)
    if not (RUN / "data/replay_verified.json").exists(): command("colorlift.spatial_replay", ["--launch"])
    cpu_env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")
    command("colorlift.spatial_data", env=cpu_env)
    render_env = dict(os.environ, CUDA_VISIBLE_DEVICES="1", MUJOCO_EGL_DEVICE_ID="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    command("colorlift.spatial_evaluate", ["--prepare"], render_env)
    files = ["colorlift/spatial_train.py", "colorlift/spatial_data.py", "colorlift/spatial_rollout.py", "colorlift/spatial_env.py", "colorlift/spatial_layouts.py"]
    freeze = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files}
    freeze.update({f"plan_seed{seed}": hashlib.sha256((RUN / f"plans/seed_{seed}.npz").read_bytes()).hexdigest() for seed in (43, 44)})
    (RUN / "training_freeze.json").write_text(json.dumps(freeze, indent=2))
    for seed in (43, 44):
        workers, handles = [], []
        for variant, gpu in (("curriculum", 1), ("shuffled", 2)):
            out = RUN / "models" / f"{variant}_seed{seed}"
            if (out / "completed.json").exists(): continue
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), MUJOCO_EGL_DEVICE_ID=str(gpu), OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1",
                       CUBLAS_WORKSPACE_CONFIG=":4096:8")
            handle = (RUN / f"train_{variant}_{seed}.log").open("a"); handles.append(handle)
            workers.append(subprocess.Popen([sys.executable, "-u", "-m", "colorlift.spatial_train", "--seed", str(seed), "--variant", variant],
                                            cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT))
        codes = [worker.wait() for worker in workers]
        for handle in handles: handle.close()
        assert all(code == 0 for code in codes), codes
    assert all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == freeze[name] for name in files)
    recommended = []
    for path in (RUN / "models").glob("*/completed.json"):
        row = json.loads(path.read_text()); recommended.append((row["best_val"], path.parent.name))
    best = min(recommended)
    (RUN / "recommended_model.json").write_text(json.dumps({"model": best[1], "selection": "lowest held-out balanced validation score; no test results used",
                                                            "val_score": best[0]}, indent=2))
    if not (RUN / "validation_evaluation_completed.json").exists(): command("colorlift.spatial_evaluate", ["--launch", "--validation"])
    if not (RUN / "evaluation_completed.json").exists(): command("colorlift.spatial_evaluate", ["--launch"])
    (RUN / "computation_completed.json").write_text(json.dumps({"complete": True, "gate_trials": 3000, "trained_models": 4,
                                                                "test_trials": 2250, "counterfactual_trials": 1080, "stage_validation_trials": 216}, indent=2))
    print("SPATIAL COMPUTATIONS COMPLETE", flush=True)


if __name__ == "__main__": main()
