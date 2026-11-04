"""After all frozen tests finish, compute results and audit before rendering the report."""
import argparse
import json
import os
import subprocess
import sys
import time
from .spatial_gate import ROOT, RUN


def main():
    p = argparse.ArgumentParser(); p.add_argument("--detach", action="store_true"); p.add_argument("--pipeline-pid", type=int, required=True)
    args = p.parse_args()
    if args.detach:
        with (RUN / "finish.log").open("a") as log:
            child = subprocess.Popen([sys.executable, "-u", "-m", "colorlift.spatial_finish", "--pipeline-pid", str(args.pipeline_pid)],
                                     cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({"pid": child.pid})); return
    while not (RUN / "computation_completed.json").exists():
        os.kill(args.pipeline_pid, 0)
        time.sleep(5)
    for module in ("colorlift.spatial_results", "colorlift.spatial_audit", "colorlift.spatial_report"):
        print("BEGIN", module, flush=True)
        subprocess.run([sys.executable, "-u", "-m", module], cwd=ROOT, check=True)
    (RUN / "completed.json").write_text(json.dumps({"complete": True, "gate_trials": 3000, "new_train_geometries": 180,
                                                    "new_val_geometries": 30, "trained_models": 4, "test_and_CF_trials": 3330,
                                                    "phase_validation_trials": 216, "audited": True}, indent=2))
    print("SPATIAL EXPERIMENT COMPLETE", flush=True)


if __name__ == "__main__": main()
