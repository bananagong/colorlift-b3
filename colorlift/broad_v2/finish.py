import argparse
import json
import os
import subprocess
import sys
import time
from .spec import ROOT, RUN


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--detach", action="store_true"); args = parser.parse_args()
    if args.detach:
        with (RUN / "finish.log").open("a") as log:
            child = subprocess.Popen([sys.executable, "-u", "-m", "colorlift.broad_v2.finish"], cwd=ROOT,
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({"pid": child.pid})); return
    while not (RUN / "generation_completed.json").exists(): time.sleep(10)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="1", MUJOCO_EGL_DEVICE_ID="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    for module in ("align_supervision", "data_contract", "audit_supervised", "report", "package"):
        print("BEGIN", module, flush=True)
        subprocess.run([sys.executable, "-u", "-m", "colorlift.broad_v2."+module], cwd=ROOT, env=env, check=True)


if __name__ == "__main__": main()
