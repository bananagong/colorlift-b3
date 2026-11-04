"""Run already-declared pending jobs alongside the four long collection workers."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from .spec import ROOT, RUN
from .generate import jobs


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--parent-pid", type=int, required=True); args = parser.parse_args()
    cmdline = Path(f"/proc/{args.parent_pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    assert "colorlift.broad_v2.generate --launch" in cmdline, cmdline
    # Pause only our coordinator, never its four running training-data workers.
    # This prevents it from duplicating the pending jobs while we run them.
    os.kill(args.parent_pid, signal.SIGSTOP)
    started = time.time()
    protocol = json.loads((RUN / "final_protocol.json").read_text())
    def run(item):
        i, (kind, category, count, seed, subtype) = item
        gpu = (1, 2)[i%2]; label = kind+"_"+(subtype or category)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), MUJOCO_EGL_DEVICE_ID=str(gpu), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        command = [sys.executable, "-u", "-m", "colorlift.broad_v2.generate", "--kind", kind, "--category", category,
                   "--count", str(count), "--seed", str(seed), "--gpu", str(gpu)]
        if subtype: command += ["--subtype", subtype]
        with (RUN / ("assisted_generation_"+label+".log")).open("a") as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        return {"job": label, "exit_code": result.returncode, "physical_gpu": gpu}
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(run, list(enumerate(jobs(protocol)))[4:]))
        (RUN / "assisted_schedule.json").write_text(json.dumps({"results": results, "seconds": time.time()-started,
                                                                 "algorithm_and_seeds_unchanged": True, "max_physical_GPUs": [1, 2]}, indent=2))
        assert all(r["exit_code"] == 0 for r in results), results
    finally:
        os.kill(args.parent_pid, signal.SIGCONT)


if __name__ == "__main__": main()
