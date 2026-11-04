"""Two-GPU bounded experiment launcher; detaches cleanly from SSH on request."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "experiments/colorlift_event_efficiency_v1"


def fingerprint():
    files = ["colorlift/event_control.py", "colorlift/event_efficiency.py", "colorlift/recovery_control.py",
             "colorlift/efficient_expert.py", "colorlift/env.py", "experiments/colorlift_recovery_v1/model/policy.pt"]
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=("dev", "test", "collect", "replay"), required=True)
    p.add_argument("--detach", action="store_true")
    args = p.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    if args.detach:
        with (RUN / f"{args.split}_launch.log").open("a") as log:
            child = subprocess.Popen([sys.executable, "-u", "-m", "colorlift.event_launch", "--split", args.split],
                                     cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({"pid": child.pid, "split": args.split}))
        return
    phase_lock = None
    if args.split in ("collect", "replay"):
        completed = RUN / f"{args.split}_completed.json"
        if completed.exists():
            print("ALREADY COMPLETE", args.split, flush=True)
            return
        phase_lock = RUN / f"{args.split}_running.json"
        try:
            with phase_lock.open("x") as lock:
                json.dump({"pid": os.getpid()}, lock)
        except FileExistsError:
            # A manually overlapped phase may already be working on the same two GPUs.
            # Wait for it instead of starting duplicate writers.
            while not completed.exists():
                owner = json.loads(phase_lock.read_text())["pid"]
                os.kill(owner, 0)
                time.sleep(5)
            print("EXTERNAL PHASE COMPLETE", args.split, flush=True)
            return
    from .event_control import protocol
    freeze = fingerprint()
    if args.split == "test":
        assert (RUN / "dev/metrics.json").exists()
        assert not (RUN / "test/metrics.json").exists(), "Preserve completed formal results"
        (RUN / "protocol.json").write_text(json.dumps(protocol(), indent=2))
        freeze["protocol"] = hashlib.sha256((RUN / "protocol.json").read_bytes()).hexdigest()
        (RUN / "formal_freeze.json").write_text(json.dumps(freeze, indent=2))
    workers, handles = [], []
    for shard, gpu in enumerate((1, 2)):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), MUJOCO_EGL_DEVICE_ID=str(gpu), OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
        log = (RUN / f"{args.split}_{shard}.log").open("a")
        handles.append(log)
        module = "colorlift.event_replay" if args.split == "replay" else "colorlift.event_efficiency"
        command = [sys.executable, "-u", "-m", module, "--gpu", str(gpu), "--shard", str(shard)]
        if args.split != "replay":
            command += ["--split", args.split]
        workers.append(subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT))
    codes = [worker.wait() for worker in workers]
    for handle in handles:
        handle.close()
    assert codes == [0, 0], codes
    if args.split == "test":
        after = fingerprint()
        after["protocol"] = hashlib.sha256((RUN / "protocol.json").read_bytes()).hexdigest()
        assert freeze == after
    else:
        assert freeze == fingerprint()
    if args.split in ("dev", "test"):
        subprocess.run([sys.executable, "-m", "colorlift.event_results", "--split", args.split], cwd=ROOT, check=True)
    (RUN / f"{args.split}_completed.json").write_text(json.dumps({"complete": True, "GPUs": [1, 2], "fingerprints_unchanged": True}, indent=2))
    if phase_lock is not None:
        phase_lock.unlink()
    print("COMPLETED", args.split, flush=True)


if __name__ == "__main__":
    main()
