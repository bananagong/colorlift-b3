"""Finish the authorized evaluation, expert collection, physical replay and audit."""
import argparse
import json
import subprocess
import sys
from .event_launch import ROOT, RUN


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--detach", action="store_true")
    args = p.parse_args()
    if args.detach:
        with (RUN / "pipeline.log").open("a") as log:
            process = subprocess.Popen([sys.executable, "-u", "-m", "colorlift.event_pipeline"], cwd=ROOT,
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({"pid": process.pid, "GPUs": [1, 2]}))
        return
    for split in ("test", "collect", "replay"):
        if (RUN / f"{split}_completed.json").exists():
            continue
        print("BEGIN", split, flush=True)
        subprocess.run([sys.executable, "-u", "-m", "colorlift.event_launch", "--split", split], cwd=ROOT, check=True)
    for module in ("colorlift.event_audit", "colorlift.event_report"):
        print("BEGIN", module, flush=True)
        subprocess.run([sys.executable, "-u", "-m", module], cwd=ROOT, check=True)
    (RUN / "completed.json").write_text(json.dumps({"complete": True, "formal_episodes": 900, "candidate_episodes": 210,
                                                    "checkpoints_retrained": False, "data_requires_successful_physical_replay": True}, indent=2))
    print("EVENT EFFICIENCY PIPELINE COMPLETE", flush=True)


if __name__ == "__main__":
    main()
