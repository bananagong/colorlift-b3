"""Reload the coordinator at a completed-job boundary without interrupting training."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from .spec import ROOT,RUN


def main(pid):
    command=Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0",b" ").decode()
    assert "colorlift.broad_act.pipeline" in command
    os.kill(pid,signal.SIGSTOP)
    children=Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
    names=[]
    for child in children:
        args=Path(f"/proc/{child}/cmdline").read_bytes().split(b"\0")
        words=[a.decode() for a in args if a]
        assert "colorlift.broad_act.train" in words,words
        names.append(words[words.index("--name")+1])
    assert names
    try:
        while not all((RUN / "models" / name / "training_completed.json").exists() for name in names):
            for child,name in zip(children,names):
                state=Path(f"/proc/{child}/stat").read_text().split()[2]
                if state=="Z" and not (RUN / "models" / name / "training_completed.json").exists():
                    raise RuntimeError("A training worker exited before its completed marker")
            time.sleep(10)
        os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
        with (RUN / "pipeline.log").open("a") as log:
            replacement=subprocess.Popen([sys.executable,"-u","-m","colorlift.broad_act.pipeline"],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        (RUN / "coordinator_revision.json").write_text(json.dumps({"at_UTC":datetime.now(timezone.utc).isoformat(),"old_pid":pid,"new_pid":replacement.pid,
            "waited_for_completed_models":names,"training_workers_interrupted":False,"training_updates_discarded":0,
            "reason":"load completed B4 deployment-consistent selection helper; B0-B3 rules unchanged"},indent=2))
    except BaseException:
        os.kill(pid,signal.SIGCONT);raise


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--pid",type=int,required=True);a=p.parse_args();main(a.pid)
