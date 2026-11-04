"""Finish B0 first, then launch paired B1 training and physical checkpoint selection."""
import argparse
import json
import os
import subprocess
import sys
from .spec import ROOT,RUN,INITIAL,SEEDS,freeze,sha
from .evaluate import evaluate,select_model


def train_runs(configurations):
    for start in range(0,len(configurations),2):
        processes=[];handles=[]
        for offset,config in enumerate(configurations[start:start+2]):
            gpu=(1,2)[offset]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),MUJOCO_EGL_DEVICE_ID=str(gpu),OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="1",CUBLAS_WORKSPACE_CONFIG=":4096:8")
            log=(RUN / ("train_"+config["name"]+".log")).open("a");handles.append(log)
            command=[sys.executable,"-u","-m","colorlift.broad_act.train","--name",config["name"],"--seed",str(config["seed"]),"--mode",config["mode"],
                     "--relative",str(config.get("relative",0.)),"--pregrasp",str(config.get("pregrasp",0.))]
            if config.get("b4"):command.append("--b4")
            processes.append(subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT))
        codes=[p.wait() for p in processes]
        for h in handles:h.close()
        assert all(code==0 for code in codes),codes


def stage():
    freeze()
    assert (RUN / "smoke.json").exists() and (RUN / "evaluations/B0_initial/metrics.json").exists()
    files=["spec.py","data.py","model.py","runtime.py","evaluate.py","train.py"]
    snapshot={name:sha(ROOT / "colorlift/broad_act" / name) for name in files}
    marker=RUN / "stage1_source_freeze.json"
    if marker.exists():assert json.loads(marker.read_text())==snapshot
    else:marker.write_text(json.dumps(snapshot,indent=2))
    configurations=[{"name":f"B1_s{seed}","seed":seed,"mode":"random"} for seed in SEEDS]
    train_runs(configurations)
    choices=[select_model(row["name"]) for row in configurations]
    (RUN / "B1_completed.json").write_text(json.dumps({"complete":True,"models":choices,"selection":"physical development closed loop","final_used":False},indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--detach",action="store_true");a=p.parse_args()
    if a.detach:
        with (RUN / "stage1.log").open("a") as log:
            child=subprocess.Popen([sys.executable,"-u","-m","colorlift.broad_act.stage1"],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps({"pid":child.pid,"stage":"B1","physical_GPUs":[1,2]}))
    else:stage()
