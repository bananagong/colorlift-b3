"""Finite approved stage1: seal -> paired training -> lock -> confirm -> analyze."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
from .spec import ROOT,RUN,SEEDS,read,sha,atomic,now,status,verify,all_model_definitions


def module(name,args=(),gpu=None):
    env=dict(os.environ,OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="1")
    if gpu is not None:env.update(CUDA_VISIBLE_DEVICES=str(gpu),MUJOCO_EGL_DEVICE_ID=str(gpu),CUBLAS_WORKSPACE_CONFIG=":4096:8")
    subprocess.run([sys.executable,"-u","-m","colorlift.repro_confirm."+name,*map(str,args)],cwd=ROOT,env=env,check=True)


def train_pair(index,seed):
    processes=[];handles=[]
    for group,gpu in zip(("B2","B3"),((1,2) if index%2==0 else (2,1))):
        name=f"{group}_s{seed}";done=RUN / "models" / name / "training_completed.json"
        if done.exists():assert read(done)["steps"]==6000;continue
        assert not (RUN / "model_lock.json").exists()
        handle=(RUN / ("train_"+name+".log")).open("a");handles.append(handle)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),MUJOCO_EGL_DEVICE_ID=str(gpu),OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="1",CUBLAS_WORKSPACE_CONFIG=":4096:8")
        processes.append(subprocess.Popen([sys.executable,"-u","-m","colorlift.repro_confirm.train","--group",group,"--seed",str(seed)],cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT))
    codes=[p.wait() for p in processes]
    for handle in handles:handle.close()
    assert all(code==0 for code in codes),codes


def main():
    module("statistics",["--self-test"])
    module("prepare")
    verify()
    module("smoke",gpu=1)
    module("seal",["--launch"])
    verify(require_seal=True)
    if not (RUN / "training_started.json").exists():
        atomic(RUN / "training_started.json",{"at_UTC":now(),"cohort_presealed":True,
            "seal_sha256":sha(RUN / "seal.json"),"confirmation_inputs_read":False,"new_training_runs":12})
    from .evaluate import select,evaluate
    selections={}
    for index,seed in enumerate(SEEDS):
        status("paired_training_and_development_selection",active_seed=seed,physical_GPUs=[1,2],fresh_cohort_private=True)
        train_pair(index,seed)
        for group in ("B2","B3"):
            name=f"{group}_s{seed}";selections[name]=select(name)
    atomic(RUN / "all_development_selections.json",{"models":selections,"fresh_confirmation_used":False,"old768_used":False})
    entries={}
    for name,config in all_model_definitions().items():
        path=config["path"]
        assert path.is_file()
        if config["cohort"]=="new":assert sha(path)==selections[name]["sha256"]
        entries[name]={k:v for k,v in config.items() if k!="path"}|{"path":str(path.relative_to(ROOT)),"sha256":sha(path),"controller":"original"}
    marker=RUN / "model_lock.json"
    if marker.exists():assert read(marker)["models"]==entries
    else:atomic(marker,{"locked_at_UTC":now(),"models":entries,"seal_sha256":sha(RUN / "seal.json"),
        "protocol_sha256":sha(RUN / "protocol.json"),"selection_sources":"original development only", "all_seeds_retained":True})
    status("fresh_sealed_confirmation_evaluation",planned_episodes=86016,primary_new_seed_episodes=64512,legacy_supplement_episodes=21504,old768_confirmation_episodes=0)
    metrics=evaluate(all_model_definitions(),"confirm_locked",kind="confirm")
    assert len(metrics)==16 and sum(row["episodes"] for row in metrics.values())==86016
    atomic(RUN / "confirmation_completed.json",{"completed_at_UTC":now(),"total_episodes":86016,
        "metrics":metrics,"fresh_geometries":768,"old768_confirmation_episodes":0})
    status("prespecified_paired_statistical_analysis",confirmation_complete=True)
    module("statistics")
    status("report_and_export",confirmation_complete=True,analysis_complete=True)
    module("results")
    result=read(RUN / "confirmatory_analysis.json")
    status("stage1_completed_waiting_stage2_protocol",verdict=result["verdict"],report=str(RUN / "REPORT.zh-CN.md"),
        stage2_started=False,new_layouts=768,confirmation_episodes=86016)


def guarded_main():
    RUN.mkdir(parents=True,exist_ok=True)
    with (RUN / "coordinator.lock").open("a") as lock:
        try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print("Coordinator already running; no duplicate work started",flush=True);return
        try:main()
        except BaseException as exc:
            status("failed",error=repr(exc),stage2_started=False);raise


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--detach",action="store_true");a=p.parse_args()
    if a.detach:
        with (RUN / "pipeline.log").open("a") as log:
            process=subprocess.Popen([sys.executable,"-u","-m","colorlift.repro_confirm.pipeline"],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print({"pid":process.pid,"physical_GPUs":[1,2],"approved_variant":"B"},flush=True)
    else:guarded_main()
