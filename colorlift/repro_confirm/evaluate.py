"""Frozen development selection and post-lock-only fresh cohort evaluation."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from .spec import ROOT,RUN,DATA,PRIVATE,COLORS,SEEDS,ATTEMPTS,read,sha,atomic,now,once,verify
from ..broad_act.evaluate import score,rank
from ..broad_v2.layouts import recolor


def scenes(kind):
    if kind=="development":
        source=read(DATA / "training_manifest.json")
        return [row["layout"]|{"eval_suite":"validation"} for row in source["layouts"] if row["split"]=="val"]
    assert kind=="confirm" and (RUN / "model_lock.json").exists()
    # This is the first code path allowed to load the new private coordinates.
    manifest=read(PRIVATE / "manifest.json")
    return [row["layout"]|{"eval_suite":row["split"]} for row in manifest["layouts"]]


def expected_cases(kind):
    cases=[]
    for layout in scenes(kind):
        for permutation in (range(3) if kind=="confirm" else range(1)):
            colored=recolor(layout,permutation)
            colors=[colored["colors_by_slot"][layout["spatial_target_slot"]]] if layout["eval_suite"]=="spatial_ood" else COLORS
            for color in colors:cases.append((layout["geometry_id"],permutation,color))
    assert len(cases)==(ATTEMPTS if kind=="confirm" else 144)
    return set(cases)


def worker(plan_path,shard,gpu):
    import torch
    from ..broad_act.model import load_policy
    from .runtime import make_env,episode
    torch.set_num_threads(1)
    verify(require_seal=True)
    plan=read(plan_path);kind=plan["kind"];layouts=scenes(kind)
    first={}
    for index,layout in enumerate(layouts):
        first.setdefault((layout["eval_suite"],layout["category"],layout.get("relation_subtype")),index)
    env=make_env(gpu)
    try:
        for name,config in plan["models"].items():
            path=ROOT / config["path"];assert sha(path)==config["sha256"]
            model,checkpoint=load_policy(path)
            for index,base in enumerate(layouts):
                if index%8!=shard:continue
                for permutation in (range(3) if kind=="confirm" else range(1)):
                    layout=recolor(base,permutation)
                    colors=[layout["colors_by_slot"][base["spatial_target_slot"]]] if base["eval_suite"]=="spatial_ood" else COLORS
                    for color in colors:
                        directory=RUN / "evaluations" / plan["tag"] / name / base["eval_suite"] / base["geometry_id"] / f"p{permutation}" / color
                        representative=(index==first[base["eval_suite"],base["category"],base.get("relation_subtype")] and permutation==0 and color==colors[0])
                        row=episode(env,layout,color,directory,name,model,checkpoint,representative)
                        print("CLOSED_LOOP",plan["tag"],name,index,permutation,color,row["success"],row["eligible"],row["control_steps"],flush=True)
            del model;torch.cuda.empty_cache()
    finally:env.close()


def evaluate(models,tag,kind="development"):
    verify(require_seal=True)
    entries={}
    for name,value in models.items():
        path=Path(value["path"])
        entries[name]={k:v for k,v in value.items() if k!="path"}|{"path":str(path.relative_to(ROOT)),"sha256":sha(path),"controller":"original"}
    if kind=="confirm":
        locked=read(RUN / "model_lock.json")["models"]
        assert entries==locked
        seal=read(RUN / "seal.json")
        assert all(sha(PRIVATE / relative)==digest for relative,digest in seal["sha256"].items())
    plan={"kind":kind,"tag":tag,"models":entries,"old768_used_for_confirmation":False}
    directory=RUN / "evaluation_plans";directory.mkdir(exist_ok=True)
    plan_path=directory / (tag+".json");once(plan_path,plan)
    target=RUN / "evaluations" / tag / "metrics.json"
    if target.exists():return read(target)
    processes=[];handles=[]
    for shard in range(8):
        gpu=(1,2)[shard%2]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),MUJOCO_EGL_DEVICE_ID=str(gpu),OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
        handle=(RUN / f"eval_{tag}_{shard}.log").open("a");handles.append(handle)
        processes.append(subprocess.Popen([sys.executable,"-u","-m","colorlift.repro_confirm.evaluate","--plan",str(plan_path),"--shard",str(shard),"--gpu",str(gpu)],cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT))
    codes=[p.wait() for p in processes]
    for handle in handles:handle.close()
    assert codes==[0]*8,codes
    expected=expected_cases(kind);metrics={}
    for name in entries:
        rows=[read(p) for p in (RUN / "evaluations" / tag / name).rglob("metadata.json")]
        assert len(rows)==len(expected)
        assert {(r["geometry_id"],r["layout"]["color_assignment_index"],r["target_color"]) for r in rows}==expected
        metrics[name]=score(rows)
    atomic(target,metrics);return metrics


def select(name):
    directory=RUN / "models" / name;marker=directory / "selection.json"
    if marker.exists():return read(marker)
    assert not (RUN / "model_lock.json").exists()
    assert read(directory / "training_completed.json")["steps"]==6000
    models={name+f"_step{step}":{"path":directory / f"step_{step}.pt"} for step in (1500,3000,4500,6000)}
    metrics=evaluate(models,"select_"+name)
    losses={row["step"]:row["loss"] for row in [json.loads(line) for line in (directory / "validation.jsonl").read_text().splitlines()]}
    best=max(models,key=lambda key:rank(metrics[key],losses[int(key.rsplit("step",1)[1])]))
    shutil.copy2(models[best]["path"],directory / "policy.pt")
    result={"model":name,"chosen_checkpoint":best,"metrics":metrics[best],"all_checkpoints":metrics,
        "policy_path":str((directory / "policy.pt").relative_to(ROOT)),"sha256":sha(directory / "policy.pt"),
        "selected_at_UTC":now(),"selection_used":"original48-layout development only","fresh_cohort_read":False,"old768_used":False}
    atomic(marker,result);return result


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--plan",type=Path,required=True);p.add_argument("--shard",type=int,required=True);p.add_argument("--gpu",type=int,required=True)
    a=p.parse_args();worker(a.plan,a.shard,a.gpu)
