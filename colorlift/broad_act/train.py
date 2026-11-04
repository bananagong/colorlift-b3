"""Matched-budget ACT fine-tuning; physical closed-loop selection runs separately."""
import argparse
import json
import os
import random
import time
import numpy as np
import torch
from .spec import ROOT,RUN,INITIAL,STEPS,BATCH,CHECKPOINTS,freeze,sha
from .data import Dataset
from .model import make_model,recognition,auxiliary_loss
from ..recovery_train import loss as act_loss
from lift_act.train import save_atomic


def train(name,seed,mode,relative_weight=0.,pregrasp_weight=0.,b4=False):
    protocol=freeze(); out=RUN / "models" / name;out.mkdir(parents=True,exist_ok=True)
    if (out / "training_completed.json").exists(): return
    assert torch.cuda.device_count()==1
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    initial=torch.load(INITIAL,map_location="cpu",weights_only=False)
    if b4:
        from .safety_data import SafetyDataset
        dataset=SafetyDataset("train",initial["stats"])
    else: dataset=Dataset("train",initial["stats"])
    val=Dataset("val",initial["stats"])
    plan=dataset.plan(seed,mode)
    auxiliary=relative_weight>0 or pregrasp_weight>0
    model=make_model(initial,auxiliary,warm_start=auxiliary).cuda()
    groups=[[],[],[],[]]
    for key,parameter in model.named_parameters():
        group=3 if key.startswith(("relative_head.","pregrasp_head.")) else 2 if "backbones" in key else 1 if key.startswith(("recognizer.","color_embedding.")) else 0
        groups[group].append(parameter)
    optimizer=torch.optim.AdamW([{"params":p,"lr":lr} for p,lr in zip(groups,[3e-5,1e-5,3e-6,1e-4]) if p],weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=STEPS,eta_min=1e-6)
    config={"name":name,"seed":seed,"sampler":mode,"steps":STEPS,"batch":BATCH,"max_chunk":4,
            "relative_lambda":relative_weight,"pregrasp_lambda":pregrasp_weight,"B4_data":b4,"initial_sha256":sha(INITIAL),
            "selection":"physical closed-loop development success; val loss tie only","forecast_capacity":4,
            "contact_prefix":2,"execution_lengths":[4,2,1,4],"physical_gpu":os.environ["CUDA_VISIBLE_DEVICES"],
            "warm_start_history_caveat":protocol["OOD_interpretation"]}
    (out / "config.json").write_text(json.dumps(config,indent=2))
    rng=np.random.default_rng(930001)
    validation_ids=np.concatenate([rng.choice(np.flatnonzero(val.arrays["category"]==category),90,replace=False) for category in range(4)])
    start_step=0
    if (out / "last.pt").exists():
        previous=torch.load(out / "last.pt",map_location="cpu",weights_only=False)
        model.load_state_dict(previous["model"]);optimizer.load_state_dict(previous["optimizer"]);scheduler.load_state_dict(previous["scheduler"])
        start_step=previous["step"]
        torch.set_rng_state(previous["torch_rng"]);torch.cuda.set_rng_state(previous["cuda_rng"])
    else:
        # Same dropout/CVAE random stream despite additional head initialization.
        torch.manual_seed(seed+40000);torch.cuda.manual_seed_all(seed+40000)
    started=time.time()
    for step in range(start_step+1,STEPS+1):
        model.train();batch=dataset.batch(plan[step-1])
        output,logits,_=model(batch["states"],batch["rgb"],batch["colors"],batch["actions"],batch["pad"])
        base,xyz,phase,accuracy,kl=act_loss(output,batch["actions"],batch["pad"])
        percept=recognition(logits,batch["classes"])
        zero=base*0;relative=pre=rotation=zero
        if auxiliary: relative,pre,rotation=auxiliary_loss(model,batch)
        objective=base+.5*percept+relative_weight*relative+pregrasp_weight*pre+(0.01*rotation if pregrasp_weight else 0)
        assert torch.isfinite(objective)
        optimizer.zero_grad(set_to_none=True);objective.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step();scheduler.step()
        if step==1 or step%100==0:
            row={"step":step,"objective":objective.item(),"ACT":base.item(),"XYZ":xyz.item(),"recognition":percept.item(),
                 "relative_L1_m":relative.item(),"pregrasp_L1_m":pre.item(),"rotation_L1":rotation.item(),"seconds":time.time()-started}
            with (out / "metrics.jsonl").open("a") as stream: stream.write(json.dumps(row)+"\n")
            print(json.dumps(row),flush=True)
        if step%500==0:
            payload={"model":model.state_dict(),"stats":initial["stats"],"config":config,"step":step,"auxiliary_heads":auxiliary,
                     "pregrasp_auxiliary":pregrasp_weight>0,"embedding_initial":initial.get("embedding_initial"),
                     "initial_checkpoint_sha256":sha(INITIAL)}
            save_atomic(payload|{"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state()},out / "last.pt")
            if step in CHECKPOINTS:
                model.eval();scores=[]
                with torch.inference_mode():
                    for start in range(0,len(validation_ids),BATCH):
                        b=val.batch(validation_ids[start:start+BATCH]);pred,_,_=model(b["states"],b["rgb"],b["colors"])
                        z=torch.zeros(len(b["states"]),32,device="cuda")
                        value,*_=act_loss((pred[0],pred[1],(z,z)),b["actions"],b["pad"]);scores.append(value.item())
                score=float(np.mean(scores));payload["val_loss"]=score
                save_atomic(payload,out / f"step_{step}.pt")
                with (out / "validation.jsonl").open("a") as stream:stream.write(json.dumps({"step":step,"loss":score})+"\n")
                print("CHECKPOINT",name,step,score,"await_physical_selection",flush=True)
    (out / "training_completed.json").write_text(json.dumps({"complete":True,"steps":STEPS,"frame_exposures":STEPS*BATCH,"seconds":time.time()-started,"physical_selection_pending":True},indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--name",required=True);p.add_argument("--seed",type=int,required=True);p.add_argument("--mode",choices=("random","balanced","curriculum"),required=True)
    p.add_argument("--relative",type=float,default=0.);p.add_argument("--pregrasp",type=float,default=0.);p.add_argument("--b4",action="store_true")
    a=p.parse_args();train(a.name,a.seed,a.mode,a.relative,a.pregrasp,a.b4)
