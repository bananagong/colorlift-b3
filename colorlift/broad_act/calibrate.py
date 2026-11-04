"""Measure meter-L1/shared-backbone gradients before freezing the B3 lambda grid."""
import json
import numpy as np
import torch
from .spec import RUN,INITIAL
from .data import Dataset
from .model import make_model,recognition,auxiliary_loss
from ..recovery_train import loss


def main(mode):
    path=RUN / "gradient_calibration.json"
    if path.exists(): return json.loads(path.read_text())
    torch.set_num_threads(2);torch.manual_seed(43001);torch.cuda.manual_seed_all(43001)
    initial=torch.load(INITIAL,map_location="cpu",weights_only=False)
    dataset=Dataset("train",initial["stats"]);plan=dataset.plan(43,mode)
    model=make_model(initial,True,True).cuda().train()
    rows=[]
    for ids in plan[:8]:
        batch=dataset.batch(ids)
        out,logits,_=model(batch["states"],batch["rgb"],batch["colors"],batch["actions"],batch["pad"])
        base,*_=loss(out,batch["actions"],batch["pad"]);base=base+.5*recognition(logits,batch["classes"])
        relative,pre,rotation=auxiliary_loss(model,batch)
        parameter=model.net.input_proj.weight
        norms=[float(torch.autograd.grad(value,parameter,retain_graph=True)[0].norm()) for value in (base,relative,pre)]
        assert min(norms)>1e-10 and np.isfinite(norms).all()
        rows.append({"ACT_loss":float(base.detach()),"relative_L1_m":float(relative.detach()),"pregrasp_L1_m":float(pre.detach()),
                     "shared_ACT_grad":norms[0],"shared_relative_grad":norms[1],"shared_pregrasp_grad":norms[2]})
    ratio=float(np.mean([r["shared_relative_grad"]/r["shared_ACT_grad"] for r in rows]))
    candidates=np.array([.1,.3,1.]);scale=1.
    if np.all(candidates*ratio<.01) or np.all(candidates*ratio>1.):scale=float(np.clip(10**np.round(np.log10(.1/ratio)),.1,100))
    pre_ratio=float(np.mean([r["shared_pregrasp_grad"]/r["shared_ACT_grad"] for r in rows]))
    pre_scale=float(np.clip(10**np.round(np.log10(.1/max(pre_ratio,1e-12))),.1,100)) if not .01<=.3*pre_ratio<=1 else 1.
    result={"mode":mode,"loss_units":{"ACT":"normalized action/CE/KL","relative":"world meters L1","pregrasp":"world meters L1"},
            "initial_candidates":[.1,.3,1.],"relative_to_ACT_shared_gradient_ratio_at_lambda1":ratio,
            "grid_scale":scale,"lambda_candidates":(candidates*scale).tolist(),"pregrasp_lambda":.3*pre_scale,
            "batches":rows,"no_parameters_updated":True,"final_read":False}
    path.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
    return result


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--mode",required=True);a=p.parse_args();main(a.mode)
