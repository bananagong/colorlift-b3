"""Offline compatibility, auxiliary-label isolation and finite-gradient checks."""
import numpy as np
import torch
from .spec import RUN,INITIAL,read,atomic,now,verify


def main():
    if (RUN / "offline_smoke.json").exists():return
    verify()
    torch.set_num_threads(2);torch.manual_seed(430001);torch.cuda.manual_seed_all(430001)
    from ..broad_act import data as original_data
    from ..broad_act.model import make_model,recognition,auxiliary_loss
    from ..recovery_train import loss
    original_data.RUN=RUN
    initial=torch.load(INITIAL,map_location="cpu",weights_only=False)
    data=original_data.Dataset("train",initial["stats"])
    ids=np.load(RUN / "plans/balanced_45.npz")["ids"][0,:6]
    batch=data.batch(ids)
    plain=make_model(initial,False).cuda().eval();geo=make_model(initial,True,True).cuda().eval()
    with torch.no_grad():
        a=plain(batch["states"],batch["rgb"],batch["colors"])
        b=geo(batch["states"],batch["rgb"],batch["colors"])
        difference=float((a[0][0]-b[0][0]).abs().max());assert difference<1e-7
        geo(batch["states"],batch["rgb"],batch["colors"],batch["actions"],batch["pad"])
        before=geo.last_aux["relative"].clone()
        geo(batch["states"],batch["rgb"],batch["colors"],batch["actions"].flip(0)*1.5,batch["pad"])
        leak=float((before-geo.last_aux["relative"]).abs().max());assert leak==0.
    geo.train();prediction,logits,_=geo(batch["states"],batch["rgb"],batch["colors"],batch["actions"],batch["pad"])
    base,*_=loss(prediction,batch["actions"],batch["pad"]);relative,_,_=auxiliary_loss(geo,batch)
    objective=base+.5*recognition(logits,batch["classes"])+.1*relative
    assert torch.isfinite(objective);objective.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in geo.parameters())
    atomic(RUN / "offline_smoke.json",{"completed_at_UTC":now(),"base_action_max_difference":difference,
        "auxiliary_posterior_action_leak_max_difference":leak,"finite_objective":float(objective.detach()),
        "finite_backward":True,"optimizer_updates":0,"fresh_confirmation_inputs_read":False})
    print("OFFLINE_COMPATIBILITY_PASSED",difference,leak,flush=True)


if __name__=="__main__":main()
