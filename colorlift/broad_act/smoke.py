"""Check checkpoint compatibility, masked batches and action-independent aux features."""
import json
import numpy as np
import torch
from .spec import RUN,INITIAL
from .data import Dataset
from .model import make_model,recognition,auxiliary_loss
from ..recovery_train import loss


def main():
    torch.set_num_threads(2)
    initial=torch.load(INITIAL,map_location="cpu",weights_only=False)
    data=Dataset("train",initial["stats"])
    batch=data.batch(data.pools[0,0,0][:12])
    plain=make_model(initial).cuda().eval()
    geometry=make_model(initial,True,True).cuda().eval()
    with torch.no_grad():
        a=plain(batch["states"],batch["rgb"],batch["colors"])[0][0]
        b=geometry(batch["states"],batch["rgb"],batch["colors"])[0][0]
        first=geometry.last_aux["relative"].clone()
        geometry(batch["states"],batch["rgb"],batch["colors"],batch["actions"],batch["pad"])
        second=geometry.last_aux["relative"].clone()
    action_error=float((a-b).abs().max());leak_error=float((first-second).abs().max())
    assert action_error<1e-6 and leak_error<1e-7
    geometry.train()
    output,logits,_=geometry(batch["states"],batch["rgb"],batch["colors"],batch["actions"],batch["pad"])
    base,*_=loss(output,batch["actions"],batch["pad"])
    rel,pre,rot=auxiliary_loss(geometry,batch)
    objective=base+.5*recognition(logits,batch["classes"])+.3*rel
    objective.backward()
    assert torch.isfinite(objective) and geometry.relative_head[0].weight.grad is not None
    result={"old_action_output_max_difference":action_error,"auxiliary_posterior_action_leak_max_difference":leak_error,
            "finite_objective":float(objective),"relative_L1_m":float(rel),"backward_passed":True,
            "train_cache_summary":json.loads((RUN / "cache/completed.json").read_text())["summary"]}
    (RUN / "smoke.json").write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":main()
