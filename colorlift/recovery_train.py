"""Fine-tune a four-query ACT on clean and corrective expert data with contact-biased sampling."""
import json
import os
import random
import time
import numpy as np
import torch
from .conditional import RUN as ORIGINAL, ConditionalACT, recognition_loss
from .recovery_data import RUN, RecoveryDataset, merge_data, CATEGORIES, PROBABILITIES
from .model import loss_components
from lift_act.train import save_atomic


def loss(output, target, pad):
    _, xyz, phase, acc, kl=loss_components(output, target, pad, 10, .25)
    difference=(output[0]-target).abs()
    valid=(~pad).float()
    motion=((2*difference[..., 0]+2*difference[..., 1]+difference[..., 2])*valid).sum()/(5*valid.sum()).clamp_min(1)
    return motion+.25*phase+10*kl, motion, phase, acc, kl


def main():
    assert torch.cuda.device_count()==1
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True
    random.seed(43); np.random.seed(43); torch.manual_seed(43); torch.cuda.manual_seed_all(43)
    rng=np.random.default_rng(43)
    protocol=json.loads((RUN / "protocol.json").read_text())
    manifest=merge_data()
    checkpoint=torch.load(ORIGINAL / "model/policy.pt", map_location="cpu", weights_only=False)
    train=RecoveryDataset("train", checkpoint["stats"])
    val=RecoveryDataset("val", checkpoint["stats"])
    model=ConditionalACT(4, pretrained=False).cuda()
    weights=dict(checkpoint["model"])
    weights["net.query_embed.weight"]=weights["net.query_embed.weight"][:4]
    weights["net.pos_table"]=weights["net.pos_table"][:, :6]
    model.load_state_dict(weights, strict=True)
    groups=[[], [], []]
    for name, parameter in model.named_parameters():
        group=2 if "backbones" in name else 1 if name.startswith(("recognizer", "color_embedding")) else 0
        groups[group].append(parameter)
    optimizer=torch.optim.AdamW([{"params": groups[0], "lr": 3e-5}, {"params": groups[1], "lr": 1e-5}, {"params": groups[2], "lr": 3e-6}], weight_decay=1e-4)
    steps=5000
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=1e-6)
    out=RUN / "model"; out.mkdir(parents=True, exist_ok=True)
    if (out / "completed.json").exists(): raise RuntimeError("Recovery training already exists")
    config={"max_chunk": 4, "contact_recovery_supervised_prefix": 2, "free_space_prefix": 4, "steps": steps, "batch_size": 36,
            "seed": 43, "initial_checkpoint": str(ORIGINAL / "model/policy.pt"), "normalization": "preserved old training statistics",
            "sampling_probabilities": dict(zip(CATEGORIES, PROBABILITIES.tolist())), "source_sampling": "recovery:new only; other categories old/new 50:50",
            "masked_actions": "all injected offset actions, future horizons crossing injections and episode boundaries",
            "action_labels": "ACT task macro target; new phase0/1 XY uses observed expert target body pose, phase1 nominal grasp Z; temporary IK retreat goals remain separate in raw records",
            "loss": "normalized XYZ L1 with weights(2,2,1) + .25 phase CE +10 KL +.5 recognition CE",
            "lr_ACT": 3e-5, "lr_recognizer_embedding": 1e-5, "lr_backbone": 3e-6,
            "new_data_counts": manifest["eligible_counts"], "train_frames": len(train.states), "train_valid_frames": int(train.valid.sum()),
            "validation_frames": len(val.states), "query_schedule": protocol["selected_query_schedule"], "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "selection": "minimum fixed phase-balanced validation prior action loss"}
    (out / "config.json").write_text(json.dumps(config, indent=2))
    print("CONFIG", json.dumps(config), flush=True)
    validation_ids=val.sample_indices(np.random.default_rng(2030), 360)
    started=time.time(); best=float("inf")
    for step in range(1, steps+1):
        model.train()
        states, rgb, colors, actions, pad, masks=train.batch(train.sample_indices(rng, 36))
        output, logits, uv=model(states, rgb, colors, actions, pad)
        action_loss, xyz, phase, accuracy, kl=loss(output, actions, pad)
        percept=recognition_loss(logits, masks)
        objective=action_loss+.5*percept
        assert torch.isfinite(objective)
        optimizer.zero_grad(set_to_none=True); objective.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step(); scheduler.step()
        if step%100==0 or step==1:
            row={"step": step, "loss": objective.item(), "weighted_xyz_l1": xyz.item(), "phase_ce": phase.item(), "phase_accuracy": accuracy.item(),
                 "recognition_loss": percept.item(), "seconds": time.time()-started, "sampling_counts": dict(zip(CATEGORIES, train.samples.tolist()))}
            print(json.dumps(row), flush=True)
            with (out / "metrics.jsonl").open("a") as stream: stream.write(json.dumps(row)+"\n")
        if step%500==0 or step==steps:
            model.eval(); scores=[]
            with torch.inference_mode():
                for start in range(0, len(validation_ids), 36):
                    states, rgb, colors, actions, pad, masks=val.batch(validation_ids[start:start+36])
                    prediction, _, _=model(states, rgb, colors)
                    zero=torch.zeros(len(states), 32, device="cuda")
                    value, xyz, phase, accuracy, _=loss((prediction[0], prediction[1], (zero, zero)), actions, pad)
                    scores.append(value.item())
            score=float(np.mean(scores))
            payload={"model": model.state_dict(), "config": config, "stats": checkpoint["stats"], "step": step,
                     "val_score": score, "initial_model_step": checkpoint["step"], "embedding_initial": checkpoint["embedding_initial"]}
            save_atomic(payload | {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()}, out / "last.pt")
            if score<best: best=score; save_atomic(payload, out / "policy.pt")
            print("VALIDATION", step, score, best, flush=True)
    (out / "completed.json").write_text(json.dumps({"completed": True, "steps": steps, "seconds": time.time()-started,
                                                    "best_validation_score": best, "sampling_counts": dict(zip(CATEGORIES, train.samples.tolist()))}, indent=2))
    print("RECOVERY TRAINING COMPLETE", flush=True)


if __name__ == "__main__": main()
