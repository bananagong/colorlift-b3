"""Two-seed curriculum vs exact-exposure shuffled-order ACT fine-tuning."""
import argparse
import json
import os
import random
import time
import numpy as np
import torch
from .conditional import ConditionalACT, recognition_loss
from .recovery_train import loss
from .spatial_gate import RUN, CHECKPOINT
from .spatial_collect import require_gate
from .spatial_data import SpatialDataset, sampling_plan, STEPS, BATCH
from lift_act.train import save_atomic


def main():
    p = argparse.ArgumentParser(); p.add_argument("--seed", type=int, required=True)
    p.add_argument("--variant", choices=("curriculum", "shuffled"), required=True)
    args = p.parse_args()
    require_gate()
    assert (RUN / "data/replay_verified.json").exists()
    assert torch.cuda.device_count() == 1
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    initial = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    train = SpatialDataset("train", initial["stats"])
    val = SpatialDataset("val", initial["stats"])
    plans = sampling_plan(args.seed, train)
    plan = plans[0 if args.variant == "curriculum" else 1]
    model = ConditionalACT(4, pretrained=False).cuda()
    model.load_state_dict(initial["model"])
    groups = [[], [], []]
    for name, parameter in model.named_parameters():
        group = 2 if "backbones" in name else 1 if name.startswith(("recognizer", "color_embedding")) else 0
        groups[group].append(parameter)
    optimizer = torch.optim.AdamW([{"params": groups[0], "lr": 3e-5}, {"params": groups[1], "lr": 1e-5}, {"params": groups[2], "lr": 3e-6}], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STEPS, eta_min=1e-6)
    out = RUN / "models" / f"{args.variant}_seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / "completed.json").exists(), "Preserve completed models"
    config = {"max_chunk": 4, "steps": STEPS, "batch_size": BATCH, "seed": args.seed, "variant": args.variant,
              "initial_checkpoint": str(CHECKPOINT), "controller": "unchanged EventExecutor efficiency 4/2/1/4",
              "normalization": "unchanged baseline stats", "free_space_supervised_prefix": 4, "contact_prefix": 2,
              "lr_ACT": 3e-5, "lr_recognizer_embedding": 1e-5, "lr_backbone": 3e-6,
              "loss": "XYZ L1(2,2,1) +.25 phase CE +10 KL +.5 recognition CE",
              "selection": "minimum fixed balanced validation prior action loss among steps3500/4000/4500/5000",
              "exact_exposure_matched": True, "OOD_in_training": False, "physical_gpu": os.environ["CUDA_VISIBLE_DEVICES"]}
    (out / "config.json").write_text(json.dumps(config, indent=2))
    rng = np.random.default_rng(20301030)
    validation_ids = np.concatenate([val.indices(rng, np.eye(3)[level], 120) for level in range(3)])
    started = time.time(); best = float("inf")
    for step, ids in enumerate(plan, start=1):
        model.train()
        states, rgb, colors, actions, pad, masks = train.batch(ids)
        output, logits, _ = model(states, rgb, colors, actions, pad)
        action_loss, xyz, phase, accuracy, kl = loss(output, actions, pad)
        recognition = recognition_loss(logits, masks)
        objective = action_loss+.5*recognition
        assert torch.isfinite(objective)
        optimizer.zero_grad(set_to_none=True); objective.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step(); scheduler.step()
        if step == 1 or step%100 == 0:
            levels = train.arrays["levels"][ids]
            row = {"step": step, "loss": objective.item(), "XYZ": xyz.item(), "phase_ce": phase.item(), "recognition_ce": recognition.item(),
                   "seconds": time.time()-started, "batch_levels": np.bincount(levels, minlength=3).tolist()}
            with (out / "metrics.jsonl").open("a") as stream: stream.write(json.dumps(row)+"\n")
            print(json.dumps(row), flush=True)
        if step%500 == 0:
            model.eval(); scores = []
            with torch.inference_mode():
                for start in range(0, len(validation_ids), BATCH):
                    states, rgb, colors, actions, pad, masks = val.batch(validation_ids[start:start+BATCH])
                    predicted, _, _ = model(states, rgb, colors)
                    zero = torch.zeros(len(states), 32, device="cuda")
                    value, _, _, _, _ = loss((predicted[0], predicted[1], (zero, zero)), actions, pad)
                    scores.append(value.item())
            score = float(np.mean(scores))
            payload = {"model": model.state_dict(), "config": config, "stats": initial["stats"], "step": step, "val_score": score,
                       "embedding_initial": initial["embedding_initial"]}
            if step in (1500, 3000, 5000): save_atomic(payload, out / f"stage_{step}.pt")
            save_atomic(payload | {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()}, out / "last.pt")
            if step >= 3500 and score < best:
                best = score; save_atomic(payload, out / "policy.pt")
            with (out / "validation.jsonl").open("a") as stream:
                stream.write(json.dumps({"step": step, "balanced_prior_action_loss": score, "best_final_stage": best if np.isfinite(best) else None})+"\n")
            print("VAL", step, score, best, flush=True)
    (out / "completed.json").write_text(json.dumps({"complete": True, "steps": STEPS, "best_val": best, "seconds": time.time()-started,
                                                    "sampled_frames": int(plan.size), "training_seed": args.seed}, indent=2))
    print("SPATIAL TRAIN COMPLETE", args.variant, args.seed, flush=True)


if __name__ == "__main__": main()
