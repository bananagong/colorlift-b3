"""Train a single color-conditioned ACT with a 4 -> 8 -> 12 chunk curriculum."""
import argparse
from dataclasses import asdict
import json
import os
import random
import time
import numpy as np
import torch
from .conditional import RUN, ConditionalACT, ConditionalDataset, recognition_loss, recognized_masks, configuration
from .config import COLORS
from .model import loss_components
from lift_act.train import save_atomic


def merge_data():
    entries = []
    for shard in range(2):
        assert (RUN / f"data/shard_{shard}_completed.json").exists()
        entries.extend(json.loads((RUN / f"data/manifest_shard_{shard}.json").read_text())["episodes"])
    groups = {}
    for entry in entries:
        groups.setdefault((entry["split"], entry["seed"]), []).append(entry)
    for key, group in groups.items():
        eligible = len(group) == 3 and all(row["success"] and row["collision_steps"] == 0 for row in group)
        for entry in group:
            entry["eligible"] = eligible
        for color in COLORS:
            assert all(np.array_equal(row["initial_positions"][color], group[0]["initial_positions"][color]) for row in group)
    counts = {split: {color: sum(row["split"] == split and row["target_color"] == color and row["eligible"] for row in entries) for color in COLORS} for split in ("train", "val")}
    assert len(set(counts["train"].values())) == 1 and min(counts["train"].values()) >= 150
    assert len(set(counts["val"].values())) == 1 and min(counts["val"].values()) >= 20
    manifest = {"episodes": entries, "counts_by_color": counts, "excluded_scene_groups": sum(not group[0]["eligible"] for group in groups.values()),
                "selection": "retain only complete successful collision-free three-color scene triples for balanced training; all attempts remain recorded"}
    (RUN / "data/manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--batch-size", type=int, default=36)
    args = parser.parse_args()
    assert torch.cuda.device_count() == 1
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    rng = np.random.default_rng(42)
    out = RUN / "model"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "completed.json").exists():
        raise RuntimeError("Preserve completed training; choose another experiment")
    manifest = merge_data()
    train = ConditionalDataset("train")
    val = ConditionalDataset("val", train.stats)
    model = ConditionalACT(12).cuda()
    embedding_initial = model.color_embedding.weight.detach().cpu().clone()
    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters() if "backbones" not in n], "lr": 1e-4},
        {"params": [p for n, p in model.named_parameters() if "backbones" in n], "lr": 1e-5},
    ], weight_decay=1e-4)
    config = {"steps": args.steps, "batch_size": args.batch_size, "seed": 42, "max_chunk": 12,
              "curriculum": [{"start": 1, "end": 4000, "chunk": 4}, {"start": 4001, "end": 8000, "chunk": 8}, {"start": 8001, "end": args.steps, "chunk": 12}],
              "lr": 1e-4, "lr_backbone": 1e-5, "kl_weight": 10, "phase_weight": .25, "recognition_weight": 2,
              "embedding_dim": 32, "robot_state_dim": 16, "action_dim": 7, "perception_pretrain_steps": 1800, "pixel_coordinate_attention_power": 4,
              "recognition_objective": "class-balanced four-way cross entropy: background, red, yellow, blue",
              "input": "two RGB cameras + proprioception and phase + integer target color", "HSV_at_inference": False,
              "counts_by_color": manifest["counts_by_color"], "train_frames": len(train.states), "val_frames": len(val.states),
              "frame_sampling": "color-balanced and phase-balanced", "checkpoint_selection": "minimum validation prior loss within final chunk-12 stage",
              "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "task_config": asdict(configuration()[0]), "expert_config": asdict(configuration()[1])}
    (out / "config.json").write_text(json.dumps(config, indent=2))
    print("CONFIG", json.dumps(config), flush=True)
    started = time.time()
    for step in range(1, 1801):
        states, rgb, colors, actions, pad, masks = train.batch(train.sample_indices(rng, args.batch_size), 4)
        logits, _, _ = model.recognizer(rgb, model.color_embedding.weight, colors)
        loss = recognition_loss(logits, masks)
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        if step % 300 == 0:
            print("PERCEPTION", step, loss.item(), "seconds", time.time()-started, flush=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=1e-6)
    val_ids = val.sample_indices(np.random.default_rng(2027), 288)
    best = float("inf")
    for step in range(1, args.steps+1):
        active = 4 if step <= 4000 else 8 if step <= 8000 else 12
        model.train()
        states, rgb, colors, actions, pad, masks = train.batch(train.sample_indices(rng, args.batch_size), active)
        output, logits, uv = model(states, rgb, colors, actions, pad)
        action_loss, xyz, phase, acc, kl = loss_components(output, actions, pad, 10, .25)
        percept = recognition_loss(logits, masks)
        loss = action_loss+2*percept
        assert torch.isfinite(loss)
        optimizer.zero_grad(set_to_none=True); loss.backward()
        embedding_grad = model.color_embedding.weight.grad.norm().item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step(); scheduler.step()
        if step % 100 == 0 or step == 1:
            row = {"step": step, "active_chunk": active, "loss": loss.item(), "xyz_l1": xyz.item(), "phase_ce": phase.item(),
                   "phase_accuracy": acc.item(), "kl": kl.item(), "recognition_loss": percept.item(), "embedding_grad_norm": embedding_grad,
                   "seconds": time.time()-started}
            print(json.dumps(row), flush=True)
            with (out / "metrics.jsonl").open("a") as f: f.write(json.dumps(row)+"\n")
        if step % 1000 == 0 or step == args.steps:
            model.eval()
            results = []
            with torch.inference_mode():
                for start in range(0, len(val_ids), args.batch_size):
                    states, rgb, colors, actions, pad, masks = val.batch(val_ids[start:start+args.batch_size], active)
                    prediction, logits, uv = model(states, rgb, colors)
                    zero = torch.zeros(len(states), 32, device="cuda")
                    score, xyz, phase, acc, _ = loss_components((prediction[0], prediction[1], (zero, zero)), actions, pad, 0, .25)
                    recognized = recognized_masks(logits)
                    intersection = (recognized & (masks > .5)).sum().item()
                    union = (recognized | (masks > .5)).sum().item()
                    results.append((score.item(), xyz.item(), phase.item(), acc.item(), intersection/max(1, union)))
            avg = np.mean(results, axis=0)
            row = {"step": step, "active_chunk": active, "val_score": avg[0], "val_xyz_l1": avg[1], "val_phase_ce": avg[2], "val_phase_accuracy": avg[3], "val_recognition_iou": avg[4]}
            print("VALIDATION", json.dumps(row), flush=True)
            with (out / "metrics.jsonl").open("a") as f: f.write(json.dumps(row)+"\n")
            payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                       "stats": train.stats, "step": step, "config": config, "val_score": float(avg[0]), "embedding_initial": embedding_initial}
            save_atomic(payload, out / "last.pt")
            save_atomic({k: v for k, v in payload.items() if k not in ("optimizer", "scheduler")}, out / "latest_policy.pt")
            if step in (4000, 8000, args.steps):
                save_atomic({k: v for k, v in payload.items() if k not in ("optimizer", "scheduler")}, out / f"stage_chunk_{active}.pt")
            if active == 12 and avg[0] < best:
                best = float(avg[0])
                save_atomic({k: v for k, v in payload.items() if k not in ("optimizer", "scheduler")}, out / "policy.pt")
    embedding = model.color_embedding.weight.detach().cpu()
    report = {"steps": args.steps, "best_final_stage_validation": best, "seconds": time.time()-started,
              "embedding_change_l2_by_color": (embedding-embedding_initial).norm(dim=1).tolist(),
              "embedding_pairwise_distance": torch.cdist(embedding, embedding).tolist()}
    (out / "completed.json").write_text(json.dumps(report, indent=2))
    print("TRAINING COMPLETE", json.dumps(report), flush=True)


if __name__ == "__main__": main()
