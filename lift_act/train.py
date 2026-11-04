"""Train original ACT on 20 or 50 demonstrations, with held-out episode validation."""
import argparse
import json
import os
import random
import time
import numpy as np
import torch
from .data import Demonstrations
from .env import EXPERIMENT
from .model import ACT, act_loss


def save_atomic(payload, path):
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demos", type=int, choices=[20, 50], required=True)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if torch.cuda.device_count() != 1:
        raise RuntimeError("This experiment expects exactly one GPU exposed via CUDA_VISIBLE_DEVICES")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    rng = np.random.default_rng(args.seed)
    out = EXPERIMENT / f"act_{args.demos}"
    out.mkdir(parents=True, exist_ok=True)
    train = Demonstrations("train", args.demos)
    val = Demonstrations("val", 10, train.stats)
    model = ACT(args.chunk_size).cuda()
    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters() if "backbones" not in n], "lr": 1e-4},
        {"params": [p for n, p in model.named_parameters() if "backbones" in n], "lr": 1e-5},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=1e-6)
    start, best = 0, float("inf")
    if args.resume:
        ckpt = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start, best = ckpt["step"], ckpt["best_val_prior_l1"]
        rng.bit_generator.state = ckpt["numpy_rng"]
        torch.set_rng_state(ckpt["torch_rng"])
        torch.cuda.set_rng_state(ckpt["cuda_rng"])
    config = vars(args) | {"train_frames": len(train.states), "val_frames": len(val.states),
                          "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "torch": torch.__version__,
                          "state_dim": 12, "action_dim": 4, "kl_weight": 10,
                          "backbone": "ImageNet ResNet18", "hidden_dim": 256,
                          "encoder_layers": 2, "decoder_layers": 3, "feedforward_dim": 1024,
                          "checkpoint_selection": "held-out prior L1, no test episodes"}
    (out / "config.json").write_text(json.dumps(config, indent=2))
    print(json.dumps(config), flush=True)
    val_indices = np.random.default_rng(2026).integers(0, len(val.states), size=256)
    started = time.time()
    log_path = out / "metrics.jsonl"
    for step in range(start + 1, args.steps + 1):
        model.train()
        indices = rng.integers(0, len(train.states), size=args.batch_size)
        states, image, actions, pad = train.batch(indices, args.chunk_size)
        optimizer.zero_grad(set_to_none=True)
        loss, l1, kl = act_loss(model(states, image, actions, pad), actions, pad)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite training loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        scheduler.step()
        if step % 100 == 0 or step == 1:
            row = {"step": step, "loss": loss.item(), "l1": l1.item(), "kl": kl.item(),
                   "elapsed_seconds": round(time.time() - started, 2)}
            print(json.dumps(row), flush=True)
            with log_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
        if step % 500 == 0 or step == args.steps:
            model.eval()
            val_l1 = []
            with torch.inference_mode():
                for offset in range(0, len(val_indices), args.batch_size):
                    states, image, actions, pad = val.batch(val_indices[offset:offset + args.batch_size], args.chunk_size)
                    prediction = model(states, image)[0]
                    valid = (~pad).unsqueeze(-1)
                    value = ((prediction - actions).abs() * valid).sum() / (valid.sum() * 4)
                    val_l1.append(value.item())
            score = float(np.mean(val_l1))
            improved = score < best
            best = min(best, score)
            payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                       "scheduler": scheduler.state_dict(), "step": step, "stats": train.stats,
                       "config": config, "val_prior_l1": score, "best_val_prior_l1": best,
                       "numpy_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                       "cuda_rng": torch.cuda.get_rng_state()}
            save_atomic(payload, out / "last.pt")
            if improved:
                save_atomic(payload, out / "best.pt")
            row = {"step": step, "val_prior_l1": score, "best_val_prior_l1": best,
                   "elapsed_seconds": round(time.time() - started, 2)}
            print(json.dumps(row), flush=True)
            with log_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
    (out / "completed.json").write_text(json.dumps({"steps": args.steps, "best_val_prior_l1": best, "seconds": time.time() - started}, indent=2))
    print("TRAINING COMPLETE", flush=True)


if __name__ == "__main__":
    main()
