"""Train one visual ACT structure on the fixed ColorLift data and evaluate on held-out episodes."""
import argparse
import json
import os
import random
import time
import numpy as np
import torch
from .config import EXPERIMENT
from .data import Dataset
from .model import ColorACT, VARIANTS, loss_components
from lift_act.train import save_atomic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--demos", type=int, default=100)
    parser.add_argument("--val-demos", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-backbone", type=float, default=1e-5)
    parser.add_argument("--kl-weight", type=float, default=10)
    parser.add_argument("--phase-weight", type=float, default=.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=__import__("pathlib").Path)
    args = parser.parse_args()
    assert torch.cuda.device_count() == 1, "Expose exactly one GPU for each run"
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    rng = np.random.default_rng(args.seed)
    out = args.output.resolve() if args.output else EXPERIMENT / args.variant
    out.mkdir(parents=True, exist_ok=True)
    if (out / "completed.json").exists() and not args.resume:
        raise RuntimeError("Completed run already exists; use --output for a new hyperparameter trial")
    started = time.time()
    train = Dataset("train", args.demos)
    val = Dataset("val", args.val_demos, train.stats)
    print(f"Loaded {len(train.states)} train / {len(val.states)} validation frames", flush=True)
    model = ColorACT(args.variant, args.chunk_size).cuda()
    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters() if "backbones" not in n], "lr": args.lr},
        {"params": [p for n, p in model.named_parameters() if "backbones" in n], "lr": args.lr_backbone},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=1e-6)
    start, best = 0, float("inf")
    if args.resume:
        checkpoint = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start, best = checkpoint["step"], checkpoint["best_val_score"]
        rng.bit_generator.state = checkpoint["numpy_rng"]
        torch.set_rng_state(checkpoint["torch_rng"])
        torch.cuda.set_rng_state(checkpoint["cuda_rng"])
    config = {key: str(value) if isinstance(value, __import__("pathlib").Path) else value for key, value in vars(args).items()} | {"state_dim": 19, "action_dim": 7, "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
                          "input_channels_per_camera": 6 if args.variant == "rgb_mask" else 3,
                          "train_frames": len(train.states), "validation_frames": len(val.states),
                          "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                          "checkpoint_selection": "validation prior XYZ L1 + phase_weight * phase CE"}
    (out / "config.json").write_text(json.dumps(config, indent=2))
    print(json.dumps(config), flush=True)
    val_ids = np.random.default_rng(2026).integers(0, len(val.states), size=256)
    for step in range(start + 1, args.steps + 1):
        model.train()
        states, image, actions, pad = train.batch(rng.integers(0, len(train.states), size=args.batch_size), args.chunk_size, args.variant)
        optimizer.zero_grad(set_to_none=True)
        loss, xyz, phase, acc, kl = loss_components(model(states, image, actions, pad), actions, pad, args.kl_weight, args.phase_weight)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite loss at {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        scheduler.step()
        if step % 100 == 0 or step == 1:
            row = {"step": step, "loss": loss.item(), "xyz_l1": xyz.item(), "phase_ce": phase.item(),
                   "phase_accuracy": acc.item(), "kl": kl.item(), "seconds": round(time.time() - started, 2)}
            print(json.dumps(row), flush=True)
            with (out / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
        if step % 500 == 0 or step == args.steps:
            model.eval()
            scores, xyz_values, phase_values, accuracies = [], [], [], []
            with torch.inference_mode():
                for offset in range(0, len(val_ids), args.batch_size):
                    states, image, actions, pad = val.batch(val_ids[offset:offset + args.batch_size], args.chunk_size, args.variant)
                    prediction = model(states, image)
                    # Prior predictions have no posterior statistics; compute matching reconstruction metrics.
                    fake_mu = torch.zeros(len(states), 32, device="cuda")
                    prior = (prediction[0], prediction[1], (fake_mu, fake_mu))
                    score, xyz, phase, accuracy, _ = loss_components(prior, actions, pad, 0, args.phase_weight)
                    scores.append(score.item()); xyz_values.append(xyz.item()); phase_values.append(phase.item()); accuracies.append(accuracy.item())
            value = float(np.mean(scores))
            improved = value < best
            best = min(best, value)
            payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                       "stats": train.stats, "step": step, "config": config, "val_score": value, "best_val_score": best,
                       "numpy_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state()}
            save_atomic(payload, out / "last.pt")
            if improved:
                save_atomic(payload, out / "best.pt")
            row = {"step": step, "val_score": value, "val_xyz_l1": float(np.mean(xyz_values)),
                   "val_phase_ce": float(np.mean(phase_values)), "val_phase_accuracy": float(np.mean(accuracies)),
                   "best_val_score": best, "seconds": round(time.time() - started, 2)}
            print(json.dumps(row), flush=True)
            with (out / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
    (out / "completed.json").write_text(json.dumps({"steps": args.steps, "best_val_score": best, "seconds": time.time() - started}, indent=2))


if __name__ == "__main__":
    main()
