"""Local episode-separated demonstration data and training-only normalization."""
import numpy as np
import torch
from .env import EXPERIMENT


class Demonstrations:
    def __init__(self, split, count, stats=None):
        records, ends, offset = [], [], 0
        for index in range(count):
            with np.load(EXPERIMENT / "data" / f"{split}_{index:03d}.npz") as data:
                records.append({key: data[key] for key in ("states", "images", "actions")})
                length = len(data["states"])
                ends.extend([offset + length - 1] * length)
                offset += length
        self.states = np.concatenate([r["states"] for r in records])
        self.images = np.concatenate([r["images"] for r in records])
        self.actions = np.concatenate([r["actions"] for r in records])
        self.ends = np.asarray(ends)
        if stats is None:
            stats = {
                "state_mean": self.states.mean(0), "state_std": self.states.std(0).clip(0.01),
                "action_mean": self.actions.mean(0), "action_std": self.actions.std(0).clip(0.01),
            }
        self.stats = stats
        self.states = ((self.states - stats["state_mean"]) / stats["state_std"]).astype(np.float32)
        self.actions = ((self.actions - stats["action_mean"]) / stats["action_std"]).astype(np.float32)

    def batch(self, indices, chunk, device="cuda"):
        future = indices[:, None] + np.arange(chunk)[None]
        ends = self.ends[indices, None]
        pad = future > ends
        future = np.minimum(future, ends)
        states = torch.as_tensor(self.states[indices], device=device)
        images = torch.as_tensor(self.images[indices].transpose(0, 1, 4, 2, 3).copy(), device=device).float() / 255
        actions = torch.as_tensor(self.actions[future], device=device)
        pad = torch.as_tensor(pad, device=device)
        return states, images, actions, pad
