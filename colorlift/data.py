"""Identical episode splits and training-only statistics for all visual modalities."""
import numpy as np
import torch
from .config import EXPERIMENT


class Dataset:
    def __init__(self, split, count, stats=None):
        states, rgb, masks, actions, ends = [], [], [], [], []
        offset = 0
        for index in range(count):
            with np.load(EXPERIMENT / "data" / f"{split}_{index:03d}.npz") as item:
                states.append(item["states"])
                rgb.append(item["images"])
                masks.append(item["masks"])
                actions.append(item["actions"])
                length = len(states[-1])
                ends.extend([offset + length - 1] * length)
                offset += length
        self.states, self.rgb, self.masks, self.actions = map(np.concatenate, (states, rgb, masks, actions))
        self.ends = np.asarray(ends)
        self.stats = stats or {
            "state_mean": self.states.mean(0), "state_std": self.states.std(0).clip(.01),
            "xyz_mean": self.actions[:, :3].mean(0), "xyz_std": self.actions[:, :3].std(0).clip(.01),
        }
        self.states = ((self.states - self.stats["state_mean"]) / self.stats["state_std"]).astype(np.float32)
        self.actions[:, :3] = (self.actions[:, :3] - self.stats["xyz_mean"]) / self.stats["xyz_std"]

    def batch(self, indices, chunk, variant, device="cuda"):
        future = indices[:, None] + np.arange(chunk)
        final = self.ends[indices, None]
        pad = future > final
        future = np.minimum(future, final)
        if variant == "rgb":
            image = self.rgb[indices]
        elif variant == "mask":
            image = self.masks[indices]
        else:
            image = np.concatenate([self.rgb[indices], self.masks[indices]], axis=-1)
        return (torch.as_tensor(self.states[indices], device=device),
                torch.as_tensor(image.transpose(0, 1, 4, 2, 3).copy(), device=device).float() / 255,
                torch.as_tensor(self.actions[future], device=device),
                torch.as_tensor(pad, device=device))
