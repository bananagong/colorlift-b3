"""One RGB ACT policy conditioned on learned color queries; no HSV at inference."""
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from lift_act.model import ACT
from .config import ROOT, Config, COLORS
from .efficient_expert import ExpertConfig

RUN = ROOT / "experiments" / "colorlift_conditional_v1"


def configuration(color="red"):
    base = Config.load(ROOT / "experiments/colorlift_v1/config.json")
    expert = ExpertConfig(**json.loads((ROOT / "experiments/colorlift_short_expert_v1/expert_config.json").read_text()))
    return replace(base, target_color=color, osc_kp=expert.kp), expert


class LearnedColorVision(nn.Module):
    """Learn pixel keys and compare against the same color embedding used by ACT."""
    def __init__(self):
        super().__init__()
        self.keys = nn.Sequential(nn.Conv2d(3, 32, 1), nn.ReLU(), nn.Conv2d(32, 32, 1))
        self.bias = nn.Parameter(torch.full((3,), -2.0))

    def forward(self, images, embeddings, target):
        b, v, _, h, w = images.shape
        feature = self.keys(images.reshape(b*v, 3, h, w))
        logits = torch.einsum("bdhw,cd->bchw", feature, embeddings) / np.sqrt(32) + self.bias[None, :, None, None]
        logits = logits.reshape(b, v, 3, h, w)
        probabilities = torch.cat((torch.zeros_like(logits[:, :, :1]), logits), dim=2).softmax(2)[:, :, 1:]
        selected = probabilities.gather(2, target[:, None, None, None, None].expand(b, v, 1, h, w))
        attention = selected.pow(4)
        mass = attention.sum((-2, -1)).clamp_min(1e-6)
        x = torch.linspace(-1, 1, w, device=images.device)
        y = torch.linspace(-1, 1, h, device=images.device)
        uv = torch.stack(((attention * x[None, None, None, None, :]).sum((-2, -1)) / mass,
                          (attention * y[None, None, None, :, None]).sum((-2, -1)) / mass), dim=-1).flatten(1)
        return logits, selected, uv


class ConditionalACT(ACT):
    def __init__(self, max_chunk=12, pretrained=True):
        super().__init__(max_chunk, pretrained)
        self.color_embedding = nn.Embedding(3, 32)
        self.recognizer = LearnedColorVision()
        self.net.camera_names = ["color_front", "color_side"]
        # 16-D robot/phase state + learned 32-D color query + 4 learned pixel coordinates.
        self.net.input_proj_robot_state = nn.Linear(52, 256)
        self.net.encoder_joint_proj = nn.Linear(52, 256)
        self.net.encoder_action_proj = nn.Linear(7, 256)
        self.net.action_head = nn.Linear(256, 7)

    def forward(self, states, images, colors, actions=None, is_pad=None):
        logits, selected, uv = self.recognizer(images, self.color_embedding.weight, colors)
        conditioned = torch.cat((states, self.color_embedding(colors), uv), dim=-1)
        # Keep contextual RGB while emphasizing the learned target-color pixels.
        focused = images * (.25 + .75 * selected)
        out = self.net(conditioned, (focused-self.image_mean)/self.image_std, None, actions, is_pad)
        return out, logits, uv


def recognition_loss(logits, masks):
    target = masks.float()
    assert target.shape == logits.shape
    background = 1-target.amax(2, keepdim=True)
    categorical = torch.cat((background, target), dim=2)
    labels = categorical.argmax(2)
    counts = categorical.sum((0, 1, 3, 4)).clamp_min(1)
    scores = torch.cat((torch.zeros_like(logits[:, :, :1]), logits), dim=2)
    return F.cross_entropy(scores.flatten(0, 1), labels.flatten(0, 1), weight=counts.reciprocal())


def recognized_masks(logits):
    scores = torch.cat((torch.zeros_like(logits[:, :, :1]), logits), dim=2)
    classes = scores.argmax(2)
    return F.one_hot(classes, 4)[..., 1:].permute(0, 1, 4, 2, 3).bool()


class FastExecutor:
    """Target-independent OSC adapter. Only the learned ACT supplies target XY."""
    def __init__(self, config, expert):
        self.cfg, self.expert = config, expert
        self.phase = self.phase_steps = self.close_steps = self.stationary_steps = 0
        self.transitions = 0
        self.grasp_xy = None
        self.last_fingers = None
        self.filtered_xy = None
        self.last_error = None

    def action(self, obs, requested_phase, xyz):
        e, cfg = self.expert, self.cfg
        fingers = np.asarray(obs["robot0_gripper_qpos"])
        stable_fingers = self.last_fingers is not None and np.linalg.norm(fingers-self.last_fingers) < e.finger_motion_tolerance
        self.stationary_steps = self.stationary_steps + 1 if stable_fingers else 0
        # Completion gates use own robot state and the previous commanded waypoint.
        if self.last_error is not None:
            reached = np.linalg.norm(self.last_error[:2]) <= e.xy_tolerance and abs(self.last_error[2]) <= e.xyz_tolerance
            advance = self.phase in (0, 1) and reached and self.phase_steps >= 3
            advance |= self.phase == 2 and self.close_steps >= e.minimum_close_steps and self.stationary_steps >= e.stationary_finger_steps
            if advance:
                self.phase += 1
                self.phase_steps = 0
                self.transitions += 1
                self.stationary_steps = 0
                if self.phase == 2:
                    self.grasp_xy = np.asarray(obs["robot0_eef_pos"][:2]).copy()
        xy = np.asarray(xyz[:2], dtype=float).clip(-.15, .15)
        self.filtered_xy = xy if self.filtered_xy is None else .6*xy+.4*self.filtered_xy
        goal = np.r_[self.filtered_xy, cfg.grasp_height]
        if self.phase == 0:
            goal[2] = cfg.table_height + cfg.cube_size/2 + e.pregrasp_clearance
        if self.phase >= 2:
            goal[:2] = self.grasp_xy
            self.close_steps += 1
        if self.phase == 3:
            goal[2] = cfg.table_height + cfg.cube_size/2 + cfg.lift_threshold + e.lift_margin
        delta = goal-np.asarray(obs["robot0_eef_pos"])
        delta[np.abs(delta) < e.deadband] = 0
        maximum = e.align_maximum_step if self.phase == 1 else e.maximum_step
        if np.linalg.norm(delta) > maximum:
            delta *= maximum/np.linalg.norm(delta)
        grip = (-1. if np.abs(fingers).sum() < e.open_gap else 0.) if self.phase < 2 else 1.
        self.last_fingers = fingers.copy()
        self.phase_steps += 1
        control = {"phase": self.phase, "requested_phase": int(requested_phase), "close_steps": self.close_steps,
                   "goal_xyz": goal, "command_translation_m": delta.copy()}
        return np.r_[delta/.05, np.zeros(3), grip], control

    def after_step(self, obs, control):
        self.last_error = np.asarray(obs["robot0_eef_pos"])-control["goal_xyz"]


class ConditionalDataset:
    def __init__(self, split, stats=None):
        metadata = json.loads((RUN / "data/manifest.json").read_text())
        selected = [row for row in metadata["episodes"] if row["split"] == split and row["success"] and row["eligible"]]
        arrays = {key: [] for key in ("states", "images", "masks", "actions")}
        ends, colors = [], []
        offset = 0
        for row in selected:
            with np.load(RUN / row["file"], allow_pickle=False) as d:
                n = len(d["states"])
                for key in arrays:
                    arrays[key].append(d[key][:, :16] if key == "states" else d[key])
                ends.extend([offset+n-1]*n)
                colors.extend([COLORS.index(row["target_color"])]*n)
                offset += n
        self.states, self.images, self.masks, self.actions = (np.concatenate(arrays[key]) for key in arrays)
        self.ends, self.colors = np.array(ends), np.array(colors)
        self.stats = stats or {"state_mean": self.states.mean(0), "state_std": self.states.std(0).clip(.01),
                               "xyz_mean": self.actions[:, :3].mean(0), "xyz_std": self.actions[:, :3].std(0).clip(.01)}
        self.states = ((self.states-self.stats["state_mean"])/self.stats["state_std"]).astype(np.float32)
        self.actions[:, :3] = (self.actions[:, :3]-self.stats["xyz_mean"])/self.stats["xyz_std"]
        self.indices = [np.flatnonzero(self.colors == c) for c in range(3)]
        phases = self.states[:, 12:16].argmax(1)
        self.phase_indices = [[np.flatnonzero((self.colors == c) & (phases == phase)) for phase in range(4)] for c in range(3)]

    def sample_indices(self, rng, batch):
        colors = np.arange(batch) % 3
        rng.shuffle(colors)
        phases = rng.integers(0, 4, batch)
        return np.array([rng.choice(self.phase_indices[c][p]) for c, p in zip(colors, phases)])

    def batch(self, ids, active_chunk, maximum=12):
        future = ids[:, None]+np.arange(maximum)
        pad = (future > self.ends[ids, None]) | (np.arange(maximum)[None] >= active_chunk)
        future = np.minimum(future, self.ends[ids, None])
        convert = lambda x: torch.as_tensor(x, device="cuda")
        return (convert(self.states[ids]), convert(self.images[ids].transpose(0, 1, 4, 2, 3).copy()).float()/255,
                convert(self.colors[ids]), convert(self.actions[future]), convert(pad),
                convert(self.masks[ids].transpose(0, 1, 4, 2, 3).copy()).float()/255)
