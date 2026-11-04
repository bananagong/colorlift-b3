"""RGB / three HSV-mask channels / RGB+Mask early-fusion ACT ablation."""
import torch
from torch import nn
from torch.nn import functional as F
from lift_act.model import ACT

VARIANTS = ("rgb", "mask", "rgb_mask")


class ColorACT(ACT):
    def __init__(self, variant="rgb", chunk_size=16, pretrained=True):
        if variant not in VARIANTS:
            raise ValueError(variant)
        super().__init__(chunk_size=chunk_size, pretrained=pretrained)
        self.variant = variant
        self.net.camera_names = ["color_front", "color_side"]
        self.net.input_proj_robot_state = nn.Linear(19, 256)
        self.net.encoder_joint_proj = nn.Linear(19, 256)
        self.net.encoder_action_proj = nn.Linear(7, 256)
        self.net.action_head = nn.Linear(256, 7)
        if variant == "rgb_mask":
            stem = self.net.backbones[0][0].body["conv1"]
            replacement = nn.Conv2d(6, stem.out_channels, stem.kernel_size, stem.stride, stem.padding, bias=False)
            with torch.no_grad():
                replacement.weight.zero_()
                replacement.weight[:, :3] = stem.weight
            self.net.backbones[0][0].body["conv1"] = replacement
            mean, std = [.485, .456, .406, .5, .5, .5], [.229, .224, .225, .5, .5, .5]
        elif variant == "mask":
            mean, std = [.5] * 3, [.5] * 3
        else:
            mean, std = [.485, .456, .406], [.229, .224, .225]
        self.image_mean = torch.tensor(mean).view(1, 1, -1, 1, 1)
        self.image_std = torch.tensor(std).view(1, 1, -1, 1, 1)


def loss_components(output, actions, pad, kl_weight=10, phase_weight=0.25):
    prediction, _, (mu, logvar) = output
    valid = ~pad
    xyz_l1 = ((prediction[..., :3] - actions[..., :3]).abs() * valid[..., None]).sum() / (valid.sum() * 3).clamp_min(1)
    labels = actions[..., 3:].argmax(-1)
    phase_ce = F.cross_entropy(prediction[..., 3:].transpose(1, 2), labels, reduction="none")
    phase_ce = (phase_ce * valid).sum() / valid.sum().clamp_min(1)
    accuracy = (((prediction[..., 3:].argmax(-1) == labels) & valid).sum() / valid.sum().clamp_min(1)).float()
    kl = (-0.5 * (1 + logvar - mu.square() - logvar.exp())).sum(-1).mean()
    return xyz_l1 + phase_weight * phase_ce + kl_weight * kl, xyz_l1, phase_ce, accuracy, kl
