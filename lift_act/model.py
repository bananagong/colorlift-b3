"""Thin dimension adapter around the original tonyzhaozh/act DETR-VAE."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys
import torch
from torch import nn

UPSTREAM = Path(__file__).resolve().parents[1] / "third_party" / "act"
sys.path.insert(0, str(UPSTREAM))
sys.path.insert(0, str(UPSTREAM / "detr"))


class ACT(nn.Module):
    def __init__(self, chunk_size=20, pretrained=True):
        super().__init__()
        from detr.models.detr_vae import build
        from detr.models import backbone
        args = SimpleNamespace(
            backbone="resnet18", dilation=False, position_embedding="sine",
            masks=False, lr_backbone=1e-5, hidden_dim=256,
            enc_layers=2, dec_layers=3, dim_feedforward=1024, dropout=0.1,
            nheads=8, num_queries=chunk_size, pre_norm=False,
            camera_names=["lift_front", "lift_side"],
        )
        # Upstream uses this function to decide whether to load ImageNet weights.
        with patch.object(backbone, "is_main_process", return_value=pretrained):
            self.net = build(args)
        # The original ALOHA implementation hardcodes 14-D state/actions.
        # Keep its ResNet, transformer, CVAE, latent=32 and query mechanism intact.
        self.net.input_proj_robot_state = nn.Linear(12, 256)
        self.net.encoder_joint_proj = nn.Linear(12, 256)
        self.net.encoder_action_proj = nn.Linear(4, 256)
        self.net.action_head = nn.Linear(256, 4)
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1))

    def forward(self, states, images, actions=None, is_pad=None):
        images = (images - self.image_mean) / self.image_std
        return self.net(states, images, None, actions, is_pad)


def act_loss(output, target, is_pad, kl_weight=10.0):
    prediction, _, (mu, logvar) = output
    valid = (~is_pad).unsqueeze(-1)
    l1 = ((prediction - target).abs() * valid).sum() / (valid.sum() * target.shape[-1]).clamp_min(1)
    kl = (-0.5 * (1 + logvar - mu.square() - logvar.exp())).sum(-1).mean()
    return l1 + kl_weight * kl, l1, kl
