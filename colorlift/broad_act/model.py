"""B3a: auxiliary geometry from shared visual maps before action-conditioned latent."""
import torch
from torch import nn
from torch.nn import functional as F
from ..conditional import ConditionalACT


class GeometryACT(ConditionalACT):
    def __init__(self, max_chunk=4):
        super().__init__(max_chunk, pretrained=False)
        def head(out):
            return nn.Sequential(nn.Linear(2100,256), nn.ReLU(), nn.Linear(256,128), nn.ReLU(), nn.Linear(128,out))
        self.relative_head = head(3)
        self.pregrasp_head = head(9)
        self._maps = []; self.last_aux = {}; self.aux_trace = []; self.record_aux = False
        self.net.input_proj.register_forward_hook(self._capture)

    def _capture(self, module, inputs, output): self._maps.append(output)

    def forward(self, states, images, colors, actions=None, is_pad=None):
        self._maps = []
        out, logits, uv = super().forward(states, images, colors, actions, is_pad)
        assert len(self._maps) == 2
        shared = torch.cat([F.adaptive_avg_pool2d(x, (2,2)).flatten(1) for x in self._maps]+
                           [states, self.color_embedding(colors), uv], dim=1)
        pre = self.pregrasp_head(shared)
        self.last_aux = {"relative": .1*self.relative_head(shared), "pre_relative": .1*pre[:,:3], "pre_rotation": pre[:,3:]}
        if self.record_aux:
            self.aux_trace.append({key: value.detach().cpu().numpy() for key,value in self.last_aux.items()})
        return out, logits, uv


def make_model(checkpoint, auxiliary=False, warm_start=False):
    model = GeometryACT(4) if auxiliary else ConditionalACT(4, pretrained=False)
    result = model.load_state_dict(checkpoint["model"], strict=not (auxiliary and warm_start))
    if auxiliary and warm_start:
        assert not result.unexpected_keys
        assert all(k.startswith(("relative_head.", "pregrasp_head.")) for k in result.missing_keys)
    return model


def load_policy(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = make_model(checkpoint, checkpoint.get("auxiliary_heads", False)).cuda().eval()
    return model, checkpoint


def recognition(logits, classes):
    scores = torch.cat((torch.zeros_like(logits[:,:,:1]), logits), dim=2)
    counts = torch.bincount(classes.flatten(), minlength=4).float().clamp_min(1)
    return F.cross_entropy(scores.flatten(0,1), classes.flatten(0,1), weight=counts.reciprocal())


def auxiliary_loss(model, batch):
    valid = batch["aux_valid"]
    zero = model.last_aux["relative"].sum()*0
    relative = (model.last_aux["relative"][valid]-batch["relative"][valid]).abs().mean() if valid.any() else zero
    pre_valid = valid & (batch["states"][:,12:16].argmax(1)<2)
    # Phase components may be normalized; use one-hot ordering retained by old
    # statistics (all phases have positive scale). The caller supplies raw phase.
    if "phases" in batch: pre_valid = valid & (batch["phases"]<2)
    pre = (model.last_aux["pre_relative"][pre_valid]-batch["pre_relative"][pre_valid]).abs().mean() if pre_valid.any() else zero
    rotation = (model.last_aux["pre_rotation"][pre_valid]-batch["pre_rotation"][pre_valid]).abs().mean() if pre_valid.any() else zero
    return relative, pre, rotation
