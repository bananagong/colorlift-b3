"""Check the official ACT adapter's real GPU forward/backward and inference shapes."""
import json
import numpy as np
import torch
from .data import Demonstrations
from .model import ACT, act_loss
from .env import EXPERIMENT


def main():
    assert torch.cuda.device_count() == 1
    torch.set_num_threads(4)
    data = Demonstrations("train", 20)
    model = ACT().cuda()
    states, images, actions, pad = data.batch(np.array([0, 10]), 20)
    output = model(states, images, actions, pad)
    loss, l1, kl = act_loss(output, actions, pad)
    assert output[0].shape == (2, 20, 4) and torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    model.eval()
    with torch.inference_mode():
        inference = model(states, images)[0]
    assert inference.shape == (2, 20, 4) and torch.isfinite(inference).all()
    report = {"passed": True, "torch": torch.__version__, "cuda": torch.version.cuda,
              "visible_gpu_count": torch.cuda.device_count(), "device": torch.cuda.get_device_name(),
              "state_shape": list(states.shape), "image_shape": list(images.shape),
              "action_chunk_shape": list(inference.shape), "loss": loss.item(),
              "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    (EXPERIMENT / "environment" / "model_smoke.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
