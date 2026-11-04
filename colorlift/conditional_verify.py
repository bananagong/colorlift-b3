"""Check gradient flow, curriculum dimensions and target-independent OSC execution."""
import argparse
import json
import numpy as np
import torch
from .conditional import RUN, ConditionalACT, FastExecutor, configuration, recognition_loss
from .config import COLORS
from .env import make_env
from .vision import observation_images, observation_masks, locate_on_table
from .primitives import policy_state
from .model import loss_components


def main():
    p = argparse.ArgumentParser(); p.add_argument("--gpu", type=int, default=4); args=p.parse_args()
    torch.set_num_threads(4)
    model = ConditionalACT(pretrained=False).cuda()
    output, logits, uv = model(torch.randn(3, 16, device="cuda"), torch.rand(3, 2, 3, 128, 128, device="cuda"),
                               torch.arange(3, device="cuda"), torch.randn(3, 12, 7, device="cuda"),
                               torch.arange(12, device="cuda")[None].expand(3, -1) >= 4)
    assert output[0].shape == (3, 12, 7) and logits.shape == (3, 2, 3, 128, 128) and uv.shape == (3, 4)
    output[0].square().mean().backward()
    norms = model.color_embedding.weight.grad.norm(dim=1).tolist()
    assert min(norms) > 0
    del model
    results = []
    for color in COLORS:
        cfg, expert = configuration(color)
        env = make_env(args.gpu, cfg)
        try:
            for seed in range(73000, 73005):
                obs = env.reset_seed(seed)
                location, _ = locate_on_table(env, obs)
                executor = FastExecutor(cfg, expert)
                info = {}
                for step in range(cfg.horizon):
                    action, control = executor.action(obs, executor.phase+1, location)
                    obs, _, done, info = env.step(action)
                    executor.after_step(obs, control)
                    if done: break
                results.append({"seed": seed, "target_color": color, "success": info["success"], "steps": info["control_steps"], "collision_steps": info["collision_steps"]})
                print("EXECUTOR ORACLE FIXTURE", results[-1], flush=True)
        finally: env.close()
    assert all(r["success"] and not r["collision_steps"] for r in results)
    out = RUN / "verification"; out.mkdir(parents=True, exist_ok=True)
    (out / "preflight.json").write_text(json.dumps({"model_shape_check": True, "color_embedding_gradient_norms": norms,
                                                    "oracle_fixture_only_not_ACT_results": results}, indent=2))


if __name__ == "__main__": main()
