"""Autonomous one-attempt grasp: choose one target color for the shared ACT."""
import argparse
import json
import torch
from .conditional import RUN, COLORS, ConditionalACT, configuration
from .conditional_evaluate import run_episode
from .env import make_env


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--color", choices=COLORS, required=True)
    p.add_argument("--seed", type=int, default=76000)
    p.add_argument("--gpu", type=int, default=4)
    p.add_argument("--tag", default="demo")
    args=p.parse_args()
    assert args.tag.replace("_", "").replace("-", "").isalnum()
    out=RUN / "interactive" / args.tag / f"scene_{args.seed}" / args.color
    if (out / "metadata.json").exists(): raise RuntimeError("Choose another --tag to preserve the previous result")
    torch.set_num_threads(4)
    checkpoint=torch.load(RUN / "model/policy.pt", map_location="cpu", weights_only=False)
    model=ConditionalACT(12, pretrained=False).cuda().eval()
    model.load_state_dict(checkpoint["model"])
    env=make_env(args.gpu, configuration(args.color)[0])
    try:
        result=run_episode(env, model, checkpoint, args.seed, out)
    finally: env.close()
    print(json.dumps({key: result[key] for key in ("target_color", "first_picked_color", "success", "control_steps", "outcome", "directory")}, indent=2))


if __name__ == "__main__": main()
