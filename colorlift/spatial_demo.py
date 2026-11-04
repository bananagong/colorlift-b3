"""One new random spatial task, with actual physical recoloring and a learned color command."""
import argparse
import json
import torch
from .config import COLORS
from .spatial_gate import RUN, CHECKPOINT
from .spatial_evaluate import MODELS, LEVELS
from .spatial_layouts import counterfactual_descriptor
from .spatial_env import make_spatial_env
from .spatial_rollout import load_policy, episode


def main():
    p = argparse.ArgumentParser(); p.add_argument("--gpu", type=int, default=1)
    p.add_argument("--model", choices=("recommended",)+MODELS, default="recommended")
    p.add_argument("--level", choices=LEVELS, default="easy"); p.add_argument("--color", choices=COLORS, default="red")
    p.add_argument("--seed", type=int, default=2400000); p.add_argument("--permutation", type=int, choices=range(6), default=0)
    p.add_argument("--tag", required=True); args = p.parse_args()
    assert args.tag and all(c.isalnum() or c in "_-" for c in args.tag)
    name = json.loads((RUN / "recommended_model.json").read_text())["model"] if args.model == "recommended" else args.model
    path = CHECKPOINT if name == "baseline" else RUN / "models" / name / "policy.pt"
    directory = RUN / "demos" / args.tag / f"{name}_{args.level}_{args.seed}_p{args.permutation}_{args.color}"
    assert not directory.exists(), "Choose a new tag to preserve existing videos"
    layout = counterfactual_descriptor(args.seed, args.level, args.permutation)
    torch.set_num_threads(1)
    model, checkpoint = load_policy(path)
    env = make_spatial_env(args.gpu)
    try:
        result = episode(env, layout, args.color, directory, name, model, checkpoint)
        print(json.dumps({key: result[key] for key in ("success", "eligible", "control_steps", "target_physical_slot", "first_picked_slot", "directory")}, indent=2))
    finally:
        env.close()


if __name__ == "__main__": main()
