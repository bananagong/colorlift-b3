"""Run the trained short-chunk ACT with an optional pair of physical 3cm offsets."""
import argparse
from dataclasses import replace
import json
import torch
from .config import COLORS
from .conditional import ConditionalACT, configuration
from .env import make_env
from .recovery_evaluate import RUN, episode


def main():
    p=argparse.ArgumentParser(); p.add_argument("--gpu", type=int, default=1); p.add_argument("--color", choices=COLORS, required=True)
    p.add_argument("--seed", type=int, default=83000); p.add_argument("--disturbed", action="store_true"); p.add_argument("--tag", default="demo"); args=p.parse_args()
    assert args.tag.replace("_", "").replace("-", "").isalnum()
    directory=RUN / "interactive" / args.tag / f"scene_{args.seed}" / args.color
    if (directory / "metadata.json").exists(): raise RuntimeError("Choose a new tag to preserve earlier runs")
    torch.set_num_threads(4)
    checkpoint=torch.load(RUN / "model/policy.pt", map_location="cpu", weights_only=False)
    model=ConditionalACT(4, pretrained=False).cuda().eval(); model.load_state_dict(checkpoint["model"])
    protocol=json.loads((RUN / "protocol.json").read_text())
    cfg=configuration(args.color)[0]
    if args.disturbed: cfg=replace(cfg, horizon=240)
    env=make_env(args.gpu, cfg)
    try: result=episode(env, model, checkpoint, args.seed, directory, "new_recovery", protocol["selected_query_schedule"], args.disturbed)
    finally: env.close()
    print(json.dumps({key: result[key] for key in ("target_color", "success", "outcome", "control_steps", "collision_steps", "directory")}, indent=2))


if __name__ == "__main__": main()
