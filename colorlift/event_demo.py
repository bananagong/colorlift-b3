"""Run one frozen ACT or HSV expert with selectable event/chunk execution."""
import argparse
import hashlib
import json
import torch
from .config import COLORS
from .conditional import configuration
from .env import make_env
from .event_control import SCHEMES
from .event_efficiency import RUN, CHECKPOINT, load_model, run_episode


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=1)
    p.add_argument("--color", choices=COLORS, default="red")
    p.add_argument("--seed", type=int, default=88000)
    p.add_argument("--actor", choices=("act", "expert"), default="act")
    p.add_argument("--scheme", choices=tuple(SCHEMES), default="efficiency")
    p.add_argument("--tag", required=True)
    args = p.parse_args()
    assert args.tag and all(character.isalnum() or character in "_-" for character in args.tag)
    torch.set_num_threads(4)
    directory = RUN / "demos" / args.tag / f"{args.actor}_{args.scheme}_{args.seed}_{args.color}"
    assert not directory.exists(), "Choose a new --tag to preserve existing results"
    before = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    model, checkpoint = load_model() if args.actor == "act" else (None, None)
    env = make_env(args.gpu, configuration(args.color)[0])
    try:
        result = run_episode(env, args.seed, directory, args.actor, args.scheme, model, checkpoint)
        print(json.dumps({key: result[key] for key in ("success", "control_steps", "contact_qualified_collision_free_success", "physical_eef_path_m", "directory")}, indent=2))
    finally:
        env.close()
    assert hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() == before


if __name__ == "__main__":
    main()
