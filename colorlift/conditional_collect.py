"""Balanced scene-level RGB demonstrations for all three target colors."""
import argparse
from dataclasses import asdict
import json
import time
import numpy as np
from .conditional import RUN, configuration
from .config import COLORS
from .env import make_env
from .efficient_expert import rollout
from .step_recording import save_rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--shards", type=int, default=2)
    p.add_argument("--train-scenes", type=int, default=180)
    p.add_argument("--val-scenes", type=int, default=30)
    args = p.parse_args()
    out = RUN / "data"
    out.mkdir(parents=True, exist_ok=True)
    envs = {c: make_env(args.gpu, configuration(c)[0]) for c in COLORS}
    expert = configuration()[1]
    entries = []
    try:
        for split, count, base in (("train", args.train_scenes, 71000), ("val", args.val_scenes, 72000)):
            for index in range(args.shard, count, args.shards):
                seed = base+index
                for color in COLORS:
                    path = out / f"{split}_{seed}_{color}.npz"
                    if path.exists():
                        with np.load(path, allow_pickle=False) as d:
                            result = json.loads(str(d["metadata"]))
                    else:
                        result, history, rows = rollout(envs[color], seed, expert, record=True)
                        result.update(split=split, file=str(path.relative_to(RUN)))
                        np.savez_compressed(path, **history, metadata=json.dumps(result))
                        save_rows(path.with_suffix(".csv"), rows)
                    entries.append(result)
                    print("COLLECT", split, seed, color, result["success"], result["control_steps"], flush=True)
                (out / f"manifest_shard_{args.shard}.json").write_text(json.dumps({"episodes": entries}, indent=2))
    finally:
        for env in envs.values():
            env.close()
    (out / f"shard_{args.shard}_completed.json").write_text(json.dumps({"complete": True, "episodes": len(entries)}))


if __name__ == "__main__":
    main()
