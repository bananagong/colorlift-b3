"""Paired execution-prefix ablation with an immutable ACT checkpoint and unchanged OSC adapter."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from .conditional import RUN as ORIGINAL, ConditionalACT, configuration
from .config import ROOT, COLORS
from .env import make_env
from .conditional_evaluate import run_episode

RUN=ROOT / "experiments/colorlift_recovery_v1"
MODES={"h1": (1, None), "h2": (2, None), "h6": (6, None), "h12": (12, None), "adaptive": (4, (4, 1, 1, 3))}


def main():
    p=argparse.ArgumentParser(); p.add_argument("--gpu", type=int, required=True); p.add_argument("--shard", type=int, required=True); args=p.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    out=RUN / "execution_sweep"; out.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    checkpoint_path=ORIGINAL / "model/policy.pt"
    digest=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    checkpoint=torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model=ConditionalACT(12, pretrained=False).cuda().eval(); model.load_state_dict(checkpoint["model"])
    # Workers alternate scenes; each mode sees all 50 scenes and all three colors.
    envs={color: make_env(args.gpu, configuration(color)[0]) for color in COLORS}
    started=time.time(); results=[]
    try:
        for name, (horizon, schedule) in MODES.items():
            for seed in range(75000+args.shard, 75050, 2):
                for color in COLORS:
                    directory=out / name / f"scene_{seed}" / color
                    if (directory / "metadata.json").exists():
                        result=json.loads((directory / "metadata.json").read_text())
                    else:
                        result=run_episode(envs[color], model, checkpoint, seed, directory, horizon,
                                           query_schedule=schedule, run_root=RUN)
                    results.append(result)
                    print("SWEEP", name, seed, color, result["success"], result["control_steps"], result["collision_steps"], flush=True)
    finally:
        for env in envs.values(): env.close()
    assert hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() == digest
    (out / f"shard_{args.shard}_completed.json").write_text(json.dumps({"completed": True, "model_sha256": digest, "episodes": len(results), "seconds": time.time()-started}))


if __name__ == "__main__": main()
