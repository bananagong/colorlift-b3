"""Record corrective expert labels, separately masking every injected disturbance action."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
import cv2
from .conditional import configuration
from .config import ROOT, COLORS
from .env import make_env
from .recovery_expert import NoiseConfig, rollout
from .step_recording import StepRecorder, save_rows

RUN=ROOT / "experiments/colorlift_recovery_v1"


def save_case(env, seed, directory, noise):
    directory.mkdir(parents=True, exist_ok=True)
    with StepRecorder(env, capture_images=True) as recorder:
        result, history, rows=rollout(env, seed, noise, record=True)
    enriched=recorder.enrich(rows, seed=seed)
    save_rows(directory / "steps.csv", enriched)
    np.savez_compressed(directory / "trajectory.npz", **history, metadata=json.dumps(result))
    (directory / "metadata.json").write_text(json.dumps(result, indent=2))
    with imageio.get_writer(str(directory / "rollout.mp4"), fps=env.config.hz, codec="libx264", quality=7) as writer:
        writer.append_data(np.concatenate(list(recorder.initial_images), axis=1))
        for record, row in zip(recorder.records, enriched):
            frame=np.concatenate(list(record["images"]), axis=1)
            cv2.putText(frame, f'{row["step"]} {row["training_category"]} hold={row["hold_streak"]}/40', (2, 11), cv2.FONT_HERSHEY_SIMPLEX, .3, (0, 0, 0), 1)
            writer.append_data(frame)
    assert len(enriched) == result["control_steps"]
    assert not any(row["supervised"] for row in enriched if row["training_category"] == "injected_offset")
    assert np.all(history["supervised"][history["recovery"]])
    if result["success"]:
        assert [int(row["hold_streak"]) for row in enriched[-40:]] == list(range(1, 41))
        assert min(row["target_lift_m"] for row in enriched[-40:]) > .04
    return result


def main():
    p=argparse.ArgumentParser(); p.add_argument("--gpu", type=int, required=True); p.add_argument("--shard", type=int, default=0)
    p.add_argument("--pilot", action="store_true"); p.add_argument("--train-scenes", type=int, default=80); p.add_argument("--val-scenes", type=int, default=15)
    args=p.parse_args()
    noise=NoiseConfig()
    envs={color: make_env(args.gpu, replace(configuration(color)[0], horizon=noise.horizon)) for color in COLORS}
    episodes=[]
    try:
        splits=[("pilot", 2, 77000)] if args.pilot else [("train", args.train_scenes, 78000), ("val", args.val_scenes, 79000)]
        for split, count, base in splits:
            for index in (range(count) if args.pilot else range(args.shard, count, 2)):
                for color in COLORS:
                    seed=base+index
                    directory=RUN / ("pilot" if args.pilot else "data") / f"{split}_{seed}_{color}"
                    if (directory / "metadata.json").exists(): result=json.loads((directory / "metadata.json").read_text())
                    else: result=save_case(envs[color], seed, directory, noise)
                    result.update(split=split, directory=str(directory.relative_to(RUN)))
                    episodes.append(result)
                    print("NOISY_EXPERT", split, seed, color, result["success"], result["control_steps"], len(result["noise_events"]), result["recovery_steps"], result["maximum_ik_residual_m"], flush=True)
                    target=RUN / ("pilot/manifest.json" if args.pilot else f"data/manifest_shard_{args.shard}.json")
                    target.write_text(json.dumps({"episodes": episodes}, indent=2))
    finally:
        for env in envs.values(): env.close()
    if not args.pilot:
        (RUN / f"data/shard_{args.shard}_completed.json").write_text(json.dumps({"complete": True, "episodes": len(episodes)}))


if __name__ == "__main__": main()
