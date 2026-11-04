"""Compare visual target prediction before/after real perturbations on held-out expert data."""
import csv
import json
import numpy as np
import torch
from .conditional import RUN as ORIGINAL, ConditionalACT
from .config import ROOT, COLORS

RUN=ROOT / "experiments/colorlift_recovery_v1"


def main():
    torch.set_num_threads(4)
    cases=[row for row in json.loads((RUN / "data/manifest.json").read_text())["episodes"] if row["split"]=="val" and row["eligible"]]
    models={}; checkpoints={}
    for key, path in (("old", ORIGINAL / "model/policy.pt"), ("new", RUN / "model/policy.pt")):
        checkpoint=torch.load(path, map_location="cpu", weights_only=False)
        model=ConditionalACT(checkpoint["config"]["max_chunk"], pretrained=False).cuda().eval(); model.load_state_dict(checkpoint["model"])
        models[key]=model; checkpoints[key]=checkpoint
    measurements=[]
    for trial in cases:
        directory=RUN / trial["directory"]
        with (directory / "steps.csv").open(encoding="utf-8-sig") as stream: trace=list(csv.DictReader(stream))
        with np.load(directory / "trajectory.npz", allow_pickle=False) as data:
            for event in trial["noise_events"]:
                for when, index in (("before", event["start_step"]-1), ("after", event["end_step"])):
                    truth=np.array([float(trace[index]["target_before_"+a]) for a in "xy"])
                    for key in ("old", "new"):
                        stats=checkpoints[key]["stats"]
                        state=(data["states"][index, :16]-stats["state_mean"])/stats["state_std"]
                        rgb=data["images"][index]
                        with torch.inference_mode():
                            output, _, _=models[key](torch.tensor(state[None], device="cuda"),
                                                    torch.tensor(rgb.transpose(0, 3, 1, 2)[None].copy(), device="cuda").float()/255,
                                                    torch.tensor([COLORS.index(trial["target_color"])], device="cuda"))
                        xyz=output[0][0, 0, :3].cpu().numpy()*stats["xyz_std"]+stats["xyz_mean"]
                        measurements.append({"seed": trial["seed"], "color": trial["target_color"], "event": event["event_id"], "when": when, "model": key,
                                             "target_xy_error_m": float(np.linalg.norm(xyz[:2]-truth))})
    summary={key: {when: float(np.mean([row["target_xy_error_m"] for row in measurements if row["model"]==key and row["when"]==when])) for when in ("before", "after")} for key in ("old", "new")}
    (RUN / "validation_target_probe.json").write_text(json.dumps({"summary": summary, "events_per_model": 90, "held_out_expert_validation_only": True, "trials": measurements}, indent=2))
    print("TARGET_RECOVERY_PROBE", json.dumps(summary))


if __name__ == "__main__": main()
