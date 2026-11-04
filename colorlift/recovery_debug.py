"""Inspect completed development failures and raw learned waypoints."""
import csv
import json
import numpy as np
import torch
from PIL import Image
from .config import ROOT, COLORS
from .conditional import ConditionalACT


def main():
    root=ROOT / "experiments/colorlift_recovery_v1"
    ckpt=torch.load(root / "model/policy.pt", map_location="cpu", weights_only=False)
    model=ConditionalACT(4, pretrained=False).cuda().eval(); model.load_state_dict(ckpt["model"])
    metrics=json.loads((root / "evaluations/dev/metrics.json").read_text())
    print("SUMMARY", json.dumps(metrics["summary"]))
    for case in [row for row in metrics["trials"] if not row["success"]][:8]:
        directory=root / case["directory"]
        with (directory / "steps.csv").open(encoding="utf-8-sig") as f: data=list(csv.DictReader(f))
        row=data[-1]; phase=int(row["phase"])
        state=np.r_[[float(row[f"robot_joint_{j}"]) for j in range(7)],
                    [float(row[f"finger_qpos_{j}_after"]) for j in range(2)],
                    [float(row["eef_after_"+a]) for a in "xyz"], np.eye(4)[phase]].astype(np.float32)
        state=(state-ckpt["stats"]["state_mean"])/ckpt["stats"]["state_std"]
        image=np.stack(np.split(np.array(Image.open(directory / "final.png")), 2, axis=1))
        with torch.inference_mode(): output, _, _=model(torch.tensor(state[None], device="cuda"), torch.tensor(image.transpose(0, 3, 1, 2)[None].copy(), device="cuda").float()/255,
                                                       torch.tensor([COLORS.index(case["target_color"])], device="cuda"))
        xyz=output[0][0, 0, :3].cpu().numpy()*ckpt["stats"]["xyz_std"]+ckpt["stats"]["xyz_mean"]
        print(json.dumps({"seed": case["seed"], "color": case["target_color"], "phase": phase,
                          "last_eef": [float(row["eef_after_"+a]) for a in "xyz"], "last_target": [float(row["target_after_"+a]) for a in "xyz"],
                          "last_command_goal": [float(row["goal_"+a]) for a in "xyz"], "raw_ACT_goal": xyz.tolist()}))


if __name__ == "__main__": main()
