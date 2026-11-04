"""Offline validation of initial target selection; not a grasp-success test."""
import hashlib
import json
import numpy as np
import torch
from .conditional import RUN, COLORS, ConditionalACT


def main():
    torch.set_num_threads(4)
    checkpoint=torch.load(RUN / "model/latest_policy.pt", map_location="cpu", weights_only=False)
    model=ConditionalACT(pretrained=False).cuda().eval()
    model.load_state_dict(checkpoint["model"])
    episodes=json.loads((RUN / "data/manifest.json").read_text())["episodes"]
    selected=[row for row in episodes if row["split"] == "val" and row["eligible"]]
    stats=checkpoint["stats"]
    trials=[]
    for row in selected:
        with np.load(RUN / row["file"], allow_pickle=False) as data:
            state=(data["states"][0, :16]-stats["state_mean"])/stats["state_std"]
            image=data["images"][0]
            label=data["actions"][0, :2]
        with torch.inference_mode():
            prediction, _, uv=model(torch.as_tensor(state[None], device="cuda"),
                                    torch.as_tensor(image.transpose(0, 3, 1, 2)[None].copy(), device="cuda").float()/255,
                                    torch.tensor([COLORS.index(row["target_color"])], device="cuda"))
        xyz=prediction[0][0, 0, :3].cpu().numpy()*stats["xyz_std"]+stats["xyz_mean"]
        truth=np.array([row["initial_positions"][color][:2] for color in COLORS])
        nearest=COLORS[np.argmin(np.linalg.norm(truth-xyz[:2], axis=1))]
        trials.append({"seed": row["seed"], "color": row["target_color"], "nearest_color": nearest,
                       "correct": nearest == row["target_color"], "xy_error_to_expert_m": float(np.linalg.norm(xyz[:2]-label)),
                       "xy_error_to_true_cube_m": float(np.linalg.norm(xyz[:2]-truth[COLORS.index(row["target_color"])])),
                       "initial_RGB_hash": hashlib.sha256(image.tobytes()).hexdigest()})
    result={"checkpoint_step": checkpoint["step"], "initial_target_selection_correct": sum(row["correct"] for row in trials), "episodes": len(trials),
            "mean_initial_xy_error_to_expert_m": float(np.mean([row["xy_error_to_expert_m"] for row in trials])),
            "mean_initial_xy_error_to_true_cube_m": float(np.mean([row["xy_error_to_true_cube_m"] for row in trials])),
            "this_is_not_closed_loop_grasp_success": True, "trials": trials}
    out=RUN / "verification" / f'initial_probe_step_{checkpoint["step"]}.json'
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({key: value for key, value in result.items() if key != "trials"}))


if __name__ == "__main__": main()
