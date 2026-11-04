"""Verify held-out rendering changes are real MuJoCo changes, without moving geometry."""
import json
import numpy as np
from PIL import Image
from .spatial_gate import RUN
from .spatial_layouts import descriptor
from .spatial_env import make_spatial_env
from .vision import observation_images


def main():
    env = make_spatial_env(1)
    out = RUN / "preflight"; out.mkdir(exist_ok=True)
    layout = descriptor(1999200, "easy")
    rows = []; images = []
    try:
        for level in ("easy", "ood_light", "ood_camera"):
            current = dict(layout, level=level)
            obs = env.reset_layout(current, "red")
            rgb = observation_images(obs)
            images.append(np.concatenate(list(rgb), axis=1))
            meta = env.episode_metadata()
            rows.append({"level": level, "nlight": env.sim.model.nlight, "mean_RGB": rgb.mean((0, 1, 2)).tolist(),
                         "physical_slot_positions": meta["physical_slot_positions"], "lights": meta["actual_light_diffuse"],
                         "cameras": meta["actual_camera_positions"]})
        assert rows[0]["nlight"] > 0
        assert rows[0]["lights"] != rows[1]["lights"]
        assert rows[0]["cameras"] != rows[2]["cameras"]
        assert all(r["physical_slot_positions"] == rows[0]["physical_slot_positions"] for r in rows)
        assert np.mean(np.abs(images[0].astype(float)-images[1])) > 1
        assert np.mean(np.abs(images[0].astype(float)-images[2])) > 1
        Image.fromarray(np.concatenate(images, axis=0)).save(out / "actual_OOD_rendering.png")
        (out / "actual_OOD_rendering.json").write_text(json.dumps({"passed": True, "rows": rows}, indent=2))
        print(json.dumps({"passed": True, "nlight": rows[0]["nlight"], "meanRGB_by_condition": [r["mean_RGB"] for r in rows]}, indent=2))
    finally:
        env.close()


if __name__ == "__main__": main()
