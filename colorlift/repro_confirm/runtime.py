"""Add read-only initial-frame capture; keep the frozen action/scoring loop."""
from datetime import datetime,timezone
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .spec import RUN, read
from ..broad_act import runtime as original
from ..broad_v2.spec import VALID
from ..vision import observation_images


def make_env(gpu):
    return original.make_env(gpu)


def episode(env,layout,color,directory,actor,model,checkpoint,representative=False):
    directory.mkdir(parents=True,exist_ok=True)
    marker=directory / "metadata.json"
    if marker.exists():
        try:
            value=read(marker)
            assert value["geometry_id"]==layout["geometry_id"] and value["target_color"]==color
            assert (directory / "initial_front.png").exists() and (directory / "initial_side.png").exists()
            return value
        except (json.JSONDecodeError,KeyError):
            pass
    # Preserve incomplete files as technical evidence. No failed task outcome is
    # replaced: a valid completed metadata record above always returns unchanged.
    previous=[p for p in directory.iterdir() if p.is_file()]
    if previous:
        archive=directory / ("technical_attempt_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f"));archive.mkdir()
        for path in previous:
            path.replace(archive / ("metadata.partial.json" if path.name=="metadata.json" else path.name))
    reset=env.reset_layout
    def capture_reset(candidate,target_color):
        obs=reset(candidate,target_color)
        checks=(("max_drift_m","settle_max_drift_m"),("max_height_error_m","settle_max_height_error_m"),
                ("max_tilt_rad","settle_max_tilt_rad"),("max_linear_speed_m_s","settle_max_linear_speed_m_s"),
                ("max_angular_speed_rad_s","settle_max_angular_speed_rad_s"))
        assert all(env.settlement[a]<=VALID[b] for a,b in checks) and not env.settlement["unintended_contact_steps"], "Initial physics invalid; technical stop, never replace this layout"
        images=observation_images(obs)
        Image.fromarray(images[0]).save(directory / "initial_front.png")
        Image.fromarray(images[1]).save(directory / "initial_side.png")
        return obs
    original.RUN=RUN;env.reset_layout=capture_reset
    try:
        return original.episode(env,layout,color,directory,actor,model,checkpoint,"original",representative)
    finally:
        env.reset_layout=reset
