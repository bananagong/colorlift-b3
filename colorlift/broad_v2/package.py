"""Export development/data evidence while keeping raw final layouts sealed on catlab."""
import json
import importlib.metadata
import shutil
import tarfile
from .spec import ROOT, RUN, CATEGORIES, sha


def main():
    assert json.loads((RUN / "phase1_completed.json").read_text())["complete"]
    manifest = json.loads((RUN / "training_manifest.json").read_text())
    examples = set()
    for split in ("train", "val"):
        for category in CATEGORIES:
            row = next(r for r in manifest["layouts"] if r["split"] == split and r["layout"]["category"] == category)
            examples.update(row["episodes"])
    assert len(examples) == 72
    source = RUN / "source/colorlift/broad_v2"; source.mkdir(parents=True, exist_ok=True)
    for path in (ROOT / "colorlift/broad_v2").glob("*.py"): shutil.copy2(path, source / path.name)
    shared = RUN / "source/shared_colorlift"; shared.mkdir(exist_ok=True)
    for name in ("config.py", "env.py", "spatial_env.py", "spatial_layouts.py", "event_control.py", "step_recording.py", "event_efficiency.py", "primitives.py", "conditional.py", "recovery_expert.py"):
        shutil.copy2(ROOT / "colorlift" / name, shared / name)
    adverse = set()
    for path in (RUN / "data_candidates").rglob("metadata.json"):
        row = json.loads(path.read_text())
        if not row["eligible"]: adverse.add(str(path.parent.relative_to(RUN)))
    contents = {"all_selected_demonstrations_on_server": 2160, "raw_NPZ_examples_exported": 72,
                "all_development_metadata_steps_and_videos": True, "all_training_metadata_and_steps": True,
                "all_training_adverse_media": True, "final_raw_coordinates_and_images_exported": False,
                "final_policy_evaluations": 0, "new_ACT_checkpoints": 0,
                "server_full_directory": str(RUN), "sealed_final_directory": str(RUN / "final_sealed")}
    (RUN / "runtime_versions.json").write_text(json.dumps({name: importlib.metadata.version(name) for name in ("torch", "numpy", "scipy", "mujoco", "robosuite", "opencv-python", "imageio")}, indent=2))
    (RUN / "delivery_contents.json").write_text(json.dumps(contents, indent=2))
    archive = ROOT / "experiments/colorlift_broad_protocol_v2_verified_results.tar.gz"
    files = 0
    with tarfile.open(archive, "w:gz", compresslevel=4) as tar:
        for path in sorted(RUN.rglob("*")):
            if not path.is_file(): continue
            relative = path.relative_to(RUN)
            if relative.parts[0] in ("final_sealed", "registry") or "__pycache__" in relative.parts: continue
            if str(relative) in ("package.json", "finish.log"): continue
            parent = str(path.parent.relative_to(RUN))
            if relative.parts[0] == "data_candidates":
                if path.name == "trajectory.npz" and parent not in examples: continue
                if path.suffix.lower() in (".mp4", ".png") and parent not in examples and parent not in adverse:
                    # Keep parent scene thumbnails used to verify new layouts.
                    if path.name != "initial_two_views.png": continue
            tar.add(path, arcname=str(path.relative_to(ROOT)), recursive=False); files += 1
    result = {"archive": str(archive), "bytes": archive.stat().st_size, "files": files, "sha256": sha(archive), **contents}
    (RUN / "package.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__": main()
