"""Re-execute three physical target paths per geometry; audit all nine actual color runs."""
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import numpy as np
from .spatial_gate import ROOT, RUN
from .spatial_collect import require_gate

WORKERS = 8


def trace(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def verify_layout(env, layout_row):
    from .event_control import FingerContactSensor, event_config
    from .event_efficiency import PhysicsRecorder, eligibility
    from .primitives import MotionDiagnostics, policy_state
    from .step_recording import save_rows
    from .vision import observation_images
    out = RUN / "replays" / f'{layout_row["split"]}_{layout_row["level"]}_{layout_row["seed"]}'
    out.mkdir(parents=True, exist_ok=True)
    if (out / "verification.json").exists():
        return json.loads((out / "verification.json").read_text())
    entries = [json.loads((RUN / directory / "metadata.json").read_text()) for directory in layout_row["episodes"]]
    references = {row["target_physical_slot"]: row for row in entries if row["layout"]["color_assignment_index"] == 0}
    assert len(references) == 3
    maximum_color_control_difference = 0.
    maximum_state_error = maximum_target_error = 0.
    replay_count = 0
    for slot, reference in references.items():
        source = RUN / reference["directory"]
        with np.load(source / "trajectory.npz", allow_pickle=False) as data:
            actions = data["osc_actions"].copy(); states = data["states"].copy(); macros = data["actions"].copy()
            initial_rgb = data["images"][0].copy()
        for row in entries:
            if row["target_physical_slot"] != slot:
                continue
            with np.load(RUN / row["directory"] / "trajectory.npz", allow_pickle=False) as data:
                assert data["osc_actions"].shape == actions.shape
                difference = float(np.max(np.abs(data["osc_actions"]-actions)))
                maximum_color_control_difference = max(maximum_color_control_difference, difference)
                assert difference < 1e-10, "A color-blind teacher should execute the same slot path after recoloring"
        obs = env.reset_layout(reference["layout"], reference["target_color"])
        assert np.max(np.abs(observation_images(obs).astype(int)-initial_rgb.astype(int))) <= 1
        sensor = FingerContactSensor(env, event_config("efficiency"))
        diagnostics = MotionDiagnostics(env.config)
        close_steps = 0
        with PhysicsRecorder(env, sensor) as recorder:
            for i, action in enumerate(actions):
                phase = int(macros[i, 3:].argmax())
                maximum_state_error = max(maximum_state_error, float(np.max(np.abs(policy_state(obs, phase)[:12]-states[i, :12]))))
                close_steps += phase >= 2
                control = {"phase": phase, "requested_phase": phase, "close_steps": close_steps,
                           "goal_xyz": macros[i, :3], "command_translation_m": action[:3]*.05}
                before = obs
                obs, _, done, info = env.step(action)
                diagnostics.update(env, before, obs, control, info)
                if done: break
        rows = recorder.enrich(diagnostics.rows, seed=reference["seed"])
        assert len(rows) == len(actions)
        assert not eligibility(info, rows, env.color_body_ids[reference["target_color"]])
        original = trace(source / "steps.csv")
        target_error = max(abs(float(a["target_after_"+axis])-float(b["target_after_"+axis]))
                           for a, b in zip(original, rows) for axis in "xyz")
        maximum_target_error = max(maximum_target_error, target_error)
        assert maximum_state_error <= 2e-5 and maximum_target_error <= 2e-5
        save_rows(out / f"slot{slot}_replay.csv", rows)
        replay_count += 1
    report = {"passed": True, "geometry_id": layout_row["geometry_id"], "actual_source_rollouts": 9, "additional_actual_replays": replay_count,
              "color_control_invariance_max_error": maximum_color_control_difference,
              "maximum_state_error": maximum_state_error, "maximum_target_position_error_m": maximum_target_error}
    (out / "verification.json").write_text(json.dumps(report, indent=2))
    return report


def worker(gpu, shard):
    require_gate()
    from .spatial_env import make_spatial_env
    manifest = json.loads((RUN / "data/manifest.json").read_text())
    env = make_spatial_env(gpu)
    try:
        for i, row in enumerate(manifest["layouts"]):
            if i%WORKERS != shard: continue
            result = verify_layout(env, row)
            print("VERIFY_LAYOUT", row["seed"], row["level"], result["passed"], result["maximum_state_error"], flush=True)
    finally:
        env.close()


def launch(detach):
    require_gate()
    assert (RUN / "collection_completed.json").exists()
    if detach:
        with (RUN / "replay.log").open("a") as log:
            child = subprocess.Popen([sys.executable, "-u", "-m", "colorlift.spatial_replay", "--launch"], cwd=ROOT,
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({"pid": child.pid, "GPUs": [1, 2]})); return
    workers, handles = [], []
    for shard in range(WORKERS):
        gpu = (1, 2)[shard%2]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), MUJOCO_EGL_DEVICE_ID=str(gpu), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        handle = (RUN / f"replay_{shard}.log").open("a"); handles.append(handle)
        workers.append(subprocess.Popen([sys.executable, "-u", "-m", "colorlift.spatial_replay", "--gpu", str(gpu), "--shard", str(shard)],
                                        cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT))
    codes = [worker.wait() for worker in workers]
    for handle in handles: handle.close()
    assert codes == [0]*WORKERS, codes
    reports = [json.loads(p.read_text()) for p in (RUN / "replays").glob("*/verification.json")]
    manifest = json.loads((RUN / "data/manifest.json").read_text())
    assert len(reports) == manifest["selected_unique_geometries"] and all(r["passed"] for r in reports)
    result = {"passed": True, "verified_geometries": len(reports), "all_actual_source_demonstrations": sum(r["actual_source_rollouts"] for r in reports),
              "additional_actual_replays": sum(r["additional_actual_replays"] for r in reports),
              "maximum_robot_state_error": max(r["maximum_state_error"] for r in reports),
              "maximum_target_position_error_m": max(r["maximum_target_position_error_m"] for r in reports)}
    (RUN / "data/replay_verified.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--launch", action="store_true"); p.add_argument("--detach", action="store_true")
    p.add_argument("--gpu", type=int); p.add_argument("--shard", type=int); args = p.parse_args()
    if args.launch: launch(args.detach)
    else: worker(args.gpu, args.shard)


if __name__ == "__main__": main()
