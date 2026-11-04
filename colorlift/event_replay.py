"""Execute every candidate's unchanged OSC actions again before admitting expert data."""
import argparse
import hashlib
import json
import shutil
import time
import numpy as np
from .conditional import configuration
from .config import COLORS
from .env import make_env
from .event_control import FingerContactSensor, event_config
from .event_efficiency import RUN, PhysicsRecorder, finalize
from .primitives import MotionDiagnostics, policy_state


def replay(env, candidate):
    source = candidate / "trajectory.npz"
    meta = json.loads((candidate / "metadata.json").read_text())
    directory = RUN / "replays" / candidate.name
    if (directory / "verification.json").exists():
        cached = json.loads((directory / "verification.json").read_text())
        assert cached["source_npz_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
        return cached
    result = {"candidate": str(candidate.relative_to(RUN)), "seed": meta["seed"], "target_color": meta["target_color"],
              "source_npz_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "admitted": False,
              "split": candidate.name.split("_")[0]}
    if not meta["contact_qualified_collision_free_success"]:
        result["rejection_reasons"] = ["candidate:"+reason for reason in meta["dataset_rejection_reasons"]]
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "verification.json").write_text(json.dumps(result, indent=2))
        return result
    with np.load(source, allow_pickle=False) as arrays:
        actions = arrays["osc_actions"].copy()
        macro = arrays["actions"].copy()
        states = arrays["states"].copy()
    obs = env.reset_seed(meta["seed"])
    sensor = FingerContactSensor(env, event_config(meta["scheme"]))
    diagnostics = MotionDiagnostics(env.config)
    close_steps = 0
    maximum_state_error = 0.
    info = {}
    with PhysicsRecorder(env, sensor) as recorder:
        for index, (action, goal) in enumerate(zip(actions, macro)):
            phase = int(goal[3:].argmax())
            before_state = policy_state(obs, phase)
            maximum_state_error = max(maximum_state_error, float(np.max(np.abs(before_state[:12]-states[index, :12]))))
            close_steps += phase >= 2
            control = {"phase": phase, "requested_phase": phase, "close_steps": close_steps,
                       "goal_xyz": goal[:3], "command_translation_m": action[:3]*.05}
            before = obs
            obs, _, done, info = env.step(action)
            diagnostics.update(env, before, obs, control, info)
            if done:
                break
    rows = recorder.enrich(diagnostics.rows, seed=meta["seed"])
    summary = {**env.episode_metadata(), **info, **diagnostics.summary(), "seed": meta["seed"], "actor": "expert_replay",
               "scheme": meta["scheme"], "policy_queries": 0, "inference_seconds": 0.,
               "outcome": "success" if info["success"] else "replay_failure"}
    summary = finalize(env, directory, summary, rows, recorder)
    identical_actions = (len(rows) == len(actions)
                         and all(np.array_equal(record["action"], original.astype(float)) for record, original in zip(recorder.records, actions)))
    initial_rgb_matches = meta["initial_rgb_sha256"] == summary["initial_rgb_sha256"]
    result.update(replay_success=summary["success"], replay_contact_qualified=summary["contact_qualified_collision_free_success"],
                  identical_actions=identical_actions, maximum_state_error=maximum_state_error,
                  initial_rgb_matches=initial_rgb_matches, control_steps=summary["control_steps"],
                  original_control_steps=meta["control_steps"], action_sha256=hashlib.sha256(actions.tobytes()).hexdigest())
    reasons = list(summary["dataset_rejection_reasons"])
    if not identical_actions:
        reasons.append("replay_actions_or_length_changed")
    if maximum_state_error > 2e-5:
        reasons.append("state_replay_diverged")
    if not initial_rgb_matches:
        reasons.append("initial_rgb_changed")
    result["rejection_reasons"] = reasons
    result["admitted"] = not reasons
    if result["admitted"]:
        destination = RUN / "data" / candidate.name
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination / "trajectory.npz")
        shutil.copy2(candidate / "metadata.json", destination / "metadata.json")
        result["data_file"] = str((destination / "trajectory.npz").relative_to(RUN))
        (destination / "verification.json").write_text(json.dumps(result, indent=2))
    (directory / "verification.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--shard", type=int, required=True)
    args = p.parse_args()
    envs = {color: make_env(args.gpu, configuration(color)[0]) for color in COLORS}
    results = []
    seen = set()
    try:
        while True:
            for candidate in sorted((RUN / "candidates").glob("*_*_*")):
                seed = int(candidate.name.split("_")[1])
                if seed % 2 != args.shard or candidate.name in seen or not (candidate / "metadata.json").exists():
                    continue
                try:
                    meta = json.loads((candidate / "metadata.json").read_text())
                except json.JSONDecodeError:
                    continue  # A candidate is published only after its complete metadata is readable.
                result = replay(envs[meta["target_color"]], candidate)
                results.append(result)
                seen.add(candidate.name)
                print("REPLAY", candidate.name, result["admitted"], result["rejection_reasons"], result.get("maximum_state_error"), flush=True)
            if (RUN / "collect_completed.json").exists():
                owned = [p for p in (RUN / "candidates").glob("*_*_*") if int(p.name.split("_")[1]) % 2 == args.shard]
                if len(seen) == len(owned):
                    break
            time.sleep(2)
    finally:
        for env in envs.values():
            env.close()
    (RUN / "replays" / f"manifest_shard_{args.shard}.json").write_text(json.dumps({"episodes": results}, indent=2))


if __name__ == "__main__":
    main()
