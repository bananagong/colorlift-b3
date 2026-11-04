"""Paired, frozen-checkpoint efficiency experiments and physically replayed expert data."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
import cv2
import imageio.v2 as imageio
from PIL import Image
from .config import ROOT, COLORS
from .conditional import ConditionalACT, configuration
from .efficient_expert import rollout as old_expert_rollout
from .env import make_env
from .event_control import FingerContactSensor, EventExecutor, SCHEMES, event_config, protocol
from .recovery_control import RecoveryExecutor
from .primitives import MotionDiagnostics, policy_state
from .step_recording import StepRecorder, save_rows, first_steps_breakdown
from .vision import observation_images, observation_masks, locate_on_table

RUN = ROOT / "experiments/colorlift_event_efficiency_v1"
CHECKPOINT = ROOT / "experiments/colorlift_recovery_v1/model/policy.pt"


class PhysicsRecorder(StepRecorder):
    """Read-only 500Hz path integration and tactile evidence around every OSC action."""
    def __init__(self, env, sensor):
        super().__init__(env, capture_images=True)
        self.sensor = sensor
        self.path = self.joint_path = 0.
        self.physics_samples = 0
        self.last_time = -1.

    def __enter__(self):
        super().__enter__()
        self.original_update = self.env._update_observables
        self.env._update_observables = self.update
        return self

    def __exit__(self, *args):
        self.env._update_observables = self.original_update
        return super().__exit__(*args)

    def update(self, force=False):
        self.original_update(force=force)
        if (not self.env._capture_contacts or self.env.sim.data.time <= self.last_time+self.env.model_timestep*.5):
            return
        pos = self.env.sim.data.site_xpos[self.site_id].copy()
        joints = self.env.robots[0]._joint_positions.copy()
        self.path += float(np.linalg.norm(pos-self.last_pos))
        self.joint_path += float(np.linalg.norm(joints-self.last_joints))
        self.last_time = float(self.env.sim.data.time)
        self.last_pos, self.last_joints = pos, joints
        self.physics_samples += 1

    def step(self, action):
        obs = self.env._get_observations()
        tactile_before = self.sensor.read(obs)
        true_grasp_before = bool(self.env._check_grasp(self.env.robots[0].gripper, self.env.color_objects[self.env.target_color]))
        snap = self.snapshot()
        self.last_pos, self.last_joints, self.last_time = snap["eef"], snap["joints"], snap["time"]
        self.path = self.joint_path = 0.
        self.physics_samples = 0
        result = super().step(action)
        self.records[-1].update(tactile_before=tactile_before, tactile_after=self.sensor.read(result[0]),
                                true_grasp_before=true_grasp_before, physics_path_m=self.path,
                                physics_joint_path_rad=self.joint_path, physics_samples=self.physics_samples)
        return result

    def enrich(self, rows, **kwargs):
        rows = super().enrich(rows, **kwargs)
        for row, record in zip(rows, self.records):
            for key, value in record["tactile_before"].items():
                row[key+"_before"] = value
            for key, value in record["tactile_after"].items():
                row[key+"_after"] = value
            for key in ("true_grasp_before", "physics_path_m", "physics_joint_path_rad", "physics_samples"):
                row[key] = record[key]
        return rows


def eligibility(result, rows, target_id):
    first_lift = next((row for row in rows if row["phase"] == 3), None)
    reasons = []
    if not result["success"]:
        reasons.append("task_not_successful")
    if result["collision_steps"]:
        reasons.append("unwanted_collision")
    if first_lift is None or not (first_lift["grip_confirmed_before"] and first_lift["true_grasp_before"]
                                   and first_lift["tactile_body_id_before"] == target_id):
        reasons.append("no_force_qualified_target_grasp_before_lift")
    if result["success"]:
        final = rows[-40:]
        if len(final) != 40 or [r["hold_streak"] for r in final] != list(range(1, 41)):
            reasons.append("invalid_continuous_hold")
        if min(row["target_lift_m"] for row in final) <= .04:
            reasons.append("insufficient_height")
        if not all(row["target_grasped"] for row in final):
            reasons.append("contact_gap_during_final_hold")
    return reasons


def finalize(env, directory, result, rows, recorder, history=None):
    directory.mkdir(parents=True, exist_ok=True)
    result = dict(result)
    first_lift = next((row for row in rows if row["phase"] == 3), None)
    pre = sum(row["physics_path_m"] for row in rows[:-40]) if result["success"] else None
    result.update(physical_eef_path_m=sum(row["physics_path_m"] for row in rows),
                  physical_eef_path_before_final_hold_m=pre,
                  physical_joint_path_rad=sum(row["physics_joint_path_rad"] for row in rows),
                  physics_samples=sum(row["physics_samples"] for row in rows),
                  control_chord_path_m=sum(row["eef_move_m"] for row in rows),
                  simulated_seconds=len(rows)/env.config.hz,
                  steps_before_final_hold=len(rows)-40 if result["success"] else None,
                  mandatory_final_hold_steps=40 if result["success"] else 0,
                  lift_started_with_confirmed_contact=bool(first_lift and first_lift["grip_confirmed_before"]),
                  correct_target_contact_before_lift=bool(first_lift and first_lift["true_grasp_before"]
                                                        and first_lift["tactile_body_id_before"] == env.color_body_ids[env.target_color]),
                  first_lift_step=None if first_lift is None else first_lift["step"],
                  first70=first_steps_breakdown(rows),
                  directory=str(directory.relative_to(RUN)))
    reasons = eligibility(result, rows, env.color_body_ids[env.target_color])
    result.update(contact_qualified_collision_free_success=not reasons, dataset_rejection_reasons=reasons)
    save_rows(directory / "steps.csv", rows)
    initial = recorder.initial_images
    result["initial_rgb_sha256"] = hashlib.sha256(initial.tobytes()).hexdigest()
    result["initial_robot_state"] = recorder.records[0]["before"]["joints"].tolist()
    Image.fromarray(np.concatenate(list(initial), axis=1)).save(directory / "initial.png")
    Image.fromarray(np.concatenate(list(recorder.records[-1]["images"]), axis=1)).save(directory / "final.png")
    with imageio.get_writer(str(directory / "rollout.mp4"), fps=env.config.hz, codec="libx264", quality=7) as writer:
        writer.append_data(np.concatenate(list(initial), axis=1))
        for row, record in zip(rows, recorder.records):
            frame = np.concatenate(list(record["images"]), axis=1)
            cv2.putText(frame, f'{result["actor"]}/{result["scheme"]} {env.target_color} t={row["step"]} p={row["phase"]} hold={row["hold_streak"]}',
                        (2, 11), cv2.FONT_HERSHEY_SIMPLEX, .25, (0, 0, 0), 1)
            writer.append_data(frame)
    if history is not None:
        arrays = {key: np.asarray(value) for key, value in history.items()}
        arrays["final_images"] = recorder.records[-1]["images"]
        np.savez_compressed(directory / "trajectory.npz", **arrays, metadata=json.dumps(result))
    (directory / "metadata.json").write_text(json.dumps(result, indent=2))
    return result


def run_episode(env, seed, directory, actor, scheme, model=None, checkpoint=None, save_data=False):
    if (directory / "metadata.json").exists():
        return json.loads((directory / "metadata.json").read_text())
    cfg, expert = env.config, configuration()[1]
    tactile_sensor = FingerContactSensor(env, event_config(scheme))
    if actor == "expert" and scheme == "baseline":
        with PhysicsRecorder(env, tactile_sensor) as recorder:
            result, _, diagnostic_rows = old_expert_rollout(env, seed, expert, record=False)
        rows = recorder.enrich(diagnostic_rows, seed=seed)
        result.update(actor=actor, scheme=scheme, policy_queries=0, inference_seconds=0., seed=seed)
        return finalize(env, directory, result, rows, recorder)
    obs = env.reset_seed(seed)
    meta = env.episode_metadata()
    meta.update(actor=actor, scheme=scheme, seed=seed, event_config=asdict(event_config(scheme)))
    event_driven = SCHEMES[scheme]["event_driven"]
    executor = EventExecutor(cfg, expert, event_config(scheme)) if event_driven else RecoveryExecutor(cfg, expert)
    if actor == "expert":
        estimate, details = locate_on_table(env, obs)
        meta.update(vision_source="initial_two_camera_HSV", localization=details, visual_estimate=estimate.tolist())
    diagnostics = MotionDiagnostics(cfg)
    history = {key: [] for key in ("states", "images", "masks", "actions", "osc_actions")} if save_data else None
    schedule = SCHEMES[scheme]["schedule"]
    prediction = None
    last_query = -1000
    queries = 0
    inference_seconds = 0.
    aborted = False
    info = {}
    with PhysicsRecorder(env, tactile_sensor) as recorder:
        for step in range(cfg.horizon):
            phase_before = executor.phase
            tactile = tactile_sensor.read(obs)
            interval = int(schedule[executor.phase])
            new_query = False
            index = -1
            if actor == "expert":
                xyz, requested_phase = estimate, executor.phase
            else:
                new_query = prediction is None or step-last_query >= interval
                if new_query:
                    state = (policy_state(obs, executor.phase)[:16]-checkpoint["stats"]["state_mean"])/checkpoint["stats"]["state_std"]
                    rgb = observation_images(obs)
                    started = time.perf_counter()
                    with torch.inference_mode():
                        output, _, _ = model(torch.as_tensor(state[None], device="cuda"),
                                             torch.as_tensor(rgb.transpose(0, 3, 1, 2)[None].copy(), device="cuda").float()/255,
                                             torch.tensor([COLORS.index(env.target_color)], device="cuda"))
                    prediction = output[0][0].cpu().numpy()
                    inference_seconds += time.perf_counter()-started
                    prediction[:, :3] = prediction[:, :3]*checkpoint["stats"]["xyz_std"] + checkpoint["stats"]["xyz_mean"]
                    last_query = step
                    queries += 1
                index = step-last_query
                assert 0 <= index < len(prediction)
                xyz, requested_phase = prediction[index, :3], int(prediction[index, 3:].argmax())
            if event_driven:
                action, control = executor.action(obs, requested_phase, xyz, tactile)
            else:
                action, control = executor.action(obs, requested_phase, xyz)
            if history is not None:
                history["states"].append(policy_state(obs, control["phase"]))
                history["images"].append(observation_images(obs))
                history["masks"].append(observation_masks(obs))
                history["actions"].append(np.r_[control["goal_xyz"], np.eye(4)[control["phase"]]].astype(np.float32))
                # Preserve the exact OSC inputs for independent physical replay.
                history["osc_actions"].append(action.copy())
            before = obs
            obs, _, done, info = env.step(action)
            diagnostics.update(env, before, obs, control, info)
            row = diagnostics.rows[-1]
            row.update(new_policy_query=new_query, policy_query_age=index, execution_length=interval,
                       transition_event=control.get("transition_event", "baseline_gate" if executor.phase != phase_before else "none"),
                       predicted_xy_error_m=float(np.linalg.norm(np.asarray(before["robot0_eef_pos"])[:2]-np.asarray(xyz)[:2])))
            executor.after_step(obs, control)
            # Contact/pose events interrupt the cached chunk, without hiding a control step.
            if event_driven and executor.phase != phase_before:
                prediction = None
            empty = executor.phase >= 2 and executor.phase_steps >= 12 and np.abs(obs["robot0_gripper_qpos"]).sum() < .006
            if done or empty:
                aborted = bool(empty and not done)
                break
    rows = recorder.enrich(diagnostics.rows, seed=seed)
    result = {**meta, **info, **diagnostics.summary(), "policy_queries": queries, "schedule": schedule,
              "inference_seconds": inference_seconds, "aborted_empty_grasp": aborted,
              "outcome": "success" if info["success"] else "wrong_color" if info["wrong_color_grasp"] else "hold_failure" if info["reached_height"] else "empty_grasp" if aborted else "no_grasp"}
    return finalize(env, directory, result, rows, recorder, history)


def load_model():
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = ConditionalACT(checkpoint["config"]["max_chunk"], pretrained=False).cuda().eval()
    model.load_state_dict(checkpoint["model"])
    assert checkpoint["config"]["max_chunk"] == 4
    return model, checkpoint


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--split", choices=("dev", "test", "collect"), default="dev")
    p.add_argument("--count", type=int)
    args = p.parse_args()
    torch.set_num_threads(4)
    RUN.mkdir(parents=True, exist_ok=True)
    current_hash = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    model, checkpoint = load_model() if args.split != "collect" else (None, None)
    envs = {color: make_env(args.gpu, configuration(color)[0]) for color in COLORS}
    results = []
    try:
        if args.split == "collect":
            selected = json.loads((RUN / "selected_expert.json").read_text())["scheme"]
            jobs = [("train", 86000, 60), ("val", 87000, 10)]
            for split, base, count in jobs:
                for seed in range(base+args.shard, base+(args.count or count), 2):
                    for color in COLORS:
                        directory = RUN / "candidates" / f"{split}_{seed}_{color}"
                        result = run_episode(envs[color], seed, directory, "expert", selected, save_data=True)
                        result.update(split=split)
                        results.append(result)
                        print("CANDIDATE", split, seed, color, result["success"], result["control_steps"], result["dataset_rejection_reasons"], flush=True)
        else:
            base, count = (84000, 8) if args.split == "dev" else (85000, 50)
            for actor in ("expert", "act"):
                for scheme in (("baseline", "efficiency") if actor == "expert" else SCHEMES):
                    for seed in range(base+args.shard, base+(args.count or count), 2):
                        for color in COLORS:
                            directory = RUN / args.split / actor / scheme / f"scene_{seed}" / color
                            result = run_episode(envs[color], seed, directory, actor, scheme, model, checkpoint)
                            results.append(result)
                            print("EFFICIENCY", args.split, actor, scheme, seed, color, result["success"], result["control_steps"], result["collision_steps"], result["lift_started_with_confirmed_contact"], flush=True)
    finally:
        for env in envs.values():
            env.close()
    assert hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() == current_hash
    target = RUN / args.split
    target.mkdir(exist_ok=True)
    (target / f"shard_{args.shard}_completed.json").write_text(json.dumps({"complete": True, "episodes": len(results), "checkpoint_sha256": current_hash}, indent=2))


if __name__ == "__main__":
    main()
