"""Fresh-scene comparisons of original ACT, shorter execution and learned recovery."""
import argparse
from dataclasses import replace
import hashlib
import json
import numpy as np
import torch
import imageio.v2 as imageio
import cv2
from PIL import Image
from .conditional import RUN as ORIGINAL, ConditionalACT, FastExecutor, configuration
from .config import ROOT, COLORS
from .env import make_env
from .recovery_control import RecoveryExecutor
from .recovery_expert import NoiseConfig, OffsetInjector, DampedIK, ik_osc_action
from .vision import observation_images
from .primitives import MotionDiagnostics, policy_state
from .step_recording import StepRecorder, save_rows

RUN=ROOT / "experiments/colorlift_recovery_v1"


def episode(env, model, checkpoint, seed, directory, variant, schedule, disturbed):
    directory.mkdir(parents=True, exist_ok=True)
    cfg, expert=env.config, configuration()[1]
    obs=env.reset_seed(seed)
    initial=observation_images(obs)
    meta=env.episode_metadata()
    meta.update(seed=seed, initial_rgb_sha256=hashlib.sha256(initial.tobytes()).hexdigest(), initial_robot_state=policy_state(obs, 0)[:16].tolist())
    executor=(FastExecutor if variant in ("old_original", "old_long") else RecoveryExecutor)(cfg, expert)
    diagnostics=MotionDiagnostics(cfg)
    injector=OffsetInjector(seed, COLORS.index(env.target_color)) if disturbed else None
    ik=DampedIK(env) if disturbed else None
    nominal_steps=queries=0; last_query=-1000; prediction=None; info={}; first_picked=None; grasp_streak={c: 0 for c in COLORS}
    aborted=False
    with StepRecorder(env) as recorder, imageio.get_writer(str(directory / "rollout.mp4"), fps=cfg.hz, codec="libx264", quality=7) as writer:
        writer.append_data(np.concatenate(list(initial), axis=1))
        Image.fromarray(np.concatenate(list(initial), axis=1)).save(directory / "initial.png")
        for step in range(cfg.horizon):
            if disturbed and injector.ready(executor.phase, executor.phase_steps, nominal_steps, False): injector.start(obs, nominal_steps, step)
            injected=disturbed and injector.active is not None
            interval=int(schedule[executor.phase])
            if injected:
                goal=injector.active["goal"].copy()
                grip=-1. if abs(obs["robot0_gripper_qpos"]).sum()<expert.open_gap else 0.
                action, delta, ik_info=ik_osc_action(env, obs, goal, grip, ik)
                control={"phase": executor.phase, "requested_phase": executor.phase, "close_steps": executor.close_steps,
                         "goal_xyz": goal, "command_translation_m": delta}
                new_query=False; index=-1
            else:
                new_query=prediction is None or nominal_steps-last_query>=interval
                if new_query:
                    state=(policy_state(obs, executor.phase)[:16]-checkpoint["stats"]["state_mean"])/checkpoint["stats"]["state_std"]
                    rgb=observation_images(obs)
                    with torch.inference_mode():
                        output, _, _=model(torch.as_tensor(state[None], device="cuda"),
                                           torch.as_tensor(rgb.transpose(0, 3, 1, 2)[None].copy(), device="cuda").float()/255,
                                           torch.tensor([COLORS.index(env.target_color)], device="cuda"))
                    prediction=output[0][0].cpu().numpy()
                    prediction[:, :3]=prediction[:, :3]*checkpoint["stats"]["xyz_std"]+checkpoint["stats"]["xyz_mean"]
                    last_query=nominal_steps; queries+=1
                index=nominal_steps-last_query
                assert index<len(prediction)
                action, control=executor.action(obs, int(prediction[index, 3:].argmax()), prediction[index, :3])
            before=obs
            obs, _, done, info=env.step(action)
            diagnostics.update(env, before, obs, control, info)
            row=diagnostics.rows[-1]
            row.update(injected_offset=bool(injected), policy_query_age=index, new_policy_query=new_query, execution_length=interval)
            if injected:
                if injector.after(obs, step+1):
                    prediction=None; executor.last_error=None; executor.stationary_steps=0
            else:
                nominal_steps+=1; executor.after_step(obs, control)
            for color in COLORS:
                grasp=info["grasp_by_color"][color]
                grasp_streak[color]=grasp_streak[color]+1 if grasp else 0
                if grasp_streak[color]>=3 and first_picked is None: first_picked=color
                row["grasp_"+color]=grasp
            frame=np.concatenate(list(observation_images(obs)), axis=1)
            label="OFFSET" if injected else "ACT"
            cv2.putText(frame, f'{variant} {env.target_color} {label} t={step+1} p={executor.phase}', (2, 11), cv2.FONT_HERSHEY_SIMPLEX, .28, (0, 0, 0), 1)
            writer.append_data(frame)
            empty=not injected and executor.phase==3 and executor.close_steps>=12 and abs(obs["robot0_gripper_qpos"]).sum()<.006
            if done or empty:
                aborted=bool(empty and not done); break
    rows=recorder.enrich(diagnostics.rows, seed=seed)
    save_rows(directory / "steps.csv", rows)
    Image.fromarray(np.concatenate(list(observation_images(obs)), axis=1)).save(directory / "final.png")
    closing=next((row for row in rows if row["phase"]==2), None)
    result={**meta, **info, **diagnostics.summary(), "variant": variant, "disturbed": disturbed,
            "schedule": list(schedule), "policy_queries": queries, "max_chunk": checkpoint["config"]["max_chunk"],
            "noise_events": injector.events if disturbed else [], "first_picked_color": first_picked or "none",
            "closing_true_xy_error_m": None if closing is None else float(np.linalg.norm([closing["eef_before_"+a]-closing["target_before_"+a] for a in "xy"])),
            "outcome": "success" if info["success"] else "wrong_color" if info["wrong_color_grasp"] else "hold_failure" if info["reached_height"] else "empty_grasp" if aborted else "no_grasp",
            "directory": str(directory.relative_to(RUN))}
    if result["success"]:
        assert [int(row["hold_streak"]) for row in rows[-40:]]==list(range(1, 41))
        assert min(row["target_lift_m"] for row in rows[-40:])>.04
    (directory / "metadata.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    p=argparse.ArgumentParser(); p.add_argument("--gpu", type=int, required=True); p.add_argument("--shard", type=int, required=True)
    p.add_argument("--split", choices=("dev", "test"), default="dev"); args=p.parse_args()
    torch.set_num_threads(4)
    protocol=json.loads((RUN / "protocol.json").read_text())
    paths={"old": ORIGINAL / "model/policy.pt", "new": RUN / "model/policy.pt"}
    hashes={key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in paths.items()}
    models={}; checkpoints={}
    for key, path in paths.items():
        ckpt=torch.load(path, map_location="cpu", weights_only=False)
        model=ConditionalACT(ckpt["config"]["max_chunk"], pretrained=False).cuda().eval(); model.load_state_dict(ckpt["model"])
        models[key]=model; checkpoints[key]=ckpt
    envs={}
    for disturbed in (False, True):
        for color in COLORS:
            cfg=configuration(color)[0]
            if disturbed: cfg=replace(cfg, horizon=NoiseConfig().horizon)
            envs[disturbed, color]=make_env(args.gpu, cfg)
    out=RUN / "evaluations" / args.split; out.mkdir(parents=True, exist_ok=True)
    trials=[]
    try:
        for disturbed in (False, True):
            count=(6 if disturbed else 12) if args.split=="dev" else (30 if disturbed else 50)
            base=(80100 if disturbed else 80000) if args.split=="dev" else (82000 if disturbed else 81000)
            variants=("new_recovery",) if args.split=="dev" else ("old_long", "old_short", "new_recovery")
            for variant in variants:
                key="new" if variant=="new_recovery" else "old"
                schedule=[3]*4 if variant=="old_original" else [12]*4 if variant=="old_long" else protocol["selected_query_schedule"]
                for seed in range(base+args.shard, base+count, 2):
                    for color in COLORS:
                        directory=out / ("disturbed" if disturbed else "clean") / variant / f"scene_{seed}" / color
                        if (directory / "metadata.json").exists(): result=json.loads((directory / "metadata.json").read_text())
                        else:
                            print("EXECUTE", args.split, disturbed, variant, seed, color, flush=True)
                            result=episode(envs[disturbed, color], models[key], checkpoints[key], seed, directory, variant, schedule, disturbed)
                        trials.append(result)
                        print("RESULT", args.split, disturbed, variant, seed, color, result["success"], result["control_steps"], result["collision_steps"], len(result["noise_events"]), flush=True)
        if args.split=="test":
            failures=json.loads((ORIGINAL / "failure_cases.json").read_text())
            for index, failed in enumerate(failures):
                if index%2 != args.shard: continue
                seed, color=failed["seed"], failed["target_color"]
                directory=RUN / "regression_old_failures" / f"scene_{seed}" / color
                if (directory / "metadata.json").exists(): result=json.loads((directory / "metadata.json").read_text())
                else: result=episode(envs[False, color], models["new"], checkpoints["new"], seed, directory, "new_recovery", protocol["selected_query_schedule"], False)
                print("OLD_FAILURE_REPLAY", seed, color, result["success"], result["control_steps"], flush=True)
    finally:
        for env in envs.values(): env.close()
    assert all(hashlib.sha256(path.read_bytes()).hexdigest()==hashes[key] for key, path in paths.items())
    (out / f"shard_{args.shard}_completed.json").write_text(json.dumps({"complete": True, "model_hashes": hashes, "episodes": len(trials)}))


if __name__ == "__main__": main()
