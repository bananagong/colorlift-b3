"""Matched OSC rollouts with physical scoring; target truth is evaluation-only."""
from dataclasses import replace
import hashlib
import json
import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image
import torch
from .spec import RUN, COLORS
from ..config import CAMERAS
from ..conditional import configuration
from ..broad_v2.validity import BroadColorLift
from ..broad_v2.rollout import SafeExecutor
from ..event_control import EventExecutor, FingerContactSensor, event_config
from ..event_efficiency import PhysicsRecorder, eligibility
from ..primitives import MotionDiagnostics, policy_state
from ..step_recording import save_rows
from ..vision import observation_images


def make_env(gpu):
    return BroadColorLift(config=replace(configuration()[0], horizon=220), has_renderer=False, has_offscreen_renderer=True,
                          use_camera_obs=True, camera_names=list(CAMERAS), camera_segmentations=None,
                          render_gpu_device_id=gpu, hard_reset=False, seed=0)


class SafetyRecoveryExecutor(SafeExecutor):
    def __init__(self, cfg, expert, event=None):
        super().__init__(cfg, expert)
        self.retries = 0; self.lost = 0; self.was_gripped = False; self.reopening = False

    def action(self, obs, requested, xyz, tactile):
        eef = np.asarray(obs["robot0_eef_pos"]); gap = float(np.abs(obs["robot0_gripper_qpos"]).sum())
        xyz = np.asarray(xyz).copy(); xyz[:2] = xyz[:2].clip(-.15,.15)
        if tactile["grip_confirmed"]: self.was_gripped = True
        self.lost = self.lost+1 if self.phase == 3 and self.was_gripped and not tactile["tactile_bilateral"] else 0
        empty = self.phase == 2 and self.phase_steps >= 8 and gap < .012 and not tactile["tactile_bilateral"]
        if (empty or self.lost >= 3) and self.retries < 2 and not self.reopening:
            self.retries += 1; self.phase = self.phase_steps = self.close_steps = self.route_index = 0
            self.filtered_xy = self.grasp_xy = self.start_xy = self.transport_z = None
            self.was_gripped = False; self.lost = 0; self.reopening = True; self.align_retreat = False
        if self.reopening:
            if gap >= self.expert.open_gap and eef[2] >= .90: self.reopening = False
            else:
                goal = np.r_[eef[:2], max(float(eef[2]),.925)]
                delta = goal-eef
                if np.linalg.norm(delta) > .035: delta *= .035/np.linalg.norm(delta)
                self.phase_steps += 1; self.last_goal = goal.copy()
                return np.r_[delta/.05,np.zeros(3),-1.], {"phase":0,"requested_phase":0,"close_steps":0,"goal_xyz":goal,
                    "command_translation_m":delta,"transition_event":"contact_loss_reopen_retreat","phase_changed":True,
                    "predicted_xy_error_m":float(np.linalg.norm(xyz[:2]-eef[:2]))}
        return super().action(obs,requested,xyz,tactile)


def episode(env, layout, color, directory, actor, model, checkpoint, controller="original", representative=False):
    marker = directory / "metadata.json"
    if marker.exists(): return json.loads(marker.read_text())
    directory.mkdir(parents=True, exist_ok=True)
    obs = env.reset_layout(layout,color); initial = observation_images(obs)
    meta = env.episode_metadata()
    meta.update(actor=actor,controller=controller,seed=layout["seed"],directory=str(directory.relative_to(RUN)),
                initial_rgb_sha256=hashlib.sha256(initial.tobytes()).hexdigest(),initial_robot_state=policy_state(obs,0).tolist())
    executor = SafetyRecoveryExecutor(env.config,configuration()[1]) if controller == "safety_recovery" else EventExecutor(env.config,configuration()[1],event_config("efficiency"))
    sensor = FingerContactSensor(env); diagnostic = MotionDiagnostics(env.config)
    prediction = None; last_query = -1000; queries = 0; info = {}; query_ticks = []
    if hasattr(model,"aux_trace"): model.aux_trace = []; model.record_aux = True
    with PhysicsRecorder(env,sensor) as recorder:
        for tick in range(env.config.horizon):
            phase_before = executor.phase; interval = (4,2,1,4)[executor.phase]
            new_query = prediction is None or tick-last_query >= interval
            if new_query:
                states = (policy_state(obs,executor.phase)[:16]-checkpoint["stats"]["state_mean"])/checkpoint["stats"]["state_std"]
                rgb = observation_images(obs)
                with torch.inference_mode():
                    output,_,_ = model(torch.as_tensor(states[None],device="cuda"),torch.as_tensor(rgb.transpose(0,3,1,2)[None].copy(),device="cuda").float()/255,
                                       torch.tensor([COLORS.index(color)],device="cuda"))
                prediction = output[0][0].cpu().numpy()
                prediction[:,:3] = prediction[:,:3]*checkpoint["stats"]["xyz_std"]+checkpoint["stats"]["xyz_mean"]
                last_query = tick; queries += 1; query_ticks.append(tick)
            xyz = prediction[tick-last_query,:3]; requested = int(prediction[tick-last_query,3:].argmax())
            action,control = executor.action(obs,requested,xyz,sensor.read(obs))
            before = obs; obs,_,done,info = env.step(action)
            diagnostic.update(env,before,obs,control,info)
            diagnostic.rows[-1].update(transition_event=control["transition_event"],new_policy_query=new_query,execution_length=interval,hold_drift_m=info["hold_drift_m"])
            executor.after_step(obs,control)
            if executor.phase != phase_before: prediction = None
            empty = executor.phase >= 2 and executor.phase_steps >= 12 and np.abs(obs["robot0_gripper_qpos"]).sum()<.006
            if done or empty: break
    rows = recorder.enrich(diagnostic.rows,seed=layout["seed"])
    result = {**meta,**info,**diagnostic.summary(),"control_steps":len(rows),"policy_queries":queries,
              "physical_eef_path_m":sum(r["physics_path_m"] for r in rows),"contact_retries":getattr(executor,"retries",0)}
    reasons = eligibility(result,rows,env.color_body_ids[color]); result.update(eligible=not reasons,rejection_reasons=reasons)
    closing = next((r for r in rows if r["phase"]==2),None)
    result["close_XY_error_m"] = float(np.linalg.norm([closing["eef_before_"+a]-closing["target_before_"+a] for a in "xy"])) if closing else None
    result["outcome"] = "success" if result["success"] else "wrong_color" if result["wrong_color_grasp"] else "hold_or_slip" if result["reached_height"] else "no_grasp"
    if hasattr(model,"aux_trace"):
        assert len(model.aux_trace) == len(query_ticks)
        errors = []; pre_errors = []
        for tick,aux in zip(query_ticks,model.aux_trace):
            row = rows[tick]
            truth = np.array([row["target_before_"+a]-row["eef_before_"+a] for a in "xyz"])
            errors.append(np.abs(aux["relative"][0]-truth))
            if row["phase"]<2:
                pre = np.array([row["target_before_x"]-row["eef_before_x"],row["target_before_y"]-row["eef_before_y"],.925-row["eef_before_z"]])
                pre_errors.append(np.abs(aux["pre_relative"][0]-pre))
        result["relative_MAE_m"] = np.mean(errors,axis=0).tolist() if errors else None
        result["pregrasp_MAE_m"] = np.mean(pre_errors,axis=0).tolist() if pre_errors and checkpoint.get("pregrasp_auxiliary",False) else None
        model.record_aux = False; model.aux_trace = []
    save_rows(directory / "steps.csv",rows)
    save_media = not result["eligible"] or representative
    if save_media:
        Image.fromarray(initial[0]).save(directory / "initial_front.png")
        with imageio.get_writer(str(directory / "front.mp4"),fps=20,codec="libx264",quality=7) as writer:
            for n,im in enumerate([initial[0]]+[r["images"][0] for r in recorder.records]):
                frame = np.full((592,512,3),250,np.uint8); frame[80:] = cv2.resize(im,(512,512),interpolation=cv2.INTER_NEAREST)
                for y,text in zip((20,43,66),(f'{actor} | {color} | front camera only',f'{layout["level"]} | seed {layout["seed"]}',f'{n}/{len(rows)} | PASS {result["success"]} | contact {result["eligible"]}')):
                    cv2.putText(frame,text,(8,y),cv2.FONT_HERSHEY_SIMPLEX,.43,(20,20,20),1)
                writer.append_data(frame)
    marker.write_text(json.dumps(result,indent=2))
    return result
