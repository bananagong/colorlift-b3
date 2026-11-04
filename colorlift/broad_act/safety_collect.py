"""Real safe-approach and induced-slip recovery demonstrations on training/val IDs."""
import argparse
from collections import Counter
from dataclasses import replace
import json
import os
import subprocess
import sys
import cv2
import imageio.v2 as imageio
import numpy as np
from .spec import ROOT,DATA,RUN,COLORS
from .runtime import make_env,SafetyRecoveryExecutor
from ..conditional import configuration
from ..broad_v2.layouts import recolor
from ..event_control import FingerContactSensor
from ..event_efficiency import PhysicsRecorder,eligibility
from ..primitives import MotionDiagnostics,policy_state
from ..step_recording import save_rows
from ..vision import observation_images,observation_masks


def selected_layouts():
    manifest=json.loads((DATA / "training_manifest.json").read_text())
    selected=[]
    for split,count in (("train",8),("val",3)):
        for category in ("uniform","sobol","near","edge_adverse"):
            pool=[r for r in manifest["layouts"] if r["split"]==split and r["layout"]["category"]==category]
            # Rank by fixed geometry difficulty, never by learned-policy results.
            pool.sort(key=lambda r:(r["layout"]["features"]["minimum_center_distance_m"],r["layout"]["features"]["edge_distance_m"]))
            selected.extend(r|{"collection_split":split} for r in pool[:count])
    assert len(selected)==44
    return selected


def collect_episode(env,layout,color,directory,regime):
    marker=directory / "metadata.json"
    if marker.exists():return json.loads(marker.read_text())
    directory.mkdir(parents=True,exist_ok=True)
    obs=env.reset_layout(layout,color);meta=env.episode_metadata()
    meta.update(seed=layout["seed"],actor="safe_expert" if regime=="safe" else "contact_recovery_expert",regime=regime,
                directory=str(directory.relative_to(RUN)),teacher_uses_ground_truth=True,student_input_excludes_truth=True)
    cfg=env.config;executor=SafetyRecoveryExecutor(cfg,configuration()[1]);sensor=FingerContactSensor(env)
    diagnostic=MotionDiagnostics(cfg);info={};events=[];active=0;injected=False;contact_lost=False
    ids=sorted(env.gripper_geom_ids | env.color_geom_ids[color]);model=env.sim.model._model
    friction=model.geom_friction[ids].copy();body=env.color_body_ids[color]
    rotation=env.sim.data.site_xmat[next(iter(env.robots[0].eef_site_id.values()))].reshape(3,3).copy()
    data={k:[] for k in ("states","images","masks","actions","servo_waypoints","osc_actions","supervised","recovery")}
    initial=observation_images(obs)
    try:
        with PhysicsRecorder(env,sensor) as recorder:
            for tick in range(cfg.horizon):
                if regime=="recovery" and not injected and info.get("hold_streak",0)>=3:
                    injected=True;active=3
                    model.geom_friction[ids,0]=.005
                    events.append({"start_step":tick+1,"duration_steps":3,"low_friction":.005,"external_force_z_N":-.4,"target_body_id":body})
                forced=active>0
                env.sim.data.xfrc_applied[body,:]=[0,0,-.4,0,0,0] if forced else 0
                truth=env.sim.data.body_xpos[body].copy()
                action,control=executor.action(obs,executor.phase,truth,sensor.read(obs))
                macro=control["goal_xyz"].copy()
                if control["phase"]<2:macro=np.r_[truth[:2],.865 if control["phase"]==0 else cfg.grasp_height]
                for key,value in {"states":policy_state(obs,control["phase"]),"images":observation_images(obs),"masks":observation_masks(obs),
                                  "actions":np.r_[macro,np.eye(4)[control["phase"]]].astype(np.float32),"servo_waypoints":control["goal_xyz"].copy(),
                                  "osc_actions":action.copy(),"supervised":not forced,"recovery":executor.retries>0 and executor.phase<3}.items():data[key].append(value)
                before=obs;obs,_,done,info=env.step(action)
                diagnostic.update(env,before,obs,control,info)
                diagnostic.rows[-1].update(transition_event=control["transition_event"],hold_drift_m=info["hold_drift_m"],injected=forced)
                executor.after_step(obs,control)
                if injected and not sensor.read(obs)["tactile_bilateral"]:contact_lost=True
                if active:
                    active-=1
                    if not active:
                        model.geom_friction[ids]=friction;env.sim.data.xfrc_applied[body,:]=0
                        events[-1]["end_step"]=tick+1
                if done:break
        rows=recorder.enrich(diagnostic.rows,seed=layout["seed"])
        result={**meta,**info,**diagnostic.summary(),"control_steps":len(rows),"retries":executor.retries,"slip_events":events,
                "actual_contact_loss_observed":contact_lost,"physical_eef_path_m":sum(r["physics_path_m"] for r in rows)}
        reasons=eligibility(result,rows,body)
        if regime=="recovery" and not (injected and contact_lost and executor.retries>=1):reasons.append("no_verified_physical_contact_recovery")
        result.update(eligible=not reasons,rejection_reasons=reasons)
        save_rows(directory / "steps.csv",rows)
        target=np.array([[r["target_before_"+a] for a in "xyz"] for r in rows])
        eef=np.array([[r["eef_before_"+a] for a in "xyz"] for r in rows])
        pre=np.c_[target[:,:2],np.full(len(rows),.925)]
        n=len(rows);valid=np.array(data["supervised"],bool)
        np.savez_compressed(directory / "trajectory.npz",**{k:np.asarray(v) for k,v in data.items()},
                            relative=target-eef,pre_relative=pre-eef,pre_rotation=np.repeat(rotation[None],n,axis=0),metadata=json.dumps(result))
        with imageio.get_writer(str(directory / "front.mp4"),fps=20,codec="libx264",quality=7) as writer:
            for number,image in enumerate([initial[0]]+[r["images"][0] for r in recorder.records]):
                frame=np.full((592,512,3),250,np.uint8);frame[80:]=cv2.resize(image,(512,512),interpolation=cv2.INTER_NEAREST)
                for y,text in zip((20,43,66),(f'{regime} expert | {color} | front only',f'{layout["category"]} | seed {layout["seed"]}',f'{number}/{n} | success {result["success"]} | contact {result["eligible"]} | retries {executor.retries}')):
                    cv2.putText(frame,text,(8,y),cv2.FONT_HERSHEY_SIMPLEX,.43,(20,20,20),1)
                writer.append_data(frame)
        marker.write_text(json.dumps(result,indent=2));return result
    finally:
        model.geom_friction[ids]=friction;env.sim.data.xfrc_applied[body,:]=0


def worker(shard,gpu,smoke=False):
    env=make_env(gpu);env.config=replace(env.config,horizon=320)
    results=[]
    try:
        selection=selected_layouts()
        if smoke:
            selection=[next(r for r in selection if r["collection_split"]=="train" and r["layout"]["category"]==category) for category in ("near","edge_adverse")]
        for index,row in enumerate(selection):
            if not smoke and index%4!=shard:continue
            for regime in (("recovery",) if smoke else ("safe","recovery")):
                for permutation in (range(1) if smoke else range(3)):
                    layout=recolor(row["layout"],permutation)
                    for color in (("red",) if smoke else COLORS):
                        directory=RUN / ("safety_smoke" if smoke else "safety_data") / row["collection_split"] / row["geometry_id"] / regime / f"p{permutation}_{color}"
                        meta=collect_episode(env,layout,color,directory,regime);results.append(meta)
                        print("SAFETY_DATA",row["collection_split"],index,regime,permutation,color,meta["success"],meta["eligible"],meta["retries"],meta["control_steps"],meta["rejection_reasons"],flush=True)
    finally:env.close()
    if smoke:
        (RUN / "safety_smoke.json").write_text(json.dumps({"cases":results,"passed":all(r["eligible"] for r in results)},indent=2))
        assert all(r["eligible"] for r in results)


def assemble():
    selected=[];counts=Counter();rejected=[]
    for row in selected_layouts():
        for regime in ("safe","recovery"):
            directory=RUN / "safety_data" / row["collection_split"] / row["geometry_id"] / regime
            episodes=[json.loads(p.read_text()) for p in sorted(directory.glob("*/metadata.json"))]
            assert len(episodes)==9
            if all(r["eligible"] for r in episodes):
                selected.extend({"split":row["collection_split"],"regime":regime,"category":row["layout"]["category"],"geometry_id":row["geometry_id"],"directory":r["directory"],"target_color":r["target_color"],"steps":r["control_steps"]} for r in episodes)
                counts[row["collection_split"]+"_"+regime]+=1
            else:rejected.append({"split":row["collection_split"],"geometry_id":row["geometry_id"],"regime":regime,"failed_cases":[r["directory"] for r in episodes if not r["eligible"]]})
    assert counts["train_recovery"]>=12 and counts["val_recovery"]>=4,dict(counts)
    (RUN / "safety_manifest.json").write_text(json.dumps({"episodes":selected,"complete_geometry_regime_groups":dict(counts),"rejected_groups":rejected,
                                                         "perturbations_not_new_layouts":True,"forced_steps_masked":True,"all_actual_physics":True},indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--launch",action="store_true");p.add_argument("--smoke",action="store_true");p.add_argument("--shard",type=int,default=0);p.add_argument("--gpu",type=int,default=1);a=p.parse_args()
    if a.launch:
        handles=[];processes=[]
        for shard in range(4):
            gpu=(1,2)[shard%2];env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),MUJOCO_EGL_DEVICE_ID=str(gpu),OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
            log=(RUN / f"safety_collect_{shard}.log").open("a");handles.append(log)
            processes.append(subprocess.Popen([sys.executable,"-u","-m","colorlift.broad_act.safety_collect","--shard",str(shard),"--gpu",str(gpu)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT))
        codes=[process.wait() for process in processes]
        for h in handles:h.close()
        assert codes==[0]*4,codes
        assemble()
    else:worker(a.shard,a.gpu,a.smoke)
