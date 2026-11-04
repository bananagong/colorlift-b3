"""Physics-executed 3cm disturbances and IK-replanned scripted recovery demonstrations."""
from dataclasses import dataclass, asdict, replace
import json
import numpy as np
import mujoco
from .conditional import configuration
from .vision import observation_images, observation_masks
from .primitives import policy_state, MotionDiagnostics


@dataclass
class NoiseConfig:
    amplitude_m: float = .03
    interval_expert_steps: int = 8
    first_event_after_steps: int = 6
    maximum_events: int = 2
    maximum_disturbance_steps: int = 12
    offset_reached_tolerance_m: float = .0025
    recovery_height: float = .875
    align_tolerance_m: float = .003
    align_confirmation_steps: int = 3
    horizon: int = 240


class DampedIK:
    """Solve six-dimensional site pose IK on a private MjData, never teleport the live robot."""
    def __init__(self, env):
        self.env=env
        self.model=env.sim.model._model
        self.data=mujoco.MjData(self.model)
        self.site=next(iter(env.robots[0].eef_site_id.values()))
        joint_names=env.robots[0].robot_model.joints
        ids=[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names]
        self.qpos=np.array([self.model.jnt_qposadr[j] for j in ids])
        self.dofs=np.array([self.model.jnt_dofadr[j] for j in ids])
        self.limits=self.model.jnt_range[ids]

    def solve(self, position, maximum_iterations=40):
        self.data.qpos[:]=self.env.sim.data.qpos
        self.data.qvel[:]=0
        rotation=self.env.sim.data.site_xmat[self.site].reshape(3, 3).copy()
        jp=np.zeros((3, self.model.nv)); jr=np.zeros_like(jp)
        for iteration in range(maximum_iterations):
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)
            current=self.data.site_xmat[self.site].reshape(3, 3)
            error_position=np.asarray(position)-self.data.site_xpos[self.site]
            error_rotation=.5*sum(np.cross(current[:, i], rotation[:, i]) for i in range(3))
            if np.linalg.norm(error_position) < .0002 and np.linalg.norm(error_rotation) < .002:
                break
            mujoco.mj_jacSite(self.model, self.data, jp, jr, self.site)
            jac=np.vstack((jp[:, self.dofs], .4*jr[:, self.dofs]))
            error=np.r_[error_position, .4*error_rotation]
            change=jac.T @ np.linalg.solve(jac @ jac.T + .0001*np.eye(6), error)
            norm=np.linalg.norm(change)
            if norm > .18: change*=.18/norm
            self.data.qpos[self.qpos]=np.clip(self.data.qpos[self.qpos]+change, self.limits[:, 0]+.005, self.limits[:, 1]-.005)
        mujoco.mj_kinematics(self.model, self.data)
        realized=self.data.site_xpos[self.site].copy()
        return realized, {"ik_iterations": iteration+1, "ik_position_residual_m": float(np.linalg.norm(realized-position)),
                          "ik_joint_solution": self.data.qpos[self.qpos].copy()}


class OffsetInjector:
    """Exogenous OSC motions toward an IK-validated offset; all such actions have loss mask false."""
    def __init__(self, seed, color_index, config=None):
        self.cfg=config or NoiseConfig()
        self.rng=np.random.default_rng(seed*17+color_index+910001)
        self.events=[]
        self.active=None
        self.last_expert_step=-1000

    def ready(self, phase, phase_steps, expert_steps, recovering):
        if self.active is not None or recovering or phase >= 2 or len(self.events) >= self.cfg.maximum_events: return False
        if not self.events: return expert_steps >= self.cfg.first_event_after_steps
        return phase == 1 and phase_steps >= 2 and expert_steps-self.last_expert_step >= self.cfg.interval_expert_steps

    def start(self, obs, expert_steps, control_step):
        angle=self.rng.uniform(0, 2*np.pi)
        offset=np.array([np.cos(angle), np.sin(angle), 0])*self.cfg.amplitude_m
        origin=np.asarray(obs["robot0_eef_pos"]).copy()
        self.active={"event_id": len(self.events)+1, "start_step": control_step+1, "origin": origin,
                     "requested_offset": offset, "goal": origin+offset, "steps": 0}
        self.last_expert_step=expert_steps

    def after(self, obs, control_step):
        event=self.active
        event["steps"]+=1
        reached=np.linalg.norm(np.asarray(obs["robot0_eef_pos"])-event["goal"]) <= self.cfg.offset_reached_tolerance_m
        if reached or event["steps"] >= self.cfg.maximum_disturbance_steps:
            realized=np.asarray(obs["robot0_eef_pos"])-event["origin"]
            result={key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in event.items()}
            result.update(end_step=control_step, realized_offset=realized.tolist(), realized_offset_norm_m=float(np.linalg.norm(realized)), reached_requested_offset=bool(reached))
            self.events.append(result)
            self.active=None
            return True
        return False


def ik_osc_action(env, obs, goal, grip, ik, maximum=.045):
    # Re-solve from the actual post-disturbance joint state; execute the FK-validated waypoint via OSC.
    feasible, diagnostic=ik.solve(np.asarray(goal))
    delta=feasible-np.asarray(obs["robot0_eef_pos"])
    delta[np.abs(delta)<.0007]=0
    norm=np.linalg.norm(delta)
    if norm > maximum: delta*=maximum/norm
    return np.r_[delta/.05, np.zeros(3), grip], delta, diagnostic


def rollout(env, seed, noise=None, record=True):
    cfg=env.config
    ncfg=noise or NoiseConfig()
    expert=configuration()[1]
    obs=env.reset_seed(seed)
    metadata=env.episode_metadata()
    metadata.update(seed=seed, noise_config=asdict(ncfg), teacher_target_source="simulator body pose for expert labels only")
    injector=OffsetInjector(seed, ("red", "yellow", "blue").index(env.target_color), ncfg)
    ik=DampedIK(env)
    diagnostic=MotionDiagnostics(cfg)
    history={key: [] for key in ("states", "images", "masks", "actions", "osc_actions", "supervised", "category", "recovery", "ik_qpos")}
    phase=phase_steps=expert_steps=align_confirm=close_steps=0
    recovery=False; recovery_stage=0; retreat_xy=None; grasp_xy=None; last_fingers=None; stationary=0
    info={}
    for step in range(cfg.horizon):
        if injector.ready(phase, phase_steps, expert_steps, recovery):
            injector.start(obs, expert_steps, step)
        injected=injector.active is not None
        target=env.sim.data.body_xpos[env.color_body_ids[env.target_color]].copy()
        eef=np.asarray(obs["robot0_eef_pos"])
        if injected:
            goal=injector.active["goal"].copy()
            category="injected_offset"
        elif recovery and phase == 1:
            if recovery_stage == 0: goal=np.r_[retreat_xy, ncfg.recovery_height]
            elif recovery_stage == 1: goal=np.r_[target[:2], ncfg.recovery_height]
            else: goal=np.r_[target[:2], cfg.grasp_height]
            category="recovery"
        elif phase == 0:
            goal=np.r_[target[:2], cfg.table_height+cfg.cube_size/2+expert.pregrasp_clearance]
            category="recovery" if recovery else "approach"
        elif phase == 1:
            goal=np.r_[target[:2], cfg.grasp_height]
            category="precise_align"
        elif phase == 2:
            goal=np.r_[grasp_xy, cfg.grasp_height]
            category="closing_contact"
        else:
            goal=np.r_[grasp_xy, cfg.table_height+cfg.cube_size/2+cfg.lift_threshold+expert.lift_margin]
            category="lift_hold"
        gap=float(np.abs(obs["robot0_gripper_qpos"]).sum())
        grip=(-1. if gap<expert.open_gap else 0.) if phase<2 else 1.
        action, delta, ik_info=ik_osc_action(env, obs, goal, grip, ik)
        if record:
            history["states"].append(policy_state(obs, phase))
            history["images"].append(observation_images(obs))
            history["masks"].append(observation_masks(obs))
            history["actions"].append(np.r_[goal, np.eye(4)[phase]].astype(np.float32))
            history["osc_actions"].append(action.astype(np.float32))
            history["supervised"].append(not injected)
            history["category"].append(category)
            history["recovery"].append(recovery and not injected)
            history["ik_qpos"].append(ik_info["ik_joint_solution"])
        before=obs
        obs, _, done, info=env.step(action)
        if phase >= 2: close_steps+=1
        control={"phase": phase, "requested_phase": phase, "close_steps": close_steps,
                 "goal_xyz": goal.copy(), "command_translation_m": delta.copy()}
        diagnostic.update(env, before, obs, control, info)
        row=diagnostic.rows[-1]
        row.update(supervised=not injected, training_category=category, recovery_active=recovery and not injected,
                   noise_event_id=injector.active["event_id"] if injected else 0,
                   ik_iterations=ik_info["ik_iterations"], ik_position_residual_m=ik_info["ik_position_residual_m"])
        for j, value in enumerate(ik_info["ik_joint_solution"]): row[f"ik_joint_{j}"]=float(value)
        if injected:
            if injector.after(obs, step+1):
                recovery=True; recovery_stage=0
                retreat_xy=np.asarray(obs["robot0_eef_pos"][:2]).copy()
                align_confirm=0; phase_steps=0
        else:
            expert_steps+=1; phase_steps+=1
            error=np.asarray(obs["robot0_eef_pos"])-goal
            reached=np.linalg.norm(error[:2]) <= ncfg.align_tolerance_m and abs(error[2]) <= .004
            if recovery:
                if reached:
                    if phase == 1 and recovery_stage < 2: recovery_stage+=1
                    else: recovery=False; align_confirm=0; phase_steps=0
            elif phase == 0 and reached:
                phase=1; phase_steps=0; align_confirm=0
            elif phase == 1:
                align_confirm=align_confirm+1 if reached else 0
                if align_confirm >= ncfg.align_confirmation_steps and len(injector.events) >= ncfg.maximum_events:
                    phase=2; phase_steps=0; grasp_xy=np.asarray(obs["robot0_eef_pos"][:2]).copy()
            elif phase == 2:
                fingers=np.asarray(obs["robot0_gripper_qpos"])
                stationary=stationary+1 if last_fingers is not None and np.linalg.norm(fingers-last_fingers)<.0003 else 0
                last_fingers=fingers.copy()
                if close_steps >= 3 and stationary >= 2: phase=3; phase_steps=0
        if done: break
    result={**metadata, **info, **diagnostic.summary(), "noise_events": injector.events,
            "supervised_steps": int(sum(history["supervised"])) if record else expert_steps,
            "recovery_steps": sum(row["training_category"] == "recovery" for row in diagnostic.rows),
            "maximum_ik_residual_m": max(row["ik_position_residual_m"] for row in diagnostic.rows),
            "outcome": "success" if info.get("success") else "wrong_color" if info.get("wrong_color_grasp") else "hold_failure" if info.get("reached_height") else "no_grasp"}
    if record:
        history={key: np.asarray(value) for key, value in history.items()}
        history["final_images"]=observation_images(obs)
    return result, history, diagnostic.rows
