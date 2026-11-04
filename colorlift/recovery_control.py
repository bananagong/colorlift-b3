"""Four-phase adapter that can execute learned vertical recovery goals during alignment."""
import numpy as np


class RecoveryExecutor:
    def __init__(self, config, expert):
        self.cfg, self.expert=config, expert
        self.phase=self.phase_steps=self.close_steps=self.stationary_steps=self.transitions=0
        self.last_error=self.last_fingers=self.filtered_xy=self.grasp_xy=None
        self.recovering_alignment=False

    def action(self, obs, requested_phase, xyz):
        cfg, e=self.cfg, self.expert
        fingers=np.asarray(obs["robot0_gripper_qpos"])
        eef=np.asarray(obs["robot0_eef_pos"])
        still=self.last_fingers is not None and np.linalg.norm(fingers-self.last_fingers)<e.finger_motion_tolerance
        self.stationary_steps=self.stationary_steps+1 if still else 0
        if self.last_error is not None:
            reached=np.linalg.norm(self.last_error[:2])<=e.xy_tolerance and abs(self.last_error[2])<=e.xyz_tolerance
            advance=self.phase in (0, 1) and reached and self.phase_steps>=3
            # Reaching a raised recovery waypoint must NOT trigger gripper closure.
            if self.phase == 1: advance &= abs(eef[2]-cfg.grasp_height)<=.005 and not self.recovering_alignment
            advance |= self.phase == 2 and self.close_steps>=e.minimum_close_steps and self.stationary_steps>=e.stationary_finger_steps
            if advance:
                self.phase+=1; self.phase_steps=0; self.stationary_steps=0; self.transitions+=1
                if self.phase == 2: self.grasp_xy=eef[:2].copy()
        xy=np.asarray(xyz[:2], dtype=float).clip(-.15, .15)
        self.filtered_xy=xy if self.filtered_xy is None else .6*xy+.4*self.filtered_xy
        goal=np.r_[self.filtered_xy, cfg.grasp_height]
        if self.phase == 0: goal[2]=cfg.table_height+cfg.cube_size/2+e.pregrasp_clearance
        elif self.phase == 1:
            # Generic clearance gate based ONLY on robot state and ACT's predicted target.
            # Both old/new checkpoints use this same gate; no HSV or object truth enters it.
            xy_error=np.linalg.norm(eef[:2]-self.filtered_xy)
            if xy_error>.012: self.recovering_alignment=True
            if self.recovering_alignment:
                if eef[2]<.870:
                    goal[:2]=eef[:2]
                elif xy_error<=.004:
                    self.recovering_alignment=False
                goal[2]=.875 if self.recovering_alignment else cfg.grasp_height
        if self.phase>=2:
            goal[:2]=self.grasp_xy; self.close_steps+=1
        if self.phase == 3: goal[2]=cfg.table_height+cfg.cube_size/2+cfg.lift_threshold+e.lift_margin
        delta=goal-eef
        delta[np.abs(delta)<e.deadband]=0
        maximum=e.align_maximum_step if self.phase==1 else e.maximum_step
        if np.linalg.norm(delta)>maximum: delta*=maximum/np.linalg.norm(delta)
        grip=(-1. if np.abs(fingers).sum()<e.open_gap else 0.) if self.phase<2 else 1.
        self.last_fingers=fingers.copy(); self.phase_steps+=1
        control={"phase": self.phase, "requested_phase": int(requested_phase), "close_steps": self.close_steps,
                 "goal_xyz": goal, "command_translation_m": delta.copy()}
        return np.r_[delta/.05, np.zeros(3), grip], control

    def after_step(self, obs, control):
        self.last_error=np.asarray(obs["robot0_eef_pos"])-control["goal_xyz"]
