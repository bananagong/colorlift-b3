"""Known-signal checks for reversal, plateau, idle, and mandatory-hold exclusion counters."""
import json
from types import SimpleNamespace
import numpy as np
from .config import Config, EXPERIMENT
from .primitives import MotionDiagnostics


def main():
    cfg = Config()
    diagnostic = MotionDiagnostics(cfg)
    diagnostic._direction("command", np.array([.01, 0, 0]), 0, 1)
    diagnostic._direction("command", np.array([-.01, 0, 0]), 0, 2)
    diagnostic._direction("command", np.array([.01, 0, 0]), 1, 3)
    assert diagnostic.counts["command_reversals_all"] == 2
    assert diagnostic.counts["command_reversals_within_phase"] == 1
    diagnostic._direction("command", np.array([-.0001, 0, 0]), 1, 4)
    assert diagnostic.counts["command_reversals_all"] == 2
    env = SimpleNamespace(target_color="red", color_body_ids={"red": 0},
                          sim=SimpleNamespace(data=SimpleNamespace(body_xpos=np.array([[.10, 0, 0]]))))
    obs = {"robot0_eef_pos": np.zeros(3), "robot0_gripper_qpos": np.zeros(2)}
    info = {"control_steps": 1, "hold_streak": 0, "height_ok": False, "recent_target_grasp": False,
            "lift_m": 0., "linear_speed_m_s": 0., "angular_speed_rad_s": 0.,
            "target_grasped": False, "wrong_color_grasp": False, "collisions": []}
    control = {"phase": 0, "requested_phase": 0, "close_steps": 0, "goal_xyz": np.zeros(3), "command_translation_m": np.zeros(3)}
    plateau = MotionDiagnostics(cfg)
    for tick in range(1, 7):
        plateau.update(env, obs, obs, control, info | {"control_steps": tick})
    assert plateau.counts["no_progress_steps"] == 1
    idle = MotionDiagnostics(cfg)
    idle.update(env, obs, obs, control | {"phase": 2, "close_steps": 8}, info)
    assert idle.counts["post_close_idle_steps"] == 1 and idle.counts["empty_gripper_wait_steps"] == 1
    hold = MotionDiagnostics(cfg)
    for tick in range(1, 41):
        hold.update(env, obs, obs, control | {"phase": 3, "close_steps": 20}, info | {"control_steps": tick, "hold_streak": tick, "height_ok": True, "recent_target_grasp": True})
    assert hold.counts["valid_hold_steps"] == 40 and hold.counts["post_close_idle_steps"] == 0 and hold.counts["no_progress_steps"] == 0
    result = {"passed": True, "real_reversal_counted": True, "phase_change_and_small_motion_excluded": True,
              "known_plateau_counted": True, "closed_empty_idle_counted": True, "required_40_hold_steps_not_idle": True}
    (EXPERIMENT / "environment" / "diagnostics_verification.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
