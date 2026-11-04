"""Shared, explicit task, controller, and diagnostic thresholds."""
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import os

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = Path(os.environ.get("COLORLIFT_RUN_DIR", ROOT / "experiments" / "colorlift_v1")).expanduser().resolve()
COLORS = ("red", "yellow", "blue")
CAMERAS = ("color_front", "color_side")
PHASES = ("approach", "align", "close", "lift_hold")


@dataclass
class Config:
    target_color: str = "red"
    hz: int = 20
    horizon: int = 180
    cube_size: float = 0.04
    cube_mass: float = 0.064
    table_height: float = 0.8
    xy_limit: float = 0.11
    min_center_distance: float = 0.095
    yaw_limit: float = 0.20
    lift_threshold: float = 0.04
    hold_seconds: float = 2.0
    max_linear_speed: float = 0.025
    max_angular_speed: float = 0.4
    max_hold_drift: float = 0.005
    contact_grace_steps: int = 2
    wrong_grasp_confirm_steps: int = 3
    collision_force_n: float = 5.0
    object_collision_force_n: float = 1.0
    motion_epsilon: float = 0.0005
    reversal_angle_deg: float = 100.0
    reversal_max_gap_steps: int = 3
    progress_window: int = 5
    progress_epsilon: float = 0.001
    progress_distance_tolerance: float = 0.012
    idle_motion_threshold: float = 0.0005
    minimum_close_steps: int = 8
    phase_confirm_queries: int = 2
    auto_advance_primitives: bool = True
    minimum_reach_steps: int = 5
    approach_height: float = 0.945
    grasp_height: float = 0.818
    lift_height: float = 0.94
    lift_goal_min_offset: float = 0.12
    lift_goal_max_offset: float = 0.22
    max_step_approach: float = 0.045
    max_step_align: float = 0.035
    max_step_lift: float = 0.04
    position_deadband: float = 0.001
    target_smoothing: float = 0.65
    goal_reached_tolerance: float = 0.006
    query_every: int = 3
    image_size: int = 128
    osc_kp: float = 150.0
    osc_damping_ratio: float = 1.0
    empty_grasp_abort_steps: int = 12
    empty_gripper_gap: float = 0.006

    def __post_init__(self):
        if self.target_color not in COLORS:
            raise ValueError("target_color must be red, yellow, or blue")
        if self.hz <= 0 or self.hold_seconds <= 0 or self.lift_threshold <= 0:
            raise ValueError("Control frequency, hold duration and lift threshold must be positive")

    @property
    def hold_steps(self):
        import math
        return int(math.ceil(self.hold_seconds * self.hz))

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path=None):
        return cls(**json.loads(Path(path).read_text())) if path else cls()
