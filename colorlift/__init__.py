"""Color-conditioned lifting with HSV observations and measured OSC primitives."""

from .env import ColorLift, make_env
from robosuite.environments.base import REGISTERED_ENVS

REGISTERED_ENVS.setdefault("colorlift", ColorLift)

__all__ = ["ColorLift", "make_env"]
