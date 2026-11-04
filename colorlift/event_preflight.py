"""One separate, actual-physics round trip of the dataset admission workflow."""
import json
from .conditional import configuration
from .env import make_env
from .event_efficiency import RUN, run_episode
from . import event_replay


def main():
    directory = RUN / "validation/candidates/train_83999_red"
    env = make_env(1, configuration("red")[0])
    try:
        candidate = run_episode(env, 83999, directory, "expert", "efficiency", save_data=True)
        assert candidate["contact_qualified_collision_free_success"]
        event_replay.RUN = RUN / "validation"
        result = event_replay.replay(env, directory)
        assert result["admitted"], result
        assert result["maximum_state_error"] <= 2e-5
        print(json.dumps(result, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
