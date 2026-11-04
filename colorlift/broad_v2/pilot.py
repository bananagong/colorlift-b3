import json
from .spec import RUN, CATEGORIES
from .validity import make_env
from .rollout import episode


def main():
    examples = json.loads((RUN / "preflight/result.json").read_text())
    env = make_env(1)
    try:
        for category in CATEGORIES:
            layout = examples[category]["example"]["layout"]
            for actor in ("current_expert", "safe_expert"):
                row = episode(env, layout, "red", RUN / "preflight/rollouts" / category / actor, actor, save_data=True)
                print("PILOT", category, actor, row["success"], row["eligible"], row["control_steps"], row["rejection_reasons"], flush=True)
    finally: env.close()


if __name__ == "__main__": main()
