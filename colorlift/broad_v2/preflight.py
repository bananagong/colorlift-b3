import argparse
from collections import Counter
import json
import numpy as np
from PIL import Image
from .spec import RUN, CATEGORIES
from .layouts import Proposals
from .validity import make_env, Validator
from ..vision import observation_images


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--gpu", type=int, default=1); args = parser.parse_args()
    env = make_env(args.gpu); validator = Validator(env)
    out = RUN / "preflight"; out.mkdir(parents=True, exist_ok=True)
    report = {}
    try:
        for index, category in enumerate(CATEGORIES):
            stream = Proposals(3000+index, category)
            rejects = Counter(); example = None
            for attempt in range(200):
                layout, why = stream.next()
                if why: rejects[why] += 1; continue
                result = validator.check(layout)
                if not result["valid"]:
                    rejects[result["reason"]] += 1
                    if attempt < 50: print("REJECT", category, attempt, json.dumps(result), flush=True)
                    continue
                example = {"layout": layout, "validation": result}; break
            if example is None: raise RuntimeError((category, dict(rejects)))
            Image.fromarray(np.concatenate(list(observation_images(env._get_observations())), axis=1)).save(out / f"{category}.png")
            report[category] = {"attempts": attempt+1, "rejects": dict(rejects), "example": example}
            print("VALID", category, attempt+1, flush=True)
        (out / "result.json").write_text(json.dumps(report, indent=2))
    finally: env.close()


if __name__ == "__main__": main()
