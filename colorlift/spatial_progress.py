"""Compact, read-only progress for large gates and subsequent spatial experiments."""
from collections import defaultdict
import argparse
import json
from .spatial_gate import RUN


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--brief", action="store_true"); args = parser.parse_args()
    if args.brief:
        groups = defaultdict(lambda: [0, 0, 0, 0])
        for path in (RUN / "evaluations/test").glob("*/*/*/*/metadata.json"):
            try: row = json.loads(path.read_text())
            except json.JSONDecodeError: continue
            values = groups[row["layout"]["level"], row["actor"]]
            values[0] += 1; values[1] += row["success"]; values[2] += row["eligible"]; values[3] += row["wrong_color_grasp"]
        print(json.dumps({"models_complete": len(list((RUN / "models").glob("*/completed.json"))),
                          "test_status": {level+"/"+model: values for (level, model), values in sorted(groups.items())},
                          "columns": ["trials", "successes", "contact_qualified", "wrong_color"],
                          "counterfactual_trials": len(list((RUN / "evaluations/counterfactual").glob("*/*/*/*/*/metadata.json"))),
                          "computation_complete": (RUN / "computation_completed.json").exists()}, indent=2)); return
    colors = defaultdict(list)
    for path in (RUN / "gate/episodes").glob("scene_*/*/metadata.json"):
        try:
            value = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        colors[value["target_color"]].append(value)
    gate = []
    for color, rows in sorted(colors.items()):
        gate.append({"color": color, "completed": len(rows), "successes": sum(r["success"] for r in rows),
                     "contact_qualified": sum(r["contact_qualified_collision_free_success"] for r in rows),
                     "wrong_color": sum(r["wrong_color_grasp"] for r in rows)})
    status = {"gate": gate, "gate_complete": (RUN / "gate/completed.json").exists()}
    if (RUN / "gate/metrics.json").exists():
        status["gate_passed"] = json.loads((RUN / "gate/metrics.json").read_text())["gate_passed"]
    status["candidate_episode_count"] = len(list((RUN / "candidates").glob("*/*/metadata.json")))
    layouts = defaultdict(lambda: [0, 0])
    for path in (RUN / "candidates").glob("*/layout.json"):
        try: row = json.loads(path.read_text())
        except json.JSONDecodeError: continue
        key = row["split"]+"_"+row["level"]
        layouts[key][0] += 1; layouts[key][1] += row["eligible"]
    status["candidate_layouts_completed_and_all9_eligible"] = dict(layouts)
    status["training_runs_complete"] = len(list((RUN / "models").glob("*/completed.json")))
    status["training_steps"] = {}
    for path in (RUN / "models").glob("*/metrics.jsonl"):
        try: status["training_steps"][path.parent.name] = json.loads(path.read_text().splitlines()[-1])["step"]
        except (json.JSONDecodeError, IndexError): pass
    status["test_episode_count"] = len(list((RUN / "evaluations/test").glob("*/*/*/*/metadata.json")))
    status["counterfactual_episode_count"] = len(list((RUN / "evaluations/counterfactual").glob("*/*/*/*/*/metadata.json")))
    print(json.dumps(status, indent=2))


if __name__ == "__main__": main()
