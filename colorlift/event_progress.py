"""Read-only snapshot; does not alter trial selection or evaluation settings."""
from collections import defaultdict
import json
from .event_efficiency import RUN


def main():
    groups = defaultdict(list)
    for path in (RUN / "test").glob("*/*/scene_*/*/metadata.json"):
        row = json.loads(path.read_text())
        groups[row["actor"], row["scheme"]].append(row)
    rows = []
    for (actor, scheme), episodes in sorted(groups.items()):
        passed = [e for e in episodes if e["success"]]
        rows.append({"actor": actor, "scheme": scheme, "completed": len(episodes), "successes": len(passed),
                     "contact_qualified": sum(e["contact_qualified_collision_free_success"] for e in episodes),
                     "mean_success_steps": round(sum(e["control_steps"] for e in passed)/len(passed), 3) if passed else None})
    candidates = list((RUN / "candidates").glob("*/metadata.json"))
    replays = [json.loads(p.read_text()) for p in (RUN / "replays").glob("*/verification.json")]
    print(json.dumps({"formal": rows, "total_formal": sum(row["completed"] for row in rows), "candidates": len(candidates),
                      "replays_completed": len(replays), "admitted": sum(r["admitted"] for r in replays),
                      "completed": (RUN / "completed.json").exists()}, indent=2))


if __name__ == "__main__":
    main()
