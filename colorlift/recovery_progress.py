"""Compact read-only snapshot of completed final evaluation cases."""
import json
from collections import defaultdict
from .config import ROOT


def main():
    root=ROOT / "experiments/colorlift_recovery_v1/evaluations/test"
    totals=defaultdict(lambda: [0, 0, 0])
    for path in root.glob("*/*/scene_*/*/metadata.json"):
        try: row=json.loads(path.read_text())
        except json.JSONDecodeError: continue
        key=("disturbed" if row["disturbed"] else "clean")+"/"+row["variant"]
        totals[key][0]+=1; totals[key][1]+=row["success"]; totals[key][2]+=row["success"] and not row["collision_steps"]
    print(json.dumps({key: {"finished": value[0], "successes": value[1], "collision_free_successes": value[2]} for key, value in sorted(totals.items())}))


if __name__ == "__main__": main()
