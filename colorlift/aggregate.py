"""Flatten all ColorLift evaluation runs and their hyperparameters into one tuning CSV."""
import argparse
import csv
import json
from pathlib import Path
from .config import EXPERIMENT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=EXPERIMENT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.root.rglob("metrics.json")):
        result = json.loads(path.read_text())
        if "variant" not in result or "trials" not in result:
            continue
        row = {key: value for key, value in result.items() if not isinstance(value, (dict, list))}
        row["result_path"] = str(path)
        for prefix, key in (("task__", "task_config"), ("train__", "training_config")):
            for name, value in (result.get(key) or {}).items():
                if not isinstance(value, (dict, list)):
                    row[prefix + name] = value
        rows.append(row)
    if not rows:
        raise RuntimeError("No ColorLift evaluation metrics found")
    output = args.output or args.root / "tuning_all_runs.csv"
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Aggregated {len(rows)} evaluations: {output}")


if __name__ == "__main__":
    main()
