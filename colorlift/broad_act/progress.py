"""Compact status from real checkpoints and physical selection markers."""
import json
from .spec import RUN


def main():
    status=json.loads((RUN / "status.json").read_text()) if (RUN / "status.json").exists() else {}
    models=[]
    for directory in sorted((RUN / "models").glob("*")):
        row={"model":directory.name}
        if (directory / "metrics.jsonl").exists():
            last=json.loads((directory / "metrics.jsonl").read_text().splitlines()[-1]);row.update(step=last["step"],objective=last["objective"])
        for filename in ("selection_safety.json","selection.json"):
            if (directory / filename).exists():
                value=json.loads((directory / filename).read_text());row.update(selected=value["chosen_checkpoint"],closed_loop=value["metrics"]);break
        row["updates_complete"]=(directory / "training_completed.json").exists();models.append(row)
    result={"pipeline":status,"models":models,"final_lock_exists":(RUN / "candidate_lock.json").exists(),"final_complete":(RUN / "final_completed.json").exists()}
    (RUN / "progress.json").write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":main()
