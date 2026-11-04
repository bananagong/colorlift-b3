"""Public progress without reading private layouts or interim confirmation outcomes."""
import json
from .spec import RUN,SEEDS,read,atomic,now


def last_json(path):
    if not path.exists():return {}
    for line in reversed(path.read_text().splitlines()):
        try:return json.loads(line)
        except json.JSONDecodeError:pass
    return {}


def snapshot():
    result={"updated_at_UTC":now(),"pipeline":read(RUN / "status.json") if (RUN / "status.json").exists() else {},
        "preparation_complete":(RUN / "preparation_completed.json").exists(),"cohort_sealed":(RUN / "sealing_completed.json").exists(),
        "all_models_locked":(RUN / "model_lock.json").exists(),"confirmation_complete":(RUN / "confirmation_completed.json").exists(),
        "analysis_complete":(RUN / "analysis_completed.json").exists(),"models":[],"confirmation_completed_cases":{},
        "interim_confirmation_outcomes_read":False}
    for seed in SEEDS:
        for group in ("B2","B3"):
            name=f"{group}_s{seed}";directory=RUN / "models" / name
            row={"model":name,"step":last_json(directory / "metrics.jsonl").get("step",0),
                "training_completed":(directory / "training_completed.json").exists(),"development_selected":(directory / "selection.json").exists()}
            result["models"].append(row)
    root=RUN / "evaluations/confirm_locked"
    if root.exists():
        for directory in sorted(root.iterdir()):
            if directory.is_dir():result["confirmation_completed_cases"][directory.name]=sum(1 for _ in directory.rglob("metadata.json"))
    result["confirmation_completed_total"]=sum(result["confirmation_completed_cases"].values())
    atomic(RUN / "progress.json",result);return result


if __name__=="__main__":print(json.dumps(snapshot(),indent=2),flush=True)
