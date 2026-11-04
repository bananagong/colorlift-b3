"""Select B4 with its declared recovery runtime, and retain fixed-runtime controls."""
import json
import shutil
from .spec import ROOT,RUN,sha
from .evaluate import evaluate,rank


def select(name):
    directory=RUN / "models" / name;marker=directory / "selection_safety.json"
    if marker.exists():return json.loads(marker.read_text())
    models={name+f"_step{step}":{"path":directory / f"step_{step}.pt","controller":"safety_recovery"} for step in (1500,3000,4500,6000)}
    metrics=evaluate(models,"select_safety_"+name)
    losses={r["step"]:r["loss"] for r in [json.loads(s) for s in (directory / "validation.jsonl").read_text().splitlines()]}
    best=max(models,key=lambda key:rank(metrics[key],losses[int(key.rsplit("step",1)[1])]))
    shutil.copy2(models[best]["path"],directory / "policy.pt")
    result={"model":name,"chosen_checkpoint":best,"metrics":metrics[best],"all_checkpoints":metrics,
            "policy_path":str((directory / "policy.pt").relative_to(ROOT)),"sha256":sha(directory / "policy.pt"),
            "controller":"safety_recovery","selection_primary":"same physical development success","final_used":False}
    marker.write_text(json.dumps(result,indent=2));return result
