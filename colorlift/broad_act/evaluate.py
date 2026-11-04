"""Closed-loop checkpoint selection and one-pass sealed final evaluation."""
import argparse
from collections import defaultdict
import json
import os
import shutil
import subprocess
import sys
import numpy as np
from .spec import ROOT, DATA, RUN, INITIAL, COLORS, sha, freeze
from ..broad_v2.layouts import recolor


def layouts(kind):
    if kind == "selection":
        manifest = json.loads((DATA / "training_manifest.json").read_text())
        return [row["layout"] | {"eval_suite":"validation"} for row in manifest["layouts"] if row["split"] == "val"]
    assert (RUN / "candidate_lock.json").exists(), "Lock all choices before final coordinates are read"
    manifest = json.loads((DATA / "final_sealed/manifest.json").read_text())
    return [row["layout"] | {"eval_suite":row["split"]} for row in manifest["layouts"]]


def score(rows):
    categories = defaultdict(list)
    for row in rows: categories[row["layout"]["category"]].append(row)
    balanced = lambda key: float(np.mean([np.mean([r[key] for r in selected]) for selected in categories.values()]))
    return {"episodes":len(rows),"independent_geometries":len({r["geometry_id"] for r in rows}),
            "successes":sum(r["success"] for r in rows),"success_rate":float(np.mean([r["success"] for r in rows])),
            "balanced_success":balanced("success"),"contact_successes":sum(r["eligible"] for r in rows),"balanced_contact":balanced("eligible"),
            "wrong_colors":sum(r["wrong_color_grasp"] for r in rows),"collision_episodes":sum(r["collision_steps"]>0 for r in rows),
            "mean_steps_all":float(np.mean([r["control_steps"] for r in rows])),
            "mean_success_steps":float(np.mean([r["control_steps"] for r in rows if r["success"]])) if any(r["success"] for r in rows) else None,
            "mean_close_XY_error_m":float(np.mean([r["close_XY_error_m"] for r in rows if r["close_XY_error_m"] is not None])) if any(r["close_XY_error_m"] is not None for r in rows) else None}


def rank(metrics, val_loss=float("inf")):
    return (metrics["wrong_colors"]==0,metrics["balanced_success"],metrics["balanced_contact"],-metrics["mean_steps_all"],-val_loss)


def worker(plan_path, shard, gpu):
    import torch
    from .model import load_policy
    from .runtime import make_env, episode
    torch.set_num_threads(1)
    plan = json.loads(plan_path.read_text()); kind = plan["kind"]
    scenes = layouts(kind)
    env = make_env(gpu)
    try:
        for name, config in plan["models"].items():
            checkpoint_path = ROOT / config["path"]
            assert sha(checkpoint_path) == config["sha256"]
            model, checkpoint = load_policy(checkpoint_path)
            for index, base in enumerate(scenes):
                if index%8 != shard: continue
                permutations = range(3) if kind == "final" else range(1)
                for permutation in permutations:
                    layout = recolor(base,permutation)
                    colors = [layout["colors_by_slot"][base["spatial_target_slot"]]] if base.get("eval_suite")=="spatial_ood" else COLORS
                    for color in colors:
                        directory = RUN / "evaluations" / plan["tag"] / name / base["eval_suite"] / base["geometry_id"] / f"p{permutation}" / color
                        row = episode(env,layout,color,directory,name,model,checkpoint,config.get("controller","original"),representative=index<4 and color=="red")
                        print("CLOSED_LOOP",plan["tag"],name,index,permutation,color,row["success"],row["eligible"],row["control_steps"],flush=True)
            del model; torch.cuda.empty_cache()
    finally: env.close()


def evaluate(models, tag, kind="selection"):
    freeze()
    out = RUN / "evaluation_plans"; out.mkdir(exist_ok=True)
    entries = {}
    for name, value in models.items():
        path = value["path"] if isinstance(value,dict) else value
        path = __import__("pathlib").Path(path)
        entries[name] = {"path":str(path.relative_to(ROOT)),"sha256":sha(path),"controller":value.get("controller","original") if isinstance(value,dict) else "original"}
    if kind == "final":
        lock = json.loads((RUN / "candidate_lock.json").read_text())
        assert {k:v["sha256"] for k,v in entries.items()} == {k:v["sha256"] for k,v in lock["models"].items()}
        seal = json.loads((DATA / "final_generation_preseal.json").read_text())
        assert all(sha(DATA / "final_sealed" / rel)==digest for rel,digest in seal["sha256"].items())
    plan = {"models":entries,"tag":tag,"kind":kind,"selection_uses_final":False}
    plan_path = out / (tag+".json")
    if plan_path.exists(): assert json.loads(plan_path.read_text()) == plan
    else: plan_path.write_text(json.dumps(plan,indent=2))
    result_path = RUN / "evaluations" / tag / "metrics.json"
    if result_path.exists(): return json.loads(result_path.read_text())
    workers = []; handles = []
    for shard in range(8):
        gpu = (1,2)[shard%2]
        env = dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),MUJOCO_EGL_DEVICE_ID=str(gpu),OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
        handle = (RUN / f"eval_{tag}_{shard}.log").open("a"); handles.append(handle)
        workers.append(subprocess.Popen([sys.executable,"-u","-m","colorlift.broad_act.evaluate","--plan",str(plan_path),"--shard",str(shard),"--gpu",str(gpu)],cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT))
    codes = [p.wait() for p in workers]
    for handle in handles: handle.close()
    assert codes == [0]*8,codes
    metrics = {}
    for name in entries:
        rows = [json.loads(p.read_text()) for p in (RUN / "evaluations" / tag / name).rglob("metadata.json")]
        expected = 144 if kind=="selection" else 5376
        assert len(rows) == expected,(name,len(rows),expected)
        metrics[name] = score(rows)
    result_path.write_text(json.dumps(metrics,indent=2))
    return metrics


def select_model(name):
    directory = RUN / "models" / name
    marker = directory / "selection.json"
    if marker.exists(): return json.loads(marker.read_text())
    train = json.loads((directory / "training_completed.json").read_text())
    models = {name+f"_step{step}":directory / f"step_{step}.pt" for step in (1500,3000,4500,6000)}
    metrics = evaluate(models,"select_"+name)
    loss_by_step = {r["step"]:r["loss"] for r in [json.loads(s) for s in (directory / "validation.jsonl").read_text().splitlines()]}
    best = max(models,key=lambda key:rank(metrics[key],loss_by_step[int(key.rsplit("step",1)[1])]))
    shutil.copy2(models[best],directory / "policy.pt")
    result = {"model":name,"chosen_checkpoint":best,"metrics":metrics[best],"all_checkpoints":metrics,
              "policy_path":str((directory / "policy.pt").relative_to(ROOT)),"sha256":sha(directory / "policy.pt"),
              "selection_primary":"physical closed-loop category-balanced success","final_used":False}
    marker.write_text(json.dumps(result,indent=2))
    return result


if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("--plan");p.add_argument("--shard",type=int);p.add_argument("--gpu",type=int);p.add_argument("--baseline",action="store_true")
    args=p.parse_args()
    if args.baseline:
        freeze(); directory=RUN / "frozen";directory.mkdir(exist_ok=True)
        target=directory / "old_policy.pt"
        if not target.exists(): shutil.copy2(INITIAL,target)
        print(json.dumps(evaluate({"B0":target},"B0_initial"),indent=2),flush=True)
    else: worker(__import__("pathlib").Path(args.plan),args.shard,args.gpu)
