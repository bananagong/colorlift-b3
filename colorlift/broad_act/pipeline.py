"""Run B2--B4 after B1; lock choices before touching sealed final layouts."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from .spec import ROOT,RUN,DATA,SEEDS,sha,freeze
from .stage1 import train_runs
from .evaluate import select_model,rank,evaluate


def status(stage,**extra):
    (RUN / "status.json").write_text(json.dumps({"stage":stage,"updated_at_UTC":datetime.now(timezone.utc).isoformat(),**extra},indent=2))


def mean_rank(selections):
    return (all(r["metrics"]["wrong_colors"]==0 for r in selections),
            float(np.mean([r["metrics"]["balanced_success"] for r in selections])),
            float(np.mean([r["metrics"]["balanced_contact"] for r in selections])),
            -float(np.mean([r["metrics"]["mean_steps_all"] for r in selections])))


def choose_seed(selections):return max(selections,key=lambda r:rank(r["metrics"]))


def run_module(module,args=(),gpu=None):
    env=dict(os.environ,OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="1")
    if gpu is not None:env.update(CUDA_VISIBLE_DEVICES=str(gpu),MUJOCO_EGL_DEVICE_ID=str(gpu))
    subprocess.run([sys.executable,"-u","-m","colorlift.broad_act."+module,*map(str,args)],cwd=ROOT,env=env,check=True)


def main():
    freeze()
    while not (RUN / "B1_completed.json").exists():
        status("B1_training_or_closed_loop_selection",final_used=False)
        time.sleep(15)
    b1=json.loads((RUN / "B1_completed.json").read_text())["models"]
    from .audit_round import freeze_sources,exposure_audit
    freeze_sources()
    status("B2_balanced_curriculum_training",final_used=False)
    configurations=[{"name":f"B2_{mode}_s{seed}","seed":seed,"mode":mode} for mode in ("balanced","curriculum") for seed in SEEDS]
    train_runs(configurations)
    b2={mode:[select_model(f"B2_{mode}_s{seed}") for seed in SEEDS] for mode in ("balanced","curriculum")}
    selected_mode=max(b2,key=lambda mode:mean_rank(b2[mode]))
    (RUN / "B2_completed.json").write_text(json.dumps({"variants":b2,"selected_mode":selected_mode,"selection_uses_final":False},indent=2))
    # Confirm same examples, sources/colors/phases and update budget in the
    # curriculum comparison; order alone differs.
    for seed in SEEDS:
        a=np.load(RUN / f"plans/balanced_{seed}.npz")["ids"]
        b=np.load(RUN / f"plans/curriculum_{seed}.npz")["ids"]
        assert np.array_equal(np.sort(a.ravel()),np.sort(b.ravel()))
    exposure_audit()
    status("B3_gradient_calibration",final_used=False)
    run_module("calibrate",["--mode",selected_mode],gpu=1)
    calibration=json.loads((RUN / "gradient_calibration.json").read_text())
    lambdas=calibration["lambda_candidates"]
    tag=lambda value:format(value,"g").replace(".","p")
    configurations=[{"name":f"B3a_l{tag(weight)}_s{seed}","seed":seed,"mode":selected_mode,"relative":weight} for weight in lambdas for seed in SEEDS]
    status("B3a_lambda_sweep_training",lambdas=lambdas,final_used=False)
    train_runs(configurations)
    b3={str(weight):[select_model(f"B3a_l{tag(weight)}_s{seed}") for seed in SEEDS] for weight in lambdas}
    selected_lambda=max(lambdas,key=lambda weight:mean_rank(b3[str(weight)]))
    (RUN / "B3_completed.json").write_text(json.dumps({"lambda_results":b3,"selected_lambda":selected_lambda,"selected_mode":selected_mode,"feature_reinjection":False,"final_used":False},indent=2))
    status("B4_physical_safe_and_contact_recovery_collection",final_used=False)
    run_module("safety_collect",["--launch"])
    run_module("safety_data")
    configurations=[{"name":f"B4_s{seed}","seed":seed,"mode":selected_mode,"relative":selected_lambda,"pregrasp":calibration["pregrasp_lambda"],"b4":True} for seed in SEEDS]
    status("B4_training",final_used=False)
    train_runs(configurations)
    from .select_safety import select as select_safety
    b4=[select_safety(row["name"]) for row in configurations]
    (RUN / "B4_completed.json").write_text(json.dumps({"models":b4,"final_used":False},indent=2))
    winners={"B1":choose_seed(b1),"B2":choose_seed(b2[selected_mode]),"B3":choose_seed(b3[str(selected_lambda)]),"B4":choose_seed(b4)}
    # Separate learned/data effects from a new contact-retry control path.
    status("B4_runtime_causality_controls",final_used=False)
    controls={"B3_same_recovery_runtime":{"path":ROOT / winners["B3"]["policy_path"],"controller":"safety_recovery"},
              "B4_original_runtime":{"path":ROOT / winners["B4"]["policy_path"],"controller":"original"},
              "B4_safety_recovery_runtime":{"path":ROOT / winners["B4"]["policy_path"],"controller":"safety_recovery"}}
    evaluate(controls,"runtime_controls")
    final_models={"B0":{"path":str((RUN / "frozen/old_policy.pt").relative_to(ROOT)),"sha256":sha(RUN / "frozen/old_policy.pt"),"controller":"original"}}
    for group,row in winners.items():final_models[group]={"path":row["policy_path"],"sha256":row["sha256"],"controller":"safety_recovery" if group=="B4" else "original"}
    lock={"locked_at_UTC":datetime.now(timezone.utc).isoformat(),"models":final_models,"winner_details":winners,"selected_sampler":selected_mode,
          "selected_relative_lambda":selected_lambda,"selection_sources":"validation/development only","warm_start_history_caveat":True,
          "final_preseal_sha256":sha(DATA / "final_generation_preseal.json")}
    marker=RUN / "candidate_lock.json"
    if marker.exists():assert json.loads(marker.read_text())["models"]==final_models
    else:marker.write_text(json.dumps(lock,indent=2))
    status("sealed_final_one_pass_evaluation",models=list(final_models),planned_episodes=26880,final_used=True)
    metrics=evaluate({name:{"path":ROOT / value["path"],"controller":value["controller"]} for name,value in final_models.items()},"final_locked",kind="final")
    (RUN / "final_completed.json").write_text(json.dumps({"complete":True,"metrics":metrics,"final_was_not_used_for_selection":True},indent=2))
    run_module("results")
    status("completed",final_used_for_selection=False,report=str(RUN / "REPORT.zh-CN.md"))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--detach",action="store_true");a=p.parse_args()
    if a.detach:
        with (RUN / "pipeline.log").open("a") as log:
            child=subprocess.Popen([sys.executable,"-u","-m","colorlift.broad_act.pipeline"],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps({"pid":child.pid,"groups":["B0","B1","B2","B3","B4"],"physical_GPUs":[1,2]}))
    else:
        try:main()
        except BaseException as exc:
            status("failed",error=repr(exc));raise
