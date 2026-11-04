"""Exact paired-seed inference and crossed seed x layout confidence intervals."""
import argparse
from collections import Counter,defaultdict
from datetime import datetime
import itertools
from pathlib import Path
import numpy as np
from PIL import Image
from .spec import ROOT,RUN,PRIVATE,SEEDS,SUITES,ATTEMPTS,read,sha,atomic,now,verify


def exact_seed_test(differences):
    values=np.asarray(differences,dtype=np.int64)
    null=np.array([sum(s*d for s,d in zip(signs,values)) for signs in itertools.product((-1,1),repeat=len(values))],dtype=np.int64)
    observed=int(values.sum())
    p=min(1.,2*min(np.mean(null>=observed),np.mean(null<=observed)))
    return {"two_sided_p":float(p),"sum_count_difference":observed,"paired_seed_count":len(values),
            "per_seed_count_differences":values.tolist(),"enumerated_label_assignments":len(null),
            "best_possible_two_sided_p_without_ties":float(2/len(null)),
            "assumption":"paired result exchangeability/sign symmetry under the null"}


def holm(values):
    values=np.asarray(values,float);order=np.argsort(values,kind="stable");adjusted=np.zeros_like(values);previous=0.
    for index,which in enumerate(order):
        previous=max(previous,min(1.,(len(values)-index)*values[which]));adjusted[which]=previous
    return adjusted.tolist()


def crossed_intervals(cubes,counts,suite_indices,resamples=20000,seed=20261031):
    """cubes: metric -> [method, paired training seed, geometry] condition sums."""
    names=list(cubes);nseed=next(iter(cubes.values())).shape[1];rng=np.random.default_rng(seed)
    scopes={"overall":np.arange(len(counts)),**suite_indices}
    samples={name:{scope:[[],[]] for scope in scopes} for name in names}
    for offset in range(0,resamples,128):
        size=min(128,resamples-offset);s=rng.integers(0,nseed,(size,nseed))
        # A single common draw of each layout cluster is shared across all
        # selected seeds and both methods: the factors are crossed, not nested.
        g=np.concatenate([rng.choice(indices,(size,len(indices)),replace=True) for indices in suite_indices.values()],axis=1)
        bounds={};start=0
        for suite,indices in suite_indices.items():bounds[suite]=slice(start,start+len(indices));start+=len(indices)
        for name,cube in cubes.items():
            selected=[cube[method,s,:].sum(axis=1) for method in range(2)]
            for scope,indices in scopes.items():
                chosen=g if scope=="overall" else g[:,bounds[scope]]
                denominator=nseed*int(counts[indices].sum())
                for method in range(2):
                    value=np.take_along_axis(selected[method],chosen,axis=1).sum(1)/denominator
                    samples[name][scope][method].append(value)
    result={}
    for name,cube in cubes.items():
        result[name]={}
        for scope,indices in scopes.items():
            a,b=[np.concatenate(parts) for parts in samples[name][scope]];d=b-a
            point=[float(cube[m,:,indices].sum()/(nseed*counts[indices].sum())) for m in range(2)]
            result[name][scope]={"B2":point[0],"B3":point[1],"difference_B3_minus_B2":point[1]-point[0],
                "B2_95":np.quantile(a,[.025,.975]).tolist(),"B3_95":np.quantile(b,[.025,.975]).tolist(),
                "difference_95":np.quantile(d,[.025,.975]).tolist(),"degenerate_difference_interval":bool(np.ptp(d)==0)}
    return result


def summarize(rows):
    count=len(rows)
    mean=lambda key:float(np.mean([r[key] for r in rows if r.get(key) is not None])) if any(r.get(key) is not None for r in rows) else None
    return {"episodes":count,"geometries":len({r["geometry_id"] for r in rows}),
        "successes":sum(r["success"] for r in rows),"success_rate":sum(r["success"] for r in rows)/count,
        "contact_successes":sum(r["eligible"] for r in rows),"contact_rate":sum(r["eligible"] for r in rows)/count,
        "wrong_colors":sum(r["wrong_color_grasp"] for r in rows),"collision_episodes":sum(r["collision_steps"]>0 for r in rows),
        "mean_steps_all":mean("control_steps"),"mean_path_m":mean("physical_eef_path_m"),"mean_close_XY_error_m":mean("close_XY_error_m"),
        "mean_command_reversals":mean("command_reversals_within_phase"),"mean_motion_reversals":mean("motion_reversals_within_phase"),
        "mean_empty_gripper_wait_steps":mean("empty_gripper_wait_steps"),"mean_post_close_idle_steps":mean("post_close_idle_steps"),
        "mean_no_progress_steps":mean("no_progress_steps")}


def flatten(value):
    if isinstance(value,dict):return [x for key in sorted(value) for x in flatten(value[key])]
    if isinstance(value,list):return [x for child in value for x in flatten(child)]
    return [float(value)]


def analyze():
    if (RUN / "analysis_completed.json").exists():return read(RUN / "analysis_completed.json")
    protocol=verify(require_seal=True)
    assert read(RUN / "confirmation_completed.json")["total_episodes"]==86016
    lock=read(RUN / "model_lock.json");locked_at=datetime.fromisoformat(lock["locked_at_UTC"]).timestamp()
    seal=read(RUN / "seal.json")
    assert all(sha(PRIVATE / rel)==digest for rel,digest in seal["sha256"].items())
    assert datetime.fromisoformat(read(RUN / "training_started.json")["at_UTC"]) > datetime.fromisoformat(read(RUN / "sealing_completed.json")["completed_at_UTC"])
    manifest=read(PRIVATE / "manifest.json");layout_rows=manifest["layouts"]
    ids=[r["geometry_id"] for r in layout_rows];index={g:i for i,g in enumerate(ids)}
    counts=np.array([3 if r["split"]=="spatial_ood" else 9 for r in layout_rows])
    suites={suite:np.array([i for i,r in enumerate(layout_rows) if r["split"]==suite]) for suite in SUITES}
    assert len(ids)==len(set(ids))==768 and counts.sum()==ATTEMPTS
    from .evaluate import expected_cases
    expected=expected_cases("confirm")
    cubes={name:np.zeros((2,6,768),dtype=np.int64) for name in ("success","contact","steps")}
    models={};baseline={};input_audit={"case_comparisons":0,"RGB_hash_differences":0,"max_pixel_absolute_difference":0,
        "physical_field_max_absolute_difference":{},"examples":[],"within_palette_instruction_RGB_differences":0}
    fields=("initial_robot_state","physical_slot_positions","physical_slot_quaternions","actual_camera_positions","actual_light_diffuse")
    for name,config in lock["models"].items():
        assert sha(ROOT / config["path"])==config["sha256"]
        paths=list((RUN / "evaluations/confirm_locked" / name).rglob("metadata.json"))
        assert len(paths)==ATTEMPTS and min(p.stat().st_mtime for p in paths)>=locked_at
        rows=[read(path) for path in paths]
        assert {(r["geometry_id"],r["layout"]["color_assignment_index"],r["target_color"]) for r in rows}==expected
        assert {r["geometry_id"] for r in rows}==set(ids)
        row_summary={"group":config["group"],"training_seed":config["training_seed"],"cohort":config["cohort"],
            "overall":summarize(rows),"by_suite":{},"by_color":{},"videos":sum(1 for _ in (RUN / "evaluations/confirm_locked" / name).rglob("front.mp4"))}
        for suite in SUITES:row_summary["by_suite"][suite]=summarize([r for r in rows if r["layout"]["eval_suite"]==suite])
        for color in ("red","yellow","blue"):row_summary["by_color"][color]=summarize([r for r in rows if r["target_color"]==color])
        local_rgb={}
        for row in rows:
            key=(row["geometry_id"],row["layout"]["color_assignment_index"],row["target_color"])
            group_key=key[:2]
            if group_key in local_rgb and local_rgb[group_key]!=row["initial_rgb_sha256"]:input_audit["within_palette_instruction_RGB_differences"]+=1
            else:local_rgb[group_key]=row["initial_rgb_sha256"]
            if key not in baseline:
                baseline[key]={k:row[k] for k in fields}|{"rgb":row["initial_rgb_sha256"],"directory":row["directory"]}
            else:
                before=baseline[key];input_audit["case_comparisons"]+=1
                for field in fields:
                    a,b=flatten(before[field]),flatten(row[field]);assert len(a)==len(b)
                    change=max(abs(x-y) for x,y in zip(a,b))
                    input_audit["physical_field_max_absolute_difference"][field]=max(change,input_audit["physical_field_max_absolute_difference"].get(field,0.))
                if before["rgb"]!=row["initial_rgb_sha256"]:
                    input_audit["RGB_hash_differences"]+=1;maximum=0
                    for image in ("initial_front.png","initial_side.png"):
                        a=np.array(Image.open(RUN / before["directory"] / image),dtype=np.int16)
                        b=np.array(Image.open(RUN / row["directory"] / image),dtype=np.int16)
                        maximum=max(maximum,int(np.abs(a-b).max()))
                    input_audit["max_pixel_absolute_difference"]=max(input_audit["max_pixel_absolute_difference"],maximum)
                    if len(input_audit["examples"])<20:input_audit["examples"].append({"model":name,"case":list(key),"max_pixel_difference":maximum})
            if config["cohort"]=="new":
                method=0 if config["group"]=="B2" else 1;seed_index=SEEDS.index(config["training_seed"]);g=index[row["geometry_id"]]
                cubes["success"][method,seed_index,g]+=int(row["success"])
                cubes["contact"][method,seed_index,g]+=int(row["eligible"])
                cubes["steps"][method,seed_index,g]+=int(row["control_steps"])
        models[name]=row_summary
        print("AUDITED_MODEL",name,ATTEMPTS,flush=True)
    assert len(models)==16
    primary=exact_seed_test(cubes["success"][1].sum(1)-cubes["success"][0].sum(1))
    primary["per_seed_rate_differences"]=[v/ATTEMPTS for v in primary["per_seed_count_differences"]]
    intervals=crossed_intervals(cubes,counts,suites,protocol["primary_analysis"]["resamples"],protocol["analysis_rng_seed"])
    secondary=[]
    for suite,which in suites.items():
        for metric in ("success","contact"):
            test=exact_seed_test(cubes[metric][1,:,which].sum(0)-cubes[metric][0,:,which].sum(0))
            secondary.append({"suite":suite,"endpoint":metric,**test})
    adjusted=holm([r["two_sided_p"] for r in secondary])
    for row,value in zip(secondary,adjusted):row["Holm_adjusted_p"]=value
    gates={};failures=[]
    for group in ("B2","B3"):
        all_pass=True
        for seed in SEEDS:
            for suite in SUITES:
                row=models[f"{group}_s{seed}"]["by_suite"][suite]
                passed=row["success_rate"]>=.98 and row["contact_rate"]>=.95 and row["wrong_colors"]==0
                if not passed:failures.append({"group":group,"seed":seed,"suite":suite,"metrics":row})
                all_pass &= passed
        gates[group]=bool(all_pass)
    ci=intervals["success"]["overall"]["difference_95"];verdict="inconclusive"
    if primary["two_sided_p"]<.05:
        if ci[0]>0 and gates["B3"]:verdict="B3_superior_on_frozen_protocol"
        elif ci[1]<0 and gates["B2"]:verdict="B2_superior_on_frozen_protocol"
    result={"analyzed_at_UTC":now(),"verdict":verdict,"primary":primary,"crossed_intervals":intervals,"secondary_tests":secondary,
        "safety_gate":gates,"safety_gate_failures":failures,"models":models,"initial_input_audit":input_audit,
        "primary_training_seeds":list(SEEDS),"supplement_training_seeds":[43,44],"fresh_geometries":768,
        "total_confirm_episodes":86016,"primary_new_seed_episodes":64512,"legacy_supplement_episodes":21504,
        "old768_confirmation_episodes":0,"best_seed_not_selected_for_primary":True,"stage2_started":False,
        "limitations":["Six paired training seeds do not guarantee power or universal optimality.",
            "Fresh cohort retains the original mixture/Sobol/support restrictions, not unconditional IID.",
            "Crossed bootstrap intervals are approximate and may degenerate at ceiling.",
            "Inherited task weights can include historical spatial/relation exposure.",
            "Non-significance is not equivalence; no practical equivalence margin was introduced."]}
    atomic(RUN / "confirmatory_analysis.json",result)
    # Full cache hashing is repeated once after all new work, never per trial.
    for relative,meta in protocol["cache_fingerprints"].items():
        assert sha(__import__("colorlift.repro_confirm.spec",fromlist=["OLD"]).OLD / "cache" / relative)==meta["sha256"]
    atomic(RUN / "analysis_completed.json",{"completed_at_UTC":now(),"verdict":verdict,
        "analysis_sha256":sha(RUN / "confirmatory_analysis.json"),"data_unchanged_after_experiment":True})
    return result


def self_test():
    assert exact_seed_test([1]*6)["two_sided_p"]==.03125
    assert exact_seed_test([1]*3)["two_sided_p"]==.25
    assert exact_seed_test([0]*6)["two_sided_p"]==1
    assert exact_seed_test([1,2,3,4,5,6])["two_sided_p"]==exact_seed_test([9,18,27,36,45,54])["two_sided_p"]
    assert np.allclose(holm([.001,.01,.05]),[.003,.02,.05])
    counts=np.array([9,9,3,3,9,9]);which={"a":np.array([0,1]),"b":np.array([2,3]),"c":np.array([4,5])}
    values=np.ones((2,6,6),dtype=np.int64)
    result=crossed_intervals({"success":values},counts,which,resamples=256)
    assert result["success"]["overall"]["difference_95"]==[0.,0.]
    assert result["success"]["overall"]["degenerate_difference_interval"]
    # Numpy advanced-index shapes used for suite-wise six-seed tests.
    assert values[1,:,which["a"]].sum(0).shape==(6,)
    perfect=np.zeros((2,6,6),dtype=np.int64);perfect[1,:,:]=counts
    full=crossed_intervals({"success":perfect},counts,which,resamples=128)
    assert full["success"]["overall"]["difference_95"]==[1.,1.]
    print("Paired-seed exact tests, duplication invariance, Holm and crossed-bootstrap checks passed",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--self-test",action="store_true");a=p.parse_args()
    self_test() if a.self_test else analyze()
