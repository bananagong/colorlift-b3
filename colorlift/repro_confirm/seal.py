"""Policy-blind Valid-only generation, historical exclusion, and pretraining seal."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import secrets
import sqlite3
import subprocess
import sys
import numpy as np
from .spec import ROOT, RUN, OLD, DATA, PRIVATE, read, sha, atomic, now, status, verify


def numeric_keys(xy):
    points = np.asarray(xy, dtype=float)
    assert points.shape == (3,2) and np.isfinite(points).all()
    keys = []
    for scale in (1000,100):
        integer = np.rint(points*scale).astype(np.int64)
        canonical = sorted(tuple(int(x) for x in row) for row in integer)
        keys.append(__import__("hashlib").sha256(json.dumps(canonical).encode()).hexdigest())
    return keys


def connect():
    db = sqlite3.connect(PRIVATE / "exclusion_registry.sqlite", timeout=60)
    db.execute("PRAGMA busy_timeout=60000")
    return db


def extract_layouts(value):
    if isinstance(value, dict):
        for key in ("xy", "physical_slot_positions", "initial_positions"):
            points = value.get(key)
            if isinstance(points, dict): points = list(points.values())
            if isinstance(points, list):
                try:
                    a = np.asarray(points, dtype=float)
                    if a.shape in ((3,2),(3,3)):
                        yield a[:,:2]
                except (TypeError,ValueError):
                    pass
        for child in value.values():
            if isinstance(child, (dict,list)):
                yield from extract_layouts(child)
    elif isinstance(value, list):
        for child in value:
            if isinstance(child, (dict,list)):
                yield from extract_layouts(child)


def seed_exclusions():
    if (RUN / "exclusions_completed.json").exists(): return
    PRIVATE.mkdir(parents=True, exist_ok=True); PRIVATE.chmod(0o700)
    db = connect()
    db.execute("CREATE TABLE IF NOT EXISTS fine(key TEXT PRIMARY KEY, owner TEXT, gid TEXT)")
    db.execute("CREATE TABLE IF NOT EXISTS coarse(key TEXT PRIMARY KEY, owner TEXT, gid TEXT)")
    paths = set()
    for filename in ("metadata.json","layout.json","manifest.json","training_manifest.json","pool_completed.json","test_scenarios.json"):
        paths.update(p for p in (ROOT / "experiments").rglob(filename) if RUN not in p.parents)
    paths.update((DATA / "preflight").glob("*.json"))
    count = 0
    for index,path in enumerate(sorted(paths)):
        for xy in extract_layouts(read(path)):
            fine, coarse = numeric_keys(xy)
            db.execute("INSERT OR IGNORE INTO fine VALUES(?,?,?)", (fine,"history","historical"))
            db.execute("INSERT OR IGNORE INTO coarse VALUES(?,?,?)", (coarse,"history","historical"))
            count += 1
        if index%5000 == 0:
            db.commit(); print("EXCLUSION_INPUTS",index,len(paths),count,flush=True)
    db.commit()
    summary = {"completed_at_UTC":now(),"historical_files_read":len(paths),"historical_coordinate_rows":count,
        "fine_keys":db.execute("SELECT COUNT(*) FROM fine").fetchone()[0],
        "coarse_keys":db.execute("SELECT COUNT(*) FROM coarse").fetchone()[0],
        "old768_role":"exclusion/history only, never confirmation outcomes",
        "numeric_canonicalization":"unordered integer-mm and integer-cm coordinates; signed zero removed"}
    db.close(); atomic(RUN / "exclusions_completed.json",summary)


def reserve(db, layout, owner):
    fine, coarse = numeric_keys(layout["xy"]); gid = layout["geometry_id"]
    db.execute("BEGIN IMMEDIATE")
    try:
        a = db.execute("SELECT owner,gid FROM fine WHERE key=?",(fine,)).fetchone()
        b = db.execute("SELECT owner,gid FROM coarse WHERE key=?",(coarse,)).fetchone()
        if a or b:
            allowed = a == b == (owner,gid)
            db.rollback(); return allowed
        db.execute("INSERT INTO fine VALUES(?,?,?)",(fine,owner,gid))
        db.execute("INSERT INTO coarse VALUES(?,?,?)",(coarse,owner,gid))
        db.commit(); return True
    except BaseException:
        db.rollback(); raise


def jobs():
    result = []
    for suite in ("broad_iid","spatial_ood"):
        for category,count in (("uniform",90),("sobol",102),("near",38),("edge_adverse",26)):
            result.append({"suite":suite,"category":category,"count":count,"subtype":None,"label":suite+"_"+category})
    for subtype in ("near_gap","aligned"):
        result.append({"suite":"relation_ood","category":"uniform","count":128,"subtype":subtype,"label":"relation_ood_"+subtype})
    return result


def worker(index,gpu):
    verify()
    assert not (RUN / "sealing_completed.json").exists()
    assert not (RUN / "training_started.json").exists()
    job = jobs()[index]; owner = "new_"+job["label"]
    directory = PRIVATE / "layouts" / job["label"]; directory.mkdir(parents=True,exist_ok=True)
    if (directory / "completed.json").exists(): return
    streams = read(PRIVATE / "streams.json")
    from ..broad_v2.conditional_proposals import ConditionalProposals
    from ..broad_v2.validity import make_env, Validator
    from ..vision import observation_images
    from PIL import Image
    stream = ConditionalProposals(streams[job["label"]],job["category"],job["suite"],job["subtype"])
    env = make_env(gpu); validator = Validator(env); db = connect()
    accepted = []; rejects = Counter()
    try:
        with (directory / "proposal_log.jsonl").open("w") as log:
            for _ in range(50000):
                layout,reason = stream.next()
                case = directory / layout["geometry_id"]
                admission = case / "admission.json"
                if admission.exists():
                    row = read(admission)
                    if row["layout"]["proposal_index"]!=layout["proposal_index"] or row["layout"]["stream_seed"]!=layout["stream_seed"]:
                        rejects["same_geometry_repeated_with_new_yaw_or_colors"]+=1
                        continue
                    accepted.append(row)
                else:
                    validity = None
                    if reason is None:
                        validity = validator.check(layout)
                        if not validity["valid"]: reason = validity["reason"]
                    if reason is None and not reserve(db,layout,owner): reason = "historical_or_new_geometry_group_reuse"
                    if reason:
                        rejects[reason] += 1
                        log.write(json.dumps({"index":stream.index-1,"reason":reason,"geometry_id":layout["geometry_id"]})+"\n")
                        continue
                    case.mkdir(exist_ok=True)
                    atomic(case / "layout.json",layout); atomic(case / "validity.json",validity)
                    images = observation_images(env._get_observations())
                    Image.fromarray(np.concatenate(list(images),axis=1)).save(case / "initial_two_views.png")
                    row = {"geometry_id":layout["geometry_id"],"position_group_id":numeric_keys(layout["xy"])[1],
                        "layout":layout,"split":job["suite"],"directory":str(case.relative_to(PRIVATE)),
                        "policy_evaluations":0,"expert_rollouts":0,"valid":True}
                    atomic(admission,row); accepted.append(row)
                log.write(json.dumps({"index":stream.index-1,"accepted":True,"geometry_id":layout["geometry_id"]})+"\n")
                print("SEALED_LAYOUT_PROGRESS",job["label"],len(accepted),job["count"],stream.index,flush=True)
                if len(accepted) == job["count"]: break
        assert len(accepted) == job["count"],(job,len(accepted),dict(rejects))
        atomic(directory / "completed.json",{"job":job,"accepted":accepted,"proposal_count":stream.index,
            "rejections":dict(rejects),"policy_evaluations":0,"expert_rollouts":0})
    finally:
        db.close(); env.close()


def launch():
    verify()
    if (RUN / "sealing_completed.json").exists(): return
    assert not (RUN / "training_started.json").exists()
    status("generating_and_sealing_fresh_layouts",training_updates=0,confirmation_policy_evaluations=0)
    seed_exclusions()
    stream_path = PRIVATE / "streams.json"
    if not stream_path.exists():
        seeds = secrets.SystemRandom().sample(range(20000,30000),len(jobs()))
        atomic(stream_path,{job["label"]:seed for job,seed in zip(jobs(),seeds)})
    # The private stream commitment is fixed before the first proposal.
    stream_commit = RUN / "private_stream_commitment.json"
    if stream_commit.exists(): assert read(stream_commit)["sha256"] == sha(stream_path)
    else: atomic(stream_commit,{"committed_at_UTC":now(),"sha256":sha(stream_path),"coordinates_public":False})
    for start in range(0,len(jobs()),4):
        processes = []; handles = []
        for index in range(start,min(start+4,len(jobs()))):
            job = jobs()[index]
            if (PRIVATE / "layouts" / job["label"] / "completed.json").exists(): continue
            gpu = (1,2)[index%2]
            env = dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),MUJOCO_EGL_DEVICE_ID=str(gpu),OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
            handle = (RUN / ("seal_"+job["label"]+".log")).open("a"); handles.append(handle)
            processes.append(subprocess.Popen([sys.executable,"-u","-m","colorlift.repro_confirm.seal","--index",str(index),"--gpu",str(gpu)],cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT))
        codes = [p.wait() for p in processes]
        for handle in handles: handle.close()
        assert all(c == 0 for c in codes),codes
    rows = []; statistics = []
    for job in jobs():
        value = read(PRIVATE / "layouts" / job["label"] / "completed.json")
        rows.extend(value["accepted"]); statistics.append({k:v for k,v in value.items() if k != "accepted"})
    assert len(rows) == len({r["geometry_id"] for r in rows}) == len({r["position_group_id"] for r in rows}) == 768
    assert Counter(r["split"] for r in rows) == {"broad_iid":256,"spatial_ood":256,"relation_ood":256}
    assert all(r["policy_evaluations"] == r["expert_rollouts"] == 0 for r in rows)
    historical=read(DATA / "public_final_manifest.json")
    old_ids={r["geometry_id"] for r in historical["geometry_groups"]}
    overlap=old_ids & {r["geometry_id"] for r in rows}
    assert len(old_ids)==768 and not overlap
    db=connect()
    assert db.execute("SELECT COUNT(*) FROM fine WHERE owner LIKE 'new_%'").fetchone()[0]==768
    assert db.execute("SELECT COUNT(*) FROM coarse WHERE owner LIKE 'new_%'").fetchone()[0]==768
    db.close()
    atomic(RUN / "generation_audit.json",{"at_UTC":now(),"new_geometries":768,"old768_exact_geometry_overlap":0,
        "historical_numeric_exclusion_registry_used":True,"new_fine_and_coarse_keys_unique":True,
        "expert_or_policy_success_filtering":False,"policy_evaluations_before_seal":0})
    atomic(PRIVATE / "manifest.json",{"layouts":rows,"independent_new_geometries":768,
        "policy_evaluations":0,"expert_rollouts":0,"original768_reused":False})
    atomic(RUN / "generation_statistics.json",statistics)
    immutable = [stream_path,PRIVATE / "manifest.json"]+sorted((PRIVATE / "layouts").rglob("*"))
    hashes = {str(path.relative_to(PRIVATE)):sha(path) for path in immutable if path.is_file()}
    seal = {"sealed_at_UTC":now(),"protocol_sha256":sha(RUN / "protocol.json"),"sha256":hashes,
        "counts":{"broad_iid":256,"spatial_ood":256,"relation_ood":256},"independent_new_geometries":768,
        "policy_evaluations":0,"expert_rollouts":0,"old768_role":"historical/excluded only"}
    atomic(RUN / "seal.json",seal)
    atomic(RUN / "public_layout_manifest.json",{"independent_new_geometries":768,"counts":seal["counts"],
        "geometry_groups":[{"geometry_id":r["geometry_id"],"position_group_id":r["position_group_id"],"split":r["split"]} for r in rows],
        "coordinates_and_images_private":True,"old768_reused":False,"seal_sha256":sha(RUN / "seal.json")})
    atomic(RUN / "sealing_completed.json",{"completed_at_UTC":now(),"independent_new_geometries":768,
        "seal_sha256":sha(RUN / "seal.json"),"training_updates":0,"policy_evaluations":0})
    print("FRESH_COHORT_SEALED",768,flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--launch",action="store_true"); p.add_argument("--index",type=int); p.add_argument("--gpu",type=int)
    a = p.parse_args()
    launch() if a.launch else worker(a.index,a.gpu)
