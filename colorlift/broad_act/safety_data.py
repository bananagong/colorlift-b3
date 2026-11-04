"""Reuse the same geometry IDs; add only real safe/recovery states to B4."""
from collections import defaultdict,Counter
import json
import numpy as np
from .spec import RUN,COLORS,STEPS,BATCH,PHASE_WEIGHTS
from .data import SHAPES,Dataset,spatial


def prepare():
    destination=RUN / "cache_b4";destination.mkdir(exist_ok=True)
    if (destination / "completed.json").exists():return
    safety=json.loads((RUN / "safety_manifest.json").read_text())
    entries=[r for r in safety["episodes"] if r["split"]=="train"]
    base={key:np.load(RUN / "cache" / f"train_{key}.npy",mmap_mode="r") for key in SHAPES}
    count=len(base["states"])+sum(r["steps"] for r in entries)
    arrays={key:np.lib.format.open_memmap(destination / f"train_{key}.npy",mode="w+",dtype=dtype,shape=(count,)+shape) for key,(shape,dtype) in SHAPES.items()}
    nbase=len(base["states"])
    for key in arrays:arrays[key][:nbase]=base[key]
    origin=json.loads((RUN / "data_manifest.json").read_text())
    geo_ids={gid:i for i,gid in enumerate(sorted({r["geometry_id"] for r in origin["rows"] if r["split"]=="train"}))}
    offset=nbase
    for index,row in enumerate(entries):
        directory=RUN / row["directory"];meta=json.loads((directory / "metadata.json").read_text())
        assert meta["eligible"] and row["geometry_id"] in geo_ids
        with np.load(directory / "trajectory.npz",allow_pickle=False) as item:
            n=len(item["states"]);block=slice(offset,offset+n);phase=item["states"][:,12:16].argmax(1)
            valid=item["supervised"].astype(bool);actions=item["actions"].copy();actions[phase==0,2]=.865
            end=n-1;ends=np.empty(n,np.int64)
            for t in range(n-1,-1,-1):
                if not valid[t]:end=t-1
                ends[t]=max(t,end)+offset
            cell,difficulty_bin,difficulty=spatial(meta)
            values={"states":item["states"][:,:16],"actions":actions,"images":item["images"],
                    "classes":np.where(item["masks"].max(-1)>0,item["masks"].argmax(-1)+1,0),
                    "colors":COLORS.index(row["target_color"]),"phases":phase,"source":3 if row["regime"]=="safe" else 4,
                    "category":("uniform","sobol","near","edge_adverse").index(row["category"]),"cell":cell,"difficulty_bin":difficulty_bin,
                    "difficulty":difficulty+.3*item["recovery"],"geometry":geo_ids[row["geometry_id"]],"valid":valid,"aux_valid":valid,
                    "ends":ends,"relative":item["relative"],"pre_relative":item["pre_relative"],"pre_rotation":item["pre_rotation"][:,:,:2].reshape(n,6),"recovery":item["recovery"]}
            for key in arrays:arrays[key][block]=values[key]
        offset+=n
        if (index+1)%72==0:print("B4_CACHE",index+1,len(entries),offset,flush=True)
    for array in arrays.values():array.flush()
    summary={"frames":count,"base_frames":nbase,"additional_episodes":len(entries),"by_regime":dict(Counter(r["regime"] for r in entries)),
             "additional_independent_layouts":0,"existing_train_geometry_groups_reused":len({r["geometry_id"] for r in entries}),
             "injected_frames_not_supervised":True,"validation_demonstrations_not_imported":True}
    (destination / "completed.json").write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)


class SafetyDataset(Dataset):
    def __init__(self,split,stats):
        assert split=="train"
        self.stats=stats;self.split=split
        self.arrays={key:np.load(RUN / "cache_b4" / f"train_{key}.npy",mmap_mode="r") for key in SHAPES}
        a=self.arrays;self.pools={};self.balanced={}
        for source in range(5):
            for color in range(3):
                for phase in range(4):
                    ids=np.flatnonzero(a["valid"]&(a["source"]==source)&(a["colors"]==color)&(a["phases"]==phase))
                    assert len(ids),(source,color,phase)
                    self.pools[source,color,phase]=ids
                    if source in (0,3,4):
                        bins=defaultdict(dict)
                        for category,cell,difficulty,geometry in np.unique(np.c_[a["category"][ids],a["cell"][ids],a["difficulty_bin"][ids],a["geometry"][ids]],axis=0):
                            selected=ids[(a["category"][ids]==category)&(a["cell"][ids]==cell)&(a["difficulty_bin"][ids]==difficulty)&(a["geometry"][ids]==geometry)]
                            bins[int(category),int(cell),int(difficulty)][int(geometry)]=selected
                        categories=defaultdict(list)
                        for key,value in bins.items():categories[key[0]].append(value)
                        self.balanced[source,color,phase]=dict(categories)

    def plan(self,seed,mode):
        directory=RUN / "plans";directory.mkdir(exist_ok=True);path=directory / f"B4_{mode}_{seed}.npz"
        if path.exists():return np.load(path)["ids"]
        rng=np.random.default_rng(seed);total=STEPS*BATCH
        source=np.concatenate([np.full(int(total*w),i,np.uint8) for i,w in enumerate((.6,.1,.1,.1,.1))]);assert len(source)==total;rng.shuffle(source)
        colors=np.arange(total)%3;rng.shuffle(colors);phases=rng.choice(4,total,p=PHASE_WEIGHTS)
        ids=np.empty(total,np.int64)
        for i,(s,c,p) in enumerate(zip(source,colors,phases)):
            if s in (0,3,4):
                categories=self.balanced[int(s),int(c),int(p)];category=int(rng.choice(list(categories)));bins=categories[category]
                group=bins[int(rng.integers(len(bins)))];keys=list(group);pool=group[keys[int(rng.integers(len(keys)))]]
                recovering=pool[self.arrays["recovery"][pool]] if s==4 else []
                ids[i]=rng.choice(recovering if len(recovering) and rng.random()<.8 else pool)
            else:ids[i]=rng.choice(self.pools[int(s),int(c),int(p)])
        if mode=="curriculum":
            original=ids.copy()
            for s in range(5):
                for c in range(3):
                    for p in range(4):
                        positions=np.flatnonzero((source==s)&(colors==c)&(phases==p));values=ids[positions]
                        ids[positions]=values[np.argsort(self.arrays["difficulty"][values],kind="stable")]
            assert np.array_equal(np.sort(original),np.sort(ids))
        np.savez_compressed(path,ids=ids.reshape(STEPS,BATCH));return ids.reshape(STEPS,BATCH)


if __name__=="__main__":prepare()
