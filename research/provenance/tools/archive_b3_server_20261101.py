"""Preserve existing B3 inputs and training states; never run a policy or train."""
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import threading
import traceback

ROOT = Path('/home/catlab/mujoco_robosuite')
OLD = ROOT/'experiments/colorlift_broad_act_round1'
NEW = ROOT/'experiments/colorlift_repro_confirm_v1'
OUT = ROOT/'experiments/colorlift_b3_archive_20261101'
GUARD = threading.Lock()
STATE = {}


def now(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
def atomic(p,value):
    tmp=p.with_name(p.name+'.writing');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2));os.replace(tmp,p)
def progress(name,**values):
    with GUARD:
        STATE.setdefault(name,{}).update(values)
        atomic(OUT/'progress.json',{'at_UTC':now(),'jobs':STATE})


class Reader:
    def __init__(self,stream): self.stream=stream;self.hash=hashlib.sha256();self.bytes=0
    def read(self,n=-1):
        b=self.stream.read(n);self.hash.update(b);self.bytes+=len(b);return b


def package(name, files):
    marker=OUT/(name+'_completed.json');target=OUT/(name+'.tar.gz')
    if marker.exists():
        result=read(marker)
        assert target.is_file() and sha(target)==result['sha256']
        progress(name,status='complete',**result);return result
    total=sum(p.stat().st_size for p,e in files.values());manifest={};done=0
    progress(name,status='packing',files_total=len(files),bytes_total=total,files_done=0,bytes_done=0)
    temporary=OUT/(name+f'.partial.{os.getpid()}.tar.gz')
    with tarfile.open(temporary,'w:gz',compresslevel=1) as tar:
        for i,(relative,(path,expected)) in enumerate(sorted(files.items()),1):
            assert not relative.startswith('/') and '..' not in Path(relative).parts
            before=path.stat();info=tar.gettarinfo(str(path),arcname=relative)
            assert info.isfile()
            with path.open('rb') as stream:
                source=Reader(stream);tar.addfile(info,source)
            after=path.stat();actual=source.hash.hexdigest()
            assert source.bytes==before.st_size==after.st_size and before.st_mtime_ns==after.st_mtime_ns,path
            if expected: assert actual==expected,(path,'frozen source hash mismatch')
            manifest[relative]={'bytes':source.bytes,'sha256':actual}
            done+=source.bytes
            if i%50==0 or source.bytes>64*1024*1024 or i==len(files):
                progress(name,files_done=i,bytes_done=done,current=relative)
        data=json.dumps({'created_at_UTC':now(),'kind':name,'files':manifest},ensure_ascii=False,indent=2).encode()
        info=tarfile.TarInfo('archive_manifest.json');info.size=len(data);tar.addfile(info,io.BytesIO(data))
    assert not target.exists(),target
    os.rename(temporary,target)
    result={'archive':target.name,'files':len(manifest),'uncompressed_bytes':done,'bytes':target.stat().st_size,'sha256':sha(target)}
    atomic(marker,result);atomic(OUT/(name+'_manifest.json'),{'files':manifest})
    progress(name,status='complete',**result)
    print('ARCHIVE_COMPLETED',name,result['bytes'],flush=True)
    return result


def main():
    OUT.mkdir(exist_ok=True)
    with (OUT/'packaging.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (OUT/'package_completed.json').exists():
            result=read(OUT/'package_completed.json')
            assert all(sha(OUT/a['archive'])==a['sha256'] for a in result['archives'])
            print('ALREADY_COMPLETE',flush=True);return
        p=read(NEW/'protocol.json');data=read(OLD/'data_manifest.json');models=read(NEW/'model_lock.json')['models']
        assert read(NEW/'confirmation_completed.json')['total_episodes']==86016
        assert read(NEW/'analysis_completed.json')['data_unchanged_after_experiment']
        assert sha(NEW/'protocol.json')=='04ca3acf602ae50b56ad3e63329894775f716a960836dbe8feec6e962f86efb1'
        groups={k:{} for k in ['training_cache','expert_demonstrations','training_checkpoints','source_environment']}
        def add(group,path,expected=None,relative=None):
            path=Path(path);assert path.is_file() and not path.is_symlink(),path
            relative=relative or 'workspace/'+str(path.relative_to(ROOT))
            old=groups[group].get(relative)
            assert old is None or old[0]==path,(relative,old,path)
            groups[group][relative]=(path,expected or (old[1] if old else None))
        for rel,meta in p['cache_fingerprints'].items():add('training_cache',OLD/'cache'/rel,meta['sha256'])
        add('training_cache',OLD/'cache/completed.json')
        trajectory_paths=set()
        for row in data['rows']:
            raw=Path(row['root'])/row['path'];assert raw.resolve().is_relative_to(ROOT)
            trajectory_paths.add(str(raw))
            add('expert_demonstrations',raw,row['npz_sha256'])
            for key in ['supervision','trace']:
                if isinstance(row.get(key),str):add('expert_demonstrations',Path(row['root'])/row[key])
            if raw.name=='trajectory.npz':
                for item in raw.parent.iterdir():
                    if item.is_file():add('expert_demonstrations',item)
            else:
                for item in raw.parent.glob(raw.stem+'.*'):
                    if item.is_file():add('expert_demonstrations',item)
        assert len(trajectory_paths)==len(data['rows'])==2769
        add('expert_demonstrations',OLD/'data_manifest.json')
        for rel,expected in p['data_manifest_sha256'].items():
            if not rel.startswith('experiments/colorlift_repro_confirm_v1/'):
                add('expert_demonstrations',ROOT/rel,expected)
        for d in [ROOT/'experiments/colorlift_broad_protocol_v2',ROOT/'experiments/colorlift_conditional_v1',ROOT/'experiments/colorlift_recovery_v1']:
            for item in d.iterdir():
                if item.is_file() and item.suffix in ('.json','.md','.csv','.yml','.txt'):
                    add('expert_demonstrations',item)
        model_dirs=[]
        for parent in [OLD/'models',NEW/'models']:
            for directory in sorted(parent.iterdir()):
                if directory.is_dir() and directory.name.startswith('B3'):
                    model_dirs.append(str(directory.relative_to(ROOT)))
                    for item in directory.rglob('*'):
                        if item.is_file():
                            expected=None
                            if item.name=='policy.pt':
                                expected=read(directory/'selection.json')['sha256']
                            add('training_checkpoints',item,expected)
            for item in parent.parent.glob('train_B3*.log'):add('training_checkpoints',item)
        for name,config in models.items():
            if config['group']=='B3':
                add('training_checkpoints',ROOT/config['path'],config['sha256'])
        add('training_checkpoints',NEW/'frozen_initial.pt',p['initial_sha256'])
        for item in (NEW/'plans').glob('*.npz'):add('training_checkpoints',item)
        for item in (OLD/'plans').glob('*43*.npz'):add('training_checkpoints',item)
        for item in (OLD/'plans').glob('*44*.npz'):add('training_checkpoints',item)
        assert len(model_dirs)==12
        for rel,expected in p['source_sha256'].items():
            add('source_environment',NEW/'source_snapshot'/rel,expected,'workspace/'+rel)
        for name in ['colorlift','lift_act','catlab_assets','third_party','scripts']:
            directory=ROOT/name
            if not directory.exists():continue
            for item in directory.rglob('*'):
                if item.is_file() and not item.is_symlink() and not any(part in ['.git','__pycache__','.pytest_cache'] for part in item.relative_to(directory).parts):
                    rel=str(item.relative_to(ROOT))
                    if 'workspace/'+rel not in groups['source_environment']:add('source_environment',item)
        for item in ROOT.iterdir():
            if item.is_file() and (item.suffix in ['.md','.yml','.toml','.sh'] or item.name.startswith('requirements')):
                add('source_environment',item)
        environment=OUT/'environment_records';environment.mkdir(exist_ok=True)
        commands={'pip_freeze_20261101.txt':['python','-m','pip','freeze'],
                  'conda_explicit_linux_20261101.txt':['conda','list','--explicit'],
                  'conda_environment_20261101.yml':['conda','env','export','--no-builds']}
        for name,command in commands.items():
            target=environment/name
            if not target.exists():
                value=subprocess.run(command,capture_output=True,text=True,check=True).stdout
                target.write_text(value)
            add('source_environment',target,relative='environment_records/'+name)
        versions={name:importlib.metadata.version(name) for name in p['software_versions']}
        atomic(environment/'runtime_versions.json',{'frozen_training_versions':p['software_versions'],'current_versions':versions,'match':versions==p['software_versions']})
        add('source_environment',environment/'runtime_versions.json',relative='environment_records/runtime_versions.json')
        add('source_environment',Path(__file__).resolve(),relative='preservation_tools/archive_b3_server_20261101.py')
        summary={'client_date_Asia_Shanghai':'2026-11-01','user_selected_model_family':'B3','relative_lambda':0.1,
                 'original_frozen_reference':'legacy_B3_s44','main_reference_sha256':models['legacy_B3_s44']['sha256'],
                 'new_replication_seeds':[45,46,47,48,49,50],'historical_reference_seeds':[43,44],
                 'intermediate_model_directories':model_dirs,'logical_demonstration_episodes':len(data['rows']),
                 'training_validation_counts':data['splits'],'files_by_archive':{k:len(v) for k,v in groups.items()},
                 'uncompressed_bytes_by_archive':{k:sum(q.stat().st_size for q,e in v.values()) for k,v in groups.items()},
                 'no_new_training_or_evaluation':True,'GPU_used_for_preservation':False,'original_files_preserved':True}
        atomic(OUT/'preservation_scope.json',summary)
        print('PRESERVATION_SCOPE',json.dumps(summary),flush=True)
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures=[pool.submit(package,name,files) for name,files in groups.items()]
            results=[future.result() for future in futures]
        atomic(OUT/'package_completed.json',{'completed_at_UTC':now(),'archives':results,'scope':summary})
        print('B3_PRESERVATION_PACKAGES_COMPLETE',flush=True)


if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(exist_ok=True);atomic(OUT/'packaging_error.json',{'at_UTC':now(),'traceback':traceback.format_exc()});raise
