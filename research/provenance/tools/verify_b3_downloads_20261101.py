"""Verify every member of downloaded B3 archives without expanding the cache."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile
import time

ROOT=Path('C:/Users/LHY/Desktop/服务器/mujoco_robosuite')
RUN=ROOT/'experiments/colorlift_b3_archive_20261101'
DOWNLOAD=RUN/'downloads'
PROTOCOL=json.loads((ROOT/'experiments/colorlift_repro_confirm_v1/protocol.json').read_text(encoding='utf-8'))
LOCK=json.loads((ROOT/'experiments/colorlift_repro_confirm_v1/model_lock.json').read_text(encoding='utf-8'))


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def atomic(path,value):
    p=path.with_name(path.name+'.writing');p.write_text(json.dumps(value,indent=2),encoding='utf-8');p.replace(path)


def verify(info):
    path=DOWNLOAD/info['archive'];assert path.stat().st_size==info['bytes']
    actual=digest(path);assert actual==info['sha256'],path
    entries={};record=None;data_manifest=None
    print('VERIFYING_ARCHIVE_MEMBERS',path.name,flush=True)
    with tarfile.open(path,'r|gz',bufsize=256*1024) as tar:
        for member in tar:
            rel=PurePosixPath(member.name)
            assert member.isfile() and not rel.is_absolute() and '..' not in rel.parts and member.name not in entries
            with tar.extractfile(member) as f:
                if member.name=='archive_manifest.json':
                    assert record is None;record=json.loads(f.read());continue
                h=hashlib.sha256();n=0;parts=[]
                collect=member.name=='workspace/experiments/colorlift_broad_act_round1/data_manifest.json'
                for b in iter(lambda:f.read(4*1024*1024),b''):
                    h.update(b);n+=len(b)
                    if collect:parts.append(b)
                assert n==member.size
                if collect:data_manifest=json.loads(b''.join(parts))
                entries[member.name]={'bytes':n,'sha256':h.hexdigest()}
    assert record and record['files']==entries
    assert len(entries)==info['files'] and sum(v['bytes'] for v in entries.values())==info['uncompressed_bytes']
    audit={}
    if path.name=='training_cache.tar.gz':
        for name,expected in PROTOCOL['cache_fingerprints'].items():
            assert entries['workspace/experiments/colorlift_broad_act_round1/cache/'+name]['sha256']==expected['sha256']
        audit['frozen_cache_arrays']=len(PROTOCOL['cache_fingerprints'])
    if path.name=='training_checkpoints.tar.gz':
        for name,cfg in LOCK['models'].items():
            if cfg['group']=='B3':assert entries['workspace/'+cfg['path']]['sha256']==cfg['sha256'],name
        assert entries['workspace/experiments/colorlift_repro_confirm_v1/frozen_initial.pt']['sha256']==PROTOCOL['initial_sha256']
        audit['locked_B3_policies']=8
    if path.name=='source_environment.tar.gz':
        for name,expected in PROTOCOL['source_sha256'].items():assert entries['workspace/'+name]['sha256']==expected,name
        audit['frozen_python_sources']=len(PROTOCOL['source_sha256'])
    if path.name=='expert_demonstrations.tar.gz':
        assert data_manifest and len(data_manifest['rows'])==2769
        prefix='/home/catlab/mujoco_robosuite/'
        for row in data_manifest['rows']:
            original=row['root']+'/'+row['path'];assert original.startswith(prefix)
            assert entries['workspace/'+original[len(prefix):]]['sha256']==row['npz_sha256']
            for key in ['supervision','trace']:
                if isinstance(row.get(key),str):
                    dep=row['root']+'/'+row[key];assert 'workspace/'+dep[len(prefix):] in entries
        audit['demonstration_episodes']=2769
        audit['training_episodes']=data_manifest['splits']['train']['episodes']
        audit['validation_episodes']=data_manifest['splits']['val']['episodes']
    result={'sha256':actual,'bytes':path.stat().st_size,'inner_files':len(entries),'all_inner_file_hashes_match':True,
            'uncompressed_bytes_verified':sum(v['bytes'] for v in entries.values()),'audit':audit}
    print('VERIFIED_ARCHIVE',path.name,len(entries),flush=True)
    return result


def main():
    result={'archives':{},'all_archives_match':False}
    receipt=RUN/'download_archives_verified.json'
    if receipt.exists():result=json.loads(receipt.read_text(encoding='utf-8'))
    while True:
        marker=DOWNLOAD/'package_completed.json'
        if not marker.exists():time.sleep(20);continue
        package=json.loads(marker.read_text(encoding='utf-8'))
        for info in package['archives']:
            if info['archive'] in result['archives']:continue
            path=DOWNLOAD/info['archive']
            if not path.exists() or path.stat().st_size!=info['bytes']:continue
            result['archives'][info['archive']]=verify(info)
            result['updated_at_UTC']=datetime.now(timezone.utc).isoformat();atomic(receipt,result)
        if len(result['archives'])==len(package['archives']):
            result['all_archives_match']=True;atomic(receipt,result);print('ALL_B3_DOWNLOAD_ARCHIVES_VERIFIED',flush=True);return
        time.sleep(20)


if __name__=='__main__':main()
