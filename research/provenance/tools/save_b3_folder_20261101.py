"""Assemble the user's B3 folder from verified existing results and new backups."""
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil

WORK = Path('C:/Users/LHY/Desktop/服务器/mujoco_robosuite')
RUN = WORK/'experiments/colorlift_b3_archive_20261101'
BASE = Path('C:/Users/LHY/Desktop/模型训练/B3_模型完整归档_20261101')
CURRENT = Path('C:/Users/LHY/Desktop/模型训练/B2_B3复现确认_20261031/final_results')
FIRST = Path('C:/Users/LHY/Desktop/模型训练/ACT广域训练_第一轮_20261030/final_results')


def disk(p):
    text=str(Path(p).absolute())
    return Path('\\\\?\\'+text) if os.name=='nt' and not text.startswith('\\\\?\\') else Path(text)
def read(p):return json.loads(disk(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256();size=disk(p).stat().st_size
    with disk(p).open('rb') as f:
        for b in iter(lambda:f.read(min(8*1024*1024,max(size,8192))),b''):h.update(b)
    return h.hexdigest()
def save(p,v):
    data=json.dumps(v,ensure_ascii=False,indent=2)
    disk(p).parent.mkdir(parents=True,exist_ok=True)
    if disk(p).exists():assert disk(p).read_text(encoding='utf-8')==data,p
    else:disk(p).write_text(data,encoding='utf-8')
def copy(source,relative,expected=None):
    target=BASE/relative;expected=expected or sha(source)
    if disk(target).exists():assert sha(target)==expected,target
    else:
        disk(target).parent.mkdir(parents=True,exist_ok=True)
        tmp=disk(str(target)+f'.copying.{os.getpid()}')
        shutil.copyfile(disk(source),tmp);assert sha(tmp)==expected,relative;os.rename(tmp,disk(target))
    return {'path':str(relative).replace('\\','/'),'bytes':disk(target).stat().st_size,'sha256':expected}


def prepare():
    RUN.mkdir(exist_ok=True)
    choice={'user_decision_date_Asia_Shanghai':'2026-11-01','chosen_model_family':'B3a Goal-Relative ACT',
        'relative_lambda':0.1,'main_model_role':'existing frozen reference B3 seed 44',
        'main_model':'models/main_frozen_reference_seed44/policy.pt',
        'main_model_sha256':'d34b91b8882a764d7556a6a28663b6fe3d5d2b8a94426dbffa798d41ba41704d',
        'new_replication_seeds':[45,46,47,48,49,50],'historical_reference_seeds':[43,44],
        'replicas_kept_separately':True,'weights_not_averaged':True,'choice_not_a_new_significance_claim':True}
    save(BASE/'B3_MODEL_CHOICE.json',choice)
    lock=read(CURRENT/'model_lock.json')['models'];analysis=read(CURRENT/'confirmatory_analysis.json')
    direct=[];index={}
    for name,cfg in lock.items():
        if cfg['group']!='B3':continue
        seed=cfg['training_seed']
        if name=='legacy_B3_s44':
            prefix=Path('models/main_frozen_reference_seed44');metadata=FIRST/'models/B3a_l0p1_s44'
            source=CURRENT/'frozen_references/B3_s44.pt'
        elif name=='legacy_B3_s43':
            prefix=Path('models/historical_reference_seed43');metadata=FIRST/'models/B3a_l0p1_s43'
            source=CURRENT/'frozen_references/B3_s43.pt'
        else:
            prefix=Path(f'models/replication_seed{seed}');metadata=CURRENT/'models'/name;source=metadata/'policy.pt'
        direct.append(copy(source,prefix/'policy.pt',cfg['sha256']))
        for p in metadata.iterdir():
            if p.is_file() and p.suffix in ['.json','.jsonl']:
                direct.append(copy(p,prefix/p.name))
        index[name]={**cfg,'saved_policy':str(prefix/'policy.pt').replace('\\','/'),'confirmation_metrics':analysis['models'][name]}
    save(BASE/'MODEL_INDEX.json',index)
    report_names=['REPORT.zh-CN.md','confirmatory_analysis.json','confirmation_completed.json','analysis_completed.json',
                  'model_lock.json','all_development_selections.json','protocol.json','seal.json','sealing_completed.json',
                  'generation_audit.json','preparation_completed.json','training_started.json','reference_manifest.json',
                  'DELIVERY_COMPLETED.json','B2_B3_confirmation_comparison.png','paired_seed_summary.csv']
    for name in report_names:
        direct.append(copy(CURRENT/name,Path('reports/reproduction')/name))
    for name in ['REPORT.zh-CN.md','final_analysis.json','final_completed.json','B3_completed.json','candidate_lock.json','data_manifest.json','protocol.json']:
        if (FIRST/name).exists():direct.append(copy(FIRST/name,Path('reports/round1')/name))
    save(BASE/'reports/B3_metrics_by_model.json',{'models':{k:v for k,v in analysis['models'].items() if v['group']=='B3'},
         'new_seed_summary':{metric:{k:v for k,v in analysis['crossed_intervals'][metric]['overall'].items() if k.startswith('B3')} for metric in ['success','contact','steps']}})
    with (CURRENT/'failure_index.csv').open(encoding='utf-8-sig',newline='') as f:
        failures=[row for row in csv.DictReader(f) if row['group']=='B3']
    jobs={}
    for row in failures:
        for filename in ['front.mp4','steps.csv','metadata.json','initial_front.png','initial_side.png']:
            rel=Path(row['video']).parent/filename
            source=CURRENT/rel
            if source.exists():jobs[str(rel)]=(source,Path('failure_cases')/rel)
        row['video']='failure_cases/'+row['video'];row['trace']='failure_cases/'+row['trace']
    with ThreadPoolExecutor(max_workers=8) as pool:
        direct.extend(pool.map(lambda pair:copy(*pair),jobs.values()))
    with disk(BASE/'B3_failure_index.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(failures[0]));writer.writeheader();writer.writerows(failures)
    template=(CURRENT/'failure_videos.html').read_text(encoding='utf-8')
    start=template.index('const DATA=')+len('const DATA=');end=template.index(';let page=',start)
    template=template[:start]+json.dumps(failures,ensure_ascii=False).replace('</','<\\/')+template[end:]
    template=template.replace('B2/B3','B3').replace('<option>B2</option>','').replace('B2 / B3','B3')
    disk(BASE/'B3_failure_videos.html').write_text(template,encoding='utf-8')
    archives=[('experiments/colorlift_repro_confirm_v1/stage1_results.tar.gz','stage1_results.tar.gz',
               '09e02722d0103f5ea41191985c7b19856ac25e350022048151273f73581ee698'),
              ('experiments/colorlift_broad_act_round1/round1_results.tar.gz','round1_results.tar.gz',
               '4dc4f65ad602830c5cd991c2743f547bb4bd21dec01841b590ef6d85b5be8e77')]
    for source,name,expected in archives:
        print('COPY_RESEARCH_ARCHIVE',name,flush=True)
        direct.append(copy(WORK/source,Path('archives')/name,expected))
    save(RUN/'direct_files_prepared.json',{'selected_policies':8,'B3_failure_cases':len(failures),'copied_files':direct})
    print('B3_DIRECT_FOLDER_READY',str(BASE),'failure_cases',len(failures),flush=True)


def finalize():
    package=read(RUN/'downloads/package_completed.json');verification=read(RUN/'download_archives_verified.json')
    assert verification['all_archives_match'] and len(verification['archives'])==4
    for info in package['archives']:
        verified=verification['archives'][info['archive']]
        assert verified['sha256']==info['sha256'] and verified['all_inner_file_hashes_match']
        copy(RUN/'downloads'/info['archive'],Path('archives')/info['archive'],info['sha256'])
    for name in ['package_completed.json','preservation_scope.json']:
        copy(RUN/'downloads'/name,Path('provenance')/name)
    for name in ['download_archives_verified.json','direct_files_prepared.json']:
        copy(RUN/name,Path('provenance')/name)
    for name in ['save_b3_folder_20261101.py','verify_b3_downloads_20261101.py','archive_b3_server_20261101.py']:
        copy(WORK/'scripts'/name,Path('provenance/tools')/name)
    notes='''# B3 模型完整归档

按你2026年11月1日的决定，将B3（Goal-Relative ACT，λ=0.1）及相关数据保存于此文件夹。

## 模型入口

- 原已冻结的B3参考权重：`models/main_frozen_reference_seed44/policy.pt`。
- 六个新增训练种子：`models/replication_seed45`至`models/replication_seed50`。
- 历史种子43参考：`models/historical_reference_seed43`。
- 每个目录保留各自配置、开发集选模记录和训练日志；`MODEL_INDEX.json`记录权重SHA256及对应评测。各权重独立保存，主文件沿用原冻结参考。
- 权重内含ACT参数、颜色嵌入和归一化统计。输入为RGB、本体状态与目标颜色；执行器为原四阶段OSC，执行长度4/2/1/4。

## 数据与训练状态

| 压缩包 | 保存内容 |
|---|---|
| archives/expert_demonstrations.tar.gz | 本轮B3实际使用的2337条训练、432条验证示范，轨迹、RGB、本体状态、动作、步骤记录、几何监督标签、元数据及原始示范媒体；训练/验证划分随原清单保存 |
| archives/training_cache.tar.gz | 全部38个冻结训练/验证数组与缓存完成记录，含HSV标签、有效步掩码、相对位置和预抓取监督；解压后约28.7GB |
| archives/training_checkpoints.tar.gz | 六个新增种子、两个原λ=0.1种子以及历史λ=0.3/1候选的全部中间权重、last.pt、优化器、调度器、Torch/CUDA随机状态、初始化权重、采样计划与训练日志 |
| archives/source_environment.tar.gz | 冻结Python源码、资产XML、ACT第三方源码、已有wheelhouse、环境YAML、pip/conda版本记录和归档工具 |
| archives/stage1_results.tar.gz | 全部复现确认实验原始记录，包括新封存布局、初始PNG、逐步CSV、单相机视频、报告、权重与统计；保留B2对照记录供追溯 |
| archives/round1_results.tar.gz | 第一轮完整研究记录，含B3超参数实验、开发/最终评测与原始对照证据 |

四个新数据包内含`archive_manifest.json`，记录每个文件的SHA256；包内`workspace/`保存原服务器工作区的相对路径。原工作区为`/home/catlab/mujoco_robosuite`。源码包还保留已有离线安装包与环境配置；操作系统和驱动由运行服务器提供。

四个新数据包中的`workspace/`内容对应原工作区。`stage1_results.tar.gz`内路径相对于原`experiments/colorlift_repro_confirm_v1`；`round1_results.tar.gz`按原工作区相对路径保存。建议先分别解包查看，再按对应层级恢复。共享缓存原位于`experiments/colorlift_broad_act_round1/cache`，复现目录的cache可指向该目录。原始清单中的绝对路径保持原样，以便追溯。

## 评测与视频

- 直接打开`B3_failure_videos.html`可浏览369个B3目标失败或接触不合格案例，均为单前视视频，并可查看逐步CSV和初始PNG。
- `reports/B3_metrics_by_model.json`保存各B3版本的三色、分测试集与效率统计。
- `reports/reproduction`及`reports/round1`保留原始研究报告与统计。原始报告和统计中的路径按原实验目录记录，完整对应文件位于两个研究记录压缩包中。
- 采用B3是你的使用决定；原有实验统计结论和各个种子结果完整保留。

本次仅保存既有模型与数据。归档校验结果见`SAVE_COMPLETED.json`和`FILE_MANIFEST.json`，原目录中的文件仍保留。
'''
    disk(BASE/'README.zh-CN.md').write_text(notes,encoding='utf-8')
    print('BUILD_FINAL_FOLDER_MANIFEST',flush=True)
    entries={}
    for path in disk(BASE).rglob('*'):
        if path.is_file() and path.name not in ['FILE_MANIFEST.json','SAVE_COMPLETED.json']:
            relative=str(path.relative_to(disk(BASE))).replace('\\','/')
            entries[relative]={'bytes':path.stat().st_size,'sha256':sha(path)}
    for expected in read(RUN/'direct_files_prepared.json')['copied_files']:
        actual=entries[expected['path']]
        assert actual['sha256']==expected['sha256'] and actual['bytes']==expected['bytes'],expected['path']
    for expected in package['archives']:
        assert entries['archives/'+expected['archive']]['sha256']==expected['sha256'],expected['archive']
    save(BASE/'FILE_MANIFEST.json',{'files':entries})
    receipt={'completed_at_UTC':datetime.now(timezone.utc).isoformat(),'user_choice_date_Asia_Shanghai':'2026-11-01',
             'folder':str(BASE),'chosen_model_family':'B3','main_policy':'models/main_frozen_reference_seed44/policy.pt',
             'main_policy_sha256':sha(BASE/'models/main_frozen_reference_seed44/policy.pt'),
             'direct_selected_policies':8,'B3_failure_cases':369,'total_archives':6,
             'training_episodes':2337,'validation_episodes':432,'cache_arrays':38,
             'all_copied_and_downloaded_file_hashes_match':True,'all_new_archive_inner_file_hashes_match':True,
             'manifest_sha256':sha(BASE/'FILE_MANIFEST.json'),'files':len(entries),'bytes':sum(v['bytes'] for v in entries.values()),
             'original_files_preserved':True,'new_training_or_evaluation_started':False}
    save(BASE/'SAVE_COMPLETED.json',receipt);save(RUN/'SAVE_COMPLETED.json',receipt)
    print(json.dumps(receipt,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    import sys
    {'prepare':prepare,'finalize':finalize}[sys.argv[1]]()
