"""Human-readable complete stage1 report and verifiable compact export."""
import csv
from pathlib import Path
import tarfile
from datetime import datetime
from zoneinfo import ZoneInfo
from .spec import ROOT,RUN,PRIVATE,SEEDS,SUITES,read,sha,atomic,now


def main():
    assert (RUN / "analysis_completed.json").exists()
    analysis=read(RUN / "confirmatory_analysis.json");lock=read(RUN / "model_lock.json")
    completed=read(RUN / "confirmation_completed.json")
    assert completed["total_episodes"]==86016
    stats=analysis["crossed_intervals"]["success"]["overall"]
    verdict={"inconclusive":"证据不足，尚不能确定 B2/B3 哪个更好；不等于两者等价。",
        "B3_superior_on_frozen_protocol":"B3 在冻结协议下通过预注册的优势与观测稳定性判据。",
        "B2_superior_on_frozen_protocol":"B2 在冻结协议下通过预注册的优势与观测稳定性判据。"}[analysis["verdict"]]
    text=["# B2/B3 多种子复现与全新封存布局确认：第一阶段结果","",
        "报告时间："+datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()+"（北京时间）。","",verdict,"",
        "已完成新增 6 对训练种子、12 个模型训练与开发集选模；确认阶段在全新的 768 个布局上执行 86016 次试验。其中新种子的 64512 次用于主要确认分析，旧训练种子模型的 21504 次单列补充。旧 768 个布局用于历史排除和补充证据，没有承担本轮确认性检验。","",
        "## 冻结与封存顺序","",
        "新训练种子为 45–50；B2/B3 从同一原始共享 checkpoint 起步。B3a λ=0.1、数据、优化器、6000 更新预算、RGB/本体状态/32维颜色嵌入、原执行器与 4/2/1/4 执行长度均冻结。原 B3 参考权重未更新。每个种子共享相同采样计划，训练 GPU 在两个方法间交替分配。","",
        "先冻结代码、数据与统计方法，再运行模型无关 Valid 并封存新布局；随后训练，全部模型以原 48 个开发布局选择 checkpoint，再锁定 16 个权重并统一解封。所有种子都进入对应分析，不通过新测试挑选最好种子。","",
        "确认集继续采用 Broad-IID、Spatial-OOD、Relation-OOD 各256个布局，单模型2304/768/2304次，共5376次；3次循环颜色置换，单次环境实现。该分布仍包含 Sobol 与受限关系提案，不是整个工作空间的无条件 IID 均匀随机。","",
        "## 主要确认性结果","",
        f"新种子平均成功率：B2 {100*stats['B2']:.4f}%，B3 {100*stats['B3']:.4f}%。差值 B3−B2 为 {100*stats['difference_B3_minus_B2']:.4f} 个百分点。",
        f"配对训练种子双侧精确置换 p={analysis['primary']['two_sided_p']:.6g}；种子×布局交叉重采样95%差值区间为 [{100*stats['difference_95'][0]:.4f}, {100*stats['difference_95'][1]:.4f}] 个百分点。","",
        "p 值以 6 对新训练种子为单位枚举 64 次标签分配；不将数万条试验当作数万个独立训练重复。区间按配对种子和各测试集内 geometry_id 共同重采样，颜色/指令条件留在同一布局簇内。少种子、近乎全成功和 Sobol 采样的限制均保留，不显著不解释为等价。","",
        "| 新种子 | B2 成功 | B3 成功 | 差值 pp | B2 接触合格 | B3 接触合格 | B2/B3 平均步数 |",
        "|---:|---:|---:|---:|---:|---:|---:|"]
    for seed in SEEDS:
        a=analysis["models"][f"B2_s{seed}"]["overall"];b=analysis["models"][f"B3_s{seed}"]["overall"]
        text.append(f"| {seed} | {a['successes']}/5376 | {b['successes']}/5376 | {100*(b['success_rate']-a['success_rate']):.4f} | {a['contact_successes']}/5376 | {b['contact_successes']}/5376 | {a['mean_steps_all']:.2f}/{b['mean_steps_all']:.2f} |")
    text += ["","## 新种子测试集分项","","| 测试集 | B2 成功率 | B3 成功率 | 差值 pp（95%区间） | B2/B3 接触合格率 |","|---|---:|---:|---|---:|"]
    for suite in SUITES:
        task=analysis["crossed_intervals"]["success"][suite];contact=analysis["crossed_intervals"]["contact"][suite]
        text.append(f"| {suite} | {100*task['B2']:.4f}% | {100*task['B3']:.4f}% | {100*task['difference_B3_minus_B2']:.4f} [{100*task['difference_95'][0]:.4f}, {100*task['difference_95'][1]:.4f}] | {100*contact['B2']:.4f}%/{100*contact['B3']:.4f}% |")
    text += ["","逐新种子、逐测试集的观测稳定性门槛为目标成功≥98%、接触合格成功≥95%、零观测抓错。",
        f"全部门槛：B2={analysis['safety_gate']['B2']}，B3={analysis['safety_gate']['B3']}。未通过项数：{len(analysis['safety_gate_failures'])}。每个具体未通过项见 confirmatory_analysis.json，不能用平均数隐藏最差种子。","",
        "## 三色与运动效率","","| 方法 | 颜色 | 新种子总成功 | 新种子总接触合格 | 观测抓错 |","|---|---|---:|---:|---:|"]
    for group in ("B2","B3"):
        for color in ("red","yellow","blue"):
            rows=[analysis["models"][f"{group}_s{s}"]["by_color"][color] for s in SEEDS]
            text.append(f"| {group} | {color} | {sum(r['successes'] for r in rows)}/{sum(r['episodes'] for r in rows)} | {sum(r['contact_successes'] for r in rows)}/{sum(r['episodes'] for r in rows)} | {sum(r['wrong_colors'] for r in rows)} |")
    text += ["","所有试验仍以目标抬升超过4cm且稳定保持2s评分。全部平均步数包含保持与失败；模型分项同时保存物理末端路径、夹紧前误差、方向反转、闭合后空等、空夹爪等待和距离不减步数。夹紧前误差只对实际进入夹紧阶段的试验有定义。","",
        "## 历史训练种子模型的新布局补充结果","",
        "这些模型使用新的布局，但训练种子43、44曾参与上一轮λ/模型选择，因此不并入新的6对主要跨种子推断。固定B3参考（种子44）也在此列出。","",
        "| 模型 | 新布局成功 | 接触合格 | 抓错 | 平均步数 |","|---|---:|---:|---:|---:|"]
    for name,value in analysis["models"].items():
        if value["cohort"]!="legacy_supplement":continue
        row=value["overall"];text.append(f"| {name} | {row['successes']}/5376 | {row['contact_successes']}/5376 | {row['wrong_colors']} | {row['mean_steps_all']:.2f} |")
    text += ["","## 配对输入与次要检验","",
        f"初始输入审计：比较 {analysis['initial_input_audit']['case_comparisons']} 个模型间配对；RGB哈希差异 {analysis['initial_input_audit']['RGB_hash_differences']} 例，最大像素绝对差 {analysis['initial_input_audit']['max_pixel_absolute_difference']}。全部试验保存前、侧初始PNG，可核查像素，不只保留哈希。物理参数最大差与示例见JSON。",
        "三测试集×任务/接触指标的6个次要种子级p值统一做Holm校正，单列报告，不替代主要检验。无观测错误不保证总体错误率为零；没有引入0.2个百分点等未批准的等价界限。","",
        "## 后续边界","",
        "第一阶段完成后停止，不自动生成第二阶段场景。即使某方法在本协议下胜出，也不能声称在所有随机场景下普遍最优；继承旧权重还可能带有历史区域/关系暴露。第二阶段的IID随机、分散、并排和图案分布需要另行冻结配额、参数和新测试集；若本轮证据不足，保留两方法并明确报告，不强行选赢家。","",
        "原始文件：protocol.json、seal.json、model_lock.json、confirmatory_analysis.json、development selections、原始 steps.csv 和所有初始PNG。失败及预先指定代表性成功视频均为单前视相机。failure_index.csv 可定位每个接触不合格或目标失败案例。","",
        "统计实现参照 SciPy 配对置换检验、Owen/Eckles 交叉因素bootstrap 与 Holm 多重检验；方法已在新布局生成前冻结。","",
        "- https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html",
        "- https://arxiv.org/abs/1106.2125",
        "- https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html",""]
    (RUN / "REPORT.zh-CN.md").write_text("\n".join(text),encoding="utf-8")
    index=[]
    for model in lock["models"]:
        for path in (RUN / "evaluations/confirm_locked" / model).rglob("metadata.json"):
            row=read(path)
            if not row["eligible"]:index.append({"model":model,"group":lock["models"][model]["group"],"cohort":lock["models"][model]["cohort"],
                "training_seed":lock["models"][model]["training_seed"],"suite":row["layout"]["eval_suite"],"geometry_id":row["geometry_id"],
                "permutation":row["layout"]["color_assignment_index"],"color":row["target_color"],"success":row["success"],
                "steps":row["control_steps"],"wrong_color":row["wrong_color_grasp"],"reasons":";".join(row["rejection_reasons"]),
                "video":row["directory"]+"/front.mp4","trace":row["directory"]+"/steps.csv"})
    names=(list(index[0]) if index else ["model","suite","geometry_id","color","success","video"])
    with (RUN / "failure_index.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=names);writer.writeheader();writer.writerows(index)
    result={"at_UTC":now(),"verdict":analysis["verdict"],"failure_index_entries":len(index),"report_sha256":sha(RUN / "REPORT.zh-CN.md")}
    atomic(RUN / "report_completed.json",result)
    export()


def export():
    marker=RUN / "export_archive.json"
    if marker.exists():
        value=read(marker)
        if (RUN / "stage1_results.tar.gz").exists() and sha(RUN / "stage1_results.tar.gz")==value["sha256"]:return
    assert (RUN / "analysis_completed.json").exists()
    paths=[]
    for path in RUN.rglob("*"):
        if not path.is_file():continue
        relative=path.relative_to(RUN)
        if relative.parts[0]=="cache":continue
        if path.name in ("export_manifest.json","export_archive.json","coordinator.lock"):continue
        if path.suffix in (".gz",".sqlite",".sqlite-journal",".npy"):continue
        if path.suffix==".pt" and not (path.name=="policy.pt" or relative.parts[0]=="frozen_references" or path.name=="frozen_initial.pt"):continue
        if "technical_attempt_" in str(relative):continue
        if path.suffix in (".json",".jsonl",".py",".md",".csv",".png",".mp4",".log",".npz",".pt"):
            paths.append(path)
    manifest={str(path.relative_to(RUN)):{"bytes":path.stat().st_size,"sha256":sha(path)} for path in sorted(paths)}
    atomic(RUN / "export_manifest.json",{"at_UTC":now(),"files":manifest,"private_cohort_released_after_confirmation":True})
    paths.append(RUN / "export_manifest.json")
    archive=RUN / "stage1_results.tar.gz"
    with tarfile.open(archive,"w:gz",compresslevel=3) as tar:
        for path in sorted(paths):tar.add(path,arcname=str(path.relative_to(RUN)),recursive=False)
    atomic(marker,{"at_UTC":now(),"path":str(archive),"bytes":archive.stat().st_size,"sha256":sha(archive),"files":len(paths)})
    print("STAGE1_EXPORT_COMPLETE",len(paths),archive.stat().st_size,flush=True)


if __name__=="__main__":main()
