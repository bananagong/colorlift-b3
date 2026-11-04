"""Geometry-grouped analysis and a self-contained export after locked evaluation."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from itertools import combinations
from pathlib import Path
import tarfile

import numpy as np
from scipy.stats import beta

from .spec import ROOT, RUN, DATA, COLORS, sha


GROUPS = ("B0", "B1", "B2", "B3", "B4")
SUITES = ("broad_iid", "spatial_ood", "relation_ood")
METRICS = ("success", "eligible", "control_steps", "close_XY_error_m",
           "physical_eef_path_m", "post_close_idle_steps", "empty_gripper_wait_steps",
           "no_progress_steps", "command_reversals_within_phase", "motion_reversals_within_phase")


def read_json(path):
    return json.loads(Path(path).read_text())


def finite_mean(values):
    values = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else None


def exact_binomial(successes, total):
    """Two-sided exact interval for the separate all-variants layout endpoint."""
    return [float(beta.ppf(.025, successes, total-successes+1)) if successes else 0.,
            float(beta.ppf(.975, successes+1, total-successes)) if successes < total else 1.]


def clustered(rows, key, seed=20261030, draws=5000):
    clusters = defaultdict(list)
    for row in rows:
        value = row.get(key)
        if value is not None and np.isscalar(value) and np.isfinite(value):
            clusters[row["geometry_id"]].append(float(value))
    if not clusters:
        return {"mean": None, "layout_bootstrap_95": None, "geometries": 0}
    values = [clusters[g] for g in sorted(clusters)]
    sums = np.array([sum(v) for v in values]); counts = np.array([len(v) for v in values])
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(values), (draws, len(values)))
    bootstrap = sums[samples].sum(1)/counts[samples].sum(1)
    return {"mean": float(sums.sum()/counts.sum()),
            "layout_bootstrap_95": np.quantile(bootstrap, [.025, .975]).tolist(),
            "geometries": len(values), "attempts_with_metric": int(counts.sum()),
            "bootstrap_degenerate": bool(np.ptp(bootstrap) == 0)}


def summary(rows):
    assert rows
    geometries = defaultdict(list)
    for row in rows:
        geometries[row["geometry_id"]].append(row)
    value = {"episodes": len(rows), "independent_geometries": len(geometries),
             "successes": sum(bool(r["success"]) for r in rows),
             "contact_qualified_successes": sum(bool(r["eligible"]) for r in rows),
             "wrong_color_episodes": sum(bool(r["wrong_color_grasp"]) for r in rows),
             "wrong_color_geometries": sum(any(r["wrong_color_grasp"] for r in group) for group in geometries.values()),
             "collision_episodes": sum(r["collision_steps"] > 0 for r in rows),
             "holding_failure_episodes": sum(r.get("holding_failure_events", 0) > 0 for r in rows),
             "steps_success_mean": finite_mean(r["control_steps"] for r in rows if r["success"]),
             "steps_success_p90": float(np.quantile([r["control_steps"] for r in rows if r["success"]], .9)) if any(r["success"] for r in rows) else None,
             "metrics": {key: clustered(rows, key) for key in METRICS},
             "relative_MAE_m": finite_mean(np.mean(r["relative_MAE_m"]) for r in rows if r.get("relative_MAE_m") is not None),
             "pregrasp_MAE_m": finite_mean(np.mean(r["pregrasp_MAE_m"]) for r in rows if r.get("pregrasp_MAE_m") is not None),
             "all_variants_layout_endpoints": {}}
    for key in ("success", "eligible"):
        count = sum(all(r[key] for r in group) for group in geometries.values())
        value["all_variants_layout_endpoints"][key] = {
            "passing_geometries": count, "rate": count/len(geometries),
            "exact_95": exact_binomial(count, len(geometries))}
    failed_wrong = value["wrong_color_geometries"]
    value["any_wrong_color_layout_exact_95"] = exact_binomial(failed_wrong, len(geometries))
    value["observed_gate"] = (value["successes"]/len(rows) >= .98 and
                              value["contact_qualified_successes"]/len(rows) >= .95 and
                              value["wrong_color_episodes"] == 0)
    return value


def paired_difference(rows_a, rows_b, key):
    def group(rows):
        values = defaultdict(list)
        for row in rows:
            if row.get(key) is not None:
                values[row["geometry_id"]].append(float(row[key]))
        return {g: float(np.mean(v)) for g, v in values.items()}
    a, b = group(rows_a), group(rows_b)
    ids = sorted(a.keys() & b.keys())
    if not ids:
        return None
    delta = np.array([b[g]-a[g] for g in ids])
    rng = np.random.default_rng(20261030)
    bootstrap = delta[rng.integers(0, len(ids), (5000, len(ids)))].mean(1)
    return {"difference_B_minus_A": float(delta.mean()), "layout_bootstrap_95": np.quantile(bootstrap, [.025, .975]).tolist(),
            "paired_geometries": len(ids), "bootstrap_degenerate": bool(np.ptp(bootstrap) == 0)}


def counterfactual(rows):
    """Score command/palette interventions without treating their pairs as IID."""
    geometries = defaultdict(list)
    for row in rows:
        assert row["layout"]["colors_by_slot"].index(row["target_color"]) == row["target_physical_slot"]
        geometries[row["geometry_id"]].append(row)
    pairs = {"fixed_palette_change_instruction_change_object": [],
             "fixed_instruction_swap_palette_change_object": [],
             "swap_palette_and_instruction_track_same_object": []}
    unchanged_geometry = True
    image_groups = defaultdict(list)
    for geometry, selected in geometries.items():
        reference = np.array(selected[0]["physical_slot_positions"])[:, :2]
        unchanged_geometry &= all(np.max(np.abs(np.array(r["physical_slot_positions"])[:, :2]-reference)) < 1e-5 for r in selected)
        for row in selected:
            image_groups[geometry, row["layout"]["color_assignment_index"]].append(row["initial_rgb_sha256"])
        for a, b in combinations(selected, 2):
            same_palette = a["layout"]["color_assignment_index"] == b["layout"]["color_assignment_index"]
            same_instruction = a["target_color"] == b["target_color"]
            same_object = a["target_physical_slot"] == b["target_physical_slot"]
            kind = ("fixed_palette_change_instruction_change_object" if same_palette and not same_instruction and not same_object else
                    "fixed_instruction_swap_palette_change_object" if not same_palette and same_instruction and not same_object else
                    "swap_palette_and_instruction_track_same_object" if not same_palette and not same_instruction and same_object else None)
            if kind:
                pairs[kind].append({"geometry_id": geometry, "success": a["success"] and b["success"],
                                    "eligible": a["eligible"] and b["eligible"]})
    return {"physical_XY_unchanged_within_10um": bool(unchanged_geometry),
            "same_palette_RGB_hash_identical_across_instructions": all(len(set(v)) == 1 for v in image_groups.values()),
            "pair_endpoint": "both commanded trials succeed; strict version also requires both contact-qualified; missing conditions are not scored",
            "conditions": {name: {"pairs": len(selected), "success": clustered(selected, "success"),
                                   "contact_qualified": clustered(selected, "eligible")} for name, selected in pairs.items()}}


def analyze_final():
    assert (RUN / "candidate_lock.json").exists() and (RUN / "final_completed.json").exists()
    lock = read_json(RUN / "candidate_lock.json")
    result = {"statistical_unit": "geometry_id", "bootstrap_draws": 5000,
              "all_variants_interval": "exact binomial for layouts passing every tested color/palette; a separate stricter endpoint",
              "zero_wrong_color_interpretation": "zero observed errors is not a guarantee of zero population error",
              "interval_assumptions": "cluster bootstrap estimates layout sampling variability, not training-seed variability; binomial intervals assume IID layouts and are reference intervals under stratified/Sobol sampling",
              "OOD_scope": "current-round holdout only; inherited task checkpoint may have prior region/relation exposure",
              "groups": {}, "paired_comparisons": {}}
    all_rows = {}
    for group in GROUPS:
        assert sha(ROOT / lock["models"][group]["path"]) == lock["models"][group]["sha256"]
        rows = [read_json(p) for p in (RUN / "evaluations/final_locked" / group).rglob("metadata.json")]
        assert len(rows) == 5376, (group, len(rows))
        assert len({(r["geometry_id"], r["layout"]["color_assignment_index"], r["target_color"]) for r in rows}) == 5376
        all_rows[group] = rows
        result["groups"][group] = {"suites": {}, "by_color": {}, "relation_subtypes": {}}
        for suite in SUITES:
            selected = [r for r in rows if r["layout"]["eval_suite"] == suite]
            assert len({r["geometry_id"] for r in selected}) == 256
            assert len(selected) == (768 if suite == "spatial_ood" else 2304)
            result["groups"][group]["suites"][suite] = summary(selected)
            result["groups"][group].setdefault("counterfactual", {})[suite] = counterfactual(selected)
            result["groups"][group]["by_color"][suite] = {
                color: summary([r for r in selected if r["target_color"] == color]) for color in COLORS}
        subtypes = sorted({r["layout"]["relation_subtype"] for r in rows if r["layout"]["eval_suite"] == "relation_ood"})
        for subtype in subtypes:
            result["groups"][group]["relation_subtypes"][subtype] = summary([r for r in rows if r["layout"]["relation_subtype"] == subtype])
    for before, after in zip(GROUPS, GROUPS[1:]):
        result["paired_comparisons"][f"{after}_minus_{before}"] = {
            suite: {key: paired_difference([r for r in all_rows[before] if r["layout"]["eval_suite"] == suite],
                                           [r for r in all_rows[after] if r["layout"]["eval_suite"] == suite], key)
                    for key in ("success", "eligible", "control_steps")} for suite in SUITES}
    (RUN / "final_analysis.json").write_text(json.dumps(result, indent=2))
    return result


def figure(analysis):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
    x = np.arange(3)
    for index, group in enumerate(GROUPS):
        values = analysis["groups"][group]["suites"]
        for ax, metric, factor in zip(axes, ("success", "eligible", "control_steps"), (100, 100, 1)):
            means = np.array([values[s]["metrics"][metric]["mean"] for s in SUITES])*factor
            intervals = np.array([values[s]["metrics"][metric]["layout_bootstrap_95"] for s in SUITES])*factor
            ax.bar(x+(index-2)*.15, means, width=.14, label=group,
                   yerr=np.maximum(0, np.array([means-intervals[:, 0], intervals[:, 1]-means])), capsize=2)
    for ax, label in zip(axes, ("Target success (%)", "Contact-qualified success (%)", "Control steps / attempt")):
        ax.set_xticks(x, ("Broad-IID", "Spatial-OOD*", "Relation-OOD*"), rotation=12)
        ax.set_ylabel(label); ax.grid(axis="y", alpha=.2); ax.set_axisbelow(True)
    axes[0].set_ylim(0, 105); axes[1].set_ylim(0, 105)
    axes[2].legend(ncol=3, fontsize=8)
    fig.suptitle("Locked final evaluation; intervals resample whole initial layouts")
    fig.text(.02, .015, "* Current-round holdout. Warm-start history may include prior spatial/relation exposure. All-pass bootstrap intervals may degenerate.", fontsize=8)
    fig.tight_layout(rect=(0, .05, 1, .94))
    fig.savefig(RUN / "final_comparison.png", dpi=180); plt.close(fig)


def development_rows():
    rows = []
    b0 = RUN / "evaluations/B0_initial/metrics.json"
    if b0.exists():
        rows.append(("B0", "frozen", read_json(b0)["B0"]))
    for directory in sorted((RUN / "models").glob("*")):
        for filename in ("selection_safety.json", "selection.json"):
            path = directory / filename
            if path.exists():
                item = read_json(path)
                rows.append((directory.name, item["chosen_checkpoint"], item["metrics"]))
                break
    return rows


def report(analysis=None):
    protocol = read_json(RUN / "protocol.json")
    data = read_json(RUN / "cache/completed.json")["summary"]
    progress = read_json(RUN / "status.json") if (RUN / "status.json").exists() else {}
    text = ["# 广域随机布局 ACT 第一轮实验", "",
            f"更新 UTC：{datetime.now(timezone.utc).isoformat()}；阶段：{progress.get('stage', 'unknown')}。", "",
            "本轮使用既有共享 ACT 权重微调。B0 保持冻结；B1 检查新广域数据与旧稳定/恢复数据的混合收益；B2 比较相同数据的均衡采样与课程顺序；B3a 只加目标相对位置辅助损失；B4 再加入安全预抓取和真实接触恢复示范。", "",
            f"新训练数据为 192 个布局、1728 条示范；混入旧稳定示范 417 条和旧恢复示范 192 条。合计训练 {data['train']['episodes']} 条、{data['train']['geometries']} 个布局组、{data['train']['valid_frames']} 个可监督帧。验证单独使用 48 个新布局、432 条示范。", "",
            "B1–B3 每次训练按 80% 新广域、10% 旧稳定、10% 旧恢复采样；旧验证数据未混入训练。每组两个随机种子、每个候选 6000 次更新，分别在 1500/3000/4500/6000 次更新后做闭环选模。GPU 固定为物理卡 1、2。", "",
            "所有模型使用 RGB、本体状态、32 维目标颜色嵌入与四阶段 OSC 接口；B0–B3 执行长度保持 4/2/1/4。真值目标位置只用于标签和事后评分，不作为正式推理输入。抬升成功仍要求目标物块高于初始高度 4 cm 且稳定保持 2 s；接触合格还要求目标接触、无非预期碰撞等原始审核条件。", "",
            "## 开发集闭环选模", "",
            "每个 checkpoint 均测试同一组 48 个完整布局 × 3 个目标颜色，共 144 次。不同种子和 checkpoint 不能相加为新的独立布局。先要求零抓错，再比较类别均衡成功率、接触合格率、全部尝试平均步数，验证 Loss 只作最后平局判据。", "",
            "| 候选 | 选中 checkpoint | 目标成功 | 接触合格 | 抓错 | 全部尝试平均步数 |", "|---|---|---:|---:|---:|---:|"]
    for name, chosen, metrics in development_rows():
        text.append(f"| {name} | {chosen} | {metrics['successes']}/{metrics['episodes']} | {metrics['contact_successes']}/{metrics['episodes']} | {metrics['wrong_colors']} | {metrics['mean_steps_all']:.2f} |")
    for marker, fields in (("B2_completed.json", ("selected_mode",)), ("B3_completed.json", ("selected_lambda", "selected_mode"))):
        if (RUN / marker).exists():
            value = read_json(RUN / marker)
            text += ["", marker+"："+"，".join(f"{k}={value[k]}" for k in fields)+"。"]
    calibration = RUN / "gradient_calibration.json"
    if calibration.exists():
        text += ["", "辅助任务的量纲及共享层梯度检查见 gradient_calibration.json，λ 扫描范围在 B3 训练前确定。B3a 未把预测的相对位置重新输入动作网络；B3b 不属于本轮已完成结论。"]
    controls = RUN / "evaluations/runtime_controls/metrics.json"
    if controls.exists():
        text += ["", "## B4 执行器对照", "", "| 条件 | 成功 | 接触合格 | 平均步数 |", "|---|---:|---:|---:|"]
        for name, metrics in read_json(controls).items():
            text.append(f"| {name} | {metrics['successes']}/{metrics['episodes']} | {metrics['contact_successes']}/{metrics['episodes']} | {metrics['mean_steps_all']:.2f} |")
        text += ["", "B4 主结果同时包含安全/恢复示范、预抓取监督和安全恢复执行器。应结合 B3 使用相同恢复执行器、B4 使用原执行器的对照判断来源，不能把全部提升归因于模型辅助损失。"]
    text += ["", "## 最终评测", ""]
    if analysis is None:
        text += ["最终评测尚未完成。768 个封存布局只在 B0–B4 的模型、采样器、λ 和 checkpoint 哈希全部锁定之后解封；当前开发结果不能代替最终结果。"]
    else:
        text += ["五组锁定模型各执行 5376 次，共 26880 次。三个测试集各有 256 个独立布局；Broad-IID、Relation-OOD 每布局 3 次颜色置换 × 3 个指令，Spatial-OOD 每布局 3 次置换并指定保留区域内物体的颜色。置换不增加独立布局数。", "",
                 "| 组别 | 测试集 | 目标成功 | 接触合格 | 抓错 | 平均步数 | 观测阈值 |", "|---|---|---:|---:|---:|---:|---|"]
        for group in GROUPS:
            for suite in SUITES:
                item = analysis["groups"][group]["suites"][suite]; n = item["episodes"]
                text.append(f"| {group} | {suite} | {item['successes']}/{n} ({100*item['successes']/n:.2f}%) | {item['contact_qualified_successes']}/{n} ({100*item['contact_qualified_successes']/n:.2f}%) | {item['wrong_color_episodes']} | {item['metrics']['control_steps']['mean']:.2f} | {'通过' if item['observed_gate'] else '未通过'} |")
        text += ["", "阈值为观测成功率 ≥98%、接触合格成功率 ≥95%、观测抓错为 0。零观测不意味着总体错误率为零。final_analysis.json 另给出整布局 bootstrap 区间、每布局全部颜色/置换均通过这一更严格指标的二项参考区间，以及相邻实验组的配对布局差值。全通过时 bootstrap 会退化，不能把退化区间解释为确定性保证。布局区间不包含训练种子的不确定性；二项区间假设独立同分布布局，在分层或 Sobol 采样下仅作为参考。", "",
                 "反事实统计分别检查：同画面改指令是否换抓取对象、同指令换颜色是否跟随新对象、同时换颜色和指令是否仍追踪同一物理对象。每一对要求两次试验都正确成功，并按原始布局分组统计，不把配对数当成独立样本数。", "",
                 "颜色分项、近距离与排列关系 OOD 分项、闭合对准误差、末端路径、空等、方向反转、距离不减步数、相对位置预测误差均保留在 final_analysis.json；逐步日志与失败的单前视相机视频在 evaluations/final_locked/。", "",
                 "![最终对照](final_comparison.png)"]
    text += ["", "## 解释边界", "",
             "- Spatial-OOD 与 Relation-OOD 仅表示本轮布局/区域/关系保留。由于按用户要求继承旧任务 checkpoint，无法排除其历史训练见过相关区域或关系，不能宣称全历史严格零样本 OOD。",
             "- B0 与 B1 的差异同时包含追加训练和新数据。本轮没有预算匹配的仅旧数据继续训练对照，不能把全部差异归因于广域数据本身。",
             "- B2 均衡采样与课程顺序使用完全相同的样本曝光集合和逐批次来源/颜色/阶段计数；课程差别是这些层内的排列顺序。其收益由开发闭环和锁定后的测试结果判断。",
             "- B4 的恢复扰动是实际物理接触滑落并重新抓取，扰动帧不作为正确动作监督。它复用已有训练布局，不增加独立布局计数；是否被纳入数据集由真实成功和接触审核决定。",
             "- 预抓取姿态标签对应当前固定工具朝向；不能据此声称学会任意 SE(3) 姿态抓取。开发集满分也不能证明未见布局满分。", "",
             f"初始模型 SHA256：{protocol['initial_checkpoint_sha256']}。冻结配置、数据清单、源码快照、模型锁、原始物理记录及哈希清单与报告一同保留。"]
    target = RUN / ("REPORT.zh-CN.md" if analysis is not None else "PROGRESS.zh-CN.md")
    target.write_text("\n".join(text)+"\n", encoding="utf-8")
    return target


def export_bundle():
    """Exclude RGB training caches/optimizer states; retain all final failure evidence."""
    assert (RUN / "final_analysis.json").exists()
    lock = read_json(RUN / "candidate_lock.json")
    paths = set()
    for path in RUN.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(RUN)
        if rel.parts[0] in ("cache", "cache_b4", "safety_data", "safety_smoke", "exports"):
            continue
        if path.name in ("export_manifest.json", "export_archive.json"):
            continue
        if path.suffix in (".npy", ".npz", ".tar", ".gz", ".pt"):
            continue
        if "evaluations" in rel.parts and "final_locked" not in rel.parts:
            if path.name not in ("metrics.json",):
                continue
        if path.suffix in (".json", ".jsonl", ".md", ".csv", ".mp4", ".png", ".log"):
            paths.add(path)
    for item in lock["models"].values():
        paths.add(ROOT / item["path"])
    safety = RUN / "safety_manifest.json"
    if safety.exists():
        representatives = {}
        for item in read_json(safety)["episodes"]:
            representatives.setdefault((item["split"], item["category"], item["regime"]), item)
        for item in representatives.values():
            directory = RUN / item["directory"]
            for name in ("metadata.json", "steps.csv", "front.mp4"):
                if (directory / name).exists():
                    paths.add(directory / name)
    for path in (ROOT / "colorlift/broad_act").glob("*.py"):
        paths.add(path)
    for name in ("final_protocol.json", "final_generation_preseal.json", "supervision_manifest.json", "public_final_manifest.json"):
        if (DATA / name).exists():
            paths.add(DATA / name)
    index = {str(path.relative_to(ROOT)): {"bytes": path.stat().st_size, "sha256": sha(path)} for path in sorted(paths)}
    manifest = RUN / "export_manifest.json"
    manifest.write_text(json.dumps({"created_at_UTC": datetime.now(timezone.utc).isoformat(), "files": index}, indent=2))
    paths.add(manifest)
    archive = RUN / "round1_results.tar.gz"
    with tarfile.open(archive, "w:gz", compresslevel=3) as tar:
        for path in sorted(paths):
            tar.add(path, arcname=str(path.relative_to(ROOT)), recursive=False)
    (RUN / "export_archive.json").write_text(json.dumps({"path": str(archive), "bytes": archive.stat().st_size,
                                                        "sha256": sha(archive), "files": len(paths)}, indent=2))


def self_test():
    rows = [{"geometry_id": str(g), "success": bool(g), "eligible": bool(g), "control_steps": 50+10*g,
             "wrong_color_grasp": False, "collision_steps": 0} for g in range(2) for _ in range(3)]
    result = summary(rows)
    assert result["episodes"] == 6 and result["independent_geometries"] == 2
    assert result["metrics"]["success"]["mean"] == .5
    assert result["metrics"]["control_steps"]["mean"] == 55
    assert paired_difference(rows, rows, "success")["difference_B_minus_A"] == 0
    assert exact_binomial(256, 256)[0] > .98
    assert exact_binomial(0, 256)[1] > 0
    assert clustered([dict(r, success=True) for r in rows], "success")["bootstrap_degenerate"]
    variants = []
    for permutation in range(3):
        palette = list(np.roll(np.array(COLORS), permutation))
        for color in COLORS:
            variants.append({"geometry_id": "one", "layout": {"colors_by_slot": palette, "color_assignment_index": permutation},
                             "target_color": color, "target_physical_slot": palette.index(color), "success": True, "eligible": True,
                             "physical_slot_positions": [[0, 0, .8], [.1, 0, .8], [0, .1, .8]], "initial_rgb_sha256": str(permutation)})
    comparisons = counterfactual(variants)
    assert comparisons["physical_XY_unchanged_within_10um"] and comparisons["same_palette_RGB_hash_identical_across_instructions"]
    assert all(item["pairs"] == 9 and item["success"]["mean"] == 1 for item in comparisons["conditions"].values())
    print("Geometry-cluster analysis checks passed", flush=True)


def main(progress=False):
    if progress:
        print(report(), flush=True)
        return
    analysis = analyze_final()
    figure(analysis)
    report(analysis)
    export_bundle()
    print("Locked final results analyzed and export archive created", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--progress", action="store_true"); parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        main(args.progress)
