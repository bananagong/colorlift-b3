"""Measured execution-length, expert-data and fresh recovery-model report."""
from collections import Counter
import csv
import json
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import cv2
from .config import ROOT, COLORS
from .step_recording import save_rows

RUN=ROOT / "experiments/colorlift_recovery_v1"
ORIGINAL=ROOT / "experiments/colorlift_conditional_v1"
TITLES={"old_long": "Original ACT, execute 12", "old_short": "Original ACT, execute 2 + recovery adapter", "new_recovery": "Recovery ACT, chunk 4 / execute 2"}


def load(path): return json.loads(Path(path).read_text())


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig") as stream: return list(csv.DictReader(stream))


def comparison_video(paths, labels, output):
    sequences=[]
    for path in paths:
        reader=imageio.get_reader(str(path))
        sequences.append([frame for frame in reader]); reader.close()
    with imageio.get_writer(str(output), fps=20, codec="libx264", quality=7) as writer:
        for step in range(max(map(len, sequences))):
            views=[]
            for sequence, label in zip(sequences, labels):
                image=sequence[min(step, len(sequence)-1)].copy()
                cv2.putText(image, label, (2, 24), cv2.FONT_HERSHEY_SIMPLEX, .32, (0, 0, 0), 1)
                views.append(image)
            writer.append_data(np.concatenate(views, axis=1))


def plots(metrics, sweep, training):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes=plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    names=("h1", "h2", "h3_original", "h6", "h12", "adaptive")
    labels=("1", "2", "3", "6", "12", "4/1/1/3")
    byname={row["mode"]: row for row in sweep["summary"]}
    axes[0].bar(labels, [byname[name]["successes"]/1.5 for name in names], color="#4c78a8", alpha=.8)
    axes[0].set(ylim=(75, 103), xlabel="Executed action prefix (frozen checkpoint)", ylabel="Strict success (%)", title="Shorter execution alone is not a fix")
    for i, name in enumerate(names): axes[0].text(i, byname[name]["successes"]/1.5+.5, str(byname[name]["successes"])+"/150", ha="center", fontsize=8)
    for ax, condition in zip(axes[1:], ("clean", "disturbed")):
        group=metrics["summary"][condition]
        for i, variant in enumerate(("old_long", "old_short", "new_recovery")):
            row=group[variant]
            ax.bar(i, row["success_rate"]*100, color=("#9aa5b1", "#72b7b2", "#54a24b")[i], alpha=.85)
            ax.bar(i, row["collision_free_successes"]/row["episodes"]*100, facecolor="none", edgecolor=".3", hatch="//", linewidth=.5)
            ax.text(i, row["success_rate"]*100+1, f'{row["successes"]}/{row["episodes"]}', ha="center", fontsize=9)
        ax.set(xticks=range(3), xticklabels=("Old, 12", "Old, 2 + guard", "New, 2 + guard"), ylim=(0, 110), ylabel="Success (%)",
               title="Fresh clean scenes" if condition=="clean" else "Fresh scenes: two 3 cm offsets")
        ax.tick_params(axis="x", labelrotation=14)
        ax.text(.02, .02, "Hatched area: collision-free success", transform=ax.transAxes, fontsize=8)
    fig.savefig(RUN / "comparison.png", dpi=160); plt.close(fig)
    categories=("Approach", "Precise align", "Closing/contact", "Recovery", "Lift/hold")
    samples=training["sampling_counts"]
    counts=np.array([samples[key] for key in ("approach", "precise_align", "closing_contact", "recovery", "lift_hold")])
    fig, axes=plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    x=np.arange(5)
    axes[0].bar(x-.18, [25, 25, 25, 0, 25], .36, label="Old stage-balanced sampler", color="#9aa5b1")
    axes[0].bar(x+.18, counts/counts.sum()*100, .36, label="Actual recovery-training samples", color="#54a24b")
    axes[0].set(xticks=x, xticklabels=categories, ylabel="Sampling probability (%)", ylim=(0, 36), title="90% of samples target alignment/contact/recovery")
    axes[0].tick_params(axis="x", labelrotation=16); axes[0].legend(fontsize=8, frameon=False)
    probe=load(RUN / "validation_target_probe.json")["summary"]
    axes[1].bar(np.array([0, 1])-.18, [probe["old"][key]*1000 for key in ("before", "after")], .36, label="Original ACT", color="#9aa5b1")
    axes[1].bar(np.array([0, 1])+.18, [probe["new"][key]*1000 for key in ("before", "after")], .36, label="Recovery ACT", color="#54a24b")
    axes[1].set(xticks=[0, 1], xticklabels=["Before offset", "After offset"], ylabel="Predicted target XY error (mm)", title="Held-out expert states: same observations for both models")
    axes[1].legend(frameon=False, fontsize=8)
    fig.savefig(RUN / "sampling_and_target_error.png", dpi=160); plt.close(fig)
    pilot=load(RUN / "pilot/pilot_77000_red/metadata.json")
    rows=read_rows(RUN / "pilot/pilot_77000_red/steps.csv")
    step=np.arange(1, len(rows)+1)
    fig, axes=plt.subplots(2, 1, figsize=(10, 6), sharex=True, constrained_layout=True)
    axes[0].plot(step, [float(row["eef_after_z"]) for row in rows], label="EEF z")
    axes[0].plot(step, [float(row["target_after_z"]) for row in rows], label="Cube z")
    axes[0].plot(step, [float(row["goal_z"]) for row in rows], "--", label="Expert IK waypoint z")
    axes[0].set(ylabel="World z (m)", title="Scripted expert: physical offsets, IK recovery, grasp and hold")
    axes[0].legend(frameon=False, ncol=3)
    axes[1].plot(step, [np.linalg.norm([float(row["eef_after_"+a])-float(row["target_after_"+a]) for a in "xy"])*1000 for row in rows], label="EEF-to-target XY distance")
    axes[1].plot(step, [float(row["gripper_gap_m"])*1000 for row in rows], label="Finger gap")
    for event in pilot["noise_events"]:
        for ax in axes: ax.axvspan(event["start_step"], event["end_step"], color="#e45756", alpha=.15)
    axes[1].set(xlabel="20 Hz control step (red shading = excluded disturbance actions)", ylabel="Distance (mm)")
    axes[1].legend(frameon=False); fig.savefig(RUN / "expert_recovery_trace.png", dpi=160); plt.close(fig)


def main():
    assert load(RUN / "evaluation_completed.json")["complete"]
    metrics=load(RUN / "evaluations/test/metrics.json")
    sweep=load(RUN / "execution_sweep/comparison.json")
    data=load(RUN / "data_audit.json")
    training=load(RUN / "model/completed.json")
    protocol=load(RUN / "protocol.json")
    regression=load(RUN / "old_failures_replay.json")
    plots(metrics, sweep, training)
    shutil.copy2(RUN / "pilot/pilot_77000_red/rollout.mp4", RUN / "expert_noise_recovery.mp4")
    repaired=next((row for row in regression["trials"] if row["success"]), regression["trials"][0])
    comparison_video([ORIGINAL / "eval_test_selected" / f'scene_{repaired["seed"]}' / repaired["target_color"] / "rollout.mp4", RUN / repaired["directory"] / "rollout.mp4"],
                     ["Original: failed", "Recovery-trained ACT"], RUN / "old_failure_before_after.mp4")
    stress=metrics["trials"]
    newcases=[row for row in stress if row["disturbed"] and row["variant"]=="new_recovery" and row["success"]]
    example=next((row for row in newcases if any(old["variant"]=="old_short" and old["disturbed"] and old["seed"]==row["seed"] and old["target_color"]==row["target_color"] and not old["success"] for old in stress)), newcases[0])
    old=next(row for row in stress if row["variant"]=="old_short" and row["disturbed"] and row["seed"]==example["seed"] and row["target_color"]==example["target_color"])
    comparison_video([RUN / old["directory"] / "rollout.mp4", RUN / example["directory"] / "rollout.mp4"], ["Original weights, same adapter", "Recovery-trained weights, same adapter"], RUN / "disturbed_old_vs_new.mp4")
    failed=[row for row in metrics["trials"] if row["variant"]=="new_recovery" and not row["success"]]
    (RUN / "new_model_failure_cases.json").write_text(json.dumps(failed, indent=2))
    lines=["# 带偏移专家、短 chunk 与 ACT 恢复训练", "", "本实验先冻结原 ACT 权重做执行长度对照，再采集真实物理偏移后的专家恢复数据，缩短训练动作块并提高精确对准、夹紧和恢复阶段的采样概率。历史诊断集和最终新场景评测分开。", "",
           "## 冻结权重的执行长度对照", "", "同一 checkpoint、相同50场景与三种颜色，共每组150次；OSC FastExecutor 与成功判定不变。3步原始组复用已保存的完整正式记录，其他组重新执行，初始图像哈希和物体位置逐一匹配。", "",
           "| 每次执行步数 | 成功 | 无碰撞成功 | 救回原失败 | 新增失败 |", "|---|---:|---:|---:|---:|"]
    byname={row["mode"]: row for row in sweep["summary"]}
    for name, label in (("h1", "1"), ("h2", "2"), ("h3_original", "3（原设置）"), ("h6", "6"), ("h12", "12"), ("adaptive", "自由4/对准1/夹紧1/抬升3")):
        row=byname[name]; lines.append(f'| {label} | {row["successes"]}/150 | {row["collision_free_successes"]}/150 | {row["rescued_baseline_failures"]} | {row["regressed_baseline_successes"]} |')
    lines.extend(["", "本批场景中，单纯缩短执行长度没有提高成功率：1步反而增加失败，12步的点估计最高。12步相对原3步的场景级配对bootstrap区间包含0，不能据此推断长执行普遍更优。完整配对变化、置信区间和回合记录在 execution_sweep/。", "",
                  "在原5个进入夹紧但失败的回合，夹紧前末端与物块中心的XY距离平均约3.40cm；142个成功回合约1.23cm，另外3个失败未进入夹紧。该距离采用记录的末端site与物块body中心，不等同于手指接触面间隙。执行器进入夹紧后会锁XY，刷新得更快也不能自动消除已经锁住的偏差。", "",
                  "## 带噪声 Scripted Expert", "",
                  f'共采集 {data["attempted_episodes"]} 个回合，{data["successful_episodes"]} 个严格成功：新增每种颜色80条训练、15条验证示范，原540条干净训练和90条验证继续保留。场景种子训练78000–78079、验证79000–79014，均与新正式测试分开。', "",
                  f'每回合最多两次随机水平偏移，目标幅度3cm，间隔至少8个专家控制步；共 {data["events"]} 次，实际平均 {data["actual_offset_mean_m"]*100:.3f}cm，范围 {data["actual_offset_min_m"]*100:.3f}–{data["actual_offset_max_m"]*100:.3f}cm。Z不加随机偏移。', "",
                  "偏移通过正常OSC动作在物理仿真中执行，没有直接改qpos或瞬移机器人。专家每个控制步从当前关节状态重新求6D阻尼最小二乘IK，在独立MjData中计算可达关节姿态与FK位点，再用OSC跟踪。对准偏移后先抬高、重新水平对齐、再下降抓取，并根据物块当前位置重算目标。专家允许读取仿真目标真值作为标签；学生推理不读取。", "",
                  f'最大IK位置残差 {data["max_IK_position_residual_m"]*1000:.3f}mm。总控制步 {data["total_control_steps"]}，其中 {data["excluded_injection_steps"]} 个外部扰动步全部屏蔽，留下 {data["supervised_corrective_steps"]} 个可监督步，包含 {data["recovery_steps"]} 个恢复步。chunk的监督不跨越扰动区间，避免用扰动前图像监督不可预知的扰动后动作。', "",
                  "![专家扰动与纠偏轨迹](expert_recovery_trace.png)", "", "专家演示：`expert_noise_recovery.mp4`。原始每步末端/目标位置、双指状态、OSC命令、IK解和残差、扰动前后位移都保存在各回合 steps.csv。", "",
                  "## 训练目标与采样", "",
                  "ACT最大查询数从12裁为4，保留原权重其余部分和归一化统计。对准、夹紧、恢复样本只监督前2步，接近及抬升保持监督4步；本轮实际每2步重新观测和规划。没有把原权重下表现较差的1步设置强行用作改进结论。", "",
                  "采样概率：接近5%、精确对准30%、夹紧接触30%、恢复30%、抬升保持5%，三种颜色均衡；恢复样本来自新增数据，其他阶段在新旧数据间各取约一半。旧模型原先四阶段各25%。实际采样计数保存在 model/completed.json。", "",
                  "新增数据的训练标签采用最终抓取宏目标，临时IK避让路标与实际OSC动作另外保留。phase0/1的目标XY取该控制区间开始时的物块位置，名义高度分别0.865/0.818m。这样不会把“暂时退到当前位置”和“最终抓取目标”混在一起。原始trajectory.npz不覆盖，入训宏标签单独导出为training_macro_labels.npz。", "",
                  "开发阶段曾发现直接监督临时路标会让模型停在抬高位置；该首版模型与开发记录保留在 model_pre_macro_labels、evaluations/dev_pre_macro_labels，不参与最终分数。最终从未改动的原ACT重新微调5000次：batch36，ACT学习率3e-5，识别/颜色嵌入1e-5，ResNet3e-6；XYZ权重2:2:1，阶段CE0.25，KL10，识别CE0.5，按固定验证集先验损失选权重。", "",
                  "恢复执行器只根据ACT预测目标XY与机器人末端误差处理通用避让：较大偏差时先抬高再对齐，已对齐则下降。它不读取目标真值或HSV。旧短执行模型与新模型共用完全相同的执行器，因此该对照比较的是再训练效果；相对于原长执行模型的整体对照还同时包含执行规则变化。", "",
                  "![采样分布和扰动前后预测](sampling_and_target_error.png)", "", "验证专家状态上的目标XY预测误差：旧模型扰动前约3.75mm、后约23.96mm；新模型前约4.78mm、后约7.90mm。新模型的干净状态误差并未在此诊断中降低，主要改善出现在离轨状态；该指标不是闭环抓取成功率。", "",
                  "## 全新场景最终对照", "", "干净测试81000–81049，50场景×3颜色，每模型150次；扰动测试82000–82029，30场景×3颜色，每模型90次。成功判定均为正确物块抬升>4cm并连续稳定保持2s，其他速度、接触、漂移阈值不变。干净回合180步上限，扰动回合为240步，所有对照模型一致。", "",
                  "测试时IK只执行外部随机偏移，不把机械臂带回目标；偏移结束后，各模型都重新观测，回归抓取由ACT与其执行器完成。原权重、微调权重、控制器和协议的SHA256在评估前冻结，评估后核对未变。所有失败均计入，没有自动重试。", "",
                  "| 场景 | 系统 | 成功 | 无碰撞成功 | 碰撞回合 | 抓错颜色 | 成功平均步数 |", "|---|---|---:|---:|---:|---:|---:|"])
    chinese={"old_long": "原ACT，执行12步，原执行器", "old_short": "原ACT，执行2步，恢复执行器", "new_recovery": "恢复训练ACT，执行2步，同恢复执行器"}
    for condition, label in (("clean", "干净"), ("disturbed", "两次3cm偏移")):
        for variant in ("old_long", "old_short", "new_recovery"):
            row=metrics["summary"][condition][variant]
            lines.append(f'| {label} | {chinese[variant]} | {row["successes"]}/{row["episodes"]}（{row["success_rate"]*100:.2f}%） | {row["collision_free_successes"]}/{row["episodes"]} | {row["collision_episodes"]} | {row["wrong_color_episodes"]} | {row["mean_success_steps"]:.2f} |')
    lines.extend(["", "![固定权重对照与全新场景结果](comparison.png)", "", "新模型分颜色结果：", "", "| 颜色 | 干净场景 | 扰动场景 |", "|---|---:|---:|"])
    for color in COLORS:
        clean=metrics["summary"]["clean"]["new_recovery"]["per_color"][color]
        noisy=metrics["summary"]["disturbed"]["new_recovery"]["per_color"][color]
        lines.append(f'| {color} | {clean["successes"]}/{clean["episodes"]} | {noisy["successes"]}/{noisy["episodes"]} |')
    for condition in ("clean", "disturbed"):
        paired=metrics["paired_comparisons"][condition+"_vs_old_short"]
        lines.extend(["", f'{condition}条件下，同一执行器和执行长度的新旧模型配对：救回 {paired["rescued"]} 次、退化 {paired["regressed"]} 次；成功率差 {paired["success_delta"]*100:.2f} 个百分点，场景级bootstrap95%区间 [{paired["scene_bootstrap_95_ci"][0]*100:.2f}, {paired["scene_bootstrap_95_ci"][1]*100:.2f}]。'])
        if condition=="disturbed": lines.append(f'双方都完成两次偏移的配对子集有 {paired["both_received_two_offsets"]} 次，旧/新成功分别 {paired["two_offset_matched_old_successes"]}/{paired["two_offset_matched_new_successes"]}。全体结果仍保留未完成第二次偏移就失败的回合。')
    lines.extend(["", f'对先前8个失败案例的另行回放，新模型救回 **{regression["rescued"]}/8**，不计入上述全新场景分数。对照视频：`old_failure_before_after.mp4`；扰动下同执行器新旧模型视频：`disturbed_old_vs_new.mp4`。', "",
                  "本实验同时改变了新增离轨数据、宏目标标签、阶段采样和训练chunk，没有做逐因素训练消融，不能把全部变化单独归于某一个因素。原模型12步是强基线，不用只比较原3步来放大收益。", "",
                  "## 文件", "", "- `model/policy.pt`：最终短chunk恢复ACT，原模型未覆盖。", "- `execution_sweep/comparison.csv`：固定旧权重执行长度配对结果。", "- `evaluations/test/comparison.csv`、`episodes.csv`、`switching.csv`、`metrics.json`：720次新场景对照。", "- `new_model_failure_cases.json`：新模型所有失败，不删去重试。", "- `data/manifest.json`、`data_audit.json`、`label_semantics.json`：285条专家轨迹、标签与独立检查。", "- `data/*/trajectory.npz`：服务器完整RGB/HSV标签/本体状态/实际OSC动作与IK解；`training_macro_labels.npz`为入训宏目标；CSV和视频保留全过程。", "- `test_audit.json`与`final_freeze.json`：轨迹连续性、严格成功保持、初态一致和冻结哈希核验。", "",
                  "服务器目录：`/home/catlab/mujoco_robosuite/experiments/colorlift_recovery_v1`。最多使用GPU1、6两张卡。所有指标均针对当前固定相机、光照、初始机器人姿态和三种颜色方块；结果不外推到未测试的扰动类型。", ""])
    (RUN / "REPORT.zh-CN.md").write_text("\n".join(lines), encoding="utf-8")
    (RUN / "completed.json").write_text(json.dumps({"complete": True, "expert_episodes": 285, "fresh_test_episodes": 720, "old_failure_replays": 8}, indent=2))
    print("RECOVERY REPORT COMPLETE")


if __name__ == "__main__": main()
