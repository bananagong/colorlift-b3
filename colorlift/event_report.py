"""Produce reviewable Chinese results, physics plots and synchronized replay videos."""
import csv
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import cv2
from .event_efficiency import RUN

LABELS = {"baseline": "Baseline K=2", "single_step": "Single step K=1", "hybrid": "Hybrid 4/1/1/4", "efficiency": "Efficiency 4/2/1/4"}
ZH = {"baseline": "当前基线", "single_step": "全部单步", "hybrid": "长短结合", "efficiency": "偏向效率"}


def contact_sequence(directory, destination):
    with (directory / "steps.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    indices = [0]
    for phase in (1, 2, 3):
        indices.append(next((int(row["step"]) for row in rows if int(row["phase"]) == phase), len(rows)))
    indices += [next((int(row["step"]) for row in rows if int(row["hold_streak"]) == 1), len(rows)), len(rows)]
    reader = imageio.get_reader(str(directory / "rollout.mp4"))
    fig, axes = plt.subplots(3, 2, figsize=(10, 8.8), constrained_layout=True)
    try:
        for ax, index in zip(axes.flat, indices):
            ax.imshow(reader.get_data(index))
            if index:
                row = rows[index-1]
                force = f'L/R pre-action force: {float(row["left_pad_force_n_before"]):.1f}/{float(row["right_pad_force_n_before"]):.1f} N'
                ax.set_title(f't={index/20:.2f}s | phase {row["phase"]} | hold {row["hold_streak"]}/40\n{force}', fontsize=9)
            else:
                ax.set_title("Initial two-camera RGB", fontsize=9)
            ax.axis("off")
    finally:
        reader.close()
    fig.suptitle("Force-qualified phase transition and full 2-second hold", fontsize=14)
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def composite(sources, path, labels):
    readers = [imageio.get_reader(str(source)) for source in sources]
    lengths = [reader.count_frames() for reader in readers]
    try:
        with imageio.get_writer(str(path), fps=20, codec="libx264", quality=8) as writer:
            for index in range(max(lengths)):
                panels = []
                for reader, count, label in zip(readers, lengths, labels):
                    frame = cv2.resize(reader.get_data(min(index, count-1)), (512, 256), interpolation=cv2.INTER_NEAREST)
                    panel = np.full((304, 512, 3), 248, np.uint8)
                    panel[48:] = frame
                    suffix = " (complete)" if index >= count-1 else ""
                    cv2.putText(panel, label+suffix, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .52, (25, 25, 25), 1)
                    panels.append(panel)
                canvas = np.concatenate([np.concatenate(panels[:2], axis=1), np.concatenate(panels[2:], axis=1)], axis=0) if len(panels) == 4 else np.concatenate(panels, axis=0)
                writer.append_data(canvas)
    finally:
        for reader in readers:
            reader.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    if args.probe:
        contact_sequence(RUN / "dev/act/efficiency/scene_84000/blue", RUN / "validation/contact_sequence.png")
        print("CONTACT_SEQUENCE_READY")
        return
    metrics = json.loads((RUN / "test/metrics.json").read_text())
    audit = json.loads((RUN / "audit.json").read_text())
    manifest = json.loads((RUN / "data/manifest.json").read_text())
    from .event_labels import write_labels
    label_semantics = write_labels(manifest)
    comparison = metrics["comparison"]
    acts = {row["scheme"]: row for row in comparison if row["actor"] == "act"}
    experts = {row["scheme"]: row for row in comparison if row["actor"] == "expert"}
    pairs = {(row["actor"], row["scheme"]): row for row in metrics["paired_common_success"]}
    schemes = list(LABELS)
    colors = ["#8b98a8", "#6495ed", "#4bafa6", "#ebaa48"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    x = np.arange(4)
    success = [100*acts[s]["success_rate"] for s in schemes]
    bars = axes[0, 0].bar(x, success, color=colors, label="Task success")
    qualified = [100*acts[s]["contact_qualified_successes"]/acts[s]["trials"] for s in schemes]
    axes[0, 0].scatter(x, qualified, color="#293344", marker="D", label="Contact-qualified collision-free success", zorder=4)
    for bar, value in zip(bars, success):
        axes[0, 0].text(bar.get_x()+bar.get_width()/2, value+1, f'{value:.1f}%', ha="center", fontsize=10)
    axes[0, 0].set(ylim=(0, 113), ylabel="% of all 150 attempts", title="Frozen ACT: success and physical contact quality")
    axes[0, 0].legend(fontsize=8, loc="lower left")
    for key, color in zip(("approach", "align", "close", "lift_and_settle", "mandatory_hold"), ("#729fcf", "#e5c168", "#c283af", "#62afa0", "#d7dadf")):
        phases = {row["scheme"]: row for row in metrics["phases"] if row["actor"] == "act"}
        bottom = getattr(axes[0, 1], "_stage_bottom", np.zeros(4))
        values = np.array([phases[s][key] for s in schemes])
        axes[0, 1].bar(x, values, bottom=bottom, label=key.replace("_", " "), color=color)
        axes[0, 1]._stage_bottom = bottom+values
    axes[0, 1].set(ylabel="Actual 20Hz control steps", title="Successful trials: 40 hold steps are mandatory")
    axes[0, 1].set_ylim(0, max(axes[0, 1]._stage_bottom)+22)
    axes[0, 1].legend(fontsize=8, ncols=2, loc="upper left")
    axes[1, 0].bar(x, [acts[s]["mean_path_before_final_hold_m"] for s in schemes], color=colors)
    axes[1, 0].set(ylabel="EEF path before final hold (m)", title="Physical motion integrated at 500Hz")
    axes[1, 1].bar(x, [acts[s]["mean_queries"] for s in schemes], color=colors)
    axes[1, 1].set(ylabel="ACT forward calls per successful trial", title="Inference frequency differs from robot motion steps")
    for ax in axes.flat:
        ax.set_xticks(x, [LABELS[s].replace(" ", "\n", 1) for s in schemes], fontsize=9)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle("ColorLift: same weights, same 50 scenes, 3 color commands per scene", fontsize=15)
    fig.savefig(RUN / "ACT_efficiency_comparison.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    labels = ["Expert baseline", "Expert event", "ACT baseline", "ACT efficiency"]
    selected = [experts["baseline"], experts["efficiency"], acts["baseline"], acts["efficiency"]]
    steps = [row["mean_steps_before_final_hold"] for row in selected]
    axes[0].bar(labels, steps, color=["#8b98a8", "#4bafa6", "#8b98a8", "#ebaa48"])
    axes[0].set(ylabel="20Hz steps before final 2s hold", title="Expert vs ACT: actual movement and settling")
    for key, offset in (("mean_motion_reversals", -.24), ("mean_post_close_idle", 0), ("mean_no_progress", .24)):
        axes[1].bar(np.arange(4)+offset, [row[key] for row in selected], width=.24, label=key.replace("mean_", "").replace("_", " "))
    axes[1].set_xticks(np.arange(4), labels, rotation=12)
    axes[1].set(ylabel="Mean count / successful trial", title="Independent efficiency diagnostics")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    fig.savefig(RUN / "expert_ACT_motion.png", dpi=170)
    plt.close(fig)
    replay_dir = RUN / "highlights"
    replay_dir.mkdir(exist_ok=True)
    # Preselected first held-out scene, not a best-case search.
    composite([RUN / "test/act" / s / "scene_85000/red/rollout.mp4" for s in schemes], replay_dir / "ACT_four_schemes_red.mp4", list(LABELS.values()))
    composite([RUN / "test" / actor / s / "scene_85000/red/rollout.mp4" for actor, s in (("expert", "baseline"), ("expert", "efficiency"), ("act", "baseline"), ("act", "efficiency"))],
              replay_dir / "expert_ACT_red.mp4", ["Expert baseline", "Expert event-driven", "ACT baseline", "ACT efficiency"])
    for color in ("red", "yellow", "blue"):
        shutil.copy2(RUN / f"test/act/efficiency/scene_85000/{color}/rollout.mp4", replay_dir / f"ACT_efficiency_{color}.mp4")
    contact_sequence(RUN / "test/act/efficiency/scene_85000/red", RUN / "contact_sequence.png")
    admitted = next(v for v in manifest["episodes"] if v["admitted"])
    case = Path(admitted["candidate"]).name
    composite([RUN / admitted["candidate"] / "rollout.mp4", RUN / "replays" / case / "rollout.mp4"],
              replay_dir / "expert_candidate_and_physics_replay.mp4", ["Optimized expert candidate", "Unchanged OSC actions: fresh MuJoCo replay"])
    episodes = [json.loads(p.read_text()) for p in sorted((RUN / "test").glob("*/*/scene_*/*/metadata.json"))]
    failures = [e for e in episodes if not e["success"]]
    (RUN / "failure_cases.json").write_text(json.dumps(failures, indent=2))
    failure_dir = replay_dir / "failures"
    failure_dir.mkdir(exist_ok=True)
    for failed in failures:
        shutil.copy2(RUN / failed["directory"] / "rollout.mp4", failure_dir / f'{failed["actor"]}_{failed["scheme"]}_{failed["seed"]}_{failed["target_color"]}.mp4')
    baseline_failures = [e for e in failures if e["actor"] == "act" and e["scheme"] == "baseline"]
    if baseline_failures:
        first = baseline_failures[0]
        composite([RUN / f'test/act/{s}/scene_{first["seed"]}/{first["target_color"]}/rollout.mp4' for s in schemes],
                  replay_dir / "first_baseline_failure_four_schemes.mp4", list(LABELS.values()))
    lines = ["# ColorLift 事件驱动阶段与动作段效率实验", "", "完成日期：2026-10-29。所有结果来自 catlab 上的真实 MuJoCo 闭环执行。",
             "", "## 实验口径", "", "固定当前三色扰动恢复 ACT checkpoint（预测容量 4），不进行新的梯度训练。本轮没有注入外部3cm偏移。正式测试使用未参与开发的 50 个场景（85000–85049），每个场景分别指定红、黄、蓝，每种方案 150 次。专家两组、ACT 四组，共 900 次。GPU 只使用物理卡 1 和 2。",
             "", "成功条件保持原样：指定颜色方块相对初始高度抬升 >0.04m，正确抓取、线速度≤0.025m/s、角速度≤0.4rad/s、漂移≤5mm，连续 40 个 20Hz 控制步（2秒）。碰撞按 500Hz 物理子步检测。原任务允许接触 2 步宽限；示范入库额外要求最后 40 步每步都检测到目标抓取。",
             "", "## 四种 ACT 方案", "", "|方案|四阶段执行长度|切换方式|", "|---|---|---|",
             "|当前基线|2/2/2/2|保留原 RecoveryExecutor：最少3步接近/对准，手指静止判定抬升|",
             "|全部单步|1/1/1/1|事件切换，真实接触后抬升|",
             "|长短结合|4/1/1/4|事件切换，自由空间长段、接触单步|",
             "|偏向效率|4/2/1/4|同样的接触门控，水平误差≤8mm且高度≤0.890m时提前进入下降对准|",
             "", "事件方案没有最少阶段驻留次数。标准接近/精确对准阈值为 XY≤4.5mm、Z≤5mm。夹紧→抬升要求双侧指垫接触同一外部动态物体，两侧法向力均≥0.2N，夹爪开口12–75mm、手指速度≤0.02m/s。阶段事件立即打断缓存动作段。空夹超时仅用于判失败，不能用来触发抬升。",
             "", "本评测统一空夹失败终止为：进入夹紧或抬升阶段后，当前阶段累计≥12步且开口<6mm。这与旧恢复评测脚本按累计闭合步数终止的细节不同；基线保留原阶段适配器和执行长度，失败终止时间不作为节省步数的证据，效率比较只使用真实成功轨迹，并另做共同成功案例配对。",
             "", "ACT 的目标 XY 仍来自 RGB、可学习颜色嵌入和机器人本体状态；事件适配器增加模拟触觉信号，不读取目标真值位置、颜色 body ID 或 HSV。传感器用接触对的 body 相等关系合并两侧接触，控制不解释该 body 的颜色。专家使用初始双相机 HSV 定位。目标真值只进入评分、接触审计和统计。夹紧后锁定 XY，抬升高度由四阶段 OSC 适配器给定；模型预测的阶段只记作诊断。因此这里比较的是 ACT + 阶段适配器系统，并非端到端关节/力矩策略；夹紧、保持期间减少模型调用本身不代表实际运动减少。",
             "", "基线与事件组同时改变了门控方式；全部单步和长短结合共享同一个事件控制器，可以直接比较执行长度。偏向效率还包含提前下降规则，其效果不能全部归因于 chunk 长度。当前最大预测容量只有4，未把旧预测重复填充成长段。",
             "", "## 正式结果", "", "|ACT方案|成功|接触合格且无碰撞|成功轨迹平均总步数|进入最终保持前步数|500Hz末端路程(m)|模型调用次数|", "|---|---:|---:|---:|---:|---:|---:|"]
    for scheme in schemes:
        r = acts[scheme]
        lines.append(f'|{ZH[scheme]}|{r["successes"]}/150|{r["contact_qualified_successes"]}/150|{r["mean_steps_success"]:.2f}|{r["mean_steps_before_final_hold"]:.2f}|{r["mean_physical_path_m"]:.4f}|{r["mean_queries"]:.2f}|')
    old, new = acts["baseline"], acts["efficiency"]
    saved = old["mean_steps_success"]-new["mean_steps_success"]
    before_hold_saved_percent = 100*saved/old["mean_steps_before_final_hold"]
    path_saved_percent = 100*(1-new["mean_physical_path_m"]/old["mean_physical_path_m"])
    lines += ["", f'本轮优先采用偏向效率方案：三色各50/50成功，成功轨迹平均减少{saved:.2f}步（{saved/20:.3f}秒），最终保持前步数减少{before_hold_saved_percent:.2f}%。但末端总路程仅减少{path_saved_percent:.2f}%，改善主要是减少阶段停顿；不能称为大幅缩短运动路径。接触合格且无碰撞成功与基线同为147/150，偏向效率仍有3个碰撞回合，基线有2个；运动反向均值也从0.19略升到0.23。']
    lines += ["", "总步数包含不可省略的 40 步稳定保持。路程由每个0.002秒物理子步的末端位移积分，不等于20Hz采样点之间的直线长度，也不等于模型调用次数。失败轨迹不算作效率收益。不同组成功率不同时，另附所有方案共同成功案例的配对统计。", "", "|ACT方案|共同成功案例|比基线节省步数|按场景聚类bootstrap 95%区间|", "|---|---:|---:|---|"]
    for scheme in schemes:
        r = pairs["act", scheme]
        lines.append(f'|{ZH[scheme]}|{r["common_success_cases"]}|{r["mean_steps_saved"]:.2f}|[{r["saved_ci95_low"]:.2f}, {r["saved_ci95_high"]:.2f}]|')
    lines += ["", "## 专家与 ACT 的真实运动", "", "|控制器|成功|接触合格且无碰撞|总步数|最终保持前步数|末端路程(m)|阶段内运动反向|夹紧后空等|无明显距离进展|", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, row in zip(labels, selected):
        lines.append(f'|{name}|{row["successes"]}/150|{row["contact_qualified_successes"]}/150|{row["mean_steps_success"]:.2f}|{row["mean_steps_before_final_hold"]:.2f}|{row["mean_physical_path_m"]:.4f}|{row["mean_motion_reversals"]:.2f}|{row["mean_post_close_idle"]:.2f}|{row["mean_no_progress"]:.2f}|')
    lines += ["", "方向反向、夹紧后空等、无进展沿用原诊断定义：位移有效阈值0.5mm、夹角>100°、间隔≤3步；夹紧后空等排除有效保持；接近/对准阶段5步距离改善<1mm且距目标>12mm记为无进展。每个案例有完整逐步CSV和前70步构成，不能仅依据平均步数判断动作质量。",
              "", "## 高效专家数据集", "", f'共生成210条候选（60个训练场景×3色、10个验证场景×3色）。最终收录 {manifest["admitted"]} 条，拒绝 {manifest["rejected"]} 条；训练 {manifest["train"]}、验证 {manifest["val"]}。每条收录数据都满足原成功标准、抬升前正确目标的双侧力接触、无异常碰撞、最后40步持续目标接触，并在相同初态下逐步重放原OSC动作后再次通过。',
              "", "原始数据包括两路 RGB、HSV mask 标签、机器人本体/阶段状态、目标颜色、阶段路标、实际OSC动作、逐步位置/接触/力/执行事件、候选与重放视频、SHA256。动作以float64保存以确保复现；宏标签与状态可为float32。候选放在 candidates/，重放放在 replays/，只有通过审计的数据进入 data/，没有用时间压缩、插值、删帧替代真实执行。另提供 training_macro_labels.npz：接近/对准阶段使用初始双相机HSV目标XY和固定阶段Z，避免把临时避让路标误当物块目标；夹紧/抬升保留实际锁定的XY。原始记录不改。",
              "", f'审计通过：正式 {audit["formal_trials"]} 条、配对初态 {audit["paired_initial_scenes"]} 组；模型与控制器哈希未变化。最大重放机器人/末端状态误差 {audit["maximum_replay_state_error"]:.8g}，目标位置误差 {audit["maximum_replay_target_position_error_m"]:.8g}m。50个场景各三种颜色命令的初始RGB完全相同。数据集未用于本轮固定权重 ACT 的训练，因此不能将执行效率提升描述为重新训练的模型收益。',
              "", "服务器完整数据路径：/home/catlab/mujoco_robosuite/experiments/colorlift_event_efficiency_v1/data。正式测试 test/metrics.json、comparison.csv、paired_common_success.csv、phase_breakdown.csv，单案例 steps.csv 和 metadata.json 保存全部数值。",
              "", "数据格式：states[T,19]（7关节+2指位+3末端位置+4阶段onehot+3目标颜色onehot）、images/masks[T,2,128,128,3]、actions[T,7]（世界XYZ宏目标+4阶段onehot）、osc_actions[T,7]（实际归一化OSC平移、旋转、夹爪命令，float64）。颜色也保存在manifest元数据；保持按场景拆分训练/验证，动作chunk不得跨回合。",
              "", f'全部 {len(failures)} 个正式失败回合保留在 failure_cases.json 和 highlights/failures/，另提供首个基线失败场景的四方案同步回放（如基线存在失败）。图示与主回放固定使用第一个正式场景85000，未搜索最好看的成功案例。',
              "", "## 与参考研究的关系", "", "参考 [Adaptive Action Chunking for Robotic Imitation Learning](https://pmc.ncbi.nlm.nih.gov/articles/PMC13204913/) 提出根据视觉上下文学习 chunk 长度预测头。本实验借鉴其按阶段平衡反馈频率与连贯性的动机，先固定模型验证显式事件门控；没有实现该论文的可学习长度头，也未把本实验当作其复现。",
              "", "![ACT四方案](ACT_efficiency_comparison.png)", "", "![专家与ACT运动](expert_ACT_motion.png)"]
    (RUN / "REPORT.zh-CN.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("REPORT_READY", RUN, flush=True)


if __name__ == "__main__":
    main()
