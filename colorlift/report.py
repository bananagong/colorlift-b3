"""Measured ColorLift ablation report, diagnostic tables, plots, and inference exports."""
import csv
import argparse
from dataclasses import asdict
import hashlib
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .config import Config, EXPERIMENT, PHASES
from .vision import HSV_RANGES

NAMES = {"hsv": "HSV+OSC", "rgb": "RGB-ACT", "mask": "Mask-ACT", "rgb_mask": "RGB+Mask-ACT"}
DESCRIPTIONS = {
    "control_steps": "真实20Hz物理控制步数，包含必要的2秒保持；不将4个宏动作伪装成4个控制步",
    "command_reversals_within_phase": "同一阶段内两次非零OSC平移方向夹角>100度且间隔<=3步的次数；位移<0.5mm不参与",
    "command_reversals_all": "跨阶段也计入的命令方向反转，包含下降转抬升等有意改变",
    "motion_reversals_within_phase": "同一阶段内实际末端位移的反转次数，使用相同角度、幅度和时间阈值",
    "post_close_idle_steps": "关闭命令已至少执行8步，夹爪停止变化且末端移动<0.5mm，同时不在有效保持计时中的步数",
    "empty_gripper_wait_steps": "上述空等中夹爪总开口<6mm的子集",
    "no_progress_steps": "仅接近/对准阶段，末端距目标>12mm，过去5步距离改善<1mm的步数；含距离增加",
    "necessary_hold_steps": "成功回合固定40步，等于20Hz下2秒稳定保持",
    "non_mandatory_control_steps": "总控制步数减去本回合必要的40步；失败回合不扣除保持预算",
    "holding_failure_events": "已开始有效保持后，因高度、速度、漂移或抓取条件失效而重置计时的事件数",
    "collision_steps": "一个控制区间中任一500Hz物理子步出现超阈值非预期接触的控制步数",
    "collision_events": "连续有碰撞的控制步作为一个事件；不是接触点数量",
    "collision_physics_steps": "500Hz物理子步中出现超阈值非预期接触的次数",
    "wrong_color_grasp": "对非目标物块双指抓取连续达到3个控制采样；检测后终止并判失败",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-export", action="store_true")
    args = parser.parse_args()
    variants = list(NAMES)
    metrics = [json.loads((EXPERIMENT / v / "eval_test" / "metrics.json").read_text()) for v in variants]
    cfg = Config.load(EXPERIMENT / "config.json")
    for m in metrics:
        if "mean_non_mandatory_success_steps" not in m:
            m["mean_non_mandatory_success_steps"] = None if m["mean_success_steps"] is None else m["mean_success_steps"] - cfg.hold_steps
    protocol = json.loads((EXPERIMENT / "protocol.json").read_text())
    reference = metrics[0]["trials"]
    for m in metrics:
        assert [r["seed"] for r in m["trials"]] == protocol["test_seeds"]
        for a, b in zip(reference, m["trials"]):
            assert a["initial_positions"] == b["initial_positions"]
    scalar_keys = [key for key, value in metrics[0].items() if isinstance(value, (str, int, float)) or value is None]
    with (EXPERIMENT / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=scalar_keys)
        writer.writeheader()
        writer.writerows({key: m.get(key) for key in scalar_keys} for m in metrics)
    (EXPERIMENT / "metric_dictionary.zh-CN.json").write_text(json.dumps(DESCRIPTIONS, ensure_ascii=False, indent=2), encoding="utf-8")
    (EXPERIMENT / "hsv_ranges.json").write_text(json.dumps(HSV_RANGES, indent=2))
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    colors = ["#888888", "#2276b7", "#e67e22", "#348653"]
    x = np.arange(4)
    names = [NAMES[v].replace("-ACT", "\nACT") for v in variants]
    for i, (m, color) in enumerate(zip(metrics, colors)):
        value = 100 * m["success_rate"]
        lo, hi = np.array(m["wilson_95_ci"]) * 100
        axes[0].bar(i, value, color=color)
        axes[0].errorbar(i, value, yerr=[[max(0, value - lo)], [max(0, hi - value)]], fmt="none", color="black", capsize=4)
        axes[0].text(i, hi + 3, f'{m["successes"]}/{m["episodes"]}', ha="center")
    axes[0].set(xticks=x, xticklabels=names, ylim=(0, 120), ylabel="Success (%)", title="Same 30 held-out scenes\n95% Wilson intervals")
    axes[0].set_yticks([0, 25, 50, 75, 100])
    overhead = [m["mean_non_mandatory_success_steps"] or 0 for m in metrics]
    hold = [40 if m["successes"] else 0 for m in metrics]
    axes[1].bar(x, overhead, color=colors, label="Movement / other steps")
    axes[1].bar(x, hold, bottom=overhead, color="#d9e1e8", label="Required 2s hold (40)")
    axes[1].set(xticks=x, xticklabels=names, ylabel="20Hz control steps", title="Mean steps in successful episodes")
    axes[1].legend(fontsize=8)
    for m, color in zip(metrics[1:], colors[1:]):
        rows = [json.loads(line) for line in (EXPERIMENT / m["variant"] / "metrics.jsonl").read_text().splitlines()]
        rows = [r for r in rows if "val_score" in r]
        axes[2].plot([r["step"] for r in rows], [r["val_score"] for r in rows], marker="o", ms=3, label=NAMES[m["variant"]], color=color)
    axes[2].set(xlabel="Updates", ylabel="XYZ L1 + 0.25 phase CE", title="Independent validation")
    axes[2].legend(fontsize=8)
    fig.savefig(EXPERIMENT / "comparison.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
    for ax, key, label in zip(axes, ["mean_command_reversals_within_phase", "mean_post_close_idle_steps", "mean_no_progress_steps"], ["Direction reversals", "Post-close idle steps", "No-progress steps"]):
        ax.bar(x, [m[key] for m in metrics], color=colors)
        ax.set(xticks=x, xticklabels=names, ylabel="Mean per episode", title=label)
    axes[0].clear()
    command_values = [m["mean_command_reversals_within_phase"] for m in metrics]
    motion_values = [m["mean_motion_reversals_within_phase"] for m in metrics]
    axes[0].bar(x - .16, command_values, width=.3, color="#aab2b8", label="OSC command")
    axes[0].bar(x + .16, motion_values, width=.3, color=colors, label="End-effector motion")
    axes[0].set(xticks=x, xticklabels=names, ylabel="Mean per episode", title="Within-phase direction reversals", ylim=(0, max(.25, max(command_values + motion_values) * 1.4)))
    axes[0].legend(fontsize=8)
    fig.savefig(EXPERIMENT / "diagnostics.png", dpi=170)
    plt.close(fig)

    for variant in ([] if args.skip_export else variants[1:]):
        root = EXPERIMENT / variant
        ckpt = torch.load(root / "best.pt", map_location="cpu", weights_only=False)
        payload = {key: ckpt[key] for key in ("model", "stats", "config", "step", "val_score")}
        payload["task_config"] = asdict(cfg)
        torch.save(payload, root / "policy.pt")
        with (root / "policy.pt").open("rb") as f:
            digest = hashlib.file_digest(f, "sha256").hexdigest()
        (root / "policy.sha256").write_text(digest + "  policy.pt\n")
    data = json.loads((EXPERIMENT / "data" / "manifest.json").read_text())
    env_check = json.loads((EXPERIMENT / "environment" / "verification.json").read_text())
    lines = ["# ColorLift：HSV 与三种 ACT 输入结构对照", "",
             "## 环境与成功定义", "",
             "robosuite 环境名为 `ColorLift` / `colorlift`，使用 Panda、OSC_POSE、红黄蓝三个4cm/64g刚体。每回合三块均随机平移、随机yaw，并打乱采样颜色顺序；中心距离至少95mm，另用保守圆形包围界检查不重叠。", "",
             "默认目标是红色。高度以本回合该物块初始中心高度为基准，**抬升量严格大于0.04m**，并有正确双指抓取接触；在目标线速度≤0.025m/s、角速度≤0.4rad/s、相对保持起点漂移≤5mm的条件下，连续保持2s。20Hz计时需40个有效控制步；接触允许最多2步检测抖动宽限。保持条件失效后计时清零。", "",
             "抓错颜色即失败。目标指尖抓取和物块静置桌面接触属于预期接触；机器人碰桌面/干扰块、非指尖碰目标、自碰撞及物块互碰分别记录。机器人相关阈值5N、物块互碰1N，按500Hz物理子步检测；碰撞作为独立指标，另外报告无碰撞成功率。", "",
             f'本次模型编译的body ID：`{env_check["checks"]["randomization_body_ids_rgb_hsv"]["body_ids"]}`。环境提供 `color_body_ids`，每次模型重建都会刷新；每回合元数据保存ID、body名称与初始位置。', "",
             "RGB 与机器人本体观测来自 robosuite；策略只接收双相机图像/HSV掩码、关节与夹爪位置、末端XYZ、当前阶段和目标颜色one-hot。物体真值位置、body ID、速度和距离仅用于测评日志，不输入ACT或驱动执行器。", "",
             "## HSV 与四分动作", "",
             "HSV由RGB图像用 OpenCV RGB2HSV + inRange 得到；红色采用跨Hue零点的两个区间，黄色、蓝色各一个区间，随后3×3开运算和小区域过滤。掩码三通道固定对应红/黄/蓝，不是渲染器真值分割。环境验证时与渲染几何体分割的IoU约0.93–0.99，真值只用于验证。", "",
             "动作顺序为：接近目标 → 精确对准 → 夹紧 → 抬升并保持。ACT预测XYZ与阶段建议；共同执行器按机器人自身到位、夹爪稳定和最小闭合时间推进，限制跳阶段和倒退，冻结夹紧后的XY并在抬升到位后锁定保持目标。XYZ、速度限幅、死区和阶段推进均不读取目标物体真值。", "",
             "`macro_actions`记录实际阶段进入数；`control_steps`记录所有底层控制步。2s保持所需的40步单列，不计入有效保持期间的空等。检测到空夹爪持续12步时提前结束。四个宏动作不会把真实物理耗时伪装成4步。", "",
             "开发阶段曾发现纯阶段建议等待造成对准停滞，随后在开发初态上修正为上述共同完成驱动规则；正式测试初态未用于该调整。记录见 `rgb/eval_dev/` 与 `protocol.json` 的控制器版本。", "",
             "## 公平对照配置", "",
             "| 结构 | 每相机输入 | 其他设置 |", "| --- | --- | --- |",
             "| RGB-ACT | RGB 3通道 | 共享ResNet18与ACT Transformer/CVAE |",
             "| Mask-ACT | 红/黄/蓝二值HSV 3通道 | 同一骨干与模型规模 |",
             "| RGB+Mask-ACT | RGB+HSV共6通道，早期融合 | 扩展第一卷积；RGB权重保留、额外Mask权重零初始化 |", "",
             f'三组共用100条训练示范、20条独立验证示范，训练各{protocol["updates_per_variant"]}次更新，batch32、seed42、chunk16，每3个控制步重新查询。网络17.66M参数量级；融合仅增加9408个第一层卷积参数。ResNet18为ImageNet初值，hidden256、encoder2层、decoder3层。损失为XYZ L1 + 0.25阶段交叉熵 + 10×KL。按独立验证集prior重建误差选择检查点。', "",
             f'示范由HSV掩码、相机标定与四阶段OSC专家采集，不使用物体真值位置控制。采集保存120条成功示范，另记录{len(data["failed_attempts"])}次未成功尝试。HSV+OSC基线单独在同一批30个新初态上评测，未筛选成功回合。', "",
             "正式测试seed为40000–40029；四组的三个物块初始位置逐回合完全一致。仅使用物理GPU0、1训练，最多两张；各模型一次训练随机种子，成功率同时给出Wilson区间，不能作多种子统计显著性的结论。", "",
             "## 实测结果", "",
             "| 结构 | 成功/30 | 成功率 | 成功平均控制步 | 扣除必要40步后 | 抓错颜色率 | 碰撞回合率 | 最终保持失败率 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    fmt = lambda value: "—" if value is None else f"{value:.1f}"
    for m in metrics:
        lines.append(f'| {NAMES[m["variant"]]} | {m["successes"]}/30 | {m["success_rate"]:.1%} | {fmt(m["mean_success_steps"])} | {fmt(m["mean_non_mandatory_success_steps"])} | {m["wrong_color_rate"]:.1%} | {m["collision_episode_rate"]:.1%} | {m["hold_failure_rate"]:.1%} |')
    lines += ["", "成功回合集合可能不同，不能仅以成功平均步数判断算法快慢；失败提前结束也不能算作成功效率提升。完整无碰撞成功率、保持中断率和每个初态结果见CSV/JSON。", "", "![对照](comparison.png)", "",
              "## 无效动作诊断", "", "| 结构 | 阶段内命令反转/回合 | 阶段内末端反转/回合 | 闭合后空等步/回合 | 空夹爪空等步/回合 | 距离停滞步/回合 |", "| --- | --- | --- | --- | --- | --- |"]
    for m in metrics:
        lines.append(f'| {NAMES[m["variant"]]} | {m["mean_command_reversals_within_phase"]:.2f} | {m["mean_motion_reversals_within_phase"]:.2f} | {m["mean_post_close_idle_steps"]:.2f} | {m["mean_empty_gripper_wait_steps"]:.2f} | {m["mean_no_progress_steps"]:.2f} |')
    lines += ["", "![诊断](diagnostics.png)", "", "精确定义及阈值见 `metric_dictionary.zh-CN.json`；阶段内反转排除下降转抬升等阶段切换。距离停滞只统计接近/对准，避免把夹紧或必要保持误判成不进步。末端到目标距离使用真值测量，仅作日志。", "", "## 案例与文件", ""]
    for m in metrics:
        for successful in (True, False):
            case = next((r for r in m["trials"] if bool(r["success"]) == successful), None)
            if case:
                prefix = f'{m["variant"]}/eval_test'
                lines.append(f'- {NAMES[m["variant"]]}：seed {case["seed"]}，`{case["outcome"]}`，{case["control_steps"]}步；[视频]({prefix}/{case["video"]})、[逐步CSV]({prefix}/{case["step_trace"]})。')
    lines += ["", "`comparison.csv`为总表；各结构`eval_test/episodes.csv`为逐回合结果，`episode_NNN_steps.csv`包含目标距离、阶段、请求阶段、XYZ命令、夹爪间隙和所有检测标志。`metrics.json`保存任务参数、训练参数、body IDs与每回合初态。`policy.pt`含推理权重、归一化统计、结构和任务配置。", "",
              "## 超参数试验入口", "", "```bash", "source /home/catlab/mujoco_robosuite/activate.sh", "export CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1", "python -m colorlift.train --variant rgb_mask --lr 5e-5 --chunk-size 8 --output experiments/colorlift_v1/tuning/fusion_trial", "python -m colorlift.evaluate --variant rgb_mask --checkpoint experiments/colorlift_v1/tuning/fusion_trial/best.pt --split dev --episodes 6 --output experiments/colorlift_v1/tuning/fusion_trial/eval_dev", "python -m colorlift.aggregate", "```", "",
              "训练超参数可用CLI调整；OSC增益、限幅、死区、阶段门限、保持和诊断阈值在`config.json`。评测可用`--config`指定修改副本并用`--output`独立保存。改变任务或重新采集时设置新的`COLORLIFT_RUN_DIR`，避免混用旧示范。默认训练与正式对照针对红色目标；环境支持其他目标颜色，需相应颜色示范再训练后评估。", "",
              "参考：[robosuite控制器](https://robosuite.ai/docs/modules/controllers.html)、[OpenCV HSV](https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html)、[官方ACT源码](https://github.com/tonyzhaozh/act)。"]
    (EXPERIMENT / "REPORT.zh-CN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (EXPERIMENT / "completed.json").write_text(json.dumps({"completed": True, "results": [{k: v for k, v in m.items() if k not in ("trials", "training_config", "task_config")} for m in metrics]}, indent=2))
    print("REPORT READY: " + str(EXPERIMENT / "REPORT.zh-CN.md"), flush=True)


if __name__ == "__main__":
    main()
