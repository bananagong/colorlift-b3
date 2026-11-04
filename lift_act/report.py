"""Generate a report and plots exclusively from measured training/evaluation artifacts."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .env import EXPERIMENT


def main():
    metrics = [json.loads((EXPERIMENT / f"act_{n}" / "eval_test_q5" / "metrics.json").read_text()) for n in (20, 50)]
    manifest = json.loads((EXPERIMENT / "data" / "manifest.json").read_text())
    env = json.loads((EXPERIMENT / "environment" / "verification.json").read_text())
    colors = ["#2276b7", "#e67e22"]
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for color, m in zip(colors, metrics):
        i = (20, 50).index(m["demos"])
        value = m["success_rate"] * 100
        lo, hi = np.asarray(m["wilson_95_ci"]) * 100
        axes[0].bar(i, value, color=color, width=0.55)
        axes[0].errorbar(i, value, yerr=[[max(0, value - lo)], [max(0, hi - value)]], color="black", capsize=6, fmt="none")
        axes[0].text(i, min(112, max(value, hi) + 4), f'{m["successes"]}/{m["episodes"]}', ha="center")
    axes[0].set(xticks=[0, 1], xticklabels=["20 demos", "50 demos"], ylim=(0, 120), ylabel="Lift success (%)", title="20 fixed held-out initial states\n95% Wilson intervals")
    axes[0].set_yticks([0, 25, 50, 75, 100])
    for color, m in zip(colors, metrics):
        rows = [json.loads(line) for line in (EXPERIMENT / f'act_{m["demos"]}' / "metrics.jsonl").read_text().splitlines()]
        rows = [r for r in rows if "val_prior_l1" in r]
        axes[1].plot([r["step"] for r in rows], [r["val_prior_l1"] for r in rows], color=color, marker="o", label=f'{m["demos"]} demos')
    axes[1].set(xlabel="Optimizer updates", ylabel="Normalized action L1", title="Held-out prior prediction error")
    axes[1].legend()
    fig.savefig(EXPERIMENT / "comparison.png", dpi=180)
    plt.close(fig)
    with (EXPERIMENT / "comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["demos", "successes", "episodes", "success_rate", "mean_success_steps", "median_success_steps", "checkpoint_training_step", "validation_prior_l1"])
        writer.writeheader()
        writer.writerows({key: m[key] for key in writer.fieldnames} for m in metrics)
    lines = [
        "# 机器人模仿学习：正方体 Lift 与 ACT 对照实验", "",
        "已完成环境、正方体资产、示范采集、ACT 训练和闭环测试。实验仅使用物理 GPU 1，未超过两张卡的限制。", "",
        "## 环境与资产", "",
        "Ubuntu 24.04、MuJoCo 3.3.7、robosuite 1.5.2、Python 3.11；Panda + OSC_POSE，控制频率 20 Hz，双 RGB 相机 128×128。", "",
        "新增 CubeObject：边长 0.04 m，质量 0.064 kg，单刚体、一个自由关节、独立碰撞与视觉几何体；可调整边长、质量、摩擦和颜色。", "",
        f'环境验证结果：`{env["passed"]}`；相机尺寸、质量、控制和相同随机种子的重复重置均通过。', "",
        "## 预先固定的实验协议", "",
        "- 成功：方块中心 z≥0.90 m（桌面 0.80 m），连续保持 10 步（0.5 秒）。最长 160 步。",
        "- 方块初态：X/Y 各在 ±0.055 m 随机，yaw 在 ±0.25 rad 随机；机器人初始关节姿态固定。",
        "- 训练组：20 / 50 条示范，20 条是 50 条中的固定子集。10 条独立示范用于验证，最终两组均测试种子 5000–5019。",
        "- 每组相同 6000 次优化更新、batch 32、随机种子 42；按验证集 prior L1 选择检查点。归一化统计只取本组训练数据。",
        "- 两组都是一次训练运行；20 次测试提供本次实验的成功率，不能代替多训练种子的统计结论。", "",
        "## 示范与 ACT 实现", "",
        f'共保存 {len(manifest["episodes"])} 条成功脚本示范，其中训练 50 条、验证 10 条。脚本专家使用方块真值位置执行接近、下降、夹紧、抬升和保持；这些是真实接触仿真轨迹，未宣称人工遥操作。', "",
        "ACT 使用 [tonyzhaozh/act 官方实现](https://github.com/tonyzhaozh/act)，源码 commit `742c753c0d4a5d87076c8f69e5628c79a8cc5488`。保留 ResNet18、Transformer 编解码器、CVAE、32 维隐变量、动作分块和 L1+10×KL 损失；适配 12 维机器人状态和 4 维笛卡尔目标动作。", "",
        "网络使用 ImageNet 预训练 ResNet18，hidden 256、encoder 2 层、decoder 3 层、feedforward 1024，chunk 20。训练学习率 1e-4，视觉骨干 1e-5，AdamW。", "",
        "推理只输入双相机 RGB、7 个关节位置、2 个夹爪位置和末端 XYZ。预测绝对 XYZ 与夹爪开合，由 OSC_POSE 根据机器人末端自身状态跟踪；保持末端初始方向。每执行 5 步重新观察并预测，推理使用零隐变量。评测没有方块真值输入、专家纠正或专家回退。", "",
        "## 实测结果", "",
        "| 训练示范 | 成功 / 测试 | 成功率 | 成功回合平均步数 | 选择的训练步 | 验证 prior L1 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for m in metrics:
        mean = "—" if m["mean_success_steps"] is None else f'{m["mean_success_steps"]:.1f}'
        lines.append(f'| {m["demos"]} | {m["successes"]}/{m["episodes"]} | {m["success_rate"]:.0%} | {mean} | {m["checkpoint_training_step"]} | {m["validation_prior_l1"]:.4f} |')
    delta = 100 * (metrics[1]["success_rate"] - metrics[0]["success_rate"])
    lines += ["", f'在相同 6000 次更新预算与相同测试初态下，50 条示范相对 20 条示范的成功率变化为 {delta:+.1f} 个百分点。测试初态配对一致，减少了两组测试难度的差异。', "",
              "平均成功步数只统计成功回合；两组成功回合集合不同，不直接作为严格的执行速度优劣结论。", "",
              "示范数量、固定优化预算和测试分布共同影响结果。当前每组只训练一个随机种子，置信区间也较宽，因此这些实测结果应作为本次仿真设置中的对照，不作普遍优越性或统计显著性的结论。"]
    lines += ["", "![成功率与验证误差](comparison.png)", "", "## 成功与失败案例", ""]
    for m in metrics:
        prefix = f'act_{m["demos"]}/eval_test_q5'
        success = next((t for t in m["trials"] if t["success"]), None)
        failure = next((t for t in m["trials"] if not t["success"]), None)
        if success:
            lines.append(f'- {m["demos"]} 条示范组成功例：seed {success["seed"]}，{success["steps"]} 步；[视频]({prefix}/{success["video"]})。')
        if failure:
            lines.append(f'- {m["demos"]} 条示范组失败例：seed {failure["seed"]}，最高 z={failure["max_cube_height"]:.4f} m，观测分类 `{failure["failure_observation"]}`；[视频]({prefix}/{failure["video"]})。')
        else:
            lines.append(f'- {m["demos"]} 条示范组在本次 20 个测试初态中未出现失败；不据此声称任意初态都能成功。')
    lines += ["", "所有测试回合的视频、逐回合指标和高度轨迹均已保留，可复查个别失败。此比较反映示范数量在当前仿真分布中的效果；没有扩展到材质变化、明显更大初态范围或真实机器人。", "",
              "## 复现与文件", "",
              "项目：`/home/catlab/mujoco_robosuite`；实验：`experiments/lift_act_v1`。", "",
              "```bash", "source /home/catlab/mujoco_robosuite/activate.sh", "bash /home/catlab/mujoco_robosuite/run_lift_experiment.sh", "```", "",
              "脚本会复用已完成数据、检查点和评测结果。`data/` 保存示范；`act_20/`、`act_50/` 保存训练日志和 best/last.pt；`policy.pt` 为导出的推理权重，含模型、归一化统计与配置，不含优化器状态；`eval_test_q5/metrics.json` 保存完整评测；`lift_act/protocol.json` 保存预先设定的协议。", "",
              "重新从零运行时应另建实验输出目录，保留已有测试记录。正方体可通过 `from catlab_assets import CubeObject` 导入。原 T 方块资产继续可用。推 T 训练属于图片中的进阶选做，本次交付完成必做 Lift 项目。", ""]
    (EXPERIMENT / "REPORT.zh-CN.md").write_text("\n".join(lines), encoding="utf-8")
    (EXPERIMENT / "completed.json").write_text(json.dumps({"completed": True, "conditions": [{k: v for k, v in m.items() if k != "trials"} for m in metrics]}, indent=2))
    print("REPORT READY: " + str(EXPERIMENT / "REPORT.zh-CN.md"), flush=True)


if __name__ == "__main__":
    main()
