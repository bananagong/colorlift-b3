"""Validate recorded short trajectories and build a measured, reproducible report."""
from collections import Counter
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .config import ROOT
from .short_trajectories import OUTPUT
from .step_recording import first_steps_breakdown, save_rows

BUCKETS = ("approach", "align", "close", "lift_and_settle", "stable_hold")
LABELS = ("Approach", "Align", "Close", "Lift / settle", "Valid hold")
COLORS = ("#4c78a8", "#72b7b2", "#f2cf5b", "#f58518", "#54a24b")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def values(rows, columns):
    return np.array([[float(row[column]) for column in columns] for row in rows])


def audit_case(directory, dataset=False):
    meta = read_json(directory / "metadata.json")
    rows = read_rows(directory / "steps.csv")
    n = meta["control_steps"]
    assert len(rows) == n
    assert [int(row["step"]) for row in rows] == list(range(1, n + 1))
    assert all(int(row["target_body_id"]) == meta["body_ids"]["red"] for row in rows)
    before = values(rows, ["eef_before_" + axis for axis in "xyz"])
    after = values(rows, ["eef_after_" + axis for axis in "xyz"])
    target = values(rows, ["target_after_" + axis for axis in "xyz"])
    raw_actions = values(rows, [f"osc_action_{i}" for i in range(7)])
    motor_controls = values(rows, [f"actuator_ctrl_{i}" for i in range(9)])
    assert np.allclose(before[1:], after[:-1], atol=1e-12, rtol=0)
    assert np.allclose(after, values(rows, ["eef_" + axis for axis in "xyz"]), atol=1e-12, rtol=0)
    assert np.allclose(target, values(rows, ["target_" + axis for axis in "xyz"]), atol=1e-12, rtol=0)
    assert np.allclose(raw_actions[:, :3] * .05, values(rows, ["command_" + axis for axis in "xyz"]), atol=1e-9)
    assert np.allclose(raw_actions[:, -1], values(rows, ["gripper_command"])[:, 0])
    dt = values(rows, ["time_after_s"]) - values(rows, ["time_before_s"])
    assert np.allclose(dt, 1 / meta["task_config"]["hz"], atol=1e-9)
    assert np.isfinite(np.c_[before, after, target, raw_actions, motor_controls]).all()
    if meta["success"]:
        last = rows[-40:]
        assert [int(row["hold_streak"]) for row in last] == list(range(1, 41))
        assert min(float(row["target_lift_m"]) for row in last) > .04
        assert max(float(row["target_linear_speed"]) for row in last) <= .025
        assert max(float(row["target_angular_speed"]) for row in last) <= .4
        assert np.max(np.linalg.norm(target[-40:] - target[-40], axis=1)) <= .005 + 1e-10
        assert meta["wrong_color_grasp"] is False
    if dataset:
        with np.load(directory / "trajectory.npz", allow_pickle=False) as data:
            assert data["states"].shape == (n, 19)
            assert data["actions"].shape == (n, 7)
            assert data["images"].shape == (n, 2, 128, 128, 3)
            assert data["masks"].shape == (n, 2, 128, 128, 3)
            assert data["final_images"].shape == (2, 128, 128, 3)
            assert data["final_masks"].shape == (2, 128, 128, 3)
            assert set(np.unique(data["masks"])).issubset({0, 255})
            assert np.allclose(data["osc_actions"], raw_actions, atol=1e-7)
            assert np.allclose(data["eef_before"], before, atol=1e-12, rtol=0)
            assert np.allclose(data["eef_after"], after, atol=1e-12, rtol=0)
            assert np.allclose(data["diagnostic_target_xyz"], target, atol=1e-12, rtol=0)
            assert np.allclose(data["states"][:, 9:12], before, atol=1e-7)
            assert np.array_equal(np.argmax(data["states"][:, 12:16], axis=1), np.argmax(data["actions"][:, 3:7], axis=1))
            assert np.allclose(data["actuator_controls"], motor_controls, atol=1e-12, rtol=0)
            assert np.allclose(data["gripper_qpos"], values(rows, [f"finger_qpos_{i}_after" for i in range(2)]))
    return meta, rows


def summary(trials):
    successes = [trial for trial in trials if trial["success"]]
    mean = lambda key: float(np.mean([trial[key] for trial in trials]))
    return {
        "episodes": len(trials), "successes": len(successes),
        "success_rate": len(successes) / len(trials),
        "mean_all_steps": mean("control_steps"),
        "mean_success_steps": float(np.mean([trial["control_steps"] for trial in successes])) if successes else None,
        "minimum_steps": min(trial["control_steps"] for trial in trials),
        "maximum_steps": max(trial["control_steps"] for trial in trials),
        "wrong_color_episodes": sum(trial["wrong_color_grasp"] for trial in trials),
        "collision_episodes": sum(trial["collision_steps"] > 0 for trial in trials),
        "hold_failure_episodes": sum(trial["outcome"] == "hold_failure" for trial in trials),
        "hold_interruption_episodes": sum(trial["holding_failure_events"] > 0 for trial in trials),
        **{f"mean_{key}": mean(key) for key in ("command_reversals_within_phase", "motion_reversals_within_phase", "post_close_idle_steps", "no_progress_steps")},
    }


def main():
    assert read_json(OUTPUT / "completed.json")["completed"]
    protocol = read_json(OUTPUT / "protocol.json")
    manifest = read_json(OUTPUT / "demonstration_manifest.json")
    baseline, efficient = [], []
    traces = {"baseline": [], "efficient": []}
    for kind, results in (("baseline", baseline), ("efficient", efficient)):
        for seed in protocol["evaluation_seeds"]:
            metadata, rows = audit_case(OUTPUT / "paired" / kind / f"seed_{seed}")
            results.append(metadata)
            traces[kind].append(rows)
    comparison = []
    for old, new in zip(baseline, efficient):
        assert old["seed"] == new["seed"]
        for color in ("red", "yellow", "blue"):
            assert np.array_equal(old["initial_positions"][color], new["initial_positions"][color])
        old_config, new_config = dict(old["task_config"]), dict(new["task_config"])
        old_config.pop("osc_kp")
        new_config.pop("osc_kp")
        assert old_config == new_config
        comparison.append({"seed": old["seed"], "baseline_success": old["success"], "efficient_success": new["success"],
                           "baseline_steps": old["control_steps"], "efficient_steps": new["control_steps"],
                           "steps_saved": old["control_steps"] - new["control_steps"],
                           "baseline_collision_steps": old["collision_steps"], "efficient_collision_steps": new["collision_steps"],
                           "baseline_outcome": old["outcome"], "efficient_outcome": new["outcome"]})
    save_rows(OUTPUT / "paired_comparison.csv", comparison)
    demos, demonstration_rows = [], []
    for trial in manifest["successful"] + manifest["failed_attempts"]:
        meta, rows = audit_case(OUTPUT / trial["directory"], dataset=True)
        assert meta["success"] == trial["success"]
        if meta["success"]:
            demos.append(meta)
        demonstration_rows.append({"seed": meta["seed"], "success": meta["success"], "steps": meta["control_steps"],
                                   "seconds": meta["control_steps"] / 20, "max_lift_m": meta["max_lift_m"],
                                   "valid_hold_steps": meta["valid_hold_steps"], "collision_steps": meta["collision_steps"],
                                   "wrong_color_grasp": meta["wrong_color_grasp"], "directory": trial["directory"]})
    save_rows(OUTPUT / "demonstration_summary.csv", demonstration_rows)
    representatives = {}
    first70_rows = []
    for kind in ("baseline", "mask_act", "efficient"):
        meta, rows = audit_case(OUTPUT / "representative" / kind)
        representatives[kind] = (meta, rows)
        first70_rows.append({"controller": kind, "seed": meta["seed"], "total_steps": meta["control_steps"],
                             **meta["first70"]["composition"], "hold_steps_remaining_after70": 40 - meta["first70"]["hold_streak_at_end"]})
    save_rows(OUTPUT / "first70_summary.csv", first70_rows)
    # Adding telemetry must reproduce the old example, rather than altering its behavior.
    source = ROOT / "experiments" / "colorlift_v1"
    for kind, variant in (("baseline", "hsv"), ("mask_act", "mask")):
        original = read_json(source / variant / "eval_test" / "metrics.json")["trials"][0]
        assert original["seed"] == 40000
        replay = representatives[kind][0]
        assert original["control_steps"] == replay["control_steps"]
        assert original["phase_control_steps"] == replay["phase_control_steps"]
        assert original["success"] == replay["success"]
    old_summary, new_summary = summary(baseline), summary(efficient)
    both = [row for row in comparison if row["baseline_success"] and row["efficient_success"]]
    old_mean = float(np.mean([row["baseline_steps"] for row in both]))
    new_mean = float(np.mean([row["efficient_steps"] for row in both]))
    stats = {"baseline": old_summary, "efficient": new_summary, "demonstrations": summary(demos),
             "demonstration_attempts": manifest["attempt_count"], "demonstration_failures": len(manifest["failed_attempts"]),
             "joint_success_pairs": len(both), "paired_mean_steps_saved": old_mean - new_mean,
             "paired_step_reduction_percent": (old_mean - new_mean) / old_mean * 100,
             "paired_nonhold_reduction_percent": (old_mean - new_mean) / (old_mean - 40) * 100,
             "paired_minimum_steps_saved": min(row["steps_saved"] for row in both),
             "paired_maximum_steps_saved": max(row["steps_saved"] for row in both),
             "phase_means": {kind: {bucket: float(np.mean([first_steps_breakdown(rows, limit=10000)["composition"][bucket] for rows in recorded]))
                                    for bucket in BUCKETS} for kind, recorded in traces.items()}}
    (OUTPUT / "summary.json").write_text(json.dumps(stats, indent=2))
    audit = {"passed": True, "paired_cases": len(comparison) * 2, "demonstration_cases": manifest["attempt_count"],
             "representative_cases": len(representatives), "initial_positions_paired_identically": True,
             "original_example_reproduced_with_recorder": True, "task_config_only_osc_gain_changed": True,
             "checks": ["step count and index", "world-frame EEF before/after continuity", "OSC actions match command deltas",
                        "control intervals are 50ms", "body ID maps to red", "finite robot and actuator values",
                        "final 40 samples satisfy lift, velocity and drift thresholds", "NPZ shapes and CSV alignment",
                        "RGB/mask camera dimensions and binary mask values", "proprioceptive state aligns with action phase"]}
    (OUTPUT / "audit.json").write_text(json.dumps(audit, indent=2))
    plot_results(stats, comparison, representatives)
    write_report(stats, representatives, audit)
    print(json.dumps(stats, indent=2))
    print("AUDIT PASSED", json.dumps(audit))


def plot_results(stats, comparison, representatives):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    names = ("baseline", "mask_act", "efficient")
    labels = ("Original HSV+OSC", "Existing Mask-ACT", "Efficient HSV+OSC")
    left = np.zeros(3)
    for bucket, label, color in zip(BUCKETS, LABELS, COLORS):
        lengths = [representatives[kind][0]["first70"]["composition"][bucket] for kind in names]
        axes[0].barh(labels, lengths, left=left, color=color, label=label)
        for i, length in enumerate(lengths):
            if length:
                axes[0].text(left[i] + length / 2, i, str(length), ha="center", va="center", fontsize=9)
        left += lengths
    axes[0].set_xlim(0, 70)
    axes[0].set_xlabel("Control steps (20 Hz)")
    axes[0].set_title("First 70 steps: same seed 40000")
    axes[0].invert_yaxis()
    axes[0].legend(loc="upper center", bbox_to_anchor=(.5, -.18), ncol=2, frameon=False, fontsize=8)
    x = np.array([row["baseline_steps"] for row in comparison])
    y = np.array([row["efficient_steps"] for row in comparison])
    axes[1].scatter(x, y, color="#4c78a8", alpha=.75)
    lo, hi = min(x.min(), y.min()) - 2, max(x.max(), y.max()) + 2
    axes[1].plot([lo, hi], [lo, hi], "--", color=".6", linewidth=1)
    axes[1].set(xlim=(lo, hi), ylim=(lo, hi), xlabel="Original HSV+OSC steps", ylabel="Efficient expert steps",
                title=f"Paired held-out initial states (n={len(x)})")
    axes[1].text(.04, .94, f"Mean: {x.mean():.2f} -> {y.mean():.2f}\nSaved: {stats['paired_step_reduction_percent']:.1f}%", transform=axes[1].transAxes, va="top")
    left = np.zeros(2)
    for bucket, label, color in zip(BUCKETS, LABELS, COLORS):
        lengths = [stats["phase_means"][kind][bucket] for kind in ("baseline", "efficient")]
        axes[2].barh(("Original", "Efficient"), lengths, left=left, color=color)
        for i, length in enumerate(lengths):
            axes[2].text(left[i] + length / 2, i, f"{length:.1f}", ha="center", va="center", fontsize=8)
        left += lengths
    axes[2].set(xlabel="Mean control steps", title="Where the saved steps come from")
    axes[2].invert_yaxis()
    fig.savefig(OUTPUT / "comparison.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(4, 2, figsize=(13, 10), sharex=True, constrained_layout=True)
    for col, kind in enumerate(("baseline", "efficient")):
        meta, rows = representatives[kind]
        t = np.arange(1, len(rows) + 1)
        axes[0, col].plot(t, values(rows, ["eef_after_z"])[:, 0], label="EEF z")
        axes[0, col].plot(t, values(rows, ["target_after_z"])[:, 0], label="Red cube z")
        axes[0, col].plot(t, values(rows, ["goal_z"])[:, 0], "--", label="Command goal z")
        axes[0, col].axhline(meta["initial_heights"]["red"] + .04, color=".4", ls=":", label="4 cm threshold")
        axes[0, col].set(title=f"{kind}: {len(rows)} steps / {len(rows)/20:.2f}s", ylabel="World z (m)")
        axes[0, col].legend(fontsize=8, ncol=2)
        axes[1, col].plot(t, values(rows, ["eef_target_distance_m"])[:, 0] * 1000, label="EEF-to-cube distance")
        axes[1, col].plot(t, values(rows, ["gripper_gap_m"])[:, 0] * 1000, label="Finger gap")
        axes[1, col].set(ylabel="Distance / gap (mm)")
        axes[1, col].legend(fontsize=8)
        for index, axis in enumerate("xyz"):
            axes[2, col].plot(t, values(rows, [f"osc_action_{index}"])[:, 0], label=f"OSC {axis}")
        axes[2, col].step(t, values(rows, ["gripper_command"])[:, 0], color=".25", alpha=.65, label="Gripper command")
        axes[2, col].set(ylabel="Normalized OSC action")
        axes[2, col].legend(fontsize=8, ncol=2)
        axes[3, col].plot(t, values(rows, ["hold_streak"])[:, 0], color=COLORS[-1], label="Valid hold streak")
        axes[3, col].step(t, values(rows, ["phase"])[:, 0] * 10, label="Phase index x10", alpha=.6)
        axes[3, col].set(xlabel="Control step (20 Hz)", ylabel="Hold / phase")
        axes[3, col].legend(fontsize=8)
        for axis in axes[:, col]:
            axis.axvline(70, color=".7", ls=":", lw=1)
            axis.grid(alpha=.15)
    fig.savefig(OUTPUT / "step_telemetry.png", dpi=150)
    plt.close(fig)


def write_report(stats, representatives, audit):
    old, new, demos = (stats[key] for key in ("baseline", "efficient", "demonstrations"))
    protocol = read_json(OUTPUT / "protocol.json")
    expert = protocol["expert_config"]
    lines = ["# 红色方块：逐步记录、前 70 步拆解与短轨迹专家", "",
             "本次记录和专家采集已经完成，未重新训练 ACT。任务仍为：相对重置位置抬升超过 0.04 m，正确抓住红色方块并连续稳定保持 2 秒（20 Hz 下 40 步）。", "",
             "## 实测对照", "", "30 个独立测试初态使用种子 50000–50029，与调参/采集初态分开；原专家和高效专家逐个使用相同的红黄蓝初始位置。所有测试回合均计入，失败不会被删去。", "",
             "| 控制器 | 成功 | 成功平均步数 | 全部平均步数 | 步数范围 | 碰撞回合 | 抓错颜色 | 保持失败 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for title, item in (("原 HSV+OSC 专家", old), ("高效视觉专家", new)):
        lines.append(f"| {title} | {item['successes']}/{item['episodes']} | {item['mean_success_steps']:.2f} | {item['mean_all_steps']:.2f} | {item['minimum_steps']}–{item['maximum_steps']} | {item['collision_episodes']} | {item['wrong_color_episodes']} | {item['hold_failure_episodes']} |")
    lines.extend(["", f"共同成功的 {stats['joint_success_pairs']} 组初态中，平均减少 {stats['paired_mean_steps_saved']:.2f} 步，即 {stats['paired_step_reduction_percent']:.1f}%；扣除必须的 40 步稳定保持后，其余步骤减少 {stats['paired_nonhold_reduction_percent']:.1f}%。这是该次仿真样本的结果。", "",
                  "![配对结果与步骤构成](comparison.png)", "", "## 前 70 步到底做了什么", "",
                  "下面固定种子 40000。先从原有 4 种方法共 120 个测试 CSV 重新统计前 70 步，再为原 HSV、现有 Mask-ACT 和高效专家补录完整机器人状态。补录后的原 HSV/Mask-ACT 总步数、阶段步数和成败均与历史记录一致。", "",
                  "四个执行阶段中的“抬升并保持”进一步分成“抬升/稳定收敛”和“有效保持”，因此表中五列互斥、相加恰好为 70；有效保持属于第 4 个分动作。", "",
                  "| 控制器 | 接近 | 对准 | 夹紧 | 抬升/稳定收敛 | 有效保持 | 完整成功步数 |", "|---|---:|---:|---:|---:|---:|---:|"])
    for kind, title in (("baseline", "原 HSV+OSC"), ("mask_act", "现有 Mask-ACT"), ("efficient", "高效视觉专家")):
        meta = representatives[kind][0]
        counts = meta["first70"]["composition"]
        lines.append(f"| {title} | " + " | ".join(str(counts[key]) for key in BUCKETS) + f" | {meta['control_steps']} |")
    lines.extend(["", "对应的连续区间：", ""])
    chinese = dict(zip(BUCKETS, ("接近目标", "精确对准", "夹紧", "抬升/稳定收敛", "有效稳定保持")))
    for kind, title in (("baseline", "原 HSV+OSC"), ("mask_act", "现有 Mask-ACT"), ("efficient", "高效视觉专家")):
        segments = representatives[kind][0]["first70"]["segments"]
        lines.append(f"- {title}：" + "；".join(f"{item['start_step']}–{item['end_step']} 步{chinese[item['bucket']]}（{item['steps']} 步）" for item in segments) + "。")
    lines.extend(["", "前 70 步不能全部算作无效运动：其中已经包含不可省略的 2 秒保持。空等和距离停滞是辅助标记，可能与阶段重叠，不能再与上述五类相加。", "",
                  "## 轨迹和日志", "",
                  f"已生成 **{demos['successes']} 条成功轨迹**，共尝试 {stats['demonstration_attempts']} 回合，失败尝试 {stats['demonstration_failures']} 条也保留。成功轨迹平均 {demos['mean_success_steps']:.2f} 步，范围 {demos['minimum_steps']}–{demos['maximum_steps']} 步。每条均保留最后完整的 40 步有效保持及终帧。", "",
                  "- `demonstrations/attempt_*_seed_*/trajectory.npz`：双相机 RGB、HSV mask、本体状态、宏动作标签、实际 7 维 OSC 动作，以及真实位置和执行器诊断。", "- 每个回合的 `steps.csv`：每步完整日志；`first70_steps.csv`：前 70 步原始数据；`metadata.json`：初始位置、body ID、阈值、结局和指标。", "- `representative/{baseline,mask_act,efficient}/trajectory.mp4`：同一初态的三种控制器视频，上排 RGB，下排 HSV mask。", "- `existing_first70/episodes.csv`：原 HSV、RGB-ACT、Mask-ACT、RGB+Mask-ACT 各 30 个历史回合的前 70 步拆解。", "- `paired_comparison.csv`：30 组原专家与高效专家的逐一对照。`demonstration_summary.csv`：所有采集尝试清单。", "", "CSV 的主要字段：", "",
                  "| 字段 | 含义 |", "|---|---|", "| `time_before_s`, `time_after_s` | 本次控制前/后的仿真时间，间隔 0.05 秒 |", "| `eef_before_x/y/z`, `eef_after_x/y/z` | 末端世界坐标，单位 m |", "| `target_before_x/y/z`, `target_after_x/y/z` | 红方块真实世界坐标，仅用于诊断 |", "| `visual_target_x/y/z` | 回合开始时双相机 HSV 定位估计，与真实目标和指令目标分开 |", "| `goal_x/y/z` | 当前末端指令目标 |", "| `finger_qpos_0/1_before/after`, `finger_qvel_0/1_before/after` | 两根手指关节位置/速度，单位 m、m/s |", "| `gripper_command`, `gripper_mode`, `gripper_gap_m` | 开合命令（-1 开、0 保持开度、+1 夹紧）及实际指间距 |", "| `osc_action_0`…`osc_action_6` | 实际传给 env.step 的 [归一化 dx,dy,dz,rx,ry,rz,夹爪] |", "| `actuator_ctrl_0`…`actuator_ctrl_8` | 7 维手臂力矩 Nm + 2 维手指位置控制 m，为控制区间末尾值 |", "| `robot_joint_0`…`robot_joint_6` | 控制后 Panda 7 个关节角 rad |", "| `phase`, `phase_name`, `hold_streak` | 4 个分动作和当前连续有效保持步数 |", "| `post_close_idle`, `no_progress` | 夹紧后空等、接近/对准时距离停滞标记 |", "| `target_lift_m`, `collision`, `wrong_color_grasp` | 抬升量、碰撞与抓错颜色状态 |", "",
                  "NPZ：`images/masks` 为 `[T,2,128,128,3]`、uint8，mask 通道按红/黄/蓝排列；`states` 为 `[T,19]`（7 关节+2 手指+3 末端位置+4 阶段 one-hot+3 目标颜色 one-hot）；`actions` 为 `[T,7]`（世界系 XYZ 指令目标+4 阶段 one-hot），`osc_actions` 才是 `[T,7]` 的实际底层控制。每个状态/图像与执行前动作对齐，真实目标坐标以 `diagnostic_target_xyz` 单独存储，未加入模型本体状态。", "",
                  "![逐步位置、夹爪、动作与保持曲线](step_telemetry.png)", "", "## 高效专家改了什么", "",
                  f"1. 回合开始使用双相机 RGB 的 HSV 红色分割和相机标定估计目标位置，后续通过机器人本体状态闭环执行；这是初始视觉定位辅助，未使用方块真实坐标规划动作。", f"2. 预抓取目标改为方块中心平面上方 {expert['pregrasp_clearance']*1000:.0f} mm（世界 z={.82+expert['pregrasp_clearance']:.3f} m），减少先停在高处再向下对准的路程。", f"3. 手指开度达到 {expert['open_gap']*1000:.0f} mm 后保持开度；闭合至少 {expert['minimum_close_steps']} 步且手指变化小于 {expert['finger_motion_tolerance']*1000:.1f} mm 后切换抬升，实际闭合步数由状态决定。", f"4. 抬升指令 z={.82+.04+expert['lift_margin']:.3f} m，保留超过 4 cm 的裕量；原专家指令 z=0.940 m。实际物体高度与末端高度分开记录。", f"5. OSC kp 从 150 调为 {expert['kp']:.0f}，最大位置增量 {expert['maximum_step']*1000:.0f} mm，对准 {expert['align_maximum_step']*1000:.0f} mm，旋转命令保持 0。使用 GPU 2，仅占用 1 张卡。", "",
                  "上述改动作为一个整体进行比较；没有把总收益归因于某一个超参数。开发参数搜索仅使用 60000–60004，完整候选结果在 `pilot/profile_search.json`；随后冻结 `expert_config.json`，再运行独立测试。采集数据从种子 60000 开始，因此前 5 条包含开发初态，不能作为未见过初态的评测集。", "",
                  "## 无效动作指标与验证", "", "| 指标（每回合平均） | 原专家 | 高效专家 |", "|---|---:|---:|"])
    for title, key in (("同阶段指令方向反转", "mean_command_reversals_within_phase"), ("同阶段末端实际方向反转", "mean_motion_reversals_within_phase"), ("夹紧后空等步数", "mean_post_close_idle_steps"), ("距离未明显减小步数", "mean_no_progress_steps")):
        lines.append(f"| {title} | {old[key]:.3f} | {new[key]:.3f} |")
    lines.extend(["", "节省主要来自对准和抬升/稳定收敛阶段；接近阶段平均步数反而从 15.13 增为 17.73，因为接近目标的定义和高度已改变，应比较总和。总步数下降的同时，同阶段末端反向微调从 0 增为 0.20 次/回合，距离停滞从 0.27 增为 1.40 步/回合，说明这组参数仍有平滑性方面的优化空间，不能宣称所有指标均有改善。", "",
                  "与旧实验保持相同诊断口径：有效位移至少 0.5 mm、相邻方向夹角超过 100°、间隔不超过 3 步才计方向反转，阶段切换单独处理；距离停滞仅检查接近/对准阶段，5 步窗口内未改善 1 mm 且距离仍大于 12 mm；空等排除有效保持，沿用原累计闭合至少 8 步的统计起点。更早闭合期间的实际双指变化可以直接从逐步 CSV 查看。", "",
                  f"已核验 {audit['paired_cases']} 个配对测试回合、{audit['demonstration_cases']} 个采集回合和 {audit['representative_cases']} 个代表回合：时序连续、动作/指令一致、NPZ 与 CSV 对齐、相同测试初态、全部成功轨迹的最后 40 步均满足原判定。结果在 `audit.json`。", "",
                  "本次没有更新 RGB-ACT / Mask-ACT / RGB+Mask-ACT 权重。若后续用这批短轨迹训练，需要将验证时的阶段执行器同步到本次较低的接近/抬升目标与状态切换规则；旧执行器会覆盖接近高度并将抬升目标截到 z≥0.920 m，直接复用会丢失这次的节步收益。", "",
                  "服务器目录：`/home/catlab/mujoco_robosuite/experiments/colorlift_short_expert_v1`。复现入口：`python -m colorlift.short_trajectories --gpu 2`（已完成的目录会拒绝覆盖）；统计验证：`python -m colorlift.short_report`。", ""])
    (OUTPUT / "REPORT.zh-CN.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
