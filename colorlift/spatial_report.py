"""Chinese report, scientific figures and synchronized physical counterfactual videos."""
import csv
import json
import math
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import imageio.v2 as imageio
from .spatial_gate import RUN
from .spatial_evaluate import MODELS, LEVELS

ZH = {"easy": "Easy", "medium": "Medium", "hard": "Hard", "ood_position": "OOD位置组合", "ood_light": "OOD光照", "ood_camera": "OOD相机"}
FAMILY = {"baseline": "原模型", "curriculum": "课程顺序", "shuffled": "打乱顺序"}


def percent(value):
    return f"{100*value:.2f}%"


def composite(cases, destination, captions):
    readers = [imageio.get_reader(str(RUN / row["directory"] / "rollout.mp4")) for row in cases]
    lengths = [reader.count_frames() for reader in readers]
    cols = 3 if len(cases) in (3, 6) else 2
    rows = math.ceil(len(cases)/cols)
    try:
        with imageio.get_writer(str(destination), fps=20, codec="libx264", quality=8) as writer:
            for tick in range(max(lengths)):
                panels = []
                for reader, count, meta, caption in zip(readers, lengths, cases, captions):
                    panel = np.full((304, 512, 3), 250, np.uint8)
                    panel[48:] = cv2.resize(reader.get_data(min(tick, count-1)), (512, 256), interpolation=cv2.INTER_NEAREST)
                    cv2.putText(panel, caption, (7, 18), cv2.FONT_HERSHEY_SIMPLEX, .43, (25, 25, 25), 1)
                    label = ('PASS' if meta["success"] else 'FAIL')+f' | {meta["control_steps"]} steps | slot{meta["target_physical_slot"]}'
                    if tick >= count-1: label += " | finished"
                    cv2.putText(panel, label, (7, 39), cv2.FONT_HERSHEY_SIMPLEX, .43, (20, 115, 30) if meta["success"] else (190, 30, 30), 1)
                    panels.append(panel)
                while len(panels) < rows*cols: panels.append(np.full((304, 512, 3), 250, np.uint8))
                canvas = np.concatenate([np.concatenate(panels[r*cols:(r+1)*cols], axis=1) for r in range(rows)], axis=0)
                writer.append_data(canvas)
    finally:
        for reader in readers: reader.close()


def main():
    audit = json.loads((RUN / "audit.json").read_text()); assert audit["passed"]
    gate = json.loads((RUN / "gate/metrics.json").read_text())
    result = json.loads((RUN / "results/metrics.json").read_text())
    manifest = json.loads((RUN / "data/manifest.json").read_text())
    recommendation = json.loads((RUN / "recommended_model.json").read_text())
    recommended = recommendation["model"]
    family_rows = {(row["family"], row["level"]): row for row in result["families"]}
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 8.7), constrained_layout=True)
    x = np.arange(6); width = .24
    for i, (family, color) in enumerate((("baseline", "#96a0ad"), ("shuffled", "#548dc8"), ("curriculum", "#dd9d3d"))):
        values = [100*family_rows[family, level]["mean_success_rate"] for level in LEVELS]
        low = [values[j]-100*family_rows[family, level]["seed_min_success_rate"] for j, level in enumerate(LEVELS)]
        high = [100*family_rows[family, level]["seed_max_success_rate"]-values[j] for j, level in enumerate(LEVELS)]
        axes[0].bar(x+(i-1)*width, values, width, label=family, color=color, yerr=[low, high], capsize=3)
        axes[1].bar(x+(i-1)*width, [100*family_rows[family, level]["mean_contact_rate"] for level in LEVELS], width, label=family, color=color)
    for ax, label in zip(axes, ("Target lift success (%)", "Contact-qualified collision-free success (%)")):
        ax.set_ylim(0, 106); ax.set_ylabel(label); ax.set_xticks(x, ["Easy", "Medium", "Hard", "OOD\nposition", "OOD\nlight", "OOD\ncamera"])
        ax.grid(axis="y", alpha=.2); ax.set_axisbelow(True); ax.legend(loc="lower left", ncols=3)
    fig.suptitle("Spatial generalization: same data/budget, curriculum vs shuffled batch order\nTrained bars average two seeds; whiskers show seed range", fontsize=13)
    fig.savefig(RUN / "spatial_generalization.png", dpi=170); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10.5, 5), constrained_layout=True)
    for name in MODELS[1:]:
        path = RUN / "models" / name / "validation.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        ax.plot([r["step"] for r in rows], [r["balanced_prior_action_loss"] for r in rows],
                color="#dd9d3d" if name.startswith("curriculum") else "#548dc8", linestyle="-" if name.endswith("43") else "--", label=name)
    for step in (1500, 3000): ax.axvline(step, color="#777777", linestyle=":", alpha=.6)
    ax.set(xlabel="Optimizer updates", ylabel="Same balanced validation prior action loss", title="Validation, not training-batch difficulty, compares learning curves")
    ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.savefig(RUN / "curriculum_learning_curves.png", dpi=170); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
    for ax, level, color in zip(axes, LEVELS[:3], ("#548dc8", "#dbad50", "#ca7973")):
        points = []
        for row in manifest["layouts"]:
            if row["split"] != "train" or row["level"] != level: continue
            meta = json.loads((RUN / row["episodes"][0] / "metadata.json").read_text()); points.extend(meta["layout"]["xy"])
        points = np.array(points)
        ax.scatter(points[:, 0], points[:, 1], s=10, alpha=.6, color=color)
        ax.set(xlim=(-.16, .16), ylim=(-.16, .16), aspect="equal", xlabel="World x (m)", ylabel="World y (m)", title=f"{level}: 60 independent train layouts")
        ax.grid(alpha=.2)
    fig.suptitle("Physical position distributions; marker colors indicate difficulty, not object color", fontsize=12)
    fig.savefig(RUN / "physical_layout_distributions.png", dpi=170); plt.close(fig)
    tests = [json.loads(p.read_text()) for p in (RUN / "evaluations/test").glob("*/*/*/*/metadata.json")]
    cf = [json.loads(p.read_text()) for p in (RUN / "evaluations/counterfactual").glob("*/*/*/*/*/metadata.json")]
    highlights = RUN / "highlights"; highlights.mkdir(exist_ok=True)
    first_seed = 2300000
    selected = sorted([row for row in cf if row["actor"] == recommended and row["seed"] == first_seed], key=lambda r: (r["layout"]["color_assignment_index"], r["target_color"]))
    fixed_red = [next(row for row in selected if row["layout"]["color_assignment_index"] == p and row["target_color"] == "red") for p in range(6)]
    fixed_slot = [next(row for row in selected if row["layout"]["color_assignment_index"] == p and row["target_physical_slot"] == 0) for p in range(6)]
    same_rgb = [next(row for row in selected if row["layout"]["color_assignment_index"] == 0 and row["target_color"] == color) for color in ("red", "yellow", "blue")]
    composite(fixed_red, highlights / "counterfactual_fixed_red.mp4", [f'perm{p}: target red; color moves across fixed slots' for p in range(6)])
    composite(fixed_slot, highlights / "counterfactual_follow_slot0.mp4", [f'perm{p}: target {row["target_color"]}; same physical slot0' for p, row in enumerate(fixed_slot)])
    composite(same_rgb, highlights / "same_RGB_change_instruction.mp4", [f'same RGB, target {row["target_color"]}' for row in same_rgb])
    family_best = {}
    for family in ("curriculum", "shuffled"):
        candidates = []
        for seed in (43, 44):
            name = f"{family}_seed{seed}"
            candidates.append((json.loads((RUN / "models" / name / "completed.json").read_text())["best_val"], name))
        family_best[family] = min(candidates)[1]
    hard_cases = [next(row for row in tests if row["actor"] == name and row["seed"] == 2220000 and row["target_color"] == "red")
                  for name in ("baseline", family_best["shuffled"], family_best["curriculum"])]
    composite(hard_cases, highlights / "hard_baseline_shuffled_curriculum.mp4", [row["actor"]+"; same held-out hard scene" for row in hard_cases])
    for index, level in enumerate(LEVELS):
        row = next(row for row in tests if row["actor"] == recommended and row["seed"] == 2200000+index*10000 and row["target_color"] == "red")
        shutil.copy2(RUN / row["directory"] / "rollout.mp4", highlights / f"recommended_{level}.mp4")
    fig, axes = plt.subplots(3, 2, figsize=(10, 8), constrained_layout=True)
    for ax, row in zip(axes.flat, fixed_red):
        ax.imshow(imageio.imread(RUN / row["directory"] / "initial.png")); ax.axis("off")
        ax.set_title(f'perm{row["layout"]["color_assignment_index"]}: red at slot{row["target_physical_slot"]}', fontsize=10)
    fig.suptitle("Actual MuJoCo recoloring: physical body positions/orientations stay identical", fontsize=12)
    fig.savefig(RUN / "counterfactual_same_geometry.png", dpi=160); plt.close(fig)
    lines = ["# 三色抓取门槛、空间泛化与课程学习实验", "", "完成日期：2026-10-30。训练和所有测试均在 catlab 的真实 MuJoCo 闭环中执行；最多使用 GPU 1、2。",
             "", f'原任务三色均通过大样本门槛，随后完成4个模型的训练与独立评测。Hard目标成功率从原模型{percent(family_rows["baseline", "hard"]["mean_success_rate"])}提高到课程{percent(family_rows["curriculum", "hard"]["mean_success_rate"])}、打乱{percent(family_rows["shuffled", "hard"]["mean_success_rate"])}；相机OOD从{percent(family_rows["baseline", "ood_camera"]["mean_success_rate"])}提高到{percent(family_rows["curriculum", "ood_camera"]["mean_success_rate"])}、{percent(family_rows["shuffled", "ood_camera"]["mean_success_rate"])}。新模型数字均为两个训练种子的均值。',
             "", "课程和打乱的相对效果随难度变化，不能认定课程普遍更优。Medium及部分接触指标没有全面提升；Hard/OOD仍未全面达到原任务98%成功、95%无非预期接触的标准。原任务门槛通过与新难度全部达标是两个不同结论。",
             "", "## 先测试、后训练", "", "冻结此前的恢复 ACT 和偏向效率执行器，预先固定1000个新独立布局，每个布局分别指定红、黄、蓝，共3000次。没有按局部比例提前停止。逐色同时达到成功率≥98%、无非预期接触成功率≥95%、观察到抓错颜色为0后，才采集本轮训练示范并开始训练。",
             "", "|颜色|目标成功|成功率|无非预期接触成功|该成功率|抓错颜色|成功平均步数|", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in gate["by_color"]:
        lines.append(f'|{row["color"]}|{row["successes"]}/1000|{percent(row["target_success_rate"])}|{row["contact_qualified_successes"]}/1000|{percent(row["contact_success_rate"])}|0/1000|{row["mean_steps_success"]:.2f}|')
    lines += ["", "成功仍要求目标相对初态抬升>4cm、连续40个20Hz步稳定保持（2秒），线速度≤0.025m/s、角速度≤0.4rad/s、漂移≤5mm。总步数包含40步保持；平均步数只对成功回合计算。接触按500Hz原有力阈值判定：机器人相关5N、物体间1N，允许正常目标指垫抓取；无非预期接触成功以全部尝试为分母，并要求抬升前真实目标双侧力接触及最后40步持续目标抓取。",
              "", "0/1000表示本次观察未抓错，有限样本不能证明总体概率绝对为零。逐色Wilson95%区间保存在gate/metrics.json； pooled区间按1000个布局聚类，不把三个颜色指令当作三个独立布局。",
              "", "## 新物理示范与分割", "", f'真实候选布局 {manifest["candidate_unique_geometries"]} 个，最终选择训练180个、验证30个独立几何布局。每个几何布局进行3种循环颜色分配×3个目标指令，共9次实际物理执行，形成1620条训练、270条验证示范。每个物理槽位恰好对应过红、黄、蓝目标；每个颜色在各槽位的次数完全相等。示范数量1890不能当作1890个布局。',
              "", "训练几何先生成，再由独立随机数分配颜色。模型输入只有双相机RGB、机器人/阶段状态和可学习颜色指令；不输入难度、布局ID、目标真值位置、body ID或实际相机外参。专家控制和标签可以读取目标真值，学生只学习RGB和本体状态到动作的映射。HSV仅作训练分割标签和日志，不给学生提供启发式目标定位。",
              "", "每个候选布局只有完整9回合均真实成功、无异常接触且满足抓取要求才有资格入选；按seed顺序取固定名额，候选失败和未选中记录全部保留。每个已选几何另外逐步重放3条原始OSC目标路径，共630次；其余颜色版本也各自真实执行，并核验同物理槽位的动作一致性。重放机器人和目标位置最大误差均为0。没有用图像随机化、插值或删帧替代新位置上的物理动作；旧570次扰动没有计入本轮新布局。",
              "", "|级别|物理定义|训练/验证独立布局|", "|---|---|---:|",
              "|Easy|中央三个较疏锚点，每顶点±2cm扰动，中心距≥10.8cm|60 / 10|",
              "|Medium|簇中心±4cm平移及随机旋转，物块中心距约7.8–9.2cm|60 / 10|",
              "|Hard|目标位于机器人可达工作区边缘，坐标绝对值12.5–14.5cm，方块yaw±0.30rad|60 / 10|",
              "", "Hard的边缘指机器人当前控制器的可达工作区，已做IK可达性预检。动作仍为固定工具朝向的四阶段OSC，不把结果外推到任意抓取姿态。所有数据按几何ID划分，换色、目标指令和相机/光照版本不会跨训练/验证/测试集合。",
              "", "## 课程顺序对照", "", "两种方法均从同一已冻结ACT checkpoint微调，使用相同1890条示范、相同5000×36样本预算、优化器、学习率及分阶段监督长度；各运行种子43、44。先生成课程批次计划，再打乱完全相同的5000个批次作为对照，逐帧曝光多重集合经审计一致。",
              "", "|更新次数|Easy / Medium / Hard采样|", "|---|---|", "|1–1500|100% / 0 / 0|", "|1501–3000|30% / 70% / 0|", "|3001–5000|10% / 30% / 60%|",
              "", "本轮采用固定课程并监测同一均衡验证集。阶段闭环验证保存1500/3000/5000步模型，最终模型从3500/4000/4500/5000中按验证动作损失选择；没有按最终测试结果调训练顺序或选择checkpoint。对准/接触监督前缀2，自由空间4；部署均使用事件执行4/2/1/4。阶段采样为接近/对准/夹紧/抬升保持10/35/35/20%，两组一致。",
              "", "## 独立空间测试", "", "每模型150个新几何布局×3个颜色指令=450次；Easy/Medium/Hard各30个布局，OOD位置/光照/相机各20个。旧模型加4个微调模型共2250次。OOD未进入训练或验证，测试布局只按预先几何和可达性约束生成，不按策略成功与否筛选。",
              "", "OOD位置为未训练的近共线组合；OOD光照修改MuJoCo灯光ambient/diffuse强度和色温；OOD相机位置±1.5cm、注视点±1cm、fovy±3度，是真实相机参数扰动。训练阶段未用这些光照和相机增强。光照/相机组各使用自身的新几何组合，评估的是该综合OOD条件，不能仅比较不同组的百分比来分离纯渲染扰动效应。",
              "", "|难度|原模型成功/无异常接触|打乱顺序两seed均值|课程顺序两seed均值|", "|---|---|---|---|"]
    for level in LEVELS:
        cells = []
        for family in ("baseline", "shuffled", "curriculum"):
            row = family_rows[family, level]; cells.append(f'{percent(row["mean_success_rate"])} / {percent(row["mean_contact_rate"])}')
        lines.append(f'|{ZH[level]}|'+'|'.join(cells)+'|')
    lines += ["", "两seed是训练重复，不增加独立测试布局数。全模型逐色、接触、碰撞、成功平均/P90步数、全尝试平均步数和路径在results/comparison.csv；阶段闭环验证在stage_validation.csv。不同组的成功案例可能不同，不能仅以成功平均步数判断整体效率。新增数据相对旧模型的收益包含新示范和微调，课程顺序的额外作用应看下面同数据对照。",
              "", "|模型|空间测试成功|无非预期接触成功|空间测试错色|反事实错色|", "|---|---:|---:|---:|---:|"]
    for name in MODELS:
        rows = [row for row in tests if row["actor"] == name]
        cf_rows = [row for row in cf if row["actor"] == name]
        lines.append(f'|{name}|{sum(row["success"] for row in rows)}/450|{sum(row["eligible"] for row in rows)}/450|{sum(row["wrong_color_grasp"] for row in rows)}/450|{sum(row["wrong_color_grasp"] for row in cf_rows)}/216|')
    lines += ["",
              "", "|级别|课程−打乱：目标成功百分点|布局聚类95%区间|接触成功百分点|", "|---|---:|---|---:|"]
    for row in result["paired_order_effects"]:
        interval = row["success_pp_cluster95"]
        lines.append(f'|{ZH[row["level"]]}|{row["curriculum_minus_shuffled_success_pp"]:+.2f}|[{interval[0]:+.2f}, {interval[1]:+.2f}]|{row["curriculum_minus_shuffled_contact_pp"]:+.2f}|')
    lines += ["", "区间按独立几何布局聚类，先在每布局内平均两个训练种子和三色指令；仅两训练种子，训练随机性的变化另以seed范围展示，不能据此证明课程在所有训练条件下更优。逐色Wilson区间保留有限样本不确定性；零失败时非参数bootstrap可能退化。",
              "", "## 真实反事实测试", "", "另外12个新几何布局，每布局保持所有物理body位置、朝向不变，进行全部6种颜色交换×3种目标指令，每模型216次。5模型共1080次。重新映射实际可见颜色对应的body/geom评分身份，学生没有读取这些ID。审计确认同一排列只改变指令时初始RGB相同，换色时只有颜色像素变化，物理body状态相同。",
              "", "|模型|目标成功|无异常接触成功|首次目标点靠近正确物体|18回合全成功布局|", "|---|---:|---:|---:|---:|"]
    for row in result["counterfactual"]:
        lines.append(f'|{row["model"]}|{row["successes"]}/216|{row["contact_successes"]}/216|{row["first_goal_correct"]}/216|{row["all18_success_geometries"]}/12|')
    lines += ["", "测试同时覆盖：固定红色指令、换色后应选择不同物理body；固定物理槽位、换色并更新颜色指令后应仍抓该body；固定颜色画面、只换指令时应抓三种不同目标。首次目标点归属与最终抓取成功分开统计；目标点归属同时受感知和空间定位误差影响，不能单独等同于颜色分割识别准确率。",
              "", f'默认示例模型为 **{recommended}**，由训练前约定的验证损失选择（{recommendation["val_score"]:.6f}），不使用正式测试结果选模型。完整权重为models/{recommended}/policy.pt。动作验证损失与实际接触成功率并不完全一致，默认选择不等于在所有新测试条件中最好；所有4个权重及逐模型结果一并交付。',
              "", "## 文件与复现", "", "服务器完整目录：/home/catlab/mujoco_robosuite/experiments/colorlift_spatial_curriculum_v1。data/manifest.json为训练白名单，原始物理示范在对应candidates/目录；被拒绝/未选中记录不参加训练。模型只读选中清单和经审计的缓存。",
              "", "trajectory.npz包含states[T,19]、RGB/masks[T,2,128,128,3]、actions[T,7]规范任务目标、servo_waypoints[T,3]实际阶段路标和osc_actions[T,7]实际float64控制。每步CSV记录末端/目标前后位置、双指状态、动作、实际接触力、事件和步数；全部正式失败回合留在results/failure_cases.json及原目录视频。",
              "", "像素级审计记录：2250次空间测试中，只有一幅首帧与同场景参考图在一个颜色通道相差1灰阶（uint8范围0–255）；物理位置/朝向、本体记录、相机和灯光均相同。审计沿用物理重放预先使用的1灰阶数值容差，原图未修改。该回合额外真实复测仍成功、接触合格且未错色，步数73→72，表明单回合步数有数值敏感性；这一次诊断复测不计入主要分母，原始统计不改。详细记录在audit.json和reset_diagnosis/。",
              "", "```bash", "source /home/catlab/mujoco_robosuite/activate.sh",
              "CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1 python -m colorlift.spatial_demo --gpu 1 --level hard --color red --seed 2400000 --permutation 0 --tag demo", "```",
              "", "保持seed/level不变，更改permutation（0–5）交换颜色；更改color改变目标指令；更换tag保留已有结果。示例视频使用预先固定的首个测试/反事实场景，不搜索最好看的成功案例；同步面板提前结束时只在展示视频中停留最后一帧并标注finished，原始轨迹没有补帧或删步。",
              "", "![新布局](physical_layout_distributions.png)", "", "![学习曲线](curriculum_learning_curves.png)", "", "![空间泛化](spatial_generalization.png)", "", "![真实换色](counterfactual_same_geometry.png)"]
    (RUN / "REPORT.zh-CN.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    (RUN / "completed.json").write_text(json.dumps({"complete": True, "gate_trials": 3000, "new_train_geometries": 180,
                                                    "new_val_geometries": 30, "trained_models": 4, "test_and_CF_trials": 3330,
                                                    "phase_validation_trials": 216, "audited": True}, indent=2))
    print("SPATIAL REPORT READY", RUN, flush=True)


if __name__ == "__main__": main()
