"""Report one policy's single-color success and fixed-scene color interventions."""
from dataclasses import asdict
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import cv2
from PIL import Image
from .conditional import RUN, COLORS, configuration
from .step_recording import save_rows


def load(path): return json.loads(Path(path).read_text())


def protocol():
    RUN.mkdir(parents=True, exist_ok=True)
    p={"target_colors": list(COLORS), "scene": "all three cubes present; static cameras/lighting; no external perturbations",
       "training_scene_seeds": [71000, 71179], "validation_scene_seeds": [72000, 72029],
       "controller_verification_seeds": [73000, 73004], "development_scene_seeds": [74000, 74011],
       "final_test_scene_seeds": [75000, 75049], "episodes_per_test_color": 50,
       "single_attempt": True, "no_retry": True, "success": "correct target grasp, lift > .04m relative to reset, continuous stable hold 2s/40 steps",
       "counterfactual_test": "same scene reset, identical RGB and robot state; only target color ID changes",
       "inference_inputs": ["two RGB cameras", "16D robot/phase state", "integer target color to learned 32D embedding"],
       "HSV_usage": "training pixel-label generation and evaluation metrics only; no HSV target localization or masks as policy inputs",
       "controller": "ACT predicts XYZ/phase chunks; a target-independent four-phase OSC adapter enforces heights and completion gates. Target XY comes only from ACT.",
       "chunk_curriculum": "max capacity 12; supervise 4-step prefix during steps1-4000,8 during4001-8000,12 during8001-12000",
       "inference_replan_every": 3, "max_physical_GPU_count": 2, "physical_GPUs": [2, 4],
       "task_config": asdict(configuration()[0]), "expert_config": asdict(configuration()[1]),
       "checkpoint_selection": "best held-out prior loss within final chunk12 stage; final test not used for selection"}
    (RUN / "protocol.json").write_text(json.dumps(p, indent=2))


def make_report(evaluation="eval_test_selected"):
    metrics=load(RUN / evaluation / "metrics.json")
    manifest=load(RUN / "data/manifest.json")
    train=load(RUN / "model/completed.json")
    cfg=load(RUN / "model/config.json")
    checkpoint=torch.load(RUN / "model/policy.pt", map_location="cpu", weights_only=False)
    embedding=checkpoint["model"]["color_embedding.weight"].numpy()
    initial=checkpoint["embedding_initial"].numpy()
    embeddings={"colors": list(COLORS), "dimension": 32, "embedding": embedding.tolist(),
                "change_from_initial_l2": np.linalg.norm(embedding-initial, axis=1).tolist(),
                "pairwise_distances": np.linalg.norm(embedding[:, None]-embedding[None], axis=-1).tolist()}
    (RUN / "color_embeddings.json").write_text(json.dumps(embeddings, indent=2))
    summary=[{"color": color, **metrics["per_color"][color]} for color in COLORS]
    save_rows(RUN / "success_by_color.csv", [{key: str(value) if isinstance(value, list) else value for key, value in row.items()} for row in summary])
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes=plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    rates=np.array([metrics["per_color"][color]["success_rate"] for color in COLORS])
    intervals=np.array([metrics["per_color"][color]["wilson_95_ci"] for color in COLORS])
    palette=["#d64d4d", "#e3ba27", "#457fc5"]
    axes[0].bar(COLORS, rates*100, color=palette, alpha=.85)
    axes[0].errorbar(COLORS, rates*100, yerr=np.stack((rates-intervals[:, 0], intervals[:, 1]-rates)).clip(0)*100, fmt="none", color=".3", capsize=5)
    axes[0].set(ylim=(0, 114), ylabel="Strict grasp + lift + 2s hold success (%)", title="One conditional ACT: 50 attempts/color")
    for index, color in enumerate(COLORS):
        row=metrics["per_color"][color]
        axes[0].text(index, rates[index]*100+3, f'{row["successes"]}/{row["episodes"]}', ha="center")
    confusion=np.array([[metrics["command_vs_actual_grasp_confusion"][color][picked] for picked in (*COLORS, "none")] for color in COLORS])
    axes[1].imshow(confusion, cmap="Blues", vmin=0, vmax=50)
    axes[1].set(xticks=range(4), xticklabels=(*COLORS, "none"), yticks=range(3), yticklabels=COLORS,
                xlabel="Actual first confirmed grasp", ylabel="Commanded target", title="Object selection under color commands")
    for i in range(3):
        for j in range(4): axes[1].text(j, i, str(confusion[i, j]), ha="center", va="center", color="white" if confusion[i, j]>25 else "black")
    labels=("All 3 selected", "All 3 strict success", "Initial goals correct")
    counts=[metrics["all_three_correctly_selected_scenes"], metrics["all_three_successful_scenes"], metrics["counterfactual_initial_prediction_correct_scenes"]]
    axes[2].barh(labels, counts, color=["#72b7b2", "#54a24b", "#4c78a8"])
    axes[2].set(xlim=(0, 55), xlabel="Fixed scenes (out of 50)", title="Same RGB scene, change target color only")
    axes[2].invert_yaxis()
    for i, value in enumerate(counts): axes[2].text(value+.5, i, f"{value}/50", va="center")
    fig.savefig(RUN / "results.png", dpi=160); plt.close(fig)
    # Show behavior for one successful intervention scene and preserve its actual video.
    chosen=next((row for row in metrics["switching"] if row["all_three_grasps_successful"]), metrics["switching"][0])
    directory=RUN / evaluation / f'scene_{chosen["seed"]}'
    columns=[]
    for color in COLORS:
        final=np.array(Image.open(directory / color / "final_rgb.png"))
        initial_rgb=np.array(Image.open(directory / color / "initial_rgb.png"))
        columns.append(np.concatenate((initial_rgb, final), axis=0))
    gallery=np.concatenate(columns, axis=1)
    Image.fromarray(gallery).save(RUN / "same_scene_switching.png")
    readers=[imageio.get_reader(str(directory / color / "rollout.mp4")) for color in COLORS]
    frames=[[frame for frame in reader] for reader in readers]
    for reader in readers: reader.close()
    with imageio.get_writer(str(RUN / "same_scene_switching.mp4"), fps=20, codec="libx264", quality=7) as writer:
        for step in range(max(map(len, frames))):
            views=[]
            for color, sequence in zip(COLORS, frames):
                frame=sequence[min(step, len(sequence)-1)].copy()
                cv2.putText(frame, f"target={color}", (3, 25), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 0), 1)
                views.append(frame)
            writer.append_data(np.concatenate(views, axis=1))
    stages=[]
    for name in ("chunk4", "chunk8", "selected"):
        dev=load(RUN / f"eval_dev_{name}/metrics.json")
        stages.append({"stage": name, "checkpoint_step": dev["checkpoint_step"], "successes": dev["successes"], "episodes": dev["episodes"],
                       **{color+"_successes": dev["per_color"][color]["successes"] for color in COLORS}})
    save_rows(RUN / "curriculum_dev_comparison.csv", stages)
    failures=[row for row in metrics["trials"] if not row["success"]]
    (RUN / "failure_cases.json").write_text(json.dumps(failures, indent=2))
    clean_successes=sum(row["success"] and row["collision_steps"] == 0 for row in metrics["trials"])
    collision_types=Counter(category for row in metrics["trials"] if row["collision_steps"] for category in set(row["collision_types"]))
    analysis={"strict_successes": metrics["successes"], "episodes": metrics["episodes"], "success_rate": metrics["successes"]/metrics["episodes"],
              "collision_free_successes": clean_successes, "collision_free_success_rate": clean_successes/metrics["episodes"],
              "collision_episodes": sum(row["collision_steps"] > 0 for row in metrics["trials"]),
              "collision_category_episode_counts": dict(collision_types), "failure_outcomes": dict(Counter(row["outcome"] for row in failures)),
              "same_scene_three_color_successes": metrics["all_three_successful_scenes"], "scenes": metrics["scenes"]}
    (RUN / "analysis_summary.json").write_text(json.dumps(analysis, indent=2))
    lines=["# 三色目标条件 ACT：训练与固定场景换色测试", "",
           "训练一个共享权重的 ACT，通过可学习的目标颜色嵌入选择红、黄、蓝物块。红黄蓝同时在场，随机位置不重叠，固定光照与相机，没有外部扰动。测试均为单次抓取，无自动重试。", "",
           f"正式测试使用 50 个未见场景（75000–75049），每个场景分别指定红、黄、蓝，共 150 次。全部测试初始 RGB 哈希、本体状态与物体位置按场景核对一致；切换命令时从同一初态重置。训练阶段对照使用开发集 74000–74011；checkpoint 按验证集先验动作损失选择，正式测试不参与模型选择。", "",
           "## 单次抓取成功率", "", "严格成功保持原测评：正确颜色物块相对初始高度抬升 >0.04m，连续稳定 2s（40步），同时检查双指接触和速度/漂移阈值。", "",
           "| 目标颜色 | 成功/次数 | 成功率 | Wilson 95%区间 | 成功平均步数 | 抓错颜色 | 碰撞回合 | 保持失败 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    chinese=dict(zip(COLORS, ("红", "黄", "蓝")))
    for color in COLORS:
        item=metrics["per_color"][color]
        lo, hi=item["wilson_95_ci"]
        steps=f'{item["mean_success_steps"]:.2f}' if item["mean_success_steps"] is not None else "无成功"
        lines.append(f'| {chinese[color]} | {item["successes"]}/{item["episodes"]} | {item["success_rate"]*100:.1f}% | {lo*100:.1f}%–{hi*100:.1f}% | {steps} | {item["wrong_color_episodes"]} | {item["collision_episodes"]} | {item["hold_failure_episodes"]} |')
    lines.extend(["", f'总严格成功 {metrics["successes"]}/150（{metrics["successes"]/150*100:.2f}%）；无超阈值碰撞的成功为 {clean_successes}/150（{clean_successes/150*100:.2f}%）。碰撞作为独立指标记录，不能把严格成功率理解为无碰撞成功率。共有 {analysis["collision_episodes"]} 个碰撞回合，类别计数为 {dict(collision_types)}；同一回合可能包含多个类别。', "",
                  "碰撞仍使用既有判定：机器人接触法向力≥5N、物体间≥1N，500Hz物理子步采样；正常双指抓取目标和物块静置桌面的接触被排除。无外部干扰不代表策略执行中不会发生接触失误。", "",
                  "![正式测试与换色结果](results.png)", "", "## 同场景换色", "",
                  f'50 个固定场景中，三个颜色命令都抓到对应物块：**{metrics["all_three_correctly_selected_scenes"]}/50**；三次都达到严格成功：**{metrics["all_three_successful_scenes"]}/50**；同一初始图像下三次 ACT 初始目标点都最接近命令对应物块：**{metrics["counterfactual_initial_prediction_correct_scenes"]}/50**。', "",
                  "抓取对象按双指接触同一物块连续至少3个控制采样确认，记录首次确认对象；未确认任何对象记为 none。改变颜色输入后，RGB、初始关节和物块位置保持一致，混淆矩阵统计实际抓取对象，并非只检查目标标签或网络输出。", "",
                  f'示例场景 {chosen["seed"]}，从左到右目标红、黄、蓝；上排初态，下排终态：', "", "![同场景改变颜色](same_scene_switching.png)", "", "视频：`same_scene_switching.mp4`；所有场景的逐一结果：`eval_test_selected/switching.csv`。", "",
                  "## 模型与训练", "",
                  "模型输入是双相机 RGB、16维本体/阶段状态与目标颜色 ID。`nn.Embedding(3,32)` 产生颜色查询：一方面与可学习像素特征匹配生成三色识别图，另一方面拼接到 ACT 本体 token。识别图强调目标区域，其可微像素坐标也进入 ACT。ACT 保留 ResNet18、Transformer 与 CVAE，输出世界系 XYZ+四阶段标签的动作块。", "",
                  "识别使用背景、红、黄、蓝四类互斥的 softmax 与按类别像素数加权的交叉熵；目标识别中心以概率4次方作权重，降低低置信背景累积的影响。HSV 只生成训练识别标签，评测时可用于识别 IoU 的标签对照；策略输入中没有 HSV mask、真实物块坐标、body ID 或解析定位结果。嵌入从动作损失和识别损失共同学习，初始/训练后数值与变化见 `color_embeddings.json`。", "",
                  "使用上一轮高效专家生成三色示范，同一场景有三个目标任务。只将三色均成功且无碰撞的完整场景组用于训练，全部尝试与排除原因保留在 manifest。", "",
                  f'每种颜色训练示范 {manifest["counts_by_color"]["train"]["red"]} 条、验证示范 {manifest["counts_by_color"]["val"]["red"]} 条；共排除 {manifest["excluded_scene_groups"]} 个不完整/失败场景组。训练与验证按场景划分，颜色和阶段均衡抽样。', "",
                  "先进行1800次像素识别预训练，再进行12000次 ACT 联合更新：1–4000次监督4步动作前缀，4001–8000次监督8步，8001–12000次监督12步。最大查询容量固定12，前缀外及回合末尾位置通过 padding 屏蔽，增长时保留模型和优化器状态。", "",
                  "batch=36，AdamW，主干/其余学习率1e-5/1e-4，cosine衰减，KL权重10，阶段CE权重0.25，识别损失权重2，训练种子42。只在最终12步阶段内按验证集先验动作损失选模型；开发/正式测试均用先验 z=0。每3个控制步重新观测和规划，执行动作块的前3步。", "",
                  "底层沿用四分动作 OSC：接近、对准、夹紧、抬升并保持。阶段高度、闭合完成门限、单调推进和动作限幅由通用执行器约束；目标 XY 由 ACT 预测，执行器不能读取目标颜色、HSV 或物块位置。网络预测阶段也写入日志，推进采用机器人自身完成门限。这一结果对应带通用阶段约束的 ACT 系统。", "",
                  f'选中 checkpoint 更新步数：{metrics["checkpoint_step"]}；训练耗时 {train["seconds"]:.1f}s。使用 GPU 2、4，最多两张，采集并行，模型训练为一张卡。', "",
                  "三个训练阶段在同一开发集的结果：", "", "| 动作前缀 | checkpoint步 | 成功/次数 | 红 | 黄 | 蓝 |", "|---|---:|---:|---:|---:|---:|"])
    for row in stages: lines.append(f'| {row["stage"]} | {row["checkpoint_step"]} | {row["successes"]}/{row["episodes"]} | {row["red_successes"]}/12 | {row["yellow_successes"]}/12 | {row["blue_successes"]}/12 |')
    lines.extend(["", "这是同一模型随训练推进的阶段观察，训练步数也同时增加，不能把变化单独归因于分块长度。", "", "## 产物与复现", "",
                  "- `model/policy.pt`：包含颜色嵌入、识别器、ACT 与训练归一化统计的单模型 checkpoint。", "- `eval_test_selected/episodes.csv`、`switching.csv`、`metrics.json`：完整正式测试；每个回合有视频、逐步位置/动作、颜色接触状态和元数据。", "- `failure_cases.json`：全部失败回合，保留视频目录，未删去重跑。", "- `data/manifest.json`：示范来源、场景、颜色、成败与训练资格；原始RGB/Mask示范保存在服务器 data/*.npz。", "- `protocol.json`：固定数据/开发/正式测试范围、成功口径与输入限制。", "",
                  "服务器目录：`/home/catlab/mujoco_robosuite/experiments/colorlift_conditional_v1`。", "",
                  "```bash", "source /home/catlab/mujoco_robosuite/activate.sh", "CUDA_VISIBLE_DEVICES=4 MUJOCO_EGL_DEVICE_ID=4 python -m colorlift.conditional_evaluate --gpu 4 --split test --scenes 50 --name new_replay", "```", "",
                  "指定一种颜色单独运行同一个模型：", "", "```bash", "CUDA_VISIBLE_DEVICES=4 MUJOCO_EGL_DEVICE_ID=4 python -m colorlift.conditional_demo --gpu 4 --color red --seed 76000 --tag demo", "```", "",
                  "目标颜色可替换为 red / yellow / blue；保持 seed 一致可复现同一场景。输入/执行参数还保存在 `inference_config.json`，原始630条示范在服务器 data/*.npz，交付包包含清单与数据哈希，不重复打包原始训练图像。", "",
                  "重评估应使用新输出名，保留正式测试。结果只覆盖固定机器人初态、固定光照、静态三色立方块与本次随机位置范围；未测试外部扰动或新颜色。", ""])
    (RUN / "REPORT.zh-CN.md").write_text("\n".join(lines), encoding="utf-8")
    print("REPORT COMPLETE", json.dumps({"successes": metrics["successes"], "episodes": metrics["episodes"], "switching_successes": metrics["all_three_successful_scenes"]}))
    print("ANALYSIS", json.dumps(analysis))


if __name__ == "__main__":
    import sys
    if "--protocol-only" in sys.argv: protocol()
    else: make_report()
