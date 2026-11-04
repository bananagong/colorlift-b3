"""Deliver data/protocol evidence; no final policy scores are produced or inspected."""
import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .spec import ROOT, RUN, CATEGORIES, REGIONS


def main():
    from .motion_metrics import main as motion_metrics
    motion_metrics()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import imageio.v2 as imageio
    audit = json.loads((RUN / "audit.json").read_text()); assert audit["passed"]
    dev = json.loads((RUN / "development_results.json").read_text())
    protocol = json.loads((RUN / "final_protocol.json").read_text())
    manifest = json.loads((RUN / "training_manifest.json").read_text())
    generation = json.loads((RUN / "generation_statistics.json").read_text())
    out = RUN / "results"; out.mkdir(exist_ok=True)
    rows = []
    for key, value in dev["paired_actor_results"].items():
        category, actor = key.split("/")
        rows.append({"category": category, "actor": actor, **{k: v for k, v in value.items() if k != "failure_categories"},
                     "success_rate": value["successes"]/value["episodes"], "contact_rate": value["contact_qualified_successes"]/value["episodes"]})
    with (out / "development_actor_comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (out / "mixture_comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        items = [{"mixture": k, **{key: value for key, value in v.items() if key not in ("weights", "quota_128")},
                  **dict(zip(CATEGORIES, v["weights"]))} for k, v in dev["mixture_candidates"].items()]
        writer = csv.DictWriter(stream, fieldnames=list(items[0])); writer.writeheader(); writer.writerows(items)
    with (out / "train_val_layout_positions.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        items = [{"split": r["split"], "geometry_id": r["geometry_id"], "category": r["layout"]["category"],
                  **{f"slot{i}_{a}_m": r["layout"]["xy"][i][j] for i in range(3) for j, a in enumerate("xy")}} for r in manifest["layouts"]]
        writer = csv.DictWriter(stream, fieldnames=list(items[0])); writer.writeheader(); writer.writerows(items)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.6), constrained_layout=True)
    for ax, split in zip(axes, ("train", "val")):
        for category, color in zip(CATEGORIES, ("#377ca7", "#db9e3a", "#579c69", "#bd6b76")):
            selected = [r for r in manifest["layouts"] if r["split"] == split and r["layout"]["category"] == category]
            xy = np.array([r["layout"]["xy"] for r in selected]).reshape(-1, 2)
            ax.scatter(xy[:, 0], xy[:, 1], s=13, alpha=.55, label=f"{category} ({len(selected)} geometries)", color=color)
        for x0, x1, y0, y1 in REGIONS:
            ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, facecolor="#dcdcdc", edgecolor="#555", hatch="///", alpha=.65))
        ax.set(xlim=(-.155, .155), ylim=(-.155, .155), aspect="equal", xlabel="World X (m)", ylabel="World Y (m)", title=split+" / real reset positions")
        ax.grid(alpha=.2); ax.legend(loc="upper center", bbox_to_anchor=(.5, -.12), ncol=2, fontsize=8)
    figure.suptitle("Broad mixed layouts; shaded continuous regions excluded from every training object\nFinal policy evaluation remains sealed; this plot shows training/validation only")
    figure.savefig(RUN / "broad_training_coverage.png", dpi=160); plt.close(figure)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    x = np.arange(4)
    for i, (actor, color) in enumerate((("act", "#9b9fab"), ("current_expert", "#377ca7"), ("safe_expert", "#db9e3a"))):
        selected = [dev["paired_actor_results"][c+"/"+actor] for c in CATEGORIES]
        axes[0].bar(x+(i-1)*.25, [100*r["successes"]/r["episodes"] for r in selected], .25, label=actor, color=color)
        axes[1].bar(x+(i-1)*.25, [r["mean_success_steps"] for r in selected], .25, label=actor, color=color)
    for ax in axes: ax.set_xticks(x, ["Uniform", "Sobol6D", "Near", "Edge/adverse"]); ax.grid(axis="y", alpha=.2); ax.set_axisbelow(True)
    axes[0].set(ylabel="Target success (%)", ylim=(0, 106)); axes[1].set_ylabel("Successful control steps (incl.2s hold)")
    axes[0].legend(loc="lower left")
    fig.suptitle("Development only: 16 new geometries/category x3 colors, paired across actors\nFrozen ACT checkpoint; no training updates; not final-test evidence")
    fig.savefig(RUN / "development_baseline_and_experts.png", dpi=160); plt.close(fig)
    # A contact sheet and sequential reel use unique training geometries, never
    # final layouts and never multiple camera views of one scene in a panel.
    selected = [r for c in CATEGORIES for r in [r for r in manifest["layouts"] if r["split"] == "train" and r["layout"]["category"] == c][:4]]
    figure, axes = plt.subplots(4, 4, figsize=(11, 12), constrained_layout=True)
    examples = []
    for ax, row in zip(axes.ravel(), selected):
        meta = json.loads((RUN / row["episodes"][0] / "metadata.json").read_text())
        initial = Image.open(RUN / row["directory"] / "initial_two_views.png")
        ax.imshow(initial.crop((0, 0, initial.height, initial.height)))
        ax.set_title(f'{row["layout"]["category"]}\nseed {row["layout"]["seed"]}', fontsize=9); ax.axis("off")
        examples.append(meta)
    figure.suptitle("16 distinct physical training layouts / one front camera per panel\nEach panel has exactly one Panda and three cubes; separate episodes")
    figure.savefig(RUN / "16_distinct_training_layouts.png", dpi=145); plt.close(figure)
    with imageio.get_writer(str(RUN / "16_distinct_expert_layouts_front.mp4"), fps=20, codec="libx264", quality=8) as writer:
        for meta in examples:
            with imageio.get_reader(str(RUN / meta["directory"] / "front.mp4")) as reader:
                for frame in reader: writer.append_data(frame)
    failure_rows = []
    for path in (RUN / "development").glob("*/episodes/*/act/*/metadata.json"):
        meta = json.loads(path.read_text())
        if not meta["success"] or not meta["eligible"]: failure_rows.append(meta)
    (out / "development_ACT_failure_cases.json").write_text(json.dumps(failure_rows, indent=2))
    totals = {actor: {key: sum(dev["paired_actor_results"][c+"/"+actor][key] for c in CATEGORIES)
                     for key in ("episodes", "successes", "contact_qualified_successes", "wrong_colors", "collision_episodes")}
              for actor in ("act", "current_expert", "safe_expert")}
    expert_means = {actor: float(np.mean([dev["paired_actor_results"][c+"/"+actor]["mean_success_steps"] for c in CATEGORIES])) for actor in ("current_expert", "safe_expert")}
    comparison = "\n".join(f'|{c}|'+"|".join(f'{dev["paired_actor_results"][c+"/"+a]["successes"]}/48' for a in ("act", "current_expert", "safe_expert"))+"|" for c in CATEGORIES)
    mix_table = "\n".join(f'|{name}|'+"/".join(str(int(w*100)) for w in v["weights"])+f'|{v["XY_grid_occupancy"]:.4f}|{v["six_dimensional_centered_discrepancy"]:.5f}|{v["score"]:.5f}|' for name, v in dev["mixture_candidates"].items())
    attempted = sum(r["executed_expert_episodes"] for r in generation)
    rejected_groups = sum(r["rejections"].get("expert_group_failed_despite_valid_layout", 0) for r in generation)
    report = f'''# 广域布局与封存评测协议 v2

本轮完成数据分布、Valid检查、开发实验和新的评测地基。没有修改ACT结构、没有更新ACT权重，也没有进行新几何辅助头、安全示范或接触恢复的ACT训练。旧结果保留为开发证据。

## 完成的数据与时间顺序

先冻结开发方案；采集四类各64个有效开发布局。在其中每类前16个布局上，旧ACT、旧专家和安全专家均执行三色指令，共576次真实MuJoCo闭环。旧ACT成功{totals['act']['successes']}/{totals['act']['episodes']}（84.38%），不是最终测试成绩。它们来自64个独立布局，不能当成192个布局。旧分布99.8%左右的门槛成绩不再外推到这批新布局；本轮同时扩大位置与yaw范围，并固定增加了静置过程，不能把差异仅归因于某个单独因素。

随后根据开发数据选择候选配比，冻结final_protocol.json，时间{protocol['frozen_at_UTC']}；之后才生成192个训练、48个验证几何及768个最终几何。训练清单2160条示范全部真实执行且严格接触合格。共实际执行{attempted}条数据候选轨迹，{rejected_groups}个物理有效几何因专家组不合格而未进训练清单，原始失败保留。

最终测试集仅执行了布局有效性验证，策略推理次数为0，未参与模型选择。768个布局分别为Broad-IID、Spatial-OOD、Relation-OOD各256个。封存前后哈希已审计；最终集的坐标和首帧保留在服务器final_sealed/，公开交付件只提供分组ID、计数与哈希，不把其几何画面用于本轮开发。

## Valid的操作定义

- 旋转方形足迹用分离轴检查，投影间隙至少3mm；固定4cm方块、固定桌高0.8m。
- 真实静置0.6秒：漂移≤1.5mm，高度偏差≤1.5mm，倾斜≤0.04rad，物体线/角速度≤0.01m/s、0.1rad/s，无非预期接触。静置为reset准备，不计入任务步数，各方法相同。
- 静置后的初始帧中，两个128像素相机各至少6个可见像素、距画面边缘≥2像素；可见面积/投影面积的最好视角≥0.45、另一视角≥0.12。分割真值只供生成器检查，不保证运动中全程无遮挡。
- 三个物理槽位都检查预抓取、抓取和抬升IK，位置误差≤1mm、朝向误差≤0.025rad及关节限位。
- 在完整机器人碰撞模型上，对每段规划关节路径采样8次，检查张开夹爪的下降路径、闭合通道和抬升通道；指掌与干扰物预留至少1mm。夹爪闭合方向由真实双指几何计算。重叠、不可达、不可见和夹爪干涉反例检查通过。

基本抓取可行性是固定工具朝向、有限采样的几何/运动学检查，并不保证实际动态抓取成功。有效测试布局不能因为某个专家失败而被剔除。入选训练示范则另要求原始成功判据：目标抬升>4cm并连续稳定2秒、抬升前真实双指接触、最终40步持续正确抓取且无异常接触。Native错色确认仍为连续3个控制样本。

本轮“不利接近方向”指相邻物体相对通道的方位和物体yaw变化；机械臂工具朝向仍固定，预抓取旋转标签因此基本固定。这份数据不能用来证明任意末端接近姿态的泛化。

## 混合分布的开发比较

工作空间为每轴±14.5cm。广域均匀直接联合采样6个XY坐标；Sobol也对6维联合采样；近距类别构造80–100mm的近邻对，第三物体自由采样；边缘/不利类别在115–145mm边缘带选择一个物体并随机邻近干扰物方位。均保留几何、稳定、可见和夹爪约束，不再依靠固定三个锚点或单一等边三角形。

每个候选配比从同一256布局开发池重复抽取128个几何，重复100次。评分预先固定为0.45×XY网格覆盖+0.20×关系直方图熵+0.25×安全专家接触合格率+0.10×提案有效率。以下是等布局预算的结果：

|候选|均匀/Sobol/近距/边缘 %|10×10网格覆盖|6维CD，越小越好|开发评分|
|---|---|---:|---:|---:|
{mix_table}

选定候选为35/40/15/10，指最终接受的布局比例。无效提案要额外采样，因此原始提案比例与入选比例不同；边缘类别有效接受率较低。这是本轮开发数据质量代理目标下的选择，不是已证明的ACT最优训练比例。尚未做不同配比的ACT微调消融。

重采样稳定性另见development_diagnosis.json。所选配比相对均衡配比的评分差，100次配对重采样的2.5–97.5百分位区间包含0；因此不能宣称已稳定胜过均衡配比。这些区间也不是ACT抓取成功率的置信区间。

Sobol使用scipy.stats.qmc.Sobol(d=6,scramble=True).random_base2(16)。[SciPy官方说明](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.Sobol.html)强调2的幂次样本的平衡性质；有效性拒绝和配额抽样后不再保证保有该性质，所以报告筛选结果的实际覆盖和6维差异度。

## 新的三种互斥支持域

|集合|位置支持|关系支持|最终独立布局|
|---|---|---|---:|
|Broad-IID|与训练相同的广域混合分布，排除保留空间区|中心最小间距≥80mm，面积比≥0.080|256|
|Spatial-OOD|恰有一个目标物理槽位位于保留连续区域内，干扰物在训练支持区|与训练相同|256|
|Relation-OOD|所有物体均在训练空间支持区|128个近距62–74mm；128个面积比≤0.025的近共线组合|256|

保留连续矩形为x∈[-.145,-.100], y∈[.050,.145]及x∈[.100,.145], y∈[-.145,-.050]。训练和Broad-IID的所有物体中心都排除矩形及8mm缓冲；Spatial-OOD目标中心位于矩形内至少3mm。Spatial-OOD通过同一联合提案的条件拒绝得到，不覆盖单个坐标后破坏物块关系。三种集合不叠加光照或相机扰动，以免把不同OOD因素混在一起。Relation-OOD的近距和共线结果还应分别报告。

Broad-IID是相对于共同提案和Valid分布的定义；实际训练示范还经过专家成功筛选，不能保证它与成功筛选后的经验分布完全相同。最终测试仍保留所有有效布局，不因专家失败剔除。

旧任务采样范围±11cm与上述保留区域有交集，几何ID不同不足以保证旧ACT从未见过这些区域。因此额外冻结training_initialization_policy.json：后续正式Spatial-OOD模型对照使用各配对种子内共同的新ACT初始化，或可证明未接触保留区的初始化；旧任务ACT权重只作为开发参考，不作为干净的未见区域初始化。通用视觉预训练如使用，所有候选必须相同并记录来源/哈希。这项训练历史约束与监督时间基准补充均在任何新ACT训练和最终策略推理前锁定；没有改变已经冻结的布局采样、有效性或评分规则。

geometry_id为忽略颜色、yaw和物理编号、按1mm取整后的无序XY三元组；另有1cm取整分组隔离近重复。同一几何的颜色、指令、扰动和失败恢复全归同一集合。旧实验、开发/预检及新各集合通过原子注册表排除分组重用。按geometry_id统计，颜色/指令先在几何内部平均，再做布局聚类区间。边界格子相邻仍可能出现接近布局，所以还报告无序位置RMS最近邻统计；分组隔离不等于严格连续距离下限。

## 专家对照与后续顺序

|开发类别|旧ACT成功|原专家成功|安全路径候选成功|
|---|---:|---:|---:|
{comparison}

两个专家均192/192成功且严格接触合格，当前开发样本未显示安全路径额外成功率收益。安全路径先原地升高、高位平移，再垂直下降，闭合前更严格对准；它平均{expert_means['safe_expert']:.2f}步，旧专家{expert_means['current_expert']:.2f}步，均含40步保持。对准误差的改善伴随步数成本。按预定选择规则，本轮示范仍使用旧专家；只有真实合格的困难布局示范入清单。不能把安全候选的192/192解释为在任意布局中都安全。

数据已加入逐步目标相对位移label、规划器给出的安全预抓取位置/旋转及相对位移。supervision.npz中的goal_relative=target_position_before-eef_position_before，逐步与CSV核验最大误差{audit['maximum_goal_relative_label_error_m']:.3g}m。pregrasp_position_label高度为0.925m，是通过IK/通道检查的安全位姿；旧专家实际可能经过较低的0.865m预抓取点，这两个字段分开保存。

提前审计发现原始本体观测EEF与控制边界物理快照有微小差异，全数据最大{audit['raw_observation_vs_physics_delta_max_m']*1000:.3f}mm。原始trajectory.npz、RGB、本体输入和OSC动作全部保留；按照独立冻结的label_alignment_protocol.json，从同一控制时刻的物理末端/目标快照生成单独的supervision.npz。原观测坐标系下的差值也明确保留为诊断字段。没有用放宽容差掩盖区别，也没有重算或更改成功判定。原冻结审计源码保留，带时间基准区分的audit_supervised.py执行后继审计。

开发失败诊断中，30次旧ACT失败有15次未进入闭合阶段；其余15次的闭合时XY误差平均约4.23cm，成功回合平均约7.17mm。这是优先研究定位/接近辅助监督的关联证据，尚不能证明增加辅助头就能改善成功率。data_contract.py将RGB、本体/阶段状态及颜色索引作为唯一inputs，把几何真值放在独立labels中，并拒绝加载final作为监督分割。

trajectory.npz中的actions是规范任务目标XYZ+阶段，phase0标签使用安全预抓取高度；servo_waypoints保存专家实际阶段路标；osc_actions保存真实执行的float64七维OSC命令。训练几何辅助头之前应按这份语义选择监督目标，不能把规范任务目标当成逐步真实位移。

results/motion_efficiency.csv逐回合另存实际末端方向反转、命令方向反转、夹紧后空等和距离停滞步数，以及500Hz真实路径长度。方向反转要求位移≥0.5mm、相邻有效向量间隔≤3步且夹角>100度；稳定保持阶段不计作夹紧后空等。

开发提案接受率同时包含分割支持域条件和物理Valid过滤，因此不能把1减接受率解释为场景物理不可行比例。各阶段拒绝原因在proposal_log.jsonl中逐项保留。

后续按以下顺序做独立消融，不同时改变多个因素：

1. 固定此数据、动作执行器和训练预算，先比较原ACT与增加goal-relative L1头，再单独加入预抓取位姿辅助监督。lambda只在开发/验证中选择，正式推理只读RGB、本体状态和颜色指令。
2. 固定第一阶段选定模型，在匹配几何与预算下比较原专家和安全接近示范，记录碰撞收益及步数成本。
3. 再加入匹配布局的接触/滑落恢复数据，单独评价夹持滑落、接触恢复、空等与反向动作次数。

最终策略推理前必须先锁定所有候选checkpoint哈希和消融/超参数/验证选择规则；最终结果不能反过来选择checkpoint。本轮没有提前训练辅助头，也没有对封存测试集运行旧ACT。

[RiEMann](https://proceedings.mlr.press/v270/gao25a.html)使用点云输入的SE(3)等变学习；[SPOT](https://research.nvidia.com/publication/2025-05_spot-se3-pose-trajectory-diffusion-object-centric-manipulation)使用目标中心SE(3)轨迹表示与扩散策略。它们与此RGB-ACT不同，本轮只借鉴几何表示思想，没有移植架构或声称相同收益。

## 审计与交付

audit.json通过。ACT及原有4个模型哈希保持不变；最终768个布局均在协议冻结后生成，没有策略分数或动作数据，哈希封存通过。完整数据位于服务器{RUN}。训练/验证2160条原始示范及失败候选均留在服务器；交付包包含全部清单、步骤CSV、元数据、开发视频、代表性新布局视频、源码及72条原始NPZ示例。最终集原始坐标/图像留在服务器封存目录。

16_distinct_expert_layouts_front.mp4每段为一个不同的新训练布局、同一前相机，不是双相机拼接。development_ACT_failure_cases.json列出开发旧ACT失败及异常接触回放。新数据的所有视频均明确标注单相机。
'''
    (RUN / "REPORT.zh-CN.md").write_text(report, encoding="utf-8")
    (RUN / "phase1_completed.json").write_text(json.dumps({"phase": "data_distribution_and_sealed_protocol", "complete": True,
                                                          "ACT_updates": 0, "final_policy_attempts": 0, "audit_passed": True}, indent=2))
    print(json.dumps({"report": str(RUN / "REPORT.zh-CN.md"), "development_actor_totals": totals, "expert_means": expert_means}, indent=2), flush=True)


if __name__ == "__main__": main()
