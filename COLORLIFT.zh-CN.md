# ColorLift 环境、三种 ACT 与动作诊断

项目在服务器 `/home/catlab/mujoco_robosuite`，环境为 Conda `robosuite`。本次实验保存在 `experiments/colorlift_v1/`。

## 创建环境并读取观测

```python
import colorlift
from colorlift.config import Config
from colorlift.vision import observation_images, observation_masks

env = colorlift.make_env(gpu=1, config=Config(target_color="red"))
obs = env.reset_seed(123)
print(env.color_body_ids)  # 当前 Panda 模型为 red=25, yellow=26, blue=27
print(env.episode_metadata())
rgb = observation_images(obs)   # (2, 128, 128, 3)，正常朝上的 RGB
masks = observation_masks(obs)  # 同形状，三通道依次是红/黄/蓝的二值 HSV mask
robot = obs["robot0_proprio-state"]
env.close()
```

导入 `colorlift` 后也可使用 `robosuite.make("ColorLift", ...)` 或 `robosuite.make("colorlift", ...)`。环境默认提供 `color_front_image`、`color_side_image`、关节/末端/夹爪等本体观测，以及 `target_color_onehot`。robosuite 原图按 OpenGL 排列，上述辅助函数统一翻为正常朝上；不要再重复翻转。

`color_body_ids` 在每次模型重建后更新，不依赖固定整数。`episode_metadata()` 保存颜色与 body ID/body 名、初始位置/高度的对应关系；每条示范和每个测试回合也保存这些信息。策略输入不包含这些真值位置或 ID。

三个方块均为边长4cm、质量64g的单刚体，颜色红/黄/蓝。每次 reset 三者都在 X/Y±0.11m 范围内重新采样，yaw±0.20rad；采样顺序也随机打乱。中心距离至少95mm，并再次检查保守水平包围界不重叠。相同 seed 可以重现三者位置。

## 测评判据

- 默认目标红色。抬升量取“目标当前中心高度减去本回合目标初始中心高度”，要求严格大于0.04m。
- 正确双指抓取接触；允许2个控制步的接触检测抖动宽限。
- 稳定：目标线速度≤0.025m/s、角速度≤0.4rad/s、相对本次保持起点的位置漂移≤5mm。
- 上述条件在20Hz控制采样中连续满足40步，即2秒；条件失效则计时清零并记录保持中断。
- 非目标方块被双指抓取连续3个控制步，记录颜色并终止为失败。
- 碰撞在500Hz物理子步检查。非预期接触包括机器人碰桌/干扰块、非指尖碰目标、机器人自碰撞、物块互碰；正常指尖抓取、物块静置桌面接触不计入。
- 机器人相关碰撞法向力门限5N，物块互碰1N；碰撞与任务成功分开记录，另报告无碰撞成功率。

目标颜色可通过 Config 调整；当前三组模型训练和正式比较针对红色目标。换颜色训练应建立独立实验目录并采集对应目标的示范。

## 四个分动作与 OSC

`approach` 接近 → `align` 精确对准 → `close` 夹紧 → `lift_hold` 抬升并保持。

ACT 输出 XYZ 与四阶段建议。共同执行器使用 OSC_POSE（kp150、阻尼比1），通过机器人自身到位和夹爪稳定状态推进四个阶段，禁止倒退/跳过；有速度限幅、1mm位置死区、目标平滑、夹紧后XY锁定、抬升到位后保持锁定。阶段完成判据不读取物块真值位置。

`macro_actions` 统计进入的阶段数；`control_steps` 是真实20Hz控制步，包含40步必要保持。不会把物理控制过程隐藏成4步。空夹爪持续12步时可以提前失败结束。

可直接使用四个参数化宏动作：

```python
from colorlift.primitives import PrimitiveExecutor, step_primitive
from colorlift.vision import locate_on_table

obs = env.reset_seed(123)
xyz, details = locate_on_table(env, obs)  # HSV + 已知相机标定，投影到已知方块中心平面
executor = PrimitiveExecutor(env.config)
for phase, z, budget in [(0, .945, 40), (1, .818, 35), (2, .818, 16), (3, .94, 120)]:
    obs, transitions = step_primitive(env, executor, obs, phase, [xyz[0], xyz[1], z], budget)
    info = transitions[-1][3]
    if info["success"] or info["wrong_color_grasp"] or info.get("aborted_empty_grasp"):
        break
```

## HSV 和三种视觉结构

HSV 使用 `cv2.COLOR_RGB2HSV`，Hue为0–179；红色阈值跨零点，黄/蓝各用独立区间，随后3×3开运算、过滤小区域。阈值见 `colorlift/vision.py` 和实验产物 `hsv_ranges.json`。

| 结构 | 每相机输入 | 融合方式 |
| --- | --- | --- |
| RGB-ACT | RGB三通道 | 标准视觉骨干 |
| Mask-ACT | 红/黄/蓝三个HSV二值通道 | 相同视觉骨干，Mask独立归一化 |
| RGB+Mask-ACT | RGB+HSV六通道 | 早期融合，扩展ResNet首层，Mask权重零初始化 |

三组共享19维输入状态（关节7、夹爪2、末端XYZ3、当前阶段4、目标颜色3），相同100条训练/20条验证示范、8000次更新、batch32、chunk16、seed42、每3步查询一次。使用官方ACT的ResNet18、Transformer/CVAE与32维隐变量，输出XYZ和阶段logits。损失为 XYZ L1 + 0.25阶段CE + 10KL。

示范与 HSV+OSC 参考方法使用HSV分割和相机标定定位目标，不使用物块真值位置控制。物体真值只用于评测、诊断距离和检查HSV定位误差。Mask不是渲染器真值分割。

## 诊断指标

每回合保存 `eval_test/episodes.csv`，每控制步保存 `episode_NNN_steps.csv`；指标定义见 `metric_dictionary.zh-CN.json`。

- `command_reversals_within_phase`：同阶段内，幅度≥0.5mm的两次OSC平移方向夹角>100°，间隔≤3步。另记录包含跨阶段的反转，以及实际末端位移反转。
- `post_close_idle_steps`：闭合命令至少8步后，夹爪停止变化、末端移动<0.5mm，且不在有效稳定保持计时内的空等步。
- `empty_gripper_wait_steps`：上述空等中，夹爪总开口<6mm的子集。
- `no_progress_steps`：仅接近/对准阶段，末端到目标距离>12mm，过去5步改善<1mm的步数；距离真值仅作日志，不反馈控制。
- `holding_failure_events`：有效保持中断的事件数；`hold_failure_rate` 是已抬起但最终未完成2秒保持的失败比例。
- `collision_steps` / `collision_physics_steps`：有超阈值非预期接触的控制步 / 物理子步；`collision_events` 合并连续控制步。
- `necessary_hold_steps`：成功回合固定40步；`non_mandatory_control_steps` 为总步数减去必要40步。失败早停不会算作成功效率提升。

## 运行与超参数试验

```bash
source /home/catlab/mujoco_robosuite/activate.sh
python -m colorlift.verify --gpu 1
python -m colorlift.collect --gpu 1 --train-count 100 --val-count 20
python -m colorlift.run --gpus 0 1 --steps 8000
```

调度器最多接受两张不同物理GPU。默认两张卡各一个训练进程，三组都完成后运行HSV基线和三组ACT的30个固定新初态评测。已完成结果会保留，训练支持 `--resume`。

模型超参数试验示例，使用独立输出以保留基线：

```bash
export CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1
python -m colorlift.train --variant rgb_mask --lr 5e-5 --chunk-size 8 --phase-weight .5 --output experiments/colorlift_v1/tuning/fusion_trial
python -m colorlift.evaluate --variant rgb_mask --checkpoint experiments/colorlift_v1/tuning/fusion_trial/best.pt --split dev --episodes 6 --output experiments/colorlift_v1/tuning/fusion_trial/eval_dev
python -m colorlift.aggregate
```

`--lr-backbone`、`--kl-weight`、`--batch-size`、`--seed`、`--steps` 也可单独指定。

OSC增益/阻尼、分阶段位移限幅、死区、平滑、阶段推进、提前结束和诊断阈值在 `experiments/colorlift_v1/config.json`。可复制该文件修改，然后用 `evaluate --config /path/trial_config.json --output /path/trial_eval` 对同一权重在开发初态下评测。正式对照固定参数，不覆盖它。

`lift_height` 是专家抬升目标绝对Z；ACT执行抬升目标的上下界为桌高加 `lift_goal_min_offset` / `lift_goal_max_offset`，本次分别为0.12/0.22m。要研究降低抬升高度以减少移动步数，可在开发试验配置中一并调整这些值，保持4cm以上及2s稳定判据不变。接近、抓取和抬升高度均为绝对坐标，修改桌高/方块尺寸时需一起调整。

改变任务设置或重新采集示范时，先设置新目录：

```bash
export COLORLIFT_RUN_DIR=/home/catlab/mujoco_robosuite/experiments/colorlift_new_setting
python -m colorlift.collect --config /path/trial_config.json --gpu 1 --train-count 100 --val-count 20
```

## 实验文件

`experiments/colorlift_v1/REPORT.zh-CN.md`、`comparison.csv`、`comparison.png`、`diagnostics.png` 是实测对照。`tuning_all_runs.csv` 合并结果与完整超参数，便于筛选和画图。

每个结构目录下有训练日志、`best.pt`/`last.pt`、精简 `policy.pt` 和完整测试视频/CSV；`data/manifest.json` 保存示范来源、body IDs和初态；`environment/verification.json` 保存正例与错误颜色、保持中断等负例验证。

当前固定光照纯色场景、已知相机标定下，HSV基线用于衡量这些条件中分割和控制的可达表现。训练结果只针对本次数据分布；每组一个训练种子，报告保留成功率区间和全部失败回合。

## 红色方块短轨迹与逐步记录

`experiments/colorlift_short_expert_v1/REPORT.zh-CN.md` 记录了后续逐步审计与高效视觉专家的实测结果。`colorlift.step_recording.StepRecorder` 可包装现有 `env.step`，记录控制前后末端/目标位置、双指位置速度、7维实际OSC动作、9维执行器控制和关节状态，不改控制命令。

30组相同独立初态中，原HSV+OSC与高效专家均30/30成功，总步数均值从96.0降为77.1；另采集50/50成功示范，平均76.36步。判定仍为抬升超过4cm、稳定保持2秒。总步数减少时反向微调和距离停滞计数略有增加，完整指标、前70步区间及每步CSV均在新实验目录。

`demonstrations/attempt_*/trajectory.npz` 保存RGB、HSV mask、本体状态、宏动作及实际OSC动作；目标真实位置仅为诊断。采集与报告入口分别为 `python -m colorlift.short_trajectories --gpu 2` 和 `python -m colorlift.short_report`。采集拒绝覆盖已完成结果。本次没有重训ACT，后续使用短轨迹训练时需同步新的高度和阶段切换规则，详见报告。

## 三色共享 ACT 与颜色命令切换

`experiments/colorlift_conditional_v1/REPORT.zh-CN.md` 是后续三色条件模型的正式结果。使用一个共享 `ConditionalACT`：双相机RGB、本体/阶段状态、32维可学习目标颜色嵌入，配合目标无关的四阶段OSC执行器。HSV仅作识别训练标签与评测标签，推理不用HSV定位或物块真实坐标。动作监督前缀随训练从4增长到8、12步，每3步重新观测规划。

每种颜色180条训练、30条验证示范，均按场景拆分；完成12000次联合训练。独立50场景、每场景分别指定三色的150次单次正式测试：红48/50、黄46/50、蓝48/50，总142/150（94.67%），无抓错颜色。50场景初始RGB和机器人状态逐一相同，只有颜色命令变化；42/50场景三次都完成4cm以上抬升及2秒稳定保持。另有12个非手指部位接触目标的超阈值碰撞回合，无碰撞成功为134/150（89.33%），两种成功率须分开解释。

模型为 `experiments/colorlift_conditional_v1/model/policy.pt`。指定一种颜色自主抓取：

```bash
CUDA_VISIBLE_DEVICES=4 MUJOCO_EGL_DEVICE_ID=4 python -m colorlift.conditional_demo --gpu 4 --color red --seed 76000 --tag demo
```

`--color`可用red/yellow/blue，保持seed相同可测试同一场景；更换tag保留此前结果。全部正式逐步日志、混淆矩阵、失败回合及换色视频都在新实验目录，原始630条示范保留在服务器data目录。最终模型已冻结哈希，正式测试150个实际执行回合通过独立核验，无失败重试。

## 带3cm偏移的专家恢复训练

`experiments/colorlift_recovery_v1/REPORT.zh-CN.md` 保存后续完整实验。先冻结原checkpoint做执行长度对照：1/2/3/6/12步及4/1/1/3自适应策略在相同150次任务中分别成功132/141/142/142/146/139次，单纯缩短执行不保证改善。原3步记录复用，其余750次重新执行，权重哈希不变。

随后采集285条成功专家轨迹，每回合两次约3cm水平随机偏移，均由真实OSC物理动作施加，无qpos瞬移。专家每步从实际关节状态重求阻尼IK并纠偏；570次偏移实际均值2.993cm。4592个扰动动作不作监督，chunk不跨扰动区间。新增每色80条训练、15条验证，旧干净数据保留；接近/对准/夹紧/恢复/抬升保持采样比为5/30/30/30/5，最大训练chunk4，对准和恢复只监督2步，部署每2步重规划。

原始IK路标和OSC命令与ACT最终宏目标分开保存：`trajectory.npz` 保留原过程，`training_macro_labels.npz` 是入训标签。专家标签可用物体真值，学生推理只用RGB、本体/阶段状态和颜色ID。通用恢复执行器仅根据ACT目标与末端误差执行抬高避让，旧、新模型共用同一执行器比较，避免把规则改动当成数据训练收益。

全新50个干净场景（每色50次）中，原模型12步/原模型2步加恢复执行器/新恢复模型分别147/150、144/150、150/150成功；无碰撞成功144/150、140/150、148/150。全新30个扰动场景（每色30次）中，分别37/90、55/90、90/90成功，新模型90/90均无超阈值碰撞。成功仍要求抬升>4cm并连续稳定2秒。先前8个失败案例另行回放全部救回，未计入新场景成功率。720次新场景对照和冻结哈希已通过独立核验。

模型：`experiments/colorlift_recovery_v1/model/policy.pt`；专家数据在同目录`data/`，全部过程CSV和视频保留；报告附执行长度、采样分布、目标误差和新旧视频。

```bash
CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1 python -m colorlift.recovery_demo --gpu 1 --color red --disturbed --seed 83000 --tag demo
```

去掉`--disturbed`运行无外部偏移的抓取，更换颜色可复用同一场景。训练使用最多GPU1、6两张卡，原模型及首次恢复开发失败的模型/日志均保留，没有覆盖。

## 事件驱动阶段与真实运动效率

`experiments/colorlift_event_efficiency_v1/REPORT.zh-CN.md` 保存固定当前恢复ACT权重的后续对照：基线2/2/2/2、全部单步1/1/1/1、长短结合4/1/1/4、偏向效率4/2/1/4。执行长度与模型预测容量分开；当前模型容量为4，没有重复旧预测来伪造更长chunk。

`colorlift.event_control.EventExecutor` 按位姿误差切换阶段，夹紧后必须有双侧真实接触力、非空夹爪开口及手指速度条件才开始抬升。它仅用ACT预测目标、本体状态和模拟触觉；物块真值只用于评分和审计。偏向效率方案还允许水平已对齐时提前进入下降。末端路程按500Hz物理子步积分；总步数仍包含40步稳定保持。

正式50个新场景各指定三色，每方案150次；两组专家和四组ACT共900次。详细成功、接触合格且无碰撞成功、步数、路程、方向反向、夹紧后空等、无进展和前70步分解位于该实验的`test/`。本轮只优化执行器和示范，没有重新训练ACT，也没有注入外部3cm偏移。

高效专家数据在该实验的`data/`：180条训练、30条验证，按场景拆分。仅收录真实成功、满足接触条件、无异常碰撞且原始OSC动作再次在MuJoCo重放成功的轨迹。`trajectory.npz`保留RGB、HSV标签、本体状态、实际阶段路标和float64 OSC动作；`training_macro_labels.npz`提供独立ACT任务目标标签。候选和重放全过程在`candidates/`、`replays/`，完整审计在`audit.json`。

```bash
source /home/catlab/mujoco_robosuite/activate.sh
CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1 python -m colorlift.event_demo --gpu 1 --color red --seed 88000 --scheme efficiency --tag demo
```

`--scheme`可选`baseline`、`single_step`、`hybrid`、`efficiency`，`--actor expert`运行HSV专家。更换颜色并保持seed相同可复用同一场景；更换tag保留已有输出。本实验只使用物理GPU1、2。
