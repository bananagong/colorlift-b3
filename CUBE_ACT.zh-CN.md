# 正方体资产与 ACT Lift 实验

服务器项目：`/home/catlab/mujoco_robosuite`，Conda 环境：`robosuite`。

## 正方体资产

```python
from catlab_assets import CubeObject

cube = CubeObject(name="cube", size=0.04, mass=0.064)
```

`size` 是完整边长，单位米；`mass` 单位千克。还可传入 `friction=(0.8, 0.005, 0.0001)`、`rgba=(0.85, 0.12, 0.12, 1)`。默认一个自由关节，`joints=None` 可固定物体。原点与质心均在正方体中心。

源文件：`catlab_assets/cube.py` 和 `catlab_assets/assets/objects/cube.xml`。支持标准 robosuite 放置、命名、接触几何体和任务合并接口。

## 图片中必做项目的对应实现

1. 环境验证：`python -m lift_act.verify_env`，验证双 RGB 相机、控制、资产质量与尺寸、相同种子的重置一致性。
2. 示范与 ACT：`lift_act/collect.py` 采集真实接触仿真中的抓取、抬升和保持轨迹；`lift_act/train.py` 训练原始 ACT 模型的 Panda 适配版本。
3. 对照：20 与 50 条训练示范，其余训练配置相同；使用共同独立验证集和固定测试初态，记录成功率及成功步数。

本次选择图片中的必做 Lift 对照；推 T 是进阶选做，已有 T 资产可在后续任务复用。

## 运行

```bash
source /home/catlab/mujoco_robosuite/activate.sh
bash /home/catlab/mujoco_robosuite/run_lift_experiment.sh
```

脚本将 `CUDA_VISIBLE_DEVICES=1`、`MUJOCO_EGL_DEVICE_ID=1`，训练和渲染只使用物理 1 号 GPU，两组依次训练。没有使用其他卡。脚本复用已有数据和完成记录，中断训练可从 `last.pt` 恢复。

可分别执行：

```bash
export CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
python -m lift_act.collect --gpu 1 --count 50 --val-count 10
python -m lift_act.train --demos 20 --steps 6000
python -m lift_act.train --demos 50 --steps 6000
python -m lift_act.evaluate --demos 20 --gpu 1 --split test --episodes 20
python -m lift_act.evaluate --demos 50 --gpu 1 --split test --episodes 20
python -m lift_act.report
```

最终测试记录已存在时评测器会停止，防止覆盖结果。若需新实验，请在代码中配置新的 `EXPERIMENT` 输出目录，并记录协议变更。

## 协议和数据

协议保存在 `lift_act/protocol.json`：20 Hz、最长 160 步；成功阈值是方块中心 z≥0.90 m，连续保持 10 步。桌高 0.80 m。方块初始 X/Y 在 ±0.055 m、yaw 在 ±0.25 rad 内随机。

训练示范 seed 从 1000 开始，验证示范从 3000 开始，最终测试固定为 5000–5019。20 条训练示范是 50 条的固定子集。示范来自脚本专家，专家可访问方块位置；ACT 推理只使用双 RGB 相机和机器人自身状态。

每条 `.npz` 保存逐帧 `states`、`images`、绝对末端 XYZ 与夹爪 `actions`、物体 `heights` 和采集元数据。图像转为正常朝上的 RGB。归一化均值和标准差只从本组训练数据计算。

## ACT 来源与适配

使用 [tonyzhaozh/act 官方代码](https://github.com/tonyzhaozh/act)，源码快照 commit `742c753c0d4a5d87076c8f69e5628c79a8cc5488` 保存在 `third_party/act/`。上游源码未做算法修改；`lift_act/model.py` 将原 ALOHA 14 维状态与动作层替换为 Panda 所需的 12 维机器人状态、4 维笛卡尔动作层。

保留 ResNet18、Transformer、CVAE、32 维隐变量、动作分块与 L1+KL 损失。配置为 hidden 256、encoder 2 层、decoder 3 层、feedforward 1024、chunk 20，KL 权重 10。采用 ImageNet 视觉骨干初值。

每组 batch 32、6000 次优化更新、seed 42；独立验证集的零隐变量 prior L1 用于选择 best.pt。闭环时每执行 5 步就重新读取图像和机器人状态、预测动作；使用 OSC_POSE 跟踪预测的 XYZ，并保持初始末端方向。没有专家回退。

## 输出

`experiments/lift_act_v1/` 下：

- `environment/verification.json`：资产、相机、重置和控制验证。
- `data/manifest.json`：示范来源、成功状态和采集种子。
- `act_20/`、`act_50/`：配置、训练日志、`best.pt`、`last.pt`；`policy.pt` 为包含归一化统计和配置的精简推理权重。
- 各模型的 `eval_test_q5/`：全部回合视频、末帧、轨迹和 `metrics.json`。
- `comparison.csv`、`comparison.png`、`REPORT.zh-CN.md`：最终实测比较与报告。

基础环境依赖见 `requirements.lock.txt`，新增训练依赖见 `requirements-act.txt`；服务器上还会保存包含 CUDA 依赖的完整训练环境 freeze 文件。`third_party/act-main.tar.gz`、上游 commit 元数据和源文件哈希用于追溯。
