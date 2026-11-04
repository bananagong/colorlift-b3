# catlab MuJoCo + robosuite 环境

搭建日期：2026-10-28。SSH 别名：`catlab`。

## 已安装

| 项目 | 版本 / 路径 |
| --- | --- |
| 系统 | Ubuntu 24.04.1 LTS，x86_64 |
| Conda 环境 | `robosuite` |
| Python | 3.11.18 |
| MuJoCo | 3.3.7 |
| robosuite | 1.5.2 |
| NumPy | 1.26.4 |
| 环境路径 | `/home/catlab/miniconda3/envs/robosuite` |
| 项目路径 | `/home/catlab/mujoco_robosuite` |
| 渲染后端 | NVIDIA EGL，无需 DISPLAY |

采用固定版本组合。robosuite 上游说明 MuJoCo 3.10 改变了控制器使用的 API，因此升级 MuJoCo 前应重新检查兼容性并运行验证。

上游参考：[robosuite 安装说明](https://robosuite.ai/docs/installation.html)、[robosuite 依赖定义](https://github.com/ARISE-Initiative/robosuite/blob/master/setup.py)、[MuJoCo Python 安装说明](https://mujoco.readthedocs.io/en/stable/python.html#installation)。

## 登录和使用

在本机终端登录：

```powershell
ssh catlab
```

在服务器 Bash 中激活环境并进入项目：

```bash
source /home/catlab/mujoco_robosuite/activate.sh
```

也可以手动激活：

```bash
source /home/catlab/miniconda3/etc/profile.d/conda.sh
conda activate robosuite
cd /home/catlab/mujoco_robosuite
```

Conda 激活时会自动设置 `MUJOCO_GL=egl` 和 `PYOPENGL_PLATFORM=egl`，退出环境时由 Conda 恢复变量。

## 运行验证

```bash
python smoke_test.py --gpu 2
```

无需手动激活的运行方式：

```bash
/home/catlab/mujoco_robosuite/run_smoke_test.sh --gpu 2
```

GPU 2 是搭建时选用的空闲 EGL 设备。之后可先执行 `nvidia-smi` 检查负载，再通过 `--gpu N` 选择设备。此参数选择渲染 GPU；MuJoCo 常规物理仿真运行在 CPU。不会在环境中永久固定 GPU 编号。

脚本验证内容：

- MuJoCo 球体落地物理仿真 1000 步，以及 256×256 离屏图像。
- Panda + Lift 与 Panda + Stack 各运行 50 步，检查控制器、状态观测和数值有限性。
- Panda + Lift 运行 120 步，输出 agentview 256×256、手眼相机 128×128 的 RGB 与深度观测，并录制 MP4。
- 使用固定随机种子生成小幅动作。这是安装和接口验证，不是训练好的任务策略，也不代表完成抓取任务。

验证产物保存在 `outputs/`：

| 文件 | 内容 |
| --- | --- |
| `verification.json` | 版本、GPU 渲染器、各项验证结果 |
| `mujoco.png` | MuJoCo 渲染图 |
| `agentview.png` | Panda 外部视角 |
| `robot0_eye_in_hand.png` | Panda 手眼相机 |
| `panda_lift.mp4` | 约 6 秒随机动作视频 |

当前验证已在 `DISPLAY` 未设置的会话中通过，OpenGL vendor 为 NVIDIA Corporation，renderer 为 NVIDIA GeForce RTX 4080 SUPER。`verification.log` 保存首次完整验证日志。

## 最小仿真示例

激活环境后执行 Python：

```python
import numpy as np
import robosuite as suite

env = suite.make(
    env_name="Lift",
    robots="Panda",
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=True,
    camera_names="agentview",
    camera_heights=256,
    camera_widths=256,
    render_gpu_device_id=2,
)
try:
    obs = env.reset()
    for _ in range(100):
        obs, reward, done, info = env.step(np.zeros(env.action_dim))
    print(obs["agentview_image"].shape)
finally:
    env.close()
```

## 依赖记录与重建

`requirements.txt` 固定核心兼容版本；`requirements.lock.txt` 记录基础环境的 Python 依赖；`environment.yml` 保存基础环境的 Conda 导出；`conda-history.yml` 保存 Conda 创建规格和激活变量。后续添加的本地 T 方块包参见下方说明。

在具备 NVIDIA EGL 驱动的 Linux 机器上重建独立环境：

```bash
conda create -n robosuite_rebuild python=3.11 pip -y
conda activate robosuite_rebuild
python -m pip install -r requirements.lock.txt
conda env config vars set MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
conda deactivate
conda activate robosuite_rebuild
python smoke_test.py --gpu 2
```

当前安装覆盖标准 robosuite 机器人和任务。PyPI 1.5.2 包未包含 `robosuite.examples`，且未额外安装 `robosuite_models`；启动时相关的扩展机器人、GR1 全身 IK 提示不影响这里验证的 Panda 控制器。若需要 GR1 或扩展机器人，应根据上游说明使用包含对应代码和模型的源码安装。

## T 方块资产

已添加参数化 `TBlockObject`，默认尺寸 20 × 20 × 2.5 cm、质量 0.3 kg，支持真实 T 形碰撞与自由移动。

```python
from catlab_assets import TBlockObject
t_block = TBlockObject(name="t_block")
```

运行 Panda 场景预览：`python -m catlab_assets.preview --gpu 2`。

完整参数、任务集成、接触推动验证和迁移方式见 [T 方块说明](T_BLOCK.zh-CN.md)。

## 正方体与 ACT Lift 实验

新增可参数化的 `CubeObject`，默认边长 4 cm、质量 64 g。图片中的必做任务采用 Panda 抓取并抬升该资产，对比 20 / 50 条示范训练的 ACT。

```python
from catlab_assets import CubeObject
cube = CubeObject(name="cube")
```

运行入口为 `bash /home/catlab/mujoco_robosuite/run_lift_experiment.sh`，仅使用 GPU 1。参数、模型来源和复现步骤见 [正方体与 ACT 说明](CUBE_ACT.zh-CN.md)；完成后的实测报告保存在 `experiments/lift_act_v1/REPORT.zh-CN.md`。

## ColorLift 三色任务

新增 robosuite `ColorLift` / `colorlift` 环境，随机放置红黄蓝三色方块，默认目标红色。测评要求抬升量>0.04m且稳定保持2s；记录错色抓取、非预期碰撞、保持失败、真实控制步数和动作效率。

采用HSV分割、四阶段OSC执行器，对照RGB-ACT / Mask-ACT / RGB+Mask-ACT。接口、诊断定义和超参数试验方式见 [ColorLift说明](COLORLIFT.zh-CN.md)，实测结果位于 `experiments/colorlift_v1/REPORT.zh-CN.md`。
