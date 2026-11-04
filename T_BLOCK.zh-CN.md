# T 方块资产

已在 catlab 的 `robosuite` 环境安装 `catlab-robosuite-assets`（当前 0.2.0，包含 T 方块与正方体）。可从任意工作目录导入：

```python
from catlab_assets import TBlockObject

t_block = TBlockObject(name="t_block")
```

## 默认规格

所有长度使用米，质量使用千克。默认蓝色，顶视图为 T 形。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `width` | 0.20 | 横梁总宽，沿 X |
| `length` | 0.20 | T 形总长，沿 Y |
| `height` | 0.025 | 方块厚度，沿 Z |
| `bar_depth` | 0.05 | 横梁沿 Y 的深度 |
| `stem_width` | 0.05 | 竖杆沿 X 的宽度 |
| `mass` | 0.30 | 整个刚体质量 |
| `friction` | (0.5, 0.005, 0.0001) | MuJoCo 滑动、扭转、滚动摩擦参数 |
| `rgba` | (0.08, 0.48, 0.85, 1) | 外观颜色 |
| `joints` | `"default"` | 一个自由关节；`None` 则固定物体 |

本地坐标系中，横梁朝 +Y，竖杆末端朝 -Y，+Z 朝上。物体原点是包围盒中心；默认均匀密度下，质心为 `(0, 0.032142857, 0)`，由 MuJoCo 根据几何体质量和位置计算。

横梁和竖杆分别是一个 box，连接边界恰好相接，全部属于**同一个 body**。两块几何体按体积分配质量，没有重叠部分的重复质量。两侧凹口保留真实碰撞边界。

碰撞体位于 group 0，视觉体位于 group 1；视觉体质量为零且不参与碰撞。接触使用 `condim=4`。桌面的摩擦参数也会影响推行动力学，需要在具体任务中一起设定。

## 调整尺寸、质量和外观

```python
from catlab_assets import TBlockObject

t_block = TBlockObject(
    name="small_t",
    width=0.12,
    length=0.16,
    height=0.02,
    bar_depth=0.04,
    stem_width=0.035,
    mass=0.16,
    friction=(0.4, 0.003, 0.0001),
    rgba=(0.9, 0.2, 0.1, 1.0),
)
```

物体类会同步修改碰撞、视觉、放置偏移、半径和标记点。质量参数独立于尺寸，改变尺寸后仍保持传入的总质量。

提供标准 robosuite 属性：`root_body`、`joints`、`contact_geoms`、`visual_geoms`、`bottom_offset`、`top_offset`、`horizontal_radius`、`get_bounding_box_size()`。可直接加入 `ManipulationTask` 和 `UniformRandomSampler`。

`important_sites` 包含 `cap_center`、`stem_center`、`stem_tip`；`footprint_vertices` 返回局部 XY 平面中逆时针排列的 8 个轮廓顶点，可用于后续推 T 任务的目标覆盖率计算。

## 加入 MuJoCo / robosuite 场景

以下示例生成带自由 T 方块的桌面物理模型：

```python
import mujoco
from catlab_assets import TBlockObject
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask

table_height = 0.8
arena = TableArena(table_offset=(0, 0, table_height))
obj = TBlockObject("push_t")
obj.set_pos([0, 0, table_height - obj.bottom_offset[2] + 0.002])
task = ManipulationTask(
    mujoco_arena=arena,
    mujoco_robots=[],
    mujoco_objects=[obj],
)
model = mujoco.MjModel.from_xml_string(task.get_xml())
data = mujoco.MjData(model)
for _ in range(1000):
    mujoco.mj_step(model, data)
```

在机器人任务中，将 `mujoco_robots` 设为该任务已有的机器人模型列表即可。资产 XML 使用 robosuite 的物体模板结构，关节由 `MujocoXMLObject` 添加；独立 MuJoCo 场景示例为 `outputs/t_block/t_block_scene.xml`。

## 预览与验证

```bash
source /home/catlab/mujoco_robosuite/activate.sh

# 生成 Panda 桌面场景的相机预览
python -m catlab_assets.preview --gpu 2

# 检查物理、形状、参数、多实例和机器人集成，并录制接触推动视频
python -m catlab_assets.verify --gpu 2
```

`TBlockPreview` 是资产预览场景：使用 Panda 和桌面，奖励恒为 0，沿用 Lift 的 `cube_*` 观测键指代 T 方块。后续的推 T 训练任务可以复用这个资产，单独定义动作、目标姿态、奖励和成功判据。

2026-10-28 在 GPU 2、EGL 无窗口模式下验证通过：

- 默认与自定义尺寸的实例可同时编译，名称互不冲突，质量和惯量正确。
- 放置采样器正确处理底面偏移；物体稳定落在桌面。
- 小球探针在左右凹口无接触，在竖杆和横梁内部检测到接触。
- 1500 步实际接触推动产生约 0.0694 m 平移、0.6726 rad 转动，保持平放。
- Panda 场景连续两次重置并各运行 20 步，状态数值有限，双相机图像有效。

产物位于 `/home/catlab/mujoco_robosuite/outputs/t_block/`：

| 文件 | 内容 |
| --- | --- |
| `t_block_topview.png` | 纯资产顶视图 |
| `t_block_after_push.png` | 接触推动后的姿态 |
| `t_block_push.mp4` | 橙色推杆推动 T 方块的视频 |
| `panda_agentview.png` | Panda 场景外部相机 |
| `panda_tblock_topview.png` | Panda 场景顶视相机 |
| `t_block_scene.xml` | 可由 MuJoCo 直接加载的完整场景 |
| `verification.json` / `verification.log` | 验证结果和日志 |

## 源码与迁移

- `catlab_assets/assets/objects/t_block.xml`：默认 MJCF 资产。
- `catlab_assets/t_block.py`：参数化 `TBlockObject`。
- `catlab_assets/preview.py`：Panda 场景预览。
- `catlab_assets/verify.py`：物理和集成验证。
- `pyproject.toml`：安装元数据，包含 XML 包数据。

服务器采用 editable 安装，修改上述源文件后，新的 Python 进程会使用更新后的资产。基础环境依赖仍记录在原有 `requirements.lock.txt`；资产包的当前安装记录另存于 `requirements-assets.freeze.txt`。

在已经具备兼容 MuJoCo / robosuite 环境的机器上安装源码：

```bash
python -m pip install --no-deps --no-build-isolation -e /path/to/mujoco_robosuite
```

独立 wheel 位于 `dist/catlab_robosuite_assets-0.2.0-py3-none-any.whl`。安装 wheel 后，XML 仍随包可用。

接口依据：[robosuite 自定义物体说明](https://robosuite.ai/docs/modules/objects.html)。
