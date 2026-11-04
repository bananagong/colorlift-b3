# 三色 ACT 空间泛化与课程顺序实验

实验目录：`experiments/colorlift_spatial_curriculum_v1/`。完整结果位于其中的 `REPORT.zh-CN.md`；所有原始数据和全部回放保存在 catlab。

首先冻结此前 ACT 和事件控制器，进行1000个独立新布局×三色指令，共3000次门槛测试。红/黄/蓝目标成功率99.8%/99.9%/99.8%，无非预期接触成功率98.6%/99.1%/99.0%，观察到的抓错颜色均0/1000。完整计数、逐色置信区间和力阈值定义在 `gate/metrics.json`、`gate/protocol.json`。

门槛通过后，在新位置真正执行专家动作。最终训练180个、验证30个独立几何布局；每个几何3种实际颜色分配×3个目标指令，产生1620/270条示范。颜色排列、颜色指令、重放和扰动次数都不增加独立布局数。每个物理槽位恰好对应过每种颜色目标。630次原动作物理重放再次通过；只有完整9回合合格的布局组能进入训练清单。

Easy为中央疏锚点±2cm扰动；Medium为簇中心±4cm平移/随机旋转及较近的物体间距；Hard为可达工作区边缘12.5–14.5cm和方块yaw变化。布局描述在 `colorlift/spatial_layouts.py`，真实颜色交换及评分身份在 `spatial_env.py`；同物理body位置和朝向保持不变。

课程与打乱对照共享完全相同的批次多重集合，每种方法种子43、44各5000次更新、batch36。课程先Easy，随后Easy/Medium，再加入Hard；打乱组只重排同批次顺序。所有方法使用相同4步预测容量、对准/接触监督前缀2、事件执行4/2/1/4和同一优化器。模型在 `models/`，推荐模型由固定验证损失选择，见 `recommended_model.json`。

最终每模型另测150个新空间布局的450次指令，以及12个反事实几何的6种颜色排列×3个指令，共216次。位置组合、光照和相机OOD不进入训练/验证。正式结果分别在 `results/comparison.csv`、`family_comparison.csv`、`paired_order_effects.csv`、`counterfactual.csv`，审计在 `audit.json`。

新模型只读RGB、本体/阶段状态和目标颜色ID。专家可读目标真值以生成标签；学生控制不读目标真值、body ID或HSV目标定位。接触来自模拟触觉；所有方法保持相同原有成功判据：抬升>4cm且连续稳定2秒。

## 运行一次新场景

```bash
source /home/catlab/mujoco_robosuite/activate.sh
CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1 python -m colorlift.spatial_demo --gpu 1 --level hard --color red --seed 2400000 --permutation 0 --tag demo
```

`--level`：easy、medium、hard、ood_position、ood_light、ood_camera。

固定seed/level并改变permutation（0–5）会交换颜色，物体位置/朝向不变；改变color会改变目标指令。更换tag保留之前的视频。`--model baseline`运行原模型，或明确指定curriculum_seed43、shuffled_seed43、curriculum_seed44、shuffled_seed44。

`data/manifest.json`为训练白名单，指向 `candidates/` 中的实际物理轨迹。完整原始RGB/动作、所有成功视频、阶段权重和缓存留在服务器；交付结果包包含4个训练权重、全部元数据/逐步CSV、所有失败及异常接触媒体和27条示范示例，详见 `delivery_contents.json`。
