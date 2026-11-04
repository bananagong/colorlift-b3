"""Audit actual layout positions and export clearly labelled, single-camera media.

This module only reads completed experiments. It never trains or runs a policy.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone, timedelta
import hashlib
import itertools
import json
from pathlib import Path
import tarfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "experiments/colorlift_spatial_curriculum_v1"
OUT = RUN / "clarification_20261030"
LEVELS = ("easy", "medium", "hard")


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def position_key(points, digits=6):
    # Ignore colors, body identities, and yaw; a yaw change is not a new XY layout.
    return tuple(sorted(tuple(round(float(v), digits) for v in p[:2]) for p in points))


def physical_row(meta, split):
    points = meta.get("physical_slot_positions", list(meta["initial_positions"].values()))
    return {"split": split, "level": meta.get("layout", {}).get("level", "original"),
            "seed": meta["seed"], "xy": np.asarray(points)[:, :2].tolist(),
            "key": position_key(points), "key_1mm": position_key(points, 3),
            "source": meta["directory"]}


def nearest_errors(queries, references):
    permutations = np.array(list(itertools.permutations(range(3))))
    reference = np.array([r["xy"] for r in references])[:, permutations]
    result = []
    for row in queries:
        delta = reference-np.array(row["xy"])[None, None]
        result.append(float(np.sqrt(np.mean(np.sum(delta*delta, axis=-1), axis=-1)).min()))
    return result


def audit():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = read(RUN / "data/manifest.json")
    groups = {name: [] for name in ("train", "val", "test", "counterfactual", "gate")}
    color_counts = {}
    for layout in manifest["layouts"]:
        episodes = [read(RUN / rel / "metadata.json") for rel in layout["episodes"]]
        assert len(episodes) == 9
        assert len({position_key(r["physical_slot_positions"]) for r in episodes}) == 1
        balance = Counter((r["target_physical_slot"], r["target_color"]) for r in episodes)
        assert len(balance) == 9 and set(balance.values()) == {1}
        groups[layout["split"]].append(physical_row(episodes[0], layout["split"]))
        color_counts.setdefault(layout["split"], Counter()).update(balance)
    scenarios = read(RUN / "test_scenarios.json")
    for layout in scenarios["layouts"]:
        relative = f'evaluations/test/baseline/{layout["level"]}/layout_{layout["seed"]}/red/metadata.json'
        groups["test"].append(physical_row(read(RUN / relative), "test"))
    for layout in scenarios["counterfactual_base_layouts"]:
        relative = f'evaluations/counterfactual/baseline/{layout["level"]}/layout_{layout["seed"]}/p0/red/metadata.json'
        groups["counterfactual"].append(physical_row(read(RUN / relative), "counterfactual"))
    for seed in range(1000000, 1001000):
        groups["gate"].append(physical_row(read(RUN / f"gate/episodes/scene_{seed}/red/metadata.json"), "gate"))
    counts = {name: {"layouts": len(rows), "unique_unordered_XY_1um": len({r["key"] for r in rows}),
                     "unique_unordered_XY_rounded_1mm": len({r["key_1mm"] for r in rows})} for name, rows in groups.items()}
    intersections = {}
    for a, b in itertools.combinations(groups, 2):
        exact = len({r["key"] for r in groups[a]} & {r["key"] for r in groups[b]})
        rounded = len({r["key_1mm"] for r in groups[a]} & {r["key_1mm"] for r in groups[b]})
        intersections[a+"__"+b] = {"same_XY_1um": exact, "same_XY_rounded_1mm": rounded}
        assert exact == 0
    proximity = {}
    for split in ("val", "test"):
        for level in LEVELS:
            errors = nearest_errors([r for r in groups[split] if r["level"] == level],
                                    [r for r in groups["train"] if r["level"] == level])
            proximity[split+"_"+level] = dict(zip(("min_m", "median_m", "p95_m"), np.quantile(errors, [0, .5, .95]).tolist()))
    failures = read(RUN / "results/failure_cases.json")
    failure_counts = {}
    for suite in ("test", "counterfactual"):
        rows = [r for r in failures if r["directory"].startswith("evaluations/"+suite+"/")]
        failure_counts[suite] = {"videos": len(rows), "independent_geometries": len({position_key(r["physical_slot_positions"]) for r in rows})}
    clock = timezone(timedelta(hours=8))
    timeline_paths = {
        "old_distribution_gate_finished": "gate/completed.json",
        "expert_collection_finished": "collection_completed.json",
        "first_training_config_written": "models/curriculum_seed43/config.json",
        "last_training_finished": "models/curriculum_seed44/completed.json",
        "new_difficulty_evaluation_frozen": "evaluation_freeze.json",
        "first_new_difficulty_baseline_result": "evaluations/test/baseline/easy/layout_2200000/red/metadata.json",
    }
    timeline = {name: datetime.fromtimestamp((RUN / path).stat().st_mtime, clock).isoformat() for name, path in timeline_paths.items()}
    result = {"position_only_audit": counts, "cross_split_position_intersections": intersections,
              "nearest_train_layout_XY_RMS_ignoring_slot_order": proximity,
              "target_slot_color_counts": {split: {f"slot{s}_{c}": n for (s, c), n in counts.items()} for split, counts in color_counts.items()},
              "failure_video_repetition": failure_counts, "server_file_mtimes_Asia_Shanghai": timeline,
              "baseline_old_distribution_test_before_training": True,
              "baseline_new_difficulty_weights_unmodified": True,
              "baseline_new_difficulty_test_executed_before_training": False,
              "single_simulation_robot_count": 1, "single_simulation_cube_count": 3,
              "raw_video_views_left_right": ["color_front", "color_side"],
              "limitations": ["Easy uses three anchor neighborhoods", "Medium uses compact near-equilateral triangles",
                              "Hard uses reachable workspace edge bands", "Distinct layouts do not imply broad or OOD spatial coverage"]}
    (OUT / "position_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (OUT / "actual_layout_positions.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        fields = ["split", "level", "seed"]+[f"slot{i}_{axis}_m" for i in range(3) for axis in "xy"]+["source"]
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for rows in groups.values():
            for row in rows:
                writer.writerow({**{k: row[k] for k in ("split", "level", "seed", "source")},
                                 **{f"slot{i}_{axis}_m": row["xy"][i][j] for i in range(3) for j, axis in enumerate("xy")}})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 3, figsize=(11.5, 10.5), sharex=True, sharey=True, constrained_layout=True)
    for i, level in enumerate(LEVELS):
        for j, split in enumerate(("train", "val", "test")):
            rows = [r for r in groups[split] if r["level"] == level]
            xy = np.array([r["xy"] for r in rows]).reshape(-1, 2)
            axes[i, j].scatter(xy[:, 0], xy[:, 1], s=12, color="#377ca7", alpha=.45)
            axes[i, j].set(title=f"{level} / {split}: {len(rows)} XY layouts", xlim=(-.16, .16), ylim=(-.16, .16), aspect="equal")
            axes[i, j].grid(alpha=.2)
            if i == 2: axes[i, j].set_xlabel("World X (m)")
            if j == 0: axes[i, j].set_ylabel("World Y (m)")
    fig.suptitle("Actual MuJoCo reset positions; colors and yaw ignored\nSplit isolation does not remove the shared, constrained layout families")
    fig.savefig(OUT / "actual_layout_splits.png", dpi=150); plt.close(fig)
    explanation = f'''# 对布局划分、零样本评测与双相机视频的核查

已重新读取真实回合的初始body位置，忽略颜色、body顺序和yaw，另做纯XY去重。训练180、验证30、正式测试150、反事实12个布局均没有重复；这些集合之间没有发现重复的XY布局。精确坐标及按1mm取整的结果见position_audit.json和actual_layout_positions.csv。

每个训练/验证几何的9条颜色/指令轨迹作为整体划分，不会把同一布局的不同颜色版本拆到训练和验证。1890是实际示范条数，独立训练/验证几何是180/30。

零样本评测需要区分时间与权重：训练前完成了旧环境采样器的3000次新随机布局测试，范围每轴±11cm、物块中心距≥9.5cm。新增Easy/Medium/Hard/OOD的旧ACT对照保留了完全相同的旧checkpoint，但实际执行时间在4个新模型训练结束之后。它是未适配权重上的零样本基线，不是训练前先执行的新难度基线。之前的汇报没有把这点讲清楚。

新专家确实在每个新位置通过接近、对准、夹紧、抬升/保持四阶段OSC真实执行，使用位姿误差和触觉事件切换；专家读取目标真值位置，学生不读取。每个入选几何完整9次运行必须全部成功且接触合格，另有630次原动作物理重放。新的物理示范并非只给旧图像换色。

你对位置分布的担心成立：Easy只在三个锚点附近±2cm；Medium主要为紧凑三角形；Hard主要为工作区边缘带。新seed和不重复的精确坐标，并不等于三个物体已在整个可达区域任意独立均匀随机。训练/测试还共享这些布局族；当前结果不能代表任意随机空间布局的泛化。

导出视频每帧左右拼接了color_front与color_side。这是同一条Panda、同三个方块、同一个时刻的两个视角。原始视频缺少明确的视角分隔标签，容易被理解为两条机械臂和六个方块。single_camera/中只保留左侧color_front；训练仍使用原来的双相机输入。

正式空间失败视频{failure_counts['test']['videos']}条，实际来自{failure_counts['test']['independent_geometries']}个几何；反事实失败视频{failure_counts['counterfactual']['videos']}条，实际来自{failure_counts['counterfactual']['independent_geometries']}个几何。同一几何会因模型、目标颜色或换色排列而重复；反事实组则按设计固定位置，只换颜色/指令。因此连续观看失败目录并不等于观看互不相同的随机布局。

本补充只审计已完成实验、转换展示，不新增训练或评测次数。18_distinct_expert_layouts_front.mp4使用Easy/Medium/Hard各前6个入选训练布局，每个布局仅播放p0_red的一次原始真实专家运行，全程20Hz。不同片段是不同回合，片段内只有一个相机视角。

后续若扩展空间泛化实验，应先冻结更宽的非模板随机布局测试集，在微调前运行旧ACT；把布局类别、近邻几何和颜色变体整体隔离，训练时再生成覆盖这些新位置的实际专家轨迹。下一轮应同时报告布局覆盖及和训练布局的距离，而非只报告seed或哈希不同。
'''
    (OUT / "核查说明.md").write_text(explanation, encoding="utf-8")
    print(json.dumps({"counts": counts, "intersections": intersections, "failure_repetition": failure_counts,
                      "nearest_train_layout_RMS_m": proximity}, indent=2), flush=True)


def media():
    import cv2
    import imageio.v2 as imageio
    OUT.mkdir(exist_ok=True)
    def header(frame, meta, index, suffix=""):
        h, w = frame.shape[:2]
        assert w == 2*h, (meta["directory"], frame.shape)
        canvas = np.full((640, 512, 3), 250, np.uint8)
        canvas[96:608] = cv2.resize(frame[:, :h], (512, 512), interpolation=cv2.INTER_NEAREST)
        level = meta.get("layout", {}).get("level", "original")
        captions = ["ONE camera: color_front | same recorded episode", f'{meta["actor"]} | {level} | target {meta["target_color"]}',
                    f'seed {meta["seed"]} | frame {index}/{meta["control_steps"]} {suffix}']
        for y, text in zip((23, 50, 77), captions):
            cv2.putText(canvas, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, .45, (20, 20, 20), 1, cv2.LINE_AA)
        cv2.putText(canvas, "1 Panda + 3 cubes | original 20Hz timing", (10, 631), cv2.FONT_HERSHEY_SIMPLEX, .46, (20, 20, 20), 1, cv2.LINE_AA)
        return canvas
    cases = []
    for path in sorted((RUN / "gate/episodes").glob("*/*/metadata.json")):
        row = read(path)
        if not row["success"]: cases.append(row)
    cases.extend(read(RUN / "results/failure_cases.json"))
    for path in sorted((RUN / "evaluations/validation").rglob("metadata.json")):
        row = read(path)
        if not row["success"]: cases.append(row)
    recommended = read(RUN / "recommended_model.json")["model"]
    for suite in ("test", "counterfactual"):
        for path in sorted((RUN / "evaluations" / suite / recommended).rglob("metadata.json")):
            row = read(path)
            if row["success"] and not row["eligible"]: cases.append(row)
    assert len({r["directory"] for r in cases}) == len(cases) == 125
    index_rows = []
    for number, meta in enumerate(cases, 1):
        source = RUN / meta["directory"] / "rollout.mp4"
        destination = OUT / "single_camera" / meta["directory"] / "front.mp4"
        destination.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with imageio.get_reader(str(source)) as reader, imageio.get_writer(str(destination), fps=20, codec="libx264", quality=8) as writer:
            assert abs(reader.get_meta_data()["fps"]-20) < 1e-6
            for frame_index, frame in enumerate(reader):
                writer.append_data(header(frame, meta, frame_index)); count += 1
        assert count == meta["control_steps"]+1
        index_rows.append({"file": str(destination.relative_to(OUT)), "source": meta["directory"], "seed": meta["seed"],
                           "model": meta["actor"], "level": meta.get("layout", {}).get("level", "original"),
                           "target_color": meta["target_color"], "success": meta["success"],
                           "contact_qualified": meta.get("eligible", meta.get("contact_qualified_collision_free_success")),
                           "frames": count, "fps": 20, "source_sha256": digest(source), "sha256": digest(destination)})
        if number%10 == 0: print("SINGLE_CAMERA", number, len(cases), flush=True)
    with (OUT / "single_camera_index.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(index_rows[0])); writer.writeheader(); writer.writerows(index_rows)
    manifest = read(RUN / "data/manifest.json")
    examples = []
    for level in LEVELS:
        for layout in [r for r in manifest["layouts"] if r["split"] == "train" and r["level"] == level][:6]:
            examples.append(read(RUN / next(p for p in layout["episodes"] if p.endswith("p0_red")) / "metadata.json"))
    assert len({position_key(m["physical_slot_positions"]) for m in examples}) == 18
    destination = OUT / "18_distinct_expert_layouts_front.mp4"
    count = 0
    with imageio.get_writer(str(destination), fps=20, codec="libx264", quality=8) as writer:
        for number, meta in enumerate(examples, 1):
            with imageio.get_reader(str(RUN / meta["directory"] / "rollout.mp4")) as reader:
                for frame_index, frame in enumerate(reader):
                    writer.append_data(header(frame, meta, frame_index, f'| layout {number}/18')); count += 1
    assert count == sum(m["control_steps"]+1 for m in examples)
    (OUT / "18_expert_layout_index.json").write_text(json.dumps([{k: row[k] for k in ("directory", "seed", "layout", "physical_slot_positions", "control_steps", "success", "eligible")} for row in examples], indent=2), encoding="utf-8")
    sample = cases[0]
    original = Image.open(RUN / sample["directory"] / "initial.png").convert("RGB")
    w, h = original.size
    canvas = Image.new("RGB", (1024, 608), "white")
    canvas.paste(original.resize((1024, 512), Image.Resampling.NEAREST), (0, 64))
    draw = ImageDraw.Draw(canvas)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 22)
    draw.text((18, 20), "LEFT: color_front", fill="black", font=font)
    draw.text((530, 20), "RIGHT: color_side", fill="black", font=font)
    draw.line((512, 0, 512, 576), fill="#d9534f", width=4)
    draw.text((20, 579), "Two cameras observing the SAME 1 Panda and SAME 3 cubes.", fill="black", font=font)
    canvas.save(OUT / "two_camera_explanation.png")
    completion = {"single_camera_case_videos": len(index_rows), "distinct_expert_layouts": len(examples), "expert_video_frames": count,
                  "original_frame_counts_preserved": True, "fps": 20, "original_results_unchanged": True}
    (OUT / "media_completed.json").write_text(json.dumps(completion, indent=2))
    archive = RUN / "clarification_20261030.tar.gz"
    with tarfile.open(archive, "w:gz", compresslevel=3) as tar:
        tar.add(OUT, arcname="clarification_20261030")
    print(json.dumps({**completion, "archive": str(archive), "bytes": archive.stat().st_size, "sha256": digest(archive)}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--media", action="store_true")
    args = parser.parse_args()
    audit()
    if args.media: media()
