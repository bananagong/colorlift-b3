"""Results/checkpoints plus all adverse-case media; full raw training data stays on catlab."""
import hashlib
import importlib.metadata
import json
import shutil
import tarfile
from .spatial_gate import ROOT, RUN


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""): value.update(block)
    return value.hexdigest()


def main():
    assert json.loads((RUN / "audit.json").read_text())["passed"]
    assert (RUN / "REPORT.zh-CN.md").exists()
    manifest = json.loads((RUN / "data/manifest.json").read_text())
    sample_directories = set()
    for level in ("easy", "medium", "hard"):
        row = next(r for r in manifest["layouts"] if r["split"] == "train" and r["level"] == level)
        sample_directories.update(row["episodes"])
    source = RUN / "source/colorlift"; source.mkdir(parents=True, exist_ok=True)
    source_hashes = {}
    for path in (ROOT / "colorlift").glob("*.py"):
        shutil.copy2(path, source / path.name); source_hashes[str(path.relative_to(ROOT))] = sha(path)
    (RUN / "source_sha256.json").write_text(json.dumps(source_hashes, indent=2))
    versions = {name: importlib.metadata.version(name) for name in ("torch", "torchvision", "mujoco", "robosuite", "numpy", "opencv-python", "imageio")}
    (RUN / "runtime_versions.json").write_text(json.dumps(versions, indent=2))
    config_dir = RUN / "configuration_snapshots"; config_dir.mkdir(exist_ok=True)
    shutil.copy2(ROOT / "experiments/colorlift_v1/config.json", config_dir / "task_config.json")
    shutil.copy2(ROOT / "experiments/colorlift_short_expert_v1/expert_config.json", config_dir / "expert_config.json")
    selected_media = set()
    for root in (RUN / "gate/episodes", RUN / "evaluations", RUN / "candidates"):
        for path in root.rglob("metadata.json"):
            row = json.loads(path.read_text())
            relative = str(path.parent.relative_to(RUN))
            adverse = not row.get("eligible", row.get("contact_qualified_collision_free_success", False))
            selected = adverse or relative in sample_directories
            if relative.startswith("gate/"):
                selected |= row["seed"] == 1000000
            elif relative.startswith("evaluations/test/"):
                levels = ("easy", "medium", "hard", "ood_position", "ood_light", "ood_camera")
                selected |= row["seed"] == 2200000+levels.index(row["layout"]["level"])*10000 and row["target_color"] == "red"
            elif relative.startswith("evaluations/counterfactual/"):
                selected |= row["seed"] == 2300000
            elif relative.startswith("evaluations/validation/"):
                selected = True
            if selected: selected_media.add(relative)
    audit = json.loads((RUN / "audit.json").read_text())
    for row in audit.get("initial_RGB_numeric_differences", []):
        selected_media.add(row["directory"])
        selected_media.add(row["reference"])
    delivery = {"full_server_directory": str(RUN), "all_metrics_metadata_step_CSV_included": True,
                "all_adverse_case_videos_included": True, "representative_success_videos_included": True,
                "all_training_policies_included": 4, "raw_expert_NPZ_examples_included": len(sample_directories),
                "complete_raw_expert_demonstrations_on_server": 1890,
                "full_positive_videos_and_training_caches_on_server": True,
                "stage_checkpoints_on_server": True, "new_independent_train_layouts": 180, "new_independent_val_layouts": 30}
    (RUN / "delivery_contents.json").write_text(json.dumps(delivery, indent=2))
    (RUN / "DELIVERY.zh-CN.txt").write_text("结果包包含全部指标、逐步CSV、元数据、4个训练权重、所有失败/非预期接触回放、代表性成功视频和27条原始示范示例。\n完整1890条原始示范、全部成功视频、阶段权重和共享缓存均保存在catlab实验目录。\n数据集使用data/manifest.json白名单，完整训练数据请在服务器上读取对应candidates目录。\n新独立布局为训练180、验证30；颜色指令、换色和630次重放不增加布局数。\n", encoding="utf-8")
    archive = ROOT / "experiments/colorlift_spatial_curriculum_v1_results.tar.gz"
    files = 0; media_files = 0
    with tarfile.open(archive, "w:gz", compresslevel=4) as tar:
        for path in sorted(RUN.rglob("*")):
            if not path.is_file(): continue
            relative = path.relative_to(RUN)
            if str(relative) in ("package.json", "package.log"): continue
            if "__pycache__" in relative.parts or relative.parts[0] == "cache": continue
            if relative.parts[0] == "models" and path.suffix == ".pt" and path.name != "policy.pt": continue
            parent = str(path.parent.relative_to(RUN))
            if relative.parts[0] == "candidates" and path.name == "trajectory.npz" and parent not in sample_directories: continue
            if path.suffix.lower() in (".mp4", ".png") and relative.parts[0] in ("gate", "evaluations", "candidates") and parent not in selected_media:
                continue
            if path.suffix.lower() in (".mp4", ".png"): media_files += 1
            tar.add(path, arcname=str(path.relative_to(ROOT)), recursive=False); files += 1
    result = {"archive": str(archive), "bytes": archive.stat().st_size, "files": files, "media_files": media_files,
              "sha256": sha(archive), **delivery}
    (RUN / "package.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__": main()
