"""Freeze approved sources/data/plans before any new scene or training update."""
import importlib.metadata
import os
from pathlib import Path
import shutil
from .spec import ROOT, RUN, OLD, DATA, INITIAL, SEEDS, read, sha, atomic, now, status


def main():
    if (RUN / "preparation_completed.json").exists():
        from .spec import verify
        verify(); return
    assert not (RUN / "confirm_private").exists(), "Do not refreeze after sampling starts"
    approved = read(RUN / "approved_plan.json")
    assert approved["execution_authorized"] and approved["selected_variant"] == "B"
    selected = approved["variants"][0]
    assert selected["id"] == "B" and selected["new_paired_seeds"] == list(SEEDS)
    assert selected["new_training_runs"] == 12 and selected["total_new_sealed_layout_rollouts"] == 86016
    status("freezing_sources_data_and_paired_plans", training_updates=0, new_layouts=0)
    refs = read(RUN / "reference_manifest.json")
    initial = OLD / "frozen/old_policy.pt"
    assert sha(initial) == refs["common_initial_sha256"]
    if not INITIAL.exists():
        shutil.copy2(initial, INITIAL); INITIAL.chmod(0o444)
    assert sha(INITIAL) == refs["common_initial_sha256"]
    legacy = {}; old_selection = {}
    (RUN / "frozen_references").mkdir(exist_ok=True)
    for seed in (43, 44):
        for group, original_name in (("B2", f"B2_balanced_s{seed}"), ("B3", f"B3a_l0p1_s{seed}")):
            directory = OLD / "models" / original_name
            selection = read(directory / "selection.json")
            assert sha(directory / "policy.pt") == selection["sha256"]
            target = RUN / "frozen_references" / f"{group}_s{seed}.pt"
            if not target.exists():
                shutil.copy2(directory / "policy.pt", target); target.chmod(0o444)
            assert sha(target) == selection["sha256"]
            legacy[str(target.relative_to(RUN))] = sha(target)
            old_selection[f"{group}_s{seed}"] = selection
    assert legacy["frozen_references/B3_s44.pt"] == refs["B3"]["sha256"]
    assert legacy["frozen_references/B2_s44.pt"] == refs["B2"]["sha256"]
    cache = RUN / "cache"
    if not cache.exists():
        cache.symlink_to(OLD / "cache", target_is_directory=True)
    assert cache.resolve() == (OLD / "cache").resolve()
    fingerprints = {}
    for path in sorted((OLD / "cache").glob("*.npy")):
        stat = path.stat()
        fingerprints[path.name] = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": sha(path)}
        print("FROZEN_CACHE", path.name, stat.st_size, flush=True)
    assert len(fingerprints) >= 30
    import torch
    from ..broad_act import data as original_data
    original_data.RUN = RUN
    checkpoint = torch.load(INITIAL, map_location="cpu", weights_only=False)
    dataset = original_data.Dataset("train", checkpoint["stats"])
    plans = {}
    for seed in SEEDS:
        values = dataset.plan(seed, "balanced")
        assert values.shape == (6000, 36)
        plans[str(seed)] = sha(RUN / "plans" / f"balanced_{seed}.npz")
        print("FROZEN_PAIRED_PLAN", seed, int(values.size), flush=True)
    sources = sorted(set((ROOT / "colorlift").rglob("*.py")) | set((ROOT / "lift_act").rglob("*.py")) |
                     set((ROOT / "third_party").rglob("*.py")) | set((ROOT / "catlab_assets").rglob("*.py")))
    source_sha = {str(path.relative_to(ROOT)): sha(path) for path in sources}
    for path in sources:
        target = RUN / "source_snapshot" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, target)
    manifests = [OLD / "data_manifest.json", OLD / "protocol.json", DATA / "final_protocol.json",
                 DATA / "training_manifest.json", DATA / "supervision_manifest.json",
                 RUN / "approved_plan.json", RUN / "approved_checklist.zh-CN.md", RUN / "reference_manifest.json"]
    versions = {}
    for package in ("torch", "numpy", "scipy", "mujoco", "robosuite", "Pillow", "imageio"):
        versions[package] = importlib.metadata.version(package)
    protocol = {"version": 1, "frozen_at_UTC": now(), "client_approval_date": "2026-10-31",
        "execution_authorized": True, "selected_variant": "B", "human_approval": approved["human_approval"],
        "new_paired_seeds": list(SEEDS), "legacy_supplement_seeds": [43,44],
        "initial_sha256": sha(INITIAL), "source_sha256": source_sha,
        "data_manifest_sha256": {str(path.relative_to(ROOT)): sha(path) for path in manifests},
        "cache_fingerprints": fingerprints, "sampling_plan_sha256": plans,
        "legacy_model_sha256": legacy, "legacy_development_selections": old_selection,
        "physical_GPUs": [1,2], "training": approved["training"], "layout_protocol": approved["proposed_new_layouts"],
        "budget": {"new_training_runs":12,"updates_per_run":6000,"development_episodes":6912,
                   "primary_new_seed_episodes":64512,"legacy_supplement_episodes":21504,"total_confirm_episodes":86016},
        "primary_analysis": approved["primary_analysis"], "safety_gate_per_new_seed_per_suite": {"success":.98,"contact_qualified_success":.95,"observed_wrong_colors":0},
        "secondary_test_family": "3 suites x task/contact exact seed-label p-values, Holm adjusted; not primary",
        "analysis_rng_seed": 20261031, "no_optional_stopping": True, "stage2_requires_separate_protocol": True,
        "software_versions": versions,
        "OOD_interpretation": "fresh current-protocol holdout; inherited task checkpoint may have prior spatial/relation exposure; mixed/Sobol cohort is not unconstrained IID"}
    atomic(RUN / "protocol.json", protocol)
    atomic(RUN / "preparation_completed.json", {"completed_at_UTC":now(),"protocol_sha256":sha(RUN / "protocol.json"),
        "sources":len(sources),"cache_arrays":len(fingerprints),"training_updates":0,"new_layouts":0})
    print("PREPARATION_COMPLETE", len(sources), len(fingerprints), flush=True)


if __name__ == "__main__":
    main()
