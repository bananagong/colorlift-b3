from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "experiments/colorlift_broad_protocol_v2"
RUN = ROOT / "experiments/colorlift_broad_act_round1"
INITIAL = ROOT / "experiments/colorlift_recovery_v1/model/policy.pt"
SEEDS = (43, 44)
STEPS = 6000
BATCH = 36
CHECKPOINTS = (1500, 3000, 4500, 6000)
COLORS = ("red", "yellow", "blue")
SOURCE_WEIGHTS = (.8, .1, .1)
PHASE_WEIGHTS = (.10, .35, .35, .20)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""): h.update(block)
    return h.hexdigest()


def write_once(path, value):
    value = json.loads(json.dumps(value))
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists(): assert json.loads(path.read_text()) == value, path
    else: path.write_text(json.dumps(value, indent=2))


def freeze():
    path = RUN / "protocol.json"
    if path.exists():
        protocol = json.loads(path.read_text())
        assert sha(INITIAL) == protocol["initial_checkpoint_sha256"]
        return protocol
    value = {"frozen_at_UTC": datetime.now(timezone.utc).isoformat(), "initial_checkpoint": str(INITIAL),
             "initial_checkpoint_sha256": sha(INITIAL), "user_authorized_warm_start": True,
             "supersedes_previous_fresh_initialization_for_this_round": True,
             "OOD_interpretation": "current-round initial-layout holdout; inherited task weights may have historical region/relation exposure; not a clean all-history OOD claim",
             "groups": {"B0": "unchanged old ACT", "B1": "new broad demonstrations plus filtered old clean/recovery, random sampling",
                        "B2": "same data, spatial/difficulty balanced sampling; matched-exposure curriculum comparison",
                        "B3": "B3a auxiliary relative-position loss only, ACT output interface unchanged",
                        "B4": "B3 plus safe-pregrasp auxiliary supervision and real safe/contact-recovery demonstrations; explicit recovery-runtime controls"},
             "initialization": "all trained candidates share the same old checkpoint within paired seeds; optimizer starts fresh",
             "seeds": SEEDS, "updates_per_run": STEPS, "batch_size": BATCH, "checkpoints": CHECKPOINTS,
             "B1_B2_B3_source_fractions": SOURCE_WEIGHTS, "source_names": ["new_broad", "old_clean", "old_recovery"],
             "phase_sampling": PHASE_WEIGHTS, "forecast_capacity": 4, "contact_supervision_prefix": 2,
             "deployment_execution_lengths": [4, 2, 1, 4], "controller_B0_B1_B2_B3": "unchanged EventExecutor efficiency",
             "B2_variants": ["balanced", "curriculum"], "curriculum": "sort frame exposures by static geometry difficulty within source/color/phase strata; exact multiset and per-batch stratum counts match balanced",
             "B3_initial_lambda_candidates": [.1, .3, 1.],
             "B3_gradient_calibration": "raw L1 in meters; measure shared input-projection gradient norms on8 development batches, adjust grid only if all candidate ratios outside0.01..1; freeze before B3 training",
             "B3a_features": "shared visual input-projection maps before action-conditioned CVAE latent; robot state and32D learned color embedding; no ground-truth target input",
             "B3b": "deferred; first isolate auxiliary loss without feature reinjection",
             "optimizer": {"ACT_lr": 3e-5, "recognizer_embedding_lr": 1e-5, "backbone_lr": 3e-6, "new_head_lr": 1e-4, "weight_decay": 1e-4, "grad_clip": 1., "cosine_floor": 1e-6},
             "normalization": "preserve old checkpoint state/XYZ statistics for compatibility, identical across candidates",
             "ACT_action_labels": "same canonical macro target; map phase0 Z to original executor0.865m, keep safety pregrasp labels separate; injected controls never supervised",
             "checkpoint_selection": ["prefer zero observed wrong colors", "maximum category-balanced physical closed-loop success", "maximum category-balanced contact-qualified success", "minimum mean steps across all attempts", "validation loss only final tie-break"],
             "selection_layouts": "all48 new validation geometries x3 target colors, same scenes for every checkpoint; no final geometry used",
             "final_unseal": "after all B0-B4 choices, lambdas and checkpoint hashes are locked, run the presealed final suite once per group; choose seed only on development/validation",
             "B4_data": "32 existing training geometries and12 validation geometries, each3 cyclic palettes x3 colors, safe clean and physically induced contact slip; no new layout count for perturbations",
             "B4_input": "same RGB/proprio/color; safe pregrasp and delta are predicted, simulator labels not supplied",
             "B4_runtime_causality_controls": "evaluate B3 with same safe/recovery runtime and B4 with original runtime on development, alongside bundled B4",
             "B4_source_fractions": [.6, .1, .1, .1, .1],
             "success": "original correct target lift>4cm stable2s; all failure/force/contact/steps metrics retained",
             "statistical_unit": "initial geometry_id; color/permutation/recovery episodes remain grouped; report layout-cluster intervals",
             "physical_GPUs": [1, 2], "data_v2_protocol_sha256": sha(DATA / "final_protocol.json"),
             "data_manifest_sha256": sha(DATA / "training_manifest.json"), "data_supervision_sha256": sha(DATA / "supervision_manifest.json"),
             "final_preseal_sha256": sha(DATA / "final_generation_preseal.json")}
    write_once(path, value)
    return value
