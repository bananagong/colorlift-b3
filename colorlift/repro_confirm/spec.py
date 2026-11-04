from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "experiments/colorlift_repro_confirm_v1"
OLD = ROOT / "experiments/colorlift_broad_act_round1"
DATA = ROOT / "experiments/colorlift_broad_protocol_v2"
PRIVATE = RUN / "confirm_private"
SEEDS = (45, 46, 47, 48, 49, 50)
GPUS = (1, 2)
COLORS = ("red", "yellow", "blue")
SUITES = ("broad_iid", "spatial_ood", "relation_ood")
STEPS = 6000
CHECKPOINTS = (1500, 3000, 4500, 6000)
ATTEMPTS = 5376
INITIAL = RUN / "frozen_initial.pt"


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def once(path, value):
    if Path(path).exists():
        assert read(path) == value, path
    else:
        atomic(path, value)


def status(stage, **extra):
    atomic(RUN / "status.json", {"stage": stage, "updated_at_UTC": now(), **extra})


def verify(require_seal=False):
    protocol = read(RUN / "protocol.json")
    assert protocol["selected_variant"] == "B" and protocol["execution_authorized"]
    assert protocol["new_paired_seeds"] == list(SEEDS)
    for relative, digest in protocol["source_sha256"].items():
        assert sha(ROOT / relative) == digest, relative
    for relative, meta in protocol["cache_fingerprints"].items():
        path = OLD / "cache" / relative; stat = path.stat()
        assert stat.st_size == meta["bytes"] and stat.st_mtime_ns == meta["mtime_ns"], path
    for relative, digest in protocol["data_manifest_sha256"].items():
        assert sha(ROOT / relative) == digest, relative
    assert sha(INITIAL) == protocol["initial_sha256"]
    for relative, digest in protocol["legacy_model_sha256"].items():
        assert sha(RUN / relative) == digest, relative
    for seed, digest in protocol["sampling_plan_sha256"].items():
        assert sha(RUN / "plans" / f"balanced_{seed}.npz") == digest
    if require_seal:
        completed = read(RUN / "sealing_completed.json")
        assert completed["independent_new_geometries"] == 768
        assert sha(RUN / "seal.json") == completed["seal_sha256"]
    return protocol


def training_context():
    protocol = verify(require_seal=True)
    assert not (RUN / "model_lock.json").exists(), "Training is forbidden after model lock"
    return protocol


def all_model_definitions():
    result = {}
    for seed in SEEDS:
        for group in ("B2", "B3"):
            name = f"{group}_s{seed}"
            result[name] = {"group": group, "training_seed": seed, "cohort": "new",
                            "path": RUN / "models" / name / "policy.pt", "controller": "original"}
    for seed in (43, 44):
        for group in ("B2", "B3"):
            name = f"legacy_{group}_s{seed}"
            result[name] = {"group": group, "training_seed": seed, "cohort": "legacy_supplement",
                            "path": RUN / "frozen_references" / f"{group}_s{seed}.pt", "controller": "original"}
    return result
