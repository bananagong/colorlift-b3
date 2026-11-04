"""Create a stable launch snapshot without reading sealed final layouts."""
import contextlib
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import tarfile
from .spec import ROOT, RUN, sha
from .progress import main as progress
from .results import report


def main():
    with contextlib.redirect_stdout(io.StringIO()):
        progress()
    report()
    name = "launch_"+datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    target = RUN / "exports" / name
    target.mkdir(parents=True, exist_ok=False)
    paths = set()
    for filename in ("protocol.json", "status.json", "progress.json", "PROGRESS.zh-CN.md", "data_manifest.json",
                     "stage1_source_freeze.json", "execution_source_freeze.json", "curriculum_exposure_audit.json",
                     "smoke.json", "B1_completed.json", "B2_completed.json", "gradient_calibration.json",
                     "coordinator_revision.json", "B3_completed.json", "B4_completed.json"):
        if (RUN / filename).exists():
            paths.add(RUN / filename)
    paths.update((RUN / "source_snapshot").glob("*.py"))
    for directory in (RUN / "models").glob("*"):
        for filename in ("config.json", "training_completed.json", "selection.json", "selection_safety.json", "validation.jsonl", "metrics.jsonl"):
            if (directory / filename).exists():
                paths.add(directory / filename)
        marker = directory / "selection.json"
        if marker.exists() and directory.name.startswith("B1"):
            selection = json.loads(marker.read_text())
            selected = RUN / "evaluations" / ("select_"+directory.name) / selection["chosen_checkpoint"]
            paths.update(p for p in selected.rglob("*") if p.is_file())
    paths.update(p for p in (RUN / "evaluations/B0_initial").rglob("*") if p.is_file())
    for source in sorted(paths):
        destination = target / source.relative_to(RUN)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    files = {str(p.relative_to(target)): {"bytes": p.stat().st_size, "sha256": sha(p)} for p in target.rglob("*") if p.is_file()}
    (target / "snapshot_manifest.json").write_text(json.dumps({"created_at_UTC": datetime.now(timezone.utc).isoformat(),
                                                              "final_coordinates_read": False, "files": files}, indent=2))
    archive = target.with_suffix(".tar.gz")
    with tarfile.open(archive, "w:gz", compresslevel=3) as tar:
        tar.add(target, arcname=name)
    result = {"archive": str(archive), "bytes": archive.stat().st_size, "sha256": sha(archive), "files": len(files)+1}
    (RUN / "progress_export.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
