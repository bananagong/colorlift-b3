"""Hash completed final generation immediately, before any model work."""
from datetime import datetime, timezone
import json
from .spec import RUN, sha, write_once


def main():
    completed = list((RUN / "final_sealed").glob("*/completed.json"))
    rows = [r for path in completed for r in json.loads(path.read_text())["accepted"]]
    assert len(rows) == 768 and len({r["geometry_id"] for r in rows}) == 768
    assert all(not r["episodes"] for r in rows)
    path = RUN / "final_generation_preseal.json"
    if path.exists():
        value = json.loads(path.read_text())
        for relative, digest in value["sha256"].items(): assert sha(RUN / "final_sealed" / relative) == digest
    else:
        value = {"sealed_at_UTC": datetime.now(timezone.utc).isoformat(), "independent_geometries": 768,
                 "policy_attempts": 0, "final_protocol_sha256": sha(RUN / "final_protocol.json"),
                 "sha256": {str(p.relative_to(RUN / "final_sealed")): sha(p) for p in sorted((RUN / "final_sealed").rglob("*")) if p.is_file()}}
        write_once(path, value)
    print(json.dumps({"sealed": True, "geometries": 768, "policy_attempts": 0, "files": len(value["sha256"])}, indent=2), flush=True)


if __name__ == "__main__": main()
