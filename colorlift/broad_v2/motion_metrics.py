"""Post-process true motion efficiency without touching raw trajectories."""
from collections import Counter
import csv
import json
import numpy as np
from .spec import RUN


def main():
    results = []
    paths = list((RUN / "development").glob("*/episodes/*/*/*/metadata.json"))+list((RUN / "data_candidates").glob("*/*/*/metadata.json"))
    for path in paths:
        meta = json.loads(path.read_text())
        with (path.parent / "steps.csv").open(encoding="utf-8-sig", newline="") as stream: steps = list(csv.DictReader(stream))
        counts = Counter(); vectors = {}
        for row in steps:
            phase, tick = int(row["phase"]), int(row["step"])
            command = np.array([float(row["command_"+a]) for a in "xyz"])
            physical = np.array([float(row["eef_after_"+a])-float(row["eef_before_"+a]) for a in "xyz"])
            for kind, value in (("command", command), ("physical_motion", physical)):
                norm = np.linalg.norm(value)
                if norm < .0005: continue
                if kind in vectors:
                    previous, previous_phase, previous_tick = vectors[kind]
                    cosine = float(np.dot(previous, value)/(np.linalg.norm(previous)*norm))
                    if tick-previous_tick <= 3 and cosine < np.cos(np.radians(100)):
                        counts[kind+"_reversals_all"] += 1
                        counts[kind+"_reversals_within_phase"] += previous_phase == phase
                vectors[kind] = value, phase, tick
            counts["post_close_idle_steps"] += str(row["post_close_idle"]).lower() == "true"
            counts["no_progress_steps"] += str(row["no_progress"]).lower() == "true"
            counts["phase_"+str(phase)+"_steps"] += 1
        fixed = {key: int(counts[key]) for key in ("command_reversals_all", "command_reversals_within_phase", "physical_motion_reversals_all", "physical_motion_reversals_within_phase", "post_close_idle_steps", "no_progress_steps", "phase_0_steps", "phase_1_steps", "phase_2_steps", "phase_3_steps")}
        results.append({"directory": meta["directory"], "geometry_id": meta["geometry_id"], "actor": meta["actor"], "category": meta["layout"]["category"],
                        "target_color": meta["target_color"], "success": meta["success"], "contact_qualified": meta["eligible"], "steps": len(steps),
                        "physical_path_m_500Hz": sum(float(r["physics_path_m"]) for r in steps), **fixed})
    out = RUN / "results"; out.mkdir(exist_ok=True)
    with (out / "motion_efficiency.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0])); writer.writeheader(); writer.writerows(results)
    (out / "motion_efficiency_definition.json").write_text(json.dumps({"reversal_min_vector_m": .0005, "reversal_angle_deg": 100,
                                                                     "maximum_gap_control_steps": 3, "physical_motion": "EEF after-before at actual control boundaries",
                                                                     "idle_and_plateau": "native per-step flags; stable hold excluded from idle",
                                                                     "samples_are_not_independent_layouts": True}, indent=2))
    print("MOTION_METRICS", len(results), flush=True)


if __name__ == "__main__": main()
