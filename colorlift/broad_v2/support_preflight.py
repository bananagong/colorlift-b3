"""Check OOD support is nonempty and reject physical negative controls before freeze."""
import json
import numpy as np
from .spec import RUN
from .layouts import Proposals, footprint_gap, geometry_id
from .validity import make_env, Validator, visibility


def main():
    env = make_env(2); validator = Validator(env); report = {}
    try:
        for index, (suite, subtype) in enumerate((("spatial_ood", None), ("relation_ood", "near_gap"), ("relation_ood", "aligned"))):
            stream = Proposals(3200+index, "uniform", suite, subtype)
            found = []
            for i in range(2000):
                layout, reason = stream.next()
                if reason: continue
                result = validator.check(layout)
                if result["valid"]: found.append({"layout": layout, "validation": result})
                if len(found) == 2: break
            assert len(found) == 2, (suite, subtype)
            report[suite+"_"+str(subtype)] = {"proposals": stream.index, "examples": found}
            print("SUPPORT", suite, subtype, stream.index, flush=True)
        assert footprint_gap([[0, 0], [.01, 0], [.1, .1]], [0, 0, 0]) < 0
        _, unreachable = validator.ik.solve_pose([.8, .8, .82])
        assert unreachable["position_error_m"] > .001
        layout = report["spatial_ood_None"]["examples"][0]["layout"]
        env.reset_layout(layout, "red")
        model = env.sim.model._model
        original = model.cam_pos.copy()
        model.cam_pos[:] += np.array([2., 2., 2.]); env.sim.forward()
        invisible = visibility(env, env._get_observations(force_update=True))
        assert not all(r["valid"] for r in invisible)
        model.cam_pos[:] = original; env.sim.forward()
        # A non-overlapping cube inside the opened finger sweep must fail.
        centers = [env.sim.data.geom_xpos[side].mean(0) for side in validator.pad_sides]
        axis = (centers[1]-centers[0])[:2]; axis /= np.linalg.norm(axis)
        bad = layout | {"xy": [[0., 0.], (.063*axis).tolist(), [-.12, .10]], "yaw": [0., 0., 0.]}
        bad["geometry_id"] = geometry_id(bad["xy"])
        env.reset_layout(bad, "red")
        pinch = validator.grasp_plan(0)
        assert not pinch["valid"], pinch
        report["negative_controls"] = {"overlap_rejected": True, "unreachable_IK_rejected": unreachable,
                                       "invisible_scene_rejected": True, "pinch_interference_rejected": pinch}
        (RUN / "preflight/support_and_negatives.json").write_text(json.dumps(report, indent=2))
        print("NEGATIVE_CONTROLS_PASSED", flush=True)
    finally: env.close()


if __name__ == "__main__": main()
