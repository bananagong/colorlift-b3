"""Check real forward/backward and inference for all three ACT input structures."""
import gc
import json
import numpy as np
import torch
from .config import EXPERIMENT
from .data import Dataset
from .model import ColorACT, VARIANTS, loss_components


def main():
    assert torch.cuda.device_count() == 1
    torch.set_num_threads(4)
    data = Dataset("pilot", 3)
    results = []
    for variant in VARIANTS:
        torch.manual_seed(42)
        model = ColorACT(variant).cuda()
        s, image, action, pad = data.batch(np.array([0, 10]), 16, variant)
        prediction = model(s, image, action, pad)
        loss, xyz, phase, accuracy, kl = loss_components(prediction, action, pad)
        assert prediction[0].shape == (2, 16, 7) and torch.isfinite(loss)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        model.eval()
        with torch.inference_mode():
            output = model(s, image)[0]
        assert output.shape == (2, 16, 7) and torch.isfinite(output).all()
        results.append({"variant": variant, "image_shape": list(image.shape), "state_shape": list(s.shape),
                        "output_shape": list(output.shape), "parameters": sum(p.numel() for p in model.parameters()), "passed": True})
        del model, prediction, output, loss
        gc.collect()
        torch.cuda.empty_cache()
    (EXPERIMENT / "environment" / "models.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
