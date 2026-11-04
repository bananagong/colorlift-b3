"""Export inference-only checkpoints without optimizer state."""
import json
import hashlib
import torch
from .env import EXPERIMENT


def main():
    for count in (20, 50):
        root = EXPERIMENT / f"act_{count}"
        checkpoint = torch.load(root / "best.pt", map_location="cpu", weights_only=False)
        payload = {key: checkpoint[key] for key in ("model", "stats", "config", "step", "val_prior_l1")}
        payload["protocol"] = json.loads((EXPERIMENT / "protocol.json").read_text())
        torch.save(payload, root / "policy.pt")
        with (root / "policy.pt").open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        (root / "policy.sha256").write_text(digest + "  policy.pt\n")
        print(f'Exported {root / "policy.pt"}: step={checkpoint["step"]}', flush=True)


if __name__ == "__main__":
    main()
