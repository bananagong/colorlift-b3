"""Download verified B3 release artifacts; Python standard library only."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def download(base_url, item, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.stat().st_size == item["bytes"] and sha(target) == item["sha256"]:
            print("VERIFIED", target, flush=True)
            return
        raise RuntimeError(f"Existing file differs; preserve it and choose another output directory: {target}")
    temporary = target.with_name(target.name + ".partial")
    for attempt in range(4):
        offset = temporary.stat().st_size if temporary.exists() else 0
        if offset == item["bytes"] and sha(temporary) == item["sha256"]:
            temporary.replace(target)
            return
        headers = {"User-Agent": "colorlift-b3-artifact-downloader"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(base_url + item["asset"], headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                status = getattr(response, "status", None)
                append = bool(offset and status == 206)
                if append:
                    assert response.headers["Content-Range"].startswith(f"bytes {offset}-")
                with temporary.open("ab" if append else "wb") as stream:
                    for block in iter(lambda: response.read(4 * 1024 * 1024), b""):
                        stream.write(block)
            if temporary.stat().st_size != item["bytes"] or sha(temporary) != item["sha256"]:
                raise RuntimeError(f"SHA256 or size mismatch: {item['asset']}")
            temporary.replace(target)
            print("DOWNLOADED", target, flush=True)
            return
        except Exception:
            if attempt == 3:
                raise
            time.sleep((2, 5, 10)[attempt])


def assemble(item, parts_directory, target):
    if target.exists():
        if target.stat().st_size == item["bytes"] and sha(target) == item["sha256"]:
            print("VERIFIED", target, flush=True)
            return
        raise RuntimeError(f"Refusing to overwrite a differing file: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".assembling")
    digest = hashlib.sha256()
    with temporary.open("wb") as output:
        for part in item["parts"]:
            path = parts_directory / part["asset"]
            assert path.stat().st_size == part["bytes"] and sha(path) == part["sha256"]
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(block); output.write(block)
    assert temporary.stat().st_size == item["bytes"] and digest.hexdigest() == item["sha256"]
    temporary.replace(target)
    print("ASSEMBLED_AND_VERIFIED", target, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("main", "models", "complete"), default="main")
    parser.add_argument("--output", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path, default=ROOT / "publication/release_manifest.json")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    output = args.output.resolve()
    for item in manifest["models"]:
        if args.profile != "main" or item["path"] == manifest["main_policy"]:
            download(manifest["base_url"], item, output / item["path"])
    if args.profile == "complete":
        parts = output / ".downloads"
        for item in manifest["archives"]:
            target = output / item["path"]
            if target.exists() and target.stat().st_size == item["bytes"] and sha(target) == item["sha256"]:
                continue
            for part in item["parts"]:
                download(manifest["base_url"], part, parts / part["asset"])
            assemble(item, parts, target)
        item = manifest["supplement"]
        download(manifest["base_url"], item, output / item["path"])
    print("All requested artifacts verified. Split downloads are retained in .downloads.")


if __name__ == "__main__":
    main()
