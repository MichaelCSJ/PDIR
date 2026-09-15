#!/usr/bin/env python
"""Publish the pretrained weights and the sample captures to Hugging Face.

The counterpart of `download_assets.py`. Run it after `hf auth login`:

    python upload_assets.py                     # both, to the default repo
    python upload_assets.py --only weights
    python upload_assets.py --repo you/other-repo --private

Reads `checkpoints/pdir_best.ckpt` and packs `sample_data/<scene>/` into
`sample_data.tar.gz` on the fly, so the layout matches what the downloader
expects.
"""

import argparse
import tarfile
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

DEFAULT_REPO = "SeokjunChoi/snapshot-polarimetric-dir"
WEIGHTS = "pdir_best.ckpt"
SAMPLES = "sample_data.tar.gz"


def parse_args():
    p = argparse.ArgumentParser(description="Upload PDIR assets to Hugging Face",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--repo", default=DEFAULT_REPO)
    p.add_argument("--repo-type", default="model", choices=["model", "dataset"])
    p.add_argument("--private", action="store_true", help="Create the repo private.")
    p.add_argument("--only", choices=["weights", "samples"], default=None)
    return p.parse_args()


def pack_samples(sample_root: Path, dest: Path) -> Path:
    scenes = sorted(d for d in sample_root.iterdir()
                    if d.is_dir() and (d / "mask.png").is_file())
    if not scenes:
        raise SystemExit(f"No sample scenes under {sample_root}")
    print(f"[tar] packing {len(scenes)} scene(s): {', '.join(d.name for d in scenes)}")
    with tarfile.open(dest, "w:gz") as tf:
        for d in scenes:
            tf.add(d, arcname=d.name)
    print(f"[tar] {dest} ({dest.stat().st_size/1e6:.1f} MB)")
    return dest


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent
    api = HfApi()

    api.create_repo(args.repo, repo_type=args.repo_type,
                    private=args.private, exist_ok=True)
    print(f"[repo] {args.repo} ({args.repo_type}, "
          f"{'private' if args.private else 'public'})")

    if args.only != "samples":
        weights = root / "checkpoints" / WEIGHTS
        if not weights.is_file():
            raise SystemExit(f"Missing {weights}")
        print(f"[put] {WEIGHTS} ({weights.stat().st_size/1e6:.1f} MB)")
        api.upload_file(path_or_fileobj=str(weights), path_in_repo=WEIGHTS,
                        repo_id=args.repo, repo_type=args.repo_type)

    if args.only != "weights":
        with tempfile.TemporaryDirectory() as tmp:
            archive = pack_samples(root / "sample_data", Path(tmp) / SAMPLES)
            print(f"[put] {SAMPLES}")
            api.upload_file(path_or_fileobj=str(archive), path_in_repo=SAMPLES,
                            repo_id=args.repo, repo_type=args.repo_type)

    url = f"https://huggingface.co/{args.repo}"
    if args.repo_type == "dataset":
        url = f"https://huggingface.co/datasets/{args.repo}"
    print(f"\n[done] {url}")


if __name__ == "__main__":
    main()
