#!/usr/bin/env python
"""Fetch the pretrained weights and the sample scenes from Hugging Face.

    python download_assets.py                 # both
    python download_assets.py --only weights
    python download_assets.py --repo someone/PDIR

Lands as::

    checkpoints/pdir_best.ckpt
    sample_data/{cat,bowl,case,foil,owl}/
        mask.png
        quad{0,1,2,3}_main_hdr_{s0,s1,s2}.npy

Both directories are gitignored. If you would rather download by hand, the
files live at https://huggingface.co/<repo>/tree/main and go in the same
places.
"""

import argparse
import tarfile
import urllib.request
from pathlib import Path

# TODO: point this at the published repository.
DEFAULT_REPO = "SeokjunChoi/PDIR"

WEIGHTS = "pdir_best.ckpt"
SAMPLES = "sample_data.tar.gz"
SAMPLE_SCENES = ("cat", "bowl", "case", "foil", "owl")


def parse_args():
    p = argparse.ArgumentParser(description="Download PDIR weights and sample scenes",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--repo", default=DEFAULT_REPO, help="Hugging Face repository id.")
    p.add_argument("--base-url", default=None,
                   help="Download from this base URL instead of Hugging Face.")
    p.add_argument("--only", choices=["weights", "samples"], default=None,
                   help="Fetch just one of the two.")
    p.add_argument("--force", action="store_true", help="Re-download even if present.")
    return p.parse_args()


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[get] {url}")
    with urllib.request.urlopen(url) as response, open(tmp, "wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r      {done/1e6:7.1f} / {total/1e6:.1f} MB", end="", flush=True)
        if total:
            print()
    tmp.replace(dest)
    print(f"[ok]  {dest}")


def main():
    args = parse_args()
    base = args.base_url or f"https://huggingface.co/{args.repo}/resolve/main"
    base = base.rstrip("/")
    root = Path(__file__).resolve().parent

    if args.only != "samples":
        dest = root / "checkpoints" / WEIGHTS
        if dest.is_file() and not args.force:
            print(f"[skip] {dest} already present")
        else:
            download(f"{base}/{WEIGHTS}", dest)

    if args.only != "weights":
        sample_root = root / "sample_data"
        have = all((sample_root / s / "mask.png").is_file() for s in SAMPLE_SCENES)
        if have and not args.force:
            print(f"[skip] {sample_root} already populated")
        else:
            archive = sample_root / SAMPLES
            download(f"{base}/{SAMPLES}", archive)
            print(f"[tar] extracting into {sample_root}")
            with tarfile.open(archive) as tf:
                # `filter` exists from Python 3.12 and refuses paths outside
                # the destination; older versions just extract.
                try:
                    tf.extractall(sample_root, filter="data")
                except TypeError:
                    tf.extractall(sample_root)
            archive.unlink()
            print(f"[ok]  {sample_root}")

    print("\nTry it:")
    print("  python inference.py --ckpt checkpoints/pdir_best.ckpt \\")
    print("      --dataset real --data-root sample_data --out-dir results \\")
    print("      --exposure-norm p95_0.95")


if __name__ == "__main__":
    main()
