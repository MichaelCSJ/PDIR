#!/usr/bin/env python
"""Run PDIR inference and write per-pixel PBR maps.

For every scene under ``--data-root`` this writes, into ``<out-dir>/<scene>/``:

    normal.png  albedo.png  roughness.png  metallic.png   preview images
    maps.npz                                              float32 arrays

Examples
--------
::

    python inference.py --ckpt checkpoints/pdir.ckpt \
        --dataset real --data-root /path/to/real_captures --out-dir results

    python inference.py --ckpt checkpoints/pdir.ckpt \
        --dataset synth --data-root /path/to/synth_scenes --out-dir results
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.io.dataset_real import RealDataset
from src.io.dataset_synth import SynthDataset
from src.model.net_train_module import Net


def parse_args():
    p = argparse.ArgumentParser(description="PDIR inference",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--ckpt", type=str, required=True,
                   help="Training checkpoint (.ckpt) or a bare state dict (.pth).")
    p.add_argument("--dataset", choices=["real", "synth"], default="real")
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--split-file", type=str, default=None,
                   help="Optional list of scenes to run; one name per line.")
    p.add_argument("--out-dir", type=str, default="results")
    p.add_argument("--image-size", type=int, default=384)
    p.add_argument("--canonical-resolution", type=int, default=192)
    p.add_argument("--patch-size", type=int, default=512,
                   help="Spatial tile size for full-image inference.")
    p.add_argument("--pixel-samples", type=int, default=2048)
    p.add_argument("--network-depth", type=int, default=4)
    p.add_argument("--pattern-type", choices=["RGBbin0", "RGBbin1"], default="RGBbin0")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--limit", type=int, default=None, help="Only run the first N scenes.")
    return p.parse_args()


def load_net(args) -> Net:
    """Build the network and load weights, tolerating both checkpoint layouts."""
    net = Net(
        pixel_samples=args.pixel_samples,
        network_depth=args.network_depth,
        pattern_type=args.pattern_type,
        add_noise=False,          # never inject synthetic noise at inference time
    )

    obj = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and isinstance(obj.get("state_dict"), dict):
        obj = obj["state_dict"]

    inner = {k[len("net."):]: v for k, v in obj.items() if k.startswith("net.")}
    sd = inner if inner else dict(obj)
    for old, new in (("light_axis_1_blocks", "polar_blocks"),
                     ("light_axis_2_blocks", "rgb_blocks")):
        for k, v in list(sd.items()):
            if old in k:
                sd.setdefault(k.replace(old, new), v)
    for dead in ("image_encoder.backbone.aggregator.polar_embed.weight",
                 "image_encoder.backbone.aggregator.rgb_embed.weight"):
        sd.pop(dead, None)

    missing, unexpected = net.load_state_dict(sd, strict=False)
    print(f"[ckpt] {args.ckpt}: missing={len(missing)} unexpected={len(unexpected)}")
    return net.to(args.device).eval()


def build_dataset(args):
    common = dict(data_root=args.data_root, image_size=args.image_size,
                  mode="all", split_file=args.split_file)
    return RealDataset(**common) if args.dataset == "real" else SynthDataset(**common)


def save_png(path: Path, array: np.ndarray) -> None:
    """Write an HWC float array in [0,1] (or HW for single channel) as 8-bit PNG."""
    img = np.clip(array, 0.0, 1.0)
    if img.ndim == 3 and img.shape[2] == 1:
        img = img[:, :, 0]
    Image.fromarray((img * 255.0 + 0.5).astype(np.uint8)).save(path)


def main():
    args = parse_args()
    device_type = args.device.split(":")[0]
    net = load_net(args)
    dataset = build_dataset(args)

    n = len(dataset) if args.limit is None else min(args.limit, len(dataset))
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"[run] {n} scene(s) -> {out_root}")

    for i in range(n):
        sample = dataset[i]
        scene = sample.get("scene_name", f"scene{i:04d}")

        def batched(value):
            tensor = value if torch.is_tensor(value) else torch.from_numpy(np.asarray(value))
            return tensor.unsqueeze(0).to(args.device)

        S0, S1, S2 = batched(sample["S0"]), batched(sample["S1"]), batched(sample["S2"])
        mask = batched(sample["mask"]).to(torch.float32)

        with torch.no_grad(), torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            out = net.infer_full_image_tiled(
                S0=S0, S1=S1, S2=S2, M=mask,
                pixel_samples=args.pixel_samples,
                patch_size=args.patch_size,
                canonical_resolution=args.canonical_resolution,
                dtype=torch.bfloat16,
            )

        maps = {k: out[k].to(torch.float32)[0].permute(1, 2, 0).cpu().numpy()
                for k in ("normal", "albedo", "roughness", "metallic")}

        scene_dir = out_root / scene
        scene_dir.mkdir(parents=True, exist_ok=True)
        # Normals live in [-1,1]; the other maps are already in [0,1].
        save_png(scene_dir / "normal.png", maps["normal"] * 0.5 + 0.5)
        save_png(scene_dir / "albedo.png", maps["albedo"])
        save_png(scene_dir / "roughness.png", maps["roughness"])
        save_png(scene_dir / "metallic.png", maps["metallic"])
        np.savez_compressed(scene_dir / "maps.npz", **maps)

        print(f"[{i + 1}/{n}] {scene}", flush=True)


if __name__ == "__main__":
    main()
