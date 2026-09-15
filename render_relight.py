#!/usr/bin/env python
"""Relight the estimated PBR maps with a point light orbiting the scene.

Reads the ``maps.npz`` files written by ``inference.py`` and renders one frame
per orbit step through the same principled BRDF the training render loss uses.

Examples
--------
::

    python render_relight.py --maps-dir results --out-dir relight

    # one scene, slower orbit, brighter
    python render_relight.py --maps-dir results --out-dir relight \
        --scenes scene1_20260427_211922 --n-frames 120 --brightness 70

Writes ``<out-dir>/<scene>/frame####.png`` and, unless ``--no-video``,
``<out-dir>/<scene>.mp4``. Video writing needs either ``imageio-ffmpeg``
(H.264, plays in browsers) or OpenCV's bundled encoder as a fallback.
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.model.renderer import Principled_BRDF


def parse_args():
    p = argparse.ArgumentParser(description="Rotating-light relighting",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--maps-dir", type=str, required=True,
                   help="Directory of inference results, one subdirectory per scene.")
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--scenes", type=str, nargs="*", default=None,
                   help="Scene names to render. Default: every scene in --maps-dir.")
    p.add_argument("--n-frames", type=int, default=60)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--radius", type=float, default=0.4,
                   help="Orbit radius of the point light, in the display calibration's units.")
    p.add_argument("--center", type=float, nargs=3, default=(0.0, -0.29, 0.0),
                   metavar=("X", "Y", "Z"), help="Orbit centre.")
    p.add_argument("--brightness", type=float, default=50.0,
                   help="Exposure applied to the HDR render before clipping to [0,1].")
    p.add_argument("--no-video", action="store_true", help="Only write the PNG frames.")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_maps(path: Path):
    """Return masked (normal, albedo, roughness, metallic) as HWC float32 arrays."""
    z = np.load(path)
    mask = z["mask"][..., None].astype(np.float32) if "mask" in z.files else 1.0
    rough, metal = z["roughness"], z["metallic"]
    if rough.ndim == 2:
        rough, metal = rough[..., None], metal[..., None]
    return (z["normal"] * mask, z["albedo"] * mask, rough * mask, metal * mask)


@torch.no_grad()
def render_one_light(renderer, albedo, normal, rough, metal, light_pos):
    """Render the maps under a single point light at `light_pos` ([1,3])."""
    H, W = albedo.shape[0], albedo.shape[1]
    saved = renderer.light_positions
    renderer.light_positions = light_pos.to(saved.device, dtype=saved.dtype)
    try:
        diffuse, specular = renderer(albedo.reshape(-1, 3), normal.reshape(-1, 3),
                                     rough.reshape(-1, 1), metal.reshape(-1, 1),
                                     point_cloud=None)
    finally:
        renderer.light_positions = saved
    return (diffuse + specular).squeeze(1).reshape(H, W, 3).float().cpu().numpy()


def orbit(center, radius, n_frames) -> np.ndarray:
    """Point-light positions on a circle in the display plane."""
    cx, cy, cz = center
    theta = np.linspace(0.0, 2.0 * math.pi, n_frames, endpoint=False)
    return np.stack([cx + radius * np.cos(theta),
                     cy + radius * np.sin(theta),
                     np.full_like(theta, cz)], axis=-1).astype(np.float32)


def write_video(frames, path: Path, fps: int) -> None:
    """H.264 through imageio-ffmpeg when available, OpenCV otherwise."""
    try:
        import imageio.v2 as imageio
        imageio.mimwrite(path, frames, fps=fps, codec="libx264",
                         quality=8, macro_block_size=1)
        return
    except Exception as exc:
        print(f"[video] imageio/ffmpeg unavailable ({exc}); falling back to OpenCV")

    import cv2
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        writer.write(f[..., ::-1])
    writer.release()


def main():
    args = parse_args()
    maps_root, out_root = Path(args.maps_dir), Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    scenes = args.scenes or sorted(d.name for d in maps_root.iterdir()
                                   if (d / "maps.npz").is_file())
    if not scenes:
        raise SystemExit(f"No scenes with maps.npz under {maps_root}")

    lights = orbit(args.center, args.radius, args.n_frames)
    print(f"[orbit] centre={tuple(args.center)} radius={args.radius} frames={args.n_frames}")

    renderer = Principled_BRDF().to(args.device).eval()

    for i, scene in enumerate(scenes, 1):
        normal, albedo, rough, metal = load_maps(maps_root / scene / "maps.npz")
        tensors = [torch.from_numpy(np.ascontiguousarray(a)).to(args.device)
                   for a in (albedo, normal, rough, metal)]

        scene_dir = out_root / scene
        scene_dir.mkdir(parents=True, exist_ok=True)
        frames = []
        for f in range(args.n_frames):
            hdr = render_one_light(renderer, *tensors,
                                   torch.from_numpy(lights[f:f + 1]))
            rgb = (np.clip(hdr * args.brightness, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
            frames.append(rgb)
            Image.fromarray(rgb).save(scene_dir / f"frame{f:04d}.png")

        if not args.no_video:
            write_video(frames, out_root / f"{scene}.mp4", args.fps)
        print(f"[{i}/{len(scenes)}] {scene}: {args.n_frames} frames", flush=True)

    print(f"[done] -> {out_root}")


if __name__ == "__main__":
    main()
