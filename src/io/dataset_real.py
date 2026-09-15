"""
Real polarization-camera dataset loader for the Stokes-based PDIR pipeline.

Reads 20260427_real-format scenes:
    {scene_dir}/
        quad{0..3}_main_hdr_s0.npy           per-quadrant Stokes S0 (>=0)
        quad{0..3}_main_hdr_s1.npy           per-quadrant Stokes S1 (signed)
        quad{0..3}_main_hdr_s2.npy           per-quadrant Stokes S2 (signed)
        mask.png                             (uint8 0/255, refined object mask)

Returns raw per-quadrant Stokes for `Net.simulate_pattern_from_stokes`. No
GT material maps -> intended for real_adapt loss (render-space only).

Two split strategies:
  (1) Single root + val_stride (legacy): every val_stride-th scene -> val.
  (2) `split_file` + `data_roots` (new): explicit train/val txt files with
      lines formatted as `<prefix>/<scene_name>`. Multiple capture sessions
      can be combined: the prefix selects which physical directory hosts
      each scene via the `data_roots` dict mapping.

This loader does NOT inherit from any base class and does NOT emit legacy
Depol/Polar/CoP keys.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


# RGBbin0 per-channel quadrant weights (R = quad1+quad3, G = quad0+quad2,
# B = quad0+quad1). Used for the per-scene exposure factor: f is computed on
# fg luminance of the post-quadrant-sum image (= what the encoder ultimately
# sees as `s0_pat`).
_RGBBIN0_3x4 = np.array(
    [[0, 1, 0, 1],
     [1, 0, 1, 0],
     [1, 1, 0, 0]], dtype=np.float32,
)


class RealDataset(Dataset):
    """Stokes-based real-capture dataset (20260427_real layout)."""

    QUAD_PREFIXES = ("quad0_main", "quad1_main", "quad2_main", "quad3_main")

    def __init__(
        self,
        data_root: Optional[str] = None,
        image_size: int = 384,
        mode: str = "Train",
        min_mask_pixels: int = 1,
        scene_ids_path: Optional[str] = None,
        val_stride: int = 10,
        # max_scenes: Optional[int] = 2,
        max_scenes: Optional[int] = None,
        data_roots: Optional[Dict[str, str]] = None,
        split_file: Optional[str] = None,
        exposure_norm: Optional[str] = None,
        exposure_norm_clamp: Tuple[float, float] = (0.5, 4.0),
    ):
        """
        Per-scene HDR exposure normalization (off by default):
            exposure_norm: one of {None, "p95_0.95", "p99_0.95", "mean_0.4",
                                   "median_0.4", "max_0.99", "reinhard_0.18"}.
                Applies a single scalar `f` to S0/S1/S2 (DoLP/AoLP preserved).
                See `scripts/analyze_pattern_color_saturation.py`-companion
                analysis at /tmp/exposure_norm_analysis.py for benchmarks.
            exposure_norm_clamp: (lo, hi) bounds on f to avoid extreme noise
                amplification on dim scenes (default [0.5, 4.0]; recommended
                for "p95_0.95" whose unbounded f reaches ~10× on dim scenes).
        """
        self.data_root = Path(data_root) if data_root else None
        self.image_size = int(image_size)
        self.mode = str(mode).lower()
        self.min_mask_pixels = int(min_mask_pixels)
        self.val_stride = max(2, int(val_stride))
        self.data_roots = {k: Path(v) for k, v in (data_roots or {}).items()}
        self.split_file = Path(split_file) if split_file else None
        self.exposure_norm = exposure_norm if exposure_norm else None
        self.exposure_norm_clamp = (float(exposure_norm_clamp[0]),
                                    float(exposure_norm_clamp[1]))
        if self.exposure_norm is not None and self.exposure_norm not in (
            "p95_0.95", "p99_0.95", "mean_0.4", "median_0.4",
            "max_0.99", "reinhard_0.18",
        ):
            raise ValueError(f"Unknown exposure_norm: {self.exposure_norm!r}")

        if self.split_file is not None:
            scene_dirs = self._scene_dirs_from_split(self.split_file)
        else:
            if self.data_root is None:
                raise ValueError("RealDataset requires either `data_root` (single) "
                                 "or `split_file` + `data_roots` (multi).")
            # Any directory holding the required files counts as a scene, so
            # readable names such as `cat/` work as well as `scene29_.../`.
            scene_dirs = sorted(p for p in self.data_root.iterdir() if p.is_dir())
            if scene_ids_path is not None and Path(scene_ids_path).is_file():
                wanted = {ln.strip() for ln in Path(scene_ids_path).read_text().splitlines() if ln.strip()}
                scene_dirs = [p for p in scene_dirs if p.name in wanted]

        good = []
        for d in scene_dirs:
            if (d / "mask.png").is_file() \
                    and (d / "quad0_main_hdr_s0.npy").is_file():
                good.append(d)
        if len(good) < len(scene_dirs):
            print(f"[WARN] {len(scene_dirs) - len(good)} scene(s) skipped (missing mask.png or quad0_main_hdr_s0.npy)")
        scene_dirs = good

        if max_scenes is not None:
            scene_dirs = scene_dirs[: int(max_scenes)]
        if not scene_dirs:
            raise RuntimeError(
                f"No real scenes found "
                f"({'split_file=' + str(self.split_file) if self.split_file else 'data_root=' + str(self.data_root)})"
            )

        if self.split_file is None and self.mode in ("train", "val"):
            # Legacy val_stride split (used only when no split_file is given).
            train_dirs = [p for i, p in enumerate(scene_dirs) if (i % self.val_stride) != 0]
            val_dirs = [p for i, p in enumerate(scene_dirs) if (i % self.val_stride) == 0]
            scene_dirs = train_dirs if self.mode == "train" else val_dirs
        elif self.split_file is None and self.mode not in ("all", "test"):
            raise ValueError(f"Unsupported mode: {mode}")
        if not scene_dirs:
            raise RuntimeError(f"No real scenes remain after split for mode={mode}")

        self.scene_dirs = scene_dirs
        self.objlist = scene_dirs

    def _scene_dirs_from_split(self, split_file: Path):
        """Resolve `<prefix>/<scene_name>` lines from split_file into Paths.

        `<prefix>` is looked up in self.data_roots; if absent and a
        single self.data_root is also given, that is used as fallback
        (legacy single-session layouts). Bare `<scene_name>` lines (no `/`)
        require self.data_root.
        """
        lines = [ln.strip() for ln in Path(split_file).read_text().splitlines() if ln.strip()]
        scene_dirs = []
        missing = []
        for line in lines:
            if "/" in line:
                prefix, scene_name = line.split("/", 1)
                if prefix in self.data_roots:
                    root = self.data_roots[prefix]
                elif self.data_root is not None:
                    root = self.data_root                                  # ignore prefix fallback
                else:
                    raise ValueError(
                        f"split_file line '{line}' has prefix '{prefix}' but no "
                        f"matching `data_roots` entry; configured roots: "
                        f"{list(self.data_roots.keys())}"
                    )
            else:
                if self.data_root is None:
                    raise ValueError(
                        f"split_file line '{line}' has no prefix and no `data_root` fallback."
                    )
                scene_name = line
                root = self.data_root
            sd = root / scene_name
            if not sd.is_dir():
                missing.append(str(sd))
                continue
            scene_dirs.append(sd)
        if missing:
            print(f"[WARN] {len(missing)} scene(s) listed in {split_file.name} "
                  f"not found on disk; first 3: {missing[:3]}")
        return scene_dirs

    def __len__(self):
        return len(self.scene_dirs)

    def _crop_hwc(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        if self.image_size > h or self.image_size > w:
            raise ValueError(
                f"image_size={self.image_size} exceeds source size {(h, w)}; loader does not upscale."
            )
        if h == self.image_size and w == self.image_size:
            return img.astype(np.float32)
        top = (h - self.image_size) // 2
        left = (w - self.image_size) // 2
        return img[top : top + self.image_size, left : left + self.image_size].astype(np.float32)

    def _stack_quads(self, scene_dir: Path, modality: str) -> np.ndarray:
        # [4, H, W, 3] in quad0..quad3 order.
        arrs = [np.load(scene_dir / f"{p}_hdr_{modality}.npy").astype(np.float32) for p in self.QUAD_PREFIXES]
        stacked = np.stack(arrs, axis=0)
        if stacked.shape[1] != self.image_size or stacked.shape[2] != self.image_size:
            stacked = np.stack([self._crop_hwc(x) for x in stacked], axis=0)
        return stacked

    def _load_mask(self, scene_dir: Path) -> np.ndarray:
        mask_img = np.array(Image.open(scene_dir / "mask.png"))
        if mask_img.ndim == 3:
            mask_img = mask_img[..., 0]
        mask = (mask_img.astype(np.float32) / 255.0 > 0.5).astype(np.float32)
        if mask.shape != (self.image_size, self.image_size):
            mask = self._crop_hwc(mask[..., None])[..., 0]
        return mask[..., None]                                   # [H, W, 1]

    def _compute_exposure_factor(self, s0_q: np.ndarray, mask_hw: np.ndarray) -> float:
        """Per-scene gain `f` such that x' = clip(x · f, 0, 1) maps fg HDR
        captures to a comparable [0,1] range across scenes. Computed on the
        post-quadrant-sum luminance (i.e. the same signal the encoder sees
        as `s0_pat`)."""
        if self.exposure_norm is None:
            return 1.0
        # sim_naive = RGBbin0-weighted sum of S0 quads, [H, W, 3].
        H, W = mask_hw.shape
        sim = np.zeros((H, W, 3), dtype=np.float32)
        for ci in range(3):
            sim[..., ci] = (s0_q[..., ci] * _RGBBIN0_3x4[ci, :, None, None]).sum(axis=0)
        L = 0.299 * sim[..., 0] + 0.587 * sim[..., 1] + 0.114 * sim[..., 2]
        L_fg = L[mask_hw > 0.5]
        if L_fg.size < 100:
            return 1.0
        if self.exposure_norm == "p95_0.95":
            f = 0.95 / max(float(np.percentile(L_fg, 95)), 1e-6)
        elif self.exposure_norm == "p99_0.95":
            f = 0.95 / max(float(np.percentile(L_fg, 99)), 1e-6)
        elif self.exposure_norm == "mean_0.4":
            f = 0.4  / max(float(L_fg.mean()), 1e-6)
        elif self.exposure_norm == "median_0.4":
            f = 0.4  / max(float(np.median(L_fg)), 1e-6)
        elif self.exposure_norm == "max_0.99":
            f = 0.99 / max(float(L_fg.max()), 1e-6)
        elif self.exposure_norm == "reinhard_0.18":
            L_pos = L_fg[L_fg > 1e-6] if (L_fg > 1e-6).any() else L_fg + 1e-6
            f = 0.18 / max(float(np.exp(np.log(L_pos + 1e-3).mean())), 1e-6)
        else:
            return 1.0
        lo, hi = self.exposure_norm_clamp
        return float(np.clip(f, lo, hi))

    def __getitem__(self, index: int):
        scene_dir = self.scene_dirs[index]

        s0_q = self._stack_quads(scene_dir, "s0")                # [4, H, W, 3] >= 0
        s1_q = self._stack_quads(scene_dir, "s1")                # [4, H, W, 3] signed
        s2_q = self._stack_quads(scene_dir, "s2")                # [4, H, W, 3] signed
        s0_q = np.clip(s0_q, 0.0, None)                          # safety: S0 must be non-negative

        mask = self._load_mask(scene_dir)                         # [H, W, 1]
        valid_pixels = int(mask.sum())
        if valid_pixels < self.min_mask_pixels:
            raise RuntimeError(
                f"Real scene '{scene_dir.name}' has too few valid pixels: "
                f"{valid_pixels} < {self.min_mask_pixels}"
            )

        s0_q = s0_q * mask[None, ...]
        s1_q = s1_q * mask[None, ...]
        s2_q = s2_q * mask[None, ...]

        # Per-scene exposure normalization (off by default). Same scalar to
        # S0/S1/S2 preserves DoLP/AoLP. Downstream modality clipping in
        # simulate_pattern_from_stokes will clip the rare overshoot.
        exposure_factor = self._compute_exposure_factor(s0_q, mask[..., 0])
        if exposure_factor != 1.0:
            s0_q = s0_q * exposure_factor
            s1_q = s1_q * exposure_factor
            s2_q = s2_q * exposure_factor

        position = np.zeros((self.image_size, self.image_size, 3), dtype=np.float32)

        try:
            scene_idx = int(scene_dir.name.split("_", 1)[0].replace("scene", ""))
        except Exception:
            scene_idx = -1

        def to_chwt(arr):
            return torch.from_numpy(arr).permute(3, 1, 2, 0).contiguous().float()

        return {
            "scene_idx":  scene_idx,
            "scene_name": scene_dir.name,
            "objname":    scene_dir.name,
            # OLAT-Stokes inputs, [3, H, W, 4]
            "S0": to_chwt(s0_q),
            "S1": to_chwt(s1_q),
            "S2": to_chwt(s2_q),
            # Geometry
            "mask":     mask.transpose(2, 0, 1)[..., None],
            "position": position.transpose(2, 0, 1)[..., None],
            # Flags
            "numberOfImages": 9,
            "is_real": True,
            "exposure_factor": float(exposure_factor),
        }
