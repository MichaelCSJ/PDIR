"""
Mitsuba synth dataset loader for the Stokes-based PDIR pipeline.

Reads dataset_0424-format scenes:
    {scene_dir}/
        obs_original.npz     per-quadrant diffuse / specular / cop
        gt_maps.npz          basecolor / roughness / metallic / normal / mask

Returns raw per-quadrant Stokes for `Net.simulate_pattern_from_stokes`:
    S0_q  =  diffuse + specular        # total intensity   (>=0)
    S1_q  =  specular                  # linear pol along 0/90 axis
    S2_q  =  cop * (diffuse + specular)
                                       # network-input slot for the polarization
                                       # direction signal. For synth this is
                                       # physically S3 (circular pol amplitude),
                                       # used identically by the network.

PATTERN_KEYS ordering verified empirically by sweep_pattern_keys.py: with
this ordering and RGBbin0 per-channel weights, simulate_pattern_from_stokes
matches the renderer's obs_weighted output to ~1e-5 mean|d| in fg.

Two split strategies (mirrors RealDataset):
  (1) Single root + val_stride (legacy): every val_stride-th scene -> val.
  (2) `split_file` + `data_roots` (new): explicit train/val/test txt files
      with lines formatted as `<prefix>/<scene_NNNN>`. Multiple synth render
      passes can be combined: the prefix selects which physical directory
      hosts each scene via the `data_roots` dict mapping.

This loader does NOT inherit from any base class and does NOT emit legacy
Depol/Polar/CoP keys. Down-stream code must consume S0/S1/S2.

Optional anti-aliasing (`antialias_edges=True`) applies a median filter to
the OLAT Stokes tensors at GT material boundaries to suppress sub-pixel
boundary noise; off by default.
"""

import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from scipy.ndimage import binary_dilation, binary_erosion, median_filter
from torch.utils.data import Dataset


class _BadSceneError(RuntimeError):
    """Internal signal that a synth scene is unusable (empty mask, broken
    npz, missing key etc.) and __getitem__ should retry with another index
    instead of crashing the dataloader worker.
    """
    pass


class SynthDataset(Dataset):
    """Stokes-based synthetic Mitsuba dataset (dataset_0424 layout)."""

    PATTERN_KEYS = ("1_0", "0_0", "1_1", "0_1")

    def __init__(
        self,
        data_root: Optional[str] = None,
        image_size: int = 384,
        mode: str = "Train",
        min_mask_pixels: int = 1,
        scene_ids_path: Optional[str] = None,
        # max_scenes: Optional[int] = 2,
        max_scenes: Optional[int] = None,
        val_stride: int = 10,
        antialias_edges: bool = False,
        edge_threshold: float = 0.04,
        edge_dilate: int = 1,
        median_kernel: int = 3,
        data_roots: Optional[Dict[str, str]] = None,
        split_file: Optional[str] = None,
    ):
        self.data_root = Path(data_root) if data_root else None
        self.image_size = int(image_size)
        self.mode = str(mode).lower()
        self.min_mask_pixels = int(min_mask_pixels)
        self.val_stride = max(2, int(val_stride))
        self.data_roots = {k: Path(v) for k, v in (data_roots or {}).items()}
        self.split_file = Path(split_file) if split_file else None

        self.antialias_edges = bool(antialias_edges)
        self.edge_threshold = float(edge_threshold)
        self.edge_dilate = max(0, int(edge_dilate))
        if median_kernel < 3 or median_kernel % 2 == 0:
            raise ValueError(f"median_kernel must be odd >= 3, got {median_kernel}")
        self.median_kernel = int(median_kernel)

        # Discover scene directories.
        # Priority: split_file > scene_ids.txt > dir-listing fallback.
        if self.split_file is not None:
            scene_dirs = self._scene_dirs_from_split(self.split_file)
        else:
            if self.data_root is None:
                raise ValueError("SynthDataset requires either `data_root` (single) "
                                 "or `split_file` + `data_roots` (multi).")
            if scene_ids_path is None:
                scene_ids_path = self.data_root / "scene_ids.txt"
            scene_ids_path = Path(scene_ids_path)
            if scene_ids_path.is_file():
                scene_names = [ln.strip().zfill(4) for ln in scene_ids_path.read_text().splitlines() if ln.strip()]
                scene_dirs = [self.data_root / n for n in scene_names if (self.data_root / n).is_dir()]
            else:
                scene_dirs = sorted([p for p in self.data_root.iterdir() if p.is_dir() and p.name.isdigit()])

        if max_scenes is not None:
            scene_dirs = scene_dirs[: int(max_scenes)]
        if not scene_dirs:
            raise RuntimeError(
                f"No synth scenes found "
                f"({'split_file=' + str(self.split_file) if self.split_file else 'data_root=' + str(self.data_root)})"
            )

        # Legacy val_stride split is bypassed when an explicit split_file
        # already dictates train/val membership.
        if self.split_file is None and self.mode in ("train", "val"):
            train_dirs = [p for i, p in enumerate(scene_dirs) if (i % self.val_stride) != 0]
            val_dirs = [p for i, p in enumerate(scene_dirs) if (i % self.val_stride) == 0]
            scene_dirs = train_dirs if self.mode == "train" else val_dirs
        elif self.split_file is None and self.mode not in ("all", "test"):
            raise ValueError(f"Unsupported mode: {mode}")
        if not scene_dirs:
            raise RuntimeError(f"No synth scenes remain after split for mode={mode}")

        self.scene_dirs = scene_dirs
        self.objlist = scene_dirs

    def _scene_dirs_from_split(self, split_file: Path):
        """Resolve `<prefix>/<scene_NNNN>` lines from split_file into Paths.

        Mirrors RealDataset._scene_dirs_from_split. `<prefix>` is looked up in
        self.data_roots; if absent and self.data_root is set, that is used as
        fallback. Bare `<scene_name>` lines (no `/`) require self.data_root.
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
                    root = self.data_root
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
            print(f"[WARN] {len(missing)} synth scene(s) listed in {split_file.name} "
                  f"not found on disk; first 3: {missing[:3]}")
        return scene_dirs

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.scene_dirs)

    def _crop_hwc(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        if self.image_size > h or self.image_size > w:
            raise ValueError(f"image_size={self.image_size} exceeds source size {(h, w)}")
        if h == self.image_size and w == self.image_size:
            return img.astype(np.float32)
        top = (h - self.image_size) // 2
        left = (w - self.image_size) // 2
        return img[top : top + self.image_size, left : left + self.image_size].astype(np.float32)

    def _crop_stack(self, arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 4:
            return np.stack([self._crop_hwc(x) for x in arr], axis=0)
        return self._crop_hwc(arr)

    def _stack_obs(self, obs, prefix: str) -> np.ndarray:
        arr = np.stack([obs[f"{prefix}_{k}"] for k in self.PATTERN_KEYS], axis=0)
        if arr.shape[1] != self.image_size or arr.shape[2] != self.image_size:
            arr = self._crop_stack(arr)
        return arr.astype(np.float32)                          # [4, H, W, 3]

    @staticmethod
    def _ensure_single_channel(arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 2:
            return arr[..., None]
        if arr.ndim == 3 and arr.shape[-1] >= 1:
            return arr[..., :1]
        raise ValueError(f"Unexpected scalar-map shape: {arr.shape}")

    @staticmethod
    def _normal_display_to_training(normal_01: np.ndarray) -> np.ndarray:
        # display [0,1] -> training [-1,1]
        return 1.0 - 2.0 * normal_01.astype(np.float32)

    # ------------------------------------------------------------------
    # optional anti-aliasing on OLAT Stokes tensors
    # ------------------------------------------------------------------

    def _compute_material_edge_mask(self, basecolor, roughness, metallic):
        cat = np.concatenate([basecolor, roughness, metallic], axis=2)
        gx = np.abs(np.diff(cat, axis=1, prepend=cat[:, :1, :]))
        gy = np.abs(np.diff(cat, axis=0, prepend=cat[:1, :, :]))
        grad = np.maximum(gx, gy).max(axis=2)
        edge = grad > self.edge_threshold
        if self.edge_dilate > 0:
            edge = binary_dilation(edge, iterations=self.edge_dilate)
        return edge.astype(np.float32)

    @staticmethod
    def _median_replace_chwt(tensor: torch.Tensor, gate_hw: np.ndarray, kernel: int) -> torch.Tensor:
        arr = tensor.detach().cpu().numpy()
        out = arr.copy()
        gate = gate_hw > 0.5
        for c in range(arr.shape[0]):
            for t in range(arr.shape[3]):
                med = median_filter(arr[c, :, :, t], size=kernel, mode="reflect")
                out[c, :, :, t] = np.where(gate, med, arr[c, :, :, t])
        return torch.from_numpy(out)

    def _maybe_apply_edge_aa(self, sample, basecolor, roughness, metallic, mask_hw):
        """basecolor/roughness/metallic are pre-mask numpy [H,W,C]; mask_hw is [H,W] bool/float."""
        fg = mask_hw > 0.5
        erode_iters = self.median_kernel // 2 + self.edge_dilate
        fg_safe = binary_erosion(fg, iterations=erode_iters) if erode_iters > 0 else fg

        edge_full = self._compute_material_edge_mask(basecolor, roughness, metallic)
        edge_mask = edge_full * fg_safe.astype(np.float32)
        if edge_mask.sum() < 1:
            return sample

        for key in ("S0", "S1", "S2"):
            sample[key] = self._median_replace_chwt(sample[key], edge_mask, self.median_kernel)
        sample["edge_mask"] = torch.from_numpy(edge_mask).float()
        return sample

    # ------------------------------------------------------------------
    # __getitem__ (retry-wrapped) + _load_item (raw loader)
    # ------------------------------------------------------------------

    # Cap on retries before giving up — large enough to walk past any
    # plausible cluster of bad scenes, small enough to fail loud if the
    # dataset is truly broken.
    _MAX_BAD_SCENE_RETRIES = 16

    def __getitem__(self, index: int):
        """Retry-wrapped item access.

        If the picked scene fails the validity check (`_BadSceneError`) or
        a low-level load error fires (e.g. truncated npz, missing key), we
        log a warning and re-roll a different index. Each call retries up
        to `_MAX_BAD_SCENE_RETRIES` times. This keeps the DataLoader worker
        alive instead of crashing the whole training run.
        """
        seen: set[int] = set()
        cur = int(index)
        last_err: Exception | None = None
        for _ in range(self._MAX_BAD_SCENE_RETRIES):
            try:
                return self._load_item(cur)
            except (_BadSceneError, FileNotFoundError, KeyError, OSError, ValueError) as e:
                last_err = e
                seen.add(cur)
                # Pick a fresh random index that we haven't tried this call.
                choices = [i for i in range(len(self.scene_dirs)) if i not in seen]
                if not choices:
                    break
                new_idx = random.choice(choices)
                print(
                    f"[WARN] SynthDataset: bad scene at idx {cur} "
                    f"({self.scene_dirs[cur].name}); retrying idx {new_idx} "
                    f"({self.scene_dirs[new_idx].name}). Reason: {str(e)[:160]}",
                    flush=True,
                )
                cur = new_idx
        raise RuntimeError(
            f"SynthDataset: max retries ({self._MAX_BAD_SCENE_RETRIES}) exhausted "
            f"starting from index {index}. Last error: {last_err}"
        ) from last_err

    def _load_item(self, index: int):
        scene_dir = self.scene_dirs[index]
        obs = np.load(scene_dir / "obs_original.npz")
        gt = np.load(scene_dir / "gt_maps.npz")

        # Per-quadrant Stokes from synth obs decomposition
        diff_q = self._stack_obs(obs, "diffuse")               # [4, H, W, 3]
        spec_q = self._stack_obs(obs, "specular")
        cop_q  = self._stack_obs(obs, "cop")
        # cop_q is the per-pattern circular polarisation ratio s2/max(s0,|sqrt(s1²+s2²)|);
        # |cop| <= 1 by definition. yunseong_ablation has rare outliers (1e3–1e4) from
        # numerical issues during data generation (s0 near 0). They survive bf16 fine
        # individually but `S2_q = cop_q * S0_q` then `pol_mag = sqrt(s1²+s2²)` overflows
        # bf16 (max ≈6.5e4) -> Inf -> NaN cascade across all losses.
        cop_q = np.clip(cop_q, -1.0, 1.0)

        S0_q = diff_q + spec_q
        S1_q = spec_q
        S2_q = cop_q * S0_q                                    # circular pol amplitude (network slot)

        # GT material maps
        basecolor = gt["basecolor"].astype(np.float32)
        roughness = self._ensure_single_channel(gt["roughness"].astype(np.float32))
        metallic  = self._ensure_single_channel(gt["metallic"].astype(np.float32))
        normal_01 = gt["normal"].astype(np.float32)
        if basecolor.shape[0] != self.image_size or basecolor.shape[1] != self.image_size:
            basecolor = self._crop_hwc(basecolor)
            roughness = self._crop_hwc(roughness)
            metallic  = self._crop_hwc(metallic)
            normal_01 = self._crop_hwc(normal_01)

        # Mask
        if "mask" not in gt.files:
            raise RuntimeError(f"SynthDataset requires 'mask' in gt_maps.npz (scene {scene_dir.name})")
        mask = gt["mask"].astype(np.float32)
        if mask.ndim == 2:
            mask = mask[..., None]
        elif mask.ndim == 3 and mask.shape[-1] != 1:
            mask = mask[..., :1]
        if mask.shape[0] != self.image_size or mask.shape[1] != self.image_size:
            mask = self._crop_hwc(mask)

        valid_pixels = int(mask[..., 0].sum())
        if valid_pixels < self.min_mask_pixels:
            # Signaled to __getitem__'s retry wrapper; not a hard crash.
            raise _BadSceneError(
                f"Synth scene '{scene_dir.name}' has too few valid pixels: "
                f"{valid_pixels} < {self.min_mask_pixels}"
            )

        # Apply mask to all observations + GT
        S0_q = S0_q * mask[None, ...]
        S1_q = S1_q * mask[None, ...]
        S2_q = S2_q * mask[None, ...]
        normal = self._normal_display_to_training(normal_01) * mask
        basecolor_m = basecolor * mask
        roughness_m = roughness * mask
        metallic_m  = metallic  * mask
        position = np.zeros_like(normal, dtype=np.float32)

        # Multi-session: include parent dataset prefix to keep names unique
        # across e.g. dataset_0429_mark9/0042 vs dataset_0429_mark10/0042.
        scene_label = (
            f"{scene_dir.parent.name}/{scene_dir.name}"
            if self.split_file is not None
            else scene_dir.name
        )
        sample = {
            "scene_idx":  int(scene_dir.name),
            "scene_name": scene_label,
            "objname":    scene_label,
            # OLAT-Stokes inputs (per quadrant), [3, H, W, 4]
            "S0": torch.from_numpy(S0_q).permute(3, 1, 2, 0).contiguous().float(),
            "S1": torch.from_numpy(S1_q).permute(3, 1, 2, 0).contiguous().float(),
            "S2": torch.from_numpy(S2_q).permute(3, 1, 2, 0).contiguous().float(),
            # GT material maps, [C, H, W, 1]
            "nml":       normal.transpose(2, 0, 1)[..., None],
            "baseColor": basecolor_m.transpose(2, 0, 1)[..., None],
            "roughness": roughness_m.transpose(2, 0, 1)[..., None],
            "metallic":  metallic_m.transpose(2, 0, 1)[..., None],
            # Geometry
            "mask":     mask.transpose(2, 0, 1)[..., None],
            "position": position.transpose(2, 0, 1)[..., None],
            # Flags
            "numberOfImages": 9,
            "is_real": False,
        }

        if self.antialias_edges:
            sample = self._maybe_apply_edge_aa(
                sample, basecolor, roughness, metallic, mask[..., 0]
            )
        return sample
