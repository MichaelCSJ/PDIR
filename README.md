<div align="center">

# Snapshot Polarimetric Display Inverse Rendering

**ACM Transactions on Graphics 45(6) — Proc. SIGGRAPH Asia 2026**

[Seokjun Choi](https://michaelcsj.github.io/)\*,
[Yunseong Moon](https://yunseong0518.github.io/)\*,
[Kaizhang Kang](https://cocoakang.cn/),
[Hoon-Gyu Chung](https://sites.google.com/view/hoongyu-chung/home),
[Jin-Nyeong Kim](https://divisonofficer.github.io/),
[Giljoo Nam](https://giljoonam.github.io/),
[Seung-Hwan Baek](https://www.shbaek.com/)

<sup>\*Equal contribution</sup>

[Project page](https://michaelcsj.github.io/PDIR/) ·
[arXiv](https://arxiv.org/abs/2605.24915) ·
[Paper](https://arxiv.org/pdf/2605.24915)

</div>

---

An LCD displays a linearly polarized RGB binary pattern; an RGB polarization
camera with a quarter-wave plate in front of the lens records the scene in a
single exposure. That one capture is decomposed into nine measurements — three
spectral channels × three polarization states (unpolarized, linearly polarized,
circularly polarized) — and a feed-forward transformer turns them into per-pixel
**normal, albedo, roughness and metallicity**.

This repository holds the training and inference code. The project page lives on
the [`project_page`](https://github.com/MichaelCSJ/PDIR/tree/project_page) branch.

## Install

```bash
git clone https://github.com/MichaelCSJ/PDIR.git
cd PDIR
conda create -n pdir python=3.10 -y && conda activate pdir
pip install -r requirements.txt
```

The code was developed against PyTorch 2.4 / CUDA 12.4 on NVIDIA A6000 GPUs.

## Data layout

Two loaders are provided. Both hand the network raw per-quadrant Stokes
components; the pattern weighting and the unpolarized/LP/CP decomposition happen
inside the model, so the same checkpoint reads synthetic and real captures.

**Synthetic** (`--dataset synth`), one directory per scene:

```
<synth-root>/
  0000/
    obs_original.npz    diffuse_{1_0,0_0,1_1,0_1}, specular_{...}, cop_{...}   each [H, W, 3]
    gt_maps.npz         basecolor [H,W,3], roughness [H,W,1], metallic [H,W,1],
                        normal [H,W,3], mask [H,W]
  0001/
  ...
```

**Real captures** (`--dataset real`), one directory per scene:

```
<real-root>/
  scene0_.../
    quad{0,1,2,3}_main_hdr_s0.npy    [H, W, 3]  total intensity, >= 0
    quad{0,1,2,3}_main_hdr_s1.npy    [H, W, 3]  linear polarization, signed
    quad{0,1,2,3}_main_hdr_s2.npy    [H, W, 3]  circular polarization, signed
    mask.png                         uint8, 0/255
  scene1_.../
  ...
```

Without split files, every `--val-stride`-th scene goes to validation. Pass
`--synth-split-train` / `--synth-split-val` (and the `--real-` equivalents) to
control the split explicitly; each line is one scene directory name.

## Training

Supervised training on synthetic scenes:

```bash
python train.py --dataset synth \
    --synth-root /path/to/synth_scenes \
    --session pdir_synth --devices 4
```

Joint synthetic + real co-training, which is what the paper uses. The real
branch contributes a render loss only — it needs no ground-truth maps:

```bash
python train.py --dataset joint \
    --synth-root /path/to/synth_scenes \
    --real-root  /path/to/real_captures \
    --session pdir_joint --devices 4
```

Training writes to `outputs/<timestamp>_<session>/`: `checkpoints/` (top-3 by
validation loss, plus `last.ckpt`) and `logs/` (CSV scalars).

Things worth knowing before a long run:

- `--image-size` must be exactly twice `--canonical-resolution`. Defaults are
  384 / 192.
- Unless you pass `--no-pretrained`, training initialises from the LINO-PBR
  weights, downloaded once into `checkpoint/`. Use `--init-ckpt` to start from
  your own checkpoint instead.
- The real loss ramps in as a staircase: zero for the first
  `--real-warmup-stair-steps` steps, then one step up per block until it reaches
  full weight after `--real-warmup-stairs` blocks. While the weight is zero the
  real forward pass is skipped entirely, which keeps early training stable.
- Joint mode runs under manual optimization, so `--accumulate-grad-batches` must
  stay at 1.
- Multi-GPU uses DDP with bf16 mixed precision.

`python train.py --help` lists everything else.

## Inference

```bash
python inference.py --ckpt outputs/<run>/checkpoints/last.ckpt \
    --dataset real --data-root /path/to/real_captures \
    --out-dir results
```

For every scene this writes `<out-dir>/<scene>/`:

| File | Contents |
|---|---|
| `normal.png`, `albedo.png`, `roughness.png`, `metallic.png` | 8-bit previews |
| `maps.npz` | float32 arrays: `normal` (HW3, in [-1,1]), `albedo` (HW3), `roughness` (HW1), `metallic` (HW1), `mask` (HW) |

Use `--dataset synth` to run the same checkpoint over synthetic scenes.
Full-resolution images are processed in tiles of `--patch-size` pixels.

Two flags must match the values the checkpoint was trained with, or the encoder
sees a different input distribution than it learned on:

| Flag | Meaning |
|---|---|
| `--exposure-norm` | per-scene HDR exposure normalisation for real captures |
| `--pattern-color-strength` | RGB cross-talk calibration; 1.0 disables it |

## Relighting

The estimated maps are directly renderable. `render_relight.py` orbits a point
light around a scene and renders one frame per step through the same principled
BRDF the training render loss uses:

```bash
python render_relight.py --maps-dir results --out-dir relight
```

That writes `relight/<scene>/frame####.png` and `relight/<scene>.mp4` for every
scene under `--maps-dir`. Useful options:

| Flag | Default | Meaning |
|---|---|---|
| `--scenes` | all | render only the named scenes |
| `--n-frames` / `--fps` | 60 / 20 | orbit steps and video frame rate |
| `--radius` / `--center` | 0.4 / `0 -0.29 0` | orbit geometry, in the display calibration's units |
| `--brightness` | 50 | exposure applied to the HDR render before clipping |
| `--no-video` | off | write frames only |

`--brightness` is a single global exposure, so scenes with dark albedo or high
metallicity come out dimmer than others; raise it per scene if a render looks
too dark. H.264 video needs `imageio-ffmpeg` (see `requirements.txt`); without
it the script falls back to OpenCV's MPEG-4 encoder, which most browsers cannot
play.

## Repository layout

```
train.py                    training entrypoint
inference.py                inference entrypoint
render_relight.py           rotating-light relighting of the estimated maps
calibration/                calibrated LCD emitter positions used by the renderer
src/io/                     synthetic and real dataset loaders
src/lightning/module.py     LightningModule: losses, joint co-training, optimizer
src/model/net_train_module.py   the network: pattern simulation, encoder, heads, losses
src/model/module/utils.py       image encoder, GLC aggregation, regressor
src/model/aggregator/           DINOv2 backbone and the alternating-attention aggregator
src/model/renderer/             principled BRDF used by the render loss
```

## Citation

```bibtex
@article{choi2026snapshot,
  title     = {Snapshot Polarimetric Display Inverse Rendering},
  author    = {Choi, Seokjun and Moon, Yunseong and Kang, Kaizhang and Chung, Hoon-Gyu
               and Kim, Jin-Nyeong and Nam, Giljoo and Baek, Seung-Hwan},
  journal   = {ACM Transactions on Graphics},
  volume    = {45},
  number    = {6},
  articleno = {201},
  year      = {2026},
  doi       = {10.1145/3842531}
}
```

## Acknowledgements

The transformer backbone builds on [DINOv2](https://github.com/facebookresearch/dinov2)
and [LINO-UniPS](https://github.com/houyuanchen111/LINO_UniPS).
