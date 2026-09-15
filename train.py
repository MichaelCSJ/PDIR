#!/usr/bin/env python
"""Train PDIR.

Examples
--------
Supervised training on synthetic scenes::

    python train.py --dataset synth \
        --synth-root /path/to/synth_dataset \
        --session pdir_synth --devices 4

Joint synthetic + real co-training (the configuration used in the paper)::

    python train.py --dataset joint \
        --synth-root /path/to/synth_dataset \
        --real-root  /path/to/real_captures \
        --session pdir_joint --devices 4

Run ``python train.py --help`` for the full list of options.
"""

import argparse
import datetime as dt
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from torch.utils.data import DataLoader

from src.io.dataset_real import RealDataset
from src.io.dataset_synth import SynthDataset
from src.lightning.module import PDIRModule
from src.model.net_train_module import Net

# Pretrained LINO-PBR weights used to initialise the encoder/aggregator.
LINO_PBR_URL = "https://huggingface.co/houyuanchen/lino/resolve/main/lino_pbr.pth"


# --------------------------------------------------------------------- args


def parse_args():
    p = argparse.ArgumentParser(description="Train PDIR", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    d = p.add_argument_group("data")
    d.add_argument("--dataset", choices=["synth", "real", "joint"], default="synth",
                   help="synth: supervised. real: render-loss adaptation. joint: both.")
    d.add_argument("--synth-root", type=str, default=None,
                   help="Root of the synthetic scenes (required for synth/joint).")
    d.add_argument("--real-root", type=str, default=None,
                   help="Root of the real captures (required for real/joint).")
    d.add_argument("--synth-split-train", type=str, default=None,
                   help="Optional train split file; one scene name per line.")
    d.add_argument("--synth-split-val", type=str, default=None)
    d.add_argument("--real-split-train", type=str, default=None)
    d.add_argument("--real-split-val", type=str, default=None)
    d.add_argument("--val-stride", type=int, default=10,
                   help="Without split files, every Nth scene goes to validation.")
    d.add_argument("--exposure-norm", default=None,
                   choices=["p95_0.95", "p99_0.95", "mean_0.4", "median_0.4",
                            "max_0.99", "reinhard_0.18"],
                   help="Per-scene HDR exposure normalisation for real captures. "
                        "Inference must use the same value.")
    d.add_argument("--exposure-norm-clamp", type=float, nargs=2, default=(0.5, 4.0),
                   metavar=("LO", "HI"))
    d.add_argument("--image-size", type=int, default=384)
    d.add_argument("--num-workers", type=int, default=2)
    d.add_argument("--batch-size", type=int, default=1)

    m = p.add_argument_group("model")
    m.add_argument("--canonical-resolution", type=int, default=192,
                   help="Encoder resolution; must be image-size / 2.")
    m.add_argument("--network-depth", type=int, default=4)
    m.add_argument("--pixel-samples", type=int, default=2048)
    m.add_argument("--pattern-type", choices=["RGBbin0", "RGBbin1"], default="RGBbin0")
    m.add_argument("--add-noise", action="store_true", default=True,
                   help="Apply the calibrated sensor noise model to the simulated input.")
    m.add_argument("--no-add-noise", dest="add_noise", action="store_false")
    m.add_argument("--pattern-color-strength", type=float, default=0.80,
                   help="RGB cross-talk calibration; 1.0 disables it.")
    m.add_argument("--init-ckpt", type=str, default=None,
                   help="Checkpoint to initialise from. Defaults to the pretrained LINO-PBR weights.")
    m.add_argument("--no-pretrained", action="store_true",
                   help="Start from random initialisation.")

    o = p.add_argument_group("optimisation")
    o.add_argument("--lr", type=float, default=8e-5)
    o.add_argument("--weight-decay", type=float, default=0.05)
    o.add_argument("--max-steps", type=int, default=100000)
    o.add_argument("--scheduler", choices=["step", "cosine", "none"], default="step")
    o.add_argument("--step-size", type=int, default=10000)
    o.add_argument("--gamma", type=float, default=0.8)
    o.add_argument("--min-lr", type=float, default=1e-6)
    o.add_argument("--accumulate-grad-batches", type=int, default=1)
    o.add_argument("--synth-loss-weight", type=float, default=1.0)
    o.add_argument("--real-loss-weight", type=float, default=0.1)
    o.add_argument("--real-warmup-stair-steps", type=int, default=2000)
    o.add_argument("--real-warmup-stairs", type=int, default=5)

    r = p.add_argument_group("run")
    r.add_argument("--session", type=str, default="pdir")
    r.add_argument("--out-dir", type=str, default="outputs")
    r.add_argument("--devices", type=int, default=1)
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--val-check-interval", type=int, default=2000)
    r.add_argument("--log-every-n-steps", type=int, default=50)
    r.add_argument("--resume", type=str, default=None, help="Lightning checkpoint to resume from.")

    args = p.parse_args()

    if args.image_size != 2 * args.canonical_resolution:
        p.error("--image-size must be exactly 2 * --canonical-resolution")
    if args.dataset in ("synth", "joint") and not args.synth_root:
        p.error(f"--synth-root is required for --dataset {args.dataset}")
    if args.dataset in ("real", "joint") and not args.real_root:
        p.error(f"--real-root is required for --dataset {args.dataset}")
    if args.dataset == "joint" and args.accumulate_grad_batches != 1:
        # Joint mode runs under manual optimization, where Lightning refuses to
        # accumulate gradients on your behalf.
        p.error("--accumulate-grad-batches must be 1 for --dataset joint")
    return args


# ------------------------------------------------------------- checkpoints


def load_state_dict(path_or_url: str, cache_dir: Path) -> dict:
    """Read a state dict from a local file or download it once into `cache_dir`."""
    if path_or_url.startswith(("http://", "https://")):
        cache_dir.mkdir(parents=True, exist_ok=True)
        dest = cache_dir / Path(path_or_url).name
        if not dest.is_file():
            print(f"[ckpt] downloading {path_or_url}")
            torch.hub.download_url_to_file(path_or_url, str(dest), progress=True)
        path = dest
    else:
        path = Path(path_or_url)

    obj = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and isinstance(obj.get("state_dict"), dict):
        obj = obj["state_dict"]
    return obj


def adapt_state_dict(state_dict: dict) -> dict:
    """Strip the Lightning `net.` prefix and map legacy aggregator block names."""
    inner = {k[len("net."):]: v for k, v in state_dict.items() if k.startswith("net.")}
    sd = inner if inner else dict(state_dict)

    renames = {"light_axis_1_blocks": "polar_blocks", "light_axis_2_blocks": "rgb_blocks"}
    for old, new in renames.items():
        for k, v in list(sd.items()):
            if old in k:
                sd.setdefault(k.replace(old, new), v)

    # Defined in older checkpoints but unused by the current network.
    for dead in ("image_encoder.backbone.aggregator.polar_embed.weight",
                 "image_encoder.backbone.aggregator.rgb_embed.weight"):
        sd.pop(dead, None)
    return sd


# ------------------------------------------------------------------- data


def build_loader(dataset, *, args, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=shuffle,
        persistent_workers=args.num_workers > 0,
    )


def build_datasets(args):
    """Return (train_loader, val_loader), each possibly a CombinedLoader."""
    synth_train = synth_val = real_train = real_val = None

    if args.dataset in ("synth", "joint"):
        common = dict(data_root=args.synth_root, image_size=args.image_size,
                      val_stride=args.val_stride)
        synth_train = SynthDataset(mode="Train", split_file=args.synth_split_train, **common)
        synth_val = SynthDataset(mode="Val", split_file=args.synth_split_val, **common)
        print(f"[data] synth train={len(synth_train)} val={len(synth_val)}")

    if args.dataset in ("real", "joint"):
        common = dict(data_root=args.real_root, image_size=args.image_size,
                      val_stride=args.val_stride,
                      exposure_norm=args.exposure_norm,
                      exposure_norm_clamp=tuple(args.exposure_norm_clamp))
        real_train = RealDataset(mode="Train", split_file=args.real_split_train, **common)
        real_val = RealDataset(mode="Val", split_file=args.real_split_val, **common)
        print(f"[data] real  train={len(real_train)} val={len(real_val)}")

    if args.dataset == "synth":
        return (build_loader(synth_train, args=args, shuffle=True),
                build_loader(synth_val, args=args, shuffle=False))
    if args.dataset == "real":
        return (build_loader(real_train, args=args, shuffle=True),
                build_loader(real_val, args=args, shuffle=False))

    # joint: cycle the shorter loader so every step sees both domains.
    train = CombinedLoader({"synth": build_loader(synth_train, args=args, shuffle=True),
                            "real": build_loader(real_train, args=args, shuffle=True)},
                           mode="max_size_cycle")
    val = CombinedLoader({"synth": build_loader(synth_val, args=args, shuffle=False),
                          "real": build_loader(real_val, args=args, shuffle=False)},
                         mode="max_size_cycle")
    return train, val


# ------------------------------------------------------------------- main


def main():
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")

    run_dir = Path(args.out_dir) / f"{dt.datetime.now():%Y%m%d-%H%M%S}_{args.session}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] {run_dir}")

    net = Net(
        pixel_samples=args.pixel_samples,
        network_depth=args.network_depth,
        pattern_type=args.pattern_type,
        add_noise=args.add_noise,
        pattern_color_strength=args.pattern_color_strength,
    )

    if not args.no_pretrained:
        source = args.init_ckpt or LINO_PBR_URL
        sd = adapt_state_dict(load_state_dict(source, Path("checkpoint")))
        missing, unexpected = net.load_state_dict(sd, strict=False)
        print(f"[ckpt] init from {source}: missing={len(missing)} unexpected={len(unexpected)}")

    module = PDIRModule(
        net=net,
        canonical_resolution=args.canonical_resolution,
        dataset_type=args.dataset,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        scheduler=args.scheduler,
        step_size=args.step_size,
        gamma=args.gamma,
        max_steps=args.max_steps,
        min_lr=args.min_lr,
        synth_loss_weight=args.synth_loss_weight,
        real_loss_weight=args.real_loss_weight,
        real_warmup_stair_steps=args.real_warmup_stair_steps,
        real_warmup_stairs=args.real_warmup_stairs,
    )

    train_loader, val_loader = build_datasets(args)

    trainer = pl.Trainer(
        default_root_dir=run_dir,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.devices,
        strategy=DDPStrategy(find_unused_parameters=False) if args.devices > 1 else "auto",
        precision="bf16-mixed",
        max_steps=args.max_steps,
        max_epochs=-1,
        # Validate on a global-step cadence. Without check_val_every_n_epoch=None
        # Lightning reads val_check_interval as a within-epoch count and refuses
        # any value larger than one epoch of batches.
        val_check_interval=args.val_check_interval,
        check_val_every_n_epoch=None,
        log_every_n_steps=args.log_every_n_steps,
        accumulate_grad_batches=args.accumulate_grad_batches,
        logger=CSVLogger(save_dir=run_dir, name="logs"),
        callbacks=[
            ModelCheckpoint(dirpath=run_dir / "checkpoints", monitor="val_loss", mode="min",
                            save_top_k=3, save_last=True,
                            filename="step={step}-val_loss={val_loss:.4f}", auto_insert_metric_name=False),
            LearningRateMonitor(logging_interval="step"),
        ],
    )

    trainer.fit(module, train_loader, val_loader, ckpt_path=args.resume)


if __name__ == "__main__":
    main()
