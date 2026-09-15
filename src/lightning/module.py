"""LightningModule for PDIR training.

Supports three dataset modes:

* ``synth`` — supervised training on rendered scenes with ground-truth PBR maps.
* ``real``  — render-loss adaptation on real captures (no ground truth).
* ``joint`` — both at once, which is the configuration used in the paper.

Joint mode runs the two domains as *sequential* forward+backward passes under
manual optimization, so peak memory is ``max(synth, real)`` rather than their
sum. Two details in that path are load-bearing and must not be simplified away:

1. ``manual_backward`` runs inside ``torch.autocast(enabled=False)``. Lightning
   wraps ``training_step`` in a bf16 autocast context and, under automatic
   optimization, exits it before calling backward. Under manual optimization
   that is the caller's job; leaving backward inside autocast makes some
   op-level backward kernels accumulate in bf16 and silently produce NaN.
2. When the real weight is 0 the real graph is never built. Multiplying a NaN
   loss by 0 still yields NaN, so the only safe way to disable the real branch
   is to skip its forward pass entirely.
"""

from typing import Any, Dict

import pytorch_lightning as pl
import torch
from torchmetrics import MeanMetric

_SYNTH_SUBLOSSES = ("mse_n", "mse_b", "mse_r", "mse_m", "mse_chroma")
_REAL_SUBLOSSES = ("depol_render_loss", "polar_render_loss")


class PDIRModule(pl.LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        canonical_resolution: int,
        dataset_type: str = "synth",
        learning_rate: float = 8e-5,
        weight_decay: float = 0.05,
        scheduler: str = "step",
        step_size: int = 10000,
        gamma: float = 0.8,
        max_steps: int = 100000,
        min_lr: float = 1e-6,
        synth_loss_weight: float = 1.0,
        real_loss_weight: float = 0.1,
        # Real-loss ramp: staircase, constant inside each block of
        # `real_warmup_stair_steps` steps, reaching 1.0 after
        # `real_warmup_stairs` blocks. The first block is exactly 0.0, which
        # keeps the real graph unbuilt while the model is still unstable.
        real_warmup_stair_steps: int = 2000,
        real_warmup_stairs: int = 5,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net"])

        self.net = net
        self.canonical_resolution = int(canonical_resolution)
        self.dataset_type = str(dataset_type).lower()
        if self.dataset_type not in ("synth", "real", "joint"):
            raise ValueError(f"dataset_type must be synth|real|joint, got {dataset_type!r}")
        self.is_joint = self.dataset_type == "joint"
        self.real_adapt = self.dataset_type == "real"

        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.scheduler = str(scheduler).lower()
        if self.scheduler not in ("step", "cosine", "none"):
            raise ValueError(f"scheduler must be step|cosine|none, got {scheduler!r}")
        self.step_size = int(step_size)
        self.gamma = float(gamma)
        self.max_steps_budget = int(max_steps)
        self.min_lr = float(min_lr)

        self.synth_loss_weight = float(synth_loss_weight)
        self.real_loss_weight = float(real_loss_weight)
        self.real_warmup_stair_steps = int(real_warmup_stair_steps)
        self.real_warmup_stairs = int(real_warmup_stairs)

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        # Joint mode needs sequential backward passes, which Lightning only
        # allows under manual optimization.
        if self.is_joint:
            self.automatic_optimization = False

    # ------------------------------------------------------------------ utils

    @staticmethod
    def _batch_size(batch: Dict[str, Any]) -> int:
        return int(batch["S0"].shape[0])

    def _forward(self, batch, real_adapt: bool, fixed_sampling: bool = False):
        return self.net(
            batch,
            batch["S0"].shape[2],           # decoder resolution == input H
            self.canonical_resolution,
            fixed_sampling,
            REAL_ADAPT=real_adapt,
        )

    def real_weight(self) -> float:
        """Current real-domain loss weight, including the warmup ramp."""
        if self.real_warmup_stairs <= 0 or self.real_warmup_stair_steps <= 0:
            return self.real_loss_weight
        passed = int(self.global_step) // self.real_warmup_stair_steps
        factor = min(1.0, passed / self.real_warmup_stairs)
        return self.real_loss_weight * factor

    def _log_all(self, prefix: str, scalars: Dict[str, Any], bs: int, on_step: bool) -> None:
        for name, value in scalars.items():
            self.log(f"{prefix}/{name}", value, on_step=on_step, on_epoch=True,
                     sync_dist=True, batch_size=bs)

    # ------------------------------------------------------------------ train

    def training_step(self, batch, batch_idx: int):
        if not self.is_joint:
            metrics = self._forward(batch, real_adapt=self.real_adapt)
            loss = metrics["loss"]
            bs = self._batch_size(batch)
            self.train_loss(loss.detach())
            self._log_all("train", {"loss": self.train_loss,
                                    "lr": self.optimizers().param_groups[0]["lr"]},
                          bs, on_step=True)
            self._log_all("train", {k: metrics[k].detach() for k in
                                    (_SYNTH_SUBLOSSES + _REAL_SUBLOSSES) if k in metrics},
                          bs, on_step=False)
            return loss

        opt = self.optimizers()
        opt.zero_grad()

        synth_batch, real_batch = batch["synth"], batch["real"]
        w_s = self.synth_loss_weight
        w_r = self.real_weight()

        # 1) synthetic pass; its graph is freed by the backward below.
        metrics_s = self._forward(synth_batch, real_adapt=False)
        with torch.autocast(device_type=self.device.type, enabled=False):
            self.manual_backward(w_s * metrics_s["loss"])
        synth_loss = metrics_s["loss"].detach()
        synth_subs = {k: metrics_s[k].detach() for k in _SYNTH_SUBLOSSES if k in metrics_s}
        del metrics_s

        # 2) real pass, skipped entirely while the ramp is still at zero.
        if w_r > 0.0:
            metrics_r = self._forward(real_batch, real_adapt=True)
            with torch.autocast(device_type=self.device.type, enabled=False):
                self.manual_backward(w_r * metrics_r["loss"])
            real_loss = metrics_r["loss"].detach()
            real_subs = {k: metrics_r[k].detach() for k in _REAL_SUBLOSSES if k in metrics_r}
            del metrics_r
        else:
            real_loss = torch.zeros((), device=synth_loss.device, dtype=synth_loss.dtype)
            real_subs = {}

        opt.step()
        sched = self.lr_schedulers()
        if sched is not None:
            sched.step()

        bs = self._batch_size(synth_batch)
        combined = w_s * synth_loss + w_r * real_loss
        self.train_loss(combined)
        self._log_all("train", {"loss": self.train_loss,
                                "synth_loss": synth_loss,
                                "real_loss": real_loss,
                                "w_real": w_r,
                                "lr": opt.param_groups[0]["lr"]}, bs, on_step=True)
        self._log_all("train", {f"synth_{k}": v for k, v in synth_subs.items()}, bs, on_step=False)
        self._log_all("train", {f"real_{k}": v for k, v in real_subs.items()}, bs, on_step=False)
        return None

    # -------------------------------------------------------------- validate

    def validation_step(self, batch, batch_idx: int):
        if not self.is_joint:
            metrics = self._forward(batch, real_adapt=self.real_adapt, fixed_sampling=True)
            loss = metrics["loss"]
            bs = self._batch_size(batch)
            self.val_loss(loss.detach())
            self._log_all("val", {"loss": self.val_loss}, bs, on_step=False)
            self._log_all("val", {k: metrics[k] for k in
                                  (_SYNTH_SUBLOSSES + _REAL_SUBLOSSES) if k in metrics},
                          bs, on_step=False)
            # ModelCheckpoint monitors this name.
            self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=bs)
            return loss

        synth_batch, real_batch = batch["synth"], batch["real"]
        metrics_s = self._forward(synth_batch, real_adapt=False, fixed_sampling=True)
        metrics_r = self._forward(real_batch, real_adapt=True, fixed_sampling=True)
        w_r = self.real_weight()
        loss = self.synth_loss_weight * metrics_s["loss"] + w_r * metrics_r["loss"]

        bs = self._batch_size(synth_batch)
        self.val_loss(loss.detach())
        self._log_all("val", {"loss": self.val_loss,
                              "synth_loss": metrics_s["loss"],
                              "real_loss": metrics_r["loss"]}, bs, on_step=False)
        self._log_all("val", {f"synth_{k}": metrics_s[k] for k in _SYNTH_SUBLOSSES
                              if k in metrics_s}, bs, on_step=False)
        self._log_all("val", {f"real_{k}": metrics_r[k] for k in _REAL_SUBLOSSES
                              if k in metrics_r}, bs, on_step=False)
        # Checkpoint on the supervised term: the real render loss is not
        # comparable across weights while the ramp is still moving.
        self.log("val_loss", metrics_s["loss"], on_step=False, on_epoch=True,
                 sync_dist=True, batch_size=bs)
        return loss

    # ------------------------------------------------------------- optimizer

    def configure_optimizers(self) -> Dict[str, Any]:
        params = [p for p in self.net.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError("No trainable parameters found.")
        optimizer = torch.optim.AdamW(params, lr=self.learning_rate,
                                      weight_decay=self.weight_decay)
        if self.scheduler == "none":
            return {"optimizer": optimizer}

        if self.scheduler == "step":
            sched = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=self.step_size, gamma=self.gamma)
        else:
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.max_steps_budget, eta_min=self.min_lr)

        # Under manual optimization (joint mode) Lightning does not step
        # schedulers; `training_step` does it instead.
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": sched, "interval": "step", "frequency": 1},
        }
