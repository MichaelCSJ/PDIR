import numpy as np
import hashlib
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.module.utils import *
from src.model.utils import decompose_tensors
from src.model.utils import gauss_filter
from src.model.utils.utils import sobel_edge_map
from src.model.lora import apply_lora, LoRALinear
from src.model.renderer import Principled_BRDF

class Net(nn.Module):
    def __init__(self, pixel_samples, network_depth, pattern_type, add_noise,
                 real_loss_mode: str = "pattern",
                 pattern_color_strength: float = 1.0):
        super().__init__()
        self.pixel_samples = pixel_samples
        self.network_depth = network_depth # layer num attention block
        self.glc_smoothing = True
        # REAL_ADAPT loss target form:
        #   "pattern" (default): compare pattern-summed simulated render to
        #     pattern-summed Stokes-derived target (existing behaviour).
        #   "quad": compare per-quadrant grouped-OLAT render directly to per-
        #     quadrant raw Stokes targets (S0_q - S1_q for depol, S1_q for
        #     polar). Identity permutation [tl,tr,bl,br] = quad{0,1,2,3}
        #     verified by find_quad_permutation.py.
        if real_loss_mode not in ("pattern", "quad"):
            raise ValueError(f"real_loss_mode must be 'pattern' or 'quad', got {real_loss_mode!r}")
        self.real_loss_mode = real_loss_mode

        # RGB color cross-talk calibration for pattern simulation.
        # Naive RGBbin0 weighted sum overestimates per-pixel saturation by
        # ~22% relative to real camera captures (measured on 20260427_real +
        # 20260428_real, 324 scenes; sat_sim / sat_real = 1.216 at α=1.0,
        # MSE / chroma_dist both minimised at α≈0.80).
        # Applied as: x' = α·x + (1−α)·mean_RGB(x)  to s0_pat (intensity).
        # Default 1.0 leaves the simulation unchanged; set to 0.80 to match.
        if not (0.0 <= float(pattern_color_strength) <= 1.0):
            raise ValueError(
                f"pattern_color_strength must be in [0,1], got {pattern_color_strength!r}"
            )
        self.pattern_color_strength = float(pattern_color_strength)
        self.input_dim = 4 # RGB + mask   
        self.image_encoder = ScaleInvariantSpatialLightImageEncoder(self.input_dim, self.network_depth, use_efficient_attention=False) 
        self.input_dim = 0 # embedding
        self.glc_upsample = GLC_Upsample(256+self.input_dim, num_enc_sab=1, dim_hidden=256, dim_feedforward=1024, use_efficient_attention=True)
        self.glc_aggregation = GLC_Aggregation(256+self.input_dim, num_agg_transformer=2, dim_aggout=384, dim_feedforward=1024, use_efficient_attention=False)
        self.img_embedding = nn.Sequential(
            nn.Linear(3,32),
            nn.LeakyReLU(),
            nn.Linear(32, 256)
        )
        self.regressor = Regressor(384, num_enc_sab=1, use_efficient_attention=True, dim_feedforward=1024)
        self.criterionL2 = nn.MSELoss(reduction = 'mean')  
        self.Renderer = Principled_BRDF()
        
        self.add_noise = add_noise
        self.only_rgb = False
        
        self.pattern_target_hw = (2, 2)
        self.P = 1
        patterns = torch.zeros(1, self.P, 3, 2, 2)
        if pattern_type == 'RGBbin0':
            patterns[:, :, 0, :, :1] = 1  # Red
            patterns[:, :, 1, :, 1:] = 1  # Green
            patterns[:, :, 2, :1, :] = 1  # Blue
        elif pattern_type == 'RGBbin1':
            patterns[:, :, 0, :, 1:] = 1  # Red
            patterns[:, :, 1, :, :1] = 1  # Green
            patterns[:, :, 2, 1:, :] = 1  # Blue
        else:
            raise ValueError(f"Unsupported pattern_type: {pattern_type}")
        # patterns = torch.logit(patterns)
        # patterns = torch.sigmoid(patterns) ** 2.2
        # patterns = patterns.permute(0,1,2,4,3)
        patterns = torch.flip(patterns, dims=[-1])
        patterns = patterns.reshape(1, self.P, 3, 1, 1, -1)
        self.register_buffer("patterns", patterns, persistent=False)
        
        if self.only_rgb:
            frame_type_template = torch.tensor(
                [[0, 0], [0, 1], [0, 2]], dtype=torch.long
            )
        else:
            frame_type_template = torch.tensor(
                [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]], dtype=torch.long
            )
        self.register_buffer("frame_type_template", frame_type_template, persistent=False)
        
        frame_type_template_with_cop = torch.tensor(
            [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]],
            dtype=torch.long,
        )
        self.register_buffer("frame_type_template_with_cop", frame_type_template_with_cop, persistent=False)
        
        # Toggle for A/B memory benchmarking: explicitly delete no-longer-used tensors.
        self.delete_intermediate_tensors = True

    def apply_lora(self, target_modules, rank: int, alpha: float, dropout: float) -> int:
        replaced = 0
        for module_name in target_modules:
            if not hasattr(self, module_name):
                raise AttributeError(f"Net has no module named '{module_name}'")
            replaced += apply_lora(getattr(self, module_name), rank=rank, alpha=alpha, dropout=dropout)
        return replaced

    def freeze_base_for_lora(self) -> None:
        for name, param in self.named_parameters():
            if name.endswith("patterns"):
                param.requires_grad_(False)
            elif ".lora_" in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)

    def find_a(self, gain_db:float, device=None, dtype=torch.float32):
        """Return slope coefficient `a` for the noise variance model."""
        return torch.tensor(0.004391*gain_db+0.008185, device=device, dtype=dtype)
    
    def find_b(self, gain_db:float, device=None, dtype=torch.float32):
        """Return bias coefficient `b` for the noise variance model."""
        return torch.tensor(0.1523*gain_db-0.8041, device=device, dtype=dtype)
    
    def _desaturate_rgb(self, x: torch.Tensor, channel_dim: int) -> torch.Tensor:
        """Linear blend toward per-pixel channel mean to mirror real-camera
        color cross-talk: x' = α·x + (1−α)·mean_RGB(x).  α=1 is a no-op.
        See `pattern_color_strength` in __init__ for calibration source.
        """
        a = self.pattern_color_strength
        if a >= 1.0:
            return x
        mean = x.mean(dim=channel_dim, keepdim=True)
        return a * x + (1.0 - a) * mean

    def _group_olat_quadrants(self, x: torch.Tensor) -> torch.Tensor:
        R, Cg = 9, 16
        *prefix, RC, Ch = x.shape
        assert RC == R * Cg, f"_group_olat_quadrants: expected OLAT dim {R*Cg}, got {RC}"
        grid = x.reshape(*prefix, R, Cg, Ch)
        r_split, c_split = 5, 8
        tl = grid[..., :r_split, :c_split, :].sum(dim=(-3, -2))
        tr = grid[..., :r_split, c_split:, :].sum(dim=(-3, -2))
        bl = grid[..., r_split:, :c_split, :].sum(dim=(-3, -2))
        br = grid[..., r_split:, c_split:, :].sum(dim=(-3, -2))
        return torch.stack([tl, tr, bl, br], dim=-2)

    def simulate_pattern(self, OLAT: torch.Tensor, is_CoP: bool = False, ADD_NOISE: bool = True, gain_db: float=14.0, noise_scale: float=1.0) -> torch.Tensor:
        OLAT_weighted = OLAT.unsqueeze(1) * self.patterns # [B, P, 3, H, W, r*c]
        pattern_sim = OLAT_weighted.sum(dim=-1) # [B, P, 3, H, W]
        # Color cross-talk calibration (RGB intensity only; CoP is a signed
        # phase signal so the linear desaturation does not apply there).
        if not is_CoP:
            pattern_sim = self._desaturate_rgb(pattern_sim, channel_dim=2)
        if is_CoP:
            pattern_sim = pattern_sim.clamp(-1.0, 1.0)
            pattern_sim = ((pattern_sim + 1.0) / 2.0).clamp(0.0, 1.0)  # [B,P,3,H,W] from [-1,1] to [0,1]
        
        if ADD_NOISE:
            pattern_sim, pattern_noise, pattern_sigma = self.add_noise_model_gray_to_rgb(
                pattern_sim, gain_db=gain_db, noise_scale=noise_scale
            )
        else:
            pattern_sim = (pattern_sim / 1.0).clamp(0, 1)
            
        pattern_sim_q = self.quantize_8bit(pattern_sim, mode="round")
        pattern_sim = pattern_sim + (pattern_sim_q - pattern_sim).detach()
        
        return pattern_sim

    def simulate_pattern_from_stokes(
        self,
        S0: torch.Tensor,
        S1: torch.Tensor,
        S2: torch.Tensor,
        ADD_NOISE: bool = True,
        gain_db: float = 14.0,
        noise_scale: float = 1.0,
    ):
        """
        Stokes-domain pattern simulation. Each component is pattern-summed
        linearly, then the transformer-input modalities are derived in [0, 1]:

            polar = clip(s1_pat, 0, 1)
            depol = clip(s0_pat - s1_pat, 0, 1)
            cop   = clip( ((s2_pat / max(s0_pat, sqrt(s1_pat^2+s2_pat^2), 0)) + 1)/2,
                          0, 1 )

        Noise + 8-bit quantization are applied AFTER the [0, 1] derivation,
        because:
          - the noise model is calibrated for [0, 1] intensity-domain inputs
          - applying noise on signed S1/S2 isn't well-defined for the model
          - quantization on the final [0, 1] signal mirrors the 8-bit display
            / sensor pipeline that produced the observation in the first place.

        Args:
            S0, S1, S2: each [B, 3, H, W, T=4] (per-quadrant Stokes).

        Returns:
            depol_sim, polar_sim, cop_sim: each [B, P, 3, H, W] in [0, 1].
        """
        # Linear pattern weighted sum on each Stokes component.
        # OLAT[B,3,H,W,4] -> unsqueeze(1) -> [B,1,3,H,W,4]; patterns [1,P,3,1,1,4].
        s0_pat = (S0.unsqueeze(1) * self.patterns).sum(dim=-1)              # [B, P, 3, H, W]
        s1_pat = (S1.unsqueeze(1) * self.patterns).sum(dim=-1)
        s2_pat = (S2.unsqueeze(1) * self.patterns).sum(dim=-1)

        # RGB cross-talk calibration applied to ALL three Stokes components:
        # S0/S1/S2 are measured through the same Bayer-polarized mosaic + the
        # same display-pattern protocol, so they share the same per-channel
        # over-saturation bias. Applying `_desaturate_rgb` only to s0_pat (the
        # original behaviour) leaves polar (raw s1_pat) inconsistent with the
        # render-side spec_sim — which DOES go through `simulate_pattern` →
        # `_desaturate_rgb`. Applying to all three restores symmetry between
        # the encoder-input simulation and the render-loss prediction.
        s0_pat = self._desaturate_rgb(s0_pat, channel_dim=2)
        s1_pat = self._desaturate_rgb(s1_pat, channel_dim=2)
        s2_pat = self._desaturate_rgb(s2_pat, channel_dim=2)

        # Derived modalities in [0, 1].
        polar_sim = s1_pat.clamp(0.0, 1.0)
        depol_sim = (s0_pat - s1_pat).clamp(0.0, 1.0)

        pol_mag = torch.sqrt(s1_pat * s1_pat + s2_pat * s2_pat + 1e-12)
        s0_denom = torch.maximum(torch.maximum(s0_pat, pol_mag), torch.zeros_like(s0_pat))
        cop_signed = s2_pat / (s0_denom + 1e-8)                              # |cop_signed| <= 1
        cop_sim = ((cop_signed + 1.0) / 2.0).clamp(0.0, 1.0)

        # Noise on each [0, 1] modality.
        if ADD_NOISE:
            depol_sim, _, _ = self.add_noise_model_gray_to_rgb(depol_sim, gain_db=gain_db, noise_scale=noise_scale)
            polar_sim, _, _ = self.add_noise_model_gray_to_rgb(polar_sim, gain_db=gain_db, noise_scale=noise_scale)
            cop_sim,   _, _ = self.add_noise_model_gray_to_rgb(cop_sim,   gain_db=gain_db, noise_scale=noise_scale)

        # 8-bit quantization (straight-through gradient).
        for sim in (depol_sim, polar_sim, cop_sim):
            pass
        depol_q = self.quantize_8bit(depol_sim, mode="round")
        depol_sim = depol_sim + (depol_q - depol_sim).detach()
        polar_q = self.quantize_8bit(polar_sim, mode="round")
        polar_sim = polar_sim + (polar_q - polar_sim).detach()
        cop_q = self.quantize_8bit(cop_sim, mode="round")
        cop_sim = cop_sim + (cop_q - cop_sim).detach()

        return depol_sim, polar_sim, cop_sim

    def simulate_pattern_from_pattern_stokes(
        self,
        s0_pat: torch.Tensor,                                  # [B, 3, H, W]
        s1_pat: torch.Tensor,                                  # [B, 3, H, W]
        s2_pat: torch.Tensor,                                  # [B, 3, H, W]
        apply_desaturate: bool = False,
        apply_noise: bool = False,
        apply_quantize: bool = True,
        gain_db: float = 14.0,
        noise_scale: float = 1.0,
    ):
        """
        Modality derivation for inputs that are ALREADY pattern-summed Stokes.

        Use when each frame is captured ONCE under the full display pattern
        (e.g. 20260430_face): per-quadrant Stokes is unavailable and the
        captured signal is already at the post-pattern stage that
        `simulate_pattern_from_stokes` would otherwise synthesize.

        Compared to `simulate_pattern_from_stokes` this method:
          * skips `(S * patterns).sum(quad)` — no 4-quadrant input;
          * skips RGB cross-talk desaturation by default. The α=0.80 in
            `simulate_pattern_from_stokes` exists to compensate the ~25%
            over-saturation that the per-quadrant weighted sum introduces
            (measured: chroma(4-quad sum) ≈ 1.25·chroma(real full-pattern
            capture); see `scripts/analyze_pattern_color_saturation.py`).
            Net training-time chroma into the encoder = 1.25·real × 0.80
            ≈ real, i.e. the actual real-camera chroma. A direct full-
            pattern capture (this method's input) is already at that real
            level, so re-applying desaturation would over-correct. Enable
            only for diagnostic A/B testing.
          * skips noise injection by default — real captures are already
            noisy.
          * keeps 8-bit quantization — idempotent for true 8-bit content
            and matches the training-time signal.

        Args:
            s0_pat, s1_pat, s2_pat: [B, 3, H, W] real Stokes captures.

        Returns:
            depol_sim, polar_sim, cop_sim: each [B, P=1, 3, H, W] in [0, 1].
        """
        s0_pat = s0_pat.unsqueeze(1)                                          # [B, 1, 3, H, W]
        s1_pat = s1_pat.unsqueeze(1)
        s2_pat = s2_pat.unsqueeze(1)

        if apply_desaturate:
            # Mirror simulate_pattern_from_stokes: applied to all three Stokes
            # components for mosaic-protocol consistency.
            s0_pat = self._desaturate_rgb(s0_pat, channel_dim=2)
            s1_pat = self._desaturate_rgb(s1_pat, channel_dim=2)
            s2_pat = self._desaturate_rgb(s2_pat, channel_dim=2)

        polar_sim = s1_pat.clamp(0.0, 1.0)
        depol_sim = (s0_pat - s1_pat).clamp(0.0, 1.0)

        pol_mag = torch.sqrt(s1_pat * s1_pat + s2_pat * s2_pat + 1e-12)
        s0_denom = torch.maximum(torch.maximum(s0_pat, pol_mag), torch.zeros_like(s0_pat))
        cop_signed = s2_pat / (s0_denom + 1e-8)
        cop_sim = ((cop_signed + 1.0) / 2.0).clamp(0.0, 1.0)

        if apply_noise:
            depol_sim, _, _ = self.add_noise_model_gray_to_rgb(depol_sim, gain_db=gain_db, noise_scale=noise_scale)
            polar_sim, _, _ = self.add_noise_model_gray_to_rgb(polar_sim, gain_db=gain_db, noise_scale=noise_scale)
            cop_sim,   _, _ = self.add_noise_model_gray_to_rgb(cop_sim,   gain_db=gain_db, noise_scale=noise_scale)

        if apply_quantize:
            depol_q = self.quantize_8bit(depol_sim, mode="round")
            depol_sim = depol_sim + (depol_q - depol_sim).detach()
            polar_q = self.quantize_8bit(polar_sim, mode="round")
            polar_sim = polar_sim + (polar_q - polar_sim).detach()
            cop_q = self.quantize_8bit(cop_sim, mode="round")
            cop_sim = cop_sim + (cop_q - cop_sim).detach()

        return depol_sim, polar_sim, cop_sim

    def add_noise_model_gray_to_rgb(
        self,
        x_0_1: torch.Tensor,  # [B, P, C, ...], in [0, 1] float
        gain_db: float,
        noise_scale: float = 1.0,
        generator: torch.Generator | None = None,
    ):
        assert x_0_1.ndim >= 4 and x_0_1.size(2) == 3, f"expected [B,P,3,...], got {x_0_1.shape}"
        device = x_0_1.device
        dtype = x_0_1.dtype

        # Convert to 0..255 domain (the model was calibrated in 8-bit space).
        x255 = (x_0_1 * 255.0).clamp(0.0, 255.0).to(torch.float32)   # Keep variance computation stable in float32.

        a = self.find_a(gain_db, device=device, dtype=torch.float32)
        b = self.find_b(gain_db, device=device, dtype=torch.float32)

        # channel-wise variance: each RGB channel has its own noise statistics
        var = torch.clamp(a * x255 + b, min=0.0)
        sigma = torch.sqrt(var + 1e-8)

        # z ~ N(0, sigma^2)
        z = torch.randn(x255.shape, device=device, dtype=torch.float32, generator=generator) * sigma
        z = z * float(noise_scale)

        noisy255 = torch.clamp(x255 + z, 0.0, 255.0)

        # Convert back to 0..1 and restore original dtype.
        noisy = (noisy255 / 255.0).to(dtype)
        return noisy, (z / 255.0).to(dtype), sigma  # noisy, noise(0..1), sigma(0..255)
    
    def quantize_8bit(self, x_0_1: torch.Tensor, mode: str = "round") -> torch.Tensor:
        """
        x_0_1: float tensor in [0,1] (ideally)
        return: float tensor in {0,1/255,...,1} (same dtype as input)
        """
        x = torch.clamp(x_0_1, 0.0, 1.0)
        x255 = x * 255.0

        if mode == "round":
            q = torch.round(x255)
        elif mode == "floor":
            q = torch.floor(x255)
        elif mode == "stochastic":
            # unbiased stochastic rounding: floor(x) + Bernoulli(frac)
            f = torch.floor(x255)
            p = (x255 - f).clamp(0.0, 1.0)
            q = f + torch.bernoulli(p)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        return (q / 255.0).to(x_0_1.dtype)
    
    def rgb_to_three_channel_grayscale(self, rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        rgb: [B,P,3,H,W]
        return (R_gray3, G_gray3, B_gray3) each: [B,P,3,H,W]
        """
        r = rgb[:, :, 0:1, :, :]  # [B,P,1,H,W]
        g = rgb[:, :, 1:2, :, :]
        b = rgb[:, :, 2:3, :, :]
        r3 = r.expand(-1, -1, 3, -1, -1)
        g3 = g.expand(-1, -1, 3, -1, -1)
        b3 = b.expand(-1, -1, 3, -1, -1)
        return r3, g3, b3

    def forward(self, batch, decoder_resolution, canonical_resolution, fixed_sampling=False, REAL_ADAPT=False):
        """
        Args:
            batch: Batch dictionary from dataloader.
            decoder_resolution: Decoder-side supervision resolution.
            canonical_resolution: Encoder canonical resolution.
            fixed_sampling: If True, use deterministic object-wise pixel sampling.

        Returns:
            Dict[str, Tensor]: scalar losses and MSE metrics.
        """
        # Stokes-domain pipeline (the only supported path).
        M = batch["mask"][:, :, :, :, 0].to(torch.bfloat16)          # [B, 1, H, W]
        S0 = batch["S0"].to(torch.bfloat16)                          # [B, 3, H, W, 4]
        S1 = batch["S1"].to(torch.bfloat16)
        S2 = batch["S2"].to(torch.bfloat16)
        Pts = batch["position"].to(torch.bfloat16) if "position" in batch else None
        nImgArray = batch["numberOfImages"].to(device=S0.device).reshape(-1).long()  # [B]
        if not REAL_ADAPT:
            NM = batch["nml"][:, :, :, :, 0].to(torch.bfloat16)           # [B, 3, H, W, 1]
            BC = batch['baseColor'][:, :, :, :, 0].to(torch.bfloat16)    # [B, 3, H, W, 1]
            RG = batch['roughness'][:, :, :, :, 0].to(torch.bfloat16)    # [B, 1, H, W, 1]
            MT = batch['metallic'][:, :, :, :, 0].to(torch.bfloat16)     # [B, 1, H, W, 1]

        B, C, H, W, Nmax = S0.shape

        # Per-component pattern weighted sum -> derive depol/polar/cop in [0, 1].
        Depol_sim, Polar_sim, CoP_sim = self.simulate_pattern_from_stokes(
            S0, S1, S2, ADD_NOISE=self.add_noise
        )

        dep_r3, dep_g3, dep_b3 = self.rgb_to_three_channel_grayscale(Depol_sim)  # each [B,P,3,H,W]
        pol_r3, pol_g3, pol_b3 = self.rgb_to_three_channel_grayscale(Polar_sim)
        cop_r3, cop_g3, cop_b3 = self.rgb_to_three_channel_grayscale(CoP_sim)

        # 9 grayscale-RGB frames are fed to the transformer (3 modalities × 3 RGB).
        frame_type_ids = self.frame_type_template_with_cop.unsqueeze(0).expand(B, -1, -1).contiguous()
        I = torch.cat([dep_r3, dep_g3, dep_b3, pol_r3, pol_g3, pol_b3, cop_r3, cop_g3, cop_b3], dim=1)  # [B, 9P, 3, H, W]
        B, Nmax, C, H, W = I.shape

        # --- 2) Canonical encoding
        """ Image Encoder at Canonical Resolution """
        # img_index: [B*Nmax] bool mask, True for valid frames per sample.
        img_index = (
            torch.arange(Nmax, device=I.device).unsqueeze(0) < nImgArray.unsqueeze(1)
        ).reshape(-1)
        I_enc = I.reshape(-1, C, H, W)  # [B*Nmax, C, H, W]    
        M_enc = M.unsqueeze(1).expand(-1, Nmax, -1, -1, -1).reshape(-1, 1, H, W) # [B*Nmax, 1, H, W]
        enc_input = (I_enc * M_enc)[img_index]        # [B*N, 3, H, W]
        
        # frame_type_ids: [B, Nmax, 2] -> [B*Nmax, 2]
        frame_type_ids = frame_type_ids.reshape(-1, 2).to(device=enc_input.device, dtype=torch.long)
        frame_type_ids = frame_type_ids[img_index, :]  # [B*N, 2]
        glc,light_tokens = self.image_encoder(enc_input, nImgArray, canonical_resolution, frame_type_ids=frame_type_ids)
        # print('glc:', glc.shape)
        # --- 3) Prepare decoder-resolution tensors
        """ Sample Decoder at Original Resolution"""
        # img = I_enc[img_index]
        # I_dec = F.interpolate(img, size=(decoder_resolution, decoder_resolution), mode='bilinear', align_corners=False)
        I_dec = F.interpolate(I_enc, size=(decoder_resolution, decoder_resolution), mode='bilinear', align_corners=False)
        M_dec = F.interpolate(M,   size=(decoder_resolution, decoder_resolution), mode='nearest')
        
        # --- 4) GLC smoothing
        if self.glc_smoothing:
            f_scale = decoder_resolution//canonical_resolution # (Hd/Hc)
            smoothing = gauss_filter.gauss_filter(glc.shape[1], 10 * f_scale+1, 1).to(glc.device) # channels, kernel_size, sigma
            glc = smoothing(glc) #[B*f,256,128,128]        
        
        if REAL_ADAPT:
            if self.real_loss_mode == "quad":
                # Per-quadrant raw targets from Stokes: depol_q = clamp(S0-S1, 0),
                # polar_q = clamp(S1, 0). S0/S1 shape [B, 3, H, W, 4]. Interpolate
                # each quadrant to decoder resolution; final shape
                # [B, 3, dec_res, dec_res, 4].
                _bx, _cx, _hx, _wx, _tx = S0.shape                        # _tx == 4
                _diff_full  = torch.clamp(S0 - S1, min=0.0)               # [B,3,H,W,4]
                _polar_full = torch.clamp(S1,      min=0.0)
                _diff_4d  = _diff_full.permute(0, 4, 1, 2, 3).reshape(_bx * _tx, _cx, _hx, _wx)
                _polar_4d = _polar_full.permute(0, 4, 1, 2, 3).reshape(_bx * _tx, _cx, _hx, _wx)
                _diff_dec  = F.interpolate(_diff_4d,  size=(decoder_resolution, decoder_resolution),
                                            mode='bilinear', align_corners=False)
                _polar_dec = F.interpolate(_polar_4d, size=(decoder_resolution, decoder_resolution),
                                            mode='bilinear', align_corners=False)
                # -> [B, 4, 3, dec_res, dec_res]; keep this layout, sample later.
                Depol_target_dec = _diff_dec.reshape(_bx, _tx, _cx, decoder_resolution, decoder_resolution)
                Polar_target_dec = _polar_dec.reshape(_bx, _tx, _cx, decoder_resolution, decoder_resolution)
            else:
                Depol_target_dec = F.interpolate(
                    Depol_sim.squeeze(1),
                    size=(decoder_resolution, decoder_resolution),
                    mode='bilinear',
                    align_corners=False,
                )
                Polar_target_dec = F.interpolate(
                    Polar_sim.squeeze(1),
                    size=(decoder_resolution, decoder_resolution),
                    mode='bilinear',
                    align_corners=False,
                )
            NM_dec = BC_dec = RG_dec = MT_dec = None
            if self.delete_intermediate_tensors:
                del M, Depol_sim, Polar_sim, dep_r3, dep_g3, dep_b3, pol_r3, pol_g3, pol_b3
            
        else:
            NM_dec = F.interpolate(NM, size=(decoder_resolution, decoder_resolution), mode='bilinear', align_corners=False)
            BC_dec = F.interpolate(BC, size=(decoder_resolution, decoder_resolution), mode='bilinear', align_corners=False)
            RG_dec = F.interpolate(RG, size=(decoder_resolution, decoder_resolution), mode='bilinear', align_corners=False)
            MT_dec = F.interpolate(MT, size=(decoder_resolution, decoder_resolution), mode='bilinear', align_corners=False)
            if self.delete_intermediate_tensors:
                del NM, BC, RG, MT, M, Depol_sim, Polar_sim, dep_r3, dep_g3, dep_b3, pol_r3, pol_g3, pol_b3
                
        
        nm_true_list = [] 
        bc_true_list = [] 
        rg_true_list = [] 
        mt_true_list = [] 
        
        depol_true_list = []
        polar_true_list = []
        
        ids_len_list = []
        o_ids_list = []
        glc_ids_list = []
        p = 0
        for b in range(B):
            n_imgs_b = int(nImgArray[b].item())
            target = range(p, p + n_imgs_b)
            p += n_imgs_b
            
            # Sampling / mask: [B, 1, H, W]
            mask = M_dec[b, 0] > 0  # [H, W], bool tensor
            valid_indices = mask.nonzero(as_tuple=False)  # [num_valid, 2] (y, x)
            num_valid = int(valid_indices.shape[0])
            if num_valid == 0:
                raise RuntimeError(f"[FATAL] Empty valid mask for sample index {b}")
            if fixed_sampling:
                # Deterministic seed per object (independent of batch index).
                g = torch.Generator(device=mask.device)
                seed_val = 42 + b
                g.manual_seed(seed_val)
            else:
                g = None  # Random sampling.
            if num_valid >= self.pixel_samples:
                perm = torch.randperm(num_valid, generator=g, device=mask.device)
                selected = valid_indices[perm[:self.pixel_samples]]
            else:
                if fixed_sampling:
                    draw = torch.randint(
                        num_valid, (self.pixel_samples,), generator=g, device=mask.device
                    )
                else:
                    draw = torch.randint(num_valid, (self.pixel_samples,), device=mask.device)
                selected = valid_indices[draw]

            # (y, x) → linear index
            H, W = mask.shape
            ids = (selected[:, 0] * W + selected[:, 1]).long()

            o_ = I_dec[target].reshape(n_imgs_b, C, H * W).permute(2, 0, 1)  # [H*W, N, C]
            o_ids = o_[ids, :, :]
            glc_ids = glc[target].permute(2,3,0,1).flatten(0, 1)[ids, :, :]
            o_ids_list.append(o_ids)
            glc_ids_list.append(glc_ids)
            ids_len_list.append(int(ids.numel()))
            
            if REAL_ADAPT:
                if self.real_loss_mode == "quad":
                    # Depol_target_dec, Polar_target_dec are [B, 4, 3, H, W];
                    # gather ids -> [num_sampled, 4, 3].
                    _dq_b = Depol_target_dec[b].permute(2, 3, 0, 1).reshape(H * W, 4, 3)
                    _pq_b = Polar_target_dec[b].permute(2, 3, 0, 1).reshape(H * W, 4, 3)
                    depol_true_list.append(_dq_b[ids, :, :])              # [S, 4, 3]
                    polar_true_list.append(_pq_b[ids, :, :])
                else:
                    depol_true = Depol_target_dec[b, :, :, :].reshape(3, H * W).permute(1, 0)  # [H*W, 3]
                    polar_true = Polar_target_dec[b, :, :, :].reshape(3, H * W).permute(1, 0)  # [H*W, 3]
                    depol_true_list.append(depol_true[ids, :])            # [S, 3]
                    polar_true_list.append(polar_true[ids, :])
            else:
                # Safe normalize: F.normalize backward goes NaN at zero norm.
                # Replace with explicit divide-by-clamped-norm (eps=1e-4).
                _nm_in = NM_dec[b, :, :, :].reshape(3, H * W).permute(1, 0)
                nm_true = _nm_in / _nm_in.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-4)
                bc_true = BC_dec[b, :, :, :].reshape(3, H * W).permute(1,0)
                rg_true = RG_dec[b, :, :, :].reshape(1, H * W).permute(1,0)
                mt_true = MT_dec[b, :, :, :].reshape(1, H * W).permute(1,0)
                nm_true_list.append(nm_true[ids, :])
                bc_true_list.append(bc_true[ids, :])
                rg_true_list.append(rg_true[ids, :])
                mt_true_list.append(mt_true[ids, :])

        num_sample_set = ids_len_list[0]
        
        o_ids   = torch.cat(o_ids_list, dim=0)     # [B*S, N, 3], S=num_sample_set
        glc_ids = torch.cat(glc_ids_list, dim=0)   # [B*S, N, 256]
        
        o_ids = self.img_embedding(o_ids)          # [B*S, N, 256]
        x = o_ids + glc_ids
        glc_ids = self.glc_upsample(x)
        x = o_ids + glc_ids
        x = self.glc_aggregation(x)                # [B*S, 384]
        # print("x after aggregation:", x.shape)
        x_n, x_brdf = self.regressor(x, num_sample_set)  # [B, S, 3]

        # Safe normalize: avoid NaN backward at zero norm.
        x_n = x_n / x_n.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-4)
        x_b = torch.clip(x_brdf[0], -1.0, 1.0)*0.5+0.5  # torch.sigmoid(x_brdf[0])
        x_r = torch.clip(x_brdf[1], -1.0, 1.0)*0.5+0.5  # torch.sigmoid(x_brdf[1])
        x_m = torch.clip(x_brdf[2], -1.0, 1.0)*0.5+0.5  # torch.sigmoid(x_brdf[2])     


        if REAL_ADAPT:
            # Stacked targets — shape depends on real_loss_mode:
            #   "pattern": [B, S, 3]
            #   "quad":    [B, S, 4, 3]
            depol_true = torch.stack(depol_true_list, dim=0)
            polar_true = torch.stack(polar_true_list, dim=0)

            rendered_diff, rendered_spec = self.Renderer(x_b.reshape(-1, 3), x_n.reshape(-1, 3), x_r.reshape(-1, 1), x_m.reshape(-1, 1))
            rendered_diff = rendered_diff.reshape(B, num_sample_set, 144, 3)
            rendered_spec = rendered_spec.reshape(B, num_sample_set, 144, 3)

            diff_quad = self._group_olat_quadrants(rendered_diff)  # [B,S,4,3]
            spec_quad = self._group_olat_quadrants(rendered_spec)  # [B,S,4,3]

            if self.real_loss_mode == "quad":
                # Direct per-quadrant comparison. Identity perm
                # [tl,tr,bl,br] = quad{0..3} verified empirically.
                # Both tensors are [B, S, 4, 3]; criterionL2 averages over all dims.
                depol_loss = self.criterionL2(diff_quad, depol_true)
                polar_loss = self.criterionL2(spec_quad, polar_true)
            else:
                diff_quad_p = diff_quad.permute(0, 3, 1, 2).unsqueeze(-2)   # [B,3,S,1,4]
                spec_quad_p = spec_quad.permute(0, 3, 1, 2).unsqueeze(-2)
                diff_sim = self.simulate_pattern(diff_quad_p, ADD_NOISE=self.add_noise).squeeze(1).squeeze(-1).permute(0, 2, 1)  # [B,S,3]
                spec_sim = self.simulate_pattern(spec_quad_p, ADD_NOISE=self.add_noise).squeeze(1).squeeze(-1).permute(0, 2, 1)
                depol_loss = self.criterionL2(diff_sim, depol_true)
                polar_loss = self.criterionL2(spec_sim, polar_true)

            # Diagnostic: log NaN/Inf occurrence (does not modify training signal).
            if torch.isnan(depol_loss).any() or torch.isinf(depol_loss).any():
                print("depol_render_loss contains NaN/Inf values")
            if torch.isnan(polar_loss).any() or torch.isinf(polar_loss).any():
                print("polar_render_loss contains NaN/Inf values")

            # Sanitize BEFORE summing so backward never sees NaN/Inf.
            depol_loss = torch.nan_to_num(depol_loss, nan=0.0, posinf=0.0, neginf=0.0)
            polar_loss = torch.nan_to_num(polar_loss, nan=0.0, posinf=0.0, neginf=0.0)

            loss = depol_loss + polar_loss

            return {
                "depol_render_loss": depol_loss,
                "polar_render_loss": polar_loss,
                "loss": loss,
            }
        else:
            nm_true = torch.stack(nm_true_list, dim=0) # [B, S, 3]
            bc_true = torch.stack(bc_true_list, dim=0) # [B, S, 3]
            rg_true = torch.stack(rg_true_list, dim=0) # [B, S, 1]
            mt_true = torch.stack(mt_true_list, dim=0) # [B, S, 1]
            
            # print("x_n:", x_n.shape, "nm_true:", nm_true.shape)
            # print("x_b:", x_b.shape, "bc_true:", bc_true.shape)
            # print("x_r:", x_r.shape, "rg_true:", rg_true.shape)
            # print("x_m:", x_m.shape, "mt_true:", mt_true.shape)
            
            mse_n = self.criterionL2(x_n, nm_true)
            mse_b = self.criterionL2(x_b, bc_true)
            mse_r = self.criterionL2(x_r, rg_true)
            mse_m = self.criterionL2(x_m, mt_true)

            # Chromaticity loss: project (R,G,B) onto the unit sphere (L2 norm
            # per pixel) so the comparison is brightness-invariant, capturing
            # only the hue direction.  Mask out pixels whose GT channel mean
            # is below `chroma_threshold` because tiny brightness lets noise
            # dominate the normalization direction at near-black pixels.
            #
            # x_b, bc_true: [B, S, 3]
            chroma_threshold = 0.05
            gt_pixel_mean  = bc_true.mean(dim=-1, keepdim=True)             # [B,S,1] for mask only
            chroma_mask    = (gt_pixel_mean > chroma_threshold).to(bc_true.dtype)  # [B,S,1]
            gt_chroma      = bc_true / bc_true.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-4)
            pred_chroma    = x_b    / x_b.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-4)
            chroma_sq      = (pred_chroma - gt_chroma) ** 2                 # [B,S,3]
            mask3          = chroma_mask.expand_as(chroma_sq)
            mse_chroma     = (chroma_sq * mask3).sum() / mask3.sum().clamp(min=1.0)

            # Diagnostic: detect NaN occurrence (does not modify training signal).
            if torch.isnan(mse_n).any(): print("mse_n contains NaN values")
            if torch.isnan(mse_b).any(): print("mse_b contains NaN values")
            if torch.isnan(mse_r).any(): print("mse_r contains NaN values")
            if torch.isnan(mse_m).any(): print("mse_m contains NaN values")
            if torch.isnan(mse_chroma).any() or torch.isinf(mse_chroma).any():
                print("mse_chroma contains NaN/Inf values")

            # Sanitize BEFORE summing so backward never sees NaN.
            # nan_to_num is differentiable; replaces NaN/±Inf entries with 0.
            mse_n      = torch.nan_to_num(mse_n,      nan=0.0, posinf=0.0, neginf=0.0)
            mse_b      = torch.nan_to_num(mse_b,      nan=0.0, posinf=0.0, neginf=0.0)
            mse_r      = torch.nan_to_num(mse_r,      nan=0.0, posinf=0.0, neginf=0.0)
            mse_m      = torch.nan_to_num(mse_m,      nan=0.0, posinf=0.0, neginf=0.0)
            mse_chroma = torch.nan_to_num(mse_chroma, nan=0.0, posinf=0.0, neginf=0.0)

            loss = mse_n + mse_b + mse_r + mse_m + mse_chroma

            return {
                'mse_n':      mse_n,
                'mse_b':      mse_b,
                'mse_r':      mse_r,
                'mse_m':      mse_m,
                'mse_chroma': mse_chroma,
                'loss':       loss,
            }

    @torch.no_grad()
    def infer_full_image_tiled(
        self,
        S0: torch.Tensor,                       # [B, 3, H, W, 4] per-quadrant Stokes S0
        S1: torch.Tensor,                       # [B, 3, H, W, 4]
        S2: torch.Tensor,                       # [B, 3, H, W, 4]
        M: torch.Tensor | None = None,          # [B, 1, H, W, 1]
        N: torch.Tensor | None = None,          # [B, 3, H, W, 1]
        R: torch.Tensor | None = None,          # [B, 1, H, W, 1]
        PT: torch.Tensor | None = None,         # [B, 3, H, W, 1]
        pixel_samples: int = 2048,
        patch_size: int = 512,
        canonical_resolution: int = 256,
        dtype=torch.bfloat16,
    ):
        """
        Run full-image tiled inference and merge per-tile predictions.

        Args:
            S0, S1, S2: [B, 3, H, W, 4] per-quadrant Stokes inputs.
            M: [B, 1, H, W, 1] valid mask.
            N: [B, 3, H, W, 1] normal target tensor (used by some paths).
            R: [B, 1, H, W, 1] roughness target tensor (used by some paths).
            PT: [B, 3, H, W, 1] position tensor (used by uncertainty paths).
            pixel_samples: Max sampled pixels.
            patch_size: Spatial tile size.
            canonical_resolution: Encoder canonical resolution.
            dtype: Compute dtype for decoder/tile buffers.

        Returns:
            Dict[str, Tensor]:
                normal [B,3,H,W], albedo [B,3,H,W],
                roughness [B,1,H,W], metallic [B,1,H,W], confidence [B,1,H,W].
        """
        Depol_sim, Polar_sim, CoP_sim = self.simulate_pattern_from_stokes(
            S0, S1, S2, ADD_NOISE=self.add_noise
        )
        return self._run_tiled_inference_from_modalities(
            Depol_sim, Polar_sim, CoP_sim, M,
            pixel_samples=pixel_samples,
            patch_size=patch_size,
            canonical_resolution=canonical_resolution,
            dtype=dtype,
        )

    def infer_full_image_tiled_from_pattern_stokes(
        self,
        s0_pat: torch.Tensor,                   # [B, 3, H, W] post-pattern Stokes S0
        s1_pat: torch.Tensor,                   # [B, 3, H, W]
        s2_pat: torch.Tensor,                   # [B, 3, H, W]
        M: torch.Tensor | None = None,          # [B, 1, H, W, 1] or [B, 1, H, W]
        pixel_samples: int = 2048,
        patch_size: int = 512,
        canonical_resolution: int = 256,
        dtype=torch.bfloat16,
        apply_desaturate: bool = False,
        apply_noise: bool = False,
        apply_quantize: bool = True,
    ):
        """
        Tiled inference variant for ALREADY pattern-summed Stokes captures
        (e.g. 20260430_face: one capture per frame under the full display
        pattern). Shares the encoder/decoder/render stack with
        `infer_full_image_tiled`; only the input-format adaptation differs.

        Args:
            s0_pat, s1_pat, s2_pat: [B, 3, H, W] real Stokes captures.
            M: valid mask. [B, 1, H, W, 1] (canonical) or [B, 1, H, W] both ok.
            apply_desaturate, apply_noise, apply_quantize: forwarded to
                `simulate_pattern_from_pattern_stokes`. Defaults match the
                training-time encoder distribution for real captures
                (desat off, noise off, quantize on); see that method's
                docstring for the chroma accounting.

        Returns:
            Same dict shape as `infer_full_image_tiled`.
        """
        if M is not None and M.dim() == 4:
            M = M.unsqueeze(-1)                                               # [B, 1, H, W] -> [B, 1, H, W, 1]
        Depol_sim, Polar_sim, CoP_sim = self.simulate_pattern_from_pattern_stokes(
            s0_pat, s1_pat, s2_pat,
            apply_desaturate=apply_desaturate,
            apply_noise=apply_noise,
            apply_quantize=apply_quantize,
        )
        # Polar_sim = torch.zeros_like(Polar_sim)  # Ablate polarization modality for real-adaptation inference.
        return self._run_tiled_inference_from_modalities(
            Depol_sim, Polar_sim, CoP_sim, M,
            pixel_samples=pixel_samples,
            patch_size=patch_size,
            canonical_resolution=canonical_resolution,
            dtype=dtype,
        )

    def _run_tiled_inference_from_modalities(
        self,
        Depol_sim: torch.Tensor,                # [B, P, 3, H, W]
        Polar_sim: torch.Tensor,                # [B, P, 3, H, W]
        CoP_sim: torch.Tensor,                  # [B, P, 3, H, W]
        M: torch.Tensor | None,                 # [B, 1, H, W, 1]
        pixel_samples: int,
        patch_size: int,
        canonical_resolution: int,
        dtype,
    ):
        """Shared post-modality tiling + decoder + render path used by both
        `infer_full_image_tiled` and `infer_full_image_tiled_from_pattern_stokes`.
        """
        dep_r3, dep_g3, dep_b3 = self.rgb_to_three_channel_grayscale(Depol_sim)
        pol_r3, pol_g3, pol_b3 = self.rgb_to_three_channel_grayscale(Polar_sim)
        cop_r3, cop_g3, cop_b3 = self.rgb_to_three_channel_grayscale(CoP_sim)

        B = Depol_sim.shape[0]
        frame_type_ids = self.frame_type_template_with_cop.unsqueeze(0).expand(B, -1, -1).contiguous()
        I = torch.cat([dep_r3, dep_g3, dep_b3, pol_r3, pol_g3, pol_b3, cop_r3, cop_g3, cop_b3], dim=1)  # [B, 9P, 3, H, W]

        B, Nmax, C, H, W = I.shape
        device = I.device
        
        
        #######################################################
        # Tile-based inference on I: [B, 3, H, W, Nmax]
        # patches_I: [B, sliding_blocks, C, ph, pw, N]
        # patches_M: [B, sliding_blocks, 1, ph, pw]
        
        # --- 1) Tile decomposition
        patches_I = decompose_tensors.divide_tensor_spatial(
            I.reshape(-1, C, H, W),  # [B*N, C, H, W]
            block_size=patch_size, method='tile_stride'
        )
        patches_I = patches_I.reshape(B, Nmax, -1, C, patch_size, patch_size).permute(0, 2, 3, 4, 5, 1)
        
        patches_M = decompose_tensors.divide_tensor_spatial(
            M.reshape(-1, 1, H, W),  # [B*N, 1, H, W]
            block_size=patch_size, method='tile_stride'
        ) # [B, sliding_blocks, 1, ph, pw]
        
        patches_nml, patches_bc, patches_rg, patches_mt = [], [], [], []
        sliding_blocks = patches_I.shape[1]
        for k in range(sliding_blocks):
            I_blk = patches_I[:, k, :, :, :, :]     # [B, C, ph, pw, N]
            M_blk = patches_M[:, k, :, :, :]        # [B, 1, ph, pw]
            Bk, Ck, Hk, Wk, Nk = I_blk.shape
            dec_res = Hk
            nImgArray = torch.full((Bk,), Nk, device=device, dtype=torch.long)

            # --- 2) Canonical encoding
            img_index = (
                torch.arange(Nk, device=device).unsqueeze(0) < nImgArray.unsqueeze(1)
            ).reshape(-1)
            I_enc = I_blk.permute(0, 4, 1, 2, 3).reshape(-1, Ck, Hk, Wk)  # [B*N, C, Hk, Wk]
            M_enc = M_blk.unsqueeze(1).expand(-1, Nk, -1, -1, -1).reshape(-1, 1, Hk, Wk)
            enc_input = (I_enc * M_enc)[img_index]    # [valid, C, Hk, Wk]
        
            # frame_type_ids: [B, Nmax, 2] -> [B*Nmax, 2]
            frame_type_ids_flat = frame_type_ids.reshape(-1, 2).to(device=device, dtype=torch.long)
            frame_type_ids_selected = frame_type_ids_flat[img_index, :]  # [B*N, 2]
            glc, _ = self.image_encoder(enc_input, nImgArray, canonical_resolution, frame_type_ids=frame_type_ids_selected) # glc: [B*Nk, 256, Hc, Wc], light_tokens: token features
            
            # --- 3) Prepare decoder-resolution tensors
            # img = I_blk.permute(0, 4, 1, 2, 3).reshape(-1, Ck, Hk, Wk)  # [B*N, C, Hk, Wk]
            # I_dec = F.interpolate(img, size=(dec_res, dec_res), mode='bilinear', align_corners=False).to(device=device, dtype=dtype)
            I_dec = F.interpolate(I_enc, size=(dec_res, dec_res), mode='bilinear', align_corners=False).to(device=device, dtype=dtype)
            M_dec = F.interpolate(M_blk, size=(dec_res, dec_res), mode='nearest').to(device=device, dtype=dtype)

            _, _, Hk, Wk = I_dec.shape

            # --- 4) GLC smoothing
            f_scale = dec_res // canonical_resolution
            smoothing = gauss_filter.gauss_filter(glc.shape[1], 10 * f_scale + 1, 1).to(glc.device, dtype=glc.dtype) # (256, 21, 1)
            glc = smoothing(glc)

            # --- 5) Split inference over all valid mask pixels
            nout   = torch.zeros(Bk, Hk * Wk, 3, device=device, dtype=dtype)  # normal
            bout   = torch.zeros(Bk, Hk * Wk, 3, device=device, dtype=dtype)  # basecolor
            rout   = torch.zeros(Bk, Hk * Wk, 1, device=device, dtype=dtype)  # roughness
            mout   = torch.zeros(Bk, Hk * Wk, 1, device=device, dtype=dtype)  # metallic
            
            p = 0
            for b in range(Bk):
                n_imgs_b = int(nImgArray[b].item())
                target = range(p, p + n_imgs_b)
                p += n_imgs_b
                m_ = M_dec[b].reshape(-1, Hk * Wk).permute(1, 0)               # [Hk*Wk, 1]
                ids = torch.nonzero(m_ > 0, as_tuple=False)[:, 0].long()
                if ids.numel() == 0:
                    continue
                perm = torch.randperm(ids.numel(), device=ids.device)
                ids = ids[perm]
                ids_t = ids.to(device=device, dtype=torch.long)
                
                o_ = I_dec[target].reshape(n_imgs_b, Ck, Hk * Wk).permute(2, 0, 1)   # [Hk*Wk, N, C]
                o_ids_raw = o_[ids_t, :, :]
                glc_ids = glc[target].permute(2,3,0,1).flatten(0,1)[ids_t, :, :]    # [len, N, 256]

                o_ids = self.img_embedding(o_ids_raw.to(dtype))                       # [len,N,256]
                x = o_ids + glc_ids
                glc_ids = self.glc_upsample(x)
                x = o_ids + glc_ids
                x = self.glc_aggregation(x)                                         # [len, 384]

                # regressor: normal, ..., conf, brdf tuple(list)
                x_n, x_brdf = self.regressor(x, int(ids_t.numel()))
                
                # Safe normalize (eps=1e-4): no_grad inference path, but kept consistent.
                x_n = x_n / x_n.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-4)  # [len, 3], normalized
                x_b = torch.clip(x_brdf[0], -1.0, 1.0)*0.5+0.5             # [len, 3], [0,1]
                x_r = torch.clip(x_brdf[1], -1.0, 1.0)*0.5+0.5             # [len, 1], [0,1]
                x_m = torch.clip(x_brdf[2], -1.0, 1.0)*0.5+0.5             # [len, 1], [0,1]
                
                # scatter
                nout[b, ids_t, :]  = x_n.to(dtype)
                bout[b, ids_t, :]  = x_b.to(dtype)
                rout[b, ids_t, :]  = x_r.to(dtype)
                mout[b, ids_t, :]  = x_m.to(dtype)

            # Rearrange to dense maps: [B,3,H,W] / [B,1,H,W].
            nmmap  = nout.reshape(Bk, Hk, Wk, 3).permute(0, 3, 1, 2)
            bcmap  = bout.reshape(Bk, Hk, Wk, 3).permute(0, 3, 1, 2)
            rgmap  = rout.reshape(Bk, Hk, Wk, 1).permute(0, 3, 1, 2)
            mtmap  = mout.reshape(Bk, Hk, Wk, 1).permute(0, 3, 1, 2)

            patches_nml.append(nmmap)
            patches_bc.append(bcmap)
            patches_rg.append(rgmap)
            patches_mt.append(mtmap)

        # --- 6) Tile merge
        def _merge(patches):
            tiles = torch.stack(patches, dim=1)                         # [B, nb, C, ph, pw]
            return decompose_tensors.merge_tensor_spatial(
                tiles.permute(1,0,2,3,4), method='tile_stride'          # -> [B, C, H, W]
            )

        merged_nm = _merge(patches_nml)
        merged_bc = _merge(patches_bc)
        merged_rg = _merge(patches_rg)
        merged_mt = _merge(patches_mt)
        
        rendered_diff, rendered_spec = self.Renderer(       # [B,H,W,R*C,3]
            merged_bc.permute(0, 2, 3, 1).reshape(-1, 3),  # [B,H,W,3]
            merged_nm.permute(0, 2, 3, 1).reshape(-1, 3),  # [B,H,W,3]
            merged_rg.permute(0, 2, 3, 1).reshape(-1, 1),  # [B,H,W,1]
            merged_mt.permute(0, 2, 3, 1).reshape(-1, 1),  # [B,H,W,1]
        )

        diff_quad = self._group_olat_quadrants(rendered_diff).reshape(B,H,W,4,3)  # [B,H,W,4,3]
        spec_quad = self._group_olat_quadrants(rendered_spec).reshape(B,H,W,4,3)  # [B,H,W,4,3]
        
        diff_quad = diff_quad.permute(0,4,1,2,3)           # [B,3,H,W,4]
        spec_quad = spec_quad.permute(0,4,1,2,3)           # [B,3,H,W,4]
        
        diff_sim = self.simulate_pattern(diff_quad, ADD_NOISE=self.add_noise)  # [B,P,3,H,W]
        spec_sim = self.simulate_pattern(spec_quad, ADD_NOISE=self.add_noise)  # [B,P,3,H,W]
        
        diff_r3, diff_g3, diff_b3 = self.rgb_to_three_channel_grayscale(diff_sim)
        spec_r3, spec_g3, spec_b3 = self.rgb_to_three_channel_grayscale(spec_sim)
        rendered_input = torch.cat([diff_r3, diff_g3, diff_b3, spec_r3, spec_g3, spec_b3], dim=1).permute(0, 2, 3, 4, 1)
        self.rendered_input = rendered_input.detach().to(torch.float32).cpu().numpy()         # for visualization [B, 3, H, W, 6P]
        self.simulated_input = I.detach().permute(0,2,3,4,1).to(torch.float32).cpu().numpy()  # for visualization [B, 3, H, W, 6P/9P]

        return {
            "normal":    merged_nm,     # [B,3,H,W], unit
            "albedo":    merged_bc,    # [B,3,H,W], 0..1
            "roughness": merged_rg,    # [B,1,H,W], 0..1
            "metallic":  merged_mt,    # [B,1,H,W], 0..1
        }
