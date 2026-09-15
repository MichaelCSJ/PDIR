





import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List, Dict, Any

from einops import rearrange
import cv2  
import torchvision
from ..layers import PatchEmbed
from ..layers.block import Block
from ..layers.rope import RotaryPositionEmbedding2D, PositionGetter
from ..layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2



cv2.setNumThreads(1)
logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]

class Aggregator(nn.Module):
    """
    The Aggregator applies alternating-attention over input frames,
    as described in VGGT: Visual Geometry Grounded Transformer.


    Args:
        img_size (int): Image size in pixels.
        patch_size (int): Size of each patch for PatchEmbed.
        embed_dim (int): Dimension of the token embeddings.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
        num_register_tokens (int): Number of register tokens.
        block_fn (nn.Module): The block type used for attention (Block by default).
        qkv_bias (bool): Whether to include bias in QKV projections.
        proj_bias (bool): Whether to include bias in the output projection.
        ffn_bias (bool): Whether to include bias in MLP layers.
        patch_embed (str): Type of patch embed. e.g., "conv" or "dinov2_vitl14_reg".
        aa_order (list[str]): The order of alternating attention, e.g. ["frame", "global"].
        aa_block_size (int): How many blocks to group under each attention type before switching. If not necessary, set to 1.
        qk_norm (bool): Whether to apply QK normalization.
        rope_freq (int): Base frequency for rotary embedding. -1 to disable.
        init_values (float): Init scale for layer scale.
    """

    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=4,
        num_heads=6,
        mlp_ratio=4.0,
        num_register_tokens=4,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        patch_embed="dinov2_vitl14_reg",
        aa_order=["frame", "polar","global","rgb"],
        # aa_order=["frame", "light_axis_1","global","light_axis_2"],
        aa_block_size=1,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
    ):
        super().__init__()

        self.__build_patch_embed__(patch_embed, img_size, patch_size, num_register_tokens, embed_dim=embed_dim)

        # polar_embed / rgb_embed: defined + initialized historically but never used in
        # forward (the addition `tokens = tokens + e` at line 266 is commented out).
        # Removing them to (a) eliminate unused parameters that force DDP to run with
        # `find_unused_parameters=True`, and (b) avoid an unused-grad path that
        # interacts badly with bf16 + dual-backward in joint co-training.
        # self.polar_embed = nn.Embedding(3, embed_dim)
        # self.rgb_embed = nn.Embedding(3, embed_dim)
        
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None
        self.patch_start_idx = 7 
        self.frame_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.global_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )
        self.polar_blocks = nn.ModuleList(
        # self.light_axis_1_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )
        self.rgb_blocks = nn.ModuleList(
        # self.light_axis_2_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )
        self.depth = depth
        self.aa_order = aa_order
        self.patch_size = patch_size
        self.aa_block_size = aa_block_size
        
        if self.depth % self.aa_block_size != 0:
            raise ValueError(f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})")

        self.aa_block_num = self.depth // self.aa_block_size

    def __build_patch_embed__(
        self,
        patch_embed,
        img_size,
        patch_size,
        num_register_tokens,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        block_chunks=0,
        init_values=1.0,
        embed_dim=1024,
    ):
        """
        Build the patch embed layer. If 'conv', we use a
        simple PatchEmbed conv layer. Otherwise, we use a vision transformer.
        """

        if "conv" in patch_embed:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        else:
            vit_models = {
                "dinov2_vitl14_reg": vit_large,
                "dinov2_vitb14_reg": vit_base,
                "dinov2_vits14_reg": vit_small,
                "dinov2_vitg2_reg": vit_giant2,
            }

            self.patch_embed = vit_models[patch_embed](
                img_size=img_size,
                patch_size=patch_size,
                num_register_tokens=num_register_tokens,
                interpolate_antialias=interpolate_antialias,
                interpolate_offset=interpolate_offset,
                block_chunks=block_chunks,
                init_values=init_values
                
            )
            
            if hasattr(self.patch_embed, "mask_token"):
                self.patch_embed.mask_token.requires_grad_(False)

    def forward(
        self,
        images: torch.Tensor,
        INJECTION=False,
        inj_feature=None,
        frame_type_ids=None,
    ) -> Tuple[List[torch.Tensor], int]:
        # images:         [B, N*K, C, Hc, Wc]
        # frame_type_ids: [B, N*K, 2]
        
        B, S, C_in, H, W = images.shape         # S: N*K
        images = images.view(B * S, C_in, H, W) # [B*S, C, H, W]
        patch_tokens = self.patch_embed(images)

        if isinstance(patch_tokens, dict):
            register_tokens = patch_tokens["x_norm_regtokens"] 
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        if not INJECTION:
            # print('No injection for light tokens')
            light_tokens = register_tokens[:,:3,:] 
        elif INJECTION=='add':
            # print('Using ADD injection for light tokens')
            inj_feature = rearrange(inj_feature, 'b s num c -> (b s) num c')
            light_tokens = register_tokens[:,:3,:] + inj_feature
        elif INJECTION=='concat':
            # print('Using CONCAT injection for light tokens')
            self.patch_start_idx = 10
            inj_feature = rearrange(inj_feature, 'b s num c -> (b s) num c')
            light_tokens = torch.cat((inj_feature, register_tokens[:,:3,:]), dim=1)
        elif INJECTION=='replace':
            # print('Using REPLACE injection for light tokens')
            inj_feature = rearrange(inj_feature, 'b s num c -> (b s) num c')
            light_tokens = inj_feature


        patch_tokens = torch.cat((light_tokens,register_tokens,patch_tokens), dim=1) 
        _, T, C = patch_tokens.shape # (B*N*K, P, C), B = batch, S = 6*K, T = (L+R+P) num_light_tokens + num_register_tokens + num_patch_tokens
        tokens = patch_tokens
        
        # if frame_type_ids is None:
        #     raise RuntimeError("[FATAL] frame_type_ids is required for aggregator attention.")
        # if frame_type_ids.dim() == 2:
        #     frame_type_ids = frame_type_ids.view(B, S, 2)
        # elif frame_type_ids.dim() != 3:
        #     raise RuntimeError(f"[FATAL] Unexpected frame_type_ids shape: {tuple(frame_type_ids.shape)}")
        # if frame_type_ids.shape[:2] != (B, S):
        #     raise RuntimeError(
        #         f"[FATAL] frame_type_ids shape mismatch: expected ({B}, {S}, 2), got {tuple(frame_type_ids.shape)}"
        #     )

        pol_id = frame_type_ids[..., 0]
        rgb_id = frame_type_ids[..., 1]
        n_polar = int(pol_id.max().item()) + 1
        n_rgb = int(rgb_id.max().item()) + 1
        

        # e = self.polar_embed(pol_id) + self.rgb_embed(rgb_id)   # [B,S,C]
        # e = e.to(dtype=tokens.dtype, device=tokens.device)
            
        # tokens = tokens + e

        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device)

        if self.patch_start_idx > 0:
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(images.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)
        # print("pos shape: ", pos.shape, pos.min(), pos.max())

        
        _, T, C = tokens.shape

        frame_idx = 0
        global_idx = 0
        polar_idx = 0
        rgb_idx = 0
        # light_axis_idx_1 = 0
        # light_axis_idx_2 = 0
        output_list = []

        for _ in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, S, T, C, frame_idx, pos=pos
                    )
                elif attn_type == "global":
                    tokens, global_idx, global_intermediates = self._process_global_attention(
                        tokens, B, S, T, C, global_idx, pos=pos
                    )
                elif attn_type == "polar":
                    tokens, polar_idx, polar_intermediates = self._process_polar_attention(
                        tokens, B, S, T, C, polar_idx, n_polar, n_rgb, pos=pos
                    )
                elif attn_type == "rgb":
                    tokens, rgb_idx, rgb_intermediates = self._process_rgb_attention(
                        tokens, B, S, T, C, rgb_idx, n_polar, n_rgb, pos=pos
                    )
                # elif attn_type == "light_axis_2":
                #     tokens, light_axis_idx_2, light_axis_intermediates_2 = self._process_light_axis_2_attention(
                #         tokens, B, S, P, C, light_axis_idx_2, pos=pos
                #     )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            for i in range(len(frame_intermediates)):
                
                concat_inter = torch.cat([frame_intermediates[i],polar_intermediates[i],global_intermediates[i],rgb_intermediates[i]], dim=-1)
                # print("len(intermediates): ", len(frame_intermediates))
                # print("concat_inter shape: ", concat_inter.shape, frame_intermediates[i].shape)
                output_list.append(concat_inter)

        del concat_inter
        del frame_intermediates
        del global_intermediates
        del polar_intermediates
        del rgb_intermediates
        if self.depth == 6:
            output_list.pop(1)
            output_list.pop(2)
        elif self.depth == 8:
            output_list.pop(1)
            output_list.pop(2)
            output_list.pop(3)
            output_list.pop(4)
        return output_list, self.patch_start_idx

    def _process_frame_attention(self, tokens, B, S, T, C, frame_idx, pos=None):
        """
        Process frame attention blocks. We keep tokens in shape (B*S, T, C).
        """
        # [B*(N*K), T, C]
        if tokens.shape != (B * S, T, C):
            tokens = tokens.view(B, S, T, C).view(B * S, T, C)

        if pos is not None and pos.shape != (B * S, T, 2):
            pos = pos.view(B, S, T, 2).view(B * S, T, 2)

        intermediates = []

        
        for _ in range(self.aa_block_size):
            tokens = self.frame_blocks[frame_idx](tokens, pos=pos)
            frame_idx += 1
            intermediates.append(tokens.view(B, S, T, C))
        return tokens, frame_idx, intermediates

    def _process_global_attention(self, tokens, B, S, T, C, global_idx, pos=None):
        """
        Process global attention blocks. We keep tokens in shape (B, S*T, C).
        """
        if tokens.shape != (B, S * T, C):
            tokens = tokens.view(B, S, T, C).view(B, S * T, C)

        if pos is not None and pos.shape != (B, S * T, 2):
            pos = pos.view(B, S, T, 2).view(B, S * T, 2)

        intermediates = []

        
        for _ in range(self.aa_block_size):
            tokens = self.global_blocks[global_idx](tokens, pos=pos)
            global_idx += 1
            intermediates.append(tokens.view(B, S, T, C))

        return tokens, global_idx, intermediates
    
    def _process_polar_attention(self, tokens, B, S, T, C, polar_idx, n_polar, n_rgb, pos=None):
        """
        Process polar attention blocks. Attention is applied across the polar axis.
        """
        n_repeats = S // (n_polar * n_rgb)
        tokens = tokens.view(B, n_polar, n_rgb, n_repeats, T, C).permute(0, 1, 4, 2, 3, 5)
        tokens = tokens.reshape(B * n_polar * T, n_rgb * n_repeats, C)

        if pos is not None and pos.shape != (B * n_polar * T, n_rgb * n_repeats, 2):
            pos = pos.view(B, n_polar, n_rgb, n_repeats, T, 2).permute(0, 1, 4, 2, 3, 5)
            pos = pos.reshape(B * n_polar * T, n_rgb * n_repeats, 2)

        intermediates = []

        for _ in range(self.aa_block_size):
            tokens = self.polar_blocks[polar_idx](tokens, pos=pos)
            polar_idx += 1
            tokens = tokens.view(B, n_polar, T, n_rgb, n_repeats, C).permute(0, 1, 3, 4, 2, 5)
            tokens = tokens.reshape(B, S, T, C)
            intermediates.append(tokens)
            tokens = tokens.reshape(B * S, T, C)

        return tokens, polar_idx, intermediates
    
    def _process_rgb_attention(self, tokens, B, S, T, C, rgb_idx, n_polar, n_rgb, pos=None):
        """
        Process spectral/RGB attention blocks. Attention is applied across the spectral axis.
        """
        n_repeats = S // (n_polar * n_rgb)
        tokens = tokens.view(B, n_polar, n_rgb, n_repeats, T, C).permute(0, 2, 4, 1, 3, 5)
        tokens = tokens.reshape(B * n_rgb * T, n_polar * n_repeats, C)

        if pos is not None and pos.shape != (B * n_rgb * T, n_polar * n_repeats, 2):
            pos = pos.view(B, n_polar, n_rgb, n_repeats, T, 2).permute(0, 2, 4, 1, 3, 5)
            pos = pos.reshape(B * n_rgb * T, n_polar * n_repeats, 2)

        intermediates = []
        for _ in range(self.aa_block_size):
            tokens = self.rgb_blocks[rgb_idx](tokens, pos=pos)
            rgb_idx += 1
            tokens = tokens.view(B, n_rgb, T, n_polar, n_repeats, C).permute(0, 3, 1, 4, 2, 5)
            tokens = tokens.reshape(B, S, T, C)
            intermediates.append(tokens)
            tokens = tokens.reshape(B * S, T, C)

        return tokens, rgb_idx, intermediates
    
def slice_expand_and_flatten(token_tensor, B, S):
    """
    Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing:
    1) Uses the first position (index=0) for the first frame only
    2) Uses the second position (index=1) for all remaining frames (S-1 frames)
    3) Expands both to match batch size B
    4) Concatenates to form (B, S, X, C) where each sequence has 1 first-position token
       followed by (S-1) second-position tokens
    5) Flattens to (B*S, X, C) for processing

    Returns:
        torch.Tensor: Processed tokens with shape (B*S, X, C)
    """

    
    query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
    
    combined = torch.cat([query, others], dim=1)

    
    combined = combined.view(B * S, *combined.shape[2:])
    return combined
