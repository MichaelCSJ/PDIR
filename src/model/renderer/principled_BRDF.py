import os
from pathlib import Path
from collections import defaultdict
import math

import numpy as np
import cv2
# import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# Calibrated LCD emitter positions (9x16 superpixel grid, camera space).
# Vendored in this repo so the renderer works without an external data root.
DEFAULT_LIGHT_POS_PATH = (
    Path(__file__).resolve().parents[3] / "calibration" / "finetuned_position.npy"
)



class Principled_BRDF(nn.Module):
    def __init__(self, light_pos_path=None):
        super().__init__()

        if light_pos_path is None:
            light_pos_path = DEFAULT_LIGHT_POS_PATH
        light_positions = np.load(light_pos_path).astype(np.float32)
        light_positions = light_positions.reshape(9, 16, 3)
        light_positions = np.flip(light_positions, axis=1).reshape(-1, 3)
        light_positions = torch.from_numpy(light_positions).float()
        self.register_buffer("light_positions", light_positions, persistent=False)

    def _torch_normalize(self, v, eps=1e-4):
        # eps=1e-4: stable backward when ||v||≈0 (zero-vector inputs would
        # otherwise propagate NaN through the geometry).
        return v / (torch.linalg.norm(v, dim=-1, keepdim=True) + eps)


    def _np_normalize(self, v, eps=1e-4):
        return v / (np.linalg.norm(v, axis=-1, keepdims=True) + eps)


    def _fresnel_schlick_torch(self, LdotH, F0):
        """
        LdotH: [B*H*W,R*C,1] / [1,R*C,1]
        F0:    [B*H*W, _ ,3]
        """
        return F0 + (1.0 - F0) * (1.0 - LdotH) ** 5


    def _distribution_ggx_torch(self, NdotH, roughness, eps=1e-6):
        """
        NdotH:      [B*H*W,R*C,1]
        roughness:  [B*H*W, _ ,1]
        """
        a = roughness ** 2
        a2 = a * a
        denom = (NdotH * NdotH) * (a2 - 1.0) + 1.0
        return a2 / (math.pi * denom * denom + eps)


    def _geometry_schlick_ggx_torch(self, NdotX, roughness, eps=1e-6):
        a = roughness ** 2
        a2 = a * a
        arg = torch.clamp(a2 + (1.0 - a2) * NdotX ** 2, min=0.0)
        return 1.0 / (NdotX + torch.sqrt(arg) + eps)


    def _geometry_smith_torch(self, NdotL, NdotV, roughness):
        """
        NdotL:      [B*H*W,R*C,1]
        NdotV:      [B*H*W, 1 ,1]
        roughness:  [B*H*W, _ ,1]
        """        
        ggx_l = self._geometry_schlick_ggx_torch(NdotL, roughness)
        ggx_v = self._geometry_schlick_ggx_torch(NdotV, roughness)
        
        return ggx_l * ggx_v
    
    def _compute_light_direction(self, ptcloud_bhw3, light_pos_rc3):
        '''
        ptcloud_bhw3: [B*H*W,3], light_pos_rc3: [R*C,3]
        incident: [B*H*W,R*C,3], exitant: [B*H*W,1,3]
        '''
        incident = light_pos_rc3.unsqueeze(0) - ptcloud_bhw3.unsqueeze(-2)
        incident = self._torch_normalize(incident)
        exitant = -self._torch_normalize(ptcloud_bhw3).unsqueeze(-2)
        
        return incident, exitant


    def forward(self, base, nrm, rgh, mtl,
                point_cloud=None,
                FALLOFF=None,
                ):
        """
        basecolor:  BxHxW,3 torch
        normal:     BxHxW,3 torch
        roughness:  BxHxW,1 torch
        metallic:   BxHxW,1 torch
        light_dirs: [L,3]
        point_cloud: [3] or [B*H*W, 3]
        return:     [B*H*W,R*C,3] in [0,1]
        """
        device = base.device
        
        # [B*H*W,_,3or1]
        base = torch.clamp(base, 0.0, 1.0).unsqueeze(-2)
        nrm = self._torch_normalize(nrm).unsqueeze(-2)
        rgh = torch.clamp(rgh, 0.02, 1.0).unsqueeze(-2)
        mtl = torch.clamp(mtl, 0.0, 1.0).unsqueeze(-2)
        
        if point_cloud is None:
            # [3]
            point_cloud = torch.tensor([0.0, 0.0, 0.5], dtype=torch.float32, device=device)[None:]
        else:
            # [B*H*W, 3]
            point_cloud =  point_cloud
        # light_dirs = self.light_positions - point_cloud
        # light_dirs = self._np_normalize(light_dirs)
        # light_dirs[..., 1] *= -1.0
        # light_dirs[..., 2] *= -1.0
        
        # [B*H*W,144,3], [B*H*W,1,3]
        wi, wo = self._compute_light_direction(point_cloud, self.light_positions)   
        wi[..., 1] *= -1.0
        wi[..., 2] *= -1.0
        wo[..., 1] *= -1.0
        wo[..., 2] *= -1.0
        
        V = wo                           # [B*H*W,1,3]   / [1,1,3]
        L = wi                           # [B*H*W,R*C,3] / [1,R*C,3]
        H = self._torch_normalize(L + V) # [B*H*W,R*C,3] / [1,R*C,3]

        NdotL = torch.clamp(torch.sum(nrm * L, dim=-1, keepdim=True), 0.0, 1.0)   # [B*H*W, R*C, 1] / [B*H*W,R*C,1]
        NdotV = torch.clamp(torch.sum(nrm * V, dim=-1, keepdim=True), 1e-4, 1.0)   # [B*H*W,  1 , 1] / [B*H*W, 1 ,1]
        # NdotV = torch.clamp(torch.sum(nrm * V, dim=-1, keepdim=True), 0.0, 1.0)   # [B*H*W,  1 , 1] / [B*H*W, 1 ,1]
        NdotH = torch.clamp(torch.sum(nrm * H, dim=-1, keepdim=True), 0.0, 1.0)   # [B*H*W, R*C, 1] / [B*H*W,R*C,1]
        LdotH = torch.clamp(torch.sum(L * H, dim=-1, keepdim=True), 0.0, 1.0)     # [B*H*W, R*C, 1] / [  1  ,R*C,1]

        SPECULAR = 1.0
        F0 = 0.08 * SPECULAR * (1.0 - mtl) + base * mtl              # [B*H*W, 1 ,3]
        F = self._fresnel_schlick_torch(LdotH, F0)        # [B*H*W,R*C,3]
        D = self._distribution_ggx_torch(NdotH, rgh)      # [B*H*W,R*C,1]
        G = self._geometry_smith_torch(NdotL, NdotV, rgh) # [B*H*W,R*C,1]

        spec = F * D * G # [B*H*W,R*C,3]

        FD90 = 0.5 + 2.0 * (LdotH * LdotH * rgh)                  # [B*H*W,R*C,1]
        FD = (1.0 + (FD90 - 1.0) * ((1.0 - NdotL) ** 5)) * \
            (1.0 + (FD90 - 1.0) * ((1.0 - NdotV) ** 5))   # [B*H*W,R*C,1]
        diff = (1.0 - mtl) * (base / math.pi) * FD   # [B*H*W,R*C,3]

        shaded = (diff + spec) * NdotL
        shaded = torch.clamp(shaded, 0.0, 1.0)

        # [B*H*W,L*C,3]
        return diff*NdotL*0.12, spec*NdotL*0.12
        # return diff*NdotL/9.*0.3766, spec*NdotL/9.*0.3766 
        # return torch.clamp(diff*NdotL, 0.0, 1.0)/9.*0.3766, torch.clamp(spec*NdotL, 0.0, 1.0)/9.*0.3766
        # return torch.clamp(diff*NdotL, 0.0, 1.0)/9.*0.3857, torch.clamp(spec*NdotL, 0.0, 1.0)/9.*0.3857
        # return shaded.mean(dim=-2)  # [B*H*W,3]
        # return shaded               # [B*H*W,L*C,3]
        
        
        
'''
class Principled_BRDF(nn.Module):
    def __init__(self, light_pos_path=None):
        super().__init__()

        if light_pos_path is None:
            light_pos_path = DEFAULT_LIGHT_POS_PATH
        light_positions = np.load(light_pos_path).astype(np.float32)
        light_positions = light_positions.reshape(9, 16, 3)
        light_positions = np.flip(light_positions, axis=1).reshape(-1, 3)
        light_positions = torch.from_numpy(light_positions).float()
        self.register_buffer("light_positions", light_positions, persistent=False)

    def _torch_normalize(self, v, eps=1e-8):
        return v / (torch.linalg.norm(v, dim=-1, keepdim=True) + eps)


    def _np_normalize(self, v, eps=1e-8):
        return v / (np.linalg.norm(v, axis=-1, keepdims=True) + eps)


    def _fresnel_schlick_torch(self, LdotH, F0):
        """
        LdotH: [B,H,W,R*C,1] / [1,1,1,R*C,1]
        F0:    [B,H,W, _ ,3]
        """
        return F0 + (1.0 - F0) * (1.0 - LdotH) ** 5


    def _distribution_ggx_torch(self, NdotH, roughness, eps=1e-8):
        """
        NdotH:      [B,H,W,R*C,1]
        roughness:  [B,H,W, _ ,1]
        """
        a = roughness ** 2
        a2 = a * a
        denom = (NdotH * NdotH) * (a2 - 1.0) + 1.0
        return a2 / (math.pi * denom * denom + eps)


    def _geometry_schlick_ggx_torch(self, NdotX, roughness, eps=1e-8):
        a = roughness ** 2
        a2 = a * a
        return 1.0 / (NdotX + torch.sqrt(a2 + (1.0 - a2) * NdotX ** 2 + eps))


    def _geometry_smith_torch(self, NdotL, NdotV, roughness):
        """
        NdotL:      [B,H,W,R*C,1]
        NdotV:      [B,H,W, 1 ,1]
        roughness:  [B,H,W, _ ,1]
        """        
        ggx_l = self._geometry_schlick_ggx_torch(NdotL, roughness)
        ggx_v = self._geometry_schlick_ggx_torch(NdotV, roughness)
        
        return ggx_l * ggx_v
    
    def _compute_light_direction(self, ptcloud_bhw3, light_pos_rc3):
        """
        ptcloud_bhw3: [B,H,W,3], light_pos_rc3: [R*C,3]
        incident: [B,H,W,R*C,3], exitant: [B,H,W,1,3]
        """
        incident = light_pos_rc3.unsqueeze(0).unsqueeze(0).unsqueeze(0) - ptcloud_bhw3.unsqueeze(-2)
        incident = self._torch_normalize(incident)
        exitant = -self._torch_normalize(ptcloud_bhw3).unsqueeze(-2)
        
        return incident, exitant


    def forward(self, base, nrm, rgh, mtl,
                point_cloud=None,
                FALLOFF=None,
                ):
        """
        basecolor:  BxHxWx3 torch
        normal:     BxHxWx3 torch
        roughness:  BxHxWx1 torch
        metallic:   BxHxWx1 torch
        light_dirs: [L,3]
        point_cloud: [3] or [B, H, W, 3]
        return:     [B,H,W,R*C,3] in [0,1]
        """
        device = base.device
        
        # [B,H,W,_,3/1]
        base = torch.clamp(base, 0.0, 1.0).unsqueeze(-2)
        nrm = self._torch_normalize(nrm).unsqueeze(-2)
        rgh = torch.clamp(rgh, 0.0, 1.0).unsqueeze(-2)
        mtl = torch.clamp(mtl, 0.0, 1.0).unsqueeze(-2)
        
        if point_cloud is None:
            # [1,1,1,3]
            point_cloud = torch.tensor([0.0, 0.0, 0.5], dtype=torch.float32, device=device)[None,None,None,:]
        else:
            # [B,H,W,3]
            point_cloud =  point_cloud
        # light_dirs = self.light_positions - point_cloud
        # light_dirs = self._np_normalize(light_dirs)
        # light_dirs[..., 1] *= -1.0
        # light_dirs[..., 2] *= -1.0
        
        # [B,H,W,144,3], [B,H,W,1,3]
        wi, wo = self._compute_light_direction(point_cloud, self.light_positions)   
        wi[..., 1] *= -1.0
        wi[..., 2] *= -1.0
        # wo[..., 1] *= -1.0
        # wo[..., 2] *= -1.0
        
        V = wo                           # [B,H,W,1,3]   / [1,1,1,1,3]
        L = wi                           # [B,H,W,R*C,3] / [1,1,1,R*C,3]
        H = self._torch_normalize(L + V) # [B,H,W,R*C,3] / [1,1,1,R*C,3]

        NdotL = torch.clamp(torch.sum(nrm * L, dim=-1, keepdim=True), 0.0, 1.0)   # [B,H,W,R*C,1] / [B,H,W,R*C,1]
        NdotV = torch.clamp(torch.sum(nrm * V, dim=-1, keepdim=True), 0.0, 1.0)   # [B,H,W, 1 ,1] / [B,H,W, 1 ,1]
        NdotH = torch.clamp(torch.sum(nrm * H, dim=-1, keepdim=True), 0.0, 1.0)   # [B,H,W,R*C,1] / [B,H,W,R*C,1]
        LdotH = torch.clamp(torch.sum(L * H, dim=-1, keepdim=True), 0.0, 1.0)     # [B,H,W,R*C,1] / [1,1,1,R*C,1]

        F0 = 0.08 * (1.0 - mtl) + base * mtl              # [B,H,W,3]
        F = self._fresnel_schlick_torch(LdotH, F0)        # [B,H,W,R*C,1]
        D = self._distribution_ggx_torch(NdotH, rgh)      # [B,H,W,R*C,1]
        G = self._geometry_smith_torch(NdotL, NdotV, rgh) # [B,H,W,R*C,1]

        spec = F * D * G # [B,H,W,R*C,1]

        FD90 = 0.5 + 2.0 * (LdotH * rgh)                  # [B,H,W,R*C,1]
        FD = (1.0 + (FD90 - 1.0) * ((1.0 - NdotL) ** 5)) * \
            (1.0 + (FD90 - 1.0) * ((1.0 - NdotV) ** 5))   # [B,H,W,R*C,1]
        diff = (FD ** 2) * base * (1.0 - mtl) / math.pi   # [B,H,W,R*C,3]

        shaded = (diff + spec) * NdotL         
        shaded = torch.clamp(shaded, 0.0, 1.0)

        # return shaded.mean(dim=-2)  # [B,H,W,3]
        # return shaded               # [B,H,W,L*C,3]
        return torch.clamp(diff*NdotL, 0.0, 1.0)/9.*0.3766, torch.clamp(spec*NdotL, 0.0, 1.0)/9.*0.3766
        # return torch.clamp(diff*NdotL, 0.0, 1.0)/9.*0.3857, torch.clamp(spec*NdotL, 0.0, 1.0)/9.*0.3857
'''
