import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Type, Any
from torch import Tensor
import math
import numpy as np
from einops import rearrange

class MLP(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class MaskExtractor(nn.Module): # Mask-based Feature Extractor
    def __init__(self, mask_shape=112, embed_dim=1024, out_dim=4096, num_heads=8, mlp_dim=2048, downsample_rate=2, skip_first_layer_pe=False):
        super(MaskExtractor, self).__init__()
        self.mask_shape = mask_shape
        self.mask_pooling = MaskPooling()
        self.feat_linear = nn.Linear(embed_dim, out_dim)
        self.cross_feat_linear = nn.Linear(embed_dim, out_dim)
        self.mask_linear = MLP(mask_shape*mask_shape, embed_dim, out_dim, 3)

        self.feature_name = ['res2', 'res3', 'res4', 'res5']

        self.cross_att_res = CrossAttention(
            embedding_dim=embed_dim,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            douwnsample_rate=downsample_rate,
            skip_first_layer_pe=skip_first_layer_pe
        )
        
        self.res2 = nn.Linear(192, 1024)
        self.res3 = nn.Linear(384, 1024)
        self.res4 = nn.Linear(768, 1024)
        self.res5 = nn.Linear(1536, 1024)

        self.g_res2 = nn.Linear(16384, 1024) # h * w
        self.g_res3 = nn.Linear(4096, 1024)
        self.g_res4 = nn.Linear(1024, 1024)
        self.g_res5 = nn.Linear(256, 1024)

        self.final_mlp = nn.Linear(2 * out_dim, out_dim)
        
        self.global_vit = nn.Sequential(
            nn.Conv2d(3, 5, 1),
            nn.GELU(),
            nn.AvgPool2d(4, 4),

            nn.Conv2d(5, 1, 1),
            nn.GELU(),
            nn.AvgPool2d(4, 4),
        )
        self.is_first = 0
        
        self.sa = Attention(32 * 32, num_heads) # self-attention
        self.mlp =  MLP(32 * 32, 512, out_dim, 3)

    def cal_globa_local(self, mask_feat_raw, feat_new, res, g_res, cross_attention):
        mask_feat_flatten = mask_feat_raw.to(device=res.weight.device, dtype=res.weight.dtype) 
        mask_feat = res(mask_feat_flatten) # (b, q, 1024)
        
        feat_new = feat_new.to(device=g_res.weight.device, dtype=g_res.weight.dtype) 
        all_feat_new = g_res(feat_new) # (b, c, 1024)
        global_mask = cross_attention(mask_feat, all_feat_new)
        return mask_feat, global_mask
                    
    def forward(self, feats, masks, cropped_img):
        global_features = []
        local_features = []
        num_imgs = len(masks)

        for idx in range(num_imgs):
            mask = masks[idx].unsqueeze(0).float() #(1, q, h, w)
            cropped_ = cropped_img[idx] # (q, 3, h, w)

            num_feats = len(self.feature_name)
            mask_feats = mask.new_zeros(num_feats, mask.shape[1], 1024)
            global_masks = mask.new_zeros(num_feats, mask.shape[1], 1024)

            for i, name in enumerate(self.feature_name):
                feat = feats[name][idx].unsqueeze(0) 
                feat = feat.to(mask.dtype)
                
                mask_feat_raw = self.mask_pooling(feat, mask)
                feat_new = rearrange(feat, 'b c h w -> b c (h w)')
                
                mask_feat, global_mask = self.cal_globa_local(mask_feat_raw, feat_new, res=getattr(self, name),  g_res=getattr(self, 'g_{}'.format(name)),  cross_attention=getattr(self,"cross_att_res"))

                mask_feats[i] = mask_feat.squeeze(0) # (q, 1024)
                global_masks[i] = global_mask.squeeze(0)
            mask_feats = mask_feats.sum(0) # (1, q, 1024)
            global_masks = global_masks.sum(0)  # (1, q, 1024)
            global_masks = global_masks.to(device=self.cross_feat_linear.weight.device, dtype=self.cross_feat_linear.weight.dtype)
            global_masks_linear = self.cross_feat_linear(global_masks)
            mask_feats = mask_feats.to(device=self.feat_linear.weight.device, dtype=self.feat_linear.weight.dtype) 
            mask_feats_linear = self.feat_linear(mask_feats) #(1, q, 4096)

            query_feat = self.final_mlp(torch.cat((global_masks_linear, mask_feats_linear), dim=-1))
            global_features.append(query_feat) # global

            cropped_ = cropped_.to(device=self.feat_linear.weight.device, dtype=self.feat_linear.weight.dtype) 
            global_features = self.global_vit(cropped_).to(device=self.feat_linear.weight.device, dtype=self.feat_linear.weight.dtype)  # q, 1, 32, 32
            global_features = global_features.reshape(-1, 1, 32 * 32) # q, 1, 32 * 32
            pos_feat = self.mlp(self.sa(global_features, global_features, global_features).squeeze(1))  # q, output

            local_features.append(pos_feat) #(imgs_num, 1, q, 4096) # local
        
        return global_features, local_features
    
class MaskPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, mask):

        if not x.shape[-2:] == mask.shape[-2:]:
            # reshape mask to x
            mask = F.interpolate(mask, size=x.shape[-2:], mode='bilinear', align_corners=False)

        mask = (mask > 0).to(mask.dtype)
        denorm = mask.sum(dim=(-1, -2), keepdim=True) + 1e-8

        mask_pooled_x = torch.einsum(
            "bchw,bqhw->bqc",
            x,
            mask / denorm,
        )
        return mask_pooled_x
    

class CrossAttention(nn.Module):
    def __init__(
            self, 
            embedding_dim: int,
            num_heads: int,
            mlp_dim: int = 2048,
            douwnsample_rate: int = 2, 
            activation: Type[nn.Module] = nn.ReLU,
            skip_first_layer_pe: bool = False
        ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads =num_heads
        self.self_attn = Attention(embedding_dim, num_heads) # self-attention
        self.skip_first_layer_pe = skip_first_layer_pe
        self.norm1 = nn.LayerNorm(embedding_dim)

        # cross-attention
        self.cross_attn = Attention(embedding_dim, num_heads, downsample_rate=douwnsample_rate)
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation) # MLP

    def forward(self, queries, keys):
        attn_out = self.self_attn(queries, queries, queries)
        queries = queries + attn_out
        queries = self.norm1(queries)

        attn_out = self.cross_attn(q=queries, k=keys, v=keys)
        queries = attn_out + queries
        queries = self.norm2(queries)

        # MLP
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        return queries

class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))

class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out
