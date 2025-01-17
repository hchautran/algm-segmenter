from typing import Tuple
import math
import torch

from segm.model import segmenter, blocks, vit
from segm.model.blocks import Block, Attention
from segm.model.segmenter import Segmenter 
from segm.model.vit import VisionTransformer
from algm.utils import parse_r

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import math
from typing import Callable, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def do_nothing(x, mode=None):
    return x

def bsm(
    metric: torch.Tensor,
    ratio:float=1.0,    
    class_token: bool = False,
) -> Tuple[Callable, Callable]:
    
    protected = 0
    if class_token:
        protected += 1
    if len(metric.shape) == 2:
        metric = metric[None,...]

    # We can only reduce by a maximum of 50% tokens
    T = metric.shape[1]
    
    if ratio < 1.0:
        r = math.floor(T- T*ratio)
    else:
        return do_nothing, do_nothing


    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)

        if class_token:
            scores[..., 0, :] = -math.inf

        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

        if class_token:
            unm_idx = unm_idx.sort(dim=1)[0]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        if len(x.shape) == 2:
            x.unsqueeze_(0)
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        return torch.cat([unm, dst], dim=1)
    


    return merge

def pitome(
    metric=None,
    class_token: bool = False,
    indices:torch.Tensor=None, 
    scores:torch.Tensor=None,
    margin:float=0.9,
    r:int=None
) -> Tuple[Callable, Callable]:
    B, T, T = scores.shape
    # seperate protected token and mergeable tokens  
    merge_idx = indices[..., :2*r]
    protected_idx = indices[..., 2*r:]
    a_idx, b_idx = merge_idx[..., ::2], merge_idx[..., 1::2] 

    # get similarity scores between mergeable tokens
    scores = scores.gather(dim=-1, index=b_idx.unsqueeze(-2).expand(B, T, r)) 
    scores = scores.gather(dim=-2, index=a_idx.unsqueeze(-1).expand(B, r, r ))
    values, dst_idx = scores.max(dim=-1) 

    # if values.mean(-1) < 0.0:
        # return do_nothing
    
    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        if class_token:
            x_cls=x[:,0,:].unsqueeze(1)
            x=x[:,1:,:]

        B, T, C = x.shape
        batch_idx = torch.arange(B).unsqueeze_(1).to(metric.device)
        protected = x[batch_idx, protected_idx, :]
        src, dst = x[batch_idx, a_idx, :], x[batch_idx,  b_idx, :]
        if mode != "prune":
            dst = dst.scatter_reduce(-2, dst_idx.unsqueeze(2).expand(B, r, C), src, reduce=mode)

        if class_token:
            return torch.cat([x_cls, protected, dst], dim=1)
        return torch.cat([protected, dst], dim=1)

    return merge


def pitome_bsm(
    metric=None,
    class_token: bool = False,
    indices:torch.Tensor=None,
    scores:torch.Tensor=None,
    r:int=None
) -> Tuple[Callable, Callable]:

    with torch.no_grad():
        B, T, _ = scores.shape
        a_idx, b_idx = indices[..., ::2], indices[..., 1::2] 
        batch_idx = torch.arange(B).unsqueeze_(1).to(metric.device)
        scores = scores.gather(dim=-1, index=b_idx.unsqueeze(-2).expand(B, T, b_idx.shape[-1])) 
        scores = scores.gather(dim=-2, index=a_idx.unsqueeze(-1).expand(B, a_idx.shape[-1], b_idx.shape[-1]))
        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        if class_token:
            x_cls=x[:,0,:].unsqueeze(1)
            x=x[:,1:,:]

        src, dst = x[batch_idx, a_idx, :], x[batch_idx, b_idx, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        if mode != "prune":
            src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
            dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        if class_token:
            return torch.cat([x_cls, unm, dst], dim=1)
        return torch.cat([unm, dst], dim=1)
    return merge

def pitome_vision(
    metric: torch.Tensor, 
    ratio=1.0,
    margin:torch.Tensor=0.5,
    class_token: bool = False,
    r=None,
):
    if r == 0:
        return do_nothing, r
    else:
        with torch.no_grad():
            if class_token:
                metric=metric[:,1:,:]
            metric = F.normalize(metric, p=2, dim=-1) 
            print('metric', metric.size(1))
            sim = metric@metric.transpose(-1,-2) - torch.eye(metric.size(1), device=metric.device)[None, ...]
            energy_score = F.relu(sim - margin).mean(dim=-1)
            r = (energy_score > 0).int().sum(-1).item()
            r = min(r, metric.size(1)//2)
            if r == 0:
                return do_nothing, r
            indices =  torch.argsort(energy_score, descending=True)

        return pitome_bsm(metric=metric, class_token=class_token, indices=indices, scores=sim, r=r), r 



def progressive_pitome(
    x_cls=None,
    x_metric=None,
    class_token: bool = False,
    indices:torch.Tensor=None, 
    margin=0.88,
    mode='mean'
) -> Tuple[Callable, Callable]:
    B, T, C = x_metric.shape
    batch_idx = torch.arange(B).unsqueeze_(1).to(x_metric.device)
    # choose dst with highest energy score 
    merge_idx, dst_idx = indices[..., 1:], indices[..., :1] 
    x_merge, x_dst = x_metric[batch_idx, merge_idx, :], x_metric[batch_idx,  dst_idx, :]
    scores = x_dst @ x_merge.transpose(-1,-2) 

    src_idx = torch.nonzero(scores >= margin, as_tuple=False)
    protected_idx = torch.nonzero(scores < margin, as_tuple=False)
    x_src = x_merge[batch_idx, src_idx, :]
    x_protected = x_merge[batch_idx, protected_idx, :]
    x_dst = x_dst.scatter_reduce(-2, torch.tensor([[[0]]]).to(x_src.device).expand(B, x_src.shape[-2], C), x_src, reduce=mode)

    if class_token:
        return torch.cat([x_cls, x_protected, x_dst], dim=1)
    return torch.cat([x_protected, x_dst], dim=1)





def merge_mean(
    merge: Callable, x: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a average based on token size.
    Returns the merged tensor and the new token sizes.
    """
    x = merge(x, mode="mean")
    return x

def prune(
    merge: Callable, x: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a average based on token size.
    Returns the merged tensor and the new token sizes.
    """
    x = merge(x, mode="prune")
    return x


def merge_wavg(
    merge: Callable, x: torch.Tensor, size: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a weighted average based on token size.
    Returns the merged tensor and the new token sizes.
    """
    if size is None:
        size = torch.ones_like(x[..., 0, None])

    x = merge(x*size, mode="sum")
    size = merge(size, mode="sum")
    x = x / size

    return x, size 


def merge_source(
    merge: Callable, x: torch.Tensor, source: torch.Tensor = None
) -> torch.Tensor:
    """
    For source tracking. Source is an adjacency matrix between the initial tokens and final merged groups.
    x is used to find out how many tokens there are in case the source is None.
    """
    if source is None:
        n, t, _ = x.shape
        source = torch.eye(t, device=x.device)[None, ...].expand(n, t, t)

    source = merge(source, mode="amax")
    return source

def merge_attention_mask(
    merge, attention_mask: torch.Tensor
): 
    attention_mask = merge(attention_mask, mode="amax")
    return attention_mask 





class TurboBlock(Block):
    """
    Modifications:
     - Apply ALGM between the attention and mlp blocks
    """

    def _drop_path1(self, x):
        return self.drop_path1(x) if hasattr(self, "drop_path1") else self.drop_path(x)

    def _drop_path2(self, x):
        return self.drop_path2(x) if hasattr(self, "drop_path2") else self.drop_path(x)

    def forward(self, x: torch.Tensor ) -> torch.Tensor:
      
        attn_size = self._turbo_info["size"] if self._turbo_info["prop_attn"] else None
        x_attn, metric  = self.attn(self.norm1(x),attn_size)
        x =  x + self._drop_path1(x_attn)
        r = None
           
        if self._turbo_info["source"] is None: # if layer_idx == 1:
            for _ in range(self._turbo_info["num_merge_step"]):
                merge, r  = pitome_vision(
                    metric=x,
                    margin=self._turbo_info["margin"],
                    class_token=self._turbo_info["class_token"],
                    r=r
                )
                if self._turbo_info["trace_source"]:
                    self._turbo_info["source"] = merge_source(
                        merge, x, self._turbo_info["source"]
                    )
                x, self._turbo_info["size"] = merge_wavg(merge, x, self._turbo_info["size"])
        # else:
                # merge = turbo_matching(
                #     x,
                #     layer_idx,
                #     self._turbo_info["source"],
                #     self._turbo_info["class_token"],
                #     self._turbo_info["distill_token"],
                # )
                # if self._turbo_info["trace_source"]:
                #     self._turbo_info["source"] = merge_source(
                #         merge, x, self._turbo_info["source"]
                #     )
                # x, self._turbo_info["size"] = merge_wavg(merge, x, self._turbo_info["size"])
           
        
        x = x + self._drop_path2(self.mlp(self.norm2(x)))
        
        return x 


class TurboAttention(Attention):

    def forward(
        self, x: torch.Tensor,size: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Note: this is copied from timm.models.vision_transformer.Attention with modifications.
        # we do not change anything here, and do not use  q.mean(1)
        B, N, C = x.shape
        
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.heads, C // self.heads)
            .permute(2, 0, 3, 1, 4)
        )
        
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  

        attn = (q @ k.transpose(-2, -1)) * self.scale 
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

       
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        x = self.proj(x)
        x = self.proj_drop(x)
        return x , q.mean(1)

class TurboVisionTransformer(VisionTransformer):
   

    def forward(self, *args, **kwdargs) -> torch.Tensor:
    
        self._turbo_info["size"] = None
        self._turbo_info["source"] = None
        self._turbo_info["rel_pos"] = None
        self._turbo_info["selected_layers"] = list(self.selected_layers)
        self._turbo_info["window_size"] = self.window_size
        self._turbo_info["margin"] = self.margin
       


        return super().forward(*args, **kwdargs)


def apply_patch(
    model: Segmenter, selected_layers: list, trace_source: bool = False, prop_attn: bool = True, num_merge_step: int = 1, ratio=1.0 
    
):

    model = model.encoder
    model.__class__ = TurboVisionTransformer
    
    
    model.selected_layers = selected_layers
    model.window_size = (2,2)
    model.margin = 0.88
    model._turbo_info = {
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": model.cls_token is not None,
        "distill_token": False,
        "num_merge_step": num_merge_step,
        "rel_pos": None,
        "selected_layers":model.selected_layers,
        "window_size":model.window_size,
        "ratio": ratio,
        # "threshold":model.threshold,
     
    }

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._turbo_info["distill_token"] = True


    print('selected layer', model.selected_layers)
    len_att = 1
    len_block = 1
    for module in model.modules():
        if isinstance(module, Block):
            if len_att in model.selected_layers:
                module.__class__ = TurboBlock
                module._turbo_info = model._turbo_info
            len_att +=1 
        elif isinstance(module, Attention):
            if len_block in model.selected_layers: 
                module.__class__ = TurboAttention
                module._turbo_info = model._turbo_info
            len_block +=1 
