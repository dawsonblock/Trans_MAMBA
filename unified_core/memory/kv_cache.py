"""
KV Cache for transformer-style attention.

Provides efficient key-value caching for autoregressive generation.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class KVCacheConfig:
    """Configuration for KV cache."""
    max_seq_len: int = 2048
    n_layers: int = 4
    n_heads: int = 4
    head_dim: int = 64


class KVCache(nn.Module):
    """
    Key-Value cache for efficient autoregressive inference.

    Maintains separate K/V tensors per layer for incremental decoding.
    """

    def __init__(self, cfg: KVCacheConfig):
        super().__init__()
        self.cfg = cfg
        self.cache_k = None
        self.cache_v = None
        self.seq_len = 0

    def reset(self, batch_size: int, device: torch.device):
        """Reset cache for new sequence."""
        cfg = self.cfg
        self.cache_k = torch.zeros(
            cfg.n_layers, batch_size, cfg.max_seq_len,
            cfg.n_heads, cfg.head_dim, device=device
        )
        self.cache_v = torch.zeros(
            cfg.n_layers, batch_size, cfg.max_seq_len,
            cfg.n_heads, cfg.head_dim, device=device
        )
        self.seq_len = 0

    def update(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache and return full K/V for attention.

        Args:
            layer_idx: Which layer's cache to update
            key: [B, 1, H, D] new key
            value: [B, 1, H, D] new value

        Returns:
            full_k: [B, seq_len+1, H, D]
            full_v: [B, seq_len+1, H, D]
        """
        B = key.size(0)

        if self.cache_k is None:
            self.reset(B, key.device)

        pos = self.seq_len
        self.cache_k[layer_idx, :B, pos] = key.squeeze(1)
        self.cache_v[layer_idx, :B, pos] = value.squeeze(1)

        if layer_idx == self.cfg.n_layers - 1:
            self.seq_len += 1

        return (
            self.cache_k[layer_idx, :B, :self.seq_len + 1],
            self.cache_v[layer_idx, :B, :self.seq_len + 1],
        )

    def get(
        self, layer_idx: int, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cached K/V for a layer."""
        return (
            self.cache_k[layer_idx, :batch_size, :self.seq_len],
            self.cache_v[layer_idx, :batch_size, :self.seq_len],
        )
