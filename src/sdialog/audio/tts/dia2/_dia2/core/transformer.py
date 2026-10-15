from __future__ import annotations

from typing import Tuple, Optional

import torch
from torch import nn

from ..config import DiaConfig
from .cache import KVCache
from .precision import Precision
from .layers import (
    MultiStreamEmbedding,
    Mlp,
    Attention,
)


class TransformerDecoder(nn.Module):
    """Inference-time port of dia_v2.model.Transformer."""

    def __init__(
        self,
        config: DiaConfig,
        precision: Precision,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.config = config
        self.precision = precision
        data_cfg = config.data
        dec_cfg = config.model.decoder

        self.audio_embeds = nn.ModuleList(
            [
                nn.Embedding(
                    data_cfg.audio_vocab_size,
                    dec_cfg.n_embd,
                    device=device,
                )
                for _ in range(max(0, data_cfg.channels - 2))
            ]
        )
        self.text_embed = MultiStreamEmbedding(
            data_cfg.text_vocab_size,
            dec_cfg.n_embd,
            pad_id=data_cfg.text_pad_token_id,
            output_dtype=self.precision.compute,
            low_rank_dim=dec_cfg.low_rank_dim,
            device=device,
        )
        self.layers = nn.ModuleList([DecoderLayer(config, precision, device=device) for _ in range(dec_cfg.n_layer)])
        self.norm = nn.RMSNorm(dec_cfg.n_embd, eps=config.model.normalization_layer_epsilon, dtype=torch.float32, device=device)

        self.action_head = nn.Linear(dec_cfg.n_embd, data_cfg.action_vocab_size, bias=False, device=device)
        self.cb0_head = nn.Linear(dec_cfg.n_embd, data_cfg.audio_vocab_size, bias=False, device=device)

    def init_cache(self, batch_size: int, device: torch.device, max_steps: int) -> KVCache:
        heads = self.layers[0].attn.num_kv_heads
        head_dim = self.layers[0].attn.head_dim
        return KVCache.allocate(
            num_layers=len(self.layers),
            batch_size=batch_size,
            heads=heads,
            max_steps=max_steps,
            head_dim=head_dim,
            device=device,
            dtype=self.precision.compute,
        )

    def forward_step(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor,
        cache: KVCache,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, KVCache]:
        if cache is None:
            raise ValueError("Transformer cache must be initialized")

        B, C, T1 = tokens.shape
        if T1 != 1:
            raise ValueError("forward_step expects sequence length 1")
        num_audio_channels = max(0, C - 2)

        hidden_t = self.text_embed(tokens[:, 0, :], tokens[:, 1, :])
        for idx in range(num_audio_channels):
            audio_emb = self.audio_embeds[idx](tokens[:, idx + 2, :])
            hidden_t.add_(audio_emb)
        hidden_t = hidden_t.to(self.precision.compute)

        x = hidden_t
        for idx, layer in enumerate(self.layers):
            slot = cache.get_slot(idx)
            x, _ = layer.decode_step(x, positions, slot)

        hidden_norm = self.norm(x)
        action_logits = self.action_head(hidden_norm.to(torch.float32)).to(self.precision.logits)
        cb0_logits = self.cb0_head(hidden_norm.to(torch.float32)).to(self.precision.logits)
        return hidden_norm, action_logits, cb0_logits, cache

    def _embed(self, tokens: torch.Tensor) -> torch.Tensor:
        B, C, T1 = tokens.shape
        if T1 != 1:
            raise ValueError("_embed expects sequence length 1")
        num_audio_channels = max(0, C - 2)
        text_hidden = self.text_embed(tokens[:, 0, :], tokens[:, 1, :])
        audio_terms: list[torch.Tensor] = []
        for idx in range(num_audio_channels):
            audio_emb = self.audio_embeds[idx](tokens[:, idx + 2, :])
            audio_terms.append(audio_emb)
        hidden = text_hidden
        for term in audio_terms:
            hidden = hidden + term
        final = hidden.to(self.precision.compute)
        return final


class DecoderLayer(nn.Module):
    def __init__(
        self,
        config: DiaConfig,
        precision: Precision,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        dec = config.model.decoder
        eps = config.model.normalization_layer_epsilon
        self.pre_norm = nn.RMSNorm(dec.n_embd, eps=eps, dtype=torch.float32, device=device)
        self.attn = Attention(config, dec.n_embd, precision.compute, device=device)
        self.post_norm = nn.RMSNorm(dec.n_embd, eps=eps, dtype=torch.float32, device=device)
        self.mlp = Mlp(
            dec.n_embd,
            dec.n_hidden,
            precision.compute,
            tuple(config.model.linear.mlp_activations),
            device=device,
        )

    def decode_step(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        cache_slot,
    ) -> Tuple[torch.Tensor, object]:
        residual = x
        x_norm = self.pre_norm(x)
        attn_out, _ = self.attn(x_norm, pos, cache_slot)
        x = residual + attn_out
        residual2 = x
        x_norm2 = self.post_norm(x)
        mlp_out = self.mlp(x_norm2)
        return residual2 + mlp_out, cache_slot
