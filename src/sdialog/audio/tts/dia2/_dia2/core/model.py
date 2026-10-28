from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from ..config import DiaConfig
from .cache import KVCache
from .depformer import Depformer
from .layers import RotaryEmbedding
from .precision import Precision
from .transformer import TransformerDecoder


@dataclass
class DecodeState:
    transformer: KVCache
    depformer: KVCache


class Dia2Model(nn.Module):
    def __init__(
        self,
        config: DiaConfig,
        precision: Precision,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.config = config
        self.precision = precision
        self.transformer = TransformerDecoder(config, precision, device)
        self.depformer = Depformer(config, precision, device)
        self._cast_norms_to_compute()

    def ensure_position_capacity(self, max_position: int) -> int:
        """Grow every rotary table so absolute positions up to ``max_position`` are valid.

        ``max_context_steps`` bounds how many steps are *generated*, but with a
        voice-clone prefix the generation loop runs *absolute* positions from
        ``prefix_len - 1`` up to ``prefix_len + max_context_steps - 2``. The rotary
        table is indexed by that absolute position, so it must cover the whole
        range, not just ``max_context_steps``.
        """
        grown = 0
        for module in self.modules():
            if isinstance(module, RotaryEmbedding) and module.ensure_capacity(max_position):
                grown += 1
        self._position_capacity = max(
            int(getattr(self, "_position_capacity", 0)), int(max_position) + 1
        )
        return grown

    @property
    def position_capacity(self) -> int:
        """Largest absolute position the rotary tables can currently serve."""
        return int(getattr(self, "_position_capacity", self.config.runtime.max_context_steps))

    def init_state(self, batch_size: int, device: torch.device, max_steps: int) -> DecodeState:
        transformer_cache = self.transformer.init_cache(batch_size, device, max_steps)
        depformer_cache = self.depformer.init_cache(batch_size, device, self.depformer.num_depth)
        return DecodeState(transformer_cache, depformer_cache)

    def step_text(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor,
        state: DecodeState,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, action, cb0, cache = self.transformer.forward_step(tokens, positions, state.transformer)
        state.transformer = cache
        return hidden, action, cb0

    def step_audio_stage(
        self,
        stage_index: int,
        prev_audio: torch.Tensor,
        transformer_hidden: torch.Tensor,
        state: DecodeState,
        main_text: Optional[torch.Tensor],
        second_text: Optional[torch.Tensor],
    ) -> torch.Tensor:
        cache = state.depformer
        logits, new_cache = self.depformer.forward_step(
            prev_audio,
            transformer_hidden,
            stage_index,
            cache,
            main_text,
            second_text,
        )
        state.depformer = new_cache
        return logits

    def _cast_norms_to_compute(self) -> None:
        """Cast RMSNorm weights/biases to the compute dtype to avoid bf16 warnings."""
        def _convert(module: nn.Module) -> None:
            if isinstance(module, nn.RMSNorm):
                module.to(self.precision.compute)

        self.apply(_convert)
