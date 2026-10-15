from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from ..config import DiaConfig
from .cache import KVCache
from .depformer import Depformer
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
