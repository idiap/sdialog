from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class DataConfig:
    channels: int
    text_vocab_size: int
    audio_vocab_size: int
    action_vocab_size: int
    text_pad_token_id: int
    text_new_word_token_id: int
    text_zero_token_id: int
    audio_pad_token_id: int
    audio_bos_token_id: int
    action_pad_token_id: int
    action_new_word_token_id: int
    delay_pattern: List[int]
    first_word_min_start: int
    max_pad: int
    second_stream_ahead: int
    tokenizer_path: Optional[str] = None


@dataclass(frozen=True)
class DecoderConfig:
    n_layer: int
    n_embd: int
    n_hidden: int
    gqa_query_heads: int
    kv_heads: int
    gqa_head_dim: int
    dropout: float
    low_rank_dim: int | None = None


@dataclass(frozen=True)
class DepformerConfig:
    n_layer: int
    n_embd: int
    n_hidden: int
    gqa_query_heads: int
    kv_heads: int
    gqa_head_dim: int
    apply_rope: bool
    text_embedding: bool
    mlp_activations: List[str]


@dataclass(frozen=True)
class LinearHeadConfig:
    mlp_activations: List[str]


@dataclass(frozen=True)
class ModelConfig:
    decoder: DecoderConfig
    depformer: DepformerConfig
    linear: LinearHeadConfig
    dropout: float
    rope_min_timescale: int
    rope_max_timescale: int
    normalization_layer_epsilon: float


@dataclass(frozen=True)
class RuntimeConfig:
    weights_schedule: List[int]
    max_context_steps: int


@dataclass(frozen=True)
class AssetsConfig:
    tokenizer: Optional[str]
    mimi: Optional[str]


@dataclass(frozen=True)
class DiaConfig:
    data: DataConfig
    model: ModelConfig
    runtime: RuntimeConfig
    assets: AssetsConfig


def _resolve_runtime(block: dict | None, data_cfg: DataConfig) -> RuntimeConfig:
    block = block or {}
    weights_schedule = block.get("weights_schedule")
    if weights_schedule is None:
        audio_channels = max(0, data_cfg.channels - 2)
        weights_schedule = list(range(max(audio_channels - 1, 0)))
    max_context = block.get("max_context_steps", 1500)
    return RuntimeConfig(
        weights_schedule=list(weights_schedule),
        max_context_steps=int(max_context),
    )


def load_config(path: str | Path) -> DiaConfig:
    cfg = json.loads(Path(path).read_text())
    data = cfg["data"]
    model = cfg["model"]
    runtime_cfg_raw = cfg.get("runtime")
    if runtime_cfg_raw is None:
        raise ValueError(f"Config '{path}' is missing a runtime block")

    decoder_cfg = DecoderConfig(
        n_layer=model["decoder"]["n_layer"],
        n_embd=model["decoder"]["n_embd"],
        n_hidden=model["decoder"]["n_hidden"],
        gqa_query_heads=model["decoder"]["gqa_query_heads"],
        kv_heads=model["decoder"]["kv_heads"],
        gqa_head_dim=model["decoder"]["gqa_head_dim"],
        dropout=model.get("dropout", 0.0),
        low_rank_dim=model["decoder"].get("low_rank_dim"),
    )

    depformer_cfg = DepformerConfig(
        n_layer=model["depformer"]["n_layer"],
        n_embd=model["depformer"]["n_embd"],
        n_hidden=model["depformer"]["n_hidden"],
        gqa_query_heads=model["depformer"]["gqa_query_heads"],
        kv_heads=model["depformer"]["kv_heads"],
        gqa_head_dim=model["depformer"]["gqa_head_dim"],
        apply_rope=model["depformer"].get("apply_rope", True),
        text_embedding=model["depformer"].get("text_embedding", True),
        mlp_activations=model["depformer"].get("mlp_activations", ["silu", "linear"]),
    )

    data_cfg = DataConfig(
        channels=data["channels"],
        text_vocab_size=data["text_vocab_size"],
        audio_vocab_size=data["audio_vocab_size"],
        action_vocab_size=data["action_vocab_size"],
        text_pad_token_id=data["text_pad_token_id"],
        text_new_word_token_id=data["text_new_word_token_id"],
        text_zero_token_id=data.get("text_zero_token_id", 7),
        audio_pad_token_id=data.get("audio_pad_token_id", data["audio_vocab_size"] - 1),
        audio_bos_token_id=data.get("audio_bos_token_id", data["audio_vocab_size"] - 2),
        action_pad_token_id=data["action_pad_token_id"],
        action_new_word_token_id=data["action_new_word_token_id"],
        delay_pattern=list(data.get("delay_pattern", [])),
        first_word_min_start=data.get("first_word_min_start", 0),
        max_pad=data.get("max_pad", 0),
        second_stream_ahead=data.get("second_stream_ahead", 0),
        tokenizer_path=data.get("tokenizer_path"),
    )

    runtime_cfg = _resolve_runtime(runtime_cfg_raw, data_cfg)

    linear_cfg = LinearHeadConfig(
        mlp_activations=model.get("linear", {}).get("mlp_activations", ["silu", "linear"]),
    )

    model_cfg = ModelConfig(
        decoder=decoder_cfg,
        depformer=depformer_cfg,
        linear=linear_cfg,
        dropout=model.get("dropout", 0.0),
        rope_min_timescale=model.get("rope_min_timescale", 1),
        rope_max_timescale=model.get("rope_max_timescale", 10000),
        normalization_layer_epsilon=model.get("normalization_layer_epsilon", 1e-5),
    )

    assets_raw = cfg.get("assets") or {}
    assets_cfg = AssetsConfig(
        tokenizer=assets_raw.get("tokenizer") or data_cfg.tokenizer_path,
        mimi=assets_raw.get("mimi"),
    )

    return DiaConfig(
        data=data_cfg,
        model=model_cfg,
        runtime=runtime_cfg,
        assets=assets_cfg,
    )
