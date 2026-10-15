from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import torch

from .assets import resolve_assets
from .runtime.context import RuntimeContext, build_runtime
from .runtime.generator import (
    build_initial_state,
    decode_audio,
    run_generation_loop,
    warmup_with_prefix,
)
from .runtime.script_parser import parse_script
from .audio.grid import undelay_frames, write_wav
from .runtime.voice_clone import build_prefix_plan
from .generation import (
    GenerationConfig,
    GenerationResult,
    merge_generation_config,
    normalize_script,
)
from .runtime.logger import RuntimeLogger

class Dia2:
    def __init__(
        self,
        *,
        repo: Optional[str] = None,
        config_path: Optional[str | Path] = None,
        weights_path: Optional[str | Path] = None,
        tokenizer_id: Optional[str | Path] = None,
        mimi_id: Optional[str] = None,
        device: str = "cuda",
        dtype: str = "auto",
        default_config: Optional[GenerationConfig] = None,
    ) -> None:
        bundle = resolve_assets(
            repo=repo,
            config_path=config_path,
            weights_path=weights_path,
        )
        self._config_path = bundle.config_path
        self._weights_path = bundle.weights_path
        self._tokenizer_id = (str(tokenizer_id) if tokenizer_id else None) or bundle.tokenizer_id
        self._repo_id = bundle.repo_id
        self._mimi_id = mimi_id or bundle.mimi_id
        self.device = device
        self._dtype_pref = dtype or "auto"
        self.default_config = default_config or GenerationConfig()
        self._runtime: Optional[RuntimeContext] = None

    @classmethod
    def from_repo(
        cls,
        repo: str,
        *,
        device: str = "cuda",
        dtype: str = "auto",
        tokenizer_id: Optional[str] = None,
        mimi_id: Optional[str] = None,
    ) -> "Dia2":
        return cls(repo=repo, device=device, dtype=dtype, tokenizer_id=tokenizer_id, mimi_id=mimi_id)

    @classmethod
    def from_local(
        cls,
        config_path: str | Path,
        weights_path: str | Path,
        *,
        device: str = "cuda",
        dtype: str = "auto",
        tokenizer_id: Optional[str | Path] = None,
        mimi_id: Optional[str] = None,
    ) -> "Dia2":
        return cls(
            config_path=config_path,
            weights_path=weights_path,
            tokenizer_id=tokenizer_id,
            device=device,
            dtype=dtype,
            mimi_id=mimi_id,
        )

    def set_device(self, device: str, *, dtype: Optional[str] = None) -> None:
        desired_dtype = dtype or self._dtype_pref
        if self.device == device and desired_dtype == self._dtype_pref:
            return
        self.device = device
        self._dtype_pref = desired_dtype
        self._runtime = None

    def close(self) -> None:
        self._runtime = None

    def _ensure_runtime(self) -> RuntimeContext:
        if self._runtime is None:
            self._runtime = self._build_runtime()
        return self._runtime

    def generate(
        self,
        script: str | Sequence[str],
        *,
        config: Optional[GenerationConfig] = None,
        output_wav: Optional[str | Path] = None,
        prefix_speaker_1=None,
        prefix_speaker_2=None,
        include_prefix: Optional[bool] = None,
        verbose: bool = False,
        seed: Optional[int] = None,
        **overrides,
    ):
        runtime = self._ensure_runtime()
        if seed is not None:  # [sdialog] reproducible sampling
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        logger = RuntimeLogger(verbose)
        merged_overrides = dict(overrides)
        if prefix_speaker_1 is not None:
            merged_overrides["prefix_speaker_1"] = prefix_speaker_1
        if prefix_speaker_2 is not None:
            merged_overrides["prefix_speaker_2"] = prefix_speaker_2
        if include_prefix is not None:
            merged_overrides["include_prefix"] = include_prefix
        merged = merge_generation_config(base=config or self.default_config, overrides=merged_overrides)
        max_context = runtime.config.runtime.max_context_steps
        text = normalize_script(script)
        prefix_plan = build_prefix_plan(runtime, merged.prefix)
        entries = []
        if prefix_plan is not None:
            entries.extend(prefix_plan.entries)
        entries.extend(parse_script([text], runtime.tokenizer, runtime.constants, runtime.frame_rate))
        runtime.machine.initial_padding = merged.initial_padding
        logger.event(
            f"starting generation: max_context={max_context} cfg_scale={merged.cfg_scale:.2f} "
            f"device={self.device} dtype={self._dtype_pref}"
        )
        state = runtime.machine.new_state(entries)
        cfg_active = merged.cfg_scale != 1.0
        if cfg_active:
            logger.event(f"classifier-free guidance enabled (scale={merged.cfg_scale:.2f})")
        else:
            logger.event("classifier-free guidance disabled (scale=1.0)")
        gen_state = build_initial_state(
            runtime,
            prefix=prefix_plan,
        )
        include_prefix_audio = bool(prefix_plan and merged.prefix and merged.prefix.include_audio)
        start_step = 0
        if prefix_plan is not None:
            logger.event(f"warming up with prefix ({prefix_plan.aligned_frames} frames)")
            start_step = warmup_with_prefix(runtime, prefix_plan, state, gen_state)
            if include_prefix_audio:
                logger.event("prefix audio will be kept in output")
            else:
                logger.event("prefix audio trimmed from output")
        first_word_frame, audio_buf = run_generation_loop(
            runtime,
            state=state,
            generation=gen_state,
            config=merged,
            start_step=start_step,
            logger=logger,
        )
        aligned = undelay_frames(audio_buf[0], runtime.audio_delays, runtime.constants.audio_pad).unsqueeze(0)
        crop = 0 if include_prefix_audio else max(first_word_frame, 0)
        if crop > 0 and crop < aligned.shape[-1]:
            aligned = aligned[:, :, crop:]
        elif crop >= aligned.shape[-1]:
            crop = 0
        logger.event(f"decoding {aligned.shape[-1]} Mimi frames")
        waveform = decode_audio(runtime, aligned)
        if output_wav is not None:
            write_wav(str(output_wav), waveform.detach().cpu().numpy(), runtime.mimi.sample_rate)
            duration = waveform.shape[-1] / max(runtime.mimi.sample_rate, 1)
            logger.event(f"saved {output_wav} ({duration:.2f}s)")
        frame_rate = max(runtime.frame_rate, 1.0)
        prefix_entry_count = len(prefix_plan.entries) if prefix_plan is not None else 0
        transcript_entries = state.transcript
        if prefix_plan is not None and not include_prefix_audio:
            if len(transcript_entries) > prefix_entry_count:
                transcript_entries = transcript_entries[prefix_entry_count:]
            else:
                transcript_entries = []
        timestamps = []
        for word, step in transcript_entries:
            adj = step - crop
            if adj < 0:
                continue
            timestamps.append((word, adj / frame_rate))
        logger.event(f"generation finished in {logger.elapsed():.2f}s")
        return GenerationResult(aligned, waveform, runtime.mimi.sample_rate, timestamps)

    def save_wav(self, script: str | Sequence[str], path: str | Path, **kwargs):
        return self.generate(script, output_wav=path, **kwargs)

    @property
    def sample_rate(self) -> int:
        return self._ensure_runtime().mimi.sample_rate

    @property
    def tokenizer_id(self) -> Optional[str]:
        if self._tokenizer_id:
            return self._tokenizer_id
        if self._runtime is not None:
            return getattr(self._runtime.tokenizer, "name_or_path", None)
        return self._repo_id

    @property
    def dtype(self) -> str:
        return self._dtype_pref

    @property
    def max_context_steps(self) -> int:
        return self._ensure_runtime().config.runtime.max_context_steps

    @property
    def repo(self) -> Optional[str]:
        return self._repo_id

    def _build_runtime(self) -> RuntimeContext:
        runtime, tokenizer_ref, mimi_ref = build_runtime(
            config_path=self._config_path,
            weights_path=self._weights_path,
            tokenizer_id=self._tokenizer_id,
            repo_id=self._repo_id,
            mimi_id=self._mimi_id,
            device=self.device,
            dtype_pref=self._dtype_pref,
        )
        self._tokenizer_id = tokenizer_ref
        self._mimi_id = mimi_ref
        return runtime
