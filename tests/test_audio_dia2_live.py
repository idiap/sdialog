"""
Live end-to-end test of dialog.to_audio() with the real Dia2 model (GPU, downloads model weights).

Skipped unless SDIALOG_RUN_LIVE=1 and a CUDA GPU is available. Generated audios are kept in
$SDIALOG_LIVE_OUT (default: a pytest temporary folder) for listening.

    SDIALOG_RUN_LIVE=1 SDIALOG_LIVE_OUT=live_audios pytest -s tests/test_audio_dia2_live.py
"""
import os
import re
import sys

import numpy as np
import pytest

if os.environ.get("SDIALOG_RUN_LIVE") != "1":
    pytest.skip("live test: set SDIALOG_RUN_LIVE=1 to run it", allow_module_level=True)
if sys.version_info < (3, 10):
    pytest.skip("Dia2 requires Python >= 3.10", allow_module_level=True)
torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("live test requires a CUDA GPU", allow_module_level=True)

from sdialog import Dialog  # noqa: E402
from sdialog.audio import list_presets, to_audio  # noqa: E402

DIALOG_PATH = os.path.join(os.path.dirname(__file__), "data", "demo_dialog_doctor_patient.json")
MAX_SEGMENT_WER = 0.3


@pytest.fixture(scope="module")
def dialog():
    return Dialog.from_file(DIALOG_PATH)


@pytest.fixture(scope="module")
def out_dir(tmp_path_factory):
    path = os.environ.get("SDIALOG_LIVE_OUT") or str(tmp_path_factory.mktemp("live_audio"))
    os.makedirs(path, exist_ok=True)
    return path


@pytest.fixture(scope="module")
def asr():
    from transformers import pipeline
    return pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo",
                    device=0, torch_dtype=torch.float16)


def _normalize(text):
    return " ".join(re.sub(r"[^\w\s']", " ", text.lower()).split())


def _segment_wers(audio, asr):
    import jiwer
    x = audio.audio if audio.audio.ndim == 1 else audio.audio.sum(axis=1)
    wers = []
    for seg in audio.segments:
        a, b = int(seg.start * audio.sample_rate), int(seg.end * audio.sample_rate)
        hyp = asr({"raw": x[a:b].astype(np.float32), "sampling_rate": audio.sample_rate},
                  generate_kwargs={"language": "english"})["text"]
        wers.append(jiwer.wer(_normalize(seg.text), _normalize(hyp) or "<empty>"))
        print(f"  WER {wers[-1]:.2f} | {seg.speaker}: {seg.text}\n           -> {hyp}")
    return wers


@pytest.mark.parametrize("voices", ["random", "db"])
def test_live_dia2_voices(dialog, out_dir, asr, voices):
    audio = to_audio(dialog, voices=voices, seed=0, path=os.path.join(out_dir, f"voices_{voices}"))
    print(f"\n[{voices}] {audio.duration:.1f}s, voices={audio.voices}")
    assert len(audio.segments) == len(dialog.turns)
    assert all(seg.end > seg.start for seg in audio.segments)
    wers = _segment_wers(audio, asr)
    # the cut points are right if each segment contains (only) its own turn
    assert np.median(wers) < MAX_SEGMENT_WER, wers
    assert np.mean(np.array(wers) < 0.5) >= 0.8, wers


def test_live_dia2_reproducible(dialog):
    a = to_audio(dialog, voices="random", seed=123)
    b = to_audio(dialog, voices="random", seed=123)
    assert a.duration == pytest.approx(b.duration, abs=0.05)


@pytest.mark.parametrize("preset", sorted(list_presets()))
def test_live_presets(dialog, out_dir, preset):
    for stereo in (False, True):
        name = f"preset_{preset}_{'stereo' if stereo else 'mono'}"
        audio = to_audio(dialog, preset=preset, stereo=stereo, seed=0, path=os.path.join(out_dir, name))
        assert audio.channels == (2 if stereo else 1)
        assert np.max(np.abs(audio.audio)) <= 1.0


def test_live_design_voices(dialog, out_dir):
    qwen_tts = pytest.importorskip("qwen_tts")
    if getattr(qwen_tts.__spec__, "loader", None) is None:  # stub installed by conftest.py
        pytest.skip("qwen-tts is not installed")
    audio = to_audio(dialog, voices="design", seed=0, path=os.path.join(out_dir, "voices_design"))
    assert all(v["source"] == "design" for v in audio.voices.values())
