"""
Tests for the dialog.to_audio() synthesis engine (sdialog.audio.synthesis and helpers).

All tests run on CPU without downloading models: a tone generator stands in for the TTS engine,
and Dia2's model calls are replaced by a fake.
"""
import os
import sys
import json

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("soundfile")

from sdialog import Dialog, Turn  # noqa: E402
from sdialog.audio import effects, list_presets, register_preset  # noqa: E402
from sdialog.audio.presets import PRESETS, resolve_scene  # noqa: E402
from sdialog.audio.synthesis import to_audio, to_audio_batch  # noqa: E402
from sdialog.audio.timeline import place_turns  # noqa: E402
from sdialog.audio.tts import BaseDialogTTS, PerTurnTTS, TurnAudio  # noqa: E402
from sdialog.audio.tts.base import BaseTTS  # noqa: E402
from sdialog.audio.voice_database import VoiceDatabase  # noqa: E402
from sdialog.audio.voices import normalize_age, normalize_gender, resolve_voices  # noqa: E402

SR = 16000
FREQS = (220.0, 330.0)


class ToneTTS(BaseDialogTTS):
    """Fake dialogue TTS: one tone per speaker; two turns per window with a 0.2 s natural gap."""
    supports_random_voices = True
    supports_reference_audio = True

    def __init__(self):
        self.calls = []

    def synthesize(self, turns, voices, seed=None):
        self.calls.append({"turns": turns, "voices": voices, "seed": seed})
        sr, out, offset = 24000, [], 0.0
        for ix, (speaker, text) in enumerate(turns):
            n = int(sr * (0.4 + 0.1 * len(text.split())))
            t = np.arange(n) / sr
            x = 0.3 * np.sin(2 * np.pi * FREQS[speaker] * t)
            x = np.concatenate([np.zeros(2400), x, np.zeros(2400)]).astype(np.float32)
            if ix % 2 == 0:
                offset = 0.0
            words = [(w, 0.1 + k * 0.1) for k, w in enumerate(text.split())]
            out.append(TurnAudio(x, sr, window=ix // 2, offset=offset, words=words))
            offset += len(x) / sr + 0.2
        return out


@pytest.fixture
def dialog():
    return Dialog(turns=[
        Turn(speaker="Agent", text="Hello, how can I help you today?"),
        Turn(speaker="Caller", text="Hi, my internet has been down since this morning."),
        Turn(speaker="Agent", text="I am sorry to hear that, let me check your line."),
        Turn(speaker="Caller", text="Thank you."),
        Turn(speaker="Agent", text="It should work again in a few minutes."),
        Turn(speaker="Caller", text="Great, have a nice day!"),
    ], personas={"Agent": {"gender": "female", "age": "middle-aged"}, "Caller": {"gender": "m", "age": 30}})


def _render(dialog, **kwargs):
    kwargs.setdefault("voices", "random")
    kwargs.setdefault("seed", 0)
    return to_audio(dialog, tts=ToneTTS(), sample_rate=SR, **kwargs)


def _band_energy_db(x, sr, low, high):
    spectrum = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / sr)
    band = spectrum[(freqs >= low) & (freqs < high)].sum()
    return 10 * np.log10(band / spectrum.sum() + 1e-20)


# ------------------------------------------------------------------------------------------------------- annotations
def test_segments_match_audio(dialog):
    audio = _render(dialog)
    assert audio.audio.ndim == 1 and audio.audio.dtype == np.float32
    assert len(audio.segments) == len(dialog.turns)
    assert [s.speaker for s in audio.segments] == [t.speaker for t in dialog.turns]
    assert [s.text for s in audio.segments] == [t.text for t in dialog.turns]
    for seg in audio.segments:
        inside = audio.audio[int(seg.start * SR) + 800:int(seg.end * SR) - 800]
        assert np.sqrt(np.mean(inside ** 2)) > 0.01  # speech where the segment says
        assert seg.words and seg.start <= seg.words[0][1] <= seg.end
    for prev, seg in zip(audio.segments, audio.segments[1:]):
        gap = audio.audio[int(prev.end * SR) + 400:int(seg.start * SR) - 400]
        if len(gap):
            assert np.max(np.abs(gap)) < 1e-3  # silence between non-overlapping turns (clean preset)
    assert audio.duration == pytest.approx(len(audio.audio) / SR)


def test_rttm_and_json(dialog, tmp_path):
    audio = _render(dialog)
    lines = audio.to_rttm(file_id="d1").strip().split("\n")
    assert len(lines) == len(dialog.turns)
    fields = lines[1].split()
    assert fields[0] == "SPEAKER" and fields[1] == "d1" and fields[7] == "Caller"
    assert float(fields[3]) == pytest.approx(audio.segments[1].start, abs=1e-3)
    assert float(fields[4]) == pytest.approx(audio.segments[1].duration, abs=1e-3)

    paths = audio.save(str(tmp_path / "out" / "dialog"))
    assert all(os.path.exists(p) for p in paths.values())
    data = json.loads(open(paths["json"]).read())
    assert data["sample_rate"] == SR and data["speakers"] == ["Agent", "Caller"]
    assert data["config"]["seed"] == 0
    assert data["segments"][0]["text"] == dialog.turns[0].text
    import soundfile as sf
    x, sr = sf.read(paths["audio"])
    assert sr == SR and len(x) == len(audio.audio)


def test_path_argument_saves_files(dialog, tmp_path):
    _render(dialog, path=str(tmp_path / "d.wav"))
    assert {"d.wav", "d.json", "d.rttm"} <= set(os.listdir(tmp_path))


# ----------------------------------------------------------------------------------------------------- reproducibility
def test_same_seed_same_audio(dialog):
    a = _render(dialog, preset="hard", seed=7)
    b = _render(dialog, preset="hard", seed=7)
    c = _render(dialog, preset="hard", seed=8)
    np.testing.assert_array_equal(a.audio, b.audio)
    assert a.config == b.config
    assert not np.array_equal(a.audio, c.audio)


def test_random_seed_is_recorded(dialog):
    audio = _render(dialog, seed=None)
    again = _render(dialog, seed=audio.config["seed"])
    np.testing.assert_array_equal(audio.audio, again.audio)


# ------------------------------------------------------------------------------------------------------------ stereo
@pytest.mark.parametrize("preset", sorted(PRESETS))
def test_every_preset_mono_and_stereo(dialog, preset):
    mono = _render(dialog, preset=preset)
    stereo = _render(dialog, preset=preset, stereo=True)
    assert stereo.audio.shape == (len(mono.audio), 2)
    assert [(s.start, s.end) for s in mono.segments] == [(s.start, s.end) for s in stereo.segments]
    assert [s.channel for s in stereo.segments] == [0, 1, 0, 1, 0, 1]
    assert all(s.channel is None for s in mono.segments)
    assert np.max(np.abs(stereo.audio)) <= 1.0 and np.max(np.abs(mono.audio)) <= 1.0


def test_stereo_channels_hold_one_speaker_each(dialog):
    audio = _render(dialog, stereo=True)
    for seg in audio.segments:
        a, b = int(seg.start * SR) + 800, int(seg.end * SR) - 800
        own, other = audio.audio[a:b, seg.channel], audio.audio[a:b, 1 - seg.channel]
        assert np.sqrt(np.mean(own ** 2)) > 0.01
        assert np.max(np.abs(other)) < 1e-3


def test_mono_is_sum_of_stereo_for_speaker_specific_effects(dialog):
    kwargs = dict(preset="call_center", seed=3)  # call_center only has per-speaker noise and channel
    mono = _render(dialog, **kwargs)
    stereo = _render(dialog, stereo=True, **kwargs)
    mix = stereo.audio.sum(axis=1)
    # outputs are soft-limited, so compare where the limiter is inactive
    safe = (np.abs(mix) < 0.85) & (np.abs(mono.audio) < 0.85)
    np.testing.assert_allclose(mono.audio[safe], mix[safe], atol=1e-5)


# ---------------------------------------------------------------------------------------------------- timing/overlap
def test_timeline_no_self_overlap_and_overlap_rate():
    rng = np.random.default_rng(0)
    speakers = [k % 2 for k in range(2000)]
    durations = list(rng.uniform(0.5, 4.0, size=2000))
    starts, gaps = place_turns(speakers, durations, rng, pauses=0.4, overlap=0.3, max_overlap=0.3)
    ends = [s + d for s, d in zip(starts, durations)]
    for k in range(2, len(starts)):
        assert starts[k] >= ends[k - 2] + 0.1 - 1e-9  # same speaker as turn k-2
        assert starts[k] > starts[k - 1]
    overlaps = [g < 0 for g in gaps[1:]]
    assert np.mean(overlaps) == pytest.approx(0.3, abs=0.05)
    for k in range(1, len(gaps)):
        assert -gaps[k] <= 0.3 * durations[k - 1] + 1e-9


def test_no_overlap_by_default_and_pause_range():
    rng = np.random.default_rng(1)
    starts, gaps = place_turns([0, 1] * 50, [1.0] * 100, rng, pauses=(0.2, 0.5))
    assert all(0.2 - 1e-9 <= g <= 0.5 + 1e-9 for g in gaps[1:])


def test_natural_pauses_keep_engine_timing(dialog):
    audio = _render(dialog)  # ToneTTS: 0.2 s between turns of the same window (+ trimmed silence)
    gap_in_window = audio.segments[1].start - audio.segments[0].end
    assert 0.2 <= gap_in_window <= 0.45


def test_overlap_in_audio(dialog):
    audio = _render(dialog, overlap=1.0, seed=2)
    overlaps = [s.start < p.end for p, s in zip(audio.segments, audio.segments[1:])]
    assert any(overlaps)


@pytest.mark.parametrize("bad", ["fast", (0.5, 0.1), -1])
def test_invalid_pauses(dialog, bad):
    with pytest.raises(ValueError):
        _render(dialog, pauses=bad)


# ------------------------------------------------------------------------------------------------------------ effects
def test_telephone_removes_high_frequencies():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(SR * 2).astype(np.float32) * 0.1
    y = effects.telephone(x, SR, "g711")
    assert len(y) == len(x)
    assert _band_energy_db(y, SR, 4000, 8000) < -30
    assert _band_energy_db(x, SR, 4000, 8000) > -5
    assert _band_energy_db(effects.telephone(x, SR, "amr-wb"), SR, 4000, 7000) > -15  # wideband codec
    with pytest.raises(ValueError):
        effects.telephone(x, SR, "nope")


def test_noise_snr():
    rng = np.random.default_rng(0)
    t = np.arange(SR * 3) / SR
    x = (0.1 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    for kind in ("white", "pink", "brown"):
        y, info = effects.add_noise(x, SR, rng, snr_db=10, kind=kind)
        noise = y - x
        snr = effects.active_level_db(x, SR) - 10 * np.log10(np.mean(noise.astype(np.float64) ** 2))
        assert snr == pytest.approx(10, abs=1.0)
        assert info == {"type": kind, "snr_db": 10.0}


def test_noise_from_file(tmp_path):
    import soundfile as sf
    rng = np.random.default_rng(0)
    sf.write(tmp_path / "n.wav", rng.standard_normal(SR // 2).astype(np.float32) * 0.2, SR)
    x = np.ones(SR * 2, dtype=np.float32) * 0.1
    y, info = effects.add_noise(x, SR, rng, snr_db=(5, 15), kind=str(tmp_path))
    assert len(y) == len(x) and info["file"].endswith("n.wav") and 5 <= info["snr_db"] <= 15


def test_rir_keeps_onsets_and_level():
    pytest.importorskip("pyroomacoustics")
    rng = np.random.default_rng(0)
    x = np.zeros(SR * 2, dtype=np.float32)
    x[SR // 2:SR] = np.sin(2 * np.pi * 300 * np.arange(SR // 2) / SR)
    rirs, info = effects.simulate_rirs(rng, SR, size="medium", n_sources=2)
    assert len(rirs) == 2 and info["type"] == "simulated" and 0.4 <= info["rt60"] <= 0.7
    y = effects.apply_rir(x, rirs[0], SR)
    assert len(y) >= len(x)  # reverberation tail is kept
    onset = np.argmax(np.abs(y) > 0.05 * np.max(np.abs(y)))
    assert abs(onset - SR // 2) < 0.005 * SR  # direct path aligned: onset does not move
    assert np.sqrt(np.mean(y ** 2)) == pytest.approx(np.sqrt(np.mean(x ** 2)), rel=1e-3)


def test_rir_from_file(dialog, tmp_path):
    import soundfile as sf
    rir = np.zeros(SR // 4, dtype=np.float32)
    rir[100], rir[900] = 1.0, 0.5
    sf.write(tmp_path / "rir.wav", rir, SR)
    audio = _render(dialog, reverb=str(tmp_path / "rir.wav"))
    assert audio.config["room"]["type"] == "file"


def test_reverb_per_speaker(dialog):
    pytest.importorskip("pyroomacoustics")
    audio = _render(dialog, reverb={"Caller": "small"}, stereo=True)
    assert audio.config["room"][0] is None and audio.config["room"][1]["size"] == "small"


def test_level_jitter(dialog):
    audio = _render(dialog, level_jitter_db=6)
    offsets = list(audio.config["level_offsets_db"].values())
    assert all(abs(o) <= 3 for o in offsets) and offsets[0] != offsets[1]


def test_trim_and_levels():
    x = np.concatenate([np.zeros(SR), np.ones(SR) * 0.5, np.zeros(SR)]).astype(np.float32)
    y, start = effects.trim_silence(x, SR, pad=0.0)
    assert len(y) == SR and start == SR
    assert effects.active_level_db(effects.set_active_level(x, SR, -26), SR) == pytest.approx(-26, abs=0.01)
    limited = effects.soft_limit(np.array([2.0, -3.0, 0.5, 0.95]))
    assert np.max(np.abs(limited)) <= 1.0 and limited[2] == 0.5 and 0.9 < limited[3] < 0.95


# ------------------------------------------------------------------------------------------------------------ presets
def test_presets_and_overrides(dialog):
    assert {"clean", "conversational", "room", "call_center", "far_field", "hard"} <= set(list_presets())
    scene = resolve_scene("hard", channel=False, overlap=0.0)
    assert scene["channel"] is False and scene["overlap"] == 0.0 and scene["reverb"] == "large"
    audio = _render(dialog, preset="hard", reverb=False, channel=False, noise=False)
    assert audio.config["room"] is None and audio.config["noise"] is None and audio.config["channel"] is None
    with pytest.raises(ValueError):
        _render(dialog, preset="does_not_exist")


def test_register_preset(dialog):
    register_preset("test_phone", channel="g729", pauses=0.5)
    try:
        audio = _render(dialog, preset="test_phone")
        assert audio.config["channel"] == {"all": "g729"}
        with pytest.raises(ValueError):
            register_preset("bad", stereo=True)  # output options are not scene parameters
    finally:
        PRESETS.pop("test_phone", None)


def test_call_center_phone_only_on_caller(dialog):
    # broadband (white) noise on both sides, so that the phone band-pass is measurable on the caller only
    white = {"type": "white", "snr": 10}
    audio = _render(dialog, preset="call_center", stereo=True, noise={"Agent": white, "Caller": white})
    agent, caller = audio.audio[:, 0], audio.audio[:, 1]
    assert audio.config["channel"]["Agent"] is None and audio.config["channel"]["Caller"] in ("g711", "amr")
    assert _band_energy_db(agent, SR, 4000, 8000) > -10
    assert _band_energy_db(caller, SR, 4000, 8000) < _band_energy_db(agent, SR, 4000, 8000) - 20


# ------------------------------------------------------------------------------------------------------------- voices
def _voice_db():
    return VoiceDatabase([
        {"identifier": f"{g}_{a}_{k}", "gender": g, "age": a, "voice": f"/voices/{g}_{a}_{k}.wav",
         "language": "english", "language_code": "a"}
        for g in ("male", "female") for a in (25, 45, 70) for k in range(2)
    ])


def test_db_voices_match_personas(dialog):
    tts = ToneTTS()
    audio = to_audio(dialog, tts=tts, voices="db", voice_database=_voice_db(), seed=0, sample_rate=SR)
    agent, caller = audio.voices["Agent"], audio.voices["Caller"]
    assert agent["source"] == "db" and agent["gender"] == "female" and agent["age"] == 45  # "middle-aged"
    assert caller["gender"] == "male" and caller["age"] == 25
    assert tts.calls[0]["voices"] == [f"/voices/{agent['id']}.wav", f"/voices/{caller['id']}.wav"]


def test_db_voices_are_distinct():
    db = VoiceDatabase([{"identifier": f"v{k}", "gender": "male", "age": 30, "voice": f"v{k}",
                         "language": "english", "language_code": "a"} for k in range(2)])
    turns = [(0, "hi"), (1, "hello")]
    for seed in range(10):
        voices, info = resolve_voices(["A", "B"], {}, turns, "db", ToneTTS(), np.random.default_rng(seed),
                                      voice_database=db)
        assert voices[0] != voices[1]


def test_random_and_explicit_voices(dialog):
    tts = ToneTTS()
    audio = to_audio(dialog, tts=tts, voices={"Caller": "/my/voice.wav"}, seed=0, sample_rate=SR)
    assert tts.calls[0]["voices"] == [None, "/my/voice.wav"]
    assert audio.voices == {"Agent": {"source": "random"}, "Caller": {"source": "user", "id": "/my/voice.wav"}}
    with pytest.raises(ValueError):
        to_audio(dialog, tts=ToneTTS(), voices="nope")


def test_design_voices(dialog):
    pytest.importorskip("torch")
    tts = ToneTTS()
    audio = to_audio(dialog, tts=tts, voices="design", seed=0, sample_rate=SR,
                     voice_description="A {{ gender }} voice")
    assert audio.voices["Agent"] == {"source": "design", "description": "A female voice"}
    assert all(isinstance(v, tuple) and len(v) == 2 for v in tts.calls[0]["voices"])


def test_age_and_gender_normalization():
    rng = np.random.default_rng(0)
    assert normalize_age(40, rng) == 40
    assert normalize_age("35 years old", rng) == 35
    assert normalize_age("in her 30s", rng) == 30
    assert normalize_age("elderly", rng) == 75
    assert 20 <= normalize_age("", rng) <= 65 and 20 <= normalize_age(None, rng) <= 65
    assert normalize_gender("F", rng) == "female" and normalize_gender("Man", rng) == "male"
    assert normalize_gender("", rng) in ("male", "female")


# ---------------------------------------------------------------------------------------------------- engines / misc
class FakeClassicTTS(BaseTTS):
    def generate(self, text, speaker_voice, tts_pipeline_kwargs={}):
        return np.ones(8000, dtype=np.float32) * 0.1, 8000


def test_classic_tts_is_wrapped(dialog):
    audio = to_audio(dialog, tts=FakeClassicTTS(), voices={"Agent": "af_1", "Caller": "am_1"}, seed=0,
                     sample_rate=SR)
    assert audio.config["tts"] == {"engine": "PerTurnTTS", "wrapped": "FakeClassicTTS"}
    assert len(audio.segments) == len(dialog.turns)
    assert PerTurnTTS(FakeClassicTTS()).default_voice_database == "sdialog/voices-kokoro"
    with pytest.raises(ValueError):  # classic engines cannot make up voices
        to_audio(dialog, tts=FakeClassicTTS(), voices="random")


@pytest.mark.parametrize("speakers", [["A"], ["A", "B", "C"]])
def test_requires_two_speakers(speakers):
    d = Dialog(turns=[Turn(speaker=s, text="Hello there.") for s in speakers])
    with pytest.raises(ValueError, match="exactly two speakers"):
        to_audio(d, tts=ToneTTS())


def test_empty_turns_are_skipped():
    d = Dialog(turns=[Turn(speaker="A", text="Hi."), Turn(speaker="B", text="  "), Turn(speaker="B", text="Hey.")])
    audio = to_audio(d, tts=ToneTTS(), voices="random", seed=0)
    assert [s.turn for s in audio.segments] == [0, 2]


def test_dialog_to_audio_method(dialog):
    audio = dialog.to_audio(tts=ToneTTS(), voices="random", preset="room", stereo=True, seed=0)
    assert audio.audio.shape[1] == 2 and audio.config["preset"] == "room"


def test_batch(dialog, tmp_path):
    rows = to_audio_batch([dialog, dialog], str(tmp_path), devices=["cpu"], tts_factory=_tone_factory,
                          seed=0, voices="random")
    assert [r["index"] for r in rows] == [0, 1] and all("error" not in r for r in rows)
    assert len(open(tmp_path / "index.jsonl").read().strip().split("\n")) == 2
    assert rows[0]["seed"] != rows[1]["seed"]
    single = to_audio(dialog, tts=ToneTTS(), voices="random", seed=rows[1]["seed"])
    import soundfile as sf
    np.testing.assert_allclose(sf.read(rows[1]["audio"])[0], single.audio, atol=1e-4)


def _tone_factory(device):
    return ToneTTS()


# --------------------------------------------------------------------------------------------------------------- Dia2
def test_dia2_requires_python_310(monkeypatch):
    from sdialog.audio.tts.dia2 import tts as dia2_tts
    monkeypatch.setattr(dia2_tts.sys, "version_info", (3, 9, 18))
    with pytest.raises(RuntimeError, match="requires Python >= 3.10"):
        dia2_tts.check_python_version()


@pytest.mark.skipif(sys.version_info < (3, 10), reason="Dia2 requires Python >= 3.10")
class TestDia2Windows:
    """Dia2TTS windowing / cutting logic, with the model call replaced by a fake."""

    @staticmethod
    def _tts(timestamps_ok=True, lead=0.0, **kwargs):
        from sdialog.audio.tts.dia2 import Dia2TTS
        tts = Dia2TTS(**kwargs)
        tts._load = lambda: None
        tts.scripts = []
        sr = 24000

        def fake_generate(script, prefix_1, prefix_2, seed):
            tts.scripts.append((script, prefix_1, prefix_2))
            words = [w for w in script.split() if w not in ("[S1]", "[S2]")]
            n = int((lead + 0.3 * len(words)) * sr)
            wav = np.random.default_rng(len(words)).standard_normal(n).astype(np.float32)
            ts = [(w, lead + 0.3 * k) for k, w in enumerate(words)]
            return wav, sr, ts if timestamps_ok else ts[:-1]

        tts._generate = fake_generate
        return tts

    def test_window_cut_points(self):
        tts = self._tts()
        turns = [(0, "one two three"), (1, "four five"), (0, "six")]
        clips = tts.synthesize(turns, [None, None], seed=0)
        assert len(tts.scripts) == 1
        assert tts.scripts[0][0] == "[S1] one two three [S2] four five [S1] six"
        durations = [len(c.audio) / c.sample_rate for c in clips]
        assert durations[0] == pytest.approx(0.9 - 0.08, abs=1e-3)
        assert clips[1].offset == pytest.approx(0.9 - 0.08, abs=1e-3)
        assert [w for w, _ in clips[1].words] == ["four", "five"]
        assert clips[1].words[0][1] == pytest.approx(0.08, abs=1e-3)
        assert all(c.window == 0 for c in clips)

    def test_audio_before_first_word_is_dropped(self):
        # e.g. the end of a reference voice prompt leaking into the output before the first word
        tts = self._tts(lead=0.5)
        clips = tts.synthesize([(0, "one two"), (1, "three")], ["voice_A.wav", "voice_B.wav"])
        assert clips[0].offset == pytest.approx(0.5 - 0.08, abs=1e-3)
        assert clips[0].words[0][1] == pytest.approx(0.08, abs=1e-3)
        assert len(clips[0].audio) / 24000 == pytest.approx(0.6, abs=1e-3)
        fallback = self._tts(timestamps_ok=False, lead=0.5)
        clips = fallback.synthesize([(0, "one two"), (1, "three four")], [None, None])
        assert clips[0].words[0][1] == pytest.approx(0.08, abs=1e-3)

    def test_trailing_speech_is_trimmed(self):
        from sdialog.audio.tts.dia2 import Dia2TTS
        sr = 24000
        rng = np.random.default_rng(0)
        speech, short_pause = rng.standard_normal(sr).astype(np.float32), np.zeros(int(0.15 * sr), np.float32)
        pause, extra = np.zeros(int(0.5 * sr), np.float32), rng.standard_normal(sr // 2).astype(np.float32)
        audio = np.concatenate([speech, short_pause, speech, pause, extra])  # "... last word <pause> Okay."
        words = [("first", 0.0), ("last", 1.6)]
        trimmed = Dia2TTS._trim_tail(audio, words, sr)
        assert len(trimmed) == pytest.approx(len(speech) * 2 + len(short_pause) + 0.1 * sr, abs=0.03 * sr)
        assert len(Dia2TTS._trim_tail(audio[:len(trimmed)], words, sr)) == len(trimmed)  # nothing more to cut
        assert len(Dia2TTS._trim_tail(audio, [], sr)) == len(audio)

    def test_windows_and_voice_continuity(self):
        tts = self._tts(max_window_words=4)  # fake audio: 0.3 s per word
        turns = [(0, "a b c d"), (0, "e f g"), (1, "h i j k l m n o"), (1, "p"), (0, "q r")]
        tts.synthesize(turns, [None, None], seed=0)
        scripts = [s for s, _, _ in tts.scripts]
        assert scripts == ["[S1] a b c d", "[S1] e f g", "[S1] h i j k l m n o", "[S1] p [S2] q r"]
        # speaker 0 has only 1.2 s of audio after the first window: still random in the second one
        assert tts.scripts[1][1] is None
        # [S1] is always the first speaker of each window; random voices are reused once >= 2 s were generated
        # (across windows), so in the last window both speakers have a reference
        _, p1, p2 = tts.scripts[3]
        assert isinstance(p1, tuple) and isinstance(p2, tuple)
        assert len(p1[0]) == pytest.approx(0.3 * 24000 * 8, abs=2)  # speaker 1 (S1 here): its 8-word turn
        assert len(p2[0]) == pytest.approx(0.3 * 24000 * 7, abs=2)  # speaker 0: its first two turns

    def test_reference_voices_follow_speaker_mapping(self):
        tts = self._tts()
        tts.synthesize([(1, "hi there"), (0, "hello")], ["voice_A.wav", "voice_B.wav"])
        assert tts.scripts[0][1:] == ("voice_B.wav", "voice_A.wav")

    def test_fallback_when_timestamps_mismatch(self):
        tts = self._tts(timestamps_ok=False)
        clips = tts.synthesize([(0, "one two"), (1, "three four")], [None, None], seed=0)
        assert len(tts.scripts) == 3  # one window + one call per turn
        assert [c.window for c in clips] == [0, 1]

    def test_more_than_two_speakers(self):
        with pytest.raises(ValueError):
            self._tts().synthesize([(0, "a"), (2, "b")], [None, None, None])


@pytest.mark.skipif(sys.version_info < (3, 10), reason="Dia2 requires Python >= 3.10")
def test_vendored_dia2_imports():
    for module in ("torch", "transformers", "safetensors", "sphn"):
        pytest.importorskip(module)
    from sdialog.audio.tts.dia2._dia2 import Dia2, GenerationConfig  # noqa: F401
    from sdialog.audio.tts.dia2._dia2.runtime import voice_clone
    assert voice_clone.WHISPER_MODEL_NAME
