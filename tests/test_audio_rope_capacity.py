"""
Regression tests for the rotary (RoPE) position-table capacity.

``max_context_steps`` bounds how many steps are *generated*, but with a voice-clone
prefix the generation loop runs *absolute* positions from ``prefix_len - 1`` up to
``prefix_len + max_context_steps - 2``. ``RotaryEmbedding.forward`` gathers
``cos_cached[position_ids]``, so the table must cover that whole absolute range --
otherwise long references index past the end and CUDA raises a device-side assert
(``index out of bounds``), which also poisons the CUDA context for every later
generation in the process.

These tests are CPU-only and need no model weights.
"""
import sys

import pytest

if sys.version_info < (3, 10):
    pytest.skip("Dia2 requires Python >= 3.10", allow_module_level=True)

torch = pytest.importorskip("torch")
# Importing anything under the vendored package runs its ``__init__``, which
# pulls in runtime.audio_io -> sphn. Skip when that optional dependency is
# absent (same guard as tests/test_audio_synthesis.py::test_vendored_dia2_imports).
pytest.importorskip("sphn")

from sdialog.audio.tts.dia2._dia2.core.layers import RotaryEmbedding  # noqa: E402


def _rope(max_seq_len=1500, head_dim=64):
    return RotaryEmbedding(head_dim, 10000, 10000, max_seq_len=max_seq_len)


def test_ensure_capacity_grows_table():
    rope = _rope(max_seq_len=1500)
    assert rope.ensure_capacity(2265) is True
    assert rope.max_seq_len == 2266
    assert rope.cos_cached.shape[0] == 2266
    assert rope.sin_cached.shape[0] == 2266


def test_ensure_capacity_is_noop_when_it_already_fits():
    rope = _rope(max_seq_len=1500)
    cos_before, sin_before = rope.cos_cached.clone(), rope.sin_cached.clone()
    assert rope.ensure_capacity(1499) is False
    assert rope.max_seq_len == 1500
    assert torch.equal(rope.cos_cached, cos_before)
    assert torch.equal(rope.sin_cached, sin_before)


def test_ensure_capacity_is_idempotent():
    rope = _rope(max_seq_len=1500)
    assert rope.ensure_capacity(2265) is True
    assert rope.ensure_capacity(2265) is False
    assert rope.max_seq_len == 2266


def test_growing_leaves_existing_rows_bit_identical():
    """Row ``i`` is ``cos(i * inv_freq)``, so extending the range must not change
    rows that already existed: generations that used to fit stay byte-identical."""
    small, big = _rope(max_seq_len=1500), _rope(max_seq_len=2266)
    assert torch.equal(small.cos_cached, big.cos_cached[:1500])
    assert torch.equal(small.sin_cached, big.sin_cached[:1500])


def test_forward_accepts_positions_past_max_context_steps():
    """The actual failure mode: a position at/after ``max_context_steps`` used to
    index past the table."""
    rope = _rope(max_seq_len=1500)
    x = torch.randn(1, 1, 1, 64)

    # before growing, position 1500 is out of range for a 1500-row table
    assert rope.cos_cached.shape[0] == 1500
    with pytest.raises(IndexError):
        rope(x, torch.tensor([[1500]]))

    assert rope.ensure_capacity(2265) is True
    out = rope(x, torch.tensor([[1500]]))  # now valid
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_model_exposes_position_capacity():
    from sdialog.audio.tts.dia2._dia2.core.model import Dia2Model

    assert hasattr(Dia2Model, "ensure_position_capacity")
    assert hasattr(Dia2Model, "position_capacity")
