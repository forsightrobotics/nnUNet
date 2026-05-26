"""Tests for the 2D-batched fast path in predict_sliding_window_return_logits.

The 2D batched path is gated on ``len(configuration_manager.patch_size) == 2``
and collapses the per-tile sliding-window loop into a single (or chunked)
batched forward pass. These tests pin down:

  * Existence of the ``_internal_predict_2d_batched`` helper.
  * The batched path collapses N per-tile forwards into one (or fewer).
  * Equality between batched and legacy outputs (no TTA, no overlap).
  * Equality with in-plane overlap (``tile_step_size < 1.0``).
  * Equality with mirror-axis TTA enabled.
  * Chunking via ``OCT3DSEG_2D_BATCH_CAP`` env var produces the same result.
  * 3D configs bypass the batched path.
  * Single-slicer degenerate case still produces correct output.
  * Invalid ``OCT3DSEG_2D_BATCH_CAP`` env var falls back to the default.

The instance flag ``_use_batched_2d_path`` lets the test force the legacy loop
on the same predictor for a side-by-side comparison.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from acvl_utils.cropping_and_padding.padding import pad_nd_image

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


def _make_predictor_2d(
    patch_size=(32, 32),
    num_classes=3,
    in_channels=1,
    tta=False,
    tile_step=1.0,
):
    """Construct a minimally configured predictor with a tiny 2D Conv1x1 network."""
    p = nnUNetPredictor(
        tile_step_size=tile_step,
        use_gaussian=True,
        use_mirroring=tta,
        perform_everything_on_device=False,
        device=torch.device("cpu"),
        verbose=False,
        allow_tqdm=False,
    )
    network = nn.Conv2d(in_channels, num_classes, kernel_size=1, bias=False)
    nn.init.normal_(network.weight)
    network.eval()
    p.network = network
    p.configuration_manager = SimpleNamespace(patch_size=patch_size)
    p.label_manager = SimpleNamespace(num_segmentation_heads=num_classes)
    p.allowed_mirroring_axes = (0, 1) if tta else None
    return p


def _make_predictor_3d(patch_size=(16, 16, 16), num_classes=3, in_channels=1, tile_step=1.0):
    p = nnUNetPredictor(
        tile_step_size=tile_step,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=False,
        device=torch.device("cpu"),
        verbose=False,
        allow_tqdm=False,
    )
    network = nn.Conv3d(in_channels, num_classes, kernel_size=1, bias=False)
    nn.init.normal_(network.weight)
    network.eval()
    p.network = network
    p.configuration_manager = SimpleNamespace(patch_size=patch_size)
    p.label_manager = SimpleNamespace(num_segmentation_heads=num_classes)
    p.allowed_mirroring_axes = None
    return p


def _pad_and_slice(predictor, data):
    padded, _ = pad_nd_image(
        data,
        predictor.configuration_manager.patch_size,
        "constant",
        {"value": 0},
        True,
        None,
    )
    slicers = predictor._internal_get_sliding_window_slicers(padded.shape[1:])
    return padded, slicers


def _run_internal(predictor, padded_data, slicers, do_on_device=True):
    return predictor._internal_predict_sliding_window_return_logits(
        padded_data, slicers, do_on_device=do_on_device,
    )


# ---- Existence canary -------------------------------------------------

def test_batched_2d_helper_exists():
    assert hasattr(nnUNetPredictor, "_internal_predict_2d_batched"), (
        "Missing helper _internal_predict_2d_batched on nnUNetPredictor"
    )


# ---- Behavioural: the batched path actually batches -------------------

def test_batched_2d_collapses_per_tile_loop_into_single_forward():
    torch.manual_seed(0)
    p = _make_predictor_2d(patch_size=(32, 32), tile_step=1.0)
    data = torch.randn(1, 8, 32, 32)
    padded, slicers = _pad_and_slice(p, data)

    calls = []
    orig_forward = p.network.forward

    def counting_forward(x):
        calls.append(tuple(x.shape))
        return orig_forward(x)

    p.network.forward = counting_forward
    p._use_batched_2d_path = True

    _ = _run_internal(p, padded, slicers, do_on_device=True)

    assert len(calls) == 1, f"expected 1 forward pass, got {len(calls)}: {calls}"
    assert calls[0][0] == len(slicers), (
        f"expected batch size {len(slicers)}, got {calls[0][0]}"
    )


# ---- Correctness: batched vs legacy ----------------------------------

def test_batched_2d_output_equals_legacy_no_tta():
    torch.manual_seed(0)
    p_b = _make_predictor_2d(patch_size=(32, 32), tile_step=1.0)
    p_l = _make_predictor_2d(patch_size=(32, 32), tile_step=1.0)
    p_l.network.load_state_dict(p_b.network.state_dict())

    torch.manual_seed(42)
    data = torch.randn(1, 8, 32, 32)
    padded, slicers = _pad_and_slice(p_b, data)

    p_b._use_batched_2d_path = True
    p_l._use_batched_2d_path = False
    out_b = _run_internal(p_b, padded, slicers, do_on_device=True)
    out_l = _run_internal(p_l, padded, slicers, do_on_device=True)

    assert out_b.shape == out_l.shape
    assert torch.allclose(out_b.float(), out_l.float(), atol=1e-3, rtol=1e-3)


def test_batched_2d_output_equals_legacy_with_overlap():
    torch.manual_seed(1)
    p_b = _make_predictor_2d(patch_size=(16, 16), tile_step=0.5)
    p_l = _make_predictor_2d(patch_size=(16, 16), tile_step=0.5)
    p_l.network.load_state_dict(p_b.network.state_dict())

    torch.manual_seed(43)
    data = torch.randn(1, 4, 32, 32)
    padded, slicers = _pad_and_slice(p_b, data)

    p_b._use_batched_2d_path = True
    p_l._use_batched_2d_path = False
    out_b = _run_internal(p_b, padded, slicers, do_on_device=True)
    out_l = _run_internal(p_l, padded, slicers, do_on_device=True)

    assert torch.allclose(out_b.float(), out_l.float(), atol=1e-3, rtol=1e-3)


def test_batched_2d_with_tta_matches_legacy():
    torch.manual_seed(2)
    p_b = _make_predictor_2d(patch_size=(32, 32), tta=True, tile_step=1.0)
    p_l = _make_predictor_2d(patch_size=(32, 32), tta=True, tile_step=1.0)
    p_l.network.load_state_dict(p_b.network.state_dict())

    torch.manual_seed(44)
    data = torch.randn(1, 4, 32, 32)
    padded, slicers = _pad_and_slice(p_b, data)

    p_b._use_batched_2d_path = True
    p_l._use_batched_2d_path = False
    out_b = _run_internal(p_b, padded, slicers, do_on_device=True)
    out_l = _run_internal(p_l, padded, slicers, do_on_device=True)

    assert torch.allclose(out_b.float(), out_l.float(), atol=1e-3, rtol=1e-3)


# ---- Chunking via env var --------------------------------------------

def test_batch_cap_env_honoured(monkeypatch):
    torch.manual_seed(3)
    p_small = _make_predictor_2d(patch_size=(32, 32), tile_step=1.0)
    p_big = _make_predictor_2d(patch_size=(32, 32), tile_step=1.0)
    p_big.network.load_state_dict(p_small.network.state_dict())

    torch.manual_seed(45)
    data = torch.randn(1, 8, 32, 32)
    padded, slicers = _pad_and_slice(p_small, data)
    p_small._use_batched_2d_path = True
    p_big._use_batched_2d_path = True

    monkeypatch.setenv("OCT3DSEG_2D_BATCH_CAP", "2")
    out_small = _run_internal(p_small, padded, slicers, do_on_device=True)

    monkeypatch.setenv("OCT3DSEG_2D_BATCH_CAP", "100")
    out_big = _run_internal(p_big, padded, slicers, do_on_device=True)

    assert torch.allclose(out_small.float(), out_big.float(), atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("bad_value", ["abc", "", "0", "-1"])
def test_invalid_batch_cap_env_falls_back_to_default(monkeypatch, bad_value):
    torch.manual_seed(4)
    p = _make_predictor_2d(patch_size=(32, 32), tile_step=1.0)
    torch.manual_seed(46)
    data = torch.randn(1, 4, 32, 32)
    padded, slicers = _pad_and_slice(p, data)
    p._use_batched_2d_path = True

    monkeypatch.setenv("OCT3DSEG_2D_BATCH_CAP", bad_value)
    out = _run_internal(p, padded, slicers, do_on_device=True)
    assert out.shape == (3, 4, 32, 32)


# ---- Gate behaviour ---------------------------------------------------

def test_3d_config_uses_legacy_path():
    torch.manual_seed(5)
    p = _make_predictor_3d(patch_size=(16, 16, 16))
    p._use_batched_2d_path = True
    torch.manual_seed(47)
    data = torch.randn(1, 16, 16, 16)
    padded, slicers = _pad_and_slice(p, data)

    def _fail(*args, **kwargs):
        raise AssertionError("3D config should not enter the batched 2D path")

    p._internal_predict_2d_batched = _fail
    out = _run_internal(p, padded, slicers, do_on_device=True)
    assert out.shape == (3, 16, 16, 16)


def test_single_slicer_degenerate():
    torch.manual_seed(6)
    p = _make_predictor_2d(patch_size=(32, 32), tile_step=1.0)
    torch.manual_seed(48)
    data = torch.randn(1, 1, 32, 32)
    padded, slicers = _pad_and_slice(p, data)
    assert len(slicers) == 1, "test setup requires len(slicers) == 1"

    p._use_batched_2d_path = True
    out = _run_internal(p, padded, slicers, do_on_device=True)
    assert out.shape == (3, 1, 32, 32)
