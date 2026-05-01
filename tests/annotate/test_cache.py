"""Tests for the BarValueMeta cache contract and axes introspection."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from publiplots.annotate._cache import BarRecord, BarValueMeta, _introspect


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_bar_value_meta_minimal_construct():
    meta = BarValueMeta(orient="v", bars=[], errorbar_kind=None,
                        hue_active=False, owner_is_publiplots=False)
    assert meta.orient == "v"
    assert meta.bars == []
    assert meta.hue_active is False
    assert meta.owner_is_publiplots is False


def test_bar_record_fields():
    fig, ax = plt.subplots()
    rect = ax.bar([0], [1.0])[0]
    rec = BarRecord(patch=rect, value=1.0, err_low=None, err_high=None, hue_color=None)
    assert rec.value == 1.0
    assert rec.err_low is None
    assert rec.err_high is None


def test_introspect_vertical_bars_no_errorbars():
    fig, ax = plt.subplots()
    ax.bar([0, 1, 2], [1.0, 2.0, 3.0], width=0.8)
    meta = _introspect(ax)
    assert meta.orient == "v"
    assert len(meta.bars) == 3
    assert [b.value for b in meta.bars] == pytest.approx([1.0, 2.0, 3.0])
    assert all(b.err_low is None and b.err_high is None for b in meta.bars)
    assert meta.owner_is_publiplots is False
    assert meta.hue_active is False


def test_introspect_horizontal_bars_no_errorbars():
    fig, ax = plt.subplots()
    ax.barh([0, 1, 2], [1.0, 2.0, 3.0], height=0.8)
    meta = _introspect(ax)
    assert meta.orient == "h"
    assert [b.value for b in meta.bars] == pytest.approx([1.0, 2.0, 3.0])


def test_introspect_empty_axes():
    fig, ax = plt.subplots()
    meta = _introspect(ax)
    assert meta.bars == []


def test_introspect_vertical_bars_with_errorbars():
    fig, ax = plt.subplots()
    ax.bar([0, 1, 2], [1.0, 2.0, 3.0], yerr=[0.1, 0.2, 0.3], width=0.8)
    meta = _introspect(ax)
    assert len(meta.bars) == 3
    # Matplotlib's errorbar plots vertical segments whose endpoints match bar ± yerr.
    for bar, exp_low, exp_high, val in zip(
        meta.bars, [0.9, 1.8, 2.7], [1.1, 2.2, 3.3], [1.0, 2.0, 3.0]
    ):
        # err_low/err_high are full-extent y values (including the bar tip direction).
        # For positive bars, err_high is the top-cap y, err_low is the bottom-cap y.
        assert bar.err_high == pytest.approx(exp_high, rel=1e-2)
        assert bar.err_low == pytest.approx(exp_low, rel=1e-2)
