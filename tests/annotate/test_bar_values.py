"""Integration tests for the bar_values strategy on real axes."""
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from publiplots.annotate._cache import BarRecord, BarValueMeta
from publiplots.annotate.bar_values import _bar_values_strategy


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# -----------------------------------------------------------------------------
# Introspection path (foreign axes, no cache)
# -----------------------------------------------------------------------------

def test_foreign_vertical_bars_produces_labels():
    fig, ax = plt.subplots()
    ax.bar([0, 1, 2], [1.0, 2.0, 3.0])
    fig.canvas.draw()
    texts = _bar_values_strategy(ax, fmt=".1f", anchor="outside", offset=0.0,
                                 color="auto", pad=0.0)
    assert len(texts) == 3
    labels = [t.get_text() for t in texts]
    assert labels == ["1.0", "2.0", "3.0"]


def test_foreign_horizontal_bars_produces_labels():
    fig, ax = plt.subplots()
    ax.barh([0, 1, 2], [1.0, 2.0, 3.0])
    fig.canvas.draw()
    texts = _bar_values_strategy(ax, fmt=".1f", anchor="outside", offset=0.0,
                                 color="auto", pad=0.0)
    assert [t.get_text() for t in texts] == ["1.0", "2.0", "3.0"]


def test_empty_axes_returns_empty_and_warns():
    fig, ax = plt.subplots()
    with pytest.warns(UserWarning, match="no bars found"):
        texts = _bar_values_strategy(ax, fmt=".1f", anchor="outside", offset=0.0,
                                     color="auto", pad=0.0)
    assert texts == []


def test_nan_values_are_skipped():
    """If a bar's height is NaN, no label is drawn for it."""
    fig, ax = plt.subplots()
    ax.bar([0, 1, 2], [1.0, 2.0, 3.0])
    # Manually install a meta with one NaN value
    rects = [p for p in ax.patches]
    bars = [
        BarRecord(patch=rects[0], value=1.0, err_low=None, err_high=None, hue_color=None),
        BarRecord(patch=rects[1], value=float("nan"), err_low=None, err_high=None, hue_color=None),
        BarRecord(patch=rects[2], value=3.0, err_low=None, err_high=None, hue_color=None),
    ]
    ax._publiplots_bar_meta = BarValueMeta(
        orient="v", bars=bars, errorbar_kind=None,
        hue_active=False, owner_is_publiplots=True,
    )
    fig.canvas.draw()
    texts = _bar_values_strategy(ax, fmt=".1f", anchor="outside", offset=0.0,
                                 color="auto", pad=0.0)
    assert len(texts) == 2
    assert [t.get_text() for t in texts] == ["1.0", "3.0"]


def test_fmt_format_string_with_braces():
    fig, ax = plt.subplots()
    ax.bar([0], [3.14159])
    fig.canvas.draw()
    texts = _bar_values_strategy(ax, fmt="{:.3f}", anchor="outside", offset=0.0,
                                 color="auto", pad=0.0)
    assert texts[0].get_text() == "3.142"


def test_fmt_bare_spec():
    fig, ax = plt.subplots()
    ax.bar([0], [3.14159])
    fig.canvas.draw()
    texts = _bar_values_strategy(ax, fmt=".3f", anchor="outside", offset=0.0,
                                 color="auto", pad=0.0)
    assert texts[0].get_text() == "3.142"


# -----------------------------------------------------------------------------
# Fit-check fallback
# -----------------------------------------------------------------------------

def test_inside_fallback_to_outside_when_text_too_big():
    fig, ax = plt.subplots()
    ax.bar([0], [0.001])  # extremely short bar
    ax.set_ylim(0, 0.01)
    fig.canvas.draw()
    texts = _bar_values_strategy(
        ax, fmt=".3f", anchor="inside", offset=0.0, color="auto", pad=0.0,
        fontsize=20,
    )
    assert len(texts) == 1
    # Label should sit at or above the bar top, not inside the 0.001 height band.
    t = texts[0]
    _, y = t.get_position()
    assert t.get_va() == "bottom"
    assert y >= 0.001


# -----------------------------------------------------------------------------
# Auto limit expansion
# -----------------------------------------------------------------------------

def test_foreign_autoscale_on_expands_past_text_extent():
    """On a pre-drawn autoscale-on axes, ylim must grow past the text top,
    not just past the bar top.

    Regression: without a fresh canvas.draw() before measuring text extents,
    get_window_extent returns stale bboxes and the computed need_max
    undershoots, so labels render clipping through the axis frame.
    """
    fig, ax = plt.subplots()
    ax.bar([0, 1, 2], [1.0, 2.4, 0.7])
    fig.canvas.draw()  # simulate the foreign-axes "already drawn" case
    texts = _bar_values_strategy(
        ax, fmt=".2f", anchor="outside", offset=1.0, color="auto", pad=1.0,
    )
    # Re-draw to materialise final text positions, then verify no text bbox
    # top exceeds the axes' data limit.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    ylim_top = ax.get_ylim()[1]
    for t in texts:
        bb = t.get_window_extent(renderer).transformed(inv)
        assert bb.y1 < ylim_top, (
            f"label {t.get_text()!r} top ({bb.y1}) clips ylim ({ylim_top})"
        )


def test_publiplots_owned_expands_limits_past_seaborn_default():
    """owner_is_publiplots=True always expands regardless of autoscale state."""
    fig, ax = plt.subplots()
    rects = ax.bar([0, 1], [10.0, 10.0])
    ax.set_ylim(0, 10)  # disables autoscale (like seaborn does)
    fig.canvas.draw()
    bars = [BarRecord(patch=r, value=10.0, err_low=None, err_high=None, hue_color=None)
            for r in rects]
    ax._publiplots_bar_meta = BarValueMeta(
        orient="v", bars=bars, errorbar_kind=None,
        hue_active=False, owner_is_publiplots=True,
    )
    _bar_values_strategy(ax, fmt=".1f", anchor="outside", offset=1.5,
                         color="auto", pad=1.0)
    _, top = ax.get_ylim()
    assert top > 10.0


def test_foreign_locked_limits_warn_on_clip():
    """Autoscale off on foreign axes → warn if labels would clip."""
    fig, ax = plt.subplots()
    ax.bar([0], [10.0])
    ax.set_ylim(0, 10)  # disables autoscale
    fig.canvas.draw()
    # Introspection → owner_is_publiplots=False
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        _bar_values_strategy(ax, fmt=".1f", anchor="outside", offset=1.5,
                             color="auto", pad=1.0)
    clip_warnings = [w for w in rec if "clipped" in str(w.message)]
    assert clip_warnings
