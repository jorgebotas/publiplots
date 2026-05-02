"""Tests for the color resolution module."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgba

from publiplots.annotate._cache import BarRecord
from publiplots.annotate._color import resolve_color


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _bar(ax, facecolor="#000000", alpha=1.0, edgecolor="#000000"):
    r = ax.bar([0], [1.0])[0]
    r.set_facecolor(to_rgba(facecolor, alpha=alpha))
    r.set_edgecolor(to_rgba(edgecolor))
    return r


def test_auto_dark_fill_inside_returns_light():
    fig, ax = plt.subplots()
    r = _bar(ax, facecolor="#000000")
    bar = BarRecord(patch=r, value=1.0, err_low=None, err_high=None, hue_color=None)
    rgba = resolve_color(bar, color="auto", anchor="inside", ax=ax)
    assert rgba == to_rgba("#ffffff")


def test_auto_light_fill_inside_returns_dark():
    fig, ax = plt.subplots()
    r = _bar(ax, facecolor="#ffffff")
    bar = BarRecord(patch=r, value=1.0, err_low=None, err_high=None, hue_color=None)
    rgba = resolve_color(bar, color="auto", anchor="inside", ax=ax)
    # text.color rcParam is dark by default; compare via rgba match
    assert rgba != to_rgba("#ffffff")


def test_auto_translucent_on_white_bg_returns_dark():
    """A saturated palette color with alpha=0.1 composited onto white should be near-white → dark text."""
    fig, ax = plt.subplots()
    ax.set_facecolor("white")
    r = _bar(ax, facecolor="#2a5ea6", alpha=0.1)  # vivid blue, mostly transparent
    bar = BarRecord(patch=r, value=1.0, err_low=None, err_high=None, hue_color=None)
    rgba = resolve_color(bar, color="auto", anchor="center", ax=ax)
    assert rgba != to_rgba("#ffffff")


def test_auto_outside_ignores_fill_uses_rcparam():
    fig, ax = plt.subplots()
    r = _bar(ax, facecolor="#000000")
    bar = BarRecord(patch=r, value=1.0, err_low=None, err_high=None, hue_color=None)
    expected = to_rgba(plt.rcParams["text.color"])
    assert resolve_color(bar, color="auto", anchor="outside", ax=ax) == expected


def test_hue_returns_hue_color_when_set():
    fig, ax = plt.subplots()
    r = _bar(ax, facecolor="#112233")
    hue = to_rgba("#ff8800")
    bar = BarRecord(patch=r, value=1.0, err_low=None, err_high=None, hue_color=hue)
    assert resolve_color(bar, color="hue", anchor="outside", ax=ax) == hue


def test_literal_color_passes_through():
    fig, ax = plt.subplots()
    r = _bar(ax, facecolor="#ffffff")
    bar = BarRecord(patch=r, value=1.0, err_low=None, err_high=None, hue_color=None)
    assert resolve_color(bar, color="#ff0000", anchor="inside", ax=ax) == to_rgba("#ff0000")
    assert resolve_color(bar, color=(0, 1, 0), anchor="inside", ax=ax) == to_rgba((0, 1, 0))


def test_auto_on_translucent_dark_fill_returns_light():
    """A very dark fill at alpha=1 should still yield light text."""
    fig, ax = plt.subplots()
    ax.set_facecolor("white")
    r = _bar(ax, facecolor="#111111", alpha=1.0)
    bar = BarRecord(patch=r, value=1.0, err_low=None, err_high=None, hue_color=None)
    assert resolve_color(bar, color="auto", anchor="inside", ax=ax) == to_rgba("#ffffff")


def test_hue_with_hue_active_false_warns_and_falls_back_to_auto():
    """color='hue' on a no-hue plot should warn and behave like auto."""
    fig, ax = plt.subplots()
    ax.set_facecolor("white")
    r = _bar(ax, facecolor="#000000")
    bar = BarRecord(patch=r, value=1.0, err_low=None, err_high=None,
                    hue_color=tuple(to_rgba("#000000")))
    with pytest.warns(UserWarning, match="plot has no hue"):
        rgba = resolve_color(bar, color="hue", anchor="inside", ax=ax,
                             hue_active=False)
    # Dark bar inside anchor → light text under auto semantics.
    assert rgba == to_rgba("#ffffff")
