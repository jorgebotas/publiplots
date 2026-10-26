"""Tests for the color resolution module."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import seaborn as sns
from matplotlib.colors import to_rgba

from publiplots.annotate._cache import BarRecord
from publiplots.annotate._color import (
    contrast_ink, relative_luminance, resolve_color, white_wins,
)


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


_PALETTES = ("deep", "muted", "pastel", "bright", "dark", "colorblind", "tab10")
_WHITE = to_rgba("#ffffff")


def _ink():
    return to_rgba(plt.rcParams["text.color"])


def _sweep():
    return [rgb for pal in _PALETTES for rgb in sns.color_palette(pal)]


def test_relative_luminance_matches_seaborn():
    for rgb in _sweep():
        assert relative_luminance(rgb) == pytest.approx(
            sns.utils.relative_luminance(rgb), abs=1e-4)


def test_contrast_ink_picks_the_higher_wcag_contrast_ratio():
    for rgb in _sweep():
        y = sns.utils.relative_luminance(rgb)
        white_wins = 1.05 / (y + 0.05) > (y + 0.05) / 0.05
        got = contrast_ink(to_rgba(rgb), to_rgba("white"))
        assert got == (_WHITE if white_wins else _ink()), rgb


def test_white_wins_ignores_text_color():
    white_bg = to_rgba("white")
    with plt.rc_context({"text.color": "white"}):
        assert not white_wins(to_rgba("#5d83c3", 0.1), white_bg)
        assert white_wins(to_rgba("#0000ff"), white_bg)


@pytest.mark.parametrize("text_color", ["white", "#dddddd", "#808080"])
@pytest.mark.parametrize("fill", ["#ffff00", "#ffffff"])
def test_contrast_ink_never_returns_a_light_text_color_on_a_light_fill(
        text_color, fill):
    # dark_background sets text.color white; returning it on a light fill
    # put white on white (1.0:1) where black reads at 21:1. #808080 sits on
    # the light side of the 0.179 crossover (luminance 0.216).
    with plt.rc_context({"text.color": text_color}):
        got = contrast_ink(to_rgba(fill), to_rgba("black"))
    assert got == to_rgba("black")


def test_contrast_ink_keeps_white_on_a_dark_fill_under_light_text():
    with plt.rc_context({"text.color": "white"}):
        assert contrast_ink(to_rgba("#000080"), to_rgba("black")) == _WHITE


@pytest.mark.parametrize("text_color", ["black", "#333333", "#666666", "#1f2a44"])
def test_contrast_ink_keeps_a_dark_text_color(text_color):
    with plt.rc_context({"text.color": text_color}):
        assert contrast_ink(to_rgba("#ffff00"), to_rgba("white")) == to_rgba(text_color)


@pytest.mark.parametrize("fill, want_white", [
    ("#0000ff", True),    # HLS lightness 0.500 picked ink: contrast 2.4 vs 8.6
    ("#2ca02c", False),   # HLS lightness 0.400 picked white: 3.4 vs 6.2
    ("#5d83c3", False),   # publiplots' default colour: 5.5 vs 3.8
])
def test_auto_inside_uses_wcag_not_hls(fill, want_white):
    fig, ax = plt.subplots()
    ax.set_facecolor("white")
    r = _bar(ax, facecolor=fill, alpha=1.0)
    bar = BarRecord(patch=r, value=1.0, err_low=None, err_high=None, hue_color=None)
    got = resolve_color(bar, color="auto", anchor="inside", ax=ax)
    assert got == (_WHITE if want_white else _ink())


def test_default_alpha_fills_always_get_ink():
    """At pp.rcParams["alpha"] = 0.1 every fill composites to near-white, so
    the rule change cannot move a default-styled label."""
    for rgb in _sweep():
        assert contrast_ink(to_rgba(rgb, 0.1), to_rgba("white")) == _ink(), rgb
