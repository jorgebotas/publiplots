"""Regression tests for issue #105: barplot face color ignored when hatch= is set.

Before the fix, ``pp.barplot(color="#43adaa", hatch="enc", hatch_map=...)`` would
render bars with a black (or palette-cycled) face because the face color was
inferred from ``patch.get_edgecolor()`` in a post-hoc pass that was order- and
matplotlib-version-dependent. The fix rewrites the paint pass to set face,
hatch, and edge in one deterministic step driven by ``BarSplitSpec``.

These tests assert only the RGB channels of face/edge colors — alpha varies
with the publiplots ``alpha`` rcParam.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.colors import to_rgba

import publiplots as pp


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def df():
    return pd.DataFrame([
        ("a", "1D", 0.80), ("a", "2D", 0.82),
        ("b", "1D", 0.75), ("b", "2D", 0.79),
    ], columns=["row", "enc", "val"])


def _rgb(color) -> tuple:
    return to_rgba(color)[:3]


def _face_rgb(patch) -> tuple:
    return tuple(patch.get_facecolor()[:3])


def _edge_rgb(patch) -> tuple:
    return tuple(patch.get_edgecolor()[:3])


def _bars(ax):
    return [p for p in ax.patches if hasattr(p, "get_height")]


def test_color_honored_with_hatch_no_hue(df):
    """color=<hex> + hatch=<col> without hue: every bar face is <hex>."""
    teal = _rgb("#43adaa")
    ax = pp.barplot(
        data=df, x="val", y="row",
        color="#43adaa",
        hatch="enc",
        hatch_map={"1D": "", "2D": "///"},
    )
    bars = _bars(ax)
    assert len(bars) == 4
    for i, p in enumerate(bars):
        assert _face_rgb(p) == pytest.approx(teal, abs=1e-6), (
            f"bar {i}: expected face {teal}, got {_face_rgb(p)}"
        )


def test_color_honored_with_hatch_uniform_palette_hue(df):
    """hue=<col> + palette={k: color, ...} uniform + hatch=<col>: all bars color."""
    teal = _rgb("#43adaa")
    ax = pp.barplot(
        data=df, x="val", y="row",
        hue="enc",
        palette={"1D": "#43adaa", "2D": "#43adaa"},
        hatch="enc",
        hatch_map={"1D": "", "2D": "///"},
    )
    bars = _bars(ax)
    assert len(bars) == 4
    for i, p in enumerate(bars):
        assert _face_rgb(p) == pytest.approx(teal, abs=1e-6)
    # Hatch pattern must still distinguish the two groups.
    hatches = {p.get_hatch() for p in bars}
    assert hatches == {"", "///"}


def test_edgecolor_override_with_hatch(df):
    """edgecolor='black' + color=<hex> + hatch: face=<hex>, edge=black."""
    teal = _rgb("#43adaa")
    black = _rgb("black")
    ax = pp.barplot(
        data=df, x="val", y="row",
        color="#43adaa", edgecolor="black",
        hatch="enc",
        hatch_map={"1D": "", "2D": "///"},
    )
    bars = _bars(ax)
    assert len(bars) == 4
    for p in bars:
        assert _face_rgb(p) == pytest.approx(teal, abs=1e-6)
        assert _edge_rgb(p) == pytest.approx(black, abs=1e-6)
    # Hatch patterns still applied.
    assert {p.get_hatch() for p in bars} == {"", "///"}


def test_face_color_matches_palette_under_double_split(df):
    """Double split (hue=row, hatch=enc): each row gets its palette color."""
    teal = _rgb("#43adaa")
    red = _rgb("#b34343")
    ax = pp.barplot(
        data=df, x="val", y="row",
        hue="row",
        palette={"a": "#43adaa", "b": "#b34343"},
        hatch="enc",
        hatch_map={"1D": "", "2D": "///"},
    )
    bars = _bars(ax)
    assert len(bars) == 4
    # Every bar should match one of the two palette colors, and both colors
    # must appear — no black leakage.
    face_colors = {_face_rgb(p) for p in bars}
    assert teal in face_colors
    assert red in face_colors
    # Hatches applied.
    assert {p.get_hatch() for p in bars} == {"", "///"}
