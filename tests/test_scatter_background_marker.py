"""Tests for scatterplot's opt-in ``background_marker`` argument."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgba

import publiplots as pp
from publiplots.utils.legend_entries import get_entries


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _df(seed=0, n=30):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "x": rng.normal(size=n),
        "y": rng.normal(size=n),
        "g": rng.choice(["A", "B", "C"], size=n),
        "h": rng.uniform(0, 1, size=n),
        "s": rng.uniform(1, 5, size=n),
        "style": rng.choice(["p", "q"], size=n),
    })


def test_default_adds_no_background_layer():
    ax = pp.scatterplot(data=_df(), x="x", y="y")
    assert len(ax.collections) == 1


def test_background_marker_false_adds_no_background_layer():
    ax = pp.scatterplot(data=_df(), x="x", y="y", background_marker=False)
    assert len(ax.collections) == 1


def test_background_marker_true_adds_white_layer_below_foreground():
    ax = pp.scatterplot(data=_df(), x="x", y="y", background_marker=True)
    assert len(ax.collections) == 2

    fg, bg = ax.collections[0], ax.collections[1]
    # Background sits below foreground via zorder (insertion order preserved
    # so foreground remains ax.collections[0] for cmap/label wiring).
    assert bg.get_zorder() < fg.get_zorder()

    # Background face is solid white, no edge.
    np.testing.assert_allclose(bg.get_facecolors()[0], to_rgba("white"))
    assert bg.get_linewidths()[0] == 0

    # Geometry matches.
    np.testing.assert_allclose(bg.get_offsets(), fg.get_offsets())
    np.testing.assert_allclose(bg.get_sizes(), fg.get_sizes())


def test_background_marker_custom_color():
    ax = pp.scatterplot(
        data=_df(), x="x", y="y", background_marker="#eeeeee",
    )
    assert len(ax.collections) == 2
    bg = ax.collections[1]
    np.testing.assert_allclose(bg.get_facecolors()[0], to_rgba("#eeeeee"))


def test_background_marker_with_categorical_hue_preserves_legend():
    df = _df()
    ax = pp.scatterplot(
        data=df, x="x", y="y", hue="g", palette="pastel",
        background_marker=True,
    )
    assert len(ax.collections) == 2
    names_kinds = [(e.name, e.kind) for e in get_entries(ax)]
    assert ("g", "hue") in names_kinds
    assert ax.collections[0].get_label() == "g"


def test_background_marker_with_continuous_hue_preserves_cmap():
    df = _df()
    ax = pp.scatterplot(
        data=df, x="x", y="y", hue="h", palette="viridis",
        hue_norm=(0, 1), background_marker=True,
    )
    fg = ax.collections[0]
    assert fg.get_cmap() is not None
    assert fg.norm is not None
    bg = ax.collections[1]
    assert bg.get_zorder() < fg.get_zorder()


def test_background_marker_with_style_copies_per_point_paths():
    df = _df()
    ax = pp.scatterplot(
        data=df, x="x", y="y", style="style",
        background_marker=True,
    )
    # Seaborn emits one PathCollection carrying per-point paths for style=;
    # the background must mirror those paths so each point gets the correct
    # marker shape underneath.
    assert len(ax.collections) == 2
    fg, bg = ax.collections[0], ax.collections[1]
    fg_paths = fg.get_paths()
    bg_paths = bg.get_paths()
    assert len(fg_paths) == len(bg_paths) == len(df)
    for fp, bp in zip(fg_paths, bg_paths):
        np.testing.assert_allclose(fp.vertices, bp.vertices)
    np.testing.assert_allclose(bg.get_facecolors()[0], to_rgba("white"))


def test_background_marker_with_numeric_size_matches_sizes():
    df = _df()
    ax = pp.scatterplot(
        data=df, x="x", y="y", size="s", sizes=(20, 200),
        background_marker=True,
    )
    assert len(ax.collections) == 2
    fg, bg = ax.collections[0], ax.collections[1]
    np.testing.assert_allclose(bg.get_sizes(), fg.get_sizes())
