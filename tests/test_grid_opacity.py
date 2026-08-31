"""Grid ink comes from rcParams, and nothing renders it at full opacity.

grid.color is black ink dimmed by grid.alpha. ax.grid() inherits that
alpha automatically (verified), so the hazards are elsewhere: a
hardcoded alpha that OVERRIDES the rcParam, and non-gridline artists
(axhline, spines) that inherit nothing and would render solid black.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgba

import publiplots as pp


def _visible_gridlines(ax):
    lines = list(ax.xaxis.get_gridlines()) + list(ax.yaxis.get_gridlines())
    return [l for l in lines if l.get_visible()]


def _minor_gridlines(ax):
    """Minor gridlines are reached through the ticks, not get_gridlines().

    ``Axis.get_gridlines()`` returns major gridlines only and rejects a
    ``which=`` kwarg (TypeError), so go via the minor ticks.
    """
    ticks = list(ax.xaxis.get_minor_ticks()) + list(ax.yaxis.get_minor_ticks())
    return [t.gridline for t in ticks if t.gridline.get_visible()]


# ---- pp.add_grid ------------------------------------------------------------

def test_add_grid_matches_grid_rcparams():
    fig, ax = pp.subplots(1, 1, axes_size=(40, 30))
    pp.add_grid(ax, axis="both")
    lines = _visible_gridlines(ax)
    assert lines, "expected visible gridlines"
    for line in lines:
        assert line.get_alpha() == pytest.approx(plt.rcParams["grid.alpha"])
        assert to_rgba(line.get_color()) == to_rgba(plt.rcParams["grid.color"])
        assert line.get_linewidth() == pytest.approx(plt.rcParams["grid.linewidth"])


def test_add_grid_is_never_opaque():
    fig, ax = pp.subplots(1, 1, axes_size=(40, 30))
    pp.add_grid(ax, axis="both")
    for line in _visible_gridlines(ax):
        alpha = line.get_alpha()
        assert alpha is not None, "gridline has no alpha -> renders solid"
        assert alpha < 1.0


# ---- dot heatmap: separators AND spines ------------------------------------

@pytest.fixture
def dot_heatmap_ax():
    """The hardcoded separators live in _draw_dot_heatmap, which is only
    reached when size= is passed. A plain pp.heatmap() does not go there."""
    long_df = pd.DataFrame({
        "row": list("aaabbbccc"),
        "col": list("xyz") * 3,
        "value": np.arange(9.0),
        "size": np.arange(1, 10.0),
    })
    ax = pp.heatmap(data=long_df, x="col", y="row", value="value", size="size")
    ax.get_figure().canvas.draw()
    return ax


def test_dot_heatmap_separators_use_grid_rcparams(dot_heatmap_ax):
    lines = _minor_gridlines(dot_heatmap_ax)
    assert lines, "expected minor gridlines between cells"
    for line in lines:
        assert to_rgba(line.get_color()) == to_rgba(plt.rcParams["grid.color"])
        assert line.get_linewidth() == pytest.approx(plt.rcParams["grid.linewidth"])
        assert line.get_alpha() == pytest.approx(plt.rcParams["grid.alpha"])


def test_dot_heatmap_spines_are_not_opaque_black(dot_heatmap_ax):
    """A spine inherits nothing from grid.alpha, so the alpha has to be
    composited into its colour or the cell-matrix border lands solid."""
    for spine in dot_heatmap_ax.spines.values():
        rgba = to_rgba(spine.get_edgecolor())
        assert rgba[3] < 1.0, "spine drawn at full opacity"


# ---- upset: gridlines and the separator axhline ---------------------------

@pytest.fixture
def upset_figure():
    plt.close("all")
    pp.upsetplot(data={"A": {1, 2, 3, 4}, "B": {3, 4, 5}, "C": {4, 5, 6}})
    fig = plt.gcf()
    fig.canvas.draw()
    return fig


def test_upset_gridlines_use_grid_alpha(upset_figure):
    """The hardcoded alpha=0.3 used to override the rcParam."""
    found = False
    for ax in upset_figure.get_axes():
        for line in _visible_gridlines(ax):
            found = True
            assert line.get_alpha() == pytest.approx(plt.rcParams["grid.alpha"])
    assert found, "expected gridlines on the upset bar axes"


def test_upset_set_separators_are_not_opaque(upset_figure):
    """The separators are axhlines, which inherit no grid alpha at all."""
    grid_rgba = to_rgba(plt.rcParams["grid.color"])
    separators = [
        l for ax in upset_figure.get_axes() for l in ax.lines
        if to_rgba(l.get_color())[:3] == grid_rgba[:3]
    ]
    assert separators, "expected set-separator lines in the grid colour"
    for line in separators:
        alpha = line.get_alpha()
        assert alpha is not None, "separator axhline has no alpha -> solid black"
        assert alpha == pytest.approx(plt.rcParams["grid.alpha"])


# ---- venn ------------------------------------------------------------------

def test_venn_labels_are_not_scaled_above_body_type():
    """Set labels used to be drawn at font.size * 1.2, breaking flat 7pt.

    Note pp.venn takes `sets=`, not `data=`. Before the fix this figure
    reports two distinct text sizes (font.size and font.size * 1.2); after
    it, exactly one.
    """
    plt.close("all")
    ax = pp.venn(sets={"A": {1, 2, 3}, "B": {2, 3, 4}})
    ax.get_figure().canvas.draw()
    sizes = {round(t.get_fontsize(), 3) for t in ax.texts}
    assert sizes, "expected text on the venn diagram"
    # A set of pytest.approx objects is unhashable, so compare the single
    # expected size instead of the set literal.
    assert len(sizes) == 1 and sorted(sizes)[0] == pytest.approx(
        plt.rcParams["font.size"]
    ), (
        f"venn draws text at {sorted(sizes)}; flat type wants only "
        f"{plt.rcParams['font.size']}"
    )
