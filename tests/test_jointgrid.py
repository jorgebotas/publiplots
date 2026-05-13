"""Tests for pp.JointGrid and pp.jointplot."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import publiplots as pp
from publiplots.utils.legend_entries import get_entries, is_continuous_hue


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def df():
    rng = np.random.default_rng(0)
    n = 300
    return pd.DataFrame({
        "x": rng.normal(0.0, 1.0, n),
        "y": rng.normal(0.0, 1.0, n),
    })


# ---- Construction ----

def test_construction_exposes_attributes(df):
    g = pp.JointGrid(data=df, x="x", y="y")
    assert isinstance(g.fig, Figure)
    assert isinstance(g.ax_joint, Axes)
    assert isinstance(g.ax_marg_x, Axes)
    assert isinstance(g.ax_marg_y, Axes)
    assert g.data is df
    assert g.x == "x"
    assert g.y == "y"


def test_corner_cell_hidden(df):
    g = pp.JointGrid(data=df, x="x", y="y")
    managed = {g.ax_joint, g.ax_marg_x, g.ax_marg_y}
    corner_axes = [ax for ax in g.fig.axes if ax not in managed]
    assert len(corner_axes) == 1, (
        f"expected exactly one non-managed (corner) axes; got {len(corner_axes)}"
    )
    assert corner_axes[0].get_visible() is False


# ---- Dimensions ----

def test_default_dimensions(df):
    g = pp.JointGrid(data=df, x="x", y="y")  # height=80, ratio=5
    layout = g.fig._publiplots_layout
    # ratio=5: main = 5/6 * 80 = 66.667 mm; marg = 1/6 * 80 = 13.333 mm.
    assert layout.col_widths == pytest.approx((66.667, 13.333), abs=0.01)
    # Row order: row 0 = thin top marginal; row 1 = tall main cell.
    assert layout.row_heights == pytest.approx((13.333, 66.667), abs=0.01)


def test_custom_height_and_ratio(df):
    g = pp.JointGrid(data=df, x="x", y="y", height=60, ratio=3)
    layout = g.fig._publiplots_layout
    # ratio=3: main = 3/4 * 60 = 45 mm; marg = 1/4 * 60 = 15 mm.
    assert layout.col_widths == pytest.approx((45.0, 15.0), abs=0.01)
    assert layout.row_heights == pytest.approx((15.0, 45.0), abs=0.01)


def test_space_propagates_to_wspace_hspace(df):
    g = pp.JointGrid(data=df, x="x", y="y", space=5)
    layout = g.fig._publiplots_layout
    assert layout.wspace == 5.0
    assert layout.hspace == 5.0


def test_space_default_autoscales_with_height(df):
    """When `space` is None (default), it scales with height: 0.025 * height
    mm. Yields 2 mm at the default 80 mm grid, 1 mm at 40 mm, 5 mm at 200 mm.
    Lets casual users get a sensible gap at any size without thinking about
    it; explicit values stay absolute mm."""
    for height, expected in [(40.0, 1.0), (80.0, 2.0), (200.0, 5.0)]:
        g = pp.JointGrid(data=df, x="x", y="y", height=height)
        layout = g.fig._publiplots_layout
        assert layout.wspace == pytest.approx(expected, abs=1e-6), (
            f"height={height}: expected wspace={expected}, got {layout.wspace}"
        )
        assert layout.hspace == pytest.approx(expected, abs=1e-6), (
            f"height={height}: expected hspace={expected}, got {layout.hspace}"
        )


def test_explicit_space_not_autoscaled(df):
    """An explicit `space=` kwarg is taken as absolute mm, NOT scaled by
    height. This preserves cross-figure consistency for users composing
    multiple JointGrids."""
    g = pp.JointGrid(data=df, x="x", y="y", height=200, space=2)
    layout = g.fig._publiplots_layout
    assert layout.wspace == 2.0
    assert layout.hspace == 2.0


def test_negative_space_raises(df):
    with pytest.raises(ValueError, match="space must be non-negative"):
        pp.JointGrid(df, x="x", y="y", space=-1)


# ---- Validation ----

def test_non_positive_height_raises(df):
    with pytest.raises(ValueError):
        pp.JointGrid(df, x="x", y="y", height=0)


def test_ratio_less_than_one_raises(df):
    with pytest.raises(ValueError):
        pp.JointGrid(df, x="x", y="y", ratio=0)


# ---- Shared axes ----

def test_shared_axes(df):
    g = pp.JointGrid(data=df, x="x", y="y")
    g.ax_joint.set_xlim(-5, 5)
    assert g.ax_marg_x.get_xlim() == pytest.approx((-5, 5))
    g.ax_joint.set_ylim(-3, 3)
    assert g.ax_marg_y.get_ylim() == pytest.approx((-3, 3))


# ---- plot_joint / plot_marginals / plot ----

def test_plot_joint_forwards_and_returns_self(df):
    g = pp.JointGrid(data=df, x="x", y="y")
    result = g.plot_joint(pp.scatterplot)
    assert result is g
    assert len(g.ax_joint.collections) >= 1


def test_plot_marginals_draws_on_both(df):
    g = pp.JointGrid(data=df, x="x", y="y")
    result = g.plot_marginals(pp.histplot, bins=20)
    assert result is g
    assert len(g.ax_marg_x.patches) > 0
    assert len(g.ax_marg_y.patches) > 0


def test_plot_marginals_forwards_x_to_top_y_to_right(df):
    g = pp.JointGrid(data=df, x="x", y="y")
    calls = []

    def spy(**kw):
        calls.append(kw)

    g.plot_marginals(spy)
    assert len(calls) == 2
    # First call: top marginal gets x=..., no y
    assert calls[0]["x"] == "x"
    assert "y" not in calls[0]
    assert calls[0]["ax"] is g.ax_marg_x
    # Second call: right marginal gets y=..., no x
    assert calls[1]["y"] == "y"
    assert "x" not in calls[1]
    assert calls[1]["ax"] is g.ax_marg_y


def test_plot_composes_joint_and_marginals(df):
    g = pp.JointGrid(data=df, x="x", y="y")
    result = g.plot(pp.scatterplot, pp.histplot)
    assert result is g
    assert len(g.ax_joint.collections) >= 1
    assert len(g.ax_marg_x.patches) > 0
    assert len(g.ax_marg_y.patches) > 0


def test_chainable_api(df):
    g = pp.JointGrid(data=df, x="x", y="y")
    assert g.plot_joint(pp.scatterplot).plot_marginals(pp.histplot) is g


# ---- jointplot wrapper ----

def test_jointplot_kind_scatter(df):
    g = pp.jointplot(data=df, x="x", y="y", kind="scatter")
    assert isinstance(g, pp.JointGrid)
    # Scatter produces at least one PathCollection on the main panel.
    assert len(g.ax_joint.collections) >= 1
    # Histograms on both marginals.
    assert len(g.ax_marg_x.patches) > 0
    assert len(g.ax_marg_y.patches) > 0


def test_jointplot_kind_hex(df):
    g = pp.jointplot(data=df, x="x", y="y", kind="hex")
    assert isinstance(g, pp.JointGrid)
    # Hexbin produces a PolyCollection on the joint axes...
    assert len(g.ax_joint.collections) >= 1
    # ...and stashes a continuous-hue legend entry.
    entries = get_entries(g.ax_joint)
    assert len(entries) == 1
    assert entries[0].kind == "hue"
    assert is_continuous_hue(entries[0].handles)


def test_jointplot_kind_kde(df):
    """``kind='kde'`` puts a 2D contour on the joint and 1D KDE curves
    on the marginals — both via :func:`pp.kdeplot`."""
    g = pp.jointplot(data=df, x="x", y="y", kind="kde")
    assert isinstance(g, pp.JointGrid)
    # 2D contours land in collections (QuadContourSet → multiple
    # PathCollection / LineCollection entries).
    assert len(g.ax_joint.collections) >= 1
    # 1D KDE on marginals draws a Line2D per axes.
    assert len(g.ax_marg_x.lines) >= 1
    assert len(g.ax_marg_y.lines) >= 1


def test_jointplot_kind_reg(df):
    """``kind='reg'`` puts a regression line + scatter on the joint and
    histograms on the marginals."""
    g = pp.jointplot(data=df, x="x", y="y", kind="reg")
    assert isinstance(g, pp.JointGrid)
    # regplot draws the scatter PathCollection and a regression Line2D.
    assert len(g.ax_joint.collections) >= 1
    assert len(g.ax_joint.lines) >= 1
    # Marginals: histograms (Rectangle patches).
    assert len(g.ax_marg_x.patches) > 0
    assert len(g.ax_marg_y.patches) > 0


def test_jointplot_kind_resid(df):
    """``kind='resid'`` puts a residual scatter on the joint and
    histograms on the marginals."""
    g = pp.jointplot(data=df, x="x", y="y", kind="resid")
    assert isinstance(g, pp.JointGrid)
    assert len(g.ax_joint.collections) >= 1
    assert len(g.ax_marg_x.patches) > 0
    assert len(g.ax_marg_y.patches) > 0


def test_jointplot_kind_hist_raises(df):
    """``kind='hist'`` is deferred until publiplots ships a 2D
    histogram primitive."""
    with pytest.raises(ValueError) as exc_info:
        pp.jointplot(data=df, x="x", y="y", kind="hist")
    msg = str(exc_info.value)
    assert "hist" in msg
    # Message should list the available kinds.
    assert "scatter" in msg
    assert "hex" in msg


def test_jointplot_unknown_kind_raises(df):
    with pytest.raises(ValueError):
        pp.jointplot(data=df, x="x", y="y", kind="unknown")


def test_jointplot_passes_through_height_and_ratio(df):
    g = pp.jointplot(data=df, x="x", y="y", kind="scatter", height=60, ratio=3)
    layout = g.fig._publiplots_layout
    assert layout.col_widths == pytest.approx((45.0, 15.0), abs=0.01)


def test_jointplot_forwards_kwargs_to_both(df):
    g = pp.jointplot(data=df, x="x", y="y", kind="scatter", alpha=0.5)
    # publiplots' scatterplot bakes `alpha` into the facecolor's 4th channel
    # rather than calling collection.set_alpha(); verify via the RGBA channel.
    scatter_fc = g.ax_joint.collections[0].get_facecolor()
    scatter_alphas = np.asarray(scatter_fc)[..., 3].ravel()
    assert np.any(np.isclose(scatter_alphas, 0.5, atol=1e-6)), (
        f"expected scatter facecolor alpha=0.5; got {scatter_alphas}"
    )
    # Histogram patches should be drawn with alpha applied — check face alpha.
    marg_x_alphas = [p.get_facecolor()[3] for p in g.ax_marg_x.patches]
    marg_y_alphas = [p.get_facecolor()[3] for p in g.ax_marg_y.patches]
    assert any(a == pytest.approx(0.5, abs=1e-6) for a in marg_x_alphas), (
        f"expected a marg_x patch with alpha=0.5; got {marg_x_alphas}"
    )
    assert any(a == pytest.approx(0.5, abs=1e-6) for a in marg_y_alphas), (
        f"expected a marg_y patch with alpha=0.5; got {marg_y_alphas}"
    )


# ---- Colorbar routing ----


def test_hexbin_colorbar_routes_to_marg_y_anchor(df):
    """A hexbin's stashed continuous-hue entry is claimed by the
    pre-attached band anchored on ``ax_marg_y`` (right marginal column
    grows its ``right`` reservation), NOT the main-panel column's
    ``right`` reservation (which must stay 0). This keeps the main
    panel and right marginal flush against each other."""
    g = pp.JointGrid(data=df, x="x", y="y")
    g.plot_joint(pp.hexbinplot, gridsize=20)
    g.plot_marginals(pp.histplot, bins=20)
    g.fig.canvas.draw()  # settle the auto-layout reactor

    layout = g.fig._publiplots_layout
    # right is per-column; col 0 (joint↔marg_y gap) MUST stay 0;
    # col 1 (right marginal's outside edge) absorbs the colorbar.
    assert layout.right[0] == pytest.approx(0.0, abs=1e-3), (
        f"main-panel column's right reservation must stay 0 to keep the "
        f"joint↔right-marginal gap tight; got right[0]={layout.right[0]}"
    )
    assert layout.right[1] > 0, (
        f"right marginal's column should absorb the colorbar via its "
        f"`right` reservation; got right[1]={layout.right[1]}"
    )


def test_joint_marginal_gaps_are_symmetric(df):
    """The joint↔top-marginal gap (vertical) and joint↔right-marginal gap
    (horizontal) must match within tight tolerance after the auto-layout
    reactor settles.

    Without the per-position reservation lock in ``pp.subplots`` (passed
    by ``JointGrid.__init__``), matplotlib's ``get_tightbbox`` reports
    ~1.5 mm of baseline padding for the inside-facing marginal edges
    even when ticks/labels are hidden. That padding gets baked into
    ``xlabel_space[0]`` and ``ylabel_space[1]`` asymmetrically (the row
    side picks up ~1.5 mm, the col side picks up ~0.78 mm), producing a
    visually noticeable ~0.7 mm asymmetry. The lock pins both inside-
    facing edges to 0 mm so both gaps collapse to ``space``.
    """
    g = pp.JointGrid(data=df, x="x", y="y")
    g.plot_joint(pp.scatterplot)
    g.plot_marginals(pp.histplot, bins=20)
    g.fig.canvas.draw()

    layout = g.fig._publiplots_layout
    # Vertical gap between top-marg (row 0) and joint (row 1):
    #   xlabel_space[0] (below row 0) + hspace + title_space[1] (above row 1)
    gap_top = layout.xlabel_space[0] + layout.hspace + layout.title_space[1]
    # Horizontal gap between joint (col 0) and right-marg (col 1):
    #   right[0] + wspace + ylabel_space[1]
    gap_right = layout.right[0] + layout.wspace + layout.ylabel_space[1]
    assert abs(gap_top - gap_right) < 0.5, (
        f"joint↔marginal gaps must be symmetric within 0.5 mm; "
        f"got gap_top={gap_top:.3f} mm, gap_right={gap_right:.3f} mm "
        f"(asymmetry={abs(gap_top - gap_right):.3f} mm)"
    )
    # Both gaps should equal `space` (the default 2 mm at height=80) once
    # the inside-facing reservations are locked to 0.
    assert gap_top == pytest.approx(layout.hspace, abs=0.05)
    assert gap_right == pytest.approx(layout.wspace, abs=0.05)


@pytest.mark.parametrize("height", [40.0, 80.0, 200.0])
def test_joint_marginal_gaps_symmetric_across_heights(df, height):
    """Symmetry must hold at any default-`space` JointGrid height."""
    g = pp.JointGrid(data=df, x="x", y="y", height=height)
    g.plot_joint(pp.scatterplot)
    g.plot_marginals(pp.histplot, bins=20)
    g.fig.canvas.draw()

    layout = g.fig._publiplots_layout
    gap_top = layout.xlabel_space[0] + layout.hspace + layout.title_space[1]
    gap_right = layout.right[0] + layout.wspace + layout.ylabel_space[1]
    assert abs(gap_top - gap_right) < 0.5, (
        f"height={height}: gap_top={gap_top:.3f}, gap_right={gap_right:.3f}"
    )


def test_scatter_no_colorbar_costs_nothing(df):
    """With no colorbar (scatter), the pre-attached legend band must not
    grow the figure. The right marginal column's ``right`` reservation
    stays at 0 mm."""
    g = pp.JointGrid(data=df, x="x", y="y")
    g.plot_joint(pp.scatterplot)
    g.plot_marginals(pp.histplot, bins=20)
    g.fig.canvas.draw()

    layout = g.fig._publiplots_layout
    assert layout.right == pytest.approx((0.0, 0.0), abs=1e-3), (
        f"scatter + no legend should leave right reservations at 0; "
        f"got {layout.right}"
    )


def test_jointgrid_violinplot_marginals(df):
    from matplotlib.collections import PolyCollection
    g = pp.JointGrid(data=df, x="x", y="y")
    g.plot_joint(pp.scatterplot)
    g.plot_marginals(pp.violinplot)
    px = [c for c in g.ax_marg_x.collections if isinstance(c, PolyCollection)]
    py = [c for c in g.ax_marg_y.collections if isinstance(c, PolyCollection)]
    assert len(px) >= 1
    assert len(py) >= 1


def test_jointgrid_boxplot_marginals(df):
    from matplotlib.patches import PathPatch
    g = pp.JointGrid(data=df, x="x", y="y")
    g.plot_joint(pp.scatterplot)
    g.plot_marginals(pp.boxplot)
    nx = sum(1 for p in g.ax_marg_x.patches if isinstance(p, PathPatch))
    ny = sum(1 for p in g.ax_marg_y.patches if isinstance(p, PathPatch))
    assert nx >= 1
    assert ny >= 1
