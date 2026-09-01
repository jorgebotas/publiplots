"""Inside-axes legend placement via ``legend_kws={'inside': True, ...}``.

Default publiplots behavior places legends outside the right edge of the
axes and re-anchors them every draw via ``LayoutReactor``. When a user
needs the seaborn/matplotlib-style inside placement (e.g., a small
legend tucked in a corner), ``legend_kws={'inside': True, 'loc':
'upper right'}`` should short-circuit that machinery and hand over to
matplotlib's native axes-relative placement.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import publiplots as pp
from publiplots.utils.layout_reactor import LayoutReactor


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "cat": np.repeat(list("ABCD"), 12),
        "val": rng.normal(size=48),
        "grp": np.tile(list("XY"), 24),
    })


def _anchor_fig_frac(legend, fig):
    """Return the legend's bbox_to_anchor in figure-fraction space."""
    bbox = legend.get_bbox_to_anchor()
    return bbox.x0 / fig.bbox.width, bbox.y0 / fig.bbox.height


def test_inside_true_places_legend_inside_axes(df):
    fig, ax = pp.subplots(axes_size=(80, 50))
    pp.barplot(
        data=df, x="cat", y="val", hue="grp", palette="pastel", ax=ax,
        legend_kws={"inside": True, "loc": "upper right"},
    )
    fig.canvas.draw()
    leg = ax.get_legend()
    assert leg is not None
    x_frac, y_frac = _anchor_fig_frac(leg, fig)
    ax_pos = ax.get_position()
    # The anchor should sit inside the axes bbox, not past the right edge.
    assert x_frac < ax_pos.x1, (
        f"inside=True but anchor x={x_frac:.3f} is at/right of ax.x1={ax_pos.x1:.3f}"
    )
    assert y_frac <= ax_pos.y1


def test_inside_true_default_loc_is_best(df):
    """Without an explicit loc, matplotlib's default 'best' is used."""
    fig, ax = pp.subplots(axes_size=(80, 50))
    pp.barplot(
        data=df, x="cat", y="val", hue="grp", palette="pastel", ax=ax,
        legend_kws={"inside": True},
    )
    fig.canvas.draw()
    leg = ax.get_legend()
    assert leg is not None
    x_frac, _ = _anchor_fig_frac(leg, fig)
    ax_pos = ax.get_position()
    # 'best' picks a corner inside the axes, not the default outside-right.
    assert x_frac < ax_pos.x1


def test_outside_default_preserved(df):
    """Sanity: the default (no inside kwarg) still anchors past the right edge."""
    fig, ax = pp.subplots(axes_size=(80, 50))
    pp.barplot(data=df, x="cat", y="val", hue="grp", palette="pastel", ax=ax)
    fig.canvas.draw()
    leg = ax.get_legend()
    assert leg is not None
    x_frac, _ = _anchor_fig_frac(leg, fig)
    ax_pos = ax.get_position()
    assert x_frac > ax_pos.x1, (
        f"default outside legend: expected x_frac > ax.x1 "
        f"({x_frac:.3f} vs {ax_pos.x1:.3f})"
    )


def test_inside_true_skips_reactor_registration(df):
    fig, ax = pp.subplots(axes_size=(80, 50))
    # Capture reactor state before the plot so we can count net additions.
    reactor = LayoutReactor.get(fig)
    before = len(reactor._registrations)
    pp.barplot(
        data=df, x="cat", y="val", hue="grp", palette="pastel", ax=ax,
        legend_kws={"inside": True, "loc": "upper right"},
    )
    after = len(reactor._registrations)
    # The inside legend must not register with the reactor.
    assert after == before, (
        f"inside=True registered {after - before} artist(s) with the reactor; "
        "reactor should be bypassed for inside-axes legends."
    )


def test_inside_true_applies_across_scatter_line_point(df):
    """Every plot that forwards legend_kws honors inside=True."""
    rng = np.random.default_rng(7)
    xy = pd.DataFrame({
        "x": rng.normal(size=40),
        "y": rng.normal(size=40),
        "g": np.tile(list("AB"), 20),
    })
    for fn in (pp.scatterplot, pp.lineplot, pp.pointplot):
        fig, ax = pp.subplots(axes_size=(60, 40))
        fn(data=xy, x="x", y="y", hue="g", palette="pastel", ax=ax,
           legend_kws={"inside": True, "loc": "upper right"})
        fig.canvas.draw()
        leg = ax.get_legend()
        assert leg is not None, f"{fn.__name__}: no legend produced"
        x_frac, _ = _anchor_fig_frac(leg, fig)
        ax_pos = ax.get_position()
        assert x_frac < ax_pos.x1, (
            f"{fn.__name__}: legend not inside axes "
            f"(x_frac={x_frac:.3f}, ax.x1={ax_pos.x1:.3f})"
        )
        plt.close(fig)


def test_inside_false_explicit_still_renders_outside(df):
    """Explicit inside=False behaves identically to the default."""
    fig, ax = pp.subplots(axes_size=(80, 50))
    pp.barplot(
        data=df, x="cat", y="val", hue="grp", palette="pastel", ax=ax,
        legend_kws={"inside": False},
    )
    fig.canvas.draw()
    leg = ax.get_legend()
    assert leg is not None
    x_frac, _ = _anchor_fig_frac(leg, fig)
    ax_pos = ax.get_position()
    assert x_frac > ax_pos.x1


def test_inside_coexists_with_legend_group():
    """Per-panel inside legend + figure-level legend_group collecting a shared
    entry. The collected entry should render once via the group; each panel
    should still render the non-collected entries inside its own axes.
    """
    rng = np.random.default_rng(0)
    t = np.linspace(0, 10, 30)
    rows = []
    for p in "ABC":
        for g in ["Control", "Treated"]:
            for m in ["raw", "smoothed"]:
                for tt in t:
                    rows.append({
                        "panel": p, "time": tt,
                        "value": np.sin(tt) + rng.normal(0, 0.1),
                        "group": g, "method": m,
                    })
    df_ = pd.DataFrame(rows)

    fig, axes = pp.subplots(1, 3, axes_size=(45, 35))
    # Collect only the shared hue across panels.
    pp.legend(anchor=axes[-1], collect=["group"])
    for ax, panel in zip(axes, "ABC"):
        pp.lineplot(
            data=df_[df_["panel"] == panel], x="time", y="value",
            hue="group", style="method", palette="pastel",
            dashes={"raw": (1, 0), "smoothed": (4, 2)},
            ax=ax,
            legend_kws={"inside": True, "loc": "lower right"},
        )
    fig.canvas.draw()

    # Each non-anchor panel should render the style=method legend inside its axes.
    for ax in axes[:-1]:
        legend_titles = [
            c.get_title().get_text()
            for c in ax.get_children()
            if type(c).__name__ == "Legend"
        ]
        assert legend_titles == ["method"], (
            f"panel legend titles: expected ['method'], got {legend_titles}"
        )

    # The anchor panel (axes[-1]) hosts the legend_group's collected "group"
    # legend AND the per-panel inside style=method legend. Both artists
    # must survive: matplotlib's ax.legend() call during _materialize()
    # evicts prior Legend children, so LegendBuilder re-attaches them.
    anchor_titles = sorted(
        c.get_title().get_text()
        for c in axes[-1].get_children()
        if type(c).__name__ == "Legend"
    )
    assert anchor_titles == ["group", "method"], (
        f"anchor axes lost one of its legends after legend_group materialize: "
        f"{anchor_titles}"
    )
    # Confirm the group entry isn't duplicated inside non-anchor panels.
    for ax in axes[:-1]:
        titles = [
            c.get_title().get_text()
            for c in ax.get_children()
            if type(c).__name__ == "Legend"
        ]
        assert "group" not in titles, (
            f"group entry leaked to non-anchor panel: {titles}"
        )


# ---------------------------------------------------------------------------
# Continuous hue: the colorbar counterpart of inside=True (#215)
# ---------------------------------------------------------------------------


@pytest.fixture
def cont_df():
    rng = np.random.default_rng(3)
    return pd.DataFrame({
        "x": rng.normal(size=60),
        "y": rng.normal(size=60),
        "z": rng.normal(size=60),
    })


def _inside_cbar_axes(ax):
    """The inside colorbar strips parented to ``ax`` (child axes)."""
    return list(ax.child_axes)


def _mm(fig, frac_w, frac_h):
    w_in, h_in = fig.get_size_inches()
    return frac_w * w_in * 25.4, frac_h * h_in * 25.4


def test_inside_true_continuous_hue_renders_inside_axes(cont_df):
    """legend_kws={'inside': True} used to raise TypeError from Colorbar."""
    fig, ax = pp.subplots(axes_size=(60, 40))
    pp.scatterplot(
        data=cont_df, x="x", y="y", hue="z", ax=ax,
        legend_kws={"inside": True, "loc": "upper right"},
    )
    fig.canvas.draw()
    strips = _inside_cbar_axes(ax)
    assert len(strips) == 1, f"expected one inside colorbar, got {len(strips)}"
    pos = strips[0].get_position()
    ax_pos = ax.get_position()
    assert ax_pos.x0 <= pos.x0 and pos.x1 <= ax_pos.x1, (
        f"colorbar strip x=({pos.x0:.3f}, {pos.x1:.3f}) escapes the axes "
        f"x=({ax_pos.x0:.3f}, {ax_pos.x1:.3f})"
    )
    assert ax_pos.y0 <= pos.y0 and pos.y1 <= ax_pos.y1
    # No figure-level colorbar axes: the strip is a child of ax, so it
    # never claims an outside band.
    assert [a for a in fig.axes if a is not ax] == []


def test_inside_continuous_keeps_mm_size_after_one_draw(cont_df):
    """The strip keeps add_colorbar's mm defaults through the first draw.

    Sizing against the axes rectangle (not the figure) is what survives
    pp.subplots' mid-draw figure resize.
    """
    fig, ax = pp.subplots(axes_size=(60, 40))
    pp.scatterplot(
        data=cont_df, x="x", y="y", hue="z", ax=ax,
        legend_kws={"inside": True},
    )
    fig.canvas.draw()
    pos = _inside_cbar_axes(ax)[0].get_position()
    w_mm, h_mm = _mm(fig, pos.width, pos.height)
    assert w_mm == pytest.approx(4.5, abs=0.1), w_mm
    assert h_mm == pytest.approx(15.0, abs=0.1), h_mm


@pytest.mark.parametrize("loc", [
    "upper right", "upper left", "lower left", "lower right",
    "center", "center left", "upper center", "best",
])
def test_inside_continuous_locs_keep_decorations_inside(cont_df, loc):
    """Every supported loc keeps strip + ticklabels + label within the axes."""
    fig, ax = pp.subplots(axes_size=(60, 40))
    pp.scatterplot(
        data=cont_df, x="x", y="y", hue="z", ax=ax,
        legend_kws={"inside": True, "loc": loc},
    )
    fig.canvas.draw()
    strip = _inside_cbar_axes(ax)[0]
    tight = strip.get_tightbbox()
    ax_bbox = ax.get_window_extent()
    assert tight.x0 >= ax_bbox.x0 - 0.5 and tight.x1 <= ax_bbox.x1 + 0.5, (
        f"loc={loc!r}: colorbar decorations spill horizontally"
    )
    assert tight.y0 >= ax_bbox.y0 - 0.5 and tight.y1 <= ax_bbox.y1 + 0.5, (
        f"loc={loc!r}: colorbar decorations spill vertically"
    )


def test_inside_continuous_skips_reactor_registration(cont_df):
    fig, ax = pp.subplots(axes_size=(60, 40))
    reactor = LayoutReactor.get(fig)
    before = len(reactor._registrations)
    pp.scatterplot(
        data=cont_df, x="x", y="y", hue="z", ax=ax,
        legend_kws={"inside": True},
    )
    assert len(reactor._registrations) == before, (
        "inside colorbar registered with the reactor; it should track the "
        "axes on its own"
    )


def test_continuous_outside_default_unchanged(cont_df):
    """Without inside=, the colorbar still lands in the outside band."""
    fig, ax = pp.subplots(axes_size=(60, 40))
    pp.scatterplot(data=cont_df, x="x", y="y", hue="z", ax=ax)
    fig.canvas.draw()
    assert _inside_cbar_axes(ax) == []
    outside = [a for a in fig.axes if a is not ax]
    assert len(outside) == 1
    assert outside[0].get_position().x0 >= ax.get_position().x1


@pytest.mark.parametrize("kws", [
    {"frameon": True},
    {"ncol": 2},
    {"markerscale": 2},
    {"title_fontsize": 6},
    {"loc": "upper right"},
    {"handletextpad": 1.0, "labelspacing": 0.5},
])
def test_legend_only_keys_do_not_reach_colorbar(cont_df, kws):
    """Legend-only legend_kws are dropped, not forwarded to Colorbar.

    Every key in the builder forward set used to reach
    ``Colorbar.__init__`` and raise TypeError on a continuous hue (#215).
    """
    fig, ax = pp.subplots(axes_size=(60, 40))
    pp.scatterplot(data=cont_df, x="x", y="y", hue="z", ax=ax, legend_kws=kws)
    fig.canvas.draw()
    assert len([a for a in fig.axes if a is not ax]) == 1


def test_inside_continuous_in_cell_group(cont_df):
    """pp.legend(anchor=empty, inside=True) puts the strip in the anchor cell."""
    fig, axes = pp.subplots(nrows=1, ncols=2, axes_size=(45, 35))
    pp.legend(anchor=axes[1], inside=True)
    pp.scatterplot(data=cont_df, x="x", y="y", hue="z", ax=axes[0])
    fig.canvas.draw()
    strips = _inside_cbar_axes(axes[1])
    assert len(strips) == 1, (
        "in-cell inside=True should render the colorbar inside the anchor "
        f"cell; got {len(strips)} child axes there"
    )
    pos = strips[0].get_position()
    anchor = axes[1].get_position()
    assert anchor.x0 <= pos.x0 and pos.x1 <= anchor.x1
    assert anchor.y0 <= pos.y0 and pos.y1 <= anchor.y1


def test_inside_continuous_in_cell_group_grid_after_ordering(cont_df):
    """The documented 2x2 in-cell pattern, continuous hue, plots first.

    Pins the two things that are easy to mis-measure from outside:

    * the strip is a CHILD axes of the anchor (``anchor.child_axes``), so it
      never appears in ``fig.axes`` — enumerating ``fig.axes`` finds only the
      un-evicted per-axes colorbars and wrongly concludes nothing rendered;
    * the anchor is blanked via ``set_axis_off()``, which flips
      ``ax.axison`` and deliberately leaves ``ax.get_visible()`` True.
    """
    fig, axes = pp.subplots(2, 2, axes_size=(35, 30))
    for r, c in [(0, 0), (0, 1), (1, 0)]:
        pp.scatterplot(data=cont_df, x="x", y="y", hue="z", ax=axes[r, c])
    pp.legend(anchor=axes[1, 1], inside=True)
    for _ in range(4):
        fig.canvas.draw()

    anchor = axes[1, 1]
    strips = _inside_cbar_axes(anchor)
    assert len(strips) == 1, (
        "expected exactly one colorbar inside the anchor cell, got "
        f"{len(strips)}; fig.axes cannot see it because an inset is a child "
        "axes, not a figure axes"
    )
    strip_bbox = strips[0].get_window_extent()
    anchor_bbox = anchor.get_window_extent()
    assert anchor_bbox.x0 <= strip_bbox.x0 and strip_bbox.x1 <= anchor_bbox.x1, (
        f"strip x=({strip_bbox.x0:.0f}, {strip_bbox.x1:.0f}) escapes anchor "
        f"x=({anchor_bbox.x0:.0f}, {anchor_bbox.x1:.0f})"
    )
    assert anchor_bbox.y0 <= strip_bbox.y0 and strip_bbox.y1 <= anchor_bbox.y1

    assert anchor.axison is False, (
        "anchor cell not blanked; note set_axis_off() flips ax.axison and "
        "leaves ax.get_visible() True, so get_visible() cannot test this"
    )


def test_inside_continuous_and_categorical_coexist():
    """A hue colorbar and a style legend can both go inside one axes."""
    rng = np.random.default_rng(5)
    data = pd.DataFrame({
        "x": rng.normal(size=60),
        "y": rng.normal(size=60),
        "z": rng.normal(size=60),
        "m": np.tile(list("AB"), 30),
    })
    fig, ax = pp.subplots(axes_size=(70, 50))
    pp.scatterplot(
        data=data, x="x", y="y", hue="z", style="m", ax=ax,
        legend_kws={"inside": True, "loc": "upper left"},
    )
    fig.canvas.draw()
    assert len(_inside_cbar_axes(ax)) == 1
    assert ax.get_legend() is not None
    assert [a for a in fig.axes if a is not ax] == []


def test_inside_continuous_rejects_unknown_loc(cont_df):
    """An unusable loc fails with our own message, not a matplotlib internal."""
    fig, ax = pp.subplots(axes_size=(60, 40))
    with pytest.raises(ValueError, match="inside colorbar loc must be one of"):
        pp.scatterplot(
            data=cont_df, x="x", y="y", hue="z", ax=ax,
            legend_kws={"inside": True, "loc": "nowhere"},
        )
