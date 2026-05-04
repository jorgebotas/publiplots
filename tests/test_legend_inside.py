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
    pp.legend_group(anchor=axes[-1], collect=["group"])
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
    # legend. The style=method inside legend was also rendered on this panel,
    # but matplotlib's `ax.legend_` slot only holds the most recently attached
    # one; both artists exist as children.
    anchor_titles = [
        c.get_title().get_text()
        for c in axes[-1].get_children()
        if type(c).__name__ == "Legend"
    ]
    assert "group" in anchor_titles, (
        f"legend_group didn't render 'group' on anchor axes: {anchor_titles}"
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
