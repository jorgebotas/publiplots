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
