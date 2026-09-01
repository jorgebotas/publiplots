"""Colorbar orientation follows the band's side (#213).

``add_colorbar``'s ``orientation`` used to default to ``'vertical'``
regardless of where the band sat, so ``pp.legend(ax, side='top')``
rendered a 4.5 x 15mm vertical strip lying across the top of the figure
while the band around it stacked horizontally. ``orientation=None`` now
means "derive": horizontal on a top/bottom band, vertical on left/right.

The derivation reads ``LegendBuilder._orientation`` — the band
orientation ``MultiAxesLegendGroup`` already resolved from ``side`` — not
``side`` itself. That is what makes an explicit
``pp.legend(side='bottom', orientation='vertical')`` keep a vertical
strip: a vertical band should hold a vertical strip.

``height`` and ``width`` keep their literal mm meaning at every
orientation (``height`` is the vertical extent, ``width`` the horizontal
one); only their defaults swap, 15 x 4.5 becoming 4.5 x 15.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

import publiplots as pp
from publiplots.utils.legend import create_legend_handles


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _df(n=40, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "x": rng.normal(size=n),
        "y": rng.normal(size=n),
        "c": np.linspace(0.0, 1.0, n),
        "g": np.tile(["A", "B"], n // 2),
    })


def _mm(fig, px):
    return px / fig.dpi * 25.4


def _sm():
    return ScalarMappable(norm=Normalize(0.0, 1.0), cmap="viridis")


def _strips(group):
    return [obj for kind, obj in group._builder.elements if kind == "colorbar"]


def _strip_mm(fig, cbar):
    """The colour rectangle's (width, height) in mm."""
    r = cbar.ax.get_window_extent()
    return _mm(fig, r.width), _mm(fig, r.height)


# --- per-side default -------------------------------------------------------


_EXPECTED = {
    "right": ("vertical", 4.5, 15.0),
    "left": ("vertical", 4.5, 15.0),
    "top": ("horizontal", 15.0, 4.5),
    "bottom": ("horizontal", 15.0, 4.5),
}


@pytest.mark.parametrize("side", ["right", "left", "top", "bottom"])
def test_per_axes_band_orientation_follows_side(side):
    """A per-axes band's strip runs along the band's own axis."""
    fig, ax = pp.subplots(1, 1, axes_size=(50, 40))
    pp.scatterplot(data=_df(), x="x", y="y", hue="c", ax=ax, title="t")
    group = pp.legend(ax, side=side)
    fig.canvas.draw()

    cbar, = _strips(group)
    want_orient, want_w, want_h = _EXPECTED[side]
    assert cbar.orientation == want_orient, (
        f"side={side!r}: strip orientation should be {want_orient!r}, "
        f"got {cbar.orientation!r}"
    )
    w, h = _strip_mm(fig, cbar)
    assert (w, h) == pytest.approx((want_w, want_h), abs=0.05), (
        f"side={side!r}: strip should be {want_w} x {want_h}mm, "
        f"got {w:.2f} x {h:.2f}mm"
    )


@pytest.mark.parametrize("side", ["right", "left", "top", "bottom"])
def test_multi_axes_band_orientation_follows_side(side):
    """A multi-axes band resolves the same way — the orientation comes
    from ``MultiAxesLegendGroup._DEFAULT_ORIENTATION``, which both anchor
    modes share."""
    fig, axes = pp.subplots(1, 2, axes_size=(40, 30))
    flat = list(np.atleast_1d(axes).flat)
    for a in flat:
        pp.scatterplot(data=_df(), x="x", y="y", hue="c", ax=a, title="t")
    group = pp.legend(anchor=flat[0], axes=flat, side=side)
    fig.canvas.draw()

    cbar, = _strips(group)
    want_orient, want_w, want_h = _EXPECTED[side]
    assert cbar.orientation == want_orient
    w, h = _strip_mm(fig, cbar)
    assert (w, h) == pytest.approx((want_w, want_h), abs=0.05), (
        f"side={side!r}: strip should be {want_w} x {want_h}mm, "
        f"got {w:.2f} x {h:.2f}mm"
    )


@pytest.mark.parametrize("side", ["top", "bottom"])
def test_plot_call_legend_kws_side_gets_horizontal_strip(side):
    """``legend_kws={'side': ...}`` on the plot call reaches the same
    resolution: the plot path renders through the per-axes group's
    builder, so it inherits the band orientation."""
    fig, ax = pp.subplots(1, 1, axes_size=(50, 40))
    pp.scatterplot(data=_df(), x="x", y="y", hue="c", ax=ax, title="t",
                   legend_kws={"side": side})
    fig.canvas.draw()

    cbar, = _strips(ax._legend_group)
    assert cbar.orientation == "horizontal"
    w, h = _strip_mm(fig, cbar)
    assert (w, h) == pytest.approx((15.0, 4.5), abs=0.05), (
        f"legend_kws side={side!r}: got {w:.2f} x {h:.2f}mm"
    )


# --- explicit override, both directions -------------------------------------


def test_explicit_vertical_wins_on_a_bottom_band():
    """``orientation='vertical'`` on a bottom band keeps a vertical strip.

    This is why the default is derived from the *band* orientation rather
    than from ``side``: ``pp.legend(side='bottom', orientation='vertical')``
    asks for a vertical band, and a vertical band should hold a vertical
    strip.
    """
    fig, ax = pp.subplots(1, 1, axes_size=(50, 40))
    ax.plot([0, 1], [0, 1])
    group = pp.legend(ax, side="bottom", orientation="vertical")
    group.add_colorbar(_sm(), label="Value")
    fig.canvas.draw()

    cbar, = _strips(group)
    assert cbar.orientation == "vertical"
    w, h = _strip_mm(fig, cbar)
    assert (w, h) == pytest.approx((4.5, 15.0), abs=0.05), (
        f"expected the vertical 4.5 x 15mm default, got {w:.2f} x {h:.2f}mm"
    )


def test_explicit_horizontal_wins_on_a_right_band():
    """The mirror case: ``orientation='horizontal'`` on a right band."""
    fig, ax = pp.subplots(1, 1, axes_size=(50, 40))
    ax.plot([0, 1], [0, 1])
    group = pp.legend(ax, side="right")
    group.add_colorbar(_sm(), label="Value", orientation="horizontal")
    fig.canvas.draw()

    cbar, = _strips(group)
    assert cbar.orientation == "horizontal"
    w, h = _strip_mm(fig, cbar)
    assert (w, h) == pytest.approx((15.0, 4.5), abs=0.05), (
        f"expected the horizontal 15 x 4.5mm default, got {w:.2f} x {h:.2f}mm"
    )


def test_bad_orientation_rejected():
    fig, ax = pp.subplots(1, 1, axes_size=(50, 40))
    ax.plot([0, 1], [0, 1])
    group = pp.legend(ax, side="top")
    with pytest.raises(ValueError, match="orientation must be"):
        group.add_colorbar(_sm(), label="Value", orientation="sideways")


# --- height / width stay literal --------------------------------------------


@pytest.mark.parametrize("kwargs,want", [
    ({"height": 20}, (15.0, 20.0)),   # width still defaults for horizontal
    ({"width": 8}, (8.0, 4.5)),       # height still defaults for horizontal
    ({"height": 6, "width": 30}, (30.0, 6.0)),
])
def test_explicit_height_width_are_literal_on_a_top_band(kwargs, want):
    """``height`` is the VERTICAL extent on every side, ``width`` the
    horizontal one. Only the defaults swap with the orientation, so
    ``height=20`` on a top band is a 20mm-TALL horizontal strip — the
    behaviour change #213 documents (migration: pass ``width=20``).
    """
    fig, ax = pp.subplots(1, 1, axes_size=(50, 40))
    ax.plot([0, 1], [0, 1])
    group = pp.legend(ax, side="top")
    group.add_colorbar(_sm(), label="Value", **kwargs)
    fig.canvas.draw()

    cbar, = _strips(group)
    assert cbar.orientation == "horizontal"
    w, h = _strip_mm(fig, cbar)
    assert (w, h) == pytest.approx(want, abs=0.05), (
        f"{kwargs} should give {want[0]} x {want[1]}mm, "
        f"got {w:.2f} x {h:.2f}mm"
    )


def test_vertical_band_keeps_its_own_literal_defaults():
    """Sanity companion: nothing about the left/right default moved."""
    fig, ax = pp.subplots(1, 1, axes_size=(50, 40))
    ax.plot([0, 1], [0, 1])
    group = pp.legend(ax, side="right")
    group.add_colorbar(_sm(), label="Value", height=20)
    fig.canvas.draw()

    cbar, = _strips(group)
    w, h = _strip_mm(fig, cbar)
    assert (w, h) == pytest.approx((4.5, 20.0), abs=0.05), (
        f"expected 4.5 x 20mm, got {w:.2f} x {h:.2f}mm"
    )


# --- the top band's block still clears the axes -----------------------------


def test_top_band_horizontal_strip_clears_the_axes_rect():
    """A horizontal strip's tick labels sit BELOW it, which on a top band
    points back toward the axes. The block is stepped outward by that
    overhang so the strip's *tight* bbox — tick labels included — is what
    clears the axes edge.

    Measured with the strip placed by its bare rectangle instead, on this
    exact configuration: the tick labels reached 1.61mm INSIDE a 40mm-tall
    axes, drawing "0.0 0.5 1.0" across the top spine.
    """
    fig, ax = pp.subplots(1, 1, axes_size=(50, 40))
    pp.scatterplot(data=_df(), x="x", y="y", hue="c", ax=ax, title="t")
    group = pp.legend(ax, side="top")

    cbar, = _strips(group)
    # Draw 0 must already be right (a consumer that draws once never gets
    # a second pass); draw 1+ is steady state, which is what savefig writes.
    for draw in range(3):
        fig.canvas.draw()
        tight = cbar.ax.get_tightbbox()
        ax_bb = ax.get_window_extent()
        clearance = _mm(fig, tight.y0 - ax_bb.y1)
        assert clearance > -0.1, (
            f"[draw {draw}] the strip's tick labels dip {-clearance:.2f}mm "
            f"into the axes rectangle"
        )


# --- multiple elements in one top band --------------------------------------


def test_two_colorbars_in_one_top_band_do_not_overlap():
    """Two horizontal strips in a top band sit side by side along the
    edge, each under its own label."""
    fig, ax = pp.subplots(1, 1, axes_size=(60, 40))
    ax.plot([0, 1], [0, 1])
    group = pp.legend(ax, side="top")
    group.add_colorbar(_sm(), label="One")
    group.add_colorbar(ScalarMappable(norm=Normalize(0, 5), cmap="magma"),
                       label="Two")

    for draw in range(3):
        fig.canvas.draw()
        first, second = _strips(group)
        assert first.orientation == second.orientation == "horizontal"
        a = first.ax.get_window_extent()
        b = second.ax.get_window_extent()
        # Same outward row (they share the band's base offset), sequenced
        # along the edge with a positive gap between the rectangles.
        assert _mm(fig, abs(a.y0 - b.y0)) < 0.1, (
            f"[draw {draw}] the two strips should share one outward row; "
            f"y0 differs by {_mm(fig, abs(a.y0 - b.y0)):.2f}mm"
        )
        gap = _mm(fig, b.x0 - a.x1)
        assert gap > 0.5, (
            f"[draw {draw}] the two strips overlap or touch along the "
            f"edge; gap is {gap:+.2f}mm"
        )


def test_colorbar_and_legend_in_one_top_band_do_not_overlap():
    """A colorbar block and a categorical legend in the same top band are
    laid out side by side, not on top of one another.

    The clearance shift that keeps the strip's tick labels off the axes
    must not push the strip into a row of its own: rows are keyed on the
    band's base outward offset, and two rows centred independently on the
    same centre line would draw the strip across the legend.
    """
    fig, ax = pp.subplots(1, 1, axes_size=(60, 40))
    ax.plot([0, 1], [0, 1])
    group = pp.legend(ax, side="top")
    group.add_colorbar(_sm(), label="Cbar")
    group.add_legend(
        handles=create_legend_handles(
            labels=["A", "B"],
            colors=list(pp.color_palette("pastel", 2)),
            alpha=0.2, linewidth=1.0,
        ),
        label="Cats",
    )

    for draw in range(3):
        fig.canvas.draw()
        cbar, = _strips(group)
        legend, = [o for k, o in group._builder.elements if k == "legend"]
        strip = cbar.ax.get_window_extent()
        leg = legend.get_window_extent()
        overlap_w = _mm(fig, min(strip.x1, leg.x1) - max(strip.x0, leg.x0))
        overlap_h = _mm(fig, min(strip.y1, leg.y1) - max(strip.y0, leg.y0))
        assert not (overlap_w > 0.5 and overlap_h > 0.5), (
            f"[draw {draw}] strip and legend overlap by "
            f"{overlap_w:.2f} x {overlap_h:.2f}mm"
        )


# --- inside mode is untouched ----------------------------------------------


def test_inside_colorbar_stays_vertical():
    """``legend_kws={'inside': True}`` renders through a bare
    ``LegendBuilder`` (side 'right', orientation 'vertical'), so the
    derived orientation is the historical vertical one."""
    fig, ax = pp.subplots(1, 1, axes_size=(50, 40))
    pp.scatterplot(data=_df(), x="x", y="y", hue="c", ax=ax, title="t",
                   legend_kws={"inside": True, "loc": "upper right"})
    fig.canvas.draw()

    strips = [c for c in ax.child_axes]
    assert len(strips) == 1, f"expected one inside strip, got {len(strips)}"
    r = strips[0].get_window_extent()
    w, h = _mm(fig, r.width), _mm(fig, r.height)
    assert (w, h) == pytest.approx((4.5, 15.0), abs=0.05), (
        f"the inside strip should keep the vertical 4.5 x 15mm default, "
        f"got {w:.2f} x {h:.2f}mm"
    )


def test_explicit_inside_group_follows_its_side():
    """An explicitly constructed ``inside=True`` group DOES follow its side.

    Counterpart to the test above, and the boundary between the two is the
    point: the plot path's ``inside=True`` short-circuit builds a bare
    ``LegendBuilder`` (side 'right'), so there is no top/bottom band for an
    orientation to follow and the strip stays vertical. A caller who writes
    ``pp.legend(anchor=ax, side='top', inside=True)`` has named a side, and
    the rule is that orientation follows it — a flat strip along the top of
    the panel rather than one standing on end inside it.

    Pinned because it is a knock-on of the #213 derivation rather than
    something the change set out to do, so a future reader should see it was
    a decision and not an accident.
    """
    fig, ax = pp.subplots(1, 1, axes_size=(50, 40))
    pp.scatterplot(data=_df(), x="x", y="y", hue="c", ax=ax, title="t")
    pp.legend(anchor=ax, side="top", inside=True)
    fig.canvas.draw()

    strips = [c for c in ax.child_axes]
    assert len(strips) == 1, f"expected one inside strip, got {len(strips)}"
    r = strips[0].get_window_extent()
    w, h = _mm(fig, r.width), _mm(fig, r.height)
    assert w > h, (
        f"an inside group on side='top' should render a flat strip, "
        f"got {w:.2f} x {h:.2f}mm"
    )
