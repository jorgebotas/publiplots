"""Issue #222 — a pinned xlabel_space / ylabel_space must still get band
collision avoidance.

Pinning ``xlabel_space=`` / ``ylabel_space=`` means "do not GROW this
reservation" (callers pin to keep panel geometry aligned across separate
figures). It must not also mean "do not MOVE the legend band clear of the
tick labels and the axis label" — which is what the lock guards used to do:
the band landed 2.00 mm below the axes on a figure whose x tick labels reach
3.61 mm and whose xlabel reaches 7.39 mm below it, overlapping both.

Everything is measured in mm, signed relative to the axes rectangle's edge
on the side under test: 0 is the spine, negative is outward. The band's
inner edge must therefore be at least as negative as the outermost
decoration edge.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import publiplots as pp


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


PIN_MM = 14.0


def _frame():
    rng = np.random.default_rng(0)
    n = 60
    return pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "y": rng.normal(size=n),
            "g": rng.choice(["alpha", "beta"], size=n),
        }
    )


def _mm(px, dpi):
    return px / dpi * 25.4


def _build(side, *, pin=PIN_MM, external=False):
    """Return (fig, anchor_ax, group) for one scenario.

    ``external=False`` is ``pp.legend(ax, side=...)`` — an in-frame,
    ``external_to_axis=False`` per-axes band. ``external=True`` is the
    multi-axes ``pp.legend(anchor=..., axes=[...])`` form, which routes
    through the ``external_to_axis=True`` overhang path instead. Both were
    broken by the pin, via different guards.
    """
    space = "xlabel_space" if side == "bottom" else "ylabel_space"
    kwargs = {} if pin is None else {space: pin}
    ncols = 2 if external else 1
    fig, axes = pp.subplots(1, ncols, axes_size=(50, 40), **kwargs)
    axes = list(np.atleast_1d(axes).ravel())
    df = _frame()
    for ax in axes:
        pp.scatterplot(data=df, x="x", y="y", hue="g", ax=ax)
        ax.set_xlabel("x axis label")
        ax.set_ylabel("y axis label")
    if external:
        group = pp.legend(anchor=axes[0], axes=axes, side=side)
    else:
        group = pp.legend(axes[0], side=side)
    fig._publiplots_auto_layout.settle()
    return fig, axes[0], group


def _band_inner_mm(fig, ax, group, side):
    """Innermost (closest to the axes) edge of the legend band, in mm."""
    dpi = fig.dpi
    ax_bb = ax.get_window_extent()
    auto = fig._publiplots_auto_layout
    inner = None
    for _, obj in group._builder.elements:
        extent = auto._artist_window_extent(obj)
        if extent is None:
            continue
        value = (
            _mm(extent.y1 - ax_bb.y0, dpi)
            if side == "bottom"
            else _mm(extent.x1 - ax_bb.x0, dpi)
        )
        inner = value if inner is None else max(inner, value)
    assert inner is not None, "legend band produced no measurable artist"
    return inner


def _decoration_outer_mm(fig, ax, side):
    """Outermost edge of the tick labels and the axis label, in mm.

    Returns ``(tick_outer, label_outer)``.
    """
    dpi = fig.dpi
    ax_bb = ax.get_window_extent()
    labels = ax.get_xticklabels() if side == "bottom" else ax.get_yticklabels()
    tick_edges = []
    for text in labels:
        if not text.get_text():
            continue
        extent = text.get_window_extent()
        tick_edges.append(
            _mm(extent.y0 - ax_bb.y0, dpi)
            if side == "bottom"
            else _mm(extent.x0 - ax_bb.x0, dpi)
        )
    assert tick_edges, "axes has no tick labels to collide with"
    axis_label = ax.xaxis.label if side == "bottom" else ax.yaxis.label
    extent = axis_label.get_window_extent()
    label_outer = (
        _mm(extent.y0 - ax_bb.y0, dpi)
        if side == "bottom"
        else _mm(extent.x0 - ax_bb.x0, dpi)
    )
    return min(tick_edges), label_outer


@pytest.mark.parametrize("side", ["bottom", "left"])
@pytest.mark.parametrize("external", [False, True], ids=["in_frame", "external"])
def test_pinned_space_keeps_band_clear_of_decorations(side, external):
    fig, ax, group = _build(side, external=external)
    band = _band_inner_mm(fig, ax, group, side)
    tick, label = _decoration_outer_mm(fig, ax, side)
    worst = min(tick, label)
    assert band <= worst + 1e-6, (
        f"side={side!r} external={external}: legend band's inner edge sits at "
        f"{band:.2f}mm but the tick labels reach {tick:.2f}mm and the axis "
        f"label {label:.2f}mm — the band overlaps them by "
        f"{band - worst:.2f}mm. A pinned reservation must not disable "
        f"collision avoidance (issue #222)."
    )


@pytest.mark.parametrize("side", ["bottom", "left"])
@pytest.mark.parametrize("external", [False, True], ids=["in_frame", "external"])
def test_pin_is_still_honoured(side, external):
    """The whole point of pinning: the reservation keeps exactly its mm."""
    fig, ax, _ = _build(side, external=external)
    field = "xlabel_space" if side == "bottom" else "ylabel_space"
    reservation = getattr(fig._publiplots_layout, field)
    assert reservation[0] == pytest.approx(PIN_MM, abs=1e-9), (
        f"pinned {field} must stay at {PIN_MM}mm, got {reservation[0]}mm"
    )


@pytest.mark.parametrize("pin", [4.0, 20.0])
def test_pinned_space_extremes_still_clear_decorations(pin):
    """A pin far below or far above what the band needs behaves the same.

    Too small a pin does NOT push the band back over the decorations — the
    band simply extends past the pinned reservation, which is the caller's
    explicit choice in pinning.
    """
    fig, ax, group = _build("bottom", pin=pin)
    band = _band_inner_mm(fig, ax, group, "bottom")
    tick, label = _decoration_outer_mm(fig, ax, "bottom")
    assert band <= min(tick, label) + 1e-6, (
        f"xlabel_space={pin}: band inner edge {band:.2f}mm overlaps "
        f"decorations at {min(tick, label):.2f}mm"
    )
    assert fig._publiplots_layout.xlabel_space[0] == pytest.approx(pin, abs=1e-9)


@pytest.mark.parametrize("side", ["bottom", "left"])
def test_pinned_band_matches_unpinned_placement(side):
    """A pin fixes the reservation size, not where the band sits (#222).

    The first assertion is the control for #212 / #220 — the auto-measured
    path must keep clearing the decorations. The second is the #222
    assertion proper: the pinned figure must place the band at the same
    outward offset as the equivalent unpinned one.
    """
    fig_auto, ax_auto, group_auto = _build(side, pin=None)
    auto_band = _band_inner_mm(fig_auto, ax_auto, group_auto, side)
    tick, label = _decoration_outer_mm(fig_auto, ax_auto, side)
    assert auto_band <= min(tick, label) + 1e-6

    fig_pin, ax_pin, group_pin = _build(side, pin=PIN_MM)
    pin_band = _band_inner_mm(fig_pin, ax_pin, group_pin, side)
    assert pin_band == pytest.approx(auto_band, abs=0.05), (
        f"side={side!r}: pinned band at {pin_band:.2f}mm vs unpinned "
        f"{auto_band:.2f}mm — the pin must not change where the band sits"
    )
