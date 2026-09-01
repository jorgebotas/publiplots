"""Issue #222 — a pinned xlabel_space / ylabel_space must still get band
collision avoidance, without pushing the band off the canvas.

Pinning ``xlabel_space=`` / ``ylabel_space=`` means "do not GROW this
reservation" (callers pin to keep panel geometry aligned across separate
figures). It must not also mean "do not MOVE the legend band clear of the
tick labels and the axis label" — which is what the lock guards used to do:
the band landed 2.00 mm below the axes on a figure whose x tick labels reach
3.61 mm and whose xlabel reaches 7.39 mm below it, overlapping both.

But the pinned row/column does not grow around a band stepped outward, and
``savefig.bbox`` is ``"standard"``, so an unclamped step walks the band off
the canvas where the saved file crops it away. The contract is therefore a
priority order, and these tests encode it:

1. the band stays inside the figure whenever it physically fits there, and
   when it does not fit it degrades to exactly the pre-#222 placement — never
   further out (new clipping) and never further in (a new #222);
2. subject to that, the band steps as far past the decorations as fits, and
   clears them completely whenever the pinned space is generous enough.

Geometry is measured in mm, signed relative to the axes rectangle's edge on
the side under test: 0 is the spine, negative is outward. The reactor places
each band element at ``base_gap + offset`` outward of that edge and the
element extends its own size further out, so the band's reach decomposes as
``base_gap + offset + own_size`` — which is how these tests recover the
offset without reaching into the reactor.
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


# Pins generous enough that the decorations AND the whole band fit inside the
# reservation on a 50x40 mm axes, so the band must clear the decorations
# outright. The left band is much wider than the bottom band is tall, hence
# the different values.
GENEROUS = {"bottom": 20.0, "left": 25.0}

ALL_PINS = [4.0, 6.0, 8.0, 10.0, 14.0, 20.0, 25.0]

# Tolerance for "still on the canvas", in mm. Well below a hairline.
EPS_MM = 0.02


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


def _space_kw(side, pin):
    field = "xlabel_space" if side == "bottom" else "ylabel_space"
    return {} if pin is None else {field: pin}


def _build(side, *, pin, external=False, ncols=None):
    """Return (fig, anchor_ax, group, all_axes) for one scenario.

    ``external=False`` is ``pp.legend(ax, side=...)`` — an in-frame,
    ``external_to_axis=False`` per-axes band, the form the issue reproduced
    with. ``external=True`` is the multi-axes
    ``pp.legend(anchor=..., axes=[...])`` form, which routes through the
    ``external_to_axis=True`` overhang path in ``_measure_one_group``
    instead. The pin broke both, via different guards.
    """
    if ncols is None:
        ncols = 2 if external else 1
    fig, axes = pp.subplots(1, ncols, axes_size=(50, 40), **_space_kw(side, pin))
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
    return fig, axes[0], group, axes


class _Geometry:
    """Every number one scenario needs, all in mm relative to the axes edge."""

    def __init__(self, fig, ax, group, side):
        dpi = fig.dpi
        auto = fig._publiplots_auto_layout
        ax_bb = ax.get_window_extent()
        fig_bb = fig.get_window_extent()
        vertical = side == "bottom"

        regs = {
            id(reg.artist): reg
            for reg in group._builder._reactor._registrations
        }
        inner = outer = None
        self.base_gap = None
        self.band_size = 0.0
        for _, obj in group._builder.elements:
            extent = auto._artist_window_extent(obj)
            if extent is None:
                continue
            if vertical:
                i, o = extent.y1 - ax_bb.y0, extent.y0 - ax_bb.y0
            else:
                i, o = extent.x1 - ax_bb.x0, extent.x0 - ax_bb.x0
            inner = i if inner is None else max(inner, i)
            outer = o if outer is None else min(outer, o)
            self.band_size = max(
                self.band_size,
                _mm(extent.height if vertical else extent.width, dpi),
            )
            reg = regs.get(id(obj))
            if reg is not None:
                self.base_gap = reg.mm_x_from_right
        assert inner is not None, "legend band produced no measurable artist"
        assert self.base_gap is not None, "band element has no reactor registration"

        self.band_inner = _mm(inner, dpi)
        self.band_outer = _mm(outer, dpi)
        # Outward space between the axes edge and the figure edge.
        self.available = _mm(
            (ax_bb.y0 - fig_bb.y0) if vertical else (ax_bb.x0 - fig_bb.x0), dpi
        )
        # How far outside the canvas the band's outer edge reaches (0 = inside).
        self.clipped = max(0.0, -self.band_outer - self.available)
        # Recovered from the placement decomposition; 0 is the pre-#222 value.
        self.offset = -self.band_inner - self.base_gap
        # True when the band can sit on the canvas at all, at any offset.
        self.fits = self.base_gap + self.band_size <= self.available + EPS_MM

        labels = ax.get_xticklabels() if vertical else ax.get_yticklabels()
        edges = [
            _mm(
                (t.get_window_extent().y0 - ax_bb.y0)
                if vertical
                else (t.get_window_extent().x0 - ax_bb.x0),
                dpi,
            )
            for t in labels
            if t.get_text()
        ]
        assert edges, "axes has no tick labels to collide with"
        self.tick = min(edges)
        axis_label = ax.xaxis.label if vertical else ax.yaxis.label
        e = axis_label.get_window_extent()
        self.label = _mm(
            (e.y0 - ax_bb.y0) if vertical else (e.x0 - ax_bb.x0), dpi
        )
        self.decoration = min(self.tick, self.label)

    def __str__(self):
        return (
            f"band=[{self.band_inner:.2f}, {self.band_outer:.2f}] "
            f"available={self.available:.2f} base_gap={self.base_gap:.2f} "
            f"band_size={self.band_size:.2f} offset={self.offset:.2f} "
            f"tick={self.tick:.2f} label={self.label:.2f} "
            f"clipped={self.clipped:.2f} fits={self.fits}"
        )


def _geometry(side, **kw):
    fig, ax, group, axes = _build(side, **kw)
    return _Geometry(fig, ax, group, side), fig, axes


# --------------------------------------------------------------------------
# Priority 1 — the band must not be pushed off the canvas.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("pin", ALL_PINS)
@pytest.mark.parametrize("side", ["bottom", "left"])
@pytest.mark.parametrize("external", [False, True], ids=["in_frame", "external"])
def test_pinned_band_is_never_clipped_beyond_the_unavoidable(side, pin, external):
    """Stepping past the decorations must not walk the band off the figure.

    ``savefig.bbox`` is ``"standard"``, so whatever leaves the canvas is
    cropped out of the saved file — trading an invisible bounding-box overlap
    for deleted legend content is the wrong trade.

    Two regimes, and the test picks between them by measurement rather than
    by a hard-coded font-dependent threshold:

    * the band fits in the pinned space — then it must be entirely on the
      canvas;
    * the band is by itself wider/taller than the pinned space — a
      pre-existing situation no offset can rescue — then the offset must
      collapse to 0, i.e. exactly where the code put the band before #222,
      adding no clipping of its own.
    """
    geo, _, _ = _geometry(side, pin=pin, external=external)
    if geo.fits:
        assert geo.clipped <= EPS_MM, (
            f"side={side!r} pin={pin} external={external}: band reaches "
            f"{geo.clipped:.2f}mm past the figure edge and savefig would crop "
            f"it — {geo}"
        )
    else:
        assert geo.offset <= EPS_MM, (
            f"side={side!r} pin={pin} external={external}: the band alone "
            f"does not fit the pinned space, so the offset must collapse to "
            f"0 (the pre-#222 placement) instead of adding clipping; got "
            f"offset={geo.offset:.2f}mm — {geo}"
        )


@pytest.mark.parametrize("pin", ALL_PINS)
@pytest.mark.parametrize("side", ["bottom", "left"])
def test_pinned_band_is_never_moved_inward_of_the_pre_fix_placement(side, pin):
    """The clamp has a floor: it may shorten the step, never reverse it.

    Before #222 a pinned reservation left the offset at 0. Clamping below
    that would be a regression in the opposite direction — the band creeping
    under the axes instead of past the decorations.
    """
    geo, _, _ = _geometry(side, pin=pin)
    assert geo.offset >= -EPS_MM, (
        f"side={side!r} pin={pin}: offset went negative ({geo.offset:.2f}mm), "
        f"pulling the band inward of where it sat before the fix — {geo}"
    )


# --------------------------------------------------------------------------
# Priority 2 — subject to staying on the canvas, clear the decorations.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("side", ["bottom", "left"])
@pytest.mark.parametrize("external", [False, True], ids=["in_frame", "external"])
def test_generous_pin_clears_the_decorations_completely(side, external):
    """The #222 regression guard: a pin with room for both must not overlap.

    This is the case the issue was reported for — the pin exists to hold the
    panel geometry steady, not to disable collision avoidance — with enough
    pinned space that there is no trade-off to make.
    """
    geo, _, _ = _geometry(side, pin=GENEROUS[side], external=external)
    assert geo.fits, f"test setup: pin should be generous enough — {geo}"
    assert geo.band_inner <= geo.decoration + EPS_MM, (
        f"side={side!r} external={external}: band's inner edge sits at "
        f"{geo.band_inner:.2f}mm but the tick labels reach {geo.tick:.2f}mm "
        f"and the axis label {geo.label:.2f}mm — the band overlaps them by "
        f"{geo.band_inner - geo.decoration:.2f}mm. A pinned reservation must "
        f"not disable collision avoidance (issue #222) — {geo}"
    )
    assert geo.clipped <= EPS_MM, f"and it must stay on the canvas — {geo}"


@pytest.mark.parametrize(
    ("side", "pin"), [("bottom", 10.0), ("bottom", 14.0), ("left", 20.0)]
)
def test_tight_pin_still_improves_on_the_pre_fix_overlap(side, pin):
    """A pin too tight for full clearance must still do better than nothing.

    Priority 2 is "as far past the decorations as fits", so where there is
    any slack at all the band must use it — a strictly smaller residual
    overlap than the pre-#222 placement, while staying on the canvas.
    """
    geo, _, _ = _geometry(side, pin=pin)
    assert geo.clipped <= EPS_MM, f"must stay on the canvas — {geo}"
    assert geo.offset > EPS_MM, (
        f"side={side!r} pin={pin}: there is room to step "
        f"{geo.available - geo.base_gap - geo.band_size:.2f}mm outward, but "
        f"the band did not move at all (offset={geo.offset:.2f}mm) — {geo}"
    )


# --------------------------------------------------------------------------
# The pin itself must still be honoured — that is what the caller asked for.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("side", ["bottom", "left"])
@pytest.mark.parametrize("external", [False, True], ids=["in_frame", "external"])
def test_pin_is_still_honoured(side, external):
    pin = GENEROUS[side]
    _, fig, _ = _geometry(side, pin=pin, external=external)
    field = "xlabel_space" if side == "bottom" else "ylabel_space"
    reservation = getattr(fig._publiplots_layout, field)
    assert reservation[0] == pytest.approx(pin, abs=1e-9), (
        f"pinned {field} must stay at {pin}mm, got {reservation[0]}mm"
    )


# --------------------------------------------------------------------------
# Per-position locks — ``xlabel_space=(14.0, None)`` style. This is what
# JointGrid uses in production, and it reaches a different pair of guards
# than a whole-side pin does.
# --------------------------------------------------------------------------


def _build_per_position_lock(external):
    """1x2 with ylabel_space pinned on column 0 only, left band on column 0.

    ``ylabel_space=(25.0, None)`` pins column 0 and leaves column 1 to
    auto-measure, so ``SubplotsAutoLayout`` gets
    ``locked_positions={"ylabel_space": {0}}`` and an empty ``locked``. The
    pin is generous enough that the on-canvas clamp is inactive, which is
    what makes the assertion below a clean signal about the lock handling
    rather than about the clamp.
    """
    fig, axes = pp.subplots(1, 2, axes_size=(50, 40), ylabel_space=(25.0, None))
    axes = list(axes)
    df = _frame()
    for ax in axes:
        pp.scatterplot(data=df, x="x", y="y", hue="g", ax=ax)
        ax.set_xlabel("x axis label")
        ax.set_ylabel("y axis label")
    if external:
        group = pp.legend(anchor=axes[0], axes=axes, side="left")
    else:
        group = pp.legend(axes[0], side="left")
    fig._publiplots_auto_layout.settle()
    return fig, axes, group


@pytest.mark.parametrize("external", [False, True], ids=["in_frame", "external"])
def test_per_position_lock_keeps_collision_avoidance(external):
    """A per-position pin must behave like a whole-side pin, not like #222.

    ``external=False`` exercises the lock check inside
    ``_offset_inside_legend_past_decorations`` (which must decide whether to
    clamp, and must NOT return early); ``external=True`` exercises the
    per-position early return in ``_measure_one_group`` (which must position
    the band before it returns). Either one reverting drops the band back to
    2.00 mm from the spine, on top of the y tick labels and the ylabel.
    """
    fig, axes, group = _build_per_position_lock(external)
    geo = _Geometry(fig, axes[0], group, "left")
    assert geo.band_inner <= geo.decoration + EPS_MM, (
        f"external={external}: with ylabel_space=(25.0, None) the band's "
        f"inner edge sits at {geo.band_inner:.2f}mm but the y tick labels "
        f"reach {geo.tick:.2f}mm and the ylabel {geo.label:.2f}mm — overlap "
        f"of {geo.band_inner - geo.decoration:.2f}mm. A per-position pin must "
        f"keep collision avoidance too (issue #222) — {geo}"
    )
    assert geo.clipped <= EPS_MM, f"and stay on the canvas — {geo}"


@pytest.mark.parametrize("external", [False, True], ids=["in_frame", "external"])
def test_per_position_lock_pins_only_its_own_position(external):
    """Column 0 keeps exactly its pinned mm; column 1 still auto-measures."""
    fig, _, _ = _build_per_position_lock(external)
    ylabel_space = fig._publiplots_layout.ylabel_space
    assert ylabel_space[0] == pytest.approx(25.0, abs=1e-9), (
        f"pinned ylabel_space[0] must stay at 25.0mm, got {ylabel_space[0]}mm"
    )
    assert ylabel_space[1] != pytest.approx(25.0, abs=1e-9), (
        "ylabel_space[1] was passed as None and must auto-measure, not "
        f"inherit the pin; got {ylabel_space[1]}mm"
    )


# --------------------------------------------------------------------------
# Controls — the auto-measured path (#212 / #220) must not regress.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("side", ["bottom", "left"])
@pytest.mark.parametrize("external", [False, True], ids=["in_frame", "external"])
def test_unpinned_band_clears_decorations_and_stays_on_canvas(side, external):
    """No pin: the reservation grows to fit, so both properties hold outright.

    The clamp must be inert here. It is skipped by design when the
    reservation auto-measures — applying it would read a transiently
    undersized figure mid-convergence and could settle the band short of the
    decorations.
    """
    geo, _, _ = _geometry(side, pin=None, external=external)
    assert geo.band_inner <= geo.decoration + EPS_MM, (
        f"side={side!r} external={external}: unpinned band overlaps the "
        f"decorations by {geo.band_inner - geo.decoration:.2f}mm — {geo}"
    )
    assert geo.clipped <= EPS_MM, f"unpinned band left the canvas — {geo}"


@pytest.mark.parametrize("side", ["bottom", "left"])
def test_generous_pin_matches_the_unpinned_band_placement(side):
    """With room to spare, pinning must not move the band at all.

    The pin fixes the reservation's size; it has no business changing where
    the band sits once there is enough space for it.
    """
    auto, _, _ = _geometry(side, pin=None)
    pinned, _, _ = _geometry(side, pin=GENEROUS[side])
    assert pinned.band_inner == pytest.approx(auto.band_inner, abs=0.05), (
        f"side={side!r}: pinned band at {pinned.band_inner:.2f}mm vs "
        f"unpinned {auto.band_inner:.2f}mm — a generous pin must not change "
        f"where the band sits"
    )
