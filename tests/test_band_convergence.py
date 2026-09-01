"""Issue #230 — a multi-axes legend band must converge, whatever the anchor.

``pp.legend(anchor=axes[i], axes=[...], side=...)`` places the band past the
OUTERMOST in-scope axes on that side, but ``SubplotsAutoLayout`` used to
measure the band's overhang from the ANCHOR's edge and write it into the
ANCHOR's row/column reservation. Whenever the anchor was not that outermost
axes the two disagreed, and the disagreement fed back on itself: the overhang
then spanned every intervening axes plus the gaps, and growing the anchor's
cell pushed those axes — and with them the band — further out, so the next
draw measured more again. The figure grew by roughly one cell's width on
EVERY draw, without limit, and since ``savefig`` draws, saving the same figure
twice produced two different sizes.

These tests therefore assert *convergence*, not a final value: a single draw
cannot see this bug — the first draw of a runaway layout looks exactly like
the first draw of a healthy one. Each case draws repeatedly and requires the
figure size and every band element's position to be bit-identical across the
last ``TAIL_DRAWS`` of them, and again after ``settle()`` and after a save to
both a raster and a vector format.

The geometry tests pin the other half of the fix: converging is not enough if
the band converges to the wrong place, so an inner anchor must produce exactly
the same layout as the outermost anchor over the same scope — the case that
always converged, and whose geometry must not move.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import publiplots as pp


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


MM = 25.4
# Draws to run before sampling. A healthy layout settles in 1-3; the extra
# headroom keeps the sampled window clear of the initial settle.
WARMUP_DRAWS = 4
# Length of the sampled window. Four draws is NOT enough to tell a one-time
# settle from a runaway — that is how #230 came to be filed with a wrong
# claim about side='left'.
TAIL_DRAWS = 8

_rng = np.random.default_rng(0)
DF = pd.DataFrame(
    {
        "x": _rng.normal(size=60),
        "y": _rng.normal(size=60),
        "g": _rng.choice(["alpha", "beta", "gamma"], 60),
        "v": _rng.uniform(0.0, 10.0, size=60),
    }
)


def _make(nrows, ncols, anchor_idx, side, scope_idx=None, hue="g", **subplot_kw):
    """A grid of scatterplots with one band anchored to ``axes[anchor_idx]``."""
    fig, axes = pp.subplots(nrows, ncols, axes_size=(40, 32), **subplot_kw)
    flat = list(np.asarray(axes).flat)
    for ax in flat:
        pp.scatterplot(data=DF, x="x", y="y", hue=hue, ax=ax)
    scope = flat if scope_idx is None else [flat[i] for i in scope_idx]
    group = pp.legend(anchor=flat[anchor_idx], axes=scope, side=side)
    return fig, flat, group


def _snapshot(fig, group):
    """Figure size and every band element's box, in mm — exactly comparable.

    Millimetres, not pixels or fractions: the figure is what resizes, so a
    fraction would hide the very drift under test, and mm are dpi-invariant
    at a fixed dpi while reading directly as the layout's own unit.
    """
    auto = fig._publiplots_auto_layout
    dpi = fig.dpi
    boxes = []
    for _, obj in group._builder.elements:
        bb = auto._artist_window_extent(obj)
        if bb is None:
            continue
        boxes.append(
            (
                round(bb.x0 / dpi * MM, 6),
                round(bb.y0 / dpi * MM, 6),
                round(bb.x1 / dpi * MM, 6),
                round(bb.y1 / dpi * MM, 6),
            )
        )
    w, h = fig.get_size_inches()
    return (round(w * MM, 6), round(h * MM, 6), tuple(boxes))


def _draw_series(fig, group, n):
    out = []
    for _ in range(n):
        fig.canvas.draw()
        out.append(_snapshot(fig, group))
    return out


def _assert_converged(fig, group, label):
    """The last TAIL_DRAWS draws must all produce the identical layout."""
    series = _draw_series(fig, group, WARMUP_DRAWS + TAIL_DRAWS)
    tail = series[-TAIL_DRAWS:]
    if len(set(tail)) != 1:
        widths = [s[0] for s in series]
        deltas = [round(b - a, 3) for a, b in zip(widths, widths[1:])]
        heights = [s[1] for s in series]
        dh = [round(b - a, 3) for a, b in zip(heights, heights[1:])]
        pytest.fail(
            f"{label}: layout never converged over "
            f"{WARMUP_DRAWS + TAIL_DRAWS} draws.\n"
            f"  widths  mm: {widths}\n  per-draw dW: {deltas}\n"
            f"  heights mm: {heights}\n  per-draw dH: {dh}"
        )
    return tail[0]


SIDES = ("right", "left", "top", "bottom")


def _grid_cases():
    for nrows, ncols in ((1, 2), (1, 3), (2, 2), (3, 1)):
        for side in SIDES:
            for anchor in range(nrows * ncols):
                yield pytest.param(
                    nrows, ncols, anchor, side,
                    id=f"{nrows}x{ncols}-{side}-anchor{anchor}",
                )


@pytest.mark.parametrize("nrows,ncols,anchor,side", list(_grid_cases()))
def test_band_converges_for_every_anchor_in_scope(nrows, ncols, anchor, side):
    """Every anchor choice must converge — the bug only fires on inner ones."""
    fig, _, group = _make(nrows, ncols, anchor, side)
    _assert_converged(fig, group, f"{nrows}x{ncols} {side} anchor={anchor}")


@pytest.mark.parametrize("side", SIDES)
@pytest.mark.parametrize("anchor", [0, 1])
def test_band_converges_for_a_strict_subset_scope(side, anchor):
    """A scope narrower than the grid still has an outermost in-scope axes."""
    fig, _, group = _make(1, 3, anchor, side, scope_idx=[0, 1])
    _assert_converged(fig, group, f"1x3 {side} anchor={anchor} scope=[0,1]")


@pytest.mark.parametrize("side", SIDES)
def test_band_converges_for_a_single_axes_scope(side):
    """The degenerate scope, where anchor and outermost axes coincide."""
    fig, _, group = _make(1, 2, 0, side, scope_idx=[0])
    _assert_converged(fig, group, f"1x2 {side} single-axes scope")


@pytest.mark.parametrize("side", SIDES)
def test_in_frame_legend_converges(side):
    """``pp.legend(ax)`` — the in-frame form, measured through the tightbbox."""
    fig, axes = pp.subplots(1, 2, axes_size=(40, 32))
    for ax in axes.flat:
        pp.scatterplot(data=DF, x="x", y="y", hue="g", ax=ax)
    group = pp.legend(axes[0], side=side)
    _assert_converged(fig, group, f"in-frame {side}")


@pytest.mark.parametrize("side", SIDES)
def test_figure_anchored_band_converges(side):
    """A figure-anchored band reserves a figure-level scalar, not a cell."""
    fig, axes = pp.subplots(2, 2, axes_size=(40, 32))
    for ax in axes.flat:
        pp.scatterplot(data=DF, x="x", y="y", hue="g", ax=ax)
    group = pp.legend(side=side, figure=fig)
    _assert_converged(fig, group, f"figure-anchored {side}")


@pytest.mark.parametrize("side", SIDES)
@pytest.mark.parametrize("anchor", [0, 1, 2])
def test_colorbar_band_converges(side, anchor):
    """A continuous hue produces a colorbar band, measured by tight bbox (#221)."""
    fig, _, group = _make(1, 3, anchor, side, hue="v")
    _assert_converged(fig, group, f"colorbar 1x3 {side} anchor={anchor}")


@pytest.mark.parametrize(
    "pin",
    [
        {"right": (3.0, None, None)},
        {"right": (None, None, 3.0)},
        {"right": (None, None, 30.0)},
    ],
)
def test_band_converges_under_a_pinned_right_reservation(pin):
    """A pin freezes one cell (#222); the band must still settle around it."""
    fig, _, group = _make(1, 3, 0, "right", **pin)
    _assert_converged(fig, group, f"1x3 right anchor=0 pin={pin}")


@pytest.mark.parametrize("hue", ["g", "v"])
def test_colorbar_and_legend_converge_under_a_pinned_ylabel_space(hue):
    """The other pinned side, for both band kinds."""
    fig, _, group = _make(
        1, 3, 2, "left", hue=hue, ylabel_space=(8.0, None, None)
    )
    _assert_converged(fig, group, f"1x3 left anchor=2 pinned ylabel hue={hue}")


@pytest.mark.parametrize("side", SIDES)
def test_settle_and_savefig_do_not_move_a_converged_band(tmp_path, side):
    """savefig draws — a non-convergent layout grows every time it is saved.

    PNG and PDF both, because ``savefig`` renders at its own dpi and the
    band's extents come from text metrics, which are not dpi-invariant.
    """
    fig, _, group = _make(1, 3, 0, side)
    converged = _assert_converged(fig, group, f"1x3 {side} anchor=0")

    fig._publiplots_auto_layout.settle()
    assert _snapshot(fig, group) == converged, f"settle() moved the {side} band"

    plt.figure(fig.number)
    pp.savefig(str(tmp_path / "band.png"))
    assert _snapshot(fig, group) == converged, f"PNG save moved the {side} band"

    plt.figure(fig.number)
    pp.savefig(str(tmp_path / "band.pdf"))
    assert _snapshot(fig, group) == converged, f"PDF save moved the {side} band"


@pytest.mark.parametrize("side", SIDES)
def test_repeated_saves_keep_the_figure_the_same_size(tmp_path, side):
    """The user-visible symptom on the issue: save twice, get two sizes."""
    fig, _, group = _make(1, 2, 0, side)
    sizes = []
    for i in range(4):
        plt.figure(fig.number)
        pp.savefig(str(tmp_path / f"s{i}.png"))
        sizes.append(tuple(round(v * MM, 6) for v in fig.get_size_inches()))
    assert len(set(sizes)) == 1, f"repeated saves resized the figure: {sizes}"


# --- geometry: converging to the RIGHT place -----------------------------

_OUTERMOST = {"right": -1, "bottom": -1, "left": 0, "top": 0}


def _outer_axes_edge_mm(fig, group, side):
    """The outermost in-scope axes edge on ``side``, mm from the figure origin."""
    dpi = fig.dpi
    boxes = [ax.get_window_extent() for ax in group._scope_anchor.scope_axes()]
    if side == "right":
        return max(b.x1 for b in boxes) / dpi * MM
    if side == "left":
        return min(b.x0 for b in boxes) / dpi * MM
    if side == "top":
        return max(b.y1 for b in boxes) / dpi * MM
    return min(b.y0 for b in boxes) / dpi * MM


def _band_gap_mm(fig, group, side):
    """Clearance between the outermost in-scope axes and the band's near edge."""
    edge = _outer_axes_edge_mm(fig, group, side)
    boxes = _snapshot(fig, group)[2]
    assert boxes, "band rendered no elements"
    if side == "right":
        return min(b[0] for b in boxes) - edge
    if side == "left":
        return edge - max(b[2] for b in boxes)
    if side == "top":
        return min(b[1] for b in boxes) - edge
    return edge - max(b[3] for b in boxes)


@pytest.mark.parametrize("side", SIDES)
@pytest.mark.parametrize("nrows,ncols", [(1, 3), (2, 2)])
def test_inner_anchor_lays_out_identically_to_the_outermost_anchor(
    nrows, ncols, side
):
    """The anchor picks which entries are collected, not where the band sits.

    The outermost-anchor case is the one that already converged before #230,
    so it is the reference: every other anchor over the same scope must land
    on exactly the same figure size and the same band boxes.
    """
    n = nrows * ncols
    outer_first = side in ("left", "top")
    reference_anchor = 0 if outer_first else n - 1
    fig, _, group = _make(nrows, ncols, reference_anchor, side)
    reference = _assert_converged(fig, group, f"reference anchor={reference_anchor}")
    plt.close(fig)

    for anchor in range(n):
        if anchor == reference_anchor:
            continue
        fig, _, group = _make(nrows, ncols, anchor, side)
        got = _assert_converged(fig, group, f"anchor={anchor}")
        assert got == reference, (
            f"{nrows}x{ncols} {side}: anchor={anchor} laid out differently "
            f"from the outermost anchor={reference_anchor}\n"
            f"  outermost: {reference}\n  this one:  {got}"
        )
        plt.close(fig)


@pytest.mark.parametrize("side", SIDES)
def test_band_clears_the_outermost_in_scope_axes(side):
    """Positive clearance from the axes it is placed past, on every anchor."""
    for anchor in range(3):
        fig, _, group = _make(1, 3, anchor, side)
        _assert_converged(fig, group, f"1x3 {side} anchor={anchor}")
        gap = _band_gap_mm(fig, group, side)
        assert gap > 0.0, (
            f"1x3 {side} anchor={anchor}: band overlaps the outermost "
            f"in-scope axes by {-gap:.2f} mm"
        )
        plt.close(fig)


@pytest.mark.parametrize("side", ["right", "left"])
@pytest.mark.parametrize("anchor", [0, 1, 2])
def test_band_stays_on_the_canvas(side, anchor):
    """An auto-measured reservation grows around the band, so nothing is cropped.

    Only the two horizontal sides: a 'top' / 'bottom' band is laid out along
    the grid's width and can legitimately be wider than a one-column figure,
    which is a separate concern from #230 and unchanged by it.
    """
    fig, _, group = _make(1, 3, anchor, side)
    _assert_converged(fig, group, f"1x3 {side} anchor={anchor}")
    w_mm, h_mm, boxes = _snapshot(fig, group)
    for x0, y0, x1, y1 in boxes:
        assert x0 >= -1e-6 and x1 <= w_mm + 1e-6, (
            f"1x3 {side} anchor={anchor}: band element spans "
            f"[{x0:.2f}, {x1:.2f}] mm outside a {w_mm:.2f} mm canvas"
        )
        assert y0 >= -1e-6 and y1 <= h_mm + 1e-6, (
            f"1x3 {side} anchor={anchor}: band element spans "
            f"[{y0:.2f}, {y1:.2f}] mm outside a {h_mm:.2f} mm canvas"
        )


# --- pinned bands: geometry, not just convergence ------------------------
#
# A pin (#222) is the one case where the row/column does NOT grow around the
# band, so the band's outward step is clamped to keep it on the canvas. #230
# moved the reference for that clamp — the decoration measurement, the
# "is this cell pinned?" test and the space-available measurement all now read
# the outermost in-scope cell rather than the anchor's. Convergence alone
# cannot see that move: both the old and the new reference converge, they just
# converge to different places. These tests assert the placement.


def _band_axis_overflow_mm(fig, group, side):
    """How far the band spills off the canvas along its OWN outward axis.

    Only that axis: a 'top' / 'bottom' band is laid out along the grid's
    width and can legitimately overhang a one-column figure sideways, which
    is a pre-existing property of a wide band and nothing to do with the
    clamp under test.
    """
    w_mm, h_mm, boxes = _snapshot(fig, group)
    assert boxes, "band rendered no elements"
    if side == "right":
        return max(0.0, max(b[2] for b in boxes) - w_mm)
    if side == "left":
        return max(0.0, -min(b[0] for b in boxes))
    if side == "top":
        return max(0.0, max(b[3] for b in boxes) - h_mm)
    return max(0.0, -min(b[1] for b in boxes))


# Whole-side pins and per-position pins, generous and too-tight, on all four
# sides. The per-position ones are what separate the anchor's cell from the
# band's cell: `right=(3.0, None, None)` pins the cell an anchor=0 band used
# to reserve in and leaves the cell it actually occupies free, and the
# `(None, None, x)` forms are that case mirrored.
_PINNED_GEOMETRY_CASES = [
    ("left", {"ylabel_space": 8.0}),
    ("left", {"ylabel_space": 25.0}),
    ("left", {"ylabel_space": (8.0, None, None)}),
    ("left", {"ylabel_space": (None, None, 8.0)}),
    ("right", {"right": 3.0}),
    ("right", {"right": (3.0, None, None)}),
    ("right", {"right": (None, None, 3.0)}),
    ("right", {"right": (None, None, 30.0)}),
    ("bottom", {"xlabel_space": 8.0}),
    ("bottom", {"xlabel_space": 20.0}),
    ("bottom", {"xlabel_space": (None, None, 6.0)}),
    ("top", {"title_space": 6.0}),
]


@pytest.mark.parametrize(
    "side,pin",
    _PINNED_GEOMETRY_CASES,
    ids=[f"{s}-{sorted(p)[0]}{sorted(p.values())[0]}" for s, p in
         _PINNED_GEOMETRY_CASES],
)
def test_pinned_band_geometry_is_anchor_independent(side, pin):
    """Under a pin, the anchor must not change where the band lands.

    The anchor selects which entries the band collects. Everything
    positional — how far past the decorations it steps, whether that step is
    clamped, and how much room the clamp thinks it has — belongs to the cell
    the band occupies, which is the outermost in-scope one whatever the
    anchor. The outermost anchor is the reference: it is the configuration
    that behaved correctly before #230, so it is what the others must match.

    Also asserts the resulting spill off the canvas is no worse than the
    reference's. Both halves matter: anchor-independence catches a reference
    that moved, and the overflow bound catches one that moved for every
    anchor at once.
    """
    nrows, ncols = (3, 1) if side in ("top", "bottom") else (1, 3)
    n = nrows * ncols
    reference_anchor = 0 if side in ("left", "top") else n - 1

    fig, _, group = _make(nrows, ncols, reference_anchor, side, **pin)
    reference = _assert_converged(
        fig, group, f"{side} pin={pin} anchor={reference_anchor}"
    )
    reference_overflow = _band_axis_overflow_mm(fig, group, side)
    plt.close(fig)

    for anchor in range(n):
        if anchor == reference_anchor:
            continue
        fig, _, group = _make(nrows, ncols, anchor, side, **pin)
        got = _assert_converged(fig, group, f"{side} pin={pin} anchor={anchor}")
        overflow = _band_axis_overflow_mm(fig, group, side)
        assert got == reference, (
            f"{side} pin={pin}: anchor={anchor} placed the band differently "
            f"from the outermost anchor={reference_anchor}. The pinned-cell "
            f"decision must follow the cell the band occupies, not the "
            f"anchor.\n  outermost: {reference}\n  this one:  {got}"
        )
        assert overflow <= reference_overflow + 1e-6, (
            f"{side} pin={pin}: anchor={anchor} spills {overflow:.3f} mm off "
            f"the canvas against {reference_overflow:.3f} mm for the "
            f"outermost anchor — the clamp is measuring the wrong edge"
        )
        plt.close(fig)


# --- twin axes resolve to their parent's cell ----------------------------


def _twin_figure(nrows, ncols, side, anchor_idx=None, twin="x", scope_idx=None):
    """A grid whose data lives on twin axes, with a band over the twins.

    A twin is not in ``fig._publiplots_axes``, so before it was resolved to
    its parent's cell nothing in the scope matched the grid and the band fell
    back to the anchor — i.e. to the exact #230 loop the rest of this module
    is about, on a figure that never gets an ``anchor=`` argument.
    """
    fig, axes = pp.subplots(nrows, ncols, axes_size=(40, 32))
    flat = list(np.asarray(axes).flat)
    twins = [(ax.twinx() if twin == "x" else ax.twiny()) for ax in flat]
    for tw in twins:
        pp.scatterplot(data=DF, x="x", y="y", hue="g", ax=tw)
    scope = twins if scope_idx is None else [twins[i] for i in scope_idx]
    if anchor_idx is None:
        group = pp.legend(scope, side=side)
    else:
        group = pp.legend(anchor=twins[anchor_idx], axes=scope, side=side)
    return fig, flat, twins, group


@pytest.mark.parametrize("side", SIDES)
@pytest.mark.parametrize("twin", ["x", "y"])
def test_twin_axes_scope_converges(side, twin):
    """The plain dual-axis pattern: no ``anchor=``, band over the twins."""
    fig, _, _, group = _twin_figure(1, 3, side, twin=twin)
    _assert_converged(fig, group, f"1x3 {side} twin{twin} scope")


@pytest.mark.parametrize("side", SIDES)
@pytest.mark.parametrize("anchor", [0, 1, 2])
def test_twin_axes_scope_converges_for_every_anchor(side, anchor):
    fig, _, _, group = _twin_figure(1, 3, side, anchor_idx=anchor)
    _assert_converged(fig, group, f"1x3 {side} twin anchor={anchor}")


@pytest.mark.parametrize("nrows,ncols", [(2, 2), (3, 1)])
@pytest.mark.parametrize("side", ["right", "bottom"])
def test_twin_axes_scope_converges_on_multi_row_grids(nrows, ncols, side):
    fig, _, _, group = _twin_figure(nrows, ncols, side)
    _assert_converged(fig, group, f"{nrows}x{ncols} {side} twin scope")


@pytest.mark.parametrize("anchor", [0, 1, 2])
def test_twin_axes_band_reserves_space_in_the_twins_own_column(anchor):
    """Converging is not enough — the reservation must land in the right cell.

    A twin resolved to the anchor's cell instead of its own still converged
    whenever the anchor happened to be the outermost member (the measurement
    reference then translates with the band), but the space was reserved in
    the wrong column and the band hung off the canvas. Only the band's own
    axis is checked: the twin's tick labels are a separate matter, below.
    """
    fig, _, _, group = _twin_figure(1, 3, "right", anchor_idx=anchor)
    _assert_converged(fig, group, f"1x3 right twin anchor={anchor}")
    overflow = _band_axis_overflow_mm(fig, group, "right")
    assert overflow == pytest.approx(0.0, abs=1e-6), (
        f"twin scope anchor={anchor}: band hangs {overflow:.2f} mm off the "
        f"canvas — its space was reserved in the wrong column"
    )


def test_twin_scope_is_anchor_independent():
    """Every anchor over a twin scope must produce the same layout."""
    fig, _, _, group = _twin_figure(1, 3, "right", anchor_idx=2)
    reference = _assert_converged(fig, group, "twin reference anchor=2")
    plt.close(fig)
    for anchor in (0, 1):
        fig, _, _, group = _twin_figure(1, 3, "right", anchor_idx=anchor)
        got = _assert_converged(fig, group, f"twin anchor={anchor}")
        assert got == reference, (
            f"twin scope anchor={anchor} laid out differently from "
            f"anchor=2:\n  outermost: {reference}\n  this one:  {got}"
        )
        plt.close(fig)
