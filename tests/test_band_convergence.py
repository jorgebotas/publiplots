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
