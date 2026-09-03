"""A twin axes occupies its parent's cell and must be measured with it (#243).

``SubplotsAutoLayout._measure`` visits ``fig._publiplots_axes``, which never
contains an ``ax.twinx()`` / ``ax.twiny()`` axes. A twin's tick labels
therefore reserved no space at all and were cropped by ``savefig.bbox``,
which is deliberately ``"standard"`` in this project.

No legend is involved in any of this: the defect reproduces on a bare grid.
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from matplotlib.ticker import FixedLocator, FixedFormatter
import matplotlib.pyplot as plt

import publiplots as pp

MM = 25.4
AXES_MM = (40, 32)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    return pd.DataFrame({"x": rng.normal(size=40), "y": rng.normal(size=40)})


def _relabel(ax, labels, axis="y"):
    """Pin explicit tick labels, so width is set by us and not by the locator.

    Large *values* are not a way to widen a label: matplotlib switches to
    offset notation and the labels stay short, which is why ``y * 1e6``
    leaves the overflow at its default 2.05mm.
    """
    target = ax.yaxis if axis == "y" else ax.xaxis
    lo, hi = (ax.get_ylim() if axis == "y" else ax.get_xlim())
    target.set_major_locator(FixedLocator([lo + (hi - lo) * f for f in (0.25, 0.5, 0.75)]))
    target.set_major_formatter(FixedFormatter(list(labels)))


def _overflow_mm(fig, artists):
    """Largest excursion of any artist's tight bbox past the canvas, in mm."""
    canvas = fig.get_window_extent()
    worst = 0.0
    for art in artists:
        tight = art.get_tightbbox()
        if tight is None:
            continue
        worst = max(
            worst,
            (tight.x1 - canvas.x1) / fig.dpi * MM,
            (canvas.x0 - tight.x0) / fig.dpi * MM,
            (tight.y1 - canvas.y1) / fig.dpi * MM,
            (canvas.y0 - tight.y0) / fig.dpi * MM,
        )
    return max(0.0, worst)


@pytest.mark.parametrize("grid", [(1, 1), (1, 3), (3, 1), (2, 2)])
@pytest.mark.parametrize("kind", ["twinx", "twiny"])
def test_a_twin_is_reserved_for_on_every_grid(df, grid, kind):
    """A twin's decorations stay inside the canvas."""
    fig, axes = pp.subplots(*grid, axes_size=AXES_MM)
    twins = []
    for ax in np.atleast_1d(axes).flat:
        tw = ax.twinx() if kind == "twinx" else ax.twiny()
        pp.scatterplot(data=df, x="x", y="y", ax=tw)
        twins.append(tw)
    fig.canvas.draw()
    fig.canvas.draw()

    over = _overflow_mm(fig, twins)
    assert over == pytest.approx(0.0, abs=0.05), (
        f"{kind} on a {grid} grid overflows the canvas by {over:.2f}mm"
    )


@pytest.mark.parametrize("labels,floor_mm", [
    (["1.25", "2.50", "3.75"], 5.0),
    (["123456789012"] * 3, 15.0),
    (["A" * 30] * 3, 45.0),
])
def test_reservation_grows_with_the_twins_label_width(df, labels, floor_mm):
    """The reservation tracks the label, not a constant.

    The default-label case overflows by only 2.05mm, which a loose tolerance
    could pass on by accident, so these pin the wide cases: a 30-character
    label overflowed by 48.76mm before the fix. ``floor_mm`` is asserted
    rather than an exact value because glyph widths are font-dependent.
    """
    fig, ax = pp.subplots(1, 1, axes_size=AXES_MM)
    tw = ax.twinx()
    pp.scatterplot(data=df, x="x", y="y", ax=tw)
    _relabel(tw, labels)
    fig.canvas.draw()
    fig.canvas.draw()

    reserved = fig._publiplots_layout.right[0]
    assert reserved >= floor_mm, (
        f"labels {labels[0]!r} reserved only {reserved:.2f}mm on the right"
    )
    over = _overflow_mm(fig, [tw])
    assert over == pytest.approx(0.0, abs=0.05), f"overflows by {over:.2f}mm"


def test_a_twin_does_not_inflate_a_cell_it_fits_inside(df):
    """A twin whose decorations are the smaller ones must not grow the cell.

    The union is a max, not a sum. ``ax.twinx()`` moves the *parent's* ticks
    to the left (matplotlib calls ``ax.yaxis.tick_left()``), so the parent's
    wide labels are reserved as ``ylabel_space`` and the right side carries
    only the twin's narrow ones — the total is unchanged and each side is
    sized by whichever axes actually draws there.
    """
    fig, ax = pp.subplots(1, 1, axes_size=AXES_MM)
    pp.scatterplot(data=df, x="x", y="y", ax=ax)
    ax.yaxis.tick_right()
    _relabel(ax, ["PARENT-LONG-LABEL-XXXX"] * 3)
    tw = ax.twinx()
    pp.scatterplot(data=df, x="x", y="y", ax=tw)
    _relabel(tw, ["1"] * 3)
    fig.canvas.draw()
    fig.canvas.draw()

    layout = fig._publiplots_layout
    assert layout.ylabel_space[0] > 20.0, (
        "the parent's wide labels, which twinx moved to the left, are not "
        f"reserved: ylabel_space={layout.ylabel_space[0]:.2f}mm"
    )
    assert layout.right[0] < 10.0, (
        "the right side is sized by the twin's narrow labels, not the "
        f"parent's wide ones: right={layout.right[0]:.2f}mm"
    )
    assert _overflow_mm(fig, [ax, tw]) == pytest.approx(0.0, abs=0.05)


def test_a_bare_grid_is_unchanged(df):
    """No twins means no new reservation — the fix must be inert here."""
    fig, axes = pp.subplots(1, 3, axes_size=AXES_MM)
    for ax in axes.flat:
        pp.scatterplot(data=df, x="x", y="y", ax=ax)
    fig.canvas.draw()
    fig.canvas.draw()

    assert fig._publiplots_layout.right == pytest.approx((0.0, 0.0, 0.0))


def test_a_twin_on_only_some_cells_reserves_only_those(df):
    """Per-cell, not per-figure: an untwinned column keeps its 0mm."""
    fig, axes = pp.subplots(1, 3, axes_size=AXES_MM)
    for ax in axes.flat:
        pp.scatterplot(data=df, x="x", y="y", ax=ax)
    tw = axes[1].twinx()
    pp.scatterplot(data=df, x="x", y="y", ax=tw)
    _relabel(tw, ["123456789012"] * 3)
    fig.canvas.draw()
    fig.canvas.draw()

    right = fig._publiplots_layout.right
    assert right[1] > 15.0, f"the twinned column reserved {right[1]:.2f}mm"
    assert right[0] == pytest.approx(0.0, abs=0.05)
    assert right[2] == pytest.approx(0.0, abs=0.05)


def test_a_twins_axis_label_is_reserved_too(df):
    """Not just tick labels — the twin's ylabel counts as well."""
    fig, ax = pp.subplots(1, 1, axes_size=AXES_MM)
    tw = ax.twinx()
    pp.scatterplot(data=df, x="x", y="y", ax=tw)
    tw.set_ylabel("a secondary axis label")
    fig.canvas.draw()
    fig.canvas.draw()

    assert _overflow_mm(fig, [tw]) == pytest.approx(0.0, abs=0.05)


@pytest.mark.parametrize("side", ["right", "left", "top", "bottom"])
def test_a_twin_coexists_with_a_legend_band(df, side):
    """The #242 pairing and this measurement change touch the same cell.

    #242 made a band's reservation resolve a twin to its parent's cell; this
    changes what that cell measures. Both must hold at once: the band stays
    on the canvas and the layout still settles.
    """
    frame = df.assign(g=["a", "b"] * 20)
    fig, axes = pp.subplots(1, 2, axes_size=(35, 28))
    twins = []
    for ax in axes.flat:
        tw = ax.twinx()
        pp.scatterplot(data=frame, x="x", y="y", hue="g", ax=tw)
        twins.append(tw)
    group = pp.legend(anchor=twins[0], axes=twins, side=side)
    fig.canvas.draw()
    fig.canvas.draw()

    band = [obj for _, obj in group._builder.elements]
    assert _overflow_mm(fig, twins) == pytest.approx(0.0, abs=0.05)
    canvas = fig.get_window_extent()
    for obj in band:
        ext = obj.ax.get_window_extent() if hasattr(obj, "ax") else obj.get_window_extent()
        assert ext.x0 >= canvas.x0 - 1 and ext.x1 <= canvas.x1 + 1, (
            f"side={side}: a band element left the canvas"
        )


@pytest.mark.parametrize("kind", ["twinx", "twiny"])
def test_a_twinned_layout_settles_and_stays_settled(kind, df):
    """Eight draws, then settle, then both output formats.

    Four draws cannot tell a settle from a runaway — that is how #230 came to
    be filed with a wrong claim about its `left` side. v0.18.0 warns on
    non-convergence, so this also asserts no warning appears.
    """
    import warnings

    fig, axes = pp.subplots(2, 2, axes_size=(30, 24))
    for ax in axes.flat:
        tw = ax.twinx() if kind == "twinx" else ax.twiny()
        pp.scatterplot(data=df, x="x", y="y", ax=tw)

    sizes = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(8):
            fig.canvas.draw()
            sizes.append(tuple(round(v * MM, 4) for v in fig.get_size_inches()))
        fig._publiplots_auto_layout.settle()
        sizes.append(tuple(round(v * MM, 4) for v in fig.get_size_inches()))

    assert len(set(sizes[1:])) == 1, f"never settled: {sizes}"
    assert not [w for w in caught if isinstance(w.message, pp.LayoutConvergenceWarning)]
