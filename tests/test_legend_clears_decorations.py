"""A per-axes legend steps past the axis decorations on its side.

The per-axes legend a plot call draws (``pp.legend(ax)``, or the plot's own
``legend_kws``) used to sit a fixed 2 mm past the axes edge on 'right' and
'top', on the assumption that nothing else lives there. ``tick_right()`` /
``tick_top()`` labels and a twin's tick labels and label do, and the legend
was drawn straight over them.

What counts is deliberately narrow: the axis decorations of the axes and its
twins, and the artists publiplots registers with ``register_decoration``.
Any other artist is left exactly where it was on 'right' and 'top', and
'left' / 'bottom' keep measuring what they always measured — an arbitrary
artist can be positioned off something the layout moves, and stepping past
it then never settles.

Everything is measured in mm from the rendered geometry after a save, which
is what the reader of the file sees.
"""

import io
import warnings

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.legend import Legend
from matplotlib.patches import Ellipse, Rectangle

import publiplots as pp
from publiplots.layout.auto_layout import register_decoration

MM = 25.4
AXES_MM = (40, 30)
GAP_MM = 2.0  # the legend's default outward gap (x_offset)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def df():
    return pd.DataFrame({
        "k": list("abcab" * 2),
        "v": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "g": ["x", "y"] * 5,
    })


def _mm(fig, px):
    return float(px) / fig.dpi * MM


def _save(fig, n=2):
    """Save ``n`` times, failing on a non-convergence warning; return sizes."""
    sizes = []
    with warnings.catch_warnings():
        warnings.simplefilter("error", pp.LayoutConvergenceWarning)
        for _ in range(n):
            fig.savefig(io.BytesIO(), format="png")
            sizes.append(tuple(fig.get_size_inches() * MM))
    return sizes


def _legend(fig):
    found = [
        c for a in fig.axes for c in a.get_children() if isinstance(c, Legend)
    ]
    assert len(found) == 1
    return found[0]


def _extent(fig, artist):
    return artist.get_window_extent(fig.canvas.get_renderer())


def _bar(df, **kwargs):
    fig, ax = pp.subplots(axes_size=AXES_MM)
    pp.barplot(data=df, x="k", y="v", hue="g", ax=ax, **kwargs)
    return fig, ax


def _assert_settled(sizes):
    for size in sizes[1:]:
        assert size == pytest.approx(sizes[0], abs=1e-6)


# --------------------------------------------------------------- side='right'

def test_right_tick_labels_push_the_right_legend_out(df):
    fig, ax = _bar(df)
    ax.yaxis.tick_right()
    _assert_settled(_save(fig))

    leg = _extent(fig, _legend(fig))
    ticks_x1 = max(
        _extent(fig, t).x1 for t in ax.get_yticklabels() if t.get_text()
    )
    # Main put the legend 1.99 mm INSIDE the tick labels' right edge.
    assert _mm(fig, leg.x0 - ticks_x1) == pytest.approx(GAP_MM, abs=0.01)


def test_a_twins_right_tick_labels_and_label_are_cleared(df):
    fig, ax = _bar(df)
    twin = ax.twinx()
    twin.plot([0, 1, 2], [100, 2000, 30000])
    twin.set_ylabel("twin label")
    _assert_settled(_save(fig))

    leg = _extent(fig, _legend(fig))
    twin_x1 = twin.yaxis.get_tightbbox(fig.canvas.get_renderer()).x1
    # Main left the legend 9.90 mm inside the twin's decorations.
    assert _mm(fig, leg.x0 - twin_x1) == pytest.approx(GAP_MM, abs=0.01)


@pytest.mark.parametrize("how", ["twin_label_only", "label_position_right"])
def test_a_right_axis_label_without_tick_labels_is_cleared(df, how):
    """The label's true extent decides whether it is level with the legend.

    ``for_layout_only`` squeezes an axis label to a 1 px strip at its
    centre, which would make it look "not level" with a top-aligned legend.
    """
    fig, ax = _bar(df)
    if how == "twin_label_only":
        twin = ax.twinx()
        twin.set_yticks([])
        twin.set_ylabel("twin only label")
        yaxis = twin.yaxis
    else:
        ax.yaxis.set_label_position("right")
        ax.tick_params(axis="y", labelleft=False)
        ax.set_ylabel("right label")
        yaxis = ax.yaxis
    _assert_settled(_save(fig))

    leg = _extent(fig, _legend(fig))
    label_x1 = yaxis.get_tightbbox(fig.canvas.get_renderer()).x1
    assert _mm(fig, leg.x0 - label_x1) == pytest.approx(GAP_MM, abs=0.01)


def test_a_registered_decoration_past_the_right_edge_is_cleared(df):
    fig, ax = _bar(df)
    text = ax.text(1.02, 0.9, "LABEL", transform=ax.transAxes, clip_on=False)
    register_decoration(ax, text)
    _assert_settled(_save(fig))

    leg = _extent(fig, _legend(fig))
    assert _mm(fig, leg.x0 - _extent(fig, text).x1) == pytest.approx(
        GAP_MM, abs=0.01
    )


def test_a_registered_decoration_not_level_with_the_legend_is_ignored(df):
    """The text sits past the right edge at the bottom of the axes; the
    legend hangs from the top, so they cannot collide."""
    fig, ax = _bar(df)
    text = ax.text(1.02, 0.05, "LOW", transform=ax.transAxes, clip_on=False)
    register_decoration(ax, text)
    _save(fig)

    leg = _extent(fig, _legend(fig))
    assert _extent(fig, text).y1 < leg.y0  # precondition: not level
    axes = ax.get_window_extent()
    assert _mm(fig, leg.x0 - axes.x1) == pytest.approx(GAP_MM, abs=0.01)


def test_a_removed_or_hidden_registered_decoration_is_ignored(df):
    fig, ax = _bar(df)
    gone = ax.text(1.02, 0.9, "GONE", transform=ax.transAxes, clip_on=False)
    hidden = ax.text(1.02, 0.8, "HIDDEN", transform=ax.transAxes, clip_on=False)
    register_decoration(ax, gone, hidden)
    gone.remove()
    hidden.set_visible(False)
    _save(fig)

    leg = _extent(fig, _legend(fig))
    axes = ax.get_window_extent()
    assert _mm(fig, leg.x0 - axes.x1) == pytest.approx(GAP_MM, abs=0.01)


# ----------------------------------- what is NOT measured stays as on main

def _right_legend_gap(fig, ax):
    leg = _extent(fig, _legend(fig))
    return _mm(fig, leg.x0 - ax.get_window_extent().x1)


@pytest.mark.parametrize("coords", [
    "axes", "figure", "annotate_figure_fraction", "annotate_off_legend",
])
def test_an_unregistered_artist_past_the_right_edge_is_not_measured(df, coords):
    """Only what publiplots knows to be fixed counts. A text in figure
    coordinates moves when the layout grows the figure, and one positioned
    off the legend moves with the legend: stepping past either never
    settles. So no unregistered artist is measured on 'right', fixed or
    not — the legend stays where main puts it."""
    fig, ax = _bar(df)
    fig.canvas.draw()
    leg = _legend(fig)
    if coords == "axes":
        ax.text(1.02, 0.9, "note", transform=ax.transAxes, clip_on=False)
    elif coords == "figure":
        axes = ax.get_window_extent()
        ax.text(
            (axes.x1 + 4) / fig.bbox.width, 0.8, "note",
            transform=fig.transFigure,
        )
    elif coords == "annotate_figure_fraction":
        ax.annotate("note", xy=(0.8, 0.8), xycoords="figure fraction")
    else:
        ax.annotate(
            "note", xy=(1, 0.5), xycoords=leg, xytext=(1, 0),
            textcoords="offset points", annotation_clip=False,
        )
    _assert_settled(_save(fig, n=4))
    assert _right_legend_gap(fig, ax) == pytest.approx(GAP_MM, abs=0.01)


@pytest.mark.parametrize("side", ["left", "bottom"])
@pytest.mark.parametrize("kind", ["rectangle", "ellipse", "inset_axes"])
def test_left_and_bottom_still_measure_what_they_always_did(df, side, kind):
    """'left' / 'bottom' step past everything ``ax.get_tightbbox()`` reports
    on their side, as they always have. An earlier attempt at this fix that
    narrowed them to "provably fixed" artists dropped patches and insets
    and overlapped them by 8-13 mm."""
    fig, ax = _bar(df, legend_kws={"side": side})
    if kind == "inset_axes":
        art = ax.inset_axes(
            [-0.45, 0.3, 0.2, 0.4] if side == "left" else [0.3, -0.6, 0.4, 0.2]
        )
    else:
        cls = Rectangle if kind == "rectangle" else Ellipse
        if side == "left":
            xy, w, h = ((-0.35, 0.3), 0.25, 0.4) if cls is Rectangle else (
                (-0.25, 0.5), 0.25, 0.4)
        else:
            xy, w, h = ((0.3, -0.45), 0.4, 0.25) if cls is Rectangle else (
                (0.5, -0.35), 0.4, 0.25)
        art = cls(xy, w, h, transform=ax.transAxes, clip_on=False)
        ax.add_patch(art)
    _assert_settled(_save(fig))

    leg = _extent(fig, _legend(fig))
    other = _extent(fig, art)
    gap = other.x0 - leg.x1 if side == "left" else other.y0 - leg.y1
    # Clear of it (main: +2.00 for the patches, more for the insets, whose
    # tick labels sit further out).
    assert _mm(fig, gap) >= GAP_MM - 0.01


def test_the_title_over_figure_text_settles(df):
    """An earlier attempt lifted the title over a figure-fraction text,
    which the figure's growth then moved: the figure grew on every save."""
    fig, ax = _bar(df)
    ax.set_title("Title")
    fig.canvas.draw()
    axes = ax.get_window_extent()
    ax.text(
        (axes.x0 + axes.x1) / 2 / fig.bbox.width,
        (axes.y1 + 3) / fig.bbox.height, "note", transform=fig.transFigure,
        ha="center", va="bottom",
    )
    _assert_settled(_save(fig, n=4))
    assert getattr(ax, "_pp_deco_titlepad", None) is None


@pytest.mark.parametrize("side", ["left", "bottom", "right", "top"])
def test_axis_decorations_that_are_not_drawn_are_not_cleared(df, side):
    """After ``ax.axis("off")`` no tick label or axis label is drawn, and
    ``Axes.get_tightbbox`` skips them; so must the legend. Measuring the
    undrawn axis pushed a bottom legend 7.39 mm too far out."""
    fig, ax = _bar(df, legend_kws={"side": side})
    ax.set_xlabel("x label")
    ax.set_ylabel("y label")
    if side == "right":
        ax.yaxis.tick_right()
    elif side == "top":
        ax.xaxis.tick_top()
    ax.axis("off")
    _assert_settled(_save(fig))

    leg = _extent(fig, _legend(fig))
    axes = ax.get_window_extent()
    gap = {
        "left": axes.x0 - leg.x1, "right": leg.x0 - axes.x1,
        "bottom": axes.y0 - leg.y1, "top": leg.y0 - axes.y1,
    }[side]
    assert _mm(fig, gap) == pytest.approx(GAP_MM, abs=0.01)


def test_a_hidden_twin_is_not_cleared(df):
    fig, ax = _bar(df)
    twin = ax.twinx()
    twin.plot([0, 1, 2], [100, 2000, 30000])
    twin.set_ylabel("twin label")
    twin.set_visible(False)
    _assert_settled(_save(fig))
    assert _right_legend_gap(fig, ax) == pytest.approx(GAP_MM, abs=0.01)


def test_a_twin_label_the_user_placed_is_not_cleared(df):
    """``set_label_coords`` can put the label anywhere — here in figure
    coordinates, which move as the figure grows; stepping past it grew the
    figure on every save, without a warning. The tick labels still count."""
    fig, ax = _bar(df)
    twin = ax.twinx()
    twin.plot([0, 1], [1e3, 3e4])
    twin.set_ylabel("twin")
    twin.yaxis.set_label_coords(0.9, 0.8, transform=fig.transFigure)
    _save(fig)  # the first save settles the label's own reservation
    _assert_settled(_save(fig, n=4))

    leg = _extent(fig, _legend(fig))
    r = fig.canvas.get_renderer()
    level = [
        _extent(fig, t) for t in twin.get_yticklabels()
        if t.get_text() and t.get_visible()
    ]
    level = [b for b in level if b.y1 > leg.y0 and b.y0 < leg.y1]
    assert level  # precondition: some tick labels are level with it
    ticks_x1 = max(b.x1 for b in level)
    assert _mm(fig, leg.x0 - ticks_x1) == pytest.approx(GAP_MM, abs=0.01)
    assert twin.yaxis.label.get_window_extent(r).width > 0  # still drawn


@pytest.mark.parametrize("fmt", ["pdf", "svg"])
def test_saving_on_a_non_agg_canvas_clears_the_tick_labels(df, fmt):
    """Only Agg canvases have ``get_renderer``; the measurement must go
    through the figure's own renderer, as ``get_tightbbox`` does."""
    from matplotlib.backends.backend_pdf import FigureCanvasPdf
    from matplotlib.backends.backend_svg import FigureCanvasSVG

    fig, ax = _bar(df)
    ax.yaxis.tick_right()
    (FigureCanvasPdf if fmt == "pdf" else FigureCanvasSVG)(fig)
    assert not hasattr(fig.canvas, "get_renderer")  # precondition
    with warnings.catch_warnings():
        warnings.simplefilter("error", pp.LayoutConvergenceWarning)
        for _ in range(2):
            fig.savefig(io.BytesIO(), format=fmt)

    renderer = fig._get_renderer()
    leg = _legend(fig).get_window_extent(renderer)
    ticks_x1 = ax.yaxis.get_tightbbox(renderer).x1
    assert _mm(fig, leg.x0 - ticks_x1) == pytest.approx(GAP_MM, abs=0.01)


# ------------------------------------------------------------------- title

def test_the_title_clears_a_registered_decoration_above_the_axes(df):
    """The title keeps the clearance it has over a bare axes top."""
    ref_fig, ref_ax = _bar(df)
    ref_ax.set_title("Title")
    _save(ref_fig)
    normal = _mm(
        ref_fig,
        _extent(ref_fig, ref_ax.title).y0 - ref_ax.get_window_extent().y1,
    )

    fig, ax = _bar(df)
    ax.set_title("Title")
    text = ax.text(
        0.5, 1.08, "ABOVE", transform=ax.transAxes, ha="center", clip_on=False
    )
    register_decoration(ax, text)
    _assert_settled(_save(fig))

    title = _extent(fig, ax.title)
    txt = _extent(fig, text)
    assert _mm(fig, title.y0 - txt.y1) == pytest.approx(normal, abs=0.01)
    # ... and title_space grew to hold it: nothing is cropped.
    assert title.y1 <= fig.get_window_extent().y1


def test_an_unregistered_text_above_the_axes_leaves_the_title_alone(df):
    fig, ax = _bar(df)
    ax.set_title("Title")
    ax.text(0.5, 1.08, "ABOVE", transform=ax.transAxes, ha="center",
            clip_on=False)
    _save(fig)
    assert ax.titleOffsetTrans._t[1] * 72.0 == pytest.approx(6.0)


def test_a_pad_set_after_a_draw_is_the_new_baseline(df):
    fig, ax = _bar(df)
    ax.set_title("Title")
    text = ax.text(
        0.5, 1.08, "ABOVE", transform=ax.transAxes, ha="center", clip_on=False
    )
    register_decoration(ax, text)
    _save(fig)
    lifted = ax.titleOffsetTrans._t[1] * 72.0

    ax.set_title("Title", pad=30)
    _save(fig)
    # Lifted from the user's 30 pt, not reset to the stale 6 pt baseline.
    assert ax.titleOffsetTrans._t[1] * 72.0 == pytest.approx(
        lifted + 24.0, abs=1e-6
    )
    assert _extent(fig, ax.title).y0 > _extent(fig, text).y1


def test_a_pad_set_after_a_draw_is_honoured_over_a_top_legend(df):
    """The top-legend lift used to keep the pad stashed at the first draw,
    so a larger pad set afterwards was overwritten by the lift."""
    fig, ax = _bar(df, legend_kws={"side": "top"})
    ax.set_title("Title")
    _save(fig)
    lifted = ax.titleOffsetTrans._t[1] * 72.0
    assert lifted < 60.0  # precondition: 60 pt is above the lift

    ax.set_title("Title", pad=60)
    _save(fig)
    assert ax.titleOffsetTrans._t[1] * 72.0 == pytest.approx(60.0)


def _top_ticks_title(df, text_y=None):
    fig, ax = _bar(df)
    ax.xaxis.tick_top()
    ax.set_title("Title")
    text = None
    if text_y is not None:
        text = ax.text(
            0.5, text_y, "txt", fontsize=6, transform=ax.transAxes,
            ha="center", va="bottom", clip_on=False,
        )
        register_decoration(ax, text)
    _assert_settled(_save(fig))
    ticks_y1 = ax.xaxis.get_tightbbox(fig.canvas.get_renderer()).y1
    return fig, ax, _extent(fig, ax.title), ticks_y1, text


@pytest.mark.parametrize("text_y", [1.00, 1.02, 1.04])
def test_a_decoration_below_the_top_tick_labels_leaves_the_title_to_matplotlib(
    df, text_y
):
    """matplotlib already lifts the title above top tick labels, and that
    clears a decoration lower than them too. Growing the pad for it would
    switch matplotlib's lift off and drop the title onto the tick labels."""
    ref_fig, _, ref_title, ref_ticks, _ = _top_ticks_title(df)
    expected = _mm(ref_fig, ref_title.y0 - ref_ticks)

    fig, ax, title, ticks_y1, text = _top_ticks_title(df, text_y)
    assert _extent(fig, text).y1 < ticks_y1  # precondition: below the ticks
    assert _mm(fig, title.y0 - ticks_y1) == pytest.approx(expected, abs=0.01)


def test_a_decoration_above_the_top_tick_labels_lifts_the_title_past_both(df):
    fig, ax, title, ticks_y1, text = _top_ticks_title(df, 1.12)
    txt = _extent(fig, text)
    assert txt.y1 > ticks_y1  # precondition: above the ticks
    assert title.y0 > txt.y1


# ---------------------------------------------------------- colorbar bands

def _colorbar_band(side):
    rng = np.random.default_rng(1)
    data = pd.DataFrame({
        "x": rng.normal(size=60), "y": rng.normal(size=60),
        "z": rng.uniform(0, 1000, 60),
    })
    fig, ax = pp.subplots(axes_size=AXES_MM)
    pp.scatterplot(
        data=data, x="x", y="y", hue="z", ax=ax, legend_kws={"side": side}
    )
    return fig, ax


@pytest.mark.parametrize("side", ["top", "left", "right"])
def test_a_colorbar_band_clears_the_tick_labels_on_its_side(side):
    """The band's near edge is its innermost ink, colorbar ticks included.

    A colorbar's tick labels can hang inward of the strip: a top band's
    strip is pushed out by them (#213) and a left band's sit between the
    strip and the axes. Main: top -1.61 mm, left -3.34 mm, right (with
    ``tick_right()``) -2.05 mm.
    """
    fig, ax = _colorbar_band(side)
    if side == "top":
        ax.xaxis.tick_top()
    elif side == "right":
        ax.yaxis.tick_right()
    _assert_settled(_save(fig))

    r = fig.canvas.get_renderer()
    cax = next(a for a in fig.axes if a is not ax)
    band = cax.get_tightbbox(r)  # strip + its tick labels
    axis = ax.xaxis if side in ("top", "bottom") else ax.yaxis
    ticks = axis.get_tightbbox(r)
    gap = {
        "top": band.y0 - ticks.y1,
        "bottom": ticks.y0 - band.y1,
        "left": ticks.x0 - band.x1,
        "right": band.x0 - ticks.x1,
    }[side]
    assert _mm(fig, gap) == pytest.approx(GAP_MM, abs=0.01)


# ----------------------------------------------- left / bottom / top legends

def test_a_left_legend_clears_left_decorations_drawn_by_a_twin(df):
    """Twins are measured with their cell on the left too (#253)."""
    fig, ax = _bar(df, legend_kws={"side": "left"})
    ax.yaxis.tick_right()
    twin = ax.twinx()
    twin.plot([0, 1, 2], [100, 2000, 30000])
    twin.yaxis.tick_left()
    twin.yaxis.set_label_position("left")
    twin.set_ylabel("twin label")
    _assert_settled(_save(fig))

    leg = _extent(fig, _legend(fig))
    twin_x0 = twin.yaxis.get_tightbbox(fig.canvas.get_renderer()).x0
    assert _mm(fig, twin_x0 - leg.x1) == pytest.approx(GAP_MM, abs=0.01)


def test_a_bottom_legend_still_clears_the_x_label(df):
    fig, ax = _bar(df, legend_kws={"side": "bottom"})
    ax.set_xlabel("Category")
    _assert_settled(_save(fig))

    leg = _extent(fig, _legend(fig))
    xaxis_y0 = ax.xaxis.get_tightbbox(fig.canvas.get_renderer()).y0
    assert _mm(fig, xaxis_y0 - leg.y1) == pytest.approx(GAP_MM, abs=0.01)


def test_a_top_legend_clears_top_tick_labels_and_the_title_clears_it(df):
    ref_fig, ref_ax = _bar(df, legend_kws={"side": "top"})
    ref_ax.set_title("Title")
    _save(ref_fig)
    ref_leg = _extent(ref_fig, _legend(ref_fig))
    # The title's clearance over a top legend on a plain axes.
    expected = _mm(ref_fig, _extent(ref_fig, ref_ax.title).y0 - ref_leg.y1)

    fig, ax = _bar(df, legend_kws={"side": "top"})
    ax.xaxis.tick_top()
    ax.set_title("Title")
    _assert_settled(_save(fig))

    leg = _extent(fig, _legend(fig))
    ticks_y1 = ax.xaxis.get_tightbbox(fig.canvas.get_renderer()).y1
    # Main put the legend 1.61 mm down into the top tick labels.
    assert _mm(fig, leg.y0 - ticks_y1) == pytest.approx(GAP_MM, abs=0.01)
    # ... and the title keeps the same clearance over the legend.
    assert _mm(fig, _extent(fig, ax.title).y0 - leg.y1) == pytest.approx(
        expected, abs=0.01
    )


# --------------------------------------------- the ordinary case is unmoved

def _line(ax, rng):
    t = np.tile(np.arange(30), 2)
    data = pd.DataFrame({
        "t": t, "y": rng.normal(size=60) * 100, "g": ["a"] * 30 + ["b"] * 30,
    })
    pp.lineplot(data=data, x="t", y="y", hue="g", ax=ax)


def _tall_line(ax, rng):
    t = np.tile(np.arange(30), 12)
    data = pd.DataFrame({
        "t": t * 1000, "y": rng.normal(size=t.size),
        "g": np.repeat([f"level {i}" for i in range(12)], 30),
    })
    pp.lineplot(data=data, x="t", y="y", hue="g", ax=ax)


def _scatter(ax, rng):
    data = pd.DataFrame({
        "x": rng.normal(size=60) * 10, "y": rng.normal(size=60) * 100,
        "g": rng.choice(["ctrl", "treat"], 60),
    })
    pp.scatterplot(data=data, x="x", y="y", hue="g", ax=ax)
    ax.set_title("Scatter")


@pytest.mark.parametrize(
    "plot",
    [
        "bar", "bar_title", "bar_wide_title", "bar_top_ticks_title",
        "line", "tall_line", "scatter",
    ],
)
def test_an_ordinary_right_legend_does_not_move(df, plot):
    """Exactly 2.00 mm past the axes, as before — the regression guard.

    'line': its last x tick label overhangs the right edge, at the bottom;
    only the y axis counts on 'right'. 'tall_line' has a legend taller than
    its axes, which the builder starts at x_offset + column_spacing =
    7.0 mm, and it must stay there.
    """
    np.random.seed(0)
    rng = np.random.default_rng(0)
    fig, ax = pp.subplots(axes_size=AXES_MM)
    if plot.startswith("bar"):
        pp.barplot(data=df, x="k", y="v", hue="g", ax=ax)
        if plot == "bar_wide_title":
            ax.set_ylim(0, 10)  # the "10" label straddles the axes top
            ax.set_title("A title far wider than the axes it belongs to")
        elif plot != "bar":
            ax.set_title("Title")
        if plot == "bar_top_ticks_title":
            ax.xaxis.tick_top()
    elif plot == "line":
        _line(ax, rng)
    elif plot == "tall_line":
        plt.close(fig)
        fig, ax = pp.subplots(axes_size=(40, 20))
        _tall_line(ax, rng)
    else:
        _scatter(ax, rng)
    _assert_settled(_save(fig))

    leg = _extent(fig, _legend(fig))
    axes = ax.get_window_extent()
    regs = fig._publiplots_layout_reactor._registrations
    # Exactly where the reactor puts the legend with nothing to clear.
    assert all(r.mm_outward_decoration_offset == 0.0 for r in regs)
    near = GAP_MM if plot != "tall_line" else 7.0
    assert _mm(fig, leg.x0 - axes.x1) == pytest.approx(near, abs=0.01)
    assert leg.y1 == pytest.approx(axes.y1, abs=0.01 / MM * fig.dpi)
    if ax.title.get_text():
        assert ax.titleOffsetTrans._t[1] * 72.0 == pytest.approx(6.0)
        assert getattr(ax, "_pp_deco_titlepad", None) is None


def test_a_band_already_clear_of_the_decoration_stays_where_it_was():
    """The band moves only as far as clearing needs, not by the whole
    decoration on top of its own placement: a legend taller than its axes
    starts at 7.0 mm, which already clears these tick_right() labels by
    more than the gap."""
    rng = np.random.default_rng(0)
    fig, ax = pp.subplots(axes_size=(40, 20))
    _tall_line(ax, rng)
    ax.yaxis.tick_right()
    _assert_settled(_save(fig))

    leg = _extent(fig, _legend(fig))
    axes = ax.get_window_extent()
    ticks_x1 = ax.yaxis.get_tightbbox(fig.canvas.get_renderer()).x1
    assert _mm(fig, leg.x0 - ticks_x1) > GAP_MM  # precondition: clear
    assert _mm(fig, leg.x0 - axes.x1) == pytest.approx(7.0, abs=0.01)


@pytest.mark.parametrize("plot", ["line", "tall_line"])
def test_the_line_cases_really_have_a_right_overhang(plot):
    """Keeps the line guards above honest: the overhang they dodge exists,
    and the tall legend really is level with it."""
    rng = np.random.default_rng(0)
    if plot == "line":
        fig, ax = pp.subplots(axes_size=AXES_MM)
        _line(ax, rng)
    else:
        fig, ax = pp.subplots(axes_size=(40, 20))
        _tall_line(ax, rng)
    _save(fig)
    xaxis = ax.xaxis.get_tightbbox(fig.canvas.get_renderer())
    assert _mm(fig, xaxis.x1 - ax.get_window_extent().x1) > 0.5
    if plot == "tall_line":
        assert _extent(fig, _legend(fig)).y0 < xaxis.y1
