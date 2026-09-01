"""A colorbar's extent for band layout is its tight bbox (#221).

The rule these tests pin down:

    **strip rect for intra-block pairing, tight bbox for inter-block
    layout.**

A colorbar draws a 4.5mm colour strip plus tick labels that hang off it —
1.73mm past each end of a default horizontal strip, measured. Packing and
along-edge alignment exist to centre visible ink and to keep neighbours
from colliding, so both must count that ink. Two defects followed from
measuring the bare rectangle instead:

1. ``MultiAxesLegendGroup._apply_along_alignment`` measured a colorbar by
   ``obj.ax.get_window_extent()``. Two default colorbars sequenced along
   one top band ended up with their rects a nominal 2.00mm apart and their
   end tick labels **overlapping by 1.45mm**, and a band mixing a colorbar
   with a categorical legend sat up to 6.43mm off the band's centre line.
2. ``add_colorbar``'s overflow pre-check and its along-edge cursor advance
   used the strip's width. Six default colorbars on a 50mm top band packed
   into two rows of three that each drew 59.35mm of ink — overrunning the
   axes edge by 1.23mm on either side — instead of wrapping into three
   rows of two.

What does *not* change is #214: the label still sits over the coloured
band, centred on the strip **rectangle**, not on band-plus-ticks. The
accepted cost is the mirror image — with asymmetric tick labels, centring
the block's tight bbox leaves the strip itself slightly off the band's
centre line.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import publiplots as pp
from publiplots.utils.legend import LegendBuilder, create_legend_handles

SIDES = ["top", "bottom", "left", "right"]
ALIGNS = ["center", "start", "end"]
AXES_MM = (50, 40)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# --- measurement helpers ----------------------------------------------------


def _to_mm(px, fig):
    return px / fig.dpi * 25.4


def _rect(kind, obj):
    return obj.ax.get_window_extent() if kind == "colorbar" \
        else obj.get_window_extent()


def _tight(kind, obj):
    if kind != "colorbar":
        return obj.get_window_extent()
    bb = obj.ax.get_tightbbox()
    return bb if bb is not None else obj.ax.get_window_extent()


def _along(bb, side):
    """(lo, hi) of a bbox on the band's along-edge axis, in pixels."""
    return (bb.x0, bb.x1) if side in ("top", "bottom") else (bb.y0, bb.y1)


def _elements(group):
    return list(group._builder.elements)


def _band_rows(group, fig, side):
    """Group band elements by their outward offset (the visual row).

    Keyed off the element rectangle's outward coordinate rounded to 0.1mm,
    which is what ``add_colorbar`` / ``add_legend`` place on the outward
    cursor; two elements in one row are laid out side by side along the
    edge and are the ones that can collide.
    """
    rows = {}
    for i, (kind, obj) in enumerate(_elements(group)):
        r, t = _rect(kind, obj), _tight(kind, obj)
        key = round(_to_mm(r.y0 if side in ("top", "bottom") else r.x0, fig), 1)
        rows.setdefault(key, []).append((f"{kind}{i}", kind, r, t))
    return rows


def _axes_edge_mm(group, fig, side):
    """(lo, hi) of the anchor axes on the along-edge axis, in mm."""
    axp = group._builder._anchor_ax.get_position()
    fx = fig.get_window_extent()
    if side in ("top", "bottom"):
        return _to_mm(axp.x0 * fx.width, fig), _to_mm(axp.x1 * fx.width, fig)
    return _to_mm(axp.y0 * fx.height, fig), _to_mm(axp.y1 * fx.height, fig)


def _tight_overlaps_mm(group, fig):
    """Pairwise tight-bbox overlaps, in mm. Empty list means no collision."""
    items = [(f"{k}{i}", _tight(k, o))
             for i, (k, o) in enumerate(_elements(group))]
    out = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            (na, a), (nb, b) = items[i], items[j]
            w = _to_mm(min(a.x1, b.x1) - max(a.x0, b.x0), fig)
            h = _to_mm(min(a.y1, b.y1) - max(a.y0, b.y0), fig)
            if w > 0 and h > 0:
                out.append((na, nb, w, h))
    return out


def _outside_canvas_fraction(group, fig):
    """(name, fraction) for every element with any ink off the canvas.

    ``savefig.bbox`` is deliberately ``'standard'``, so anything outside
    the figure rectangle is cropped out of the saved file. #222 was
    rejected on review for moving a band clear of decorations and straight
    off the canvas, so every geometry change here has to be checked
    against it.
    """
    fx = fig.get_window_extent()
    out = []
    for i, (kind, obj) in enumerate(_elements(group)):
        t = _tight(kind, obj)
        area = max(t.width, 1e-9) * max(t.height, 1e-9)
        inside = (max(0.0, min(t.x1, fx.width) - max(t.x0, 0.0))
                  * max(0.0, min(t.y1, fx.height) - max(t.y0, 0.0)))
        frac = 1.0 - inside / area
        if frac > 1e-6:
            out.append((f"{kind}{i}", frac))
    return out


# --- figure builders --------------------------------------------------------


def _legend_handles(n=2):
    return create_legend_handles(
        labels=list("AB")[:n],
        colors=list(pp.color_palette("pastel", n)),
        alpha=0.2, linewidth=1.0,
    )


def _colorbar_band(side, align="center", n=2, label_fmt="cb{}"):
    fig, ax = pp.subplots(1, 1, axes_size=AXES_MM)
    ax.plot([0, 1, 2], [0, 1, 0])
    group = pp.legend(ax, side=side, align=align)
    for i in range(n):
        group.add_colorbar(cmap="viridis", vmin=0, vmax=1,
                           label=label_fmt.format(i))
    return fig, group


def _mixed_band(side, align="center", n_cbar=1):
    fig, ax = pp.subplots(1, 1, axes_size=AXES_MM)
    ax.plot([0, 1, 2], [0, 1, 0])
    group = pp.legend(ax, side=side, align=align)
    for i in range(n_cbar):
        group.add_colorbar(cmap="viridis", vmin=0, vmax=1, label=f"cb{i}")
    group.add_legend(handles=_legend_handles(), label="grp")
    return fig, group


def _multi_axes_band(side, align="center", n=2):
    """Two panels sharing one band.

    ``side='right'`` pins the anchor to the LAST panel: a multi-column
    scope whose band edge is on the far side of the axes it reserves
    against runs the figure away, and that predates this issue entirely.
    """
    fig, axes = pp.subplots(1, 2, axes_size=(35, 28))
    axl = list(np.atleast_1d(axes).flat)
    for a in axl:
        a.plot([0, 1, 2], [0, 1, 0])
    if side == "right":
        group = pp.legend(axes=axl, anchor=axl[-1], side=side, align=align)
    else:
        group = pp.legend(axl, side=side, align=align)
    for i in range(n):
        group.add_colorbar(cmap="viridis", vmin=0, vmax=1, label=f"cb{i}")
    return fig, group


def _draw(fig, times=3):
    # matplotlib's xlabel position lags a resize by one draw, and the band's
    # outward decoration offset is baked on the first draw, so measure only
    # after the geometry has settled.
    for _ in range(times):
        fig.canvas.draw()


# --- part 1: neighbouring blocks must not collide ---------------------------


@pytest.mark.parametrize("side", SIDES)
@pytest.mark.parametrize("align", ALIGNS)
@pytest.mark.parametrize("n", [2, 3])
def test_sequenced_colorbars_do_not_overlap_their_tick_labels(side, align, n):
    """Two or three colorbars in one band must not draw over each other.

    Before #221 the alignment pass measured each strip by its 4.5mm colour
    rectangle, so a top band left the rects a nominal 2.00mm apart and the
    end tick labels — 1.73mm of overhang per side — **overlapped by
    1.45mm**. Nothing was clipped; the numbers were simply drawn on top of
    one another.
    """
    fig, group = _colorbar_band(side, align, n)
    _draw(fig)
    overlaps = _tight_overlaps_mm(group, fig)
    assert not overlaps, (
        f"[{side}, align={align}, n={n}] band elements overlap: "
        + ", ".join(f"{a}/{b} {w:.2f}x{h:.2f}mm" for a, b, w, h in overlaps)
    )


@pytest.mark.parametrize("side", SIDES)
@pytest.mark.parametrize("n", [2, 3])
def test_sequenced_colorbars_keep_the_bands_gap_between_their_ink(side, n):
    """Consecutive blocks in a row are separated by the band's own ``gap``,
    measured between tight bboxes.

    The complement of the overlap test: it is not enough that the ink
    stops touching, the declared 2mm gap has to fall between the ink and
    not between two rectangles whose decorations already cross.
    """
    fig, group = _colorbar_band(side, "center", n)
    _draw(fig)
    gap_mm = group._builder._layout.gap
    for key, row in _band_rows(group, fig, side).items():
        strips = sorted(
            (it for it in row if it[1] == "colorbar"),
            key=lambda it: _along(it[3], side)[0],
        )
        for a, b in zip(strips, strips[1:]):
            measured = _to_mm(
                _along(b[3], side)[0] - _along(a[3], side)[1], fig
            )
            assert measured == pytest.approx(gap_mm, abs=0.15), (
                f"[{side}, n={n}, row@{key}] {a[0]}->{b[0]} tight gap "
                f"{measured:.2f}mm, expected the band gap {gap_mm:.2f}mm"
            )


@pytest.mark.parametrize("side", SIDES)
@pytest.mark.parametrize("case,builder", [
    ("2 colorbars", lambda s: _colorbar_band(s, "center", 2)),
    ("3 colorbars", lambda s: _colorbar_band(s, "center", 3)),
    ("1 colorbar + legend", lambda s: _mixed_band(s, "center", 1)),
    ("2 colorbars + legend", lambda s: _mixed_band(s, "center", 2)),
    ("long label", lambda s: _colorbar_band(
        s, "center", 1, "Expression level (log2 CPM){}")),
    ("multi-axes", lambda s: _multi_axes_band(s, "center", 2)),
])
def test_default_align_centres_the_bands_visible_ink(side, case, builder):
    """``align='center'`` centres what the reader sees, not the rectangles.

    Measured before #221, offset of the band's ink from the anchor edge's
    centre line: ``1 colorbar + legend`` -0.86mm on top/bottom, ``2
    colorbars + legend`` **+6.43mm**, every left/right case -0.72mm. The
    rectangles were centred and the ink was not.
    """
    fig, group = builder(side)
    _draw(fig)
    lo, hi = _axes_edge_mm(group, fig, side)
    spans = [_along(_tight(k, o), side) for k, o in _elements(group)]
    ink_lo = _to_mm(min(s[0] for s in spans), fig)
    ink_hi = _to_mm(max(s[1] for s in spans), fig)
    offset = (ink_lo + ink_hi) / 2 - (lo + hi) / 2
    assert abs(offset) < 0.35, (
        f"[{side}, {case}] the band's ink sits {offset:+.2f}mm off the "
        f"anchor edge's centre line (ink [{ink_lo:.2f},{ink_hi:.2f}], "
        f"edge [{lo:.2f},{hi:.2f}])"
    )


# --- part 2: the row must not overrun the axes edge -------------------------


@pytest.mark.parametrize("side", SIDES)
@pytest.mark.parametrize("align", ALIGNS)
@pytest.mark.parametrize("n", [2, 3, 6])
def test_band_rows_stay_inside_the_axes_edge(side, align, n):
    """No row of the band may draw past either end of the anchor edge.

    Before #221 the overflow pre-check and the cursor advance both counted
    only the colour rectangle, so a row packed more blocks than its ink
    fits: three default strips on a 50mm edge drew 59.35mm and overran by
    1.23mm at each end (``align='center'``) or 1.73mm at the leading end
    (``align='start'``). Six strips repeated it in both of its two rows.
    """
    fig, group = _colorbar_band(side, align, n)
    _draw(fig)
    lo, hi = _axes_edge_mm(group, fig, side)
    # 0.1mm of slack for the float round-trip through figure fractions.
    for key, row in _band_rows(group, fig, side).items():
        r_lo = _to_mm(min(_along(it[3], side)[0] for it in row), fig)
        r_hi = _to_mm(max(_along(it[3], side)[1] for it in row), fig)
        assert r_lo > lo - 0.1 and r_hi < hi + 0.1, (
            f"[{side}, align={align}, n={n}] row@{key} spans "
            f"[{r_lo:.2f},{r_hi:.2f}]mm, past the anchor edge "
            f"[{lo:.2f},{hi:.2f}]mm"
        )


@pytest.mark.parametrize("side", ["top", "bottom"])
def test_six_colorbars_wrap_by_block_width_not_strip_width(side):
    """Six default strips on a 50mm edge wrap into three rows of two.

    18.45mm of ink each plus the 2mm gap leaves room for two per row, not
    the three the strip-width pre-check allowed (3 x 15 + 2 x 2 = 49mm of
    rectangle "fits" a 50mm edge while 59.35mm of ink does not).
    """
    fig, group = _colorbar_band(side, "center", 6)
    _draw(fig)
    strip_rows = {
        key: [it for it in row if it[1] == "colorbar"]
        for key, row in _band_rows(group, fig, side).items()
    }
    strip_rows = {k: v for k, v in strip_rows.items() if v}
    assert len(strip_rows) == 3, (
        f"[{side}] expected 3 rows of strips, got "
        f"{ {k: len(v) for k, v in strip_rows.items()} }"
    )
    for key, row in strip_rows.items():
        assert len(row) == 2, (
            f"[{side}] row@{key} holds {len(row)} strips, expected 2"
        )


# --- #214 must survive: the label stays on the strip RECTANGLE --------------


@pytest.mark.parametrize("side", ["top", "bottom"])
@pytest.mark.parametrize("align", ALIGNS)
@pytest.mark.parametrize("label", ["e", "cb0", "Expression level (log2 CPM)"])
def test_colorbar_label_stays_centred_on_its_strip_rectangle(
    side, align, label
):
    """The intra-block half of the rule, unchanged by #221.

    A label narrower than the strip and a label twice its width must both
    sit centred on the coloured band — not on band-plus-ticks, and not on
    the block's tight bbox, which for a long label is the label itself.
    """
    fig, group = _colorbar_band(side, align, 1, label + "{}")
    _draw(fig)
    els = _elements(group)
    pairs = [(els[i][1], els[i + 1][1]) for i in range(len(els) - 1)
             if els[i][0] == "colorbar" and els[i + 1][0] == "text"]
    assert len(pairs) == 1, f"expected one strip + label pair, got {els}"
    cbar, text = pairs[0]
    strip_rect = cbar.ax.get_window_extent()
    strip_tight = cbar.ax.get_tightbbox() or strip_rect
    lbl = text.get_window_extent()
    rect_c = _to_mm((strip_rect.x0 + strip_rect.x1) / 2, fig)
    tight_c = _to_mm((strip_tight.x0 + strip_tight.x1) / 2, fig)
    lbl_c = _to_mm((lbl.x0 + lbl.x1) / 2, fig)
    assert abs(lbl_c - rect_c) < 0.1, (
        f"[{side}, align={align}, label={label!r}] the label must be "
        f"centred on the strip RECTANGLE: label centre {lbl_c:.3f}mm vs "
        f"rect centre {rect_c:.3f}mm (tight centre {tight_c:.3f}mm)"
    )


# --- the units behind both halves ------------------------------------------


@pytest.mark.parametrize("side,horizontal", [
    ("top", True), ("bottom", True), ("left", False), ("right", False),
])
def test_measure_along_extent_reports_the_tight_bbox_and_its_lead(
    side, horizontal
):
    """``_measure_along_extent`` is the inter-block measurement.

    It must report a colorbar's tight extent (18.45mm along the edge for a
    default horizontal strip, against a 15mm rectangle), the rectangle's
    own extent, and how far the ink leads that rectangle — the offset a
    caller adds to turn "put the ink here" into a registration value.
    """
    fig, group = _colorbar_band(side, "center", 1)
    _draw(fig)
    builder = group._builder
    cbar = next(o for k, o in _elements(group) if k == "colorbar")
    tight, rect, lead = builder._measure_along_extent(
        cbar, horizontal=horizontal
    )
    assert tight > rect + 1.0, (
        f"[{side}] tight extent {tight:.2f}mm should exceed the strip "
        f"rectangle {rect:.2f}mm by the tick-label overhang"
    )
    assert lead > 0.0
    assert lead <= (tight - rect) + 1e-6

    # A Legend's own window extent already IS its tight bbox, so it has
    # nothing hanging off the rectangle the reactor positions.
    legend = group.add_legend(handles=_legend_handles(), label="grp")
    _draw(fig)
    l_tight, l_rect, l_lead = builder._measure_along_extent(
        legend, horizontal=horizontal
    )
    assert l_tight == pytest.approx(l_rect)
    assert l_lead == pytest.approx(0.0)


def test_measure_object_dimensions_still_reports_the_strip_rectangle():
    """The intra-block measurement must stay the colour rectangle.

    ``_measure_object_dimensions`` is what the reactor is handed as
    ``mm_width``/``mm_height`` and what the #214 pairing reasons about, so
    #221's tight bbox must NOT have leaked into it.
    """
    fig, group = _colorbar_band("top", "center", 1)
    _draw(fig)
    cbar = next(o for k, o in _elements(group) if k == "colorbar")
    w, h = group._builder._measure_object_dimensions(cbar)
    rect = cbar.ax.get_window_extent()
    assert w == pytest.approx(_to_mm(rect.width, fig), abs=1e-6)
    assert h == pytest.approx(_to_mm(rect.height, fig), abs=1e-6)


def test_colorbar_block_along_geometry_is_pure_arithmetic():
    """The one place the two halves of the rule meet.

    Pure function, so assert it directly rather than through a figure.
    """
    geom = LegendBuilder._colorbar_block_along_geometry

    # No label: the block is the strip's ink, and the rectangle sits
    # ``lead`` into it.
    assert geom(18.45, 15.0, 1.725, 0.0) == pytest.approx(
        (18.45, 1.725, 1.725)
    )

    # A strip with no overhang and no label reduces to the historical
    # rect-based behaviour exactly.
    assert geom(15.0, 15.0, 0.0, 0.0) == pytest.approx((15.0, 0.0, 0.0))

    # Label narrower than the strip: it hides inside the strip's ink, so
    # the block is still the strip's tight extent, and the label is
    # centred on the strip RECTANGLE (#214).
    extent, strip_off, label_off = geom(18.45, 15.0, 1.725, 4.0)
    assert extent == pytest.approx(18.45)
    assert strip_off == pytest.approx(1.725)
    # label rect centre == strip rect centre
    assert label_off + 4.0 / 2 == pytest.approx(strip_off + 15.0 / 2)

    # Label wider than the whole strip block: the block grows to the
    # label's union, both offsets shift, and the label is STILL centred on
    # the strip rectangle.
    extent, strip_off, label_off = geom(18.45, 15.0, 1.725, 31.33)
    assert extent == pytest.approx(31.33)
    assert label_off == pytest.approx(0.0)
    assert label_off + 31.33 / 2 == pytest.approx(strip_off + 15.0 / 2)


# --- nothing may leave the canvas -----------------------------------------


@pytest.mark.parametrize("side", SIDES)
@pytest.mark.parametrize("align", ALIGNS)
@pytest.mark.parametrize("case,builder", [
    ("2 colorbars", lambda s, a: _colorbar_band(s, a, 2)),
    ("6 colorbars", lambda s, a: _colorbar_band(s, a, 6)),
    ("2 colorbars + legend", lambda s, a: _mixed_band(s, a, 2)),
    ("multi-axes", lambda s, a: _multi_axes_band(s, a, 2)),
])
def test_no_band_element_leaves_the_figure_canvas(side, align, case, builder):
    """``savefig.bbox`` is ``'standard'``: off-canvas ink is deleted.

    #222 was rejected on review for moving a band clear of the axes
    decorations and straight off the figure, where the save cropped up to
    100% of the legend away. Any change to band geometry has to be held to
    this.
    """
    fig, group = builder(side, align)
    _draw(fig)
    outside = _outside_canvas_fraction(group, fig)
    assert not outside, (
        f"[{side}, align={align}, {case}] elements cropped by the figure "
        "canvas: "
        + ", ".join(f"{n} {f * 100:.1f}%" for n, f in outside)
    )


# --- the reactor must converge --------------------------------------------


@pytest.mark.parametrize("side", SIDES)
@pytest.mark.parametrize("n", [2, 6])
def test_band_geometry_is_identical_across_draws_and_settle(side, n):
    """An alignment change is exactly the kind that oscillates.

    Draw 0 legitimately differs on some sides: the outward decoration
    offset is baked in by the first draw's layout pass. From draw 1 on,
    and across ``settle()``, every element's tight bbox must be bit-stable.
    """
    fig, group = _colorbar_band(side, "center", n)

    def snapshot():
        return tuple(
            tuple(round(_to_mm(v, fig), 4)
                  for v in (t.x0, t.y0, t.x1, t.y1))
            for t in (_tight(k, o) for k, o in _elements(group))
        )

    fig.canvas.draw()
    fig.canvas.draw()
    baseline = snapshot()
    fig.canvas.draw()
    assert snapshot() == baseline, f"[{side}, n={n}] draw 3 moved the band"
    fig._publiplots_auto_layout.settle()
    fig.canvas.draw()
    assert snapshot() == baseline, f"[{side}, n={n}] settle() moved the band"


def test_alignment_measurement_forces_no_nested_canvas_draws():
    """Reading a tight bbox in the alignment pass must stay draw-free.

    The pass runs as a post-refresh reactor callback where matplotlib's
    renderer cache is already current; forcing a fresh figure draw there
    cost O(panels) nested redraws. ``_measure_along_extent`` defaults to
    ``force_draw=False`` for that reason, and ``get_tightbbox()`` must not
    smuggle a draw back in.
    """
    orig = LegendBuilder._fig_canvas_draw_for_measure
    calls = {"n": 0}

    def counting(self):
        calls["n"] += 1
        return orig(self)

    LegendBuilder._fig_canvas_draw_for_measure = counting
    try:
        fig, axes = pp.subplots(2, 2, axes_size=(30, 22))
        for ax in np.atleast_2d(axes).flat:
            ax.plot([0, 1, 2], [0, 1, 0])
            g = pp.legend(ax, side="top")
            g.add_colorbar(cmap="viridis", vmin=0, vmax=1, label="cb")
        calls["n"] = 0  # count only the explicit draw below
        fig.canvas.draw()
        assert calls["n"] <= 1, (
            f"the align pass forced {calls['n']} nested canvas draws"
        )
    finally:
        LegendBuilder._fig_canvas_draw_for_measure = orig


# --- it still saves ------------------------------------------------------


@pytest.mark.parametrize("n", [2, 6])
def test_multi_colorbar_top_band_saves_to_png_and_pdf(tmp_path, n):
    """Neither format may come out blank or cropped at the default dpi."""
    fig, group = _colorbar_band("top", "center", n)
    _draw(fig)
    png, pdf = tmp_path / f"band{n}.png", tmp_path / f"band{n}.pdf"
    pp.savefig(str(png))
    pp.savefig(str(pdf))
    assert png.stat().st_size > 1000
    assert pdf.stat().st_size > 1000

    import matplotlib.image as mpimg
    arr = mpimg.imread(str(png))
    rgb = arr[..., :3] if arr.ndim == 3 and arr.shape[2] >= 3 else arr
    n_colors = len(np.unique(np.round(rgb.reshape(-1, rgb.shape[-1]), 4),
                             axis=0))
    assert n_colors > 5, f"saved PNG looks blank ({n_colors} unique colours)"

    # savefig() settles the layout; the band must still be on the canvas.
    assert not _outside_canvas_fraction(group, fig)


def test_colorbar_band_measures_on_a_pdf_canvas(tmp_path):
    """``get_tightbbox()`` must work off the cached renderer.

    The alignment callback fires during ``fig.savefig('x.pdf')``, and
    FigureCanvasPdf has no ``get_renderer()`` — the reason
    ``_measure_object_dimensions`` never passes one (#115). The tight-bbox
    measurement #221 adds has to keep that property.
    """
    df = pd.DataFrame({
        "x": [0, 1, 2] * 2,
        "y": [0, 1, 2, 0.5, 1.5, 2.5],
        "v": [0.0, 0.25, 0.5, 0.75, 1.0, 0.5],
    })
    fig, ax = pp.subplots(1, 1, axes_size=AXES_MM)
    pp.scatterplot(data=df, x="x", y="y", hue="v", ax=ax)
    group = pp.legend(ax, side="top")
    group.add_colorbar(cmap="viridis", vmin=0, vmax=1, label="second")
    for ext in ("pdf", "svg", "png"):
        fig.savefig(tmp_path / f"ok.{ext}")
        assert (tmp_path / f"ok.{ext}").exists()
