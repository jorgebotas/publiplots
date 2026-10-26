"""anchor="inside" for box_stats: on the forward side of the median line,
inside the body; half-bodies anchor on their seam."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgba

import publiplots as pp


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _df() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return pd.DataFrame([{"cat": c, "h": h, "v": float(v)}
                         for c in "ABC" for h in ("x", "y")
                         for v in rng.normal("ABC".index(c), 0.5, 40)])


def _bbox(ax, t):
    ax.figure.canvas.draw()
    return t.get_window_extent(ax.figure.canvas.get_renderer())


def _seam(box) -> float:
    return box.center_pos + (box.cat_half_width if box.half == "low"
                             else -box.cat_half_width)


def test_inside_vertical_sits_above_the_median_centred_on_the_body():
    fig, ax = pp.subplots()
    pp.violinplot(data=_df(), x="cat", y="v", inner=None, legend=False, ax=ax)
    texts = pp.annotate(ax, kind="box_stats", anchor="inside")
    for t, box in zip(texts, ax._publiplots_box_meta.boxes):
        assert t.get_position() == pytest.approx((box.center_pos, box.stats["median"]))
        assert (t.get_ha(), t.get_va()) == ("center", "baseline")
        median_px = ax.transData.transform((0, box.stats["median"]))[1]
        assert _bbox(ax, t).y0 > median_px


def test_inside_horizontal_sits_right_of_the_median_centred_on_the_row():
    fig, ax = pp.subplots()
    pp.violinplot(data=_df(), y="cat", x="v", inner=None, legend=False, ax=ax)
    texts = pp.annotate(ax, kind="box_stats", anchor="inside")
    for t, box in zip(texts, ax._publiplots_box_meta.boxes):
        assert t.get_position() == pytest.approx((box.stats["median"], box.center_pos))
        assert (t.get_ha(), t.get_va()) == ("left", "center")


def test_inside_never_flips_side_near_the_axis_end():
    df = _df()
    fig, ax = pp.subplots()
    pp.violinplot(data=df, y="cat", x="v", inner=None, legend=False, ax=ax)
    ax.set_xlim(right=df.v.max())
    texts = pp.annotate(ax, kind="box_stats", anchor="inside")
    assert {t.get_ha() for t in texts} == {"left"}


@pytest.mark.parametrize("orient", ["v", "h"])
def test_split_labels_sit_on_their_own_side_of_the_seam(orient):
    xy = dict(x="cat", y="v") if orient == "v" else dict(y="cat", x="v")
    fig, ax = pp.subplots(axes_size=(80, 60))
    pp.violinplot(data=_df(), hue="h", split=True, inner=None, legend=False,
                  ax=ax, **xy)
    texts = pp.annotate(ax, kind="box_stats", anchor="inside")
    axis = 0 if orient == "v" else 1
    for t, box in zip(texts, ax._publiplots_box_meta.boxes):
        seam_px = ax.transData.transform((_seam(box),) * 2)[axis]
        body_px = ax.transData.transform((box.center_pos,) * 2)[axis]
        bb = _bbox(ax, t)
        lo, hi = (bb.x0, bb.x1) if axis == 0 else (bb.y0, bb.y1)
        if body_px > seam_px:
            assert lo > seam_px
        else:
            assert hi < seam_px


@pytest.mark.parametrize("side", ["left", "right"])
def test_side_halves_anchor_on_the_seam(side):
    """Horizontal, so seaborn's inverted categorical y axis is exercised."""
    fig, ax = pp.subplots(axes_size=(80, 60))
    pp.violinplot(data=_df(), y="cat", x="v", side=side, inner=None,
                  legend=False, ax=ax)
    texts = pp.annotate(ax, kind="box_stats", anchor="inside")
    for t, box in zip(texts, ax._publiplots_box_meta.boxes):
        seam_px = ax.transData.transform((0, _seam(box)))[1]
        body_px = ax.transData.transform((0, box.center_pos))[1]
        bb = _bbox(ax, t)
        assert (bb.y0 > seam_px) if body_px > seam_px else (bb.y1 < seam_px)


@pytest.mark.parametrize("hue", [None, "cat"])
@pytest.mark.parametrize("orient", ["v", "h"])
def test_split_seam_at_the_slot_edge_keeps_labels_on_their_body(orient, hue):
    """split=True without a separate hue column puts the seam at a slot
    edge, not at the category position; the label must still sit on the
    seam side of its own drawn body."""
    xy = dict(x="cat", y="v") if orient == "v" else dict(y="cat", x="v")
    fig, ax = pp.subplots(axes_size=(80, 60))
    pp.violinplot(data=_df(), hue=hue, split=True, inner=None, legend=False,
                  ax=ax, **xy)
    texts = pp.annotate(ax, kind="box_stats", anchor="inside")
    renderer = fig.canvas.get_renderer()
    axis = 0 if orient == "v" else 1
    for t, box in zip(texts, ax._publiplots_box_meta.boxes):
        seam_px = ax.transData.transform((_seam(box),) * 2)[axis]
        body = box.artist.get_window_extent(renderer)
        bb = _bbox(ax, t)
        if axis == 0:
            body_lo, body_hi, lo, hi = body.x0, body.x1, bb.x0, bb.x1
        else:
            body_lo, body_hi, lo, hi = body.y0, body.y1, bb.y0, bb.y1
        assert body_lo <= lo and hi <= body_hi
        assert min(abs(lo - seam_px), abs(hi - seam_px)) < 0.25 * (body_hi - body_lo)


def test_inside_contrasts_with_the_drawn_fill():
    fig, ax = pp.subplots()
    pp.violinplot(data=_df(), x="cat", y="v", hue="cat", fill=True, alpha=1.0,
                  palette={"A": "#0000ff", "B": "#ffff00", "C": "#0000ff"},
                  inner=None, legend=False, ax=ax)
    texts = pp.annotate(ax, kind="box_stats", anchor="inside")
    white, ink = to_rgba("#ffffff"), to_rgba(plt.rcParams["text.color"])
    assert [to_rgba(t.get_color()) for t in texts] == [white, ink, white]


def test_inside_ink_under_dark_background_reads_on_a_light_fill():
    with plt.style.context("dark_background"):
        fig, ax = pp.subplots()
        pp.violinplot(data=_df(), x="cat", y="v", hue="cat", fill=True,
                      alpha=1.0, palette={"A": "#0000ff", "B": "#ffff00",
                                          "C": "#0000ff"},
                      inner=None, legend=False, ax=ax)
        texts = pp.annotate(ax, kind="box_stats", anchor="inside")
    white, black = to_rgba("white"), to_rgba("black")
    assert [to_rgba(t.get_color()) for t in texts] == [white, black, white]


@pytest.mark.parametrize("plot, kind", [(pp.boxplot, "box_stats"),
                                         (pp.violinplot, "violin_stats")])
def test_inside_contrasts_with_the_translucent_face_not_the_palette(plot, kind):
    # At the default alpha=0.1 a saturated dark palette colour is drawn as a
    # pale tint, so the label needs ink even though the hue colour is dark.
    # The hue must differ from the categorical axis: a repeated hue records
    # no hue value, and its hue_color falls back to the drawn face.
    fig, ax = pp.subplots()
    plot(data=_df(), x="cat", y="v", hue="h", fill=True, legend=False,
         palette={"x": "#000080", "y": "#000080"}, ax=ax)
    boxes = ax._publiplots_box_meta.boxes
    assert {b.hue_color for b in boxes} == {to_rgba("#000080")}
    assert [b.face_color[3] for b in boxes] == pytest.approx([0.1] * 6)
    texts = pp.annotate(ax, kind=kind, anchor="inside")
    assert len(texts) == len(boxes) == 6
    assert {to_rgba(t.get_color()) for t in texts} == {to_rgba(plt.rcParams["text.color"])}


def test_outside_anchors_keep_ink_on_the_same_fill():
    fig, ax = pp.subplots()
    pp.violinplot(data=_df(), x="cat", y="v", hue="cat", fill=True, alpha=1.0,
                  palette={"A": "#0000ff", "B": "#0000ff", "C": "#0000ff"},
                  inner=None, legend=False, ax=ax)
    texts = pp.annotate(ax, kind="box_stats", anchor="right")
    assert {to_rgba(t.get_color()) for t in texts} == {to_rgba(plt.rcParams["text.color"])}


def test_boxplot_accepts_inside():
    fig, ax = pp.subplots()
    pp.boxplot(data=_df(), x="cat", y="v", legend=False, ax=ax,
               annotate={"anchor": "inside"})
    for t, box in zip(ax.texts, ax._publiplots_box_meta.boxes):
        assert t.get_position()[0] == pytest.approx(box.center_pos)


def test_violin_custom_inside_sits_on_the_median_and_contrasts():
    fig, ax = pp.subplots()
    pp.violinplot(data=_df(), x="cat", y="v", hue="cat", fill=True, alpha=1.0,
                  palette={"A": "#0000ff", "B": "#ffff00", "C": "#0000ff"},
                  inner=None, legend=False, ax=ax)
    texts = pp.annotate(ax, kind="violin_custom", anchor="inside",
                        labels=lambda rec: str(rec.category))
    for t, box in zip(texts, ax._publiplots_box_meta.boxes):
        assert t.get_position() == pytest.approx((box.center_pos, box.stats["median"]))
    white, ink = to_rgba("#ffffff"), to_rgba(plt.rcParams["text.color"])
    assert [to_rgba(t.get_color()) for t in texts] == [white, ink, white]


def test_inside_on_an_unfilled_box_contrasts_with_the_empty_interior():
    # Nothing is drawn inside an unfilled box, so even a dark outline leaves
    # the label sitting on the axes background.
    fig, ax = pp.subplots(axes_size=(60, 40))
    pp.boxplot(data=_df(), x="cat", y="v", hue="h", fill=False,
               palette={"x": "#000080", "y": "#000080"}, legend=False, ax=ax)
    texts = pp.annotate(ax, kind="box_stats", anchor="inside")
    assert len(texts) == 6
    assert {to_rgba(t.get_color()) for t in texts} == {to_rgba(plt.rcParams["text.color"])}
