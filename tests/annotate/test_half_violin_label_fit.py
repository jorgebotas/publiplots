"""violin_stats on half-bodies: labels fit beside their seam without
colliding. A vertical body narrower than an undivided slot turns its label
90 deg; a half-body's label clears its seam by a fixed gap."""
import itertools

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import publiplots as pp
from publiplots.annotate.box_stats import _SEAM_GAP_MM


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _df(n_cat: int = 5) -> pd.DataFrame:
    # Every group from one distribution: the medians, and so the labels,
    # line up on the value axis, the worst case for a collision.
    rng = np.random.default_rng(0)
    return pd.DataFrame([{"cat": f"c{c}", "h": h, "v": float(v)}
                         for c in range(n_cat) for h in ("x", "y")
                         for v in rng.normal(0.6, 0.15, 60)])


def _violins(orient, size=(42, 32), **kwargs):
    xy = dict(x="cat", y="v") if orient == "v" else dict(y="cat", x="v")
    fig, ax = pp.subplots(axes_size=size)
    pp.violinplot(data=_df(), legend=False, ax=ax, annotate=True,
                  **xy, **kwargs)
    fig.canvas.draw()
    return ax


def _seam(box) -> float:
    return box.center_pos + (box.cat_half_width if box.half == "low"
                             else -box.cat_half_width)


def _cat_span(ax, orient, bbox):
    return (bbox.x0, bbox.x1) if orient == "v" else (bbox.y0, bbox.y1)


def _to_px(ax, orient, pos) -> float:
    return ax.transData.transform((pos, pos))[0 if orient == "v" else 1]


_HALVES = [
    pytest.param(dict(hue="h", side="right"), id="side-right-hue"),
    pytest.param(dict(hue="h", side="left"), id="side-left-hue"),
    pytest.param(dict(hue="h", split=True), id="split-hue"),
    pytest.param(dict(side="right"), id="side-right"),
    pytest.param(dict(split=True), id="split"),
]


@pytest.mark.parametrize("orient", ["v", "h"])
@pytest.mark.parametrize("kwargs", _HALVES)
def test_half_body_labels_do_not_overlap(orient, kwargs):
    ax = _violins(orient, **kwargs)
    renderer = ax.figure.canvas.get_renderer()
    bboxes = [t.get_window_extent(renderer) for t in ax.texts]
    assert len(bboxes) == len(ax._publiplots_box_meta.boxes)
    clashes = [(i, j) for (i, a), (j, b)
               in itertools.combinations(enumerate(bboxes), 2) if a.overlaps(b)]
    assert clashes == []


@pytest.mark.parametrize("orient", ["v", "h"])
@pytest.mark.parametrize("kwargs", _HALVES)
def test_half_body_labels_stay_on_their_side_of_the_seam(orient, kwargs):
    ax = _violins(orient, **kwargs)
    renderer = ax.figure.canvas.get_renderer()
    gap_px = _SEAM_GAP_MM / 25.4 * ax.figure.dpi
    for t, box in zip(ax.texts, ax._publiplots_box_meta.boxes):
        seam_px = _to_px(ax, orient, _seam(box))
        lo, hi = _cat_span(ax, orient, t.get_window_extent(renderer))
        if _to_px(ax, orient, box.center_pos) > seam_px:
            assert lo == pytest.approx(seam_px + gap_px, abs=0.5)
        else:
            assert hi == pytest.approx(seam_px - gap_px, abs=0.5)


@pytest.mark.parametrize("kwargs, rotation", [
    (dict(), 0.0),                                  # undivided slot
    (dict(hue="h"), 90.0),                          # dodged
    (dict(hue="h", split=True), 90.0),              # split by hue
    (dict(hue="h", side="right"), 90.0),            # dodged halves
    (dict(side="left"), 90.0),                      # halved slot
    (dict(split=True), 0.0),                        # split, no hue: full slot
    (dict(hue="cat", split=True), 0.0),             # hue repeats the axis
])
def test_vertical_labels_rotate_when_the_body_is_narrower_than_a_slot(
        kwargs, rotation):
    ax = _violins("v", **kwargs)
    assert {t.get_rotation() for t in ax.texts} == {rotation}


@pytest.mark.parametrize("kwargs", _HALVES)
def test_horizontal_half_labels_never_rotate(kwargs):
    ax = _violins("h", **kwargs)
    assert {t.get_rotation() for t in ax.texts} == {0.0}


@pytest.mark.parametrize("kwargs", _HALVES[:3])
def test_rotated_half_labels_rise_from_the_median(kwargs):
    ax = _violins("v", **kwargs)
    renderer = ax.figure.canvas.get_renderer()
    for t, box in zip(ax.texts, ax._publiplots_box_meta.boxes):
        assert t.get_va() == "bottom"
        median_px = ax.transData.transform((0, box.stats["median"]))[1]
        assert t.get_window_extent(renderer).y0 > median_px


@pytest.mark.parametrize("orient", ["v", "h"])
def test_seam_gap_does_not_follow_offset(orient):
    """`offset` is the gap from the median line; the seam keeps its own."""
    xy = dict(x="cat", y="v") if orient == "v" else dict(y="cat", x="v")
    fig, ax = pp.subplots(axes_size=(42, 32))
    pp.violinplot(data=_df(), hue="h", split=True, legend=False, ax=ax,
                  annotate={"offset": 3.0}, **xy)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    gap_px = _SEAM_GAP_MM / 25.4 * fig.dpi
    value_axis = 1 if orient == "v" else 0
    for t, box in zip(ax.texts, ax._publiplots_box_meta.boxes):
        seam_px = _to_px(ax, orient, _seam(box))
        bb = t.get_window_extent(renderer)
        lo, hi = _cat_span(ax, orient, bb)
        assert min(abs(lo - seam_px), abs(hi - seam_px)) == pytest.approx(
            gap_px, abs=0.5)
        median_px = ax.transData.transform((box.stats["median"],) * 2)[value_axis]
        near = bb.y0 if orient == "v" else bb.x0
        assert near - median_px > 2.5 / 25.4 * fig.dpi
