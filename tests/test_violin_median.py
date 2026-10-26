"""The median-line inner mark (publiplots/plot/_violin_median.py)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgba

import publiplots as pp
from publiplots.plot._violin_median import draw_median_inner, local_span


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


CATS = ["bimodal", "sym"]


def _df() -> pd.DataFrame:
    # The bimodal body's median falls in the trough between its modes, so
    # its line is far narrower than the body: measured 0.27 of its full
    # width at 60x40 mm. (A skewed beta does not work — KDE smoothing lifts
    # the median's width to 0.87-0.91 of the peak.)
    rng = np.random.default_rng(3)
    bimodal = np.r_[rng.normal(0.3, 0.07, 150), rng.normal(0.7, 0.07, 150)]
    symmetric = rng.normal(0.5, 0.1, 300)
    return pd.DataFrame({"cat": ["bimodal"] * 300 + ["sym"] * 300,
                         "h": ["x", "y"] * 300,
                         "v": np.r_[bimodal, symmetric]})


def _plot(orient="v", **kwargs):
    xy = dict(x="cat", y="v") if orient == "v" else dict(y="cat", x="v")
    fig, ax = pp.subplots(axes_size=(60, 40))
    pp.violinplot(data=_df(), inner=None, legend=False, ax=ax, **xy, **kwargs)
    return ax, ax._publiplots_box_meta


def _segment(line, orient="v"):
    """(value, lo, hi) of a median line on the value / position axes."""
    xs, ys = line.get_xdata(), line.get_ydata()
    val, pos = (ys, xs) if orient == "v" else (xs, ys)
    return float(val[0]), float(min(pos)), float(max(pos))


@pytest.mark.parametrize("orient", ["v", "h"])
def test_one_line_per_body_at_the_group_median(orient):
    ax, meta = _plot(orient)
    lines = draw_median_inner(ax, meta, {}, linewidth=0.75)
    assert len(lines) == len(meta.boxes) == 2
    for line, rec in zip(lines, meta.boxes):
        assert _segment(line, orient)[0] == pytest.approx(rec.stats["median"])


@pytest.mark.parametrize("orient", ["v", "h"])
def test_line_stays_inside_the_local_span(orient):
    ax, meta = _plot(orient)
    val_axis = 1 if orient == "v" else 0
    for line, rec in zip(draw_median_inner(ax, meta, {}, linewidth=0.75), meta.boxes):
        val, lo, hi = _segment(line, orient)
        span_lo, span_hi = local_span(rec.artist, val, val_axis)
        assert span_lo <= lo < hi <= span_hi


def test_line_spans_the_local_not_the_global_thickness():
    ax, meta = _plot()
    _, lo, hi = _segment(draw_median_inner(ax, meta, {"inset": 0}, linewidth=0.75)[0])
    assert hi - lo < 0.5 * (2 * meta.boxes[0].cat_half_width)


@pytest.mark.parametrize("inset, fraction, cap", [
    (0, 1.0, "butt"), (0.12, 0.88, "round"), (0.99, 0.2, "round"),
])
def test_inset_shortens_the_line(inset, fraction, cap):
    ax, meta = _plot()
    for line, rec in zip(draw_median_inner(ax, meta, {"inset": inset}, linewidth=0.75),
                         meta.boxes):
        val, lo, hi = _segment(line)
        span_lo, span_hi = local_span(rec.artist, val, 1)
        assert hi - lo == pytest.approx(fraction * (span_hi - span_lo))
        assert line.get_solid_capstyle() == cap


def test_auto_inset_is_half_the_stroke():
    ax, meta = _plot()
    line = draw_median_inner(ax, meta, {}, linewidth=2.0)[1]   # the wide body
    val, lo, _ = _segment(line)
    span_lo, _ = local_span(meta.boxes[1].artist, val, 1)
    pad_px = ax.transData.transform((lo, 0))[0] - ax.transData.transform((span_lo, 0))[0]
    assert pad_px == pytest.approx(1.0 / 72 * ax.figure.dpi, rel=1e-6)
    assert line.get_solid_capstyle() == "round"


def test_numpy_scalar_inset_is_accepted():
    ax, meta = _plot()
    numpy_lines = draw_median_inner(ax, meta, {"inset": np.float32(0.1)}, linewidth=0.75)
    float_lines = draw_median_inner(ax, meta, {"inset": 0.1}, linewidth=0.75)
    for a, b in zip(numpy_lines, float_lines):
        assert _segment(a) == pytest.approx(_segment(b), rel=1e-6)


@pytest.mark.parametrize("bad", [-0.1, 1.0, "wide", True, np.bool_(True)])
def test_invalid_inset_raises(bad):
    ax, meta = _plot()
    with pytest.raises(ValueError, match="inset"):
        draw_median_inner(ax, meta, {"inset": bad}, linewidth=0.75)


@pytest.mark.parametrize("side", ["left", "right"])
def test_half_violin_line_spans_only_its_half(side):
    ax, meta = _plot(side=side)
    for line, rec in zip(draw_median_inner(ax, meta, {"inset": 0}, linewidth=0.75),
                         meta.boxes):
        _, lo, hi = _segment(line)
        # side= clips each body at the mean of its vertices, which lands
        # ~2e-5 data units off the category position; the line must meet
        # the seam the body was actually clipped to.
        pos = rec.artist.get_paths()[0].vertices[:, 0]
        seam = pos.min() if side == "right" else pos.max()
        assert seam == pytest.approx(CATS.index(rec.category), abs=1e-4)
        if side == "right":
            assert lo == pytest.approx(seam, abs=1e-9) and hi > seam
        else:
            assert hi == pytest.approx(seam, abs=1e-9) and lo < seam


def test_split_lines_meet_the_seam_from_their_own_side():
    ax, meta = _plot(hue="h", split=True)
    for line, rec in zip(draw_median_inner(ax, meta, {"inset": 0}, linewidth=0.75),
                         meta.boxes):
        _, lo, hi = _segment(line)
        seam = CATS.index(rec.category)
        if rec.half == "low":
            assert hi == pytest.approx(seam, abs=1e-6) and lo < seam
        else:
            assert lo == pytest.approx(seam, abs=1e-6) and hi > seam


def test_colour_follows_the_body_edge():
    # Both reach 4.2:1 against their alpha=0.1 tint, above the 3:1 floor.
    ax, meta = _plot(hue="cat", palette={"bimodal": "#1f77b4", "sym": "#d62728"})
    for line, rec in zip(draw_median_inner(ax, meta, {}, linewidth=0.75), meta.boxes):
        assert to_rgba(line.get_color()) == pytest.approx(
            tuple(rec.artist.get_edgecolor()[0]))


def test_explicit_colour_wins():
    ax, meta = _plot(hue="cat")
    lines = draw_median_inner(ax, meta, {"color": "red"}, linewidth=0.75)
    assert {to_rgba(l.get_color()) for l in lines} == {to_rgba("red")}


def test_dark_opaque_fill_turns_the_line_white():
    ax, meta = _plot(hue="cat", fill=True, alpha=1.0, edgecolor="black",
                     palette={"bimodal": "#0000ff", "sym": "#ffff00"})
    dark, light = draw_median_inner(ax, meta, {}, linewidth=0.75)
    assert to_rgba(dark.get_color()) == to_rgba("white")
    assert to_rgba(light.get_color()) == to_rgba("black")


def test_linewidth_and_zorder():
    ax, meta = _plot()
    assert {l.get_linewidth() for l in draw_median_inner(ax, meta, {}, linewidth=1.3)} == {1.3}
    lines = draw_median_inner(ax, meta, {"linewidth": 2.0}, linewidth=1.3)
    assert {l.get_linewidth() for l in lines} == {2.0}
    for line, rec in zip(lines, meta.boxes):
        assert line.get_zorder() == rec.artist.get_zorder() + 1


def test_drawing_does_not_rescale_the_axes():
    ax, meta = _plot()
    before = (ax.get_xlim(), ax.get_ylim())
    draw_median_inner(ax, meta, {"inset": 0}, linewidth=0.75)
    assert (ax.get_xlim(), ax.get_ylim()) == before


def test_aliases_and_zorder_pass_through():
    ax, meta = _plot()
    lines = draw_median_inner(ax, meta, {"zorder": 10, "lw": 2, "c": "red"},
                              linewidth=0.75)
    assert len(lines) == 2
    for line in lines:
        assert line.get_zorder() == 10
        assert line.get_linewidth() == 2
        assert to_rgba(line.get_color()) == to_rgba("red")


def test_scalex_cannot_rescale_the_axes():
    ax, meta = _plot()
    before = (ax.get_xlim(), ax.get_ylim())
    draw_median_inner(ax, meta, {"inset": 0, "scalex": True, "scaley": True},
                      linewidth=0.75)
    assert (ax.get_xlim(), ax.get_ylim()) == before
