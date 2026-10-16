"""BoxStatsRecord carries its drawn face, its artist and its seam side;
BoxStatsMeta knows whether the plot dodged."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_rgba
from matplotlib.patches import PathPatch

import publiplots as pp


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _df(empty_combo: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame([{"cat": c, "h": h, "v": float(v)}
                       for c in "AB" for h in ("x", "y")
                       for v in rng.normal(0, 1, 30)])
    if empty_combo:
        df = df[~((df.cat == "A") & (df.h == "x"))]
    return df


def test_violin_records_carry_their_body_and_drawn_face():
    fig, ax = pp.subplots()
    pp.violinplot(data=_df(), x="cat", y="v", hue="h", inner=None,
                  legend=False, ax=ax)
    for rec in ax._publiplots_box_meta.boxes:
        assert isinstance(rec.artist, PolyCollection)
        assert rec.artist in ax.collections
        assert rec.face_color == pytest.approx(tuple(rec.artist.get_facecolor()[0]))
        assert rec.face_color[3] == pytest.approx(pp.rcParams["alpha"])


def test_boxplot_records_carry_their_patch():
    fig, ax = pp.subplots()
    pp.boxplot(data=_df(), x="cat", y="v", hue="h", legend=False, ax=ax)
    for rec in ax._publiplots_box_meta.boxes:
        assert isinstance(rec.artist, PathPatch)
        assert rec.face_color == pytest.approx(to_rgba(rec.artist.get_facecolor()))
        assert rec.half is None


@pytest.mark.parametrize("empty_combo", [False, True])
def test_split_halves_follow_the_hue_level_not_draw_order(empty_combo):
    # hue_order pins x as the even level: without it, dropping (A, x) makes
    # y the first-seen level, and seaborn then draws y on the low side.
    fig, ax = pp.subplots()
    pp.violinplot(data=_df(empty_combo), x="cat", y="v", hue="h", split=True,
                  hue_order=["x", "y"], inner=None, legend=False, ax=ax)
    boxes = ax._publiplots_box_meta.boxes
    got = {(r.category, r.hue_value): r.half for r in boxes}
    want = {("A", "x"): "low", ("A", "y"): "high",
            ("B", "x"): "low", ("B", "y"): "high"}
    if empty_combo:
        del want[("A", "x")]
    assert got == want
    # And each half is where seaborn actually drew it: on the side of its
    # category's seam that the label names.
    seams = {"A": 0.0, "B": 1.0}
    for r in boxes:
        xs = r.artist.get_paths()[0].vertices[:, 0]
        seam = seams[r.category]
        if r.half == "low":
            assert xs.max() == pytest.approx(seam)
        else:
            assert xs.min() == pytest.approx(seam)


def test_split_on_a_hue_that_repeats_the_category_halves_by_category():
    """hue == x records no hue value, but seaborn still halves each body,
    alternating by category level with the seam on the slot's edge."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame([{"cat": c, "v": float(v)}
                       for c in "ABC" for v in rng.normal(0, 1, 30)])
    fig, ax = pp.subplots()
    pp.violinplot(data=df, x="cat", y="v", hue="cat", split=True,
                  inner=None, legend=False, ax=ax)
    boxes = ax._publiplots_box_meta.boxes
    assert {r.category: r.half for r in boxes} == {"A": "low", "B": "high",
                                                   "C": "low"}
    for r in boxes:
        xs = r.artist.get_paths()[0].vertices[:, 0]
        # The seam is a straight edge: many vertices share its coordinate.
        seam = xs.max() if r.half == "low" else xs.min()
        assert np.isclose(xs, seam).sum() > len(xs) // 3


@pytest.mark.parametrize("orient", ["v", "h"])
def test_split_without_hue_draws_low_halves(orient):
    """With no hue, seaborn treats every body as hue level 0: a low half
    whose straight seam is the slot's high edge."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame([{"cat": c, "v": float(v)}
                       for c in "AB" for v in rng.normal(0, 1, 30)])
    kw = dict(x="cat", y="v") if orient == "v" else dict(x="v", y="cat")
    fig, ax = pp.subplots()
    pp.violinplot(data=df, split=True, inner=None, legend=False, ax=ax, **kw)
    boxes = ax._publiplots_box_meta.boxes
    assert [r.half for r in boxes] == ["low", "low"]
    col = 0 if orient == "v" else 1
    slot_half_width = 0.45  # violinplot's default width=0.9, halved
    for pos, r in enumerate(boxes):
        cs = r.artist.get_paths()[0].vertices[:, col]
        seam = r.center_pos + r.cat_half_width
        assert seam == pytest.approx(pos + slot_half_width)
        assert cs.max() == pytest.approx(seam)
        assert np.isclose(cs, seam).sum() > len(cs) // 3


@pytest.mark.parametrize("side, half", [("left", "low"), ("right", "high"), ("both", None)])
def test_side_sets_the_half(side, half):
    fig, ax = pp.subplots()
    pp.violinplot(data=_df(), x="cat", y="v", side=side, inner=None,
                  legend=False, ax=ax)
    assert {r.half for r in ax._publiplots_box_meta.boxes} == {half}


@pytest.mark.parametrize("kwargs, dodged", [
    (dict(hue="h"), True),
    (dict(hue="cat"), False),
    (dict(), False),
    (dict(hue="h", split=True), False),
])
def test_meta_knows_when_bodies_were_dodged(kwargs, dodged):
    fig, ax = pp.subplots()
    pp.violinplot(data=_df(), x="cat", y="v", inner=None, legend=False,
                  ax=ax, **kwargs)
    assert ax._publiplots_box_meta.dodged is dodged


def test_custom_center_label_contrasts_with_the_drawn_face():
    """The palette colour is pure blue, which takes white ink when opaque —
    but the body is drawn at alpha 0.1, so the label must be ink.

    The hue is a column of its own: a hue that repeats the categorical axis
    records no hue level, so its hue_color is already the drawn face."""
    fig, ax = pp.subplots()
    pp.violinplot(data=_df(), x="cat", y="v", hue="h",
                  palette={"x": "#0000ff", "y": "#0000ff"},
                  inner=None, legend=False, ax=ax)
    texts = pp.annotate(ax, kind="violin_custom", labels=lambda r: "n",
                        anchor="center")
    ink = to_rgba(plt.rcParams["text.color"])
    assert [to_rgba(t.get_color()) for t in texts] == [ink] * 4
