"""pp.violinplot's inner="median" default and width=0.9."""
import inspect

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import seaborn as sns
from matplotlib.colors import to_rgba

import publiplots as pp


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


@pytest.fixture
def df():
    rng = np.random.default_rng(4)
    return pd.DataFrame({"cat": np.repeat(list("ABC"), 40),
                         "g": np.tile(["x", "y"], 60),
                         "v": rng.normal(size=120)})


def test_defaults():
    params = inspect.signature(pp.violinplot).parameters
    assert params["inner"].default == "median"
    assert params["width"].default == 0.9


def test_default_draws_one_median_line_per_body_and_nothing_else(df):
    ax = pp.violinplot(data=df, x="cat", y="v")
    meta = ax._publiplots_box_meta
    assert len(ax.lines) == len(meta.boxes) == 3
    for line, rec in zip(ax.lines, meta.boxes):
        assert line.get_ydata()[0] == pytest.approx(rec.stats["median"])


def test_width_default_is_0_9(df):
    ax = pp.violinplot(data=df, x="cat", y="v", density_norm="width")
    assert {round(b.cat_half_width, 4) for b in ax._publiplots_box_meta.boxes} == {0.45}


@pytest.mark.parametrize("inner", ["box", "quart", "stick", "point", None])
def test_seaborn_inner_styles_draw_what_seaborn_draws(df, inner):
    ax = pp.violinplot(data=df, x="cat", y="v", inner=inner)
    fig, ref = plt.subplots()  # raw seaborn reference axes
    sns.violinplot(data=df, x="cat", y="v", inner=inner, width=0.9, ax=ref)
    assert len(ax.lines) == len(ref.lines)
    assert len(ax.collections) == len(ref.collections)


def test_unknown_inner_still_raises(df):
    with pytest.raises(ValueError, match="inner"):
        pp.violinplot(data=df, x="cat", y="v", inner="mean")


def test_median_follows_a_per_call_linewidth_and_edgecolor(df):
    ax = pp.violinplot(data=df, x="cat", y="v", linewidth=1.6, edgecolor="red")
    assert {l.get_linewidth() for l in ax.lines} == {1.6}
    assert {to_rgba(l.get_color()) for l in ax.lines} == {to_rgba("red")}


def test_median_follows_rcparams_edgecolor(df):
    saved = pp.rcParams["edgecolor"]
    pp.rcParams["edgecolor"] = "#123456"
    try:
        ax = pp.violinplot(data=df, x="cat", y="v")
    finally:
        pp.rcParams["edgecolor"] = saved
    assert {to_rgba(l.get_color()) for l in ax.lines} == {to_rgba("#123456")}


def test_inner_kws_reach_the_median(df):
    ax = pp.violinplot(data=df, x="cat", y="v",
                       inner_kws={"inset": 0, "color": "k", "linewidth": 2})
    assert {l.get_solid_capstyle() for l in ax.lines} == {"butt"}
    assert {l.get_linewidth() for l in ax.lines} == {2}


def test_a_nan_keeps_its_bodys_median_line_and_label(df):
    df.loc[0, "v"] = np.nan   # group A
    ax = pp.violinplot(data=df, x="cat", y="v", annotate={"fmt": ".3f"})
    meta = ax._publiplots_box_meta
    assert len(ax.lines) == len(meta.boxes) == 3
    clean = df.dropna().groupby("cat")["v"].median()
    for line, rec in zip(ax.lines, meta.boxes):
        assert line.get_ydata()[0] == pytest.approx(clean[rec.category])
    assert [t.get_text() for t in ax.texts] == [f"{clean[c]:.3f}" for c in "ABC"]


@pytest.mark.parametrize("key", ["box_width", "whis_width", "marker"])
def test_box_only_inner_kws_point_to_inner_box(df, key):
    with pytest.raises(ValueError, match=rf"{key}.*inner=.box."):
        pp.violinplot(data=df, x="cat", y="v", inner_kws={key: 4})


def test_box_only_inner_kws_still_style_inner_box(df):
    ax = pp.violinplot(data=df, x="cat", y="v", inner="box",
                       inner_kws={"box_width": 4, "whis_width": 2})
    assert {4, 2} <= {l.get_linewidth() for l in ax.lines}


def test_median_line_and_label_agree(df):
    ax = pp.violinplot(data=df, x="cat", y="v", annotate={"fmt": ".3f"})
    assert [t.get_text() for t in ax.texts] == [
        f"{l.get_ydata()[0]:.3f}" for l in ax.lines]


def test_raincloud_keeps_its_width(df, monkeypatch):
    import publiplots.plot.raincloud as rc
    seen = {}
    real = rc.violinplot

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(rc, "violinplot", spy)
    pp.raincloudplot(data=df, x="cat", y="v")
    assert seen["width"] == 0.8
    assert seen["inner"] is None
