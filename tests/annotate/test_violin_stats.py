"""kind="violin_stats": box_stats' statistics, labelled from inside the body."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import publiplots as pp
from publiplots.annotate._dispatcher import _DEFAULT_FMT, _STRATEGIES


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _df() -> pd.DataFrame:
    rng = np.random.default_rng(2)
    return pd.DataFrame([{"cat": c, "h": h, "v": float(v)}
                         for c in "AB" for h in ("x", "y")
                         for v in rng.normal(0, 1, 40)])


def _violins(**kwargs):
    fig, ax = pp.subplots()
    kwargs.setdefault("x", "cat")
    kwargs.setdefault("y", "v")
    pp.violinplot(data=_df(), inner=None, legend=False, ax=ax, **kwargs)
    return ax


def test_violin_stats_is_registered_with_a_two_decimal_default():
    assert "violin_stats" in _STRATEGIES
    assert _DEFAULT_FMT["violin_stats"] == ".2f"


def test_violin_stats_defaults_to_inside():
    ax = _violins()
    texts = pp.annotate(ax, kind="violin_stats")
    for t, box in zip(texts, ax._publiplots_box_meta.boxes):
        assert t.get_position() == pytest.approx((box.center_pos, box.stats["median"]))


def test_violinplot_annotate_true_labels_from_inside():
    fig, ax = pp.subplots()
    pp.violinplot(data=_df(), x="cat", y="v", inner=None, legend=False, ax=ax,
                  annotate=True)
    for t, box in zip(ax.texts, ax._publiplots_box_meta.boxes):
        assert t.get_position()[0] == pytest.approx(box.center_pos)


def test_box_stats_still_defaults_to_right():
    ax = _violins()
    texts = pp.annotate(ax, kind="box_stats")
    for t, box in zip(texts, ax._publiplots_box_meta.boxes):
        assert t.get_position()[0] == pytest.approx(box.center_pos + box.cat_half_width)


@pytest.mark.parametrize("kwargs, rotation", [
    (dict(hue="h"), 90.0),                          # dodged vertical
    (dict(y="cat", x="v", hue="h"), 0.0),           # dodged horizontal
    (dict(hue="h", split=True), 0.0),               # halves
    (dict(), 0.0),                                  # one body per category
])
def test_only_dodged_vertical_labels_rotate(kwargs, rotation):
    ax = _violins(**kwargs)
    texts = pp.annotate(ax, kind="violin_stats")
    assert {t.get_rotation() for t in texts} == {rotation}


def test_rotated_labels_rise_from_the_median():
    ax = _violins(hue="h")
    texts = pp.annotate(ax, kind="violin_stats")
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    for t, box in zip(texts, ax._publiplots_box_meta.boxes):
        assert (t.get_ha(), t.get_va()) == ("center", "bottom")
        median_px = ax.transData.transform((0, box.stats["median"]))[1]
        assert t.get_window_extent(renderer).y0 > median_px


def test_explicit_rotation_wins_on_dodged_vertical():
    ax = _violins(hue="h")
    texts = pp.annotate(ax, kind="violin_stats", rotation=0)
    assert {t.get_rotation() for t in texts} == {0.0}


def test_other_kinds_still_default_to_no_rotation():
    ax = _violins(hue="h")
    texts = pp.annotate(ax, kind="box_stats", anchor="inside")
    assert {t.get_rotation() for t in texts} == {0.0}


def test_violin_custom_defaults_to_inside():
    ax = _violins(y="cat", x="v")
    texts = pp.annotate(ax, kind="violin_custom", labels=lambda r: "n")
    for t, box in zip(texts, ax._publiplots_box_meta.boxes):
        assert t.get_position() == pytest.approx((box.stats["median"], box.center_pos))
        assert t.get_ha() == "left"
