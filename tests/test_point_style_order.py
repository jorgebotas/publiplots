"""pointplot's per-level ``linestyles`` / ``markers`` follow seaborn's level
order, not data order (issue #292).

seaborn applies a ``linestyles`` / ``markers`` list in ``hue_order`` (or its
own default level order). publiplots built those lists from
``data[hue].unique()``, so whenever the levels first appeared in the data in
another order every series was drawn with another level's dash and marker.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.colors import to_hex

import publiplots as pp


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _styles_by_colour(ax):
    """{colour: (linestyle, marker)} for each drawn series."""
    return {
        to_hex(ln.get_color()): (ln.get_linestyle(), ln.get_marker())
        for ln in ax.lines
        if ln.get_marker() not in ("None", None, "") and ln.get_linestyle() != "None"
    }


def _frame(levels):
    # Reversed, so the levels first appear in the opposite order to `levels`.
    return pd.DataFrame({
        "x": list("abc") * 2, "y": [1, 2, 3, 2, 3, 4],
        "series": [levels[0]] * 3 + [levels[1]] * 3,
    }).iloc[::-1]


@pytest.mark.parametrize("styles", [
    dict(linestyles=["-", "--"], markers=["o", "s"]),
    dict(linestyles={"first": "-", "second": "--"},
         markers={"first": "o", "second": "s"}),
], ids=["list", "dict"])
def test_styles_follow_hue_order(styles):
    df = _frame(["first", "second"])
    fig, ax = pp.subplots()
    pp.pointplot(data=df, x="x", y="y", hue="series",
                 hue_order=["first", "second"],
                 palette={"first": "#ff0000", "second": "#0000ff"},
                 errorbar=None, ax=ax, **styles)
    assert _styles_by_colour(ax) == {"#ff0000": ("-", "o"),
                                     "#0000ff": ("--", "s")}


def test_numeric_level_dicts_without_hue_order():
    # seaborn sorts numeric hue levels; data order here is 2 then 1. (A list
    # is applied in seaborn's order either way, so only a dict exposes it.)
    df = _frame([1, 2])
    fig, ax = pp.subplots()
    pp.pointplot(data=df, x="x", y="y", hue="series",
                 palette={1: "#ff0000", 2: "#0000ff"},
                 linestyles={1: "-", 2: "--"}, markers={1: "o", 2: "s"},
                 errorbar=None, ax=ax)
    assert _styles_by_colour(ax) == {"#ff0000": ("-", "o"),
                                     "#0000ff": ("--", "s")}


def test_numeric_level_lists_without_hue_order_follow_sorted_order():
    # A list maps onto seaborn's sorted levels, as `sns.pointplot` does.
    df = _frame([1, 2])
    fig, ax = pp.subplots()
    pp.pointplot(data=df, x="x", y="y", hue="series",
                 palette={1: "#ff0000", 2: "#0000ff"},
                 linestyles=["-", "--"], markers=["o", "s"],
                 errorbar=None, ax=ax)
    assert _styles_by_colour(ax) == {"#ff0000": ("-", "o"),
                                     "#0000ff": ("--", "s")}
