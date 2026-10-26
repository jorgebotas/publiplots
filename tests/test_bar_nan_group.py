"""An all-NaN group must not shift the colours of later bars (issue #291).

seaborn drops NaN-valued rows before drawing, so a (category, hue) group
whose values are all NaN gets no patch. ``_paint_bars`` paired patches with
``BarSplitSpec.iter_draw_order`` triples by position while still counting
that group, so every later bar was painted with the previous level's colour.

Each bar's value is unique, so the test recovers a patch's level from its
height and checks its face against the palette.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_hex

import publiplots as pp


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


PALETTE = {"x": "#ff0000", "y": "#0000ff", "z": "#00aa00"}


def _frame(nan_cells):
    df = pd.DataFrame({"cat": ["a"] * 3 + ["b"] * 3,
                       "arm": ["x", "y", "z"] * 2,
                       "v": [0.5, 0.3, 0.7, 0.4, 0.2, 0.6]})
    df.loc[nan_cells, "v"] = np.nan
    return df


@pytest.mark.parametrize("nan_cells", [[1, 4], [1]], ids=["level", "cell"])
@pytest.mark.parametrize("horizontal", [False, True], ids=["v", "h"])
def test_bars_after_an_all_nan_group_keep_their_colour(nan_cells, horizontal):
    df = _frame(nan_cells)
    level_of = dict(zip(df["v"].dropna().round(3), df.loc[df["v"].notna(), "arm"]))
    xy = dict(x="v", y="cat") if horizontal else dict(x="cat", y="v")

    fig, ax = pp.subplots()
    pp.barplot(data=df, **xy, hue="arm", order=["a", "b"],
               hue_order=["x", "y", "z"], palette=PALETTE,
               errorbar=None, alpha=1, ax=ax)

    drawn = [p for p in ax.patches
             if (p.get_width() if horizontal else p.get_height()) > 0]
    assert len(drawn) == len(level_of)
    for p in drawn:
        value = round(p.get_width() if horizontal else p.get_height(), 3)
        assert to_hex(p.get_facecolor()) == PALETTE[level_of[value]], value
