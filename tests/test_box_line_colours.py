"""Tests for which colour each pp.boxplot line takes from its box."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgba

import publiplots as pp


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


@pytest.mark.parametrize("dodge", [False, True])
def test_filled_overlapping_boxes_keep_their_own_colours(dodge):
    # Filled, the lines are seaborn's grey and carry no box colour, and a
    # position -> colour lookup cannot tell apart boxes that share a
    # position (dodge=False with hue != x). Each line must take the colour
    # of its own box: identified here by value, since the hue levels are
    # 5 sd apart, so every line sits nearest its own box's centre.
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({
        "c": np.repeat(list("AB"), 40),
        "g": np.tile(list("xy"), 40),
        "v": rng.normal(size=80),
    })
    frame.loc[frame["g"] == "y", "v"] += 5
    _, ax = pp.subplots(axes_size=(40, 40))
    pp.boxplot(data=frame, x="c", y="v", hue="g",
               palette={"x": "#d62728", "y": "#1f77b4"},
               dodge=dodge, ax=ax)

    boxes = []
    for patch in ax.patches:
        verts = patch.get_path().vertices
        boxes.append((verts[:, 0].mean(), verts[:, 1].mean(),
                      to_rgba(patch.get_edgecolor())[:3]))
    assert len({colour for *_, colour in boxes}) == 2

    lines = [l for l in ax.lines if len(l.get_xdata())]
    assert lines
    for line in lines:
        x, y = np.mean(line.get_xdata()), np.mean(line.get_ydata())
        *_, own = min(boxes, key=lambda b: (abs(b[0] - x), abs(b[1] - y)))
        is_flier = line.get_marker() not in (None, "", "None", " ")
        drawn = line.get_markeredgecolor() if is_flier else line.get_color()
        assert to_rgba(drawn)[:3] == own
