"""Tests for pp.boxplot with ``fill=False``.

Unfilled, seaborn draws each box outline as a closed ``Line2D`` instead of a
``PathPatch``, so the recolouring pass must take box colours from the lines.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgba

import publiplots as pp
from publiplots.utils.legend_entries import get_entries


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


PALETTE = {"A": "#d62728", "B": "#1f77b4"}


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    values = rng.normal(size=40)
    values[0] = 6.0   # guarantee an outlier in A
    values[20] = -6.0  # and in B
    return pd.DataFrame({"c": np.repeat(list("AB"), 20), "v": values})


def _structural_lines(ax):
    return [l for l in ax.lines
            if not (l.get_marker() and l.get_marker() != "None")]


def _flier_lines(ax):
    return [l for l in ax.lines
            if l.get_marker() and l.get_marker() != "None" and len(l.get_xdata())]


def _center(line, axis):
    data = np.asarray(line.get_xdata() if axis == "x" else line.get_ydata())
    return (data.min() + data.max()) / 2


def test_unfilled_does_not_raise_without_hue(df):
    _, ax = pp.subplots(axes_size=(40, 40))
    pp.boxplot(data=df, x="c", y="v", fill=False, ax=ax)
    assert not ax.patches
    assert _structural_lines(ax)


@pytest.mark.parametrize("orient", ["v", "h"])
def test_unfilled_keeps_palette_colour_per_box(df, orient):
    _, ax = pp.subplots(axes_size=(40, 40))
    if orient == "v":
        pp.boxplot(data=df, x="c", y="v", hue="c", palette=PALETTE,
                   fill=False, ax=ax)
        cat_axis = "x"
    else:
        pp.boxplot(data=df, x="v", y="c", hue="c", palette=PALETTE,
                   fill=False, ax=ax)
        cat_axis = "y"

    expected = {0: to_rgba(PALETTE["A"]), 1: to_rgba(PALETTE["B"])}
    lines = _structural_lines(ax)
    # Each box: outline + 2 whiskers + median (caps hidden by default).
    assert len(lines) >= 8
    for line in lines:
        pos = round(_center(line, cat_axis))
        assert to_rgba(line.get_color()) == expected[pos]
        assert line.get_linewidth() == pytest.approx(pp.rcParams["edgewidth"])

    fliers = _flier_lines(ax)
    assert len(fliers) == 2
    for line in fliers:
        pos = round(_center(line, cat_axis))
        assert to_rgba(line.get_markeredgecolor()) == expected[pos]


def test_unfilled_honours_explicit_edgecolor(df):
    _, ax = pp.subplots(axes_size=(40, 40))
    pp.boxplot(data=df, x="c", y="v", hue="c", palette=PALETTE,
               fill=False, edgecolor="black", ax=ax)
    for line in _structural_lines(ax):
        assert to_rgba(line.get_color()) == to_rgba("black")


def test_unfilled_overlapping_boxes_keep_their_own_colours():
    # dodge=False with hue != x puts two boxes at each position; a
    # position -> colour lookup would paint both in one colour.
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({
        "c": np.repeat(list("AB"), 40),
        "g": np.tile(list("xy"), 40),
        "v": rng.normal(size=80),
    })
    frame.loc[frame["g"] == "y", "v"] += 5
    palette = {"x": "#d62728", "y": "#1f77b4"}
    _, ax = pp.subplots(axes_size=(40, 40))
    pp.boxplot(data=frame, x="c", y="v", hue="g", palette=palette,
               dodge=False, fill=False, ax=ax)
    colours = {to_rgba(l.get_color()) for l in _structural_lines(ax)}
    assert colours == {to_rgba(c) for c in palette.values()}


def test_unfilled_legend_swatches_are_outlines_only(df):
    # Unfilled boxes have no face, so the swatch's fill layer must be
    # invisible; the edge layer stays opaque in the palette colour.
    _, ax = pp.subplots(axes_size=(40, 40))
    pp.boxplot(data=df, x="c", y="v", hue="c", palette=PALETTE,
               fill=False, ax=ax)
    (entry,) = get_entries(ax)
    assert list(entry.labels) == list(PALETTE)
    for handle, colour in zip(entry.handles, PALETTE.values()):
        assert handle.get_alpha() == 0
        assert to_rgba(handle.get_edgecolor())[:3] == to_rgba(colour)[:3]


def test_filled_legend_swatches_keep_translucent_fill(df):
    _, ax = pp.subplots(axes_size=(40, 40))
    pp.boxplot(data=df, x="c", y="v", hue="c", palette=PALETTE, ax=ax)
    (entry,) = get_entries(ax)
    for handle in entry.handles:
        assert handle.get_alpha() == pytest.approx(pp.rcParams["alpha"])
