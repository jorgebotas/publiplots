"""Regression tests for legend marker edgecolor in lineplot/pointplot.

The legend handle factory ``create_legend_handles`` stores the user's
``edgecolor`` on ``LineMarkerPatch``. ``HandlerLineMarker.create_artists``
must render the marker *ring* with that color rather than the face color —
previously it ignored the extracted ``edgecolor`` and used the face color
for both fill and ring, so ``edgecolor='black'`` had no visible effect on
the lineplot/pointplot legend markers.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgba
from matplotlib.offsetbox import DrawingArea

import publiplots as pp


@pytest.fixture
def sample_data():
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "x": list(range(5)) * 3,
        "y": rng.normal(size=15),
        "g": ["A"] * 5 + ["B"] * 5 + ["C"] * 5,
    })


@pytest.fixture(autouse=True)
def _restore_edgecolor_rcparam():
    original = pp.rcParams["edgecolor"]
    yield
    pp.rcParams["edgecolor"] = original


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _legend_marker_edgecolors(legend) -> list[tuple]:
    """Return the ``markeredgecolor`` for each rendered marker layer.

    Each entry produces three Line2D layers (line, white background marker,
    colored marker). The third is the user-visible ring; its ``mec`` is
    what we assert against.
    """
    out: list[tuple] = []
    stack: list = [legend]
    areas: list[DrawingArea] = []
    while stack:
        obj = stack.pop()
        if isinstance(obj, DrawingArea):
            areas.append(obj)
            continue
        if hasattr(obj, "get_children"):
            stack.extend(obj.get_children())
    for da in areas:
        children = da.get_children()
        if len(children) >= 3:
            out.append(to_rgba(children[2].get_markeredgecolor()))
    return out


def test_lineplot_legend_honors_edgecolor_arg(sample_data):
    fig, ax = plt.subplots()
    pp.lineplot(
        data=sample_data, x="x", y="y", hue="g", style="g",
        markers=True, ax=ax, edgecolor="black",
    )
    fig.canvas.draw()
    mecs = _legend_marker_edgecolors(ax.get_legend())
    assert mecs, "expected at least one legend marker"
    for mec in mecs:
        assert mec[:3] == pytest.approx(to_rgba("black")[:3], abs=0.01)


def test_lineplot_legend_honors_edgecolor_rcparam(sample_data):
    pp.rcParams["edgecolor"] = "red"
    fig, ax = plt.subplots()
    pp.lineplot(
        data=sample_data, x="x", y="y", hue="g", style="g",
        markers=True, ax=ax,
    )
    fig.canvas.draw()
    mecs = _legend_marker_edgecolors(ax.get_legend())
    assert mecs
    for mec in mecs:
        assert mec[:3] == pytest.approx(to_rgba("red")[:3], abs=0.01)


def test_lineplot_legend_default_edge_matches_face(sample_data):
    """When edgecolor is not set, the ring color mirrors the face color."""
    fig, ax = plt.subplots()
    pp.lineplot(
        data=sample_data, x="x", y="y", hue="g", style="g",
        markers=True, ax=ax,
    )
    fig.canvas.draw()
    mecs = _legend_marker_edgecolors(ax.get_legend())
    assert mecs
    # The three hue levels use distinct palette colors; none should be black.
    blacks = [m for m in mecs if m[:3] == pytest.approx(to_rgba("black")[:3], abs=0.01)]
    assert not blacks, "no handle should default to black when edgecolor is unset"


def test_pointplot_legend_honors_edgecolor_arg(sample_data):
    fig, ax = plt.subplots()
    pp.pointplot(
        data=sample_data, x="x", y="y", hue="g", ax=ax, edgecolor="black",
    )
    fig.canvas.draw()
    mecs = _legend_marker_edgecolors(ax.get_legend())
    assert mecs, "expected at least one legend marker"
    for mec in mecs:
        assert mec[:3] == pytest.approx(to_rgba("black")[:3], abs=0.01)


def test_pointplot_legend_honors_edgecolor_rcparam(sample_data):
    pp.rcParams["edgecolor"] = "red"
    fig, ax = plt.subplots()
    pp.pointplot(data=sample_data, x="x", y="y", hue="g", ax=ax)
    fig.canvas.draw()
    mecs = _legend_marker_edgecolors(ax.get_legend())
    assert mecs
    for mec in mecs:
        assert mec[:3] == pytest.approx(to_rgba("red")[:3], abs=0.01)
