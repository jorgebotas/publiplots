"""The shared rectangle legend handler and the hatch width of its swatch.

``HandlerRectangle`` draws a swatch's hatch at the width its handle
carries: one set on it explicitly, or else the
``rcParams["hatch.linewidth"]`` in force when the handle was created. These
pin that for a handle no pie built, since every other rectangle-swatch
plotter goes through the same handler.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.legend import Legend
from matplotlib.patches import Patch, Rectangle

import publiplots as pp

pytestmark = pytest.mark.skipif(
    not hasattr(Patch, "set_hatch_linewidth"),
    reason="Patch.set_hatch_linewidth needs matplotlib>=3.10")


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


@pytest.fixture
def df():
    return pd.DataFrame({
        "g": ["a", "a", "b", "b"],
        "h": ["x", "y", "x", "y"],
        "v": [1.0, 2.0, 3.0, 4.0],
    })


def _hatched_swatches(fig):
    fig.canvas.draw()
    return [r for leg in fig.findobj(Legend) for r in leg.findobj(Rectangle)
            if r.get_hatch()]


def test_a_barplot_hatch_swatch_is_unchanged_outside_an_rc_context(df):
    """Regression guard for every non-pie plotter: with no rc_context in
    play, a barplot swatch's hatch stays at rcParams["hatch.linewidth"] --
    not the bars' linewidth=, which barplot never puts on its handles."""
    fig, ax = pp.subplots(axes_size=(50, 40))
    pp.barplot(data=df, x="g", y="v", hue="h", hatch="h", linewidth=2.0,
               ax=ax)
    swatches = _hatched_swatches(fig)
    assert swatches
    expected = matplotlib.rcParams["hatch.linewidth"]
    assert expected == pytest.approx(1.0)
    assert all(s.get_hatch_linewidth() == pytest.approx(expected)
               for s in swatches)


def test_a_deferred_band_keeps_the_hatch_width_the_handle_was_made_with(df):
    """The handle records rcParams["hatch.linewidth"] when barplot builds it.
    A band declared inside the context renders after it has exited, and the
    swatch keeps the creation-time 3.0 rather than the render-time 1.0."""
    with matplotlib.rc_context({"hatch.linewidth": 3.0}):
        fig, ax = pp.subplots(axes_size=(50, 40))
        pp.barplot(data=df, x="g", y="v", hue="h", hatch="h", ax=ax)
        pp.legend(side="right")
    assert matplotlib.rcParams["hatch.linewidth"] == pytest.approx(1.0)
    swatches = _hatched_swatches(fig)
    assert swatches
    assert all(s.get_hatch_linewidth() == pytest.approx(3.0)
               for s in swatches)
