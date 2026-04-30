"""Tests for LegendBuilder reactivity under layout changes."""
import pytest
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import publiplots as pp
from publiplots.utils.legend import create_legend_handles


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _anchor_x_fig_frac(leg, fig):
    """Read the legend's anchor x in figure-fraction coordinates.

    `leg._bbox_to_anchor` is a TransformedBbox whose `.get_points()` returns
    display pixels. Divide by figure width to get figure fraction.
    """
    px = leg._bbox_to_anchor.get_points()[0, 0]
    return px / fig.get_window_extent().width


def test_legend_follows_axes_after_tight_layout():
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot([0, 1], [0, 1], label="line")
    handles = create_legend_handles(
        labels=["A", "B"], colors=["#5d83c3", "#c0392b"],
        alpha=0.2, linewidth=1.0,
    )
    builder = pp.legend(ax, auto=False, x_offset=2, vpad=5)
    leg = builder.add_legend(handles=handles, label="group")
    fig.canvas.draw()

    # Snapshot anchor in figure coords
    initial_anchor_x = _anchor_x_fig_frac(leg, fig)
    initial_ax_x1 = ax.get_position().x1

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # reactor warns under tight_layout; that's fine
        plt.tight_layout()
    fig.canvas.draw()

    final_anchor_x = _anchor_x_fig_frac(leg, fig)
    final_ax_x1 = ax.get_position().x1

    # Axes moved
    assert abs(final_ax_x1 - initial_ax_x1) > 1e-3
    # Anchor followed (within ~1 pixel worth of fractional space)
    anchor_vs_edge_delta = (final_anchor_x - final_ax_x1) - (initial_anchor_x - initial_ax_x1)
    assert abs(anchor_vs_edge_delta) < 2e-3
