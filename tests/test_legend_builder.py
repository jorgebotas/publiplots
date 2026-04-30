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


def test_colorbar_title_follows_axes_after_tight_layout():
    """The `fig.text` colorbar title must be registered with the reactor
    and reposition in lockstep with its colorbar after layout changes."""
    import numpy as np
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot([0, 1], [0, 1])

    sm = ScalarMappable(cmap="viridis", norm=Normalize(0, 1))
    builder = pp.legend(ax, auto=False, x_offset=2, vpad=5)
    cbar = builder.add_colorbar(mappable=sm, label="Value", height=15, width=4.5)
    fig.canvas.draw()

    # Find the title Text in builder.elements
    title_obj = next(
        (obj for kind, obj in builder.elements if kind == "text"),
        None,
    )
    assert title_obj is not None, "expected colorbar title in builder.elements"

    initial_title_pos = title_obj.get_position()
    initial_cbar_x = cbar.ax.get_position().x0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.tight_layout()
    fig.canvas.draw()

    final_title_pos = title_obj.get_position()
    final_cbar_x = cbar.ax.get_position().x0

    # Both must have moved (axes grew under tight_layout)
    assert abs(final_cbar_x - initial_cbar_x) > 1e-4
    assert abs(final_title_pos[0] - initial_title_pos[0]) > 1e-4

    # Title should still be horizontally aligned with the colorbar
    # (same left edge within 0.5 mm ≈ 2e-3 of a 5-inch figure width)
    title_vs_cbar_delta = final_title_pos[0] - final_cbar_x
    initial_delta = initial_title_pos[0] - initial_cbar_x
    assert abs(title_vs_cbar_delta - initial_delta) < 3e-3


def test_set_notebook_style_does_not_force_constrained_layout():
    """publiplots styles intentionally leave the layout engine off so users
    can declare a fixed axes size and let decorations extend the figure.
    Users who want constrained_layout can opt in per-figure."""
    pp.set_notebook_style()
    try:
        assert matplotlib.rcParams["figure.constrained_layout.use"] is False
    finally:
        pp.reset_style()


def test_set_publication_style_does_not_force_constrained_layout():
    """Same philosophy for publication style."""
    pp.set_publication_style()
    try:
        assert matplotlib.rcParams["figure.constrained_layout.use"] is False
    finally:
        pp.reset_style()
