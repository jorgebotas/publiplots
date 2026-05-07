"""Tests for MultiAxesLegendGroup — unified legends across subplots."""
import pytest
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


def _handles():
    return create_legend_handles(
        labels=["A", "B"], colors=["#5d83c3", "#c0392b"],
        alpha=0.2, linewidth=1.0,
    )


def _anchor_x_fig_frac(leg, fig):
    """Convert a TransformedBbox legend anchor to figure-fraction coords."""
    px = leg._bbox_to_anchor.get_points()[0, 0]
    return px / fig.get_window_extent().width


def _anchor_y_fig_frac(leg, fig):
    py = leg._bbox_to_anchor.get_points()[0, 1]
    return py / fig.get_window_extent().height


def test_legend_group_anchors_to_chosen_axes():
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    group = pp.legend(anchor=axes[0], x_offset=2)
    leg = group.add_legend(handles=_handles(), label="Treatment", ax=axes[0])
    fig.canvas.draw()

    anchor_x = _anchor_x_fig_frac(leg, fig)
    ax0_x1 = axes[0].get_position().x1
    assert anchor_x > ax0_x1
    assert anchor_x < axes[1].get_position().x0


def test_legend_group_stacks_elements_in_one_column():
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    group = pp.legend(anchor=axes[0], x_offset=2, gap=2)
    leg_a = group.add_legend(handles=_handles(), label="A", ax=axes[0])
    leg_b = group.add_legend(handles=_handles(), label="B", ax=axes[1])
    leg_c = group.add_legend(handles=_handles(), label="C", ax=axes[2])
    fig.canvas.draw()

    xa = _anchor_x_fig_frac(leg_a, fig)
    xb = _anchor_x_fig_frac(leg_b, fig)
    xc = _anchor_x_fig_frac(leg_c, fig)
    assert abs(xa - xb) < 3e-3
    assert abs(xa - xc) < 3e-3

    ya = _anchor_y_fig_frac(leg_a, fig)
    yb = _anchor_y_fig_frac(leg_b, fig)
    yc = _anchor_y_fig_frac(leg_c, fig)
    assert ya > yb > yc


def test_legend_group_overflow_creates_new_column():
    fig, ax = plt.subplots(figsize=(5, 3))
    group = pp.legend(anchor=ax, x_offset=2, gap=1, vpad=0)
    first = group.add_legend(handles=_handles(), label="0", ax=ax)
    others = [group.add_legend(handles=_handles(), label=str(i), ax=ax)
              for i in range(1, 25)]
    fig.canvas.draw()

    x_first = _anchor_x_fig_frac(first, fig)
    x_last = _anchor_x_fig_frac(others[-1], fig)
    assert x_last > x_first + 1e-3


def test_legend_group_follows_axes_after_tight_layout():
    import warnings
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    group = pp.legend(anchor=axes[0], x_offset=2)
    leg = group.add_legend(handles=_handles(), label="A", ax=axes[0])
    fig.canvas.draw()

    initial_anchor_x = _anchor_x_fig_frac(leg, fig)
    initial_ax0_x1 = axes[0].get_position().x1

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.tight_layout()
    fig.canvas.draw()

    final_anchor_x = _anchor_x_fig_frac(leg, fig)
    final_ax0_x1 = axes[0].get_position().x1

    assert abs(final_ax0_x1 - initial_ax0_x1) > 1e-3
    delta = (final_anchor_x - final_ax0_x1) - (initial_anchor_x - initial_ax0_x1)
    assert abs(delta) < 2e-3