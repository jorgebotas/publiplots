"""4-sided × 2-anchor test matrix for ``pp.legend_group``.

Verifies that the right ``FigureLayout`` reservation field grows for each
combination of ``side=`` and anchor mode (figure-anchored via no
``anchor=``, axes-anchored via an explicit axes). Also guards the
back-compat call signature.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import publiplots as pp
from publiplots.utils.legend import create_legend_handles
from publiplots.utils.legend_entries import LegendEntry, stash_entry


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _sample_handles():
    return create_legend_handles(
        labels=["A", "B", "C"],
        colors=list(pp.color_palette("pastel", 3)),
        alpha=0.2, linewidth=1.0,
    )


# --- figure-anchored ---------------------------------------------------------


@pytest.mark.parametrize("side,field", [
    ("right", "legend_column"),
    ("bottom", "legend_band_bottom"),
    ("left", "legend_band_left"),
    ("top", "legend_band_top"),
])
def test_figure_anchored_reserves_correct_field(side, field):
    fig, axes = pp.subplots(2, 2, axes_size=(35, 25))
    for ax in np.atleast_2d(axes).flat:
        ax.plot([0, 1, 2], [0, 1, 0])
    group = pp.legend_group(side=side)
    group.add_legend(handles=_sample_handles(), label="group")
    fig.canvas.draw()
    layout = fig._publiplots_layout
    assert getattr(layout, field) > 5.0, (
        f"side={side!r}: {field} should absorb the legend; "
        f"got {getattr(layout, field):.1f}"
    )
    for other_field in (
        "legend_column", "legend_band_bottom", "legend_band_left", "legend_band_top"
    ):
        if other_field == field:
            continue
        assert getattr(layout, other_field) < 0.5, (
            f"side={side!r}: {other_field} should stay zero; "
            f"got {getattr(layout, other_field):.1f}"
        )


# --- axes-anchored -----------------------------------------------------------


def test_axes_anchored_right_pins_to_column():
    fig, axes = pp.subplots(2, 2, axes_size=(35, 25))
    for ax in np.atleast_2d(axes).flat:
        ax.plot([0, 1, 2], [0, 1, 0])
    # Anchor to axes[0, 0] — column 0. The legend should push column 0
    # wider, not figure-level legend_column.
    group = pp.legend_group(anchor=axes[0, 0], side="right")
    group.add_legend(handles=_sample_handles(), label="group")
    fig.canvas.draw()
    layout = fig._publiplots_layout
    assert layout.right[0] > 10.0, (
        f"right[0] should absorb the legend (axes-anchored); "
        f"got {layout.right[0]:.1f}"
    )
    assert layout.right[1] < 5.0, f"right[1] baseline, got {layout.right[1]:.1f}"
    assert layout.legend_column < 0.5, (
        f"legend_column should stay zero for axes-anchored; "
        f"got {layout.legend_column:.1f}"
    )


def test_axes_anchored_bottom_pins_to_row():
    fig, axes = pp.subplots(2, 2, axes_size=(35, 25))
    for ax in np.atleast_2d(axes).flat:
        ax.plot([0, 1, 2], [0, 1, 0])
    # anchor=axes[0, 0] is in row 0. Bottom-anchored pushes xlabel_space[0]
    # wider (the row's bottom reservation), not legend_band_bottom.
    group = pp.legend_group(anchor=axes[0, 0], side="bottom")
    group.add_legend(handles=_sample_handles(), label="group")
    fig.canvas.draw()
    layout = fig._publiplots_layout
    assert layout.xlabel_space[0] > 10.0, (
        f"xlabel_space[0] should absorb legend; got {layout.xlabel_space[0]:.1f}"
    )
    assert layout.legend_band_bottom < 0.5, (
        f"legend_band_bottom should stay zero; got {layout.legend_band_bottom:.1f}"
    )


# --- validation / back-compat ------------------------------------------------


def test_side_invalid_raises():
    fig, _ = pp.subplots(1, 1, axes_size=(40, 30))
    with pytest.raises(ValueError, match="side must be"):
        pp.legend_group(side="diagonal")


def test_back_compat_anchor_axes_defaults_to_side_right():
    """The pre-0.9 signature pp.legend_group(anchor=axes[-1]) must keep
    working as axes-anchored side='right'."""
    fig, axes = pp.subplots(1, 3, axes_size=(40, 30))
    for ax in axes:
        ax.plot([0, 1, 2], [0, 1, 0])
    group = pp.legend_group(anchor=axes[-1])
    assert group._side == "right"
    assert group._anchor_kind == "axes"
    group.add_legend(handles=_sample_handles(), label="group")
    fig.canvas.draw()
    layout = fig._publiplots_layout
    # Axes-anchored right on the last column → right[-1] grows.
    assert layout.right[-1] > 10.0
    assert layout.legend_column < 0.5


def test_figure_anchored_auto_collect_still_works():
    """pp.legend_group() without anchor auto-collects stashed entries."""
    fig, axes = pp.subplots(1, 2, axes_size=(40, 30))
    stash_entry(
        axes[0],
        LegendEntry.build(
            "g", "hue",
            handles=_sample_handles(),
            labels=("A", "B", "C"),
        ),
    )
    group = pp.legend_group()
    fig.canvas.draw()
    assert group._materialized
    assert len(group._builder.elements) == 1
    assert group._builder.elements[0][1].get_title().get_text() == "g"
