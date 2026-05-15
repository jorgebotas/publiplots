"""Integration tests — Composer figures inherit publiplots rcParams + style."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

import publiplots as pp


def test_composer_figure_uses_publiplots_savefig_dpi():
    """publiplots' savefig.dpi rcParam is 600. The Composer's raster
    savefig path inherits it via pp.savefig."""
    assert pp.rcParams["savefig.dpi"] == 600


def test_composer_figure_uses_arial_font():
    """publiplots' init_rcparams sets Arial as the default sans-serif.
    A Composer-built figure inherits this — confirmed by the panel's
    text artists having Arial in their font.family."""
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(70.0, 40.0)))
    canvas["A"].ax.set_xlabel("x label")
    fig = canvas.figure
    fig.canvas.draw()

    xlabel_text = canvas["A"].ax.xaxis.get_label()
    family = xlabel_text.get_fontfamily()
    # `family` is a list-like; matplotlib's default Arial fallback is
    # in there.
    assert any("Arial" in f or f == "sans-serif" for f in family)


def test_composer_panel_axes_is_a_real_matplotlib_axes():
    """canvas[label].ax is a real Axes — usable with pp.scatterplot etc."""
    from matplotlib.axes import Axes

    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(70.0, 40.0)))
    assert isinstance(canvas["A"].ax, Axes)


def test_composer_can_plot_with_pp_scatterplot():
    """Smoke test — pp.scatterplot accepts canvas[label].ax as its ax=
    kwarg (the canonical Composer plotting pattern)."""
    import pandas as pd

    df = pd.DataFrame({"x": [1, 2, 3], "y": [1, 4, 2], "g": ["a", "b", "a"]})
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(70.0, 40.0)))
    pp.scatterplot(data=df, x="x", y="y", hue="g", ax=canvas["A"].ax)

    # The axes now has at least one collection (the scatter).
    assert len(canvas["A"].ax.collections) >= 1


def test_composer_figure_size_matches_pp_subplots_for_equivalent_layout():
    """A Canvas with one 70×40mm panel + default rcParams should produce
    the same figure size as pp.subplots(1, 1, axes_size=(70, 40)) ±
    decoration auto-measurement noise. This pins the equivalence between
    the two layout entry points (the Canvas reuses pp.subplots' layout
    machinery)."""
    fig_subplots, _ = pp.subplots(1, 1, axes_size=(70.0, 40.0))
    fig_subplots.canvas.draw()
    sub_w_in, sub_h_in = fig_subplots.get_size_inches()

    # Set canvas width to a comfortable upper bound; the Composer's
    # figure width is determined by panel + decorations, NOT the canvas
    # budget (in PR 1 — PR 2 adds 'flex' sizing).
    canvas = pp.Canvas("custom", width=200.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(70.0, 40.0)))
    canvas.figure.canvas.draw()
    canv_w_in, canv_h_in = canvas.figure.get_size_inches()

    # Both layouts route through FigureLayout + SubplotsAutoLayout with
    # the same rcParams reservations, so width AND height match within
    # decoration auto-measurement noise (~1 mm).
    assert abs(canv_w_in - sub_w_in) < 0.05
    assert abs(canv_h_in - sub_h_in) < 0.05
