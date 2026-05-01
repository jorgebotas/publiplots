"""Tests for the shared plot_legend helpers."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pytest

import publiplots as pp
from publiplots.utils.legend_entries import get_entries, LegendEntry, stash_entry
from publiplots.utils.plot_legend import stash_hue_legend, render_entries


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_stash_hue_legend_stashes_one_entry():
    """With hue + dict palette, stash_hue_legend produces one LegendEntry."""
    fig, ax = plt.subplots()
    palette = {"A": "#ff0000", "B": "#00ff00"}
    stash_hue_legend(
        ax,
        hue="group",
        palette=palette,
        edgecolor=None,
        alpha=0.5,
        linewidth=1.0,
        legend=True,
        legend_kws=None,
    )
    entries = get_entries(ax)
    assert len(entries) == 1
    assert (entries[0].name, entries[0].kind) == ("group", "hue")
    assert entries[0].labels == ("A", "B")


def test_stash_hue_legend_legend_false_short_circuits():
    fig, ax = plt.subplots()
    stash_hue_legend(
        ax,
        hue="group",
        palette={"A": "#ff0000"},
        edgecolor=None,
        alpha=0.5,
        linewidth=1.0,
        legend=False,
        legend_kws=None,
    )
    assert get_entries(ax) == []


def test_stash_hue_legend_no_hue_short_circuits():
    fig, ax = plt.subplots()
    stash_hue_legend(
        ax,
        hue=None,
        palette={"A": "#ff0000"},
        edgecolor=None,
        alpha=0.5,
        linewidth=1.0,
        legend=True,
        legend_kws=None,
    )
    assert get_entries(ax) == []


def test_stash_hue_legend_non_dict_palette_short_circuits():
    """Non-dict palette (e.g., a seaborn string name) short-circuits; caller
    should resolve palette to a dict before invoking."""
    fig, ax = plt.subplots()
    stash_hue_legend(
        ax,
        hue="group",
        palette="pastel",
        edgecolor=None,
        alpha=0.5,
        linewidth=1.0,
        legend=True,
        legend_kws=None,
    )
    assert get_entries(ax) == []


def test_stash_hue_legend_dict_flag_suppresses():
    fig, ax = plt.subplots()
    stash_hue_legend(
        ax,
        hue="group",
        palette={"A": "#ff0000"},
        edgecolor=None,
        alpha=0.5,
        linewidth=1.0,
        legend={"hue": False},
        legend_kws=None,
    )
    assert get_entries(ax) == []


def test_stash_hue_legend_respects_hue_label_kw():
    """legend_kws={'hue_label': 'Custom'} overrides the stashed name."""
    fig, ax = plt.subplots()
    stash_hue_legend(
        ax,
        hue="group",
        palette={"A": "#ff0000"},
        edgecolor=None,
        alpha=0.5,
        linewidth=1.0,
        legend=True,
        legend_kws={"hue_label": "Custom"},
    )
    entries = get_entries(ax)
    assert entries[0].name == "Custom"


def test_render_entries_renders_stashed_entries_per_axis():
    """After render_entries, a Legend artist is attached to ax."""
    from matplotlib.legend import Legend
    from publiplots.utils import create_legend_handles
    fig, ax = plt.subplots()
    handles = create_legend_handles(
        labels=["A", "B"], colors=["#ff0000", "#00ff00"],
        alpha=0.5, linewidth=1.0,
    )
    stash_entry(
        ax,
        LegendEntry.build(name="group", kind="hue", handles=handles, labels=("A", "B")),
    )
    render_entries(ax, flags={"hue": True, "size": True, "style": True, "marker": True, "hatch": True})
    per_axis_legends = [c for c in ax.get_children() if isinstance(c, Legend)]
    assert len(per_axis_legends) == 1


def test_render_entries_skips_entries_claimed_by_group():
    """When a legend_group claims an entry, render_entries does not attach a per-axis Legend."""
    from matplotlib.legend import Legend
    fig, axes = pp.subplots(1, 2, axes_size=(50, 40))
    pp.legend_group(anchor=axes[-1])
    from publiplots.utils import create_legend_handles
    handles = create_legend_handles(
        labels=["A", "B"], colors=["#ff0000", "#00ff00"],
        alpha=0.5, linewidth=1.0,
    )
    stash_entry(
        axes[0],
        LegendEntry.build(name="g", kind="hue", handles=handles, labels=("A", "B")),
    )
    render_entries(axes[0], flags={"hue": True, "size": True, "style": True, "marker": True, "hatch": True})
    per_axis_legends = [c for c in axes[0].get_children() if isinstance(c, Legend)]
    assert per_axis_legends == []


def test_render_entries_renders_continuous_hue_as_colorbar():
    """A ScalarMappable handle with empty labels routes through add_colorbar; no Legend is added to the axes."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.legend import Legend
    fig, ax = plt.subplots()
    mappable = ScalarMappable(norm=Normalize(0, 10), cmap="viridis")
    stash_entry(
        ax,
        LegendEntry.build(name="score", kind="hue", handles=[mappable], labels=[]),
    )
    render_entries(ax, flags={"hue": True, "size": True, "style": True, "marker": True, "hatch": True})
    # Colorbars are separate axes artists, not Legend artists on ax
    per_axis_legends = [c for c in ax.get_children() if isinstance(c, Legend)]
    assert per_axis_legends == []
