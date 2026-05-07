"""Tests for scatterplot legend stashing via LegendEntry."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pytest

import publiplots as pp
from publiplots.utils.legend_entries import get_entries


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _scatter_df(seed=0):
    rng = np.random.default_rng(seed)
    n = 30
    return pd.DataFrame({
        "x": rng.normal(size=n),
        "y": rng.normal(size=n),
        "g": rng.choice(["A", "B", "C"], size=n),
        "m": rng.uniform(1, 5, size=n),
    })


def test_scatterplot_stashes_hue_entry():
    df = _scatter_df()
    ax = pp.scatterplot(data=df, x="x", y="y", hue="g", palette="pastel")
    entries = get_entries(ax)
    names_kinds = [(e.name, e.kind) for e in entries]
    assert ("g", "hue") in names_kinds


def test_scatterplot_stashes_size_entry():
    df = _scatter_df()
    ax = pp.scatterplot(data=df, x="x", y="y", size="m")
    entries = get_entries(ax)
    names_kinds = [(e.name, e.kind) for e in entries]
    assert ("m", "size") in names_kinds


def test_scatterplot_legend_dict_suppresses_hue():
    df = _scatter_df()
    ax = pp.scatterplot(
        data=df, x="x", y="y", hue="g", size="m",
        palette="pastel",
        legend={"hue": False},
    )
    entries = get_entries(ax)
    names = [e.name for e in entries]
    # hue is suppressed, size is still stashed
    assert "g" not in names
    assert "m" in names


def test_scatterplot_legend_false_stashes_nothing():
    df = _scatter_df()
    ax = pp.scatterplot(
        data=df, x="x", y="y", hue="g", size="m",
        palette="pastel",
        legend=False,
    )
    assert get_entries(ax) == []


def test_scatterplot_in_group_suppresses_per_axis_render():
    """With a legend_group active, the scatter should NOT attach a per-axis
    Legend artist to ax (the group handles it)."""
    from matplotlib.legend import Legend
    df = _scatter_df()
    fig, axes = pp.subplots(1, 2, axes_size=(50, 40))
    pp.legend(anchor=axes[-1])
    pp.scatterplot(data=df, x="x", y="y", hue="g", palette="pastel", ax=axes[0])
    fig.canvas.draw()
    per_axis_legends = [c for c in axes[0].get_children() if isinstance(c, Legend)]
    assert per_axis_legends == []


def test_scatterplot_categorical_size_does_not_crash():
    """Regression: previously crashed in get_size_ticks via np.isnan(str)."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "x": rng.normal(size=20),
        "y": rng.normal(size=20),
        "tier": rng.choice(["low", "med", "high"], size=20),
    })
    ax = pp.scatterplot(data=df, x="x", y="y", size="tier")
    entries = get_entries(ax)
    [entry] = [e for e in entries if e.kind == "size"]
    assert entry.name == "tier"


def test_scatterplot_categorical_size_explicit_dict():
    """With an explicit ``{cat: area_points²}`` map, the legend markers use
    the diameter sqrt(area/pi)*2."""
    import math
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "x": rng.normal(size=20),
        "y": rng.normal(size=20),
        "tier": rng.choice(["low", "med", "high"], size=20),
    })
    ax = pp.scatterplot(
        data=df, x="x", y="y", size="tier",
        sizes={"low": 20, "med": 100, "high": 400},
    )
    [entry] = [e for e in get_entries(ax) if e.kind == "size"]
    by_label = dict(zip(entry.labels, entry.handles))
    for label, area in [("low", 20), ("med", 100), ("high", 400)]:
        expected = math.sqrt(area / math.pi) * 2
        assert abs(by_label[label].get_markersize() - expected) < 1e-6


def test_scatterplot_hue_equals_style_single_entry():
    """When hue and style reference the same categorical column, stash one
    merged LegendEntry (colored shaped marker per row) instead of duplicate
    hue + style entries."""
    df = _scatter_df()
    ax = pp.scatterplot(
        data=df, x="x", y="y",
        hue="g", style="g",
        palette="pastel",
        markers=["o", "^", "s"],
    )
    entries = get_entries(ax)
    assert len(entries) == 1
    entry = entries[0]
    assert entry.name == "g"
    assert entry.kind == "hue"
    # Each handle must carry a marker (verifies the style dimension made
    # it onto the merged swatch).
    markers = [h.get_marker() for h in entry.handles]
    assert set(markers) == {"o", "^", "s"}


def test_scatterplot_hue_equals_style_different_columns_kept_separate():
    """When hue and style reference *different* columns, keep them as two
    separate entries."""
    df = _scatter_df().assign(m_cat=np.tile(["r", "s"], len(_scatter_df()) // 2))
    ax = pp.scatterplot(
        data=df, x="x", y="y",
        hue="g", style="m_cat",
        palette="pastel",
    )
    kinds = {e.kind for e in get_entries(ax)}
    assert {"hue", "style"} <= kinds


def test_scatterplot_no_group_renders_per_axis_legend():
    from matplotlib.legend import Legend
    df = _scatter_df()
    ax = pp.scatterplot(data=df, x="x", y="y", hue="g", palette="pastel")
    ax.get_figure().canvas.draw()
    per_axis_legends = [c for c in ax.get_children() if isinstance(c, Legend)]
    assert len(per_axis_legends) >= 1


# ---------------------------------------------------------------------------
# stripplot migration
# ---------------------------------------------------------------------------


def test_stripplot_stashes_hue_entry():
    df = _scatter_df()
    ax = pp.stripplot(data=df, x="g", y="y", hue="g", palette="pastel")
    entries = get_entries(ax)
    names_kinds = [(e.name, e.kind) for e in entries]
    assert ("g", "hue") in names_kinds


def test_stripplot_legend_false_stashes_nothing():
    df = _scatter_df()
    ax = pp.stripplot(
        data=df, x="g", y="y", hue="g", palette="pastel", legend=False,
    )
    assert get_entries(ax) == []


# ---------------------------------------------------------------------------
# swarmplot migration
# ---------------------------------------------------------------------------


def test_swarmplot_stashes_hue_entry():
    df = _scatter_df()
    ax = pp.swarmplot(data=df, x="g", y="y", hue="g", palette="pastel")
    entries = get_entries(ax)
    names_kinds = [(e.name, e.kind) for e in entries]
    assert ("g", "hue") in names_kinds


def test_swarmplot_legend_false_stashes_nothing():
    df = _scatter_df()
    ax = pp.swarmplot(
        data=df, x="g", y="y", hue="g", palette="pastel", legend=False,
    )
    assert get_entries(ax) == []


# ---------------------------------------------------------------------------
# pointplot migration
# ---------------------------------------------------------------------------


def test_pointplot_stashes_hue_entry():
    df = _scatter_df()
    ax = pp.pointplot(data=df, x="g", y="y", hue="g", palette="pastel")
    entries = get_entries(ax)
    names_kinds = [(e.name, e.kind) for e in entries]
    assert ("g", "hue") in names_kinds


def test_pointplot_legend_false_stashes_nothing():
    df = _scatter_df()
    ax = pp.pointplot(
        data=df, x="g", y="y", hue="g", palette="pastel", legend=False,
    )
    assert get_entries(ax) == []
