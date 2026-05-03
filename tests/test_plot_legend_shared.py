"""Direct tests for the shared plot_legend helpers lifted from scatter/point/etc."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
import pytest

from publiplots.utils.legend_entries import get_entries
from publiplots.utils.legend import LinePatch, MarkerPatch
from publiplots.utils.plot_legend import (
    get_size_ticks,
    merge_categorical_entries,
    resolve_size_map,
    stash_continuous_hue,
    resolve_style_maps,
)


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


# ---- get_size_ticks ----

def test_get_size_ticks_returns_labels_and_sizes():
    """With simple numeric input, returns equal-length labels and sizes."""
    values = np.linspace(1, 10, 20)
    tick_labels, tick_sizes = get_size_ticks(
        values=values,
        sizes=(20, 200),
        size_norm=Normalize(vmin=1, vmax=10),
        nbins=4,
        min_n_ticks=3,
        include_min_max=False,
    )
    assert len(tick_labels) == len(tick_sizes)
    assert len(tick_labels) >= 2
    # All size values should fall within the requested (min, max) range.
    # Note: helper converts points^2 to a markersize via sqrt(size/pi)*2.
    min_ms = np.sqrt(20 / np.pi) * 2
    max_ms = np.sqrt(200 / np.pi) * 2
    assert all(min_ms <= s <= max_ms + 1e-6 for s in tick_sizes)


# ---- stash_continuous_hue ----

def test_stash_continuous_hue_stashes_scalarmappable():
    fig, ax = plt.subplots()
    stash_continuous_hue(ax, name="score", palette="viridis",
                         hue_norm=Normalize(0, 10))
    entries = get_entries(ax)
    assert len(entries) == 1
    entry = entries[0]
    assert entry.name == "score"
    assert entry.kind == "hue"
    assert entry.labels == ()
    assert isinstance(entry.handles[0], ScalarMappable)


# ---- resolve_style_maps ----

def test_resolve_style_maps_style_none_returns_empty_dicts():
    df = pd.DataFrame({"x": [1, 2, 3]})
    marker_map, linestyle_map = resolve_style_maps(
        style=None, data=df, style_order=None, markers=True, dashes=True,
    )
    assert marker_map == {}
    assert linestyle_map == {}


def test_resolve_style_maps_markers_only():
    df = pd.DataFrame({"g": ["A", "B", "C"]})
    marker_map, linestyle_map = resolve_style_maps(
        style="g", data=df, style_order=None, markers=True, dashes=None,
    )
    assert set(marker_map.keys()) == {"A", "B", "C"}
    assert linestyle_map == {}


def test_resolve_style_maps_dashes_only():
    df = pd.DataFrame({"g": ["A", "B", "C"]})
    marker_map, linestyle_map = resolve_style_maps(
        style="g", data=df, style_order=None, markers=None, dashes=True,
    )
    assert marker_map == {}
    assert set(linestyle_map.keys()) == {"A", "B", "C"}


def test_resolve_style_maps_both():
    df = pd.DataFrame({"g": ["A", "B"]})
    marker_map, linestyle_map = resolve_style_maps(
        style="g", data=df, style_order=None, markers=True, dashes=True,
    )
    assert set(marker_map.keys()) == {"A", "B"}
    assert set(linestyle_map.keys()) == {"A", "B"}


def test_resolve_style_maps_respects_style_order():
    df = pd.DataFrame({"g": ["A", "B", "C"]})
    marker_map, _ = resolve_style_maps(
        style="g", data=df, style_order=["C", "A", "B"],
        markers=True, dashes=None,
    )
    assert list(marker_map.keys()) == ["C", "A", "B"]


# ---- merge_categorical_entries ----

def test_merge_categorical_entries_colors_plus_linestyles_yields_line_patches():
    """Merged hue + style (linestyles) for lineplot: one colored dashed line
    per row, stashed under kind='hue'."""
    entry = merge_categorical_entries(
        name="g",
        labels=["A", "B"],
        colors=["#ff0000", "#0000ff"],
        linestyles=[(4, 2), (1, 1)],
    )
    assert entry.name == "g"
    assert entry.kind == "hue"
    assert len(entry.handles) == 2
    assert all(isinstance(h, LinePatch) for h in entry.handles)
    linestyles = [h.get_linestyle() for h in entry.handles]
    assert (4, 2) in linestyles and (1, 1) in linestyles


def test_merge_categorical_entries_colors_plus_markers_yields_marker_patches():
    """Merged hue + style (markers) for scatter: one colored shaped marker
    per row."""
    entry = merge_categorical_entries(
        name="g",
        labels=["A", "B", "C"],
        colors=["#ff0000", "#00ff00", "#0000ff"],
        markers=["o", "^", "s"],
    )
    assert entry.name == "g"
    assert entry.kind == "hue"
    assert len(entry.handles) == 3
    assert all(isinstance(h, MarkerPatch) for h in entry.handles)
    assert [h.get_marker() for h in entry.handles] == ["o", "^", "s"]


# ---- resolve_size_map ----

def test_resolve_size_map_none_returns_empty():
    df = pd.DataFrame({"g": ["A", "B"]})
    assert resolve_size_map(size=None, data=df) == {}


def test_resolve_size_map_dict_is_passthrough_filtered_to_order():
    df = pd.DataFrame({"g": ["A", "B", "C"]})
    out = resolve_size_map(
        size="g", data=df, size_order=["C", "A"],
        sizes={"A": 1.0, "B": 2.0, "C": 3.0},
    )
    assert list(out.keys()) == ["C", "A"]
    assert out == {"C": 3.0, "A": 1.0}


def test_resolve_size_map_tuple_interpolates_across_categories():
    df = pd.DataFrame({"g": ["low", "med", "high"]})
    out = resolve_size_map(
        size="g", data=df, size_order=["low", "med", "high"],
        sizes=(1.0, 5.0),
    )
    assert list(out.keys()) == ["low", "med", "high"]
    assert out["low"] == 1.0
    assert out["med"] == 3.0
    assert out["high"] == 5.0


def test_resolve_size_map_list_cycles():
    df = pd.DataFrame({"g": ["A", "B", "C", "D"]})
    out = resolve_size_map(
        size="g", data=df, size_order=["A", "B", "C", "D"],
        sizes=[1.0, 2.0],
    )
    assert out == {"A": 1.0, "B": 2.0, "C": 1.0, "D": 2.0}


def test_resolve_size_map_default_interpolates_1_to_4():
    """When sizes is None, default falls back to (1.0, 4.0)."""
    df = pd.DataFrame({"g": ["A", "B", "C"]})
    out = resolve_size_map(size="g", data=df,
                           size_order=["A", "B", "C"], sizes=None)
    assert out == {"A": 1.0, "B": 2.5, "C": 4.0}


def test_resolve_size_map_single_category_uses_upper_bound():
    df = pd.DataFrame({"g": ["only"]})
    out = resolve_size_map(size="g", data=df, sizes=(1.0, 5.0))
    assert out == {"only": 5.0}


def test_merge_categorical_entries_respects_handle_kwargs():
    """Alpha / linewidth from handle_kwargs reach the underlying handles."""
    entry = merge_categorical_entries(
        name="g",
        labels=["A"],
        colors=["#123456"],
        linestyles=["-"],
        handle_kwargs={"alpha": 0.5, "linewidth": 3.0},
    )
    handle = entry.handles[0]
    assert handle.get_alpha() == 0.5
    assert handle.get_linewidth() == 3.0
