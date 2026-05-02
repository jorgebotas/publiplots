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
from publiplots.utils.plot_legend import (
    get_size_ticks,
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
