"""Tests for heatmap legend stashing via LegendEntry."""
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


def _matrix_df(rows=4, cols=5, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(size=(rows, cols)),
                        index=[f"r{i}" for i in range(rows)],
                        columns=[f"c{i}" for i in range(cols)])


def test_categorical_heatmap_stashes_continuous_hue_entry():
    """Categorical heatmap stashes one continuous-hue (colorbar) entry."""
    from publiplots.utils.legend_entries import is_continuous_hue
    df = _matrix_df()
    fig, ax = pp.heatmap(data=df)
    entries = get_entries(ax)
    assert len(entries) == 1
    assert entries[0].kind == "hue"
    assert is_continuous_hue(entries[0].handles)


def test_categorical_heatmap_legend_false_stashes_nothing():
    df = _matrix_df()
    fig, ax = pp.heatmap(data=df, legend=False)
    assert get_entries(ax) == []
