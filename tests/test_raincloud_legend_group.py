"""Integration tests for raincloudplot inside a pp.legend_group."""
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


def _rain_df(seed=0):
    rng = np.random.default_rng(seed)
    n = 90
    return pd.DataFrame({
        "condition": rng.choice(["Control", "Low", "High"], size=n),
        "response": rng.normal(50, 8, size=n),
    })


def test_raincloud_in_group_stashes_via_violin():
    """Raincloud forwards legend to its inner violin; the hue entry must land
    on the same ax the user sees."""
    from matplotlib.legend import Legend
    df = _rain_df()
    fig, axes = pp.subplots(1, 2, axes_size=(50, 40))
    pp.legend_group(anchor=axes[-1])
    pp.raincloudplot(
        data=df, x="condition", y="response", hue="condition",
        palette="pastel", ax=axes[0],
    )
    fig.canvas.draw()
    # Entry stashed on the raincloud axis (by the inner violin call)
    names_kinds = [(e.name, e.kind) for e in get_entries(axes[0])]
    assert ("condition", "hue") in names_kinds
    # No per-axis legend artist (group claimed it)
    per_axis_legends = [c for c in axes[0].get_children() if isinstance(c, Legend)]
    assert per_axis_legends == []


def test_raincloud_accepts_legend_dict():
    """Type widening: legend=dict is accepted and forwarded to violin."""
    df = _rain_df()
    fig, ax = pp.subplots(axes_size=(50, 40))
    # Should not raise a TypeError
    pp.raincloudplot(
        data=df, x="condition", y="response", hue="condition",
        palette="pastel", legend={"hue": False}, ax=ax,
    )
    # With hue suppressed, nothing is stashed
    assert get_entries(ax) == []
