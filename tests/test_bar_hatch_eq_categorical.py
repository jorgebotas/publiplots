"""Regression tests for the `hatch == categorical_axis` path in barplot.

When hatch is the same column as the categorical axis AND hue is a separate
column with fewer levels than the categorical axis, the old code computed
`bar_color = palette[hue_order[hue_idx]]` with `hue_idx = axis_idx`, which
indexes past `hue_order` when `n_hue < n_axis`. The fix moves the bar_color
computation inside the recolor gate (it's dead work on this path).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pytest

import publiplots as pp


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _mismatched_df(seed=0):
    rng = np.random.default_rng(seed)
    n = 60
    return pd.DataFrame({
        "cond": rng.choice(["A", "B", "C"], size=n),     # 3 levels (categorical axis)
        "treat": rng.choice(["ctrl", "trt"], size=n),    # 2 levels (hue, < n_axis)
        "value": rng.normal(size=n),
    })


def test_bar_hatch_equals_categorical_with_fewer_hue_levels_does_not_raise():
    """Regression: hue=treat (2), hatch=cond=x (3) must not raise IndexError."""
    df = _mismatched_df()
    fig, ax = pp.barplot(
        data=df, x="cond", y="value",
        hue="treat", hatch="cond",
        palette={"ctrl": "#ff0000", "trt": "#00ff00"},
    )
    # If we get here without IndexError, the fix holds. Also assert the
    # expected 6 bars were drawn (n_axis × n_hue).
    bars = [p for p in ax.patches if hasattr(p, "get_height")]
    assert len(bars) == 6


def test_bar_hatch_equals_categorical_applies_distinct_hatch_per_column():
    """The hatch patterns must still match the categorical axis column."""
    df = _mismatched_df()
    fig, ax = pp.barplot(
        data=df, x="cond", y="value",
        hue="treat", hatch="cond",
        palette={"ctrl": "#ff0000", "trt": "#00ff00"},
        hatch_map={"A": "", "B": "///", "C": "xxx"},
    )
    bars = [p for p in ax.patches if hasattr(p, "get_height")]
    # Each bar's hatch pattern should be one of the three from hatch_map.
    patterns = {p.get_hatch() for p in bars}
    assert patterns <= {"", "///", "xxx"}
    # All three patterns should appear (every category represented).
    assert patterns == {"", "///", "xxx"}
