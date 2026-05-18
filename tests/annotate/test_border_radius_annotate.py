"""Regression tests for #173 — annotate must work with border_radius set.

When ``pp.rcParams['bar.border_radius']`` (or ``box.border_radius``) is
non-zero, ``apply_border_radius`` swaps matplotlib's ``Rectangle`` /
``PathPatch`` artists for publiplots' custom ``_RoundedBarPatch``. The
annotate subsystem's isinstance gates previously rejected those rounded
patches and silently emitted zero labels.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import publiplots as pp


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def _reset_border_radius():
    """Restore zero border_radius after each test, even on failure."""
    try:
        yield
    finally:
        pp.rcParams["bar.border_radius"] = (0.0, 0.0)
        pp.rcParams["box.border_radius"] = (0.0, 0.0)


def _bar_df():
    return pd.DataFrame({
        "cat": pd.Categorical(["A", "B", "C"], categories=["A", "B", "C"]),
        "value": [1.0, 2.0, 3.0],
        "n": [10, 20, 30],
    })


def _stacked_df():
    return pd.DataFrame({
        "cat": pd.Categorical(["A", "A", "B", "B"], categories=["A", "B"]),
        "hue": pd.Categorical(["x", "y", "x", "y"], categories=["x", "y"]),
        "value": [1.0, 2.0, 3.0, 4.0],
    })


def _box_df(seed: int = 0):
    rng = np.random.default_rng(seed)
    rows = []
    for g, base in zip(("A", "B", "C"), (1.0, 2.0, 3.0)):
        for v in rng.normal(base, 0.5, 30):
            rows.append({"g": g, "y": float(v)})
    df = pd.DataFrame(rows)
    df["g"] = df["g"].astype("category")
    return df


def test_bar_values_with_border_radius(_reset_border_radius):
    """bar_values must label every rounded bar (was: silently 0 labels)."""
    pp.rcParams["bar.border_radius"] = (1.0, 0.0)
    fig, ax = plt.subplots()
    pp.barplot(data=_bar_df(), x="cat", y="value", ax=ax)
    pp.annotate(ax, kind="bar_values", color="hue")
    assert len(ax.texts) == 3, (
        f"expected 3 labels on rounded bars, got {len(ax.texts)}"
    )
    assert [t.get_text() for t in ax.texts] == ["1.00", "2.00", "3.00"]
    plt.close(fig)


def test_bar_custom_with_border_radius(_reset_border_radius):
    """bar_custom must label every rounded bar with explicit labels."""
    pp.rcParams["bar.border_radius"] = (1.0, 0.0)
    fig, ax = plt.subplots()
    pp.barplot(data=_bar_df(), x="cat", y="value", ax=ax)
    pp.annotate(ax, kind="bar_custom", labels="n")
    assert len(ax.texts) == 3
    assert [t.get_text() for t in ax.texts] == ["10", "20", "30"]
    plt.close(fig)


def test_stacked_barplot_with_border_radius(_reset_border_radius):
    """Stacked barplot patch walker must accept rounded patches."""
    pp.rcParams["bar.border_radius"] = (1.0, 0.0)
    fig, ax = plt.subplots()
    pp.barplot(
        data=_stacked_df(),
        x="cat",
        y="value",
        hue="hue",
        multiple="stack",
        ax=ax,
        annotate=True,
    )
    # 2 cats × 2 hues = 4 stacked segments, each gets a label.
    assert len(ax.texts) == 4, (
        f"expected 4 labels on rounded stacked bars, got {len(ax.texts)}"
    )
    plt.close(fig)


def test_box_stats_with_border_radius(_reset_border_radius):
    """box_stats must label every rounded box (was: silently 0 labels)."""
    pp.rcParams["box.border_radius"] = (1.0, 0.0)
    fig, ax = plt.subplots()
    pp.boxplot(data=_box_df(), x="g", y="y", ax=ax)
    pp.annotate(ax, kind="box_stats", stats=["median"])
    # 3 boxes × 1 stat = 3 labels.
    assert len(ax.texts) == 3, (
        f"expected 3 labels on rounded boxes, got {len(ax.texts)}"
    )
    plt.close(fig)
