"""Tests for pp.boxplot 1D (univariate) mode."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.patches import PathPatch

import publiplots as pp


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    n = 30
    return pd.DataFrame({
        "cat": np.repeat(["A", "B", "C"], n // 3),
        "val": rng.normal(size=n),
        "g": np.tile(["x", "y"], n // 2),
    })


def test_univariate_y_only(df):
    ax = pp.boxplot(data=df, y="val")
    assert any(isinstance(p, PathPatch) for p in ax.patches)


def test_univariate_x_only(df):
    ax = pp.boxplot(data=df, x="val")
    assert any(isinstance(p, PathPatch) for p in ax.patches)


def test_univariate_y_with_hue(df):
    ax = pp.boxplot(data=df, y="val", hue="g")
    n_pp = sum(1 for p in ax.patches if isinstance(p, PathPatch))
    assert n_pp >= 2


def test_univariate_annotate(df):
    ax = pp.boxplot(data=df, y="val", annotate=True)
    assert ax._publiplots_box_meta is not None
    assert len(ax._publiplots_box_meta.boxes) >= 1


def test_univariate_hides_synthetic_axis(df):
    ax = pp.boxplot(data=df, y="val")
    assert list(ax.get_xticks()) == []
    assert ax.spines["bottom"].get_visible() is False


def test_neither_x_nor_y_raises(df):
    with pytest.raises(ValueError, match="at least one of"):
        pp.boxplot(data=df)


def test_2d_still_works_smoke(df):
    ax = pp.boxplot(data=df, x="cat", y="val")
    n_pp = sum(1 for p in ax.patches if isinstance(p, PathPatch))
    assert n_pp == 3
