"""Tests for pp.violinplot 1D (univariate) mode."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import PolyCollection

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


def _polys(ax):
    return [c for c in ax.collections if isinstance(c, PolyCollection)]


def test_univariate_y_only(df):
    ax = pp.violinplot(data=df, y="val")
    assert len(_polys(ax)) >= 1


def test_univariate_x_only(df):
    ax = pp.violinplot(data=df, x="val")
    assert len(_polys(ax)) >= 1


def test_univariate_y_with_hue(df):
    ax = pp.violinplot(data=df, y="val", hue="g")
    assert len(_polys(ax)) >= 2


def test_univariate_annotate(df):
    ax = pp.violinplot(data=df, y="val", annotate=True)
    assert ax._publiplots_box_meta is not None


def test_univariate_hides_synthetic_axis(df):
    ax = pp.violinplot(data=df, y="val")
    assert list(ax.get_xticks()) == []
    assert ax.spines["bottom"].get_visible() is True


def test_univariate_side_clip(df):
    ax = pp.violinplot(data=df, y="val", side="left")
    assert len(_polys(ax)) >= 1


def test_neither_x_nor_y_raises(df):
    with pytest.raises(ValueError, match="at least one of"):
        pp.violinplot(data=df)


def test_2d_still_works_smoke(df):
    ax = pp.violinplot(data=df, x="cat", y="val")
    assert len(_polys(ax)) == 3
