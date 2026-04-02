"""Tests for unified edgecolor support across plot types."""
import pytest
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

import publiplots as pp


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    n = 60
    return pd.DataFrame({
        "category": pd.Categorical(np.repeat(["A", "B", "C"], n // 3)),
        "group": pd.Categorical(np.tile(["X", "Y"], n // 2)),
        "value": np.random.randn(n),
    })


@pytest.fixture(autouse=True)
def close_figures():
    """Close all figures after each test."""
    yield
    plt.close("all")


class TestCreateLegendHandles:
    """Tests for create_legend_handles edgecolors parameter."""

    def test_edgecolors_none_defaults_to_facecolor(self):
        handles = pp.create_legend_handles(
            labels=["A", "B"],
            colors=["red", "blue"],
            edgecolors=None,
            alpha=0.1,
            linewidth=1.0,
        )
        for h, fc in zip(handles, ["red", "blue"]):
            # Check RGB components (ignore alpha channel)
            assert to_rgba(h.get_facecolor())[:3] == to_rgba(fc)[:3]
            assert to_rgba(h.get_edgecolor())[:3] == to_rgba(fc)[:3]

    def test_edgecolors_single_string_broadcasts(self):
        handles = pp.create_legend_handles(
            labels=["A", "B"],
            colors=["red", "blue"],
            edgecolors="black",
            alpha=0.1,
            linewidth=1.0,
        )
        for h in handles:
            assert to_rgba(h.get_edgecolor())[:3] == to_rgba("black")[:3]
        assert to_rgba(handles[0].get_facecolor())[:3] == to_rgba("red")[:3]
        assert to_rgba(handles[1].get_facecolor())[:3] == to_rgba("blue")[:3]

    def test_edgecolors_list_per_handle(self):
        handles = pp.create_legend_handles(
            labels=["A", "B"],
            colors=["red", "blue"],
            edgecolors=["black", "gray"],
            alpha=0.1,
            linewidth=1.0,
        )
        assert to_rgba(handles[0].get_edgecolor())[:3] == to_rgba("black")[:3]
        assert to_rgba(handles[1].get_edgecolor())[:3] == to_rgba("gray")[:3]

    def test_edgecolors_with_markers(self):
        handles = pp.create_legend_handles(
            labels=["A", "B"],
            colors=["red", "blue"],
            edgecolors="black",
            markers=["o", "^"],
            alpha=0.1,
            linewidth=1.0,
        )
        for h in handles:
            assert to_rgba(h.get_edgecolor())[:3] == to_rgba("black")[:3]

    def test_edgecolors_with_linemarkers(self):
        handles = pp.create_legend_handles(
            labels=["A", "B"],
            colors=["red", "blue"],
            edgecolors="black",
            markers=["o", "^"],
            linestyles=["-", "--"],
            alpha=0.1,
            linewidth=1.0,
        )
        for h in handles:
            assert to_rgba(h.get_edgecolor())[:3] == to_rgba("black")[:3]
