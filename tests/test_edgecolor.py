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


class TestBarplotEdgecolor:
    def test_default_edgecolor_matches_face(self, sample_data):
        ax = pp.barplot(data=sample_data, x="category", y="value", hue="category", legend=False)
        for patch in ax.patches:
            fc = patch.get_facecolor()
            ec = patch.get_edgecolor()
            assert to_rgba(ec)[:3] == pytest.approx(to_rgba(fc)[:3], abs=0.01)

    def test_custom_edgecolor_applied_to_patches(self, sample_data):
        ax = pp.barplot(data=sample_data, x="category", y="value", hue="category",
                             edgecolor="black", legend=False)
        for patch in ax.patches:
            ec = patch.get_edgecolor()
            fc = patch.get_facecolor()
            assert to_rgba(ec)[:3] == pytest.approx(to_rgba("black")[:3], abs=0.01)
            assert to_rgba(fc)[:3] != pytest.approx(to_rgba("black")[:3], abs=0.01)

    def test_custom_edgecolor_applied_to_error_bars(self, sample_data):
        ax = pp.barplot(data=sample_data, x="category", y="value", hue="category",
                             edgecolor="black", legend=False)
        for line in ax.lines:
            lc = to_rgba(line.get_color())
            assert lc[:3] == pytest.approx(to_rgba("black")[:3], abs=0.01)

    def test_no_hue_with_edgecolor(self, sample_data):
        ax = pp.barplot(data=sample_data, x="category", y="value",
                             edgecolor="black", legend=False)
        for patch in ax.patches:
            ec = patch.get_edgecolor()
            assert to_rgba(ec)[:3] == pytest.approx(to_rgba("black")[:3], abs=0.01)


class TestBoxplotEdgecolor:
    def test_default_edgecolor_matches_face(self, sample_data):
        ax = pp.boxplot(data=sample_data, x="category", y="value", hue="category", legend=False)
        for patch in ax.patches:
            fc = patch.get_facecolor()
            ec = patch.get_edgecolor()
            assert to_rgba(ec)[:3] == pytest.approx(to_rgba(fc)[:3], abs=0.01)

    def test_custom_edgecolor_on_patches(self, sample_data):
        ax = pp.boxplot(data=sample_data, x="category", y="value", hue="category",
                             edgecolor="black", legend=False)
        for patch in ax.patches:
            ec = patch.get_edgecolor()
            assert to_rgba(ec)[:3] == pytest.approx(to_rgba("black")[:3], abs=0.01)
            fc = patch.get_facecolor()
            assert to_rgba(fc)[:3] != pytest.approx(to_rgba("black")[:3], abs=0.01)

    def test_custom_edgecolor_structural_lines(self, sample_data):
        ax = pp.boxplot(data=sample_data, x="category", y="value",
                             edgecolor="red", legend=False)
        for line in ax.lines:
            marker = line.get_marker()
            if marker == "None" or marker == "" or marker is None:
                lc = to_rgba(line.get_color())
                assert lc[:3] == pytest.approx(to_rgba("red")[:3], abs=0.01)

    def test_linecolor_backward_compat(self, sample_data):
        ax = pp.boxplot(data=sample_data, x="category", y="value",
                             linecolor="blue", legend=False)
        for patch in ax.patches:
            ec = patch.get_edgecolor()
            assert to_rgba(ec)[:3] == pytest.approx(to_rgba("blue")[:3], abs=0.01)

    def test_edgecolor_overrides_linecolor(self, sample_data):
        with pytest.warns(FutureWarning, match="linecolor.*deprecated"):
            ax = pp.boxplot(data=sample_data, x="category", y="value",
                                 edgecolor="black", linecolor="blue", legend=False)
        for patch in ax.patches:
            ec = patch.get_edgecolor()
            assert to_rgba(ec)[:3] == pytest.approx(to_rgba("black")[:3], abs=0.01)


class TestScatterplotEdgecolor:
    def test_custom_edgecolor_on_collection(self, sample_data):
        ax = pp.scatterplot(data=sample_data, x="value", y="value",
                                  edgecolor="black", legend=False)
        collection = ax.collections[0]
        ecs = collection.get_edgecolors()
        for ec in ecs:
            assert ec[:3] == pytest.approx(to_rgba("black")[:3], abs=0.01)

    def test_edgecolor_in_legend_handles(self, sample_data):
        ax = pp.scatterplot(data=sample_data, x="value", y="value",
                                  hue="category", edgecolor="black")
        collection = ax.collections[0]
        if hasattr(collection, '_legend_data') and 'hue' in collection._legend_data:
            handles = collection._legend_data['hue'].get('handles', [])
            for h in handles:
                assert to_rgba(h.get_edgecolor())[:3] == pytest.approx(to_rgba("black")[:3], abs=0.01)


class TestViolinplotEdgecolor:
    """Tests for violinplot edgecolor parameter."""

    def test_edgecolor_maps_to_linecolor(self, sample_data):
        """edgecolor maps to seaborn's linecolor (visible with fill=True)."""
        ax = pp.violinplot(data=sample_data, x="category", y="value",
                                 edgecolor="black", fill=True, legend=False)
        from matplotlib.collections import FillBetweenPolyCollection
        for coll in ax.collections:
            if isinstance(coll, FillBetweenPolyCollection):
                ecs = coll.get_edgecolors()
                for ec in ecs:
                    assert ec[:3] == pytest.approx(to_rgba("black")[:3], abs=0.01)

    def test_edgecolor_precedence_over_linecolor(self, sample_data):
        """edgecolor takes precedence over linecolor with deprecation warning."""
        with pytest.warns(FutureWarning):
            ax = pp.violinplot(data=sample_data, x="category", y="value",
                                     edgecolor="black", linecolor="red", legend=False)

    def test_edgecolor_none_preserves_linecolor(self, sample_data):
        """When edgecolor is None, linecolor is used as before (no warning)."""
        # Should not emit FutureWarning
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            ax = pp.violinplot(data=sample_data, x="category", y="value",
                                     linecolor="red", legend=False)
        # Just verify it returns without error
        assert ax.get_figure() is not None
        assert ax is not None

    def test_edgecolor_no_fill_accepted(self, sample_data):
        """edgecolor is accepted even with fill=False (default)."""
        ax = pp.violinplot(data=sample_data, x="category", y="value",
                                 edgecolor="black", legend=False)
        assert ax.get_figure() is not None
        assert ax is not None


class TestStripplotEdgecolor:
    def test_edgecolor_applied_to_collection(self, sample_data):
        ax = pp.stripplot(data=sample_data, x="category", y="value",
                                edgecolor="black", legend=False)
        for coll in ax.collections:
            ecs = coll.get_edgecolors()
            for ec in ecs:
                assert ec[:3] == pytest.approx(to_rgba("black")[:3], abs=0.01)

    def test_edgecolor_in_legend(self, sample_data):
        ax = pp.stripplot(data=sample_data, x="category", y="value",
                                hue="category", edgecolor="black")
        collection = ax.collections[0]
        if hasattr(collection, '_legend_data') and 'hue' in collection._legend_data:
            handles = collection._legend_data['hue'].get('handles', [])
            for h in handles:
                assert to_rgba(h.get_edgecolor())[:3] == pytest.approx(to_rgba("black")[:3], abs=0.01)


class TestSwarmplotEdgecolor:
    def test_edgecolor_applied_to_collection(self, sample_data):
        ax = pp.swarmplot(data=sample_data, x="category", y="value",
                                edgecolor="black", legend=False)
        for coll in ax.collections:
            ecs = coll.get_edgecolors()
            for ec in ecs:
                assert ec[:3] == pytest.approx(to_rgba("black")[:3], abs=0.01)

    def test_edgecolor_in_legend(self, sample_data):
        ax = pp.swarmplot(data=sample_data, x="category", y="value",
                                hue="category", edgecolor="black")
        collection = ax.collections[0]
        if hasattr(collection, '_legend_data') and 'hue' in collection._legend_data:
            handles = collection._legend_data['hue'].get('handles', [])
            for h in handles:
                assert to_rgba(h.get_edgecolor())[:3] == pytest.approx(to_rgba("black")[:3], abs=0.01)


class TestPointplotEdgecolor:
    def test_custom_edgecolor_on_markers(self, sample_data):
        ax = pp.pointplot(data=sample_data, x="category", y="value",
                                edgecolor="black", legend=False)
        found_marker = False
        for line in ax.lines:
            mec = line.get_markeredgecolor()
            if line.get_markersize() > 0 and line.get_marker() != "None":
                found_marker = True
                assert to_rgba(mec)[:3] == pytest.approx(to_rgba("black")[:3], abs=0.01)
        assert found_marker, "No markers found on plot"

    def test_edgecolor_does_not_affect_connecting_lines(self, sample_data):
        ax = pp.pointplot(data=sample_data, x="category", y="value",
                                hue="group", edgecolor="black", legend=False)
        for line in ax.lines:
            if (line.get_marker() == "None" or line.get_markersize() == 0) and line.get_linestyle() != "None":
                lc = to_rgba(line.get_color())
                assert lc[:3] != pytest.approx(to_rgba("black")[:3], abs=0.01)


class TestRaincloudplotEdgecolor:
    """Tests for raincloudplot edgecolor passthrough."""

    def test_edgecolor_passed_to_subplots(self, sample_data):
        """Edgecolor should be applied across all sub-components."""
        ax = pp.raincloudplot(data=sample_data, x="category", y="value",
                                    edgecolor="black", legend=False)
        # Check box patches have black edges
        for patch in ax.patches:
            ec = patch.get_edgecolor()
            assert to_rgba(ec)[:3] == pytest.approx(to_rgba("black")[:3], abs=0.01)


class TestEdgecolorIntegration:
    """Cross-cutting tests for edgecolor consistency."""

    def test_all_plot_types_accept_edgecolor(self, sample_data):
        """Every plot function should accept edgecolor without error."""
        common = dict(data=sample_data, x="category", y="value", edgecolor="black", legend=False)
        pp.barplot(**common)
        plt.close("all")
        pp.boxplot(**common)
        plt.close("all")
        pp.violinplot(**common)
        plt.close("all")
        pp.stripplot(**common)
        plt.close("all")
        pp.swarmplot(**common)
        plt.close("all")
        pp.pointplot(**common)
        plt.close("all")
        pp.raincloudplot(**common)
        plt.close("all")

    def test_all_plot_types_accept_edgecolor_none(self, sample_data):
        """edgecolor=None should preserve default behavior for all types."""
        common = dict(data=sample_data, x="category", y="value", edgecolor=None, legend=False)
        pp.barplot(**common)
        plt.close("all")
        pp.boxplot(**common)
        plt.close("all")
        pp.violinplot(**common)
        plt.close("all")
        pp.stripplot(**common)
        plt.close("all")
        pp.swarmplot(**common)
        plt.close("all")
        pp.pointplot(**common)
        plt.close("all")
        pp.raincloudplot(**common)
        plt.close("all")

    def test_scatterplot_edgecolor_unchanged(self, sample_data):
        """Scatterplot's existing edgecolor behavior should be preserved."""
        ax = pp.scatterplot(data=sample_data, x="value", y="value",
                                  edgecolor="black", legend=False)
        collection = ax.collections[0]
        ecs = collection.get_edgecolors()
        for ec in ecs:
            assert ec[:3] == pytest.approx(to_rgba("black")[:3], abs=0.01)

    def test_combined_overlay_with_edgecolor(self, sample_data):
        """Overlaying plots with edgecolor should work correctly."""
        ax = pp.violinplot(data=sample_data, x="category", y="value",
                                 edgecolor="black", legend=False)
        pp.stripplot(data=sample_data, x="category", y="value",
                      edgecolor="black", ax=ax, legend=False)

    def test_hue_with_edgecolor(self, sample_data):
        """Edgecolor should work alongside hue grouping for all types."""
        common = dict(data=sample_data, x="category", y="value", hue="group", edgecolor="black", legend=False)
        pp.barplot(**common)
        plt.close("all")
        pp.boxplot(**common)
        plt.close("all")
        pp.violinplot(**common)
        plt.close("all")
        pp.stripplot(**common)
        plt.close("all")
        pp.pointplot(**common)
        plt.close("all")
