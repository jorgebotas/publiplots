"""Contract tests for PR D — axes-only returns from plot functions.

Simple plots return a single Axes.
Composites (upsetplot, complex_heatmap.build) return a dict[str, Axes].
pp.subplots is unchanged (returns (fig, ax)) and not tested here.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import pandas as pd
import numpy as np
import pytest

import publiplots as pp


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def scatter_df():
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "x": rng.normal(size=50),
        "y": rng.normal(size=50),
        "g": rng.choice(["A", "B"], size=50),
    })


@pytest.fixture(scope="module")
def cat_df():
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "cat": rng.choice(["A", "B", "C"], size=60),
        "val": rng.normal(size=60),
        "g": rng.choice(["ctrl", "trt"], size=60),
    })


@pytest.fixture(scope="module")
def matrix_df():
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.normal(size=(4, 5)))


@pytest.fixture(scope="module")
def dot_df():
    rng = np.random.default_rng(0)
    rows, cols = ["r0", "r1", "r2"], ["c0", "c1", "c2", "c3"]
    data = []
    for r in rows:
        for c in cols:
            data.append({"row": r, "col": c, "val": rng.normal(), "sz": rng.uniform(1, 10)})
    return pd.DataFrame(data)


@pytest.fixture(scope="module")
def line_df():
    rng = np.random.default_rng(0)
    n = 40
    return pd.DataFrame({
        "t": np.tile(np.linspace(0, 10, 20), 2),
        "y": rng.normal(size=n),
        "g": np.repeat(["A", "B"], 20),
    })


SIMPLE_PLOTS = [
    ("barplot",       "cat_df",     {"x": "cat", "y": "val"}),
    ("boxplot",       "cat_df",     {"x": "cat", "y": "val"}),
    ("violinplot",    "cat_df",     {"x": "cat", "y": "val"}),
    ("raincloudplot", "cat_df",     {"x": "cat", "y": "val"}),
    ("scatterplot",   "scatter_df", {"x": "x",   "y": "y"}),
    ("stripplot",     "cat_df",     {"x": "cat", "y": "val"}),
    ("swarmplot",     "cat_df",     {"x": "cat", "y": "val"}),
    ("pointplot",     "cat_df",     {"x": "cat", "y": "val"}),
    ("lineplot",      "line_df",    {"x": "t",   "y": "y"}),
    ("heatmap",       "matrix_df",  {}),
    ("heatmap",       "dot_df",     {"x": "col", "y": "row", "value": "val", "size": "sz"}),
]

_SIMPLE_IDS = [
    "barplot", "boxplot", "violinplot", "raincloudplot", "scatterplot",
    "stripplot", "swarmplot", "pointplot", "lineplot",
    "heatmap-categorical", "heatmap-dot",
]


@pytest.mark.parametrize("fn_name,df_fixture,kwargs", SIMPLE_PLOTS, ids=_SIMPLE_IDS)
def test_simple_plot_returns_axes(fn_name, df_fixture, kwargs, request):
    df = request.getfixturevalue(df_fixture)
    fn = getattr(pp, fn_name)
    result = fn(data=df, **kwargs)
    assert isinstance(result, Axes), \
        f"{fn_name} returned {type(result).__name__}; expected matplotlib.axes.Axes"


@pytest.mark.parametrize("fn_name,df_fixture,kwargs", SIMPLE_PLOTS, ids=_SIMPLE_IDS)
def test_simple_plot_figure_accessible(fn_name, df_fixture, kwargs, request):
    df = request.getfixturevalue(df_fixture)
    fn = getattr(pp, fn_name)
    ax = fn(data=df, **kwargs)
    assert ax.get_figure() is not None


@pytest.mark.parametrize("fn_name,df_fixture,kwargs", SIMPLE_PLOTS, ids=_SIMPLE_IDS)
def test_simple_plot_ax_kwarg_returns_same_ax(fn_name, df_fixture, kwargs, request):
    df = request.getfixturevalue(df_fixture)
    fn = getattr(pp, fn_name)
    fig, ax = pp.subplots(axes_size=(50, 30))
    result = fn(data=df, **kwargs, ax=ax)
    assert result is ax


@pytest.mark.parametrize("fn_name,df_fixture,kwargs", SIMPLE_PLOTS, ids=_SIMPLE_IDS)
def test_simple_plot_tuple_unpack_raises(fn_name, df_fixture, kwargs, request):
    """Regression guard: plot return is NOT a tuple."""
    df = request.getfixturevalue(df_fixture)
    fn = getattr(pp, fn_name)
    with pytest.raises(TypeError):
        fig, ax = fn(data=df, **kwargs)  # noqa: F841


# ---- Composite plots ----


def test_upsetplot_returns_dict():
    sets = {"A": {1, 2}, "B": {2, 3}, "C": {3, 4}}
    result = pp.upsetplot(sets)
    assert isinstance(result, dict)
    assert set(result.keys()) == {"intersections", "matrix", "sets"}
    for k, v in result.items():
        assert isinstance(v, Axes), f"upsetplot['{k}'] is {type(v).__name__}, expected Axes"


def test_upsetplot_figure_accessible():
    sets = {"A": {1, 2}, "B": {2, 3}, "C": {3, 4}}
    axes = pp.upsetplot(sets)
    assert axes["intersections"].get_figure() is not None


def test_upsetplot_tuple_unpack_raises():
    """dict return: 2-unpack raises ValueError (too many values)."""
    sets = {"A": {1, 2}, "B": {2, 3}, "C": {3, 4}}
    with pytest.raises(ValueError):
        fig, axes = pp.upsetplot(sets)  # noqa: F841


def test_complex_heatmap_build_returns_dict(matrix_df):
    axes = pp.complex_heatmap(matrix_df).build()
    assert isinstance(axes, dict)
    assert "main" in axes
    assert isinstance(axes["main"], Axes)


def test_complex_heatmap_build_figure_accessible(matrix_df):
    axes = pp.complex_heatmap(matrix_df).build()
    assert axes["main"].get_figure() is not None


def test_complex_heatmap_build_tuple_unpack_raises(matrix_df):
    """dict return: 2-unpack raises ValueError (too many values)."""
    with pytest.raises(ValueError):
        fig, axes = pp.complex_heatmap(matrix_df).build()  # noqa: F841


# ---- Dendrogram (separate public API) ----


def test_dendrogram_returns_axes():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(8, 5)))
    result = pp.dendrogram(data=df)
    assert isinstance(result, Axes)


def test_dendrogram_figure_accessible():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(8, 5)))
    ax = pp.dendrogram(data=df)
    assert ax.get_figure() is not None


def test_dendrogram_tuple_unpack_raises():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(8, 5)))
    with pytest.raises(TypeError):
        fig, ax = pp.dendrogram(data=df)  # noqa: F841
