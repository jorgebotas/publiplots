"""Contract tests for PR C — figsize removal across simple plot functions.

Each migrated plot function must:
  1. Raise TypeError if called with figsize=.
  2. Install SubplotsAutoLayout on its figure by default.
  3. Honor ax= passed from pp.subplots without creating a second figure.
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
    """Pivot-able long-format data for the dot-heatmap path."""
    rng = np.random.default_rng(0)
    rows, cols = ["r0", "r1", "r2"], ["c0", "c1", "c2", "c3"]
    data = []
    for r in rows:
        for c in cols:
            data.append({
                "row": r, "col": c,
                "val": rng.normal(),
                "sz": rng.uniform(1, 10),
            })
    return pd.DataFrame(data)


MIGRATED = [
    ("barplot",       "cat_df",     {"x": "cat", "y": "val"}),
    ("boxplot",       "cat_df",     {"x": "cat", "y": "val"}),
    ("violinplot",    "cat_df",     {"x": "cat", "y": "val"}),
    ("raincloudplot", "cat_df",     {"x": "cat", "y": "val"}),
    ("scatterplot",   "scatter_df", {"x": "x",   "y": "y"}),
    ("stripplot",     "cat_df",     {"x": "cat", "y": "val"}),
    ("swarmplot",     "cat_df",     {"x": "cat", "y": "val"}),
    ("pointplot",     "cat_df",     {"x": "cat", "y": "val"}),
    ("heatmap",       "matrix_df",  {}),
    ("heatmap",       "dot_df",     {"x": "col", "y": "row", "value": "val", "size": "sz"}),
]

_IDS = [
    "barplot", "boxplot", "violinplot", "raincloudplot", "scatterplot",
    "stripplot", "swarmplot", "pointplot", "heatmap-categorical", "heatmap-dot",
]


@pytest.mark.parametrize("fn_name,df_fixture,kwargs", MIGRATED, ids=_IDS)
def test_figsize_kwarg_is_rejected(fn_name, df_fixture, kwargs, request):
    df = request.getfixturevalue(df_fixture)
    fn = getattr(pp, fn_name)
    with pytest.raises(TypeError, match="figsize"):
        fn(data=df, **kwargs, figsize=(4, 3))


@pytest.mark.parametrize("fn_name,df_fixture,kwargs", MIGRATED, ids=_IDS)
def test_default_call_installs_auto_layout(fn_name, df_fixture, kwargs, request):
    df = request.getfixturevalue(df_fixture)
    fn = getattr(pp, fn_name)
    fig, _ = fn(data=df, **kwargs)
    assert hasattr(fig, "_publiplots_auto_layout"), \
        f"{fn_name}: figure has no SubplotsAutoLayout — did it take a non-pp.subplots path?"


@pytest.mark.parametrize("fn_name,df_fixture,kwargs", MIGRATED, ids=_IDS)
def test_ax_kwarg_reuses_existing_figure(fn_name, df_fixture, kwargs, request):
    df = request.getfixturevalue(df_fixture)
    fn = getattr(pp, fn_name)
    fig0, ax0 = pp.subplots(axes_size=(50, 30))
    fig1, ax1 = fn(data=df, **kwargs, ax=ax0)
    assert fig1 is fig0
    assert ax1 is ax0


def test_venn_rejects_figsize():
    """pp.venn no longer accepts figsize=."""
    with pytest.raises(TypeError, match="figsize"):
        pp.venn(sets=[{1, 2}, {2, 3}], figsize=(6, 6))


def test_venn_default_installs_auto_layout():
    fig, _ = pp.venn(sets=[{1, 2, 3}, {2, 3, 4}])
    assert hasattr(fig, "_publiplots_auto_layout")


def test_venn_ax_reuses_existing_figure():
    fig0, ax0 = pp.subplots(axes_size=(80, 80))
    fig1, ax1 = pp.venn(sets=[{1, 2, 3}, {2, 3, 4}], ax=ax0)
    assert fig1 is fig0
    assert ax1 is ax0


def test_complex_heatmap_rejects_figsize(matrix_df):
    """complex_heatmap no longer accepts figsize=; use axes_size= (mm)."""
    with pytest.raises(TypeError, match="figsize"):
        pp.complex_heatmap(matrix_df, figsize=(5, 5))


def test_complex_heatmap_accepts_axes_size(matrix_df):
    """complex_heatmap accepts axes_size=(w_mm, h_mm) and stores it in inches."""
    builder = pp.complex_heatmap(matrix_df, axes_size=(90, 60))
    # The builder stores figsize internally in inches
    assert builder._figsize == pytest.approx((90 / 25.4, 60 / 25.4), rel=1e-6)


def test_rcparam_figure_figsize_not_overridden_by_publiplots():
    """After PR C, publiplots must not override matplotlib's figure.figsize.

    `figure.figsize` remains a matplotlib-native rcParam (pp.rcParams proxies
    matplotlib.rcParams), but publiplots no longer sets it to a publication
    default — figure dimensions are computed by pp.subplots(axes_size=...)
    instead.
    """
    from publiplots.themes.rcparams import MATPLOTLIB_RCPARAMS
    assert "figure.figsize" not in MATPLOTLIB_RCPARAMS
