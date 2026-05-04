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
    ax = pp.heatmap(data=df)
    entries = get_entries(ax)
    assert len(entries) == 1
    assert entries[0].kind == "hue"
    assert is_continuous_hue(entries[0].handles)


def test_categorical_heatmap_legend_false_stashes_nothing():
    df = _matrix_df()
    ax = pp.heatmap(data=df, legend=False)
    assert get_entries(ax) == []


def _dot_df(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i, row in enumerate(["r0", "r1", "r2"]):
        for j, col in enumerate(["c0", "c1", "c2", "c3"]):
            rows.append({
                "row": row, "col": col,
                "value": rng.normal(),
                "size_var": rng.uniform(1, 10),
            })
    return pd.DataFrame(rows)


def test_dot_heatmap_stashes_hue_and_size_entries():
    """Dot heatmap (value_col + size_col) stashes one continuous-hue + one size entry."""
    from publiplots.utils.legend_entries import is_continuous_hue
    df = _dot_df()
    ax = pp.heatmap(
        data=df, x="col", y="row", value="value", size="size_var",
    )
    entries = get_entries(ax)
    kinds = [e.kind for e in entries]
    assert "hue" in kinds
    assert "size" in kinds
    hue_entry = next(e for e in entries if e.kind == "hue")
    assert is_continuous_hue(hue_entry.handles)


def test_dot_heatmap_legend_dict_suppresses_size():
    df = _dot_df()
    ax = pp.heatmap(
        data=df, x="col", y="row", value="value", size="size_var",
        legend={"size": False},
    )
    entries = get_entries(ax)
    kinds = [e.kind for e in entries]
    assert "hue" in kinds
    assert "size" not in kinds


def test_dot_heatmap_legend_false_stashes_nothing():
    df = _dot_df()
    ax = pp.heatmap(
        data=df, x="col", y="row", value="value", size="size_var",
        legend=False,
    )
    assert get_entries(ax) == []


def test_dot_heatmap_default_edges_match_facecolors_not_steel_blue():
    """Regression: dot heatmap's bubble edges must follow the face color
    (publiplots double-layer style), not matplotlib's default steel-blue
    line color. Previously ``ax.scatter(edgecolors='face')`` was resolved
    before the figure drew, so ``apply_transparency`` locked in C0 as
    the edge color for every bubble."""
    from matplotlib.colors import to_rgba
    df = _dot_df()
    ax = pp.heatmap(
        data=df, x="col", y="row", value="value", size="size_var",
        cmap="Reds",
    )
    coll = ax.collections[0]
    ecs = coll.get_edgecolors()
    fcs = coll.get_facecolors()
    assert len(ecs) == len(fcs) > 0
    steel_blue = to_rgba("C0")
    # Not all edges steel-blue
    assert not all(
        abs(e[0] - steel_blue[0]) < 0.01 and abs(e[1] - steel_blue[1]) < 0.01
        for e in ecs
    )
    # Each edge RGB matches its face RGB (alphas differ: face dimmed, edge opaque).
    for e, f in zip(ecs, fcs):
        assert abs(e[0] - f[0]) < 0.01
        assert abs(e[1] - f[1]) < 0.01
        assert abs(e[2] - f[2]) < 0.01


def test_dot_heatmap_uses_half_cell_margins():
    """Axis limits must be exactly half a cell past the data on each side,
    independent of cell count. Regression for scatter's default margins
    leaving huge empty space around a small categorical grid."""
    df = _dot_df()  # 4 cols, 3 rows
    ax = pp.heatmap(data=df, x="col", y="row", value="value", size="size_var")
    x0, x1 = ax.get_xlim()
    # 4 cols → positions 0..3 → expected xlim (-0.5, 3.5)
    assert abs(x0 - (-0.5)) < 1e-6, f"xlim[0]={x0}, want -0.5"
    assert abs(x1 - 3.5) < 1e-6, f"xlim[1]={x1}, want 3.5"
    y0, y1 = ax.get_ylim()
    # y is inverted by scatter's categorical handling → (2.5, -0.5)
    assert abs(y0 - 2.5) < 1e-6 and abs(y1 - (-0.5)) < 1e-6


def test_dot_heatmap_edgecolor_override_applies():
    """Explicit ``edgecolor=`` should override the facecolor-derived default."""
    from matplotlib.colors import to_rgba
    df = _dot_df()
    ax = pp.heatmap(
        data=df, x="col", y="row", value="value", size="size_var",
        cmap="Reds", edgecolor="black",
    )
    black = to_rgba("black")
    ecs = ax.collections[0].get_edgecolors()
    assert len(ecs) > 0
    for e in ecs:
        assert abs(e[0] - black[0]) < 0.01
        assert abs(e[1] - black[1]) < 0.01
        assert abs(e[2] - black[2]) < 0.01
