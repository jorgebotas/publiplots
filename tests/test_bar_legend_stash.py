"""Tests for barplot legend stashing via LegendEntry."""
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


def _bar_df(seed=0):
    rng = np.random.default_rng(seed)
    n = 60
    return pd.DataFrame({
        # NB: same cardinality for cond & treat so the hatch-only path
        # (hue="treat", hatch="cond") doesn't trip the unrelated
        # _apply_hatches_and_override_colors IndexError when axis_idx
        # indexes into palette (pre-existing bug outside Task 4 scope).
        "cond": rng.choice(["A", "B"], size=n),
        "treat": rng.choice(["ctrl", "trt"], size=n),
        "value": rng.normal(size=n),
    })


def _double_split_df(seed=0):
    rng = np.random.default_rng(seed)
    n = 60
    df = pd.DataFrame({
        "cond": rng.choice(["A", "B", "C"], size=n),
        "treat": rng.choice(["ctrl", "trt"], size=n),
        "value": rng.normal(size=n),
    })
    df["role"] = np.tile(["lead", "sub"], n // 2 + 1)[:n]
    return df


def test_bar_hue_only_stashes_one_hue_entry():
    """hatch == categorical_axis -> only a hue entry."""
    df = _bar_df()
    ax = pp.barplot(
        data=df, x="cond", y="value",
        hue="treat", hatch="cond",
        palette={"ctrl": "#ff0000", "trt": "#00ff00"},
    )
    names_kinds = [(e.name, e.kind) for e in get_entries(ax)]
    assert ("treat", "hue") in names_kinds
    assert not any(k == "hatch" for _, k in names_kinds)


def test_bar_hatch_only_stashes_one_hatch_entry():
    """hue == categorical_axis -> only a hatch entry."""
    df = _bar_df()
    ax = pp.barplot(
        data=df, x="cond", y="value",
        hue="cond", hatch="treat",
        palette={"A": "#ff0000", "B": "#00ff00"},
    )
    names_kinds = [(e.name, e.kind) for e in get_entries(ax)]
    assert ("treat", "hatch") in names_kinds
    assert not any(k == "hue" for _, k in names_kinds)


def test_bar_hatch_legend_is_gray_when_hue_equals_cat_axis():
    """hue == categorical_axis with a separate hatch column: the hatch
    legend swatches must be gray (the hatch represents a different
    dimension than the bar color, matching the double-split convention).
    """
    from matplotlib.colors import to_rgba
    df = _bar_df()
    ax = pp.barplot(
        data=df, x="cond", y="value",
        hue="cond", hatch="treat",
        palette={"A": "#ff0000", "B": "#00ff00"},
    )
    [entry] = [e for e in get_entries(ax) if e.kind == "hatch"]
    expected_rgb = to_rgba("gray")[:3]
    for handle in entry.handles:
        fc = handle.get_facecolor()
        # Alpha may be dimmed by the rcParams alpha; compare RGB only.
        assert fc[:3] == expected_rgb, (
            f"hatch swatch should be gray, got {fc}"
        )


def test_bar_hue_equals_cat_axis_no_hatch_stashes_nothing():
    """hue == categorical_axis without ``hatch``: nothing to stash (an
    empty hatch entry would render as a blank legend frame on the axes).
    Regression: previously stashed ``LegendEntry(name=None, kind='hatch',
    handles=[], labels=())``.
    """
    from matplotlib.legend import Legend
    df = _bar_df()
    ax = pp.barplot(data=df, x="cond", y="value", hue="cond",
                    palette="pastel")
    assert get_entries(ax) == []
    ax.get_figure().canvas.draw()
    assert [c for c in ax.get_children() if isinstance(c, Legend)] == []


def test_bar_combined_stashes_one_hue_entry():
    """hue == hatch -> one combined entry under kind=hue."""
    df = _bar_df()
    ax = pp.barplot(
        data=df, x="cond", y="value",
        hue="cond", hatch="cond",
        palette={"A": "#ff0000", "B": "#00ff00"},
    )
    entries = get_entries(ax)
    names_kinds = [(e.name, e.kind) for e in entries]
    assert ("cond", "hue") in names_kinds
    assert len(entries) == 1


def test_bar_double_split_stashes_hue_and_hatch():
    """hue != hatch and neither == categorical_axis -> two entries."""
    df = _double_split_df()
    fig, ax = pp.subplots(axes_size=(60, 40))
    pp.barplot(
        data=df, x="cond", y="value",
        hue="treat", hatch="role",
        palette={"ctrl": "#ff0000", "trt": "#00ff00"},
        ax=ax,
    )
    names_kinds = [(e.name, e.kind) for e in get_entries(ax)]
    assert ("treat", "hue") in names_kinds
    assert ("role", "hatch") in names_kinds


def test_bar_legend_false_stashes_nothing():
    df = _bar_df()
    ax = pp.barplot(
        data=df, x="cond", y="value",
        hue="treat", hatch="cond",
        palette={"ctrl": "#ff0000", "trt": "#00ff00"},
        legend=False,
    )
    assert get_entries(ax) == []


def test_bar_legend_dict_suppresses_hatch():
    """legend={'hatch': False} in a double-split scenario -> only hue stashed."""
    df = _double_split_df()
    fig, ax = pp.subplots(axes_size=(60, 40))
    pp.barplot(
        data=df, x="cond", y="value",
        hue="treat", hatch="role",
        palette={"ctrl": "#ff0000", "trt": "#00ff00"},
        legend={"hatch": False},
        ax=ax,
    )
    names_kinds = [(e.name, e.kind) for e in get_entries(ax)]
    assert ("treat", "hue") in names_kinds
    assert not any(k == "hatch" for _, k in names_kinds)


def test_bar_in_group_suppresses_per_axis_render():
    from matplotlib.legend import Legend
    df = _bar_df()
    fig, axes = pp.subplots(1, 2, axes_size=(50, 40))
    pp.legend_group(anchor=axes[-1])
    pp.barplot(
        data=df, x="cond", y="value",
        hue="treat", hatch="cond",
        palette={"ctrl": "#ff0000", "trt": "#00ff00"},
        ax=axes[0],
    )
    fig.canvas.draw()
    per_axis_legends = [c for c in axes[0].get_children() if isinstance(c, Legend)]
    assert per_axis_legends == []
