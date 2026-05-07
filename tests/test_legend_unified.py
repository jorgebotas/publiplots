"""Tests for the unified pp.legend() factory (v0.10.0)."""
import matplotlib.pyplot as plt
import pandas as pd
import pytest

import publiplots as pp
from publiplots.utils.legend_group import MultiAxesLegendGroup


@pytest.fixture
def df():
    return pd.DataFrame({
        'x': list(range(10)) * 2,
        'y': list(range(20)),
        'g': ['a'] * 10 + ['b'] * 10,
    })


def test_legend_single_axes_returns_group(df):
    fig, ax = plt.subplots()
    pp.scatterplot(df, x='x', y='y', hue='g', ax=ax, legend=False)
    result = pp.legend(ax)
    assert isinstance(result, MultiAxesLegendGroup)
    plt.close(fig)


def test_legend_single_axes_external_to_axis_is_false(df):
    fig, ax = plt.subplots()
    pp.scatterplot(df, x='x', y='y', hue='g', ax=ax, legend=False)
    group = pp.legend(ax)
    assert group._external_to_axis is False, (
        "single-axes scope must live inside tightbbox (external_to_axis=False) "
        "so _side_extent counts it toward the per-cell reservation"
    )
    plt.close(fig)


def test_legend_figure_level_external_to_axis_is_true(df):
    fig, axes = pp.subplots(1, 2)
    for ax in axes.flat:
        pp.scatterplot(df, x='x', y='y', hue='g', ax=ax, legend=False)
    group = pp.legend(side='right')
    assert group._external_to_axis is True
    plt.close(fig)


def test_legend_multi_axes_scope_external_to_axis_is_true(df):
    fig, axes = pp.subplots(1, 3)
    for ax in axes.flat:
        pp.scatterplot(df, x='x', y='y', hue='g', ax=ax, legend=False)
    group = pp.legend(list(axes.flat), side='top')
    assert group._external_to_axis is True
    plt.close(fig)


def test_legend_matches_legend_group_for_side_right(df):
    """Unified pp.legend(side='right') should be byte-compatible with the
    old pp.legend_group(side='right') for the common figure-level case."""
    import io
    fig1, axes1 = pp.subplots(1, 2)
    for ax in axes1.flat:
        pp.scatterplot(df, x='x', y='y', hue='g', ax=ax, legend=False)
    pp.legend_group(side='right')
    buf1 = io.BytesIO()
    fig1.savefig(buf1, format='png')

    fig2, axes2 = pp.subplots(1, 2)
    for ax in axes2.flat:
        pp.scatterplot(df, x='x', y='y', hue='g', ax=ax, legend=False)
    pp.legend(side='right')
    buf2 = io.BytesIO()
    fig2.savefig(buf2, format='png')

    assert buf1.getvalue() == buf2.getvalue(), "byte-identical rendering expected"
    plt.close(fig1)
    plt.close(fig2)


def test_legend_collect_empty_skips_auto(df):
    """collect=[] disables auto-collection — used when the caller wants to
    add manual handles only (replaces old auto=False semantics)."""
    fig, ax = plt.subplots()
    pp.scatterplot(df, x='x', y='y', hue='g', ax=ax, legend=False)
    group = pp.legend(ax, collect=[])
    group._materialize()
    # With collect=[], no auto-collected entries should materialize.
    # Manual .add_legend() calls would still work — tested implicitly
    # via the existing test_legend_builder.py migration.
    assert len(group._builder.elements) == 0, (
        f"collect=[] must skip auto-collect; got {len(group._builder.elements)} elements"
    )
    plt.close(fig)
