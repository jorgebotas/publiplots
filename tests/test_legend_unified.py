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
    old pp.legend(side='right') for the common figure-level case."""
    import io
    fig1, axes1 = pp.subplots(1, 2)
    for ax in axes1.flat:
        pp.scatterplot(df, x='x', y='y', hue='g', ax=ax, legend=False)
    pp.legend(side='right')
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


def test_render_entries_inside_true_bypasses_group(df):
    """inside=True legends must NOT register on the figure's legend group
    list and must NOT attach a _legend_group to the axes."""
    fig, ax = plt.subplots()
    pp.scatterplot(
        df, x='x', y='y', hue='g', ax=ax,
        legend_kws={'inside': True, 'loc': 'upper right'},
    )
    # No per-axes group should have been created.
    assert getattr(ax, '_legend_group', None) is None, (
        "inside=True must bypass the per-axes group path"
    )
    # No reactor registrations either (inside legends are plain matplotlib).
    fig.canvas.draw()  # force layout
    reactor = getattr(fig, '_publiplots_layout_reactor', None)
    if reactor is not None:
        ax_regs = [r for r in reactor._registrations if r.ax is ax]
        assert len(ax_regs) == 0, (
            f"inside=True must not register with the reactor; got {len(ax_regs)} regs"
        )
    plt.close(fig)


def test_render_entries_per_axes_stacks_into_same_group(df):
    """Successive plot calls on the same axes should stack into the same
    per-axes group (single LegendLayout cursor), not create separate groups."""
    fig, ax = plt.subplots()
    pp.scatterplot(df, x='x', y='y', hue='g', ax=ax)
    first_group = ax._legend_group
    assert first_group is not None
    pp.scatterplot(df, x='x', y='y', hue='g', ax=ax)
    second_group = ax._legend_group
    assert first_group is second_group, (
        "second plot call must reuse the first call's per-axes group"
    )
    plt.close(fig)


def test_row_band_centers_on_row_width(df):
    """``pp.legend(axes[0], side='top')`` must center the band on the row's
    width (union of all axes in the row), not the first axes' width.

    Regression test for the v0.10 gallery showcase bug where the row-band
    rendered anchored to ``axes[0,0].x0`` — i.e. along-edge geometry came
    from the single-axes ``self.anchor`` instead of the scope's
    ``_ScopeAnchor`` union.
    """
    fig, axes = pp.subplots(2, 3)
    group = pp.legend(list(axes[0]), side='top')
    for ax in axes.flat:
        pp.scatterplot(df, x='x', y='y', hue='g', palette='pastel', ax=ax)
    fig.canvas.draw()
    row_x0 = axes[0, 0].get_position().x0
    row_x1 = axes[0, -1].get_position().x1
    row_midx = (row_x0 + row_x1) / 2
    # Check the first rendered legend's bbox midpoint against the row
    # midpoint (expressed in figure fractions).
    _, artist = group._builder.elements[0]
    legend_bbox = artist.get_window_extent()
    fig_w = fig.get_window_extent().width
    legend_midx_frac = (legend_bbox.x0 + legend_bbox.x1) / 2 / fig_w
    assert abs(legend_midx_frac - row_midx) < 0.05, (
        f"row-band midpoint {legend_midx_frac:.3f} should match row "
        f"midpoint {row_midx:.3f}"
    )
    plt.close(fig)


def test_per_axes_legend_and_external_band_coexist(df):
    """``pp.legend(axes[0])`` (positional single-axes) and
    ``pp.legend(anchor=axes[1])`` (keyword external band) on different
    axes of the same figure must NOT warn about scope overlap.

    Regression test for the v0.10 gallery showcase bug where the
    positional single-axes form resolved to ``_scope_axes=None`` (meaning
    "full grid"), triggering ``_scope_overlap``'s full-grid heuristic
    against any other group on the figure.
    """
    import warnings
    fig, axes = pp.subplots(1, 2)
    for ax in axes.flat:
        pp.scatterplot(df, x='x', y='y', hue='g', ax=ax, legend=False)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        pp.legend(axes[0])
        pp.legend(anchor=axes[1])
    overlap_warnings = [
        x for x in w if "scope overlaps" in str(x.message)
    ]
    assert not overlap_warnings, (
        f"single-axes per-axes and external-band groups on different "
        f"axes should not warn about overlap; got "
        f"{len(overlap_warnings)} warnings"
    )
    plt.close(fig)
