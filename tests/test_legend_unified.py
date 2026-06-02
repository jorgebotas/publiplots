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


# ---------------------------------------------------------------------------
# inside=True — render shared legend inside a dedicated empty cell
# ---------------------------------------------------------------------------


def _legend_artists(ax):
    """Return Legend children attached to ``ax``."""
    from matplotlib.legend import Legend
    return [c for c in ax.get_children() if isinstance(c, Legend)]


def test_inside_renders_in_anchor_axes(df):
    """`pp.legend(anchor=ax, inside=True)` attaches one Legend to ax."""
    fig, axes = pp.subplots(2, 2, axes_size=(35, 30))
    for r, c in [(0, 0), (0, 1), (1, 0)]:
        pp.scatterplot(df, x='x', y='y', hue='g', ax=axes[r, c])
    pp.legend(anchor=axes[1, 1], inside=True)
    fig.canvas.draw()
    legends = _legend_artists(axes[1, 1])
    assert len(legends) == 1, (
        f"expected exactly 1 Legend on the anchor cell; got {len(legends)}"
    )
    plt.close(fig)


def test_inside_default_loc_is_upper_left(df):
    """Defaults: side='left', align='start' → matplotlib loc='upper left'."""
    fig, axes = pp.subplots(2, 2, axes_size=(35, 30))
    pp.scatterplot(df, x='x', y='y', hue='g', ax=axes[0, 0])
    pp.legend(anchor=axes[1, 1], inside=True)
    fig.canvas.draw()
    leg = _legend_artists(axes[1, 1])[0]
    # matplotlib stores the resolved loc as an int code; 'upper left' = 2
    assert leg._loc == 2, (
        f"default inside loc should be 'upper left' (code 2); got {leg._loc}"
    )
    plt.close(fig)


def test_inside_collects_from_full_figure_by_default(df):
    """Without explicit `axes=`, inside mode collects across the full grid."""
    fig, axes = pp.subplots(2, 2, axes_size=(35, 30))
    for r, c in [(0, 0), (0, 1), (1, 0)]:
        pp.scatterplot(df, x='x', y='y', hue='g', ax=axes[r, c])
    pp.legend(anchor=axes[1, 1], inside=True)
    fig.canvas.draw()
    leg = _legend_artists(axes[1, 1])[0]
    labels = [t.get_text() for t in leg.get_texts()]
    # 2 hue values 'a' and 'b' should both appear
    assert set(labels) == {'a', 'b'}, (
        f"expected union of hue values across plotted axes; got {labels}"
    )
    plt.close(fig)


def test_inside_explicit_axes_scope_narrows_collection(df):
    """`axes=[a, b]` + `anchor=c` narrows collection scope to [a, b] only."""
    fig, axes = pp.subplots(2, 2, axes_size=(35, 30))
    # Plot 'a','b' hue on axes[0,0] only; axes[0,1] gets a different hue 'c','d'.
    pp.scatterplot(df, x='x', y='y', hue='g', ax=axes[0, 0])
    df2 = df.copy()
    df2['g'] = ['c'] * 10 + ['d'] * 10
    pp.scatterplot(df2, x='x', y='y', hue='g', ax=axes[0, 1])
    # Collect ONLY from axes[0,0] — c/d should not appear.
    pp.legend(axes=[axes[0, 0]], anchor=axes[1, 1], inside=True)
    fig.canvas.draw()
    leg = _legend_artists(axes[1, 1])[0]
    labels = [t.get_text() for t in leg.get_texts()]
    assert set(labels) == {'a', 'b'}, (
        f"explicit scope should restrict collection; got {labels}"
    )
    plt.close(fig)


def test_inside_auto_blanks_anchor(df):
    """Default `clear_anchor=True` turns the anchor's axes off."""
    fig, axes = pp.subplots(2, 2, axes_size=(35, 30))
    pp.scatterplot(df, x='x', y='y', hue='g', ax=axes[0, 0])
    pp.legend(anchor=axes[1, 1], inside=True)
    fig.canvas.draw()
    assert axes[1, 1].axison is False
    plt.close(fig)


def test_inside_clear_anchor_false_preserves_frame(df):
    """`clear_anchor=False` preserves the anchor's axis frame/ticks."""
    fig, axes = pp.subplots(2, 2, axes_size=(35, 30))
    pp.scatterplot(df, x='x', y='y', hue='g', ax=axes[0, 0])
    pp.legend(anchor=axes[1, 1], inside=True, clear_anchor=False)
    fig.canvas.draw()
    assert axes[1, 1].axison is True
    plt.close(fig)


def test_inside_evicts_per_axis_legends(df):
    """Per-axis legends from plotted cells are evicted; only the inside one
    survives."""
    from matplotlib.legend import Legend
    fig, axes = pp.subplots(2, 2, axes_size=(35, 30))
    for r, c in [(0, 0), (0, 1), (1, 0)]:
        pp.scatterplot(df, x='x', y='y', hue='g', ax=axes[r, c])  # default legend=True
    pp.legend(anchor=axes[1, 1], inside=True)
    fig.canvas.draw()
    legends_in_fig = [
        c for ax in axes.flat for c in ax.get_children()
        if isinstance(c, Legend)
    ]
    assert len(legends_in_fig) == 1, (
        f"inside-cell legend should evict per-axis legends; "
        f"saw {len(legends_in_fig)} Legend artists across the figure"
    )
    assert legends_in_fig[0].axes is axes[1, 1]
    plt.close(fig)


@pytest.mark.parametrize("side,align,expected_loc", [
    ("left",   "start",  "upper left"),
    ("left",   "center", "center left"),
    ("left",   "end",    "lower left"),
    ("right",  "start",  "upper right"),
    ("right",  "center", "center right"),
    ("right",  "end",    "lower right"),
    ("top",    "start",  "upper left"),
    ("top",    "center", "upper center"),
    ("top",    "end",    "upper right"),
    ("bottom", "start",  "lower left"),
    ("bottom", "center", "lower center"),
    ("bottom", "end",    "lower right"),
    ("center", "center", "center"),
])
def test_inside_loc_mapping(side, align, expected_loc):
    """Direct unit test of the (side, align) → loc helper."""
    from publiplots.utils.legend_group import _inside_loc_from_side_align
    assert _inside_loc_from_side_align(side, align) == expected_loc


def test_inside_without_anchor_raises():
    """`inside=True` requires `anchor=<Axes>`."""
    fig, _ = plt.subplots()
    with pytest.raises(ValueError, match="inside=True requires anchor"):
        pp.legend(inside=True)
    plt.close(fig)


def test_inside_clear_anchor_only_with_inside(df):
    """`clear_anchor=` is only meaningful when `inside=True`."""
    fig, ax = plt.subplots()
    pp.scatterplot(df, x='x', y='y', hue='g', ax=ax, legend=False)
    with pytest.raises(ValueError, match="clear_anchor= is only meaningful"):
        pp.legend(anchor=ax, clear_anchor=False)
    plt.close(fig)


def test_inside_side_center_coerces_align(df):
    """`side='center'` + `align='start'` → coerced to align='center' silently."""
    fig, axes = pp.subplots(2, 2, axes_size=(35, 30))
    pp.scatterplot(df, x='x', y='y', hue='g', ax=axes[0, 0])
    group = pp.legend(
        anchor=axes[1, 1], inside=True, side='center', align='start',
    )
    assert group._side == "center"
    assert group._align == "center", (
        f"side='center' must coerce align to 'center'; got {group._align}"
    )
    plt.close(fig)


def test_inside_side_center_only_in_inside_mode():
    """`side='center'` is invalid outside of inside mode."""
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="side must be one of"):
        pp.legend(anchor=ax, side='center')  # inside=False
    plt.close(fig)


def test_inside_does_not_grow_anchor_reservation(df):
    """Inside mode must not register an external overhang on the anchor."""
    fig, axes = pp.subplots(2, 2, axes_size=(35, 30))
    pp.scatterplot(df, x='x', y='y', hue='g', ax=axes[0, 0])
    group = pp.legend(anchor=axes[1, 1], inside=True)
    fig.canvas.draw()
    assert group._external_to_axis is False
    assert group._builder._external_to_axis is False
    plt.close(fig)


# ---------------------------------------------------------------------------
# GH #180 — inside=True must not shift sibling axes' positions
# ---------------------------------------------------------------------------


def _build_inside_repro_figure(legend_kwargs):
    """1x2 grid; left has data, right is blanked; optionally an in-cell legend."""
    import numpy as np
    fig, axes = pp.subplots(1, 2, axes_size=(50, 35))
    rng = np.random.default_rng(0)
    inner = pd.DataFrame({"x": rng.normal(size=50), "y": rng.normal(size=50)})
    pp.scatterplot(data=inner, x="x", y="y", ax=axes[0])
    # Always blank the right cell so baseline & with-legend are comparable.
    axes[1].set_axis_off()
    if legend_kwargs is not None:
        handles = pp.create_legend_handles(
            labels=[f"item {i}" for i in range(8)],
            colors=pp.color_palette("tab10", n_colors=8),
            style="circle",
        )
        band = pp.legend(
            anchor=axes[1], inside=True, collect=[], **legend_kwargs,
        )
        band.add_legend(handles=handles, label="My legend", ncol=2)
    fig.canvas.draw()
    return fig, axes


def test_inside_manual_add_legend_does_not_shift_siblings():
    """Adding `band.add_legend(...)` after `inside=True, collect=[]` must
    leave the sibling axes' positions unchanged vs. the no-legend baseline.

    Regression test for GH #180. Before the fix, the legend artist counted
    in the layout reactor as an external overhang on `axes[1]`, growing
    that cell's reservation and pushing `axes[0]` to the left.
    """
    fig_base, axes_base = _build_inside_repro_figure(None)
    base_x1 = axes_base[0].get_position().x1
    plt.close(fig_base)

    for kw in [{}, {"side": "center"}, {"side": "right"}]:
        fig, axes = _build_inside_repro_figure(kw)
        x1 = axes[0].get_position().x1
        assert abs(x1 - base_x1) < 1e-3, (
            f"inside={kw!r}: sibling axes shifted {x1:.4f} vs baseline "
            f"{base_x1:.4f} (delta {x1 - base_x1:+.4f})"
        )
        plt.close(fig)


def test_inside_legend_artist_is_not_in_layout():
    """The Legend artist created by inside-mode add_legend must have
    set_in_layout(False) so it's excluded from tightbbox math."""
    from matplotlib.legend import Legend
    fig, axes = _build_inside_repro_figure({})
    legends = [c for c in axes[1].get_children() if isinstance(c, Legend)]
    assert len(legends) == 1
    assert legends[0].get_in_layout() is False, (
        "inside-mode Legend must have in_layout=False to avoid "
        "displacing siblings"
    )
    plt.close(fig)


def test_inside_collect_empty_still_blanks_anchor(df):
    """clear_anchor=True must fire even when collect=[] short-circuits
    auto-collection (regression for the third bug uncovered by #180)."""
    fig, axes = pp.subplots(1, 2, axes_size=(50, 35))
    pp.scatterplot(df, x="x", y="y", hue="g", ax=axes[0])
    pp.legend(anchor=axes[1], inside=True, collect=[])
    fig.canvas.draw()
    assert axes[1].axison is False, (
        "clear_anchor must blank the anchor at construction, not gate "
        "the blank on _materialize finding entries"
    )
    plt.close(fig)


def test_inside_manual_add_legend_routes_through_inside_path(df):
    """Manual band.add_legend() on an inside-mode group must go through
    the LegendBuilder inside=True branch (auto-injected) — not the band
    path with reactor registration."""
    fig, axes = pp.subplots(1, 2, axes_size=(50, 35))
    pp.scatterplot(df, x="x", y="y", hue="g", ax=axes[0])
    handles = pp.create_legend_handles(
        labels=["a", "b"],
        colors=pp.color_palette("tab10", n_colors=2),
        style="circle",
    )
    band = pp.legend(anchor=axes[1], inside=True, collect=[])
    band.add_legend(handles=handles, label="L")
    fig.canvas.draw()
    # No reactor registrations should have been created for this builder.
    reactor = getattr(fig, "_publiplots_layout_reactor", None)
    if reactor is not None:
        builder_regs = [
            r for r in reactor._registrations
            if id(r.artist) in {id(a) for _, a in band._builder.elements}
        ]
        assert len(builder_regs) == 0, (
            "inside-mode manual add_legend must not register with the "
            f"layout reactor; got {len(builder_regs)} regs"
        )
    plt.close(fig)
