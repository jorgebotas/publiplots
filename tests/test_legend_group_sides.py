"""4-sided × 2-anchor test matrix for ``pp.legend_group``.

Verifies that the right ``FigureLayout`` reservation field grows for each
combination of ``side=`` and anchor mode (figure-anchored via no
``anchor=``, axes-anchored via an explicit axes). Also guards the
back-compat call signature.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import publiplots as pp
from publiplots.utils.legend import create_legend_handles
from publiplots.utils.legend_entries import LegendEntry, stash_entry


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _sample_handles():
    return create_legend_handles(
        labels=["A", "B", "C"],
        colors=list(pp.color_palette("pastel", 3)),
        alpha=0.2, linewidth=1.0,
    )


# --- figure-anchored ---------------------------------------------------------


@pytest.mark.parametrize("side,field", [
    ("right", "legend_column"),
    ("bottom", "legend_band_bottom"),
    ("left", "legend_band_left"),
    ("top", "legend_band_top"),
])
def test_figure_anchored_reserves_correct_field(side, field):
    fig, axes = pp.subplots(2, 2, axes_size=(35, 25))
    for ax in np.atleast_2d(axes).flat:
        ax.plot([0, 1, 2], [0, 1, 0])
    group = pp.legend(side=side)
    group.add_legend(handles=_sample_handles(), label="group")
    fig.canvas.draw()
    layout = fig._publiplots_layout
    assert getattr(layout, field) > 5.0, (
        f"side={side!r}: {field} should absorb the legend; "
        f"got {getattr(layout, field):.1f}"
    )
    for other_field in (
        "legend_column", "legend_band_bottom", "legend_band_left", "legend_band_top"
    ):
        if other_field == field:
            continue
        assert getattr(layout, other_field) < 0.5, (
            f"side={side!r}: {other_field} should stay zero; "
            f"got {getattr(layout, other_field):.1f}"
        )


# --- axes-anchored -----------------------------------------------------------


def test_axes_anchored_right_pins_to_column():
    fig, axes = pp.subplots(2, 2, axes_size=(35, 25))
    for ax in np.atleast_2d(axes).flat:
        ax.plot([0, 1, 2], [0, 1, 0])
    # Anchor to axes[0, 0] — column 0. The legend should push column 0
    # wider, not figure-level legend_column.
    group = pp.legend(anchor=axes[0, 0], side="right")
    group.add_legend(handles=_sample_handles(), label="group")
    fig.canvas.draw()
    layout = fig._publiplots_layout
    assert layout.right[0] > 10.0, (
        f"right[0] should absorb the legend (axes-anchored); "
        f"got {layout.right[0]:.1f}"
    )
    assert layout.right[1] < 5.0, f"right[1] baseline, got {layout.right[1]:.1f}"
    assert layout.legend_column < 0.5, (
        f"legend_column should stay zero for axes-anchored; "
        f"got {layout.legend_column:.1f}"
    )


def test_axes_anchored_bottom_pins_to_row():
    fig, axes = pp.subplots(2, 2, axes_size=(35, 25))
    for ax in np.atleast_2d(axes).flat:
        ax.plot([0, 1, 2], [0, 1, 0])
    # anchor=axes[0, 0] is in row 0. Bottom-anchored pushes xlabel_space[0]
    # wider (the row's bottom reservation), not legend_band_bottom.
    group = pp.legend(anchor=axes[0, 0], side="bottom")
    group.add_legend(handles=_sample_handles(), label="group")
    fig.canvas.draw()
    layout = fig._publiplots_layout
    assert layout.xlabel_space[0] > 10.0, (
        f"xlabel_space[0] should absorb legend; got {layout.xlabel_space[0]:.1f}"
    )
    assert layout.legend_band_bottom < 0.5, (
        f"legend_band_bottom should stay zero; got {layout.legend_band_bottom:.1f}"
    )


# --- validation / back-compat ------------------------------------------------


def test_side_invalid_raises():
    fig, _ = pp.subplots(1, 1, axes_size=(40, 30))
    with pytest.raises(ValueError, match="side must be"):
        pp.legend(side="diagonal")


def test_back_compat_anchor_axes_defaults_to_side_right():
    """The pre-0.9 signature pp.legend(anchor=axes[-1]) must keep
    working as axes-anchored side='right'."""
    fig, axes = pp.subplots(1, 3, axes_size=(40, 30))
    for ax in axes:
        ax.plot([0, 1, 2], [0, 1, 0])
    group = pp.legend(anchor=axes[-1])
    assert group._side == "right"
    assert group._anchor_kind == "axes"
    group.add_legend(handles=_sample_handles(), label="group")
    fig.canvas.draw()
    layout = fig._publiplots_layout
    # Axes-anchored right on the last column → right[-1] grows.
    assert layout.right[-1] > 10.0
    assert layout.legend_column < 0.5


def test_figure_anchored_auto_collect_still_works():
    """pp.legend() without anchor auto-collects stashed entries."""
    fig, axes = pp.subplots(1, 2, axes_size=(40, 30))
    stash_entry(
        axes[0],
        LegendEntry.build(
            "g", "hue",
            handles=_sample_handles(),
            labels=("A", "B", "C"),
        ),
    )
    group = pp.legend()
    fig.canvas.draw()
    assert group._materialized
    assert len(group._builder.elements) == 1
    assert group._builder.elements[0][1].get_title().get_text() == "g"


# --- spacing / overlap guards -----------------------------------------------


@pytest.mark.parametrize("side,expected_gap_mm", [
    ("right", 2.0),
    ("left", 2.0),
    ("bottom", 2.0),
    ("top", 2.0),
])
def test_figure_anchored_default_outward_gap_per_side(side, expected_gap_mm):
    """The outward gap between the decorated-grid edge and the legend's
    near side defaults to 2 mm on every side. The per-side mapping
    exists so defaults can be tuned independently later."""
    fig, axes = pp.subplots(2, 2, axes_size=(35, 25))
    for ax in np.atleast_2d(axes).flat:
        ax.plot([0, 1, 2], [0, 1, 0])
    group = pp.legend(side=side)
    group.add_legend(handles=_sample_handles(), label="group")
    fig.canvas.draw()

    anc = group.anchor.get_window_extent()
    legend = group._builder.elements[0][1]
    lb = legend.get_window_extent()
    dpi = fig.dpi
    if side == "right":
        gap_mm = (lb.x0 - anc.x1) / dpi * 25.4
    elif side == "left":
        gap_mm = (anc.x0 - lb.x1) / dpi * 25.4
    elif side == "bottom":
        gap_mm = (anc.y0 - lb.y1) / dpi * 25.4
    else:  # "top"
        gap_mm = (lb.y0 - anc.y1) / dpi * 25.4
    assert abs(gap_mm - expected_gap_mm) < 0.5, (
        f"side={side}: expected ~{expected_gap_mm:.1f} mm gap, got {gap_mm:.2f}"
    )


def test_figure_anchored_grid_bbox_excludes_xlabel_space():
    """The _GridAnchor's bottom edge sits at the bottom of the xlabel_space
    zone (not the raw axes rectangle), so a bottom-anchored legend never
    overlaps with x-tick labels.
    """
    from matplotlib.legend import Legend
    fig, axes = pp.subplots(2, 2, axes_size=(35, 25))
    for ax in np.atleast_2d(axes).flat:
        ax.plot([0, 1, 2], [0, 1, 0])
    group = pp.legend(side="bottom")
    group.add_legend(handles=_sample_handles(), label="group")
    fig.canvas.draw()

    # Measure each axes' tight bbox WITHOUT the group's Legend child —
    # that legend is a visual sibling of the tick labels, not part of
    # the panel's own decoration footprint.
    anc_y0 = group.anchor.get_window_extent().y0
    group_legend_id = id(group._builder.elements[0][1])
    for ax in np.atleast_2d(axes).flat:
        toggled = []
        for child in ax.get_children():
            if isinstance(child, Legend) and id(child) == group_legend_id:
                child.set_in_layout(False)
                toggled.append(child)
        try:
            tb = ax.get_tightbbox()
        finally:
            for c in toggled:
                c.set_in_layout(True)
        if tb is None:
            continue
        assert tb.y0 >= anc_y0 - 0.5, (
            f"anchor y0={anc_y0:.2f} px but ax tight y0={tb.y0:.2f} px: "
            "anchor must sit below tick labels."
        )


# --- orientation + align ---------------------------------------------------


@pytest.mark.parametrize("side,expected_orientation,expected_align", [
    ("right", "vertical", "start"),
    ("left", "vertical", "start"),
    ("bottom", "horizontal", "center"),
    ("top", "horizontal", "center"),
])
def test_default_orientation_and_align_per_side(side, expected_orientation, expected_align):
    fig, axes = pp.subplots(2, 2, axes_size=(35, 25))
    for ax in np.atleast_2d(axes).flat:
        ax.plot([0, 1, 2], [0, 1, 0])
    group = pp.legend(side=side)
    assert group._orientation == expected_orientation
    assert group._align == expected_align


def test_bottom_horizontal_legend_is_wider_than_tall():
    """side='bottom' defaults to horizontal layout: the legend's pixel
    width should comfortably exceed its height (>= 2x)."""
    fig, axes = pp.subplots(2, 2, axes_size=(35, 25))
    for ax in np.atleast_2d(axes).flat:
        ax.plot([0, 1, 2], [0, 1, 0])
    group = pp.legend(side="bottom")
    group.add_legend(handles=_sample_handles(), label="group")
    fig.canvas.draw()
    bb = group._builder.elements[0][1].get_window_extent()
    assert (bb.x1 - bb.x0) > 2 * (bb.y1 - bb.y0), (
        f"expected horizontal layout (w>>h); got w={bb.x1-bb.x0:.1f} "
        f"h={bb.y1-bb.y0:.1f}"
    )


def test_right_vertical_legend_is_taller_than_wide():
    """side='right' keeps the historical vertical layout."""
    fig, axes = pp.subplots(2, 2, axes_size=(35, 25))
    for ax in np.atleast_2d(axes).flat:
        ax.plot([0, 1, 2], [0, 1, 0])
    group = pp.legend(side="right")
    group.add_legend(handles=_sample_handles(), label="group")
    fig.canvas.draw()
    bb = group._builder.elements[0][1].get_window_extent()
    assert (bb.y1 - bb.y0) > (bb.x1 - bb.x0), (
        f"expected vertical layout (h>w); got w={bb.x1-bb.x0:.1f} "
        f"h={bb.y1-bb.y0:.1f}"
    )


def test_bottom_center_align_places_legend_at_grid_midpoint():
    fig, axes = pp.subplots(2, 2, axes_size=(35, 25))
    for ax in np.atleast_2d(axes).flat:
        ax.plot([0, 1, 2], [0, 1, 0])
    group = pp.legend(side="bottom")
    group.add_legend(handles=_sample_handles(), label="group")
    fig.canvas.draw()
    anc = group.anchor.get_window_extent()
    bb = group._builder.elements[0][1].get_window_extent()
    anc_mid = (anc.x0 + anc.x1) / 2
    leg_mid = (bb.x0 + bb.x1) / 2
    delta_mm = abs(anc_mid - leg_mid) / fig.dpi * 25.4
    assert delta_mm < 1.0, f"expected legend centered on anchor; got delta={delta_mm:.2f} mm"


def test_explicit_align_end_on_bottom():
    fig, axes = pp.subplots(2, 2, axes_size=(35, 25))
    for ax in np.atleast_2d(axes).flat:
        ax.plot([0, 1, 2], [0, 1, 0])
    group = pp.legend(side="bottom", align="end")
    group.add_legend(handles=_sample_handles(), label="group")
    fig.canvas.draw()
    anc = group.anchor.get_window_extent()
    bb = group._builder.elements[0][1].get_window_extent()
    # Legend right edge should be close to anchor right edge.
    right_gap_mm = (anc.x1 - bb.x1) / fig.dpi * 25.4
    # ... and far from the left edge.
    left_gap_mm = (bb.x0 - anc.x0) / fig.dpi * 25.4
    assert right_gap_mm < 10.0 and left_gap_mm > 30.0, (
        f"expected end-aligned; got right_gap={right_gap_mm:.1f}, "
        f"left_gap={left_gap_mm:.1f}"
    )


def test_explicit_orientation_vertical_on_bottom():
    """Explicit orientation='vertical' on a bottom-side group produces
    a stacked (tall) legend even though bottom defaults to horizontal."""
    fig, axes = pp.subplots(2, 2, axes_size=(35, 25))
    for ax in np.atleast_2d(axes).flat:
        ax.plot([0, 1, 2], [0, 1, 0])
    group = pp.legend(side="bottom", orientation="vertical")
    assert group._orientation == "vertical"
    group.add_legend(handles=_sample_handles(), label="group")
    fig.canvas.draw()
    bb = group._builder.elements[0][1].get_window_extent()
    # Vertical layout: height > width.
    assert (bb.y1 - bb.y0) > (bb.x1 - bb.x0), (
        f"explicit orientation=vertical should stack; got w={bb.x1-bb.x0:.1f} "
        f"h={bb.y1-bb.y0:.1f}"
    )


def test_user_ncol_override_wins_on_horizontal():
    """Passing ncol=1 via add_legend on a horizontal-orientation group
    still stacks vertically (user override wins over the orientation
    default)."""
    fig, axes = pp.subplots(2, 2, axes_size=(35, 25))
    for ax in np.atleast_2d(axes).flat:
        ax.plot([0, 1, 2], [0, 1, 0])
    group = pp.legend(side="bottom")
    group.add_legend(handles=_sample_handles(), label="group", ncol=1)
    fig.canvas.draw()
    bb = group._builder.elements[0][1].get_window_extent()
    assert (bb.y1 - bb.y0) > (bb.x1 - bb.x0), (
        f"ncol=1 should stack vertically; got w={bb.x1-bb.x0:.1f} "
        f"h={bb.y1-bb.y0:.1f}"
    )


def test_orientation_invalid_raises():
    with pytest.raises(ValueError, match="orientation must be"):
        pp.legend(orientation="diagonal")


def test_align_invalid_raises():
    with pytest.raises(ValueError, match="align must be"):
        pp.legend(align="middle")


@pytest.mark.parametrize("anchor_kind", ["figure", "axes"])
def test_right_side_legend_top_aligned_with_axes_top(anchor_kind):
    """Regression: axes-anchored groups used to leave ~vpad extra white
    space above the legend because the default vpad=5 landed the legend
    below axes.y1 by 5 mm, while figure-anchored groups reach the same
    visual position because the _GridAnchor sits higher. Both modes now
    put the legend top within ~1 mm of the anchor axes' top."""
    fig, axes = pp.subplots(1, 3, axes_size=(40, 30))
    for ax in axes:
        ax.plot([0, 1, 2], [0, 1, 0])
        ax.set_title("title")
    if anchor_kind == "figure":
        group = pp.legend(side="right")
    else:
        group = pp.legend(anchor=axes[-1], side="right")
    group.add_legend(handles=_sample_handles(), label="group")
    fig.canvas.draw()

    ax_top_px = axes[-1].get_window_extent().y1
    legend_top_px = group._builder.elements[0][1].get_window_extent().y1
    delta_mm = (ax_top_px - legend_top_px) / fig.dpi * 25.4
    assert abs(delta_mm) < 1.5, (
        f"{anchor_kind}-anchored: legend top should be within ~1 mm of "
        f"axes top; got delta={delta_mm:.2f} mm"
    )


def test_materialize_merges_entries_with_subset_labels_across_axes():
    """When each axes stashes a subset of a hue's levels,
    ``legend_group`` should render the UNION of labels (first-seen
    order), not just the first axes' subset. A single warning should
    fire announcing the merge."""
    import warnings as _w
    from publiplots.utils.legend_entries import LegendEntry, stash_entry
    fig, axes = pp.subplots(1, 3, axes_size=(40, 30))
    for ax in axes:
        ax.plot([0, 1, 2], [0, 1, 0])
    # Panel 0: only 'A', panel 1: 'A' + 'B', panel 2: 'B' + 'C'.
    # Union should be A, B, C.
    palette_full = list(pp.color_palette("pastel", 3))
    subsets = [
        (["A"], palette_full[:1]),
        (["A", "B"], palette_full[:2]),
        (["B", "C"], palette_full[1:3]),
    ]
    for ax, (labels, colors) in zip(axes, subsets):
        stash_entry(
            ax,
            LegendEntry.build(
                name="group", kind="hue",
                handles=pp.create_legend_handles(
                    labels=labels, colors=colors, alpha=0.2, linewidth=1.0,
                ),
                labels=labels,
            ),
        )
    group = pp.legend(anchor=axes[-1])
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        group._materialize()
    labels_rendered = [
        t.get_text() for t in group._builder.elements[0][1].get_texts()
    ]
    assert labels_rendered == ["A", "B", "C"], (
        f"expected union of labels; got {labels_rendered}"
    )
    merge_warnings = [w for w in caught if "merged across" in str(w.message)]
    assert len(merge_warnings) == 1, (
        f"expected one merge warning; got {[str(w.message) for w in caught]}"
    )


def test_per_axis_outside_right_legend_top_aligned_with_axes():
    """Regression: ``pp.scatterplot(..., title=...)`` with the default
    outside-right legend used to sit 5 mm below the axes top because
    LegendBuilder defaulted vpad=5 regardless of anchor kind. Now
    LegendBuilder resolves the default vpad from self._anchor_ax:
    a real Axes gets vpad=0 (flush with axes top), a _GridAnchor gets
    vpad=5. This regression covers the per-axis path (no legend_group)."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "x": rng.normal(size=40),
        "y": rng.normal(size=40),
        "g": np.tile(list("AB"), 20),
    })
    ax = pp.scatterplot(
        data=df, x="x", y="y", hue="g", palette="pastel",
        title="per-axis",
    )
    fig = ax.get_figure()
    fig.canvas.draw()
    assert ax.legend_ is not None
    ax_top_px = ax.get_window_extent().y1
    legend_top_px = ax.legend_.get_window_extent().y1
    delta_mm = (ax_top_px - legend_top_px) / fig.dpi * 25.4
    assert abs(delta_mm) < 1.5, (
        f"per-axis legend top should be within ~1 mm of axes top; "
        f"got delta={delta_mm:.2f} mm"
    )


# --- seamless "after-plot" attachment ---------------------------------------


def test_legend_group_attached_after_plots_evicts_per_axis_legends():
    """pp.legend_group attached AFTER plot calls should remove the
    per-axis legend artists that each plot drew, since it'll render a
    shared legend itself. No ghost legends on the per-axis panels."""
    from matplotlib.legend import Legend
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "x": rng.normal(size=90),
        "y": rng.normal(size=90),
        "g": np.tile(list("ABC"), 30),
    })
    fig, axes = pp.subplots(1, 3, axes_size=(35, 25))
    for ax in axes:
        pp.scatterplot(data=df, x="x", y="y", hue="g", palette="pastel", ax=ax)
    # Before attaching the group, every panel has its own 'g' legend.
    pre = [sum(1 for c in ax.get_children() if isinstance(c, Legend)) for ax in axes]
    assert pre == [1, 1, 1], f"expected one per-axis legend each, got {pre}"
    # Attach the group AFTER the plots — should evict claimed entries.
    group = pp.legend(side="right")
    fig.canvas.draw()
    # Only the anchor axes should still carry a Legend (the group's own
    # shared legend); the other panels should have no Legend left.
    counts = [sum(1 for c in ax.get_children() if isinstance(c, Legend)) for ax in axes]
    assert counts[:-1] == [0, 0], (
        f"non-anchor panels should have no legend, got {counts[:-1]}"
    )
    assert counts[-1] == 1, f"anchor panel should host the group legend, got {counts[-1]}"
    assert len(group._builder.elements) == 1


def test_legend_group_after_plots_preserves_unclaimed_inside_legends():
    """A scatter with hue + style + ``legend_kws={'inside': True}`` draws
    a per-axis legend with BOTH entries merged. When ``legend_group(...
    collect=['g'])`` attaches after the plots, only the 'g' entry is
    evicted — the style legend stays inside each panel."""
    from matplotlib.legend import Legend
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "x": rng.normal(size=180),
        "y": rng.normal(size=180),
        "g": np.tile(list("ABC"), 60),
        "r": np.tile(list("12"), 90),
    })
    fig, axes = pp.subplots(1, 3, axes_size=(35, 25))
    for ax in axes:
        pp.scatterplot(
            data=df, x="x", y="y", hue="g", style="r", palette="pastel", ax=ax,
            legend_kws={"inside": True, "loc": "upper right"},
        )
    group = pp.legend(side="right", collect=["g"])
    fig.canvas.draw()
    for ax in axes[:-1]:
        titles = [
            c.get_title().get_text()
            for c in ax.get_children() if isinstance(c, Legend)
        ]
        assert titles == ["r"], (
            f"non-anchor panel should keep only the 'r' inside legend; "
            f"got {titles}"
        )
    anchor_titles = sorted(
        c.get_title().get_text()
        for c in axes[-1].get_children() if isinstance(c, Legend)
    )
    # The anchor still carries its own 'r' inside legend plus the
    # group's shared 'g' legend.
    assert anchor_titles == ["g", "r"], anchor_titles


def test_legend_group_after_plots_shrinks_per_axis_reservation():
    """After eviction, SubplotsAutoLayout should shrink the per-column
    ``right`` reservation (which used to accommodate the per-axis legend)
    back to baseline — the freed space becomes the figure-level
    ``legend_column`` band instead."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "x": rng.normal(size=90),
        "y": rng.normal(size=90),
        "g": np.tile(list("ABC"), 30),
    })
    fig, axes = pp.subplots(1, 3, axes_size=(35, 25))
    for ax in axes:
        pp.scatterplot(data=df, x="x", y="y", hue="g", palette="pastel", ax=ax)
    # First draw with per-axis legends present → right[-1] inflated.
    fig.canvas.draw()
    # Now attach the group after the fact.
    pp.legend(side="right")
    fig.canvas.draw()
    layout = fig._publiplots_layout
    # All per-column right reservations should be near zero; the width
    # went into legend_column instead.
    for i, r in enumerate(layout.right):
        assert r < 5.0, f"right[{i}] should shrink after eviction, got {r:.1f}"
    assert layout.legend_column > 5.0, (
        f"legend_column should absorb the legend width; "
        f"got {layout.legend_column:.1f}"
    )


def test_legend_group_saves_to_pdf_without_renderer_error(tmp_path):
    """Regression: LegendBuilder._measure_object_dimensions used to call
    ``self.fig.canvas.get_renderer()`` which exists on FigureCanvasAgg
    but NOT on FigureCanvasPdf / Svg / Ps. The alignment callback fires
    during ``fig.savefig('x.pdf')`` and the measurement path used to
    crash with AttributeError. Closes #115."""
    df = pd.DataFrame({
        "x": [0, 1, 2] * 2,
        "y": [0, 1, 2, 0.5, 1.5, 2.5],
        "m": ["A"] * 3 + ["B"] * 3,
    })
    fig, axes = pp.subplots(nrows=1, ncols=2)
    for ax in axes:
        pp.lineplot(data=df, x="x", y="y", hue="m", ax=ax,
                    errorbar=None, legend=True)
    pp.legend(figure=fig, side="bottom")
    # Each of these triggers a _measure_object_dimensions call during
    # the align hook. Previously this crashed on non-AGG canvases.
    fig.savefig(tmp_path / "ok.pdf")
    fig.savefig(tmp_path / "ok.svg")
    fig.savefig(tmp_path / "ok.png")
    assert (tmp_path / "ok.pdf").exists()
    assert (tmp_path / "ok.svg").exists()
    assert (tmp_path / "ok.png").exists()


# --- per-axes adopt: pp.legend(ax) reuses the plot-created group ------------


def _scatter_df(n=40, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "x": rng.normal(size=n),
        "y": rng.normal(size=n),
        "g": np.tile(list("AB"), n // 2),
    })


def test_per_axis_legend_adopts_cached_group():
    """pp.legend(ax) after a plot adopts the cached per-axes group rather
    than building a second competing one. Exactly one group on the figure,
    and it IS the cached group on the axes."""
    from matplotlib.legend import Legend
    df = _scatter_df()
    fig, ax = pp.subplots(1, 1, axes_size=(40, 30))
    pp.scatterplot(data=df, x="x", y="y", hue="g", palette="pastel", ax=ax)
    cached = ax._legend_group
    g = pp.legend(ax, side="left")
    fig.canvas.draw()
    assert len(fig._publiplots_legend_groups) == 1
    assert g is cached
    assert g is ax._legend_group
    # Exactly one Legend artist rendered (no double render).
    n_legends = sum(1 for c in ax.get_children() if isinstance(c, Legend))
    assert n_legends == 1, f"expected one Legend, got {n_legends}"


def test_per_axis_legend_repeated_call_re_adopts():
    """Calling pp.legend(ax) twice (e.g. to change side=) must re-adopt the
    same cached group, not build a second competing one. Regression: the
    adopt guard used to key on the mutable ``_collect == []`` which the
    first adopt flips to None, so repeats fell through to construction —
    duplicating groups, re-emitting the overlap warning, and stacking
    align callbacks."""
    import warnings as _w
    from matplotlib.legend import Legend
    df = _scatter_df()
    fig, ax = pp.subplots(1, 1, axes_size=(40, 30))
    pp.scatterplot(data=df, x="x", y="y", hue="g", palette="pastel", ax=ax)
    cached = ax._legend_group
    g1 = pp.legend(ax, side="top")
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        g2 = pp.legend(ax, side="left")
        fig.canvas.draw()
    overlap = [w for w in caught if "scope overlaps" in str(w.message)]
    assert g1 is cached and g2 is cached
    assert len(fig._publiplots_legend_groups) == 1
    assert overlap == [], f"repeat adopt must not warn; got {len(overlap)}"
    assert g2._side == "left"
    n_legends = sum(1 for c in ax.get_children() if isinstance(c, Legend))
    assert n_legends == 1, f"expected one Legend after re-adopt, got {n_legends}"


def test_per_axis_legend_no_prior_plot_constructs_fresh():
    """pp.legend(ax) on an axes that never plotted through publiplots
    (no cached group) must still construct a fresh group."""
    fig, ax = pp.subplots(1, 1, axes_size=(40, 30))
    ax.plot([0, 1, 2], [0, 1, 0])
    assert getattr(ax, "_legend_group", None) is None
    g = pp.legend(ax, side="left")
    g.add_legend(handles=_sample_handles(), label="group")
    fig.canvas.draw()
    assert g is not None
    assert g._side == "left"
    assert len(fig._publiplots_legend_groups) == 1


@pytest.mark.parametrize("side", ["right", "left", "top", "bottom"])
def test_per_axis_internal_side_places_on_correct_edge(side):
    """An adopted per-axes legend lands on the correct edge of the axes."""
    df = _scatter_df()
    fig, ax = pp.subplots(1, 1, axes_size=(50, 40))
    pp.scatterplot(data=df, x="x", y="y", hue="g", palette="pastel", ax=ax)
    g = pp.legend(ax, side=side)
    fig.canvas.draw()
    legend = g._builder.elements[0][1]
    lb = legend.get_window_extent()
    axb = ax.get_window_extent()
    if side == "right":
        assert lb.x0 >= axb.x1 - 2, (lb.x0, axb.x1)
    elif side == "left":
        assert lb.x1 <= axb.x0 + 2, (lb.x1, axb.x0)
    elif side == "top":
        assert lb.y0 >= axb.y1 - 2, (lb.y0, axb.y1)
    else:  # bottom
        assert lb.y1 <= axb.y0 + 2, (lb.y1, axb.y0)


def test_per_axis_internal_top_title_is_outermost():
    """Issue B: for side='top', stacking from the axes outward must be
    AXES -> LEGEND -> TITLE (title OUTERMOST, above the legend), with no
    overlaps. The legend sits between the axes top and the title.
    """
    df = _scatter_df()
    fig, ax = pp.subplots(1, 1, axes_size=(50, 40))
    pp.scatterplot(data=df, x="x", y="y", hue="g", palette="pastel", ax=ax,
                   title="my title")
    g = pp.legend(ax, side="top")
    # Settle: the title lift converges over a couple of draws, so assert on
    # the STEADY state, not the transient first-draw geometry.
    for _ in range(3):
        fig.canvas.draw()
    legend = g._builder.elements[0][1]
    lb = legend.get_window_extent()
    axb = ax.get_window_extent()
    title_bb = ax.title.get_window_extent()
    # Legend sits above the axes top.
    assert lb.y0 >= axb.y1 - 1.0, (
        f"legend y0={lb.y0:.1f} should be above axes top y1={axb.y1:.1f}"
    )
    # Title sits ABOVE the legend (title outermost), no overlap at steady
    # state (no negative tolerance — the gap must be genuinely positive).
    assert title_bb.y0 >= lb.y1, (
        f"title y0={title_bb.y0:.1f} should be above legend top "
        f"y1={lb.y1:.1f} (title must be outermost)"
    )


def test_per_axis_internal_top_no_gap_without_title():
    """Issue B/C: without a title, a top legend sits ~2mm above the axes
    top — no fixed 7mm empty padding."""
    df = _scatter_df()
    fig, ax = pp.subplots(1, 1, axes_size=(50, 40))
    pp.scatterplot(data=df, x="x", y="y", hue="g", palette="pastel", ax=ax)
    g = pp.legend(ax, side="top")
    fig.canvas.draw()
    legend = g._builder.elements[0][1]
    lb = legend.get_window_extent()
    axb = ax.get_window_extent()
    gap_mm = (lb.y0 - axb.y1) / fig.dpi * 25.4
    assert -0.5 <= gap_mm <= 4.0, (
        f"top legend gap above axes should be ~2mm, got {gap_mm:.2f}mm"
    )


def test_per_axis_top_title_lift_preserves_title_styling():
    """Issue B regression: lifting the title above a side='top' legend must
    NOT reset the title's font/color/weight. The lift uses the title offset
    transform, not ax.set_title (which re-merges the default fontdict). Also
    a user-set title pad LARGER than the band lift must be preserved, and
    neither must drift across repeated draws."""
    df = _scatter_df()
    fig, ax = pp.subplots(1, 1, axes_size=(50, 40))
    pp.scatterplot(data=df, x="x", y="y", hue="g", palette="pastel", ax=ax)
    ax.set_title("styled", color="red", fontsize=20, fontweight="bold", pad=40)
    g = pp.legend(ax, side="top")
    pads = []
    for _ in range(4):
        fig.canvas.draw()
        pads.append(ax.titleOffsetTrans._t[1] * 72.0)
    # Styling survives every draw.
    assert ax.title.get_color() in ("red", (1.0, 0.0, 0.0, 1.0))
    assert ax.title.get_fontsize() == 20.0
    assert ax.title.get_fontweight() == "bold"
    # User pad (40pt) is honoured (>= 40) and converges (no drift).
    assert all(p >= 40.0 - 0.5 for p in pads), f"user pad lost: {pads}"
    assert abs(pads[-1] - pads[-2]) < 0.5, f"title pad drifts: {pads}"


def test_per_axis_internal_left_clears_yticklabels():
    """An adopted left legend sits left of (clears) the y-tick labels,
    with no large fixed gap (Issue B: dynamic offset)."""
    df = _scatter_df()
    fig, ax = pp.subplots(1, 1, axes_size=(50, 40))
    pp.scatterplot(data=df, x="x", y="y", hue="g", palette="pastel", ax=ax)
    g = pp.legend(ax, side="left")
    fig.canvas.draw()
    legend = g._builder.elements[0][1]
    lb = legend.get_window_extent()
    ytick_x0s = [
        t.get_window_extent().x0
        for t in ax.get_yticklabels() if t.get_text()
    ]
    if ytick_x0s:
        min_x0 = min(ytick_x0s)
        # Clears the y-ticklabels (legend right edge at/left of them).
        assert lb.x1 <= min_x0 + 1.0, (
            f"left legend x1={lb.x1:.1f} should clear y-ticklabel x0={min_x0:.1f}"
        )
        # ... but not by a large fixed gap (dynamic, ~2mm).
        gap_mm = (min_x0 - lb.x1) / fig.dpi * 25.4
        assert gap_mm <= 5.0, (
            f"left legend gap to y-ticklabels should be small/dynamic, "
            f"got {gap_mm:.2f}mm"
        )


def test_per_axis_top_title_space_reserves_legend_plus_title():
    """Issue C: the reserved title_space for a per-axes top legend should
    be ~ legend_height + gap + title_height — not extra vertical-stacking
    padding."""
    df = _scatter_df()
    fig, ax = pp.subplots(1, 1, axes_size=(50, 40))
    pp.scatterplot(data=df, x="x", y="y", hue="g", palette="pastel", ax=ax,
                   title="my title")
    g = pp.legend(ax, side="top")
    fig.canvas.draw()
    legend = g._builder.elements[0][1]
    lb = legend.get_window_extent()
    axb = ax.get_window_extent()
    title_bb = ax.title.get_window_extent()
    # Total decoration above the axes top, in mm.
    decoration_mm = (title_bb.y1 - axb.y1) / fig.dpi * 25.4
    legend_h_mm = lb.height / fig.dpi * 25.4
    title_h_mm = title_bb.height / fig.dpi * 25.4
    # Expected: legend + ~2mm gap + title + a small slack for inter-gaps.
    expected = legend_h_mm + title_h_mm + 2.0
    assert decoration_mm <= expected + 5.0, (
        f"reserved decoration {decoration_mm:.2f}mm exceeds "
        f"legend+title+gap ~= {expected:.2f}mm (excess padding)"
    )


def test_per_axis_right_still_top_aligned():
    """Regression: an adopted internal right legend keeps its top within
    ~1.5mm of the axes top (mirrors test_per_axis_outside_right_*)."""
    df = _scatter_df()
    fig, ax = pp.subplots(1, 1, axes_size=(50, 40))
    pp.scatterplot(data=df, x="x", y="y", hue="g", palette="pastel", ax=ax,
                   title="t")
    g = pp.legend(ax, side="right")
    fig.canvas.draw()
    ax_top_px = ax.get_window_extent().y1
    legend_top_px = g._builder.elements[0][1].get_window_extent().y1
    delta_mm = (ax_top_px - legend_top_px) / fig.dpi * 25.4
    assert abs(delta_mm) < 1.5, (
        f"right legend top should be within ~1.5mm of axes top; "
        f"got delta={delta_mm:.2f} mm"
    )


def test_no_overlap_warning_on_scatter_then_per_axis_legend():
    """scatter then pp.legend(ax) emits zero 'scope overlaps' warnings."""
    import warnings as _w
    df = _scatter_df()
    fig, ax = pp.subplots(1, 1, axes_size=(40, 30))
    pp.scatterplot(data=df, x="x", y="y", hue="g", palette="pastel", ax=ax)
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        pp.legend(ax, side="left")
        fig.canvas.draw()
    overlap = [w for w in caught if "scope overlaps" in str(w.message)]
    assert len(overlap) == 0, [str(w.message) for w in overlap]


def test_overlap_warning_still_fires_for_two_figure_groups():
    """Two figure-level collect=None groups on a grid genuinely compete →
    one 'scope overlaps' warning. And two overlapping collect=['g'] groups
    also warn."""
    import warnings as _w
    fig, axes = pp.subplots(2, 2, axes_size=(35, 25))
    for ax in np.atleast_2d(axes).flat:
        ax.plot([0, 1, 2], [0, 1, 0])
    pp.legend(side="right")
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        pp.legend(side="left")
    overlap = [w for w in caught if "scope overlaps" in str(w.message)]
    assert len(overlap) == 1, [str(w.message) for w in caught]

    # Two groups with the same explicit collect on the same scope.
    fig2, axes2 = pp.subplots(2, 2, axes_size=(35, 25))
    for ax in np.atleast_2d(axes2).flat:
        ax.plot([0, 1, 2], [0, 1, 0])
    pp.legend(figure=fig2, side="right", collect=["g"])
    with _w.catch_warnings(record=True) as caught2:
        _w.simplefilter("always")
        pp.legend(figure=fig2, side="left", collect=["g"])
    overlap2 = [w for w in caught2 if "scope overlaps" in str(w.message)]
    assert len(overlap2) == 1, [str(w.message) for w in caught2]


def test_legend_kws_side_places_per_axis_legend():
    """pp.scatterplot(..., legend_kws={'side': 'left'}) places the legend
    on the left in a single group with no overlap warning."""
    import warnings as _w
    df = _scatter_df()
    fig, ax = pp.subplots(1, 1, axes_size=(50, 40))
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        pp.scatterplot(data=df, x="x", y="y", hue="g", palette="pastel",
                       ax=ax, legend_kws={"side": "left"})
        fig.canvas.draw()
    overlap = [w for w in caught if "scope overlaps" in str(w.message)]
    assert len(overlap) == 0, [str(w.message) for w in overlap]
    assert len(fig._publiplots_legend_groups) == 1
    g = ax._legend_group
    assert g._side == "left"
    legend = g._builder.elements[0][1]
    lb = legend.get_window_extent()
    axb = ax.get_window_extent()
    assert lb.x1 <= axb.x0 + 2, (lb.x1, axb.x0)


def test_per_axis_external_band_side_reserves():
    """pp.legend(anchor=axes[0,0], side=...) (external-band form) grows the
    correct per-cell reservation while figure legend bands stay ~0."""
    fig, axes = pp.subplots(2, 2, axes_size=(35, 25))
    for ax in np.atleast_2d(axes).flat:
        ax.plot([0, 1, 2], [0, 1, 0])
        ax.set_title("t")
        ax.set_ylabel("yl")
    group = pp.legend(anchor=axes[0, 0], side="top")
    group.add_legend(handles=_sample_handles(), label="group")
    fig.canvas.draw()
    layout = fig._publiplots_layout
    assert layout.title_space[0] > 10.0, (
        f"title_space[0] should absorb the top band; "
        f"got {layout.title_space[0]:.1f}"
    )
    assert layout.legend_band_top < 0.5, (
        f"legend_band_top should stay zero; got {layout.legend_band_top:.1f}"
    )

    fig2, axes2 = pp.subplots(2, 2, axes_size=(35, 25))
    for ax in np.atleast_2d(axes2).flat:
        ax.plot([0, 1, 2], [0, 1, 0])
        ax.set_ylabel("yl")
    group2 = pp.legend(anchor=axes2[0, 0], side="left")
    group2.add_legend(handles=_sample_handles(), label="group")
    fig2.canvas.draw()
    layout2 = fig2._publiplots_layout
    assert layout2.ylabel_space[0] > 10.0, (
        f"ylabel_space[0] should absorb the left band; "
        f"got {layout2.ylabel_space[0]:.1f}"
    )
    assert layout2.legend_band_left < 0.5, (
        f"legend_band_left should stay zero; got {layout2.legend_band_left:.1f}"
    )


def test_top_align_does_not_force_per_panel_canvas_draws():
    """Issue A (perf): one fig.canvas.draw() on an N-panel grid of
    side='top' per-axes legends must NOT trigger O(panels) nested full
    canvas.draws via _measure_object_dimensions. The align pass should
    measure without forcing a fresh figure redraw per element.

    Structural (not wall-clock): count _measure_object_dimensions calls
    that hit a real fig.canvas.draw() during a single user draw and
    assert it stays small and constant w.r.t. panel count.
    """
    from publiplots.utils.legend import LegendBuilder

    df = _scatter_df(n=40)

    def count_draws_in_one_draw(n):
        orig_draw = LegendBuilder._fig_canvas_draw_for_measure
        calls = {"n": 0}

        def counting(self):
            calls["n"] += 1
            return orig_draw(self)

        LegendBuilder._fig_canvas_draw_for_measure = counting
        try:
            fig, axes = pp.subplots(n, n, axes_size=(30, 22))
            for ax in np.atleast_2d(axes).flat:
                pp.scatterplot(data=df, x="x", y="y", hue="g",
                               palette="pastel", ax=ax, title="t")
                pp.legend(ax, side="top")
            calls["n"] = 0  # reset; count only the explicit draw below
            fig.canvas.draw()
            return calls["n"]
        finally:
            LegendBuilder._fig_canvas_draw_for_measure = orig_draw

    n2 = count_draws_in_one_draw(2)  # 4 panels
    n3 = count_draws_in_one_draw(3)  # 9 panels
    # Must not scale with panel count and must be small (ideally 0).
    assert n2 <= 1, f"2x2 forced {n2} nested canvas.draws in align measure"
    assert n3 <= 1, f"3x3 forced {n3} nested canvas.draws in align measure"
    assert n3 <= n2, f"draw count scales with panels: 2x2={n2}, 3x3={n3}"


def test_multi_panel_no_duplicate_render_or_hooks():
    """3x3 grid, plot each panel, pp.legend(ax) per panel. Exactly one
    Legend artist per panel (no double render) and the reactor registers
    at most one align callback per axes (no duplicate hook)."""
    from matplotlib.legend import Legend
    df = _scatter_df(n=60)
    fig, axes = pp.subplots(3, 3, axes_size=(30, 22))
    flat = list(np.atleast_2d(axes).flat)
    for ax in flat:
        pp.scatterplot(data=df, x="x", y="y", hue="g", palette="pastel", ax=ax)
    groups = [pp.legend(ax, side="top") for ax in flat]
    fig.canvas.draw()
    # One group per axes; no duplicate groups on the figure.
    assert len(fig._publiplots_legend_groups) == 9
    total_legends = sum(
        1 for ax in flat
        for c in ax.get_children() if isinstance(c, Legend)
    )
    assert total_legends == 9, f"expected 9 Legend artists, got {total_legends}"
    # The reactor wraps _refresh_all once; each group contributes exactly
    # one align callback (side='top' default align='center' connects).
    reactor = groups[0]._builder._reactor
    callbacks = getattr(reactor, "_align_callbacks", [])
    assert len(callbacks) == len(set(id(c) for c in callbacks)), (
        "duplicate align callbacks registered"
    )
    assert len(callbacks) <= 9, f"expected <=9 align callbacks, got {len(callbacks)}"


# --- legend spacing rcParam defaults -----------------------------------------


def _rendered_legend_width_mm(handletextpad=None, borderpad=None):
    """Build a horizontal side='top' 3-category legend and return its
    rendered width in mm. Optional explicit kwargs override the rcParam."""
    fig, axes = pp.subplots(1, 3, axes_size=(40, 30))
    for ax in axes:
        ax.plot([0, 1, 2], [0, 1, 0])
    group = pp.legend(side="top")
    kw = {}
    if handletextpad is not None:
        kw["handletextpad"] = handletextpad
    if borderpad is not None:
        kw["borderpad"] = borderpad
    group.add_legend(handles=_sample_handles(), label="group", **kw)
    fig.canvas.draw()
    bb = group._builder.elements[0][1].get_window_extent()
    return (bb.x1 - bb.x0) / fig.dpi * 25.4


def test_legend_handletextpad_respects_rcparam():
    """A smaller legend.handletextpad rcParam renders a narrower
    horizontal legend (the render path must read the rcParam)."""
    import matplotlib as mpl
    with mpl.rc_context({"legend.handletextpad": 0.2}):
        narrow = _rendered_legend_width_mm()
    with mpl.rc_context({"legend.handletextpad": 2.0}):
        wide = _rendered_legend_width_mm()
    assert wide > narrow + 1.0, (
        f"larger handletextpad should widen legend; "
        f"narrow={narrow:.2f} mm wide={wide:.2f} mm"
    )


def test_legend_borderpad_respects_rcparam():
    """legend.borderpad rcParam changes the rendered legend extent."""
    import matplotlib as mpl
    with mpl.rc_context({"legend.borderpad": 0.2}):
        narrow = _rendered_legend_width_mm()
    with mpl.rc_context({"legend.borderpad": 2.0}):
        wide = _rendered_legend_width_mm()
    assert wide > narrow + 1.0, (
        f"larger borderpad should widen legend; "
        f"narrow={narrow:.2f} mm wide={wide:.2f} mm"
    )


def test_legend_per_call_kwarg_overrides_rcparam():
    """An explicit handletextpad passed to add_legend wins over the
    rcParam default."""
    import matplotlib as mpl
    with mpl.rc_context({"legend.handletextpad": 0.2}):
        rcparam_narrow = _rendered_legend_width_mm()
        explicit_wide = _rendered_legend_width_mm(handletextpad=2.0)
    assert explicit_wide > rcparam_narrow + 1.0, (
        f"explicit handletextpad=2 should override rcParam 0.2; "
        f"rcparam_narrow={rcparam_narrow:.2f} mm explicit_wide={explicit_wide:.2f} mm"
    )


def test_legend_columnspacing_still_rcparam_controllable():
    """Regression guard: legend.columnspacing was never hard-coded and
    must keep responding to the rcParam (multi-column horizontal legend)."""
    import matplotlib as mpl

    def _width(cs):
        with mpl.rc_context({"legend.columnspacing": cs}):
            fig, axes = pp.subplots(1, 3, axes_size=(40, 30))
            for ax in axes:
                ax.plot([0, 1, 2], [0, 1, 0])
            group = pp.legend(side="top")
            group.add_legend(handles=_sample_handles(), label="group")
            fig.canvas.draw()
            bb = group._builder.elements[0][1].get_window_extent()
            return (bb.x1 - bb.x0) / fig.dpi * 25.4

    narrow = _width(0.2)
    wide = _width(4.0)
    assert wide > narrow + 1.0, (
        f"larger columnspacing should widen multi-column legend; "
        f"narrow={narrow:.2f} mm wide={wide:.2f} mm"
    )


def test_legend_reservation_tracks_handletextpad_rcparam():
    """The layout reservation must grow with the legend.handletextpad
    rcParam, proving the width ESTIMATE and the RENDER read the same
    rcParam (otherwise reserved != drawn). A vertical side='right'
    legend reserves its outward extent (width) in ``legend_column``, so
    that field tracks the handle/text padding that drives width."""
    import matplotlib as mpl

    def _reserved(htp):
        with mpl.rc_context({"legend.handletextpad": htp}):
            fig, axes = pp.subplots(1, 3, axes_size=(40, 30))
            for ax in axes:
                ax.plot([0, 1, 2], [0, 1, 0])
            group = pp.legend(side="right")
            group.add_legend(handles=_sample_handles(), label="group")
            fig.canvas.draw()
            return fig._publiplots_layout.legend_column

    small = _reserved(0.2)
    big = _reserved(3.0)
    assert big > small + 0.5, (
        f"reservation should grow with handletextpad rcParam; "
        f"small={small:.2f} mm big={big:.2f} mm"
    )
