"""Tests for publiplots.utils.rounding and pp.barplot(border_radius=).

The unit tests for :func:`normalize_border_radius` land in commit 2
alongside the helper; the ``pp.barplot`` integration tests (and the
rounded-patch type assertions) land in commit 3 when the kwarg is
wired in.
"""

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.patches import Rectangle

import publiplots as pp
from publiplots.utils.rounding import (
    _RoundedBarPatch,
    apply_border_radius,
    normalize_border_radius,
)


# ---------------------------------------------------------------------------
# normalize_border_radius — pure helper
# ---------------------------------------------------------------------------


def test_normalize_scalar():
    """Scalar int/float maps to symmetric (v, v)."""
    assert normalize_border_radius(1.5) == (1.5, 1.5)
    assert normalize_border_radius(2) == (2.0, 2.0)
    assert normalize_border_radius(0) == (0.0, 0.0)


def test_normalize_tuple():
    """2-tuple / list passes through as (top, bottom) cast to float."""
    assert normalize_border_radius((2, 0)) == (2.0, 0.0)
    assert normalize_border_radius([1.5, 0.5]) == (1.5, 0.5)
    assert normalize_border_radius((0, 0)) == (0.0, 0.0)


def test_normalize_none_is_flat():
    """None -> (0.0, 0.0) — matches the rcParam default."""
    assert normalize_border_radius(None) == (0.0, 0.0)


def test_normalize_invalid_raises():
    """String, 3-tuple, dict, bool — all TypeError."""
    with pytest.raises(TypeError):
        normalize_border_radius("rounded")
    with pytest.raises(TypeError):
        normalize_border_radius((1, 2, 3))
    with pytest.raises(TypeError):
        normalize_border_radius({"top": 1, "bottom": 0})
    with pytest.raises(TypeError):
        normalize_border_radius(True)


# ---------------------------------------------------------------------------
# pp.barplot integration — border_radius kwarg, rcParam, preservation
# ---------------------------------------------------------------------------


def _bar_patches(ax):
    """Return patches that look like bars (have ``get_height``)."""
    return [p for p in ax.patches if hasattr(p, "get_height")]


def test_barplot_no_radius_keeps_rectangles():
    """Default (no kwarg, default rcParam) leaves Rectangle bars intact."""
    df = pd.DataFrame({"x": ["a", "b"], "y": [1, 2]})
    fig, ax = plt.subplots()
    pp.barplot(data=df, x="x", y="y", ax=ax)
    patches = _bar_patches(ax)
    assert len(patches) == 2
    assert all(isinstance(p, Rectangle) for p in patches)
    assert not any(isinstance(p, _RoundedBarPatch) for p in patches)
    plt.close(fig)


def test_barplot_symmetric_radius_produces_rounded_patches():
    """border_radius=1.5 swaps every Rectangle for a _RoundedBarPatch."""
    df = pd.DataFrame({"x": ["a", "b"], "y": [1, 2]})
    fig, ax = plt.subplots()
    pp.barplot(data=df, x="x", y="y", ax=ax, border_radius=1.5)
    patches = _bar_patches(ax)
    rounded = [p for p in patches if isinstance(p, _RoundedBarPatch)]
    assert len(rounded) == 2, (
        f"expected 2 _RoundedBarPatches, got {len(rounded)} "
        f"(types: {[type(p).__name__ for p in patches]})"
    )
    # No leftover Rectangles (the exact-type check avoids matching the
    # _RoundedBarPatch itself, which is a Patch subclass but not Rectangle).
    assert not any(
        type(p) is Rectangle for p in patches
    ), f"Rectangle survivor: {[type(p).__name__ for p in patches]}"
    plt.close(fig)


def test_barplot_asymmetric_radius():
    """(top, bottom) = (1.5, 0) — top rounded, bottom flat, no errors."""
    df = pd.DataFrame({"x": ["a", "b"], "y": [1, 2]})
    fig, ax = plt.subplots()
    pp.barplot(data=df, x="x", y="y", ax=ax, border_radius=(1.5, 0))
    rounded = [p for p in _bar_patches(ax) if isinstance(p, _RoundedBarPatch)]
    assert len(rounded) == 2
    # Force a draw so get_path() is exercised — catches path code /
    # vert mismatches that would otherwise only fire at render time.
    fig.canvas.draw()
    plt.close(fig)


def test_rcparam_applies_globally():
    """Setting pp.rcParams['bar.border_radius'] affects bars without kwarg."""
    df = pd.DataFrame({"x": ["a", "b"], "y": [1, 2]})
    pp.rcParams["bar.border_radius"] = 2.0
    try:
        fig, ax = plt.subplots()
        pp.barplot(data=df, x="x", y="y", ax=ax)
        rounded = [
            p for p in _bar_patches(ax) if isinstance(p, _RoundedBarPatch)
        ]
        assert len(rounded) == 2
        plt.close(fig)
    finally:
        pp.rcParams["bar.border_radius"] = (0.0, 0.0)


def test_kwarg_overrides_rcparam_with_zero():
    """border_radius=0 forces flat bars even when rcParam is nonzero."""
    df = pd.DataFrame({"x": ["a", "b"], "y": [1, 2]})
    pp.rcParams["bar.border_radius"] = 2.0
    try:
        fig, ax = plt.subplots()
        pp.barplot(data=df, x="x", y="y", ax=ax, border_radius=0)
        rounded = [
            p for p in _bar_patches(ax) if isinstance(p, _RoundedBarPatch)
        ]
        # 0 normalizes to (0, 0) -> apply_border_radius is a no-op -> Rectangle.
        assert len(rounded) == 0
        plt.close(fig)
    finally:
        pp.rcParams["bar.border_radius"] = (0.0, 0.0)


def test_hatch_preserved_after_rounding():
    """Bars keep their hatch pattern after the Rectangle->rounded swap."""
    df = pd.DataFrame({"x": ["a", "b"], "y": [1, 2], "g": ["x", "y"]})
    fig, ax = plt.subplots()
    pp.barplot(
        data=df,
        x="x",
        y="y",
        hatch="g",
        hue="x",
        hatch_map={"x": "", "y": "///"},
        ax=ax,
        border_radius=1.5,
    )
    rounded = [p for p in _bar_patches(ax) if isinstance(p, _RoundedBarPatch)]
    hatches = [p.get_hatch() for p in rounded]
    assert "///" in hatches, f"hatch lost after rounding; got {hatches}"
    plt.close(fig)


def test_apply_border_radius_noop_on_zero():
    """radius_mm=(0, 0) leaves the Rectangle list untouched."""
    fig, ax = plt.subplots()
    rect = Rectangle((0, 0), 1, 1)
    ax.add_patch(rect)
    apply_border_radius([rect], (0.0, 0.0), ax)
    # Rectangle still present, unchanged.
    assert rect in ax.patches
    assert isinstance(ax.patches[0], Rectangle)
    plt.close(fig)


def test_apply_border_radius_swaps_rectangle():
    """Nonzero radius replaces the Rectangle with a _RoundedBarPatch."""
    fig, ax = plt.subplots()
    rect = Rectangle((0.5, 0.5), 1.0, 2.0, facecolor="red", edgecolor="blue")
    ax.add_patch(rect)
    apply_border_radius([rect], (1.5, 1.5), ax)
    assert rect not in ax.patches
    assert len(ax.patches) == 1
    new = ax.patches[0]
    assert isinstance(new, _RoundedBarPatch)
    assert new.get_x() == 0.5
    assert new.get_y() == 0.5
    assert new.get_width() == 1.0
    assert new.get_height() == 2.0
    plt.close(fig)


# ---------------------------------------------------------------------------
# pp.boxplot integration — border_radius kwarg, rcParam, invariants
# ---------------------------------------------------------------------------


def _box_pathpatches(ax):
    """Return IQR box PathPatches seaborn added to ax."""
    from matplotlib.patches import PathPatch
    return [p for p in ax.patches if isinstance(p, PathPatch)]


def test_boxplot_no_radius_keeps_pathpatches():
    """Default pp.boxplot leaves seaborn's PathPatches in place."""
    df = pd.DataFrame(
        {"g": ["a"] * 20 + ["b"] * 20, "y": list(range(40))}
    )
    fig, ax = plt.subplots()
    pp.boxplot(data=df, x="g", y="y", ax=ax)
    # No rounded patches when border_radius is unset (default (0, 0)).
    assert not any(isinstance(p, _RoundedBarPatch) for p in ax.patches)
    # Seaborn drew one PathPatch per group.
    assert len(_box_pathpatches(ax)) == 2
    plt.close(fig)


def test_boxplot_symmetric_radius_produces_rounded_patches():
    """border_radius=1.5 swaps IQR PathPatches for _RoundedBarPatches."""
    df = pd.DataFrame(
        {"g": ["a"] * 20 + ["b"] * 20, "y": list(range(40))}
    )
    fig, ax = plt.subplots()
    pp.boxplot(data=df, x="g", y="y", ax=ax, border_radius=1.5)
    rounded = [p for p in ax.patches if isinstance(p, _RoundedBarPatch)]
    assert len(rounded) == 2
    # Original PathPatches removed.
    assert len(_box_pathpatches(ax)) == 0
    plt.close(fig)


def test_boxplot_asymmetric_radius():
    """(top, bottom) = (1.5, 0) — top rounded, bottom flat, draws cleanly."""
    df = pd.DataFrame({"g": ["a"] * 20, "y": list(range(20))})
    fig, ax = plt.subplots()
    pp.boxplot(data=df, x="g", y="y", ax=ax, border_radius=(1.5, 0))
    # Force draw so get_path() runs — catches vert/code mismatches.
    fig.canvas.draw()
    rounded = [p for p in ax.patches if isinstance(p, _RoundedBarPatch)]
    assert len(rounded) == 1
    plt.close(fig)


def test_boxplot_rcparam_applies_globally():
    """pp.rcParams['box.border_radius'] affects boxes without kwarg."""
    df = pd.DataFrame({"g": ["a"] * 20, "y": list(range(20))})
    pp.rcParams["box.border_radius"] = 2.0
    try:
        fig, ax = plt.subplots()
        pp.boxplot(data=df, x="g", y="y", ax=ax)
        assert any(isinstance(p, _RoundedBarPatch) for p in ax.patches)
        plt.close(fig)
    finally:
        pp.rcParams["box.border_radius"] = (0.0, 0.0)


def test_boxplot_kwarg_overrides_rcparam_with_zero():
    """border_radius=0 forces flat boxes even when rcParam is nonzero."""
    df = pd.DataFrame({"g": ["a"] * 20, "y": list(range(20))})
    pp.rcParams["box.border_radius"] = 2.0
    try:
        fig, ax = plt.subplots()
        pp.boxplot(data=df, x="g", y="y", ax=ax, border_radius=0)
        rounded = [
            p for p in ax.patches if isinstance(p, _RoundedBarPatch)
        ]
        assert len(rounded) == 0
        plt.close(fig)
    finally:
        pp.rcParams["box.border_radius"] = (0.0, 0.0)


def test_boxplot_whiskers_untouched():
    """Whiskers/caps/medians are Line2Ds; rounding must not touch them."""
    df = pd.DataFrame({"g": ["a"] * 20, "y": list(range(20))})
    fig, ax = plt.subplots()
    pp.boxplot(data=df, x="g", y="y", ax=ax, border_radius=1.5)
    # Boxplot line family still present (whiskers + caps + median).
    assert len(ax.lines) >= 4
    plt.close(fig)


def test_boxplot_horizontal_orient_radius():
    """Horizontal orient boxplot draws and rounds without transform errors."""
    df = pd.DataFrame({"g": ["a"] * 20, "y": list(range(20))})
    fig, ax = plt.subplots()
    # Categorical on y, value on x → horizontal boxplot.
    pp.boxplot(data=df, x="y", y="g", ax=ax, border_radius=1.5)
    fig.canvas.draw()
    rounded = [p for p in ax.patches if isinstance(p, _RoundedBarPatch)]
    assert len(rounded) == 1
    plt.close(fig)


# ---------------------------------------------------------------------------
# Orientation- and sign-aware corner placement
# ---------------------------------------------------------------------------
#
# `border_radius=(top_mm, bottom_mm)` is interpreted as (free-end, base-end).
# `_RoundedBarPatch._corner_pts` carries per-corner radii in points, ordered
# (TL, TR, BR, BL). The mapping that `apply_border_radius` performs is:
#   vertical:   (TL, TR, BR, BL) = (top, top, bottom, bottom)
#   horizontal: (TL, TR, BR, BL) = (bottom, top, top, bottom)
# Sign-awareness then comes for free from matplotlib's signed width/height
# storage — the patch's bbox extents flip and the "TR" / "TL" labels
# correspond to the visually-free vs visually-base corners regardless.

# Index of each corner in `_corner_pts` for readability in assertions.
_TL, _TR, _BR, _BL = 0, 1, 2, 3


def _free_base_corners(orient: str, sign: float):
    """Return (free_idx_pair, base_idx_pair) into _corner_pts.

    The "free" pair is the two corners at the extreme end of the bar
    (visually opposite the baseline); the "base" pair sits on the
    baseline. Determined by orientation + sign of the bar value.
    """
    if orient == "v":
        # Vertical positive: free = top edge (TL, TR); base = bottom (BL, BR).
        # Vertical negative: matplotlib stores h<0 so y1 < y0; the patch's
        # "TR/TL" corners (at y1) end up visually below — still the free
        # end. Pair labels relative to _corner_pts don't change with sign.
        return (_TL, _TR), (_BL, _BR)
    else:  # "h"
        # Horizontal: free = right edge of bbox (TR, BR); base = left (TL, BL).
        return (_TR, _BR), (_TL, _BL)


def _rounded(ax):
    return [p for p in ax.patches if isinstance(p, _RoundedBarPatch)]


def test_apply_border_radius_orient_v_default():
    """orient='v' (default) maps (top, bottom) -> (TL=top, TR=top, BR=bot, BL=bot)."""
    fig, ax = plt.subplots()
    rect = Rectangle((0, 0), 1, 2)
    ax.add_patch(rect)
    apply_border_radius([rect], (1.5, 0.5), ax)  # default orient="v"
    new = ax.patches[0]
    assert isinstance(new, _RoundedBarPatch)
    tl, tr, br, bl = new._corner_pts
    assert tl == tr > 0  # top corners share top_mm
    assert br == bl > 0  # bottom corners share bottom_mm
    assert tl > br  # top_mm=1.5 > bottom_mm=0.5
    plt.close(fig)


def test_apply_border_radius_orient_h_rotates_90():
    """orient='h' maps (top, bottom) -> (TL=bot, TR=top, BR=top, BL=bot)."""
    fig, ax = plt.subplots()
    rect = Rectangle((0, 0), 2, 1)
    ax.add_patch(rect)
    apply_border_radius([rect], (1.5, 0.5), ax, orient="h")
    new = ax.patches[0]
    assert isinstance(new, _RoundedBarPatch)
    tl, tr, br, bl = new._corner_pts
    # Right-edge corners (TR, BR) carry top_mm (the free-end radius).
    assert tr == br > 0
    # Left-edge corners (TL, BL) carry bottom_mm (the base-end radius).
    assert tl == bl > 0
    assert tr > tl  # top_mm=1.5 > bottom_mm=0.5
    plt.close(fig)


def test_apply_border_radius_invalid_orient_raises():
    fig, ax = plt.subplots()
    rect = Rectangle((0, 0), 1, 1)
    ax.add_patch(rect)
    with pytest.raises(ValueError, match="orient"):
        apply_border_radius([rect], (1.0, 0.0), ax, orient="diagonal")
    plt.close(fig)


def test_barplot_horizontal_symmetric_radius():
    """Horizontal pp.barplot with symmetric radius rounds all 4 corners."""
    df = pd.DataFrame({"g": ["a", "b"], "v": [1.0, 2.0]})
    fig, ax = plt.subplots()
    # Categorical on y → horizontal bars.
    pp.barplot(data=df, x="v", y="g", ax=ax, border_radius=1.5)
    rounded = _rounded(ax)
    assert len(rounded) == 2
    for p in rounded:
        assert all(c > 0 for c in p._corner_pts), (
            f"symmetric radius should round all 4 corners, got "
            f"{p._corner_pts}"
        )
    plt.close(fig)


def test_barplot_horizontal_asymmetric_radius_rounds_free_end_only():
    """border_radius=(1.5, 0) on horizontal positive bars → right corners only."""
    df = pd.DataFrame({"g": ["a", "b"], "v": [1.0, 2.0]})
    fig, ax = plt.subplots()
    pp.barplot(data=df, x="v", y="g", ax=ax, border_radius=(1.5, 0))
    rounded = _rounded(ax)
    assert len(rounded) == 2
    free, base = _free_base_corners("h", sign=1.0)
    for p in rounded:
        cps = p._corner_pts
        for fi in free:
            assert cps[fi] > 0, (
                f"horizontal positive: free-end corner {fi} should be "
                f"rounded, got {cps}"
            )
        for bi in base:
            assert cps[bi] == 0, (
                f"horizontal positive: base-end corner {bi} should be "
                f"flat, got {cps}"
            )
    plt.close(fig)


def test_barplot_horizontal_negative_values_round_free_end():
    """Negative-valued horizontal bars: free end is the LEFT (x1 < x0).

    The patch's "TR/BR" labels in _corner_pts always point to the
    free-end (top_mm) per the orient='h' mapping. Sign awareness is
    delivered by matplotlib's signed width — width<0 makes the bbox's
    x1 numerically smaller than x0 and the rounding visually lands at
    the lower-x edge.
    """
    df = pd.DataFrame({"g": ["a", "b"], "v": [-1.0, -2.0]})
    fig, ax = plt.subplots()
    pp.barplot(data=df, x="v", y="g", ax=ax, border_radius=(1.5, 0))
    rounded = _rounded(ax)
    assert len(rounded) == 2
    free, base = _free_base_corners("h", sign=-1.0)
    for p in rounded:
        cps = p._corner_pts
        # Width should be negative (bar extends from 0 to negative x).
        assert p.get_width() < 0, (
            f"expected negative-width bar; got width={p.get_width()}"
        )
        for fi in free:
            assert cps[fi] > 0
        for bi in base:
            assert cps[bi] == 0
    plt.close(fig)


def test_barplot_vertical_negative_values_round_free_end():
    """Negative vertical bars: free end is at the visual bottom (y1 < y0)."""
    df = pd.DataFrame({"g": ["a", "b"], "v": [-1.0, -2.0]})
    fig, ax = plt.subplots()
    pp.barplot(data=df, x="g", y="v", ax=ax, border_radius=(1.5, 0))
    rounded = _rounded(ax)
    assert len(rounded) == 2
    free, base = _free_base_corners("v", sign=-1.0)
    for p in rounded:
        cps = p._corner_pts
        assert p.get_height() < 0, (
            f"expected negative-height bar; got height={p.get_height()}"
        )
        # Free corners (TL, TR) carry top_mm = 1.5; base (BL, BR) flat.
        for fi in free:
            assert cps[fi] > 0
        for bi in base:
            assert cps[bi] == 0
    plt.close(fig)


def test_boxplot_horizontal_asymmetric_radius_rounds_right_end():
    """Horizontal boxplot with (1.5, 0) rounds the right end only."""
    df = pd.DataFrame({"g": ["a"] * 20, "v": list(range(20))})
    fig, ax = plt.subplots()
    pp.boxplot(data=df, x="v", y="g", ax=ax, border_radius=(1.5, 0))
    fig.canvas.draw()
    rounded = _rounded(ax)
    assert len(rounded) == 1
    free, base = _free_base_corners("h", sign=1.0)
    cps = rounded[0]._corner_pts
    for fi in free:
        assert cps[fi] > 0, f"free-end corner {fi} flat: {cps}"
    for bi in base:
        assert cps[bi] == 0, f"base-end corner {bi} rounded: {cps}"
    plt.close(fig)


# ---------------------------------------------------------------------------
# pp.raincloudplot — auto-propagation via the internal pp.boxplot call
# ---------------------------------------------------------------------------


def test_raincloud_picks_up_box_border_radius_via_rcparam():
    """rcParam propagates through pp.raincloudplot's internal pp.boxplot."""
    df = pd.DataFrame({"g": ["a"] * 40, "y": list(range(40))})
    pp.rcParams["box.border_radius"] = 1.5
    try:
        fig, ax = plt.subplots()
        pp.raincloudplot(data=df, x="g", y="y", ax=ax)
        assert any(
            isinstance(p, _RoundedBarPatch) for p in ax.patches
        ), (
            "raincloud should auto-propagate box.border_radius via "
            "its internal pp.boxplot call"
        )
        plt.close(fig)
    finally:
        pp.rcParams["box.border_radius"] = (0.0, 0.0)


def test_raincloud_box_kws_override():
    """Per-call box_kws={'border_radius': ...} works via box_kws passthrough."""
    df = pd.DataFrame({"g": ["a"] * 40, "y": list(range(40))})
    fig, ax = plt.subplots()
    pp.raincloudplot(
        data=df,
        x="g",
        y="y",
        ax=ax,
        box_kws={"border_radius": 2.0},
    )
    assert any(isinstance(p, _RoundedBarPatch) for p in ax.patches)
    plt.close(fig)


# ---------------------------------------------------------------------------
# tracker-scoping invariants — pre-existing patches must NOT be rounded
# ---------------------------------------------------------------------------


def test_barplot_border_radius_leaves_preexisting_patches_untouched():
    """Pre-existing non-bar patches must NOT be rounded by pp.barplot."""
    fig, ax = plt.subplots()
    pre = Rectangle((0, 5), 0.5, 1, facecolor="red")
    ax.add_patch(pre)
    df = pd.DataFrame({"x": ["A", "B"], "y": [1, 2]})
    pp.barplot(data=df, x="x", y="y", ax=ax, border_radius=1.5)
    # Pre-existing Rectangle survived the barplot call.
    assert pre in ax.patches
    assert type(pre) is Rectangle
    assert not isinstance(pre, _RoundedBarPatch)
    plt.close(fig)


def test_boxplot_border_radius_leaves_preexisting_patches_untouched():
    """Pre-existing non-box patches must NOT be rounded by pp.boxplot."""
    fig, ax = plt.subplots()
    pre = Rectangle((0, 0.5), 0.1, 0.1, facecolor="green")
    ax.add_patch(pre)
    df = pd.DataFrame({"g": ["a"] * 20, "y": list(range(20))})
    pp.boxplot(data=df, x="g", y="y", ax=ax, border_radius=1.5)
    assert pre in ax.patches
    assert type(pre) is Rectangle
    assert not isinstance(pre, _RoundedBarPatch)
    plt.close(fig)


def test_apply_border_radius_handles_pathpatch():
    """Feeding a rectangular PathPatch produces a _RoundedBarPatch.

    Mirrors the shape seaborn 0.13+ uses for boxplot IQR boxes — a
    PathPatch with 5 vertices (MOVETO + 3 LINETO + CLOSEPOLY). The
    helper should derive bounds from the path extents and swap.
    """
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path

    fig, ax = plt.subplots()
    pp_rect = PathPatch(
        Path(
            [(1, 1), (2, 1), (2, 3), (1, 3), (1, 1)],
            [
                Path.MOVETO,
                Path.LINETO,
                Path.LINETO,
                Path.LINETO,
                Path.CLOSEPOLY,
            ],
        ),
        facecolor="#abc",
    )
    ax.add_patch(pp_rect)
    apply_border_radius([pp_rect], (1.0, 1.0), ax)
    assert any(isinstance(p, _RoundedBarPatch) for p in ax.patches)
    # Swapped out — original PathPatch no longer in ax.patches.
    assert pp_rect not in ax.patches
    # Bounds derived from path extents.
    new = [p for p in ax.patches if isinstance(p, _RoundedBarPatch)][0]
    assert new.get_x() == 1.0
    assert new.get_y() == 1.0
    assert new.get_width() == 1.0
    assert new.get_height() == 2.0
    plt.close(fig)


# ---------------------------------------------------------------------------
# Raincloud offset — regression for the _RoundedBarPatch shift bug
# ---------------------------------------------------------------------------
# In v0.10.6 development, pp.raincloudplot was visibly shifting its IQR box
# TOWARD the violin when box.border_radius was set — the box lost its
# box_offset shift while the whiskers/caps/median (Line2D) kept it. Root
# cause: offset_patches mutated patch.get_path().vertices in place, which
# is a no-op for _RoundedBarPatch (path is rebuilt at draw time from
# _rounded_xy). Fixed by type-dispatching in offset_patches to use
# _RoundedBarPatch.set_xy().


def test_raincloud_rounded_box_offset_survives_draw():
    """pp.raincloudplot(box.border_radius=...) box must land at the same
    x-coord as the flat baseline (i.e., same box_offset applied), and the
    offset must persist across fig.canvas.draw().
    """
    from matplotlib.patches import PathPatch

    df = pd.DataFrame({"g": ["a"] * 40, "y": list(range(40))})

    # Baseline: flat raincloud — capture the IQR-box bbox x0.
    fig, ax = plt.subplots()
    pp.raincloudplot(data=df, x="g", y="y", ax=ax)
    flat_x0 = [
        p.get_path().get_extents().x0
        for p in ax.patches
        if isinstance(p, PathPatch)
    ]
    plt.close(fig)
    assert flat_x0, "baseline raincloud produced no PathPatch box"

    # Rounded raincloud: the _RoundedBarPatch xy must match the flat x0.
    pp.rcParams["box.border_radius"] = 1.5
    try:
        fig, ax = plt.subplots()
        pp.raincloudplot(data=df, x="g", y="y", ax=ax)
        rounded = [p for p in ax.patches if isinstance(p, _RoundedBarPatch)]
        assert rounded, "rounded raincloud produced no _RoundedBarPatch"
        x_before = rounded[0].get_xy()[0]
        assert abs(x_before - flat_x0[0]) < 1e-6, (
            f"offset drifted: flat={flat_x0[0]}, rounded={x_before}"
        )
        # Draw the canvas — _RoundedBarPatch rebuilds its path on draw,
        # so we need the set_xy mutation to persist.
        fig.canvas.draw()
        x_after = rounded[0].get_xy()[0]
        assert abs(x_after - x_before) < 1e-6, (
            f"offset lost on draw: before={x_before}, after={x_after}"
        )
        plt.close(fig)
    finally:
        pp.rcParams["box.border_radius"] = (0.0, 0.0)


def test_offset_patches_handles_rounded_bar_patch():
    """Unit test: offset_patches on a _RoundedBarPatch shifts via set_xy,
    not path-vertex mutation.
    """
    from publiplots.utils.offset import offset_patches

    fig, ax = plt.subplots()
    rounded = _RoundedBarPatch(
        xy=(1.0, 2.0),
        width=1.0,
        height=1.0,
        top_pts=5.0,
        bottom_pts=5.0,
    )
    ax.add_patch(rounded)
    offset_patches([rounded], offset=0.3, orientation="vertical")
    assert rounded.get_xy()[0] == pytest.approx(1.3)
    assert rounded.get_xy()[1] == pytest.approx(2.0)
    # Horizontal orientation shifts y, not x.
    offset_patches([rounded], offset=-0.1, orientation="horizontal")
    assert rounded.get_xy()[0] == pytest.approx(1.3)
    assert rounded.get_xy()[1] == pytest.approx(1.9)
    plt.close(fig)
