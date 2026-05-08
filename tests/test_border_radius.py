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
