"""Grid ink comes from rcParams, and nothing renders it at full opacity.

grid.color is black ink dimmed by grid.alpha. ax.grid() inherits that
alpha automatically (verified), so the hazards are elsewhere: a
hardcoded alpha that OVERRIDES the rcParam, and non-gridline artists
(axhline, spines) that inherit nothing and would render solid black.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgba

import publiplots as pp


@pytest.fixture(autouse=True)
def _close_figures():
    """Repo convention (see tests/test_upset_layout.py, test_venn_orientation.py,
    test_heatmap_legend_stash.py): close every figure between tests so none leak."""
    yield
    plt.close("all")


def _visible_gridlines(ax):
    lines = list(ax.xaxis.get_gridlines()) + list(ax.yaxis.get_gridlines())
    return [l for l in lines if l.get_visible()]


def _minor_gridlines(ax):
    """Minor gridlines are reached through the ticks, not get_gridlines().

    ``Axis.get_gridlines()`` returns major gridlines only and rejects a
    ``which=`` kwarg (TypeError), so go via the minor ticks.
    """
    ticks = list(ax.xaxis.get_minor_ticks()) + list(ax.yaxis.get_minor_ticks())
    return [t.gridline for t in ticks if t.gridline.get_visible()]


# ---- pp.add_grid ------------------------------------------------------------

def test_add_grid_matches_grid_rcparams():
    fig, ax = pp.subplots(1, 1, axes_size=(40, 30))
    pp.add_grid(ax, axis="both")
    lines = _visible_gridlines(ax)
    assert lines, "expected visible gridlines"
    for line in lines:
        assert line.get_alpha() == pytest.approx(plt.rcParams["grid.alpha"])
        assert to_rgba(line.get_color()) == to_rgba(plt.rcParams["grid.color"])
        assert line.get_linewidth() == pytest.approx(plt.rcParams["grid.linewidth"])


def test_add_grid_is_never_opaque():
    fig, ax = pp.subplots(1, 1, axes_size=(40, 30))
    pp.add_grid(ax, axis="both")
    for line in _visible_gridlines(ax):
        alpha = line.get_alpha()
        assert alpha is not None, "gridline has no alpha -> renders solid"
        assert alpha < 1.0


# ---- dot heatmap: separators AND spines ------------------------------------

def _dot_heatmap(n_rows, n_cols, **kwargs):
    """Build a dot heatmap of a given shape.

    The chrome under test lives in _draw_dot_heatmap, reached only when
    ``size=`` is passed; a plain pp.heatmap() does not go there.
    """
    rows = [
        (f"r{r}", f"c{c}", float(r * n_cols + c), 5.0)
        for r in range(n_rows)
        for c in range(n_cols)
    ]
    long_df = pd.DataFrame(rows, columns=["row", "col", "value", "size"])
    ax = pp.heatmap(
        data=long_df, x="col", y="row", value="value", size="size", **kwargs
    )
    ax.get_figure().canvas.draw()
    return ax


@pytest.fixture
def dot_heatmap_ax():
    """The hardcoded separators live in _draw_dot_heatmap, which is only
    reached when size= is passed. A plain pp.heatmap() does not go there."""
    long_df = pd.DataFrame({
        "row": list("aaabbbccc"),
        "col": list("xyz") * 3,
        "value": np.arange(9.0),
        "size": np.arange(1, 10.0),
    })
    ax = pp.heatmap(data=long_df, x="col", y="row", value="value", size="size")
    ax.get_figure().canvas.draw()
    return ax


def test_dot_heatmap_separators_use_grid_rcparams(dot_heatmap_ax):
    lines = _minor_gridlines(dot_heatmap_ax)
    assert lines, "expected minor gridlines between cells"
    for line in lines:
        assert to_rgba(line.get_color()) == to_rgba(plt.rcParams["grid.color"])
        assert line.get_linewidth() == pytest.approx(plt.rcParams["grid.linewidth"])
        assert line.get_alpha() == pytest.approx(plt.rcParams["grid.alpha"])


def test_dot_heatmap_border_is_one_tone_with_the_inner_grid(dot_heatmap_ax):
    """The border and the inner lattice must render as ONE tone.

    The spine composites grid.alpha into its colour, while a minor gridline
    carries the same colour at grid.alpha as a separate channel -- so the two
    match only if the effective RGBA agrees. This also pins the reason the
    minor ticks stop at the INTERIOR boundaries: the axis limits sit exactly
    on the outer boundaries, so a gridline there would overlay the spine and
    the two alphas would compound into a heavier border (measured 1.44x
    before the fix, 1.00x after).
    """
    lines = _minor_gridlines(dot_heatmap_ax)
    assert lines, "expected minor gridlines between cells"

    def effective(color, alpha):
        r, g, b, a = to_rgba(color)
        return (r, g, b, a if alpha is None else a * alpha)

    grid_tone = effective(lines[0].get_color(), lines[0].get_alpha())
    for spine in dot_heatmap_ax.spines.values():
        spine_tone = effective(spine.get_edgecolor(), spine.get_alpha())
        assert spine_tone == pytest.approx(grid_tone), (
            f"border tone {spine_tone} != inner grid tone {grid_tone}"
        )
        assert spine.get_linewidth() == pytest.approx(lines[0].get_linewidth())

    # No gridline may coincide with a spine, or the alphas compound.
    xlim, ylim = dot_heatmap_ax.get_xlim(), dot_heatmap_ax.get_ylim()
    edges = {round(v, 6) for v in (*xlim, *ylim)}
    ticks = [
        *dot_heatmap_ax.xaxis.get_minorticklocs(),
        *dot_heatmap_ax.yaxis.get_minorticklocs(),
    ]
    assert not (edges & {round(t, 6) for t in ticks}), (
        "a minor gridline sits on the axes limit, doubling the border ink"
    )


@pytest.mark.parametrize("n_rows,n_cols", [(1, 1), (1, 4), (4, 1), (2, 2), (3, 5)])
def test_dot_heatmap_grid_is_adjacent_and_complete(n_rows, n_cols):
    """Tone alone is not enough -- pin ADJACENCY and COMPLETENESS too.

    Tone equality says the border and the lattice are the same colour; it
    says nothing about whether they touch, or whether every separator is
    present. Two failures slip past a tone-only check:

    - a margin change pulling the limits off the outer cell boundaries would
      open a white gutter between the outer cells and the border;
    - a wrong tick range (say ``np.arange(2, n) - 0.5``) would drop a
      separator while every surviving one still matched in tone.
    """
    ax = _dot_heatmap(n_rows, n_cols)

    # ADJACENCY: the limits sit exactly on the outer cell boundaries, so the
    # spines ARE the outer cell edges -- no gutter, no overlap.
    assert ax.get_xlim() == pytest.approx((-0.5, n_cols - 0.5))
    assert ax.get_ylim() == pytest.approx((n_rows - 0.5, -0.5)), (
        "y must stay inverted with the first row at the top"
    )

    # COMPLETENESS: exactly one separator per interior boundary.
    assert len(ax.xaxis.get_minorticklocs()) == n_cols - 1
    assert len(ax.yaxis.get_minorticklocs()) == n_rows - 1
    assert list(ax.xaxis.get_minorticklocs()) == pytest.approx(
        list(np.arange(1, n_cols) - 0.5)
    )
    assert list(ax.yaxis.get_minorticklocs()) == pytest.approx(
        list(np.arange(1, n_rows) - 0.5)
    )

    # The border is always drawn, even when there are no interior separators.
    for spine in ax.spines.values():
        assert spine.get_visible()
        assert to_rgba(spine.get_edgecolor())[3] == pytest.approx(
            plt.rcParams["grid.alpha"]
        )


@pytest.mark.parametrize("n_rows,n_cols", [(1, 4), (4, 1), (1, 1), (3, 3), (3, 5)])
def test_dot_heatmap_square_gives_square_cells(n_rows, n_cols):
    """square=True must produce square RENDERED cells at any shape.

    A degenerate axis (single row or column) has zero data extent, so a
    margin expressed as a fraction of that extent collapses and matplotlib
    falls back to +/-0.1 -- which silently made 1xN cells 5x wider than tall
    under set_aspect("equal"). The limits are set explicitly to prevent that.
    """
    ax = _dot_heatmap(n_rows, n_cols, square=True)
    box = ax.get_window_extent()
    cell_aspect = (box.width / n_cols) / (box.height / n_rows)
    assert cell_aspect == pytest.approx(1.0, abs=0.01), (
        f"{n_rows}x{n_cols} square=True gives cell aspect {cell_aspect:.4f}; "
        f"box {box.width:.1f}x{box.height:.1f}"
    )


def test_dot_heatmap_spines_are_not_opaque_black(dot_heatmap_ax):
    """A spine inherits nothing from grid.alpha, so the alpha has to be
    composited into its colour or the cell-matrix border lands solid."""
    for spine in dot_heatmap_ax.spines.values():
        rgba = to_rgba(spine.get_edgecolor())
        assert rgba[3] < 1.0, "spine drawn at full opacity"


# ---- upset: gridlines and the separator axhline ---------------------------

@pytest.fixture
def upset_axes():
    """pp.upsetplot returns {"intersections", "matrix", "sets"} -> Axes, which
    lets the separator test scope itself to the matrix axes."""
    axes = pp.upsetplot(data={"A": {1, 2, 3, 4}, "B": {3, 4, 5}, "C": {4, 5, 6}})
    fig = axes["matrix"].get_figure()
    fig.canvas.draw()
    return {**axes, "figure": fig}


def test_upset_gridlines_use_grid_alpha(upset_axes):
    """The hardcoded alpha=0.3 used to override the rcParam, and the width
    came from a GRID_LINEWIDTH = 1 constant instead of grid.linewidth."""
    found = False
    for ax in upset_axes["figure"].get_axes():
        for line in _visible_gridlines(ax):
            found = True
            assert line.get_alpha() == pytest.approx(plt.rcParams["grid.alpha"])
            assert to_rgba(line.get_color()) == to_rgba(plt.rcParams["grid.color"])
            assert line.get_linewidth() == pytest.approx(
                plt.rcParams["grid.linewidth"]
            )
    assert found, "expected gridlines on the upset bar axes"


def test_upset_set_separators_are_not_opaque(upset_axes):
    """The separators are axhlines, which inherit no grid alpha at all.

    Scoped to the matrix axes rather than matching every black line in the
    figure: grid.color is plain black now, so a figure-wide colour match
    would sweep in any future black line and force it to carry grid.alpha.
    """
    grid_rgba = to_rgba(plt.rcParams["grid.color"])
    separators = [
        l for l in upset_axes["matrix"].lines
        if to_rgba(l.get_color())[:3] == grid_rgba[:3]
    ]
    assert separators, "expected set-separator lines in the grid colour"
    for line in separators:
        alpha = line.get_alpha()
        assert alpha is not None, "separator axhline has no alpha -> solid black"
        assert alpha == pytest.approx(plt.rcParams["grid.alpha"])


# ---- venn ------------------------------------------------------------------

def test_venn_labels_are_not_scaled_above_body_type():
    """Set labels used to be drawn at font.size * 1.2, breaking flat 7pt.

    Note pp.venn takes `sets=`, not `data=`. Before the fix this figure
    reports two distinct text sizes (font.size and font.size * 1.2); after
    it, exactly one.
    """
    ax = pp.venn(sets={"A": {1, 2, 3}, "B": {2, 3, 4}})
    ax.get_figure().canvas.draw()
    sizes = {round(t.get_fontsize(), 3) for t in ax.texts}
    assert sizes, "expected text on the venn diagram"
    # A set of pytest.approx objects is unhashable, so compare the single
    # expected size instead of the set literal.
    assert len(sizes) == 1 and sorted(sizes)[0] == pytest.approx(
        plt.rcParams["font.size"]
    ), (
        f"venn draws text at {sorted(sizes)}; flat type wants only "
        f"{plt.rcParams['font.size']}"
    )
