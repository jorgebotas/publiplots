"""Regression tests for UpSet layout fix (Path A).

UpSet's gridspec spans top=1/bottom=0 (zero margin), so bar-top
annotations, xlabels, titles, and ticklabels clip on plt.show / non-tight
savefig. The fix measures post-draw overflow and grows the figure +
shifts the gridspec inward to reserve space.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

import publiplots as pp


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def simple_sets():
    rng = np.random.default_rng(42)
    return {
        "A": set(rng.integers(1, 100, 50)),
        "B": set(rng.integers(20, 120, 50)),
        "C": set(rng.integers(40, 140, 50)),
        "D": set(rng.integers(60, 160, 50)),
    }


def _overflow(fig):
    """Return (top_overflow_in, bottom_overflow_in) after drawing."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tight = fig.get_tightbbox(renderer)
    fig_bb = fig.bbox_inches
    return (tight.y1 - fig_bb.y1, -tight.y0)


def test_upsetplot_bare_fits_inside_canvas(simple_sets):
    """Bare UpSet (no title, no labels) must not clip bar annotations or ticklabels."""
    axes = pp.upsetplot(simple_sets)
    fig = axes["intersections"].get_figure()
    top, bottom = _overflow(fig)
    assert top <= 0, f"top overflow {top:+.3f} in — bar annotations clipped"
    assert bottom <= 0, f"bottom overflow {bottom:+.3f} in — xticklabels clipped"


def test_upsetplot_with_decorations_fits_inside_canvas(simple_sets):
    """UpSet with title + ylabel + xlabel must reserve space for all decorations."""
    axes = pp.upsetplot(
        simple_sets,
        title="Set Overlap Analysis",
        intersection_label="Intersection size",
        set_label="Set size",
    )
    fig = axes["intersections"].get_figure()
    top, bottom = _overflow(fig)
    assert top <= 0, f"top overflow {top:+.3f} in — title/annotations clipped"
    assert bottom <= 0, f"bottom overflow {bottom:+.3f} in — xlabel clipped"


def test_upsetplot_decorations_grow_figure_vs_bare(simple_sets):
    """Adding a title + labels should grow the figure beyond the bare version."""
    axes_bare = pp.upsetplot(simple_sets)
    h_bare = axes_bare["intersections"].get_figure().get_figheight()

    axes_decorated = pp.upsetplot(
        simple_sets,
        title="Overlap",
        intersection_label="Size",
        set_label="Size",
    )
    h_decorated = axes_decorated["intersections"].get_figure().get_figheight()

    assert h_decorated > h_bare, (
        f"decorated figure ({h_decorated:.3f} in) should be taller than "
        f"bare ({h_bare:.3f} in) — more decorations = more reserved space"
    )
