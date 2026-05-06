"""Tests for pp.legend_group(axes=...) scoping and multi-group figures.

Two independent legend groups on a single figure, each scoped to a
disjoint subset of axes via ``axes=``, should:

- each collect only from the axes it was given
- not evict per-axis legends outside its scope
- coexist in ``fig._publiplots_legend_groups``
- both contribute to the figure's reservation band (top/bottom/right/left)

Overlapping scope with overlapping ``collect=`` emits a ``UserWarning``.
"""
from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
from matplotlib.legend import Legend
from matplotlib.patches import Rectangle

import publiplots as pp
from publiplots.utils.legend_entries import LegendEntry, stash_entry


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _stub_handle(color="red", label="A"):
    return Rectangle((0, 0), 1, 1, facecolor=color, label=label)


def _stash_hue(ax, name, color="red", labels=("A",)):
    stash_entry(
        ax,
        LegendEntry.build(
            name=name,
            kind="hue",
            handles=[_stub_handle(color, label=lab) for lab in labels],
            labels=list(labels),
        ),
    )


def _legend_titles(group) -> list[str]:
    return [
        e[1].get_title().get_text()
        for e in group._builder.elements
        if e[0] == "legend"
    ]


def test_axes_scope_single_axes_only_collects_from_that_axes():
    fig, axes = pp.subplots(1, 3, axes_size=(40, 30))
    _stash_hue(axes[0], "treatment")
    _stash_hue(axes[1], "dose")
    _stash_hue(axes[2], "response")
    group = pp.legend_group(anchor=axes[1], axes=[axes[1]])
    group._materialize()
    assert _legend_titles(group) == ["dose"]


def test_axes_scope_list_collects_only_from_listed_axes():
    fig, axes = pp.subplots(1, 3, axes_size=(40, 30))
    _stash_hue(axes[0], "treatment")
    _stash_hue(axes[1], "dose")
    _stash_hue(axes[2], "response")
    group = pp.legend_group(anchor=axes[-1], axes=[axes[0], axes[2]])
    group._materialize()
    assert sorted(_legend_titles(group)) == ["response", "treatment"]


def test_axes_scope_none_preserves_full_grid_collection():
    """Regression: axes=None still walks every axes in the grid."""
    fig, axes = pp.subplots(1, 2, axes_size=(40, 30))
    _stash_hue(axes[0], "treatment")
    _stash_hue(axes[1], "dose")
    group = pp.legend_group(anchor=axes[-1])
    group._materialize()
    assert sorted(_legend_titles(group)) == ["dose", "treatment"]


def test_axes_scope_accepts_bare_axes():
    fig, axes = pp.subplots(1, 2, axes_size=(40, 30))
    _stash_hue(axes[0], "treatment")
    _stash_hue(axes[1], "dose")
    # Single Axes instead of a list
    group = pp.legend_group(anchor=axes[0], axes=axes[0])
    group._materialize()
    assert _legend_titles(group) == ["treatment"]


def test_axes_scope_rejects_non_axes():
    fig, axes = pp.subplots(1, 2, axes_size=(40, 30))
    with pytest.raises(TypeError, match="sequence of Axes"):
        pp.legend_group(anchor=axes[0], axes=["not an axes"])


def test_multiple_groups_register_on_figure():
    fig, axes = pp.subplots(1, 2, axes_size=(40, 30))
    g1 = pp.legend_group(anchor=axes[0], axes=[axes[0]])
    g2 = pp.legend_group(anchor=axes[1], axes=[axes[1]])
    assert fig._publiplots_legend_groups == [g1, g2]


def test_two_disjoint_groups_collect_independently():
    fig, axes = pp.subplots(1, 2, axes_size=(40, 30))
    _stash_hue(axes[0], "treatment")
    _stash_hue(axes[1], "dose")
    g1 = pp.legend_group(anchor=axes[0], axes=[axes[0]], side="top")
    g2 = pp.legend_group(anchor=axes[1], axes=[axes[1]], side="top")
    g1._materialize()
    g2._materialize()
    assert _legend_titles(g1) == ["treatment"]
    assert _legend_titles(g2) == ["dose"]


def test_two_disjoint_groups_emit_no_overlap_warning():
    fig, axes = pp.subplots(1, 2, axes_size=(40, 30))
    pp.legend_group(anchor=axes[0], axes=[axes[0]])
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        # Must not raise — axes scopes are disjoint.
        pp.legend_group(anchor=axes[1], axes=[axes[1]])


def test_overlapping_axes_scope_warns():
    """Two full-grid (axes=None) groups compete for every axes."""
    fig, axes = pp.subplots(1, 2, axes_size=(40, 30))
    pp.legend_group(anchor=axes[0])
    with pytest.warns(UserWarning, match="scope overlaps"):
        pp.legend_group(anchor=axes[1])


def test_two_figure_anchored_groups_rejected():
    """Two figure-anchored groups corrupt the settle pass; must raise."""
    fig, axes = pp.subplots(2, 2, axes_size=(40, 30))
    pp.legend_group(side="top")
    with pytest.raises(ValueError, match="Only one figure-anchored"):
        pp.legend_group(side="bottom")


def test_overlapping_scope_disjoint_collect_emits_no_warning():
    """Overlapping axes but disjoint collect= means no entry-claim conflict."""
    fig, axes = pp.subplots(1, 2, axes_size=(40, 30))
    pp.legend_group(anchor=axes[0], collect=["treatment"])
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        pp.legend_group(anchor=axes[1], collect=["dose"])


def test_eviction_scoped_to_axes():
    """A scoped group must not evict per-axis legends outside its scope.

    We attach legends to every axes, then create a group scoped to just
    one axis. The legend on the out-of-scope axis must survive.
    """
    fig, axes = pp.subplots(1, 2, axes_size=(40, 30))
    _stash_hue(axes[0], "treatment")
    _stash_hue(axes[1], "treatment")

    # Render per-axis legends on both axes first (simulating the
    # "group attached after plotting" path).
    from publiplots.utils.plot_legend import render_entries
    from publiplots.utils.legend_entries import resolve_legend_flags
    flags = resolve_legend_flags(True)
    render_entries(axes[0], flags=flags)
    render_entries(axes[1], flags=flags)
    assert axes[0].legend_ is not None
    assert axes[1].legend_ is not None

    # Scope the group to axes[0] only — axes[1]'s legend must survive.
    pp.legend_group(anchor=axes[0], axes=[axes[0]], side="top")
    assert axes[0].legend_ is None
    assert axes[1].legend_ is not None


def test_two_groups_materialize_independently_on_draw():
    """End-to-end: two bands on the same figure, both render at draw time."""
    fig, axes = pp.subplots(1, 2, axes_size=(40, 30))
    _stash_hue(axes[0], "treatment")
    _stash_hue(axes[1], "dose")
    pp.legend_group(anchor=axes[0], axes=[axes[0]], side="top")
    pp.legend_group(anchor=axes[1], axes=[axes[1]], side="top")
    fig.canvas.draw()
    # Each axes should carry its group's Legend child after the settle pass.
    ax0_legends = [c for c in axes[0].get_children() if isinstance(c, Legend)]
    ax1_legends = [c for c in axes[1].get_children() if isinstance(c, Legend)]
    assert len(ax0_legends) == 1
    assert len(ax1_legends) == 1
    assert ax0_legends[0].get_title().get_text() == "treatment"
    assert ax1_legends[0].get_title().get_text() == "dose"
