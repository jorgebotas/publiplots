"""Each stashed LegendEntry is rendered exactly once per axes (#227).

The stash on an axes is cumulative and every plot call renders from it,
so before the fix the second call re-drew the first call's entries
alongside its own: N calls produced ~N(N+1)/2 legends/colorbars, each
one reserving its own layout space.

Two paths deliberately render entries LATER than they were stashed and
must keep working: a legend group created before the plot calls (which
materializes stashed entries on a later pass), and ``pp.legend(ax)``'s
adopt rebuild (which drains its builder and re-renders what it drained).
Neither consults the per-axes render record, and the tests below pin
that.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.legend import Legend
import numpy as np
import pandas as pd
import pytest

import publiplots as pp
from publiplots.utils.legend_entries import (
    LegendEntry,
    resolve_legend_flags,
    stash_entry,
)
from publiplots.utils.plot_legend import render_entries


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


MM = 25.4
CONT = ["nc", "nd", "ne", "nf", "ng"]
CAT = ["kc", "kd", "ke", "kf", "kg"]


def _df(seed=0):
    rng = np.random.default_rng(seed)
    n = 60
    cols = {"x": rng.normal(size=n), "y": rng.normal(size=n)}
    for i, name in enumerate(CONT):
        cols[name] = rng.normal(size=n) + i
    for name in CAT:
        cols[name] = rng.choice(["a", "b"], size=n)
    return pd.DataFrame(cols)


def _walk_axes(fig):
    """Every axes on the figure, including inset children.

    An ``inside=True`` colorbar is an ``ax.inset_axes`` child and never
    appears in ``fig.get_axes()``.
    """
    out = []

    def walk(ax):
        out.append(ax)
        for child in (getattr(ax, "child_axes", None) or []):
            walk(child)

    for ax in list(fig.get_axes()):
        walk(ax)
    return out


def _colorbar_strips(fig):
    """Colorbar strip axes, identified by the back-reference
    ``Colorbar.__init__`` sets unconditionally."""
    return [ax for ax in _walk_axes(fig)
            if getattr(ax, "_colorbar", None) is not None]


def _legend_refs(fig):
    """Every Legend reference in the figure's child lists.

    References, not objects: re-parenting a legend used to list the same
    artist several times under one axes.
    """
    refs = []
    for ax in _walk_axes(fig):
        refs += [c for c in ax.get_children() if isinstance(c, Legend)]
    return refs


def _legends(fig):
    """Distinct Legend objects, by identity."""
    out = []
    for leg in _legend_refs(fig):
        if not any(leg is seen for seen in out):
            out.append(leg)
    return out


def _colorbar_labels(fig):
    """The floating labels colorbar strips are titled with."""
    return [t.get_text() for t in fig.texts if t.get_text()]


def _legend_titles(fig):
    return [leg.get_title().get_text() for leg in _legends(fig)]


def _fig_mm(fig):
    w, h = fig.get_size_inches()
    return (w * MM, h * MM)


def _settle(fig):
    fig.canvas.draw()
    layout = getattr(fig, "_publiplots_auto_layout", None)
    if layout is not None:
        layout.settle()


# --------------------------------------------------------------------
# The core defect: successive calls on one axes.
# --------------------------------------------------------------------

@pytest.mark.parametrize("n_calls", [1, 2, 3, 5])
def test_continuous_hue_renders_one_strip_per_call(n_calls):
    """N calls -> N colorbars, not N(N+1)/2.

    Three or more calls matter here: with two, the triangular growth
    (3) and a single stale duplicate (3) are indistinguishable.
    """
    df = _df()
    fig, ax = pp.subplots(1, 1, axes_size=(40, 35))
    for col in CONT[:n_calls]:
        pp.scatterplot(data=df, x="x", y="y", hue=col, ax=ax)
    _settle(fig)

    assert len(_colorbar_strips(fig)) == n_calls
    assert _colorbar_labels(fig) == CONT[:n_calls]


@pytest.mark.parametrize("n_calls", [1, 2, 3, 5])
def test_categorical_hue_renders_one_legend_per_call(n_calls):
    """N calls -> N legends, counted both by object and by child reference.

    The child-reference count is the stricter of the two: the reported
    "4 legends after 2 calls" was 3 distinct Legend objects (the
    triangular re-render) plus one artist listed twice in the axes'
    child list by the re-parenting in ``LegendBuilder.add_legend``.
    """
    df = _df()
    fig, ax = pp.subplots(1, 1, axes_size=(40, 35))
    for col in CAT[:n_calls]:
        pp.scatterplot(data=df, x="x", y="y", hue=col, ax=ax)
    _settle(fig)

    assert len(_legends(fig)) == n_calls
    assert len(_legend_refs(fig)) == n_calls
    assert _legend_titles(fig) == CAT[:n_calls]


def test_mixed_kinds_render_once_each():
    """Alternating continuous and categorical hues, five calls."""
    df = _df()
    seq = ["nc", "kc", "nd", "kd", "ne"]
    fig, ax = pp.subplots(1, 1, axes_size=(40, 35))
    for col in seq:
        pp.scatterplot(data=df, x="x", y="y", hue=col, ax=ax)
    _settle(fig)

    assert len(_colorbar_strips(fig)) == 3
    assert _colorbar_labels(fig) == ["nc", "nd", "ne"]
    assert len(_legends(fig)) == 2
    assert len(_legend_refs(fig)) == 2
    assert _legend_titles(fig) == ["kc", "kd"]


@pytest.mark.parametrize("col,count_fn,label_fn", [
    ("nc", _colorbar_strips, _colorbar_labels),
    ("kc", _legends, _legend_titles),
])
def test_same_column_twice_renders_both_entries(col, count_fn, label_fn):
    """Two calls on the same column stash two entries, and each is owed
    one render.

    The record is keyed by entry identity, not by name: keying by name
    would collapse these two into one and silently drop the second
    call's legend.
    """
    df = _df()
    fig, ax = pp.subplots(1, 1, axes_size=(40, 35))
    pp.scatterplot(data=df, x="x", y="y", hue=col, ax=ax)
    pp.scatterplot(data=df, x="x", y="y", hue=col, ax=ax)
    _settle(fig)

    assert len(count_fn(fig)) == 2
    assert label_fn(fig) == [col, col]


def test_heatmap_renders_one_strip_per_call():
    rng = np.random.default_rng(1)
    data = pd.DataFrame(rng.normal(size=(5, 4)), columns=list("PQRS"),
                        index=list("VWXYZ"))
    fig, ax = pp.subplots(1, 1, axes_size=(40, 35))
    pp.heatmap(data=data, ax=ax)
    pp.heatmap(data=data, ax=ax)
    _settle(fig)

    assert len(_colorbar_strips(fig)) == 2


# --------------------------------------------------------------------
# Reclaimed layout space.
# --------------------------------------------------------------------

@pytest.mark.parametrize("n_calls", [2, 3, 5])
def test_n_calls_reserve_the_space_of_n_entries(n_calls):
    """The figure must shrink to the size N entries need — no more, and
    no less.

    The control draws the same data artists and then renders the same N
    entries in a single pass, which is what one call per axes would have
    produced. Comparing against a control rather than against "smaller
    than before" is the point: duplicates reserved real space, so a
    merely-smaller figure would still be wrong.
    """
    from matplotlib.cm import ScalarMappable

    df = _df()
    cols = CONT[:n_calls]
    fig_a, ax_a = pp.subplots(1, 1, axes_size=(40, 35))
    for col in cols:
        pp.scatterplot(data=df, x="x", y="y", hue=col, ax=ax_a)
    _settle(fig_a)
    mappables = [(e.name, e.handles[0].norm, e.handles[0].cmap)
                 for e in ax_a._publiplots_legend_entries]
    size_a = _fig_mm(fig_a)
    rect_a = ax_a.get_position().bounds
    assert len(mappables) == n_calls

    fig_b, ax_b = pp.subplots(1, 1, axes_size=(40, 35))
    for col in cols:
        pp.scatterplot(data=df, x="x", y="y", hue=col, ax=ax_b, legend=False)
    for name, norm, cmap in mappables:
        stash_entry(ax_b, LegendEntry.build(
            name=name, kind="hue",
            handles=[ScalarMappable(norm=norm, cmap=cmap)], labels=[]))
    render_entries(ax_b, flags=resolve_legend_flags(True), legend_kws={})
    _settle(fig_b)

    assert len(_colorbar_strips(fig_b)) == n_calls
    assert size_a == pytest.approx(_fig_mm(fig_b))
    assert rect_a == pytest.approx(ax_b.get_position().bounds)


@pytest.mark.parametrize("n_calls", [1, 2, 3, 5])
def test_panel_keeps_its_millimetre_size(n_calls):
    """Every duplicate reserved layout space; the panel itself must keep
    the millimetre size it was asked for regardless."""
    df = _df()
    fig, ax = pp.subplots(1, 1, axes_size=(40, 35))
    for col in CONT[:n_calls]:
        pp.scatterplot(data=df, x="x", y="y", hue=col, ax=ax)
    _settle(fig)

    fig_w, fig_h = fig.get_size_inches()
    pos = ax.get_position()
    assert pos.width * fig_w * MM == pytest.approx(40.0, abs=1e-6)
    assert pos.height * fig_h * MM == pytest.approx(35.0, abs=1e-6)


# --------------------------------------------------------------------
# The two paths that render entries later than they were stashed.
# --------------------------------------------------------------------

def test_group_created_before_the_plots_still_materializes():
    """The documented "before" ordering: the band is registered first and
    picks the stashed entries up afterwards."""
    df = _df()
    fig, axes = pp.subplots(1, 2, axes_size=(35, 30))
    pp.legend(side="right")
    for ax in axes:
        pp.scatterplot(data=df, x="x", y="y", hue="nc", ax=ax)
        pp.scatterplot(data=df, x="x", y="y", hue="kd", ax=ax)
    _settle(fig)

    assert len(_colorbar_strips(fig)) == 1
    assert _colorbar_labels(fig) == ["nc"]
    assert len(_legends(fig)) == 1
    assert _legend_titles(fig) == ["kd"]


@pytest.mark.parametrize("n_calls", [2, 3])
def test_pp_legend_adopt_re_renders_what_it_drained(n_calls):
    """``pp.legend(ax)`` tears its builder down and rebuilds it under the
    requested side; the per-axes render record must not block that."""
    df = _df()
    fig, ax = pp.subplots(1, 1, axes_size=(40, 35))
    for col in CONT[:n_calls]:
        pp.scatterplot(data=df, x="x", y="y", hue=col, ax=ax)
    pp.legend(ax, side="top")
    _settle(fig)

    assert len(_colorbar_strips(fig)) == n_calls
    assert _colorbar_labels(fig) == CONT[:n_calls]


def test_adopt_after_a_categorical_stack():
    df = _df()
    fig, ax = pp.subplots(1, 1, axes_size=(40, 35))
    for col in CAT[:3]:
        pp.scatterplot(data=df, x="x", y="y", hue=col, ax=ax)
    pp.legend(ax, side="top")
    _settle(fig)

    assert len(_legends(fig)) == 3
    assert len(_legend_refs(fig)) == 3
    assert sorted(_legend_titles(fig)) == CAT[:3]


def test_band_claiming_some_entries_after_the_plots():
    """A band built after the plots evicts the per-axes artists it
    claims and renders them itself; unclaimed entries stay per-axes.

    If the marker made an evicted entry un-renderable the band would
    silently render nothing.
    """
    df = _df()
    fig, axes = pp.subplots(1, 2, axes_size=(35, 30))
    for ax in axes:
        pp.scatterplot(data=df, x="x", y="y", hue="kc", ax=ax)
        pp.scatterplot(data=df, x="x", y="y", hue="kd", ax=ax)
    pp.legend(axes[0:2], side="top", collect=["kc"])
    _settle(fig)

    titles = _legend_titles(fig)
    assert titles.count("kc") == 1      # the band, once
    assert titles.count("kd") == 2      # one per axes, still per-axes
    assert len(_legend_refs(fig)) == 3


def test_figure_band_claims_every_entry():
    df = _df()
    fig, axes = pp.subplots(2, 2, axes_size=(30, 25))
    for ax in np.ravel(axes):
        pp.scatterplot(data=df, x="x", y="y", hue="kc", ax=ax)
        pp.scatterplot(data=df, x="x", y="y", hue="kd", ax=ax)
    pp.legend(side="bottom")
    _settle(fig)

    assert sorted(_legend_titles(fig)) == ["kc", "kd"]
    assert len(_legend_refs(fig)) == 2


# --------------------------------------------------------------------
# inside=True (a fresh builder per call, no reactor registration).
# --------------------------------------------------------------------

def test_inside_legends_render_once_each():
    df = _df()
    fig, ax = pp.subplots(1, 1, axes_size=(40, 35))
    for col in CAT[:3]:
        pp.scatterplot(data=df, x="x", y="y", hue=col, ax=ax,
                       legend_kws={"inside": True, "loc": "upper right"})
    _settle(fig)

    assert len(_legends(fig)) == 3
    assert len(_legend_refs(fig)) == 3
    assert _legend_titles(fig) == CAT[:3]


def test_inside_colorbars_render_once_each():
    df = _df()
    fig, ax = pp.subplots(1, 1, axes_size=(40, 35))
    for col in CONT[:3]:
        pp.scatterplot(data=df, x="x", y="y", hue=col, ax=ax,
                       legend_kws={"inside": True, "loc": "upper right"})
    _settle(fig)

    # The strips are ax.inset_axes children, not figure axes.
    assert len(_colorbar_strips(fig)) == 3


# --------------------------------------------------------------------
# Convergence: the fix must not leave the layout chasing itself.
# --------------------------------------------------------------------

def test_repeated_draws_and_saves_are_stable(tmp_path):
    df = _df()
    fig, ax = pp.subplots(1, 1, axes_size=(40, 35))
    for col in CONT[:3]:
        pp.scatterplot(data=df, x="x", y="y", hue=col, ax=ax)

    def snapshot():
        w, h = fig.get_size_inches()
        rects = [tuple(s.get_position().bounds) for s in _colorbar_strips(fig)]
        return (round(w * MM, 6), round(h * MM, 6),
                tuple(ax.get_position().bounds), tuple(rects))

    fig.canvas.draw()
    baseline = snapshot()
    for _ in range(7):
        fig.canvas.draw()
        assert snapshot() == baseline
    fig._publiplots_auto_layout.settle()
    assert snapshot() == baseline

    plt.figure(fig.number)
    pp.savefig(str(tmp_path / "conv.png"))
    assert snapshot() == baseline
    plt.figure(fig.number)
    pp.savefig(str(tmp_path / "conv.pdf"))
    assert snapshot() == baseline
    assert len(_colorbar_strips(fig)) == 3
