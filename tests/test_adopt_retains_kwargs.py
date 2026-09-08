"""``pp.legend(ax)`` keeps the plot call's ``legend_kws`` (#258) and yields
shared entries to a band that already claims them (#233).

Both defects live on the adopt path, where a claimed entry is re-rendered
from the stash. #258: a ``LegendEntry`` stored nothing about how it had
been presented, so ``_render_entry`` could only rebuild a default artist
— ``legend_kws={'width': 30, 'height': 8}`` came back 15 x 4.5mm and
``legend_kws={'ncol': 3}`` came back at the label count. #233: the
re-render consulted the raw stash rather than asking whether anyone else
already owned the entry, so a band created first ended up with a second
copy beside one of its panels, silently.

The record is a side table keyed by entry identity
(``record_entry_kwargs``), the same shape as #227's render record, so the
frozen entry and its cross-axes ``signature`` are untouched.
"""
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import publiplots as pp


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


AXES_SIZE = (50, 40)
_DEFAULT_HORIZONTAL = (15.0, 4.5)


def _df(n=60, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "x": rng.normal(size=n),
        "y": rng.normal(size=n),
        "c": np.linspace(0.0, 1.0, n),
        "g": np.tile(list("ABCDEF"), n // 6),
    })


def _mm(fig, px):
    return px / fig.dpi * 25.4


def _strips(ax):
    """Every colorbar any per-axes builder drew for ``ax``.

    Covers the cached ``collect=[]`` group, the ``inside=True``
    short-circuit and the adopted group (``_reconfigure_for_adopt``
    re-registers its rebuilt builder). An inside strip is an
    ``ax.inset_axes`` child and never appears in ``fig.get_axes()``.
    """
    out = []
    for builder in getattr(ax, "_publiplots_legend_builders", []):
        for kind, artist in builder.elements:
            if kind == "colorbar":
                out.append(artist)
    return out


def _strip_mm(fig, cbar):
    rect = cbar.ax.get_window_extent()
    return (_mm(fig, rect.width), _mm(fig, rect.height))


def _legends(fig):
    return [child for ax in fig.get_axes() for child in ax.get_children()
            if type(child).__name__ == "Legend"]


def _all_colorbars(fig):
    """Every colorbar strip on ``fig``, inside ones included.

    ``fig.get_axes()`` misses an ``inside=True`` strip (it lives in the
    parent's ``child_axes``), so walk both and identify a strip by the
    ``_colorbar`` back-reference matplotlib sets on it.
    """
    seen = []
    candidates = list(fig.get_axes())
    for ax in list(candidates):
        candidates.extend(getattr(ax, "child_axes", ()) or ())
    for ax in candidates:
        cbar = getattr(ax, "_colorbar", None)
        if cbar is not None and not any(cbar is c for c in seen):
            seen.append(cbar)
    return seen


def _overlap(caught):
    return [w for w in caught if "scope overlaps" in str(w.message)]


# ---------------------------------------------------------------------------
# #258 — the adopt path re-applies the plot call's legend_kws
# ---------------------------------------------------------------------------


def test_adopt_retains_colorbar_geometry():
    """``legend_kws={'width': 30, 'height': 8}`` survives ``pp.legend(ax)``.

    The measurement from the issue: 30 x 8 before, 15 x 4.5 after.
    """
    fig, ax = pp.subplots(1, 1, axes_size=AXES_SIZE)
    pp.scatterplot(data=_df(), x="x", y="y", hue="c", ax=ax,
                   legend_kws={"side": "top", "width": 30, "height": 8})
    fig.canvas.draw()
    strips = _strips(ax)
    assert len(strips) == 1
    assert _strip_mm(fig, strips[0]) == pytest.approx((30.0, 8.0), abs=1e-6)

    pp.legend(ax, side="top")
    fig.canvas.draw()
    fig.canvas.draw()
    strips = _strips(ax)
    assert len(strips) == 1, f"expected one strip after adoption, got {len(strips)}"
    after = _strip_mm(fig, strips[0])
    assert after == pytest.approx((30.0, 8.0), abs=1e-6), (
        f"adoption re-rendered a default strip instead of the sized one: {after}"
    )
    assert after != pytest.approx(_DEFAULT_HORIZONTAL, abs=1e-6)


def test_adopt_retains_categorical_ncol():
    """``legend_kws={'ncol': 3}`` survives ``pp.legend(ax)`` — 3, not 6."""
    fig, ax = pp.subplots(1, 1, axes_size=AXES_SIZE)
    pp.scatterplot(data=_df(), x="x", y="y", hue="g", ax=ax,
                   legend_kws={"side": "top", "ncol": 3})
    fig.canvas.draw()
    legends = _legends(fig)
    assert len(legends) == 1 and legends[0]._ncols == 3

    pp.legend(ax, side="top")
    fig.canvas.draw()
    legends = _legends(fig)
    assert len(legends) == 1, f"expected one Legend, got {len(legends)}"
    assert legends[0]._ncols == 3, (
        f"adoption reset ncol to the label count: {legends[0]._ncols}"
    )


def test_adopt_with_no_side_keeps_the_plot_calls_side():
    """A bare ``pp.legend(ax)`` must not drag the legend back to the right.

    ``side`` is the one placement key whose signature default ('right')
    is a real value, so without the sentinel check the factory cannot
    tell an omitted ``side`` from an explicit one.
    """
    fig, ax = pp.subplots(1, 1, axes_size=AXES_SIZE)
    pp.scatterplot(data=_df(), x="x", y="y", hue="g", ax=ax,
                   legend_kws={"side": "top"})
    assert ax._legend_group._side == "top"

    group = pp.legend(ax)
    fig.canvas.draw()
    assert group._side == "top", (
        f"bare adoption reset side to {group._side!r}"
    )


def test_explicit_side_beats_the_stashed_one():
    """``pp.legend(ax, side='left')`` overrides ``legend_kws={'side': 'top'}``.

    The ordering that makes the whole record safe: if the stash won, the
    documented override idiom would stop working.
    """
    fig, ax = pp.subplots(1, 1, axes_size=AXES_SIZE)
    pp.scatterplot(data=_df(), x="x", y="y", hue="g", ax=ax,
                   legend_kws={"side": "top"})
    group = pp.legend(ax, side="left")
    fig.canvas.draw()
    assert group._side == "left", (
        f"the stashed side won over the explicit one: {group._side!r}"
    )
    legends = _legends(fig)
    assert len(legends) == 1
    lb = legends[0].get_window_extent()
    ab = ax.get_window_extent()
    assert lb.x1 <= ab.x0 + 2, (lb.x1, ab.x0)


def test_explicit_side_beats_stashed_for_a_colorbar_too():
    """Same ordering on the continuous branch, geometry still inherited."""
    fig, ax = pp.subplots(1, 1, axes_size=AXES_SIZE)
    pp.scatterplot(data=_df(), x="x", y="y", hue="c", ax=ax,
                   legend_kws={"side": "top", "width": 30, "height": 8})
    group = pp.legend(ax, side="bottom")
    fig.canvas.draw()
    fig.canvas.draw()
    assert group._side == "bottom"
    strips = _strips(ax)
    assert len(strips) == 1
    assert _strip_mm(fig, strips[0]) == pytest.approx((30.0, 8.0), abs=1e-6)


def test_second_plot_calls_kwargs_do_not_overwrite_the_first():
    """The record is per entry, so two calls on one axes each keep theirs.

    First writer wins is a per-``LegendEntry`` rule, not a per-axes one:
    ``ncol=3`` on the ``g`` entry and ``ncol=2`` on the ``h`` entry both
    render as asked.
    """
    frame = _df()
    frame["h"] = np.tile(list("XY"), len(frame) // 2)
    fig, ax = pp.subplots(1, 1, axes_size=AXES_SIZE)
    pp.scatterplot(data=frame, x="x", y="y", hue="g", ax=ax,
                   legend_kws={"ncol": 3})
    pp.scatterplot(data=frame, x="x", y="y", hue="h", ax=ax,
                   legend_kws={"ncol": 2})
    pp.legend(ax)
    fig.canvas.draw()
    by_title = {lg.get_title().get_text(): lg._ncols for lg in _legends(fig)}
    assert by_title == {"g": 3, "h": 2}, by_title


def test_a_band_ignores_the_stashed_kwargs():
    """Only the per-axes form inherits ``legend_kws``; a band does not.

    Two panels stash the same entry with different geometry. A shared
    strip has no non-arbitrary way to pick one, and ``legend_kws`` on a
    plot call describes that panel's own legend — so the band renders its
    default and the conflict never reaches a pixel. This is why
    ``_merge_entries`` resolves conflicting kwargs first-wins without
    warning.
    """
    fig, axes = pp.subplots(1, 2, axes_size=(35, 30))
    pp.scatterplot(data=_df(), x="x", y="y", hue="c", ax=axes[0],
                   legend_kws={"width": 30, "height": 8})
    pp.scatterplot(data=_df(), x="x", y="y", hue="c", ax=axes[1],
                   legend_kws={"width": 10, "height": 3})
    pp.legend(side="right")
    fig.canvas.draw()
    fig.canvas.draw()
    strips = _all_colorbars(fig)
    assert len(strips) == 1, f"expected one shared strip, got {len(strips)}"
    rect = strips[0].ax.get_window_extent()
    got = (_mm(fig, rect.width), _mm(fig, rect.height))
    assert got == pytest.approx((4.5, 15.0), abs=1e-6), (
        f"a band must render its own default vertical strip, got {got}"
    )


def test_geometry_key_never_reaches_ax_legend():
    """A colorbar-only key stashed on a categorical entry is still dropped.

    The re-render routes through the same two disjoint filters the plot
    path uses, so ``height`` cannot reach ``Legend.__init__`` (#231's
    trap) even though the record carries it verbatim.
    """
    fig, ax = pp.subplots(1, 1, axes_size=AXES_SIZE)
    pp.scatterplot(data=_df(), x="x", y="y", hue="g", ax=ax,
                   legend_kws={"ncol": 2, "height": 8, "width": 30})
    pp.legend(ax)
    fig.canvas.draw()
    legends = _legends(fig)
    assert len(legends) == 1 and legends[0]._ncols == 2


# ---------------------------------------------------------------------------
# #233 — an adopt on an axes a band already covers
# ---------------------------------------------------------------------------


def test_band_then_adopt_leaves_one_colorbar_and_warns():
    """The issue's snippet: 2 colorbars and no warning, now 1 and a warning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig, axes = pp.subplots(1, 2, axes_size=(35, 30))
        for ax in axes.flat:
            pp.scatterplot(data=_df(), x="x", y="y", hue="c", ax=ax)
        pp.legend(anchor=axes[1])
        pp.legend(axes[0], side="left")
        fig.canvas.draw()
        fig.canvas.draw()
    strips = _all_colorbars(fig)
    assert len(strips) == 1, f"expected one colorbar, got {len(strips)}"
    assert len(_overlap(caught)) == 1, [str(w.message) for w in caught]


def test_band_then_adopt_leaves_one_legend_and_warns():
    """Kind-agnostic: the categorical equivalent gave ``['g', 'g']``."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig, axes = pp.subplots(1, 2, axes_size=(35, 30))
        for ax in axes.flat:
            pp.scatterplot(data=_df(), x="x", y="y", hue="g", ax=ax)
        pp.legend(anchor=axes[1])
        pp.legend(axes[0], side="left")
        fig.canvas.draw()
    titles = [lg.get_title().get_text() for lg in _legends(fig)]
    assert titles == ["g"], titles
    assert len(_overlap(caught)) == 1, [str(w.message) for w in caught]


def test_adopt_still_renders_what_the_band_did_not_claim():
    """A ``collect=`` band owns only its names; the rest still adopt.

    Guards the claim filter against becoming "an adopt next to a band
    renders nothing": the band takes ``g``, and ``c`` (continuous size)
    still renders per panel.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig, axes = pp.subplots(1, 2, axes_size=(35, 30))
        for ax in axes.flat:
            pp.scatterplot(data=_df(), x="x", y="y", hue="g", size="c", ax=ax)
        pp.legend(anchor=axes[1], collect=["g"])
        pp.legend(axes[0], side="left")
        fig.canvas.draw()
    titles = sorted(lg.get_title().get_text() for lg in _legends(fig))
    assert titles == ["c", "c", "g"], titles
    assert len(_overlap(caught)) == 1, [str(w.message) for w in caught]


def test_plain_adopt_does_not_warn():
    """No other group, no overlap, no warning.

    The warning has to be driven by a measured collision — a blanket
    "you adopted next to something" would fire here and make the signal
    worthless.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig, ax = pp.subplots(1, 1, axes_size=AXES_SIZE)
        pp.scatterplot(data=_df(), x="x", y="y", hue="g", ax=ax)
        pp.legend(ax, side="left")
        fig.canvas.draw()
    assert _overlap(caught) == [], [str(w.message) for w in caught]
    assert len(_legends(fig)) == 1


def test_two_adopts_on_different_axes_do_not_warn():
    """Per-axes legends on separate panels never collide."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig, axes = pp.subplots(1, 2, axes_size=(35, 30))
        for ax in axes.flat:
            pp.scatterplot(data=_df(), x="x", y="y", hue="g", ax=ax)
        pp.legend(axes[0], side="left")
        pp.legend(axes[1], side="right")
        fig.canvas.draw()
    assert _overlap(caught) == [], [str(w.message) for w in caught]
    assert len(_legends(fig)) == 2


def test_adopt_then_band_still_leaves_one():
    """The other ordering keeps working — the band evicts, and stays silent."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig, axes = pp.subplots(1, 2, axes_size=(35, 30))
        for ax in axes.flat:
            pp.scatterplot(data=_df(), x="x", y="y", hue="g", ax=ax)
        pp.legend(axes[0], side="left")
        pp.legend(anchor=axes[1])
        fig.canvas.draw()
    titles = [lg.get_title().get_text() for lg in _legends(fig)]
    assert titles == ["g"], titles
    assert _overlap(caught) == [], [str(w.message) for w in caught]


def test_figure_band_then_adopt_leaves_one():
    """A figure-level band covers every cell, so the adopt yields to it."""
    fig, axes = pp.subplots(1, 2, axes_size=(35, 30))
    for ax in axes.flat:
        pp.scatterplot(data=_df(), x="x", y="y", hue="g", ax=ax)
    pp.legend(side="right")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pp.legend(axes[0], side="left")
        fig.canvas.draw()
    titles = [lg.get_title().get_text() for lg in _legends(fig)]
    assert titles == ["g"], titles
    assert len(_overlap(caught)) == 1, [str(w.message) for w in caught]


def test_band_created_before_the_plots_then_adopt():
    """The band-first-before-plotting ordering, where no cache ever forms.

    The entries are claimed by the time ``render_entries`` runs, so it
    returns before creating a per-axes group and ``pp.legend(axes[0])``
    constructs a fresh one instead of adopting. Both paths have to
    decline, which is why the check lives in ``_materialize``.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig, axes = pp.subplots(1, 2, axes_size=(35, 30))
        pp.legend(anchor=axes[1])
        for ax in axes.flat:
            pp.scatterplot(data=_df(), x="x", y="y", hue="g", ax=ax)
        pp.legend(axes[0], side="left")
        fig.canvas.draw()
        fig.canvas.draw()
    titles = [lg.get_title().get_text() for lg in _legends(fig)]
    assert titles == ["g"], titles
    assert len(_overlap(caught)) == 1, [str(w.message) for w in caught]


# ---------------------------------------------------------------------------
# convergence
# ---------------------------------------------------------------------------


def _snapshot(fig, group):
    """Figure size in mm plus every element's rendered rect, rounded."""
    rects = []
    for _kind, artist in group._builder.elements:
        target = getattr(artist, "ax", artist)
        rect = target.get_window_extent()
        rects.append(tuple(round(v, 6) for v in
                           (rect.x0, rect.y0, rect.width, rect.height)))
    size = tuple(round(v * 25.4, 6) for v in fig.get_size_inches())
    return size, tuple(rects)


@pytest.mark.parametrize("hue,kws", [
    ("c", {"side": "top", "width": 30, "height": 8}),
    ("g", {"side": "top", "ncol": 3}),
])
def test_adopted_geometry_converges(tmp_path, hue, kws):
    """Eight draws, then ``settle()``, then PNG and PDF — all identical.

    The re-render happens inside ``_reconfigure_for_adopt``, before the
    layout has settled around the new artist, so a size that keeps
    creeping would surface here. ``savefig`` renders at its own dpi and
    the extents come from text metrics, so it is a separate risk from
    the draw loop.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig, ax = pp.subplots(1, 1, axes_size=AXES_SIZE)
        pp.scatterplot(data=_df(), x="x", y="y", hue=hue, ax=ax,
                       legend_kws=dict(kws))
        group = pp.legend(ax, side="top")
        shots = []
        for _ in range(8):
            fig.canvas.draw()
            shots.append(_snapshot(fig, group))
        fig._publiplots_auto_layout.settle()
        shots.append(_snapshot(fig, group))
        plt.figure(fig.number)
        pp.savefig(str(tmp_path / "adopt.png"))
        shots.append(_snapshot(fig, group))
        plt.figure(fig.number)
        pp.savefig(str(tmp_path / "adopt.pdf"))
        shots.append(_snapshot(fig, group))
    assert len(set(shots[1:])) == 1, shots
    convergence = [w for w in caught
                   if type(w.message).__name__ == "LayoutConvergenceWarning"]
    assert convergence == [], [str(w.message) for w in convergence]
