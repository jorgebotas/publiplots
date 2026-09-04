"""``legend_kws`` reaches ``add_colorbar``'s geometry parameters (#231).

``legend_kws={'width': 30, 'height': 8}`` on a continuous hue used to
render the default 15 x 4.5mm strip and drop both arguments without a
word, while the same arguments through the group API
(``g.add_colorbar(width=30, height=20)``) worked.

The keys were dropped one filter earlier than the issue first reported:
not by ``_colorbar_kwargs``, but by ``_builder_kwargs``, because
``height``/``width`` are not in ``_BUILDER_FORWARD_KEYS`` at all.

That set cannot simply be widened. ``_BUILDER_FORWARD_KEYS`` is
forwarded verbatim to ``add_legend`` -> ``ax.legend()``, which takes no
``height``/``width`` but does take ``**kwargs``, so a geometry key added
there raises ``Legend.__init__() got an unexpected keyword argument
'height'`` on every *categorical* legend — the mirror image of #215,
where legend-only keys reaching ``Colorbar.__init__`` raised on every
continuous one. So ``_COLORBAR_FORWARD_KEYS`` is now applied to
``legend_kws`` directly and is deliberately DISJOINT from the legend
passthrough set rather than a subset of it.

Four keys are forwarded — ``height``, ``width``, ``orientation``,
``ticks``. ``vmin``/``vmax``/``center``/``cmap``/``label`` are not: the
plot call derives each from the data it just drew, so a second value
from ``legend_kws`` would put two sources of truth behind one strip.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import publiplots as pp
from publiplots.utils.plot_legend import (
    _BUILDER_FORWARD_KEYS,
    _COLORBAR_FORWARD_KEYS,
    _colorbar_kwargs,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


AXES_SIZE = (50, 40)


def _df(n=40, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "x": rng.normal(size=n),
        "y": rng.normal(size=n),
        "c": np.linspace(0.0, 1.0, n),
        "g": np.tile(["A", "B"], n // 2),
    })


def _matrix():
    return pd.DataFrame(
        np.arange(20).reshape(4, 5).astype(float),
        index=list("abcd"), columns=list("vwxyz"),
    )


def _mm(fig, px):
    return px / fig.dpi * 25.4


def _strips(ax):
    """Every colorbar rendered per-axes on ``ax``.

    Covers both render paths: the cached ``collect=[]`` group and the
    ``inside=True`` short-circuit. An inside colorbar is an
    ``ax.inset_axes`` child, so it never appears in ``fig.get_axes()``.
    """
    out = []
    for builder in getattr(ax, "_publiplots_legend_builders", []):
        out += [obj for kind, obj in builder.elements if kind == "colorbar"]
    return out


def _only_strip(ax):
    strips = _strips(ax)
    assert len(strips) == 1, f"expected one strip, got {len(strips)}"
    return strips[0]


def _render(legend_kws, hue="c", kind="scatter", axes_size=AXES_SIZE):
    """Draw one plot call and return ``(fig, ax)``, drawn twice.

    Twice because matplotlib's xlabel position lags a resize by one
    draw, and a bottom/left band shifts the axes origin on the first.
    """
    fig, ax = pp.subplots(1, 1, axes_size=axes_size)
    if kind == "scatter":
        pp.scatterplot(data=_df(), x="x", y="y", hue=hue, ax=ax,
                       legend_kws=dict(legend_kws))
    elif kind == "heatmap":
        pp.heatmap(data=_matrix(), ax=ax, legend_kws=dict(legend_kws))
    else:  # pragma: no cover - guard
        raise AssertionError(kind)
    fig.canvas.draw()
    fig.canvas.draw()
    return fig, ax


def _strip_mm(legend_kws, **kw):
    """(width_mm, height_mm) of the strip a single plot call renders."""
    fig, ax = _render(legend_kws, **kw)
    r = _only_strip(ax).ax.get_window_extent()
    return _mm(fig, r.width), _mm(fig, r.height)


# --- the reported bug -------------------------------------------------------

# The defaults the strip falls back to when geometry is dropped, so a
# regression reads as "you got the default" rather than a bare mismatch.
_DEFAULT_HORIZONTAL = (15.0, 4.5)


def test_height_alone_is_honoured():
    w, h = _strip_mm({"side": "top", "height": 8})
    assert h == pytest.approx(8.0, abs=1e-6), (
        f"height=8 gave {w:.2f} x {h:.2f}mm; "
        f"default is {_DEFAULT_HORIZONTAL[1]}mm (#231)"
    )
    # width untouched keeps its per-orientation default.
    assert w == pytest.approx(_DEFAULT_HORIZONTAL[0], abs=1e-6)


def test_width_alone_is_honoured():
    w, h = _strip_mm({"side": "top", "width": 30})
    assert w == pytest.approx(30.0, abs=1e-6), (
        f"width=30 gave {w:.2f} x {h:.2f}mm; "
        f"default is {_DEFAULT_HORIZONTAL[0]}mm (#231)"
    )
    assert h == pytest.approx(_DEFAULT_HORIZONTAL[1], abs=1e-6)


def test_width_and_height_together():
    w, h = _strip_mm({"side": "top", "width": 30, "height": 8})
    assert (w, h) == pytest.approx((30.0, 8.0), abs=1e-6), (
        f"got {w:.2f} x {h:.2f}mm, expected 30.00 x 8.00mm (#231)"
    )


def test_all_four_forwarded_keys_together():
    kws = {"side": "top", "width": 30, "height": 8,
           "orientation": "horizontal", "ticks": [0.0, 0.5, 1.0]}
    fig, ax = _render(kws)
    cbar = _only_strip(ax)
    r = cbar.ax.get_window_extent()
    assert (_mm(fig, r.width), _mm(fig, r.height)) == pytest.approx(
        (30.0, 8.0), abs=1e-6)
    assert cbar.orientation == "horizontal"
    assert list(cbar.get_ticks()) == pytest.approx([0.0, 0.5, 1.0])


# ``height`` is the vertical extent and ``width`` the horizontal one at
# every orientation, so the same numbers are expected on every side.
@pytest.mark.parametrize("side", ["top", "bottom", "left", "right"])
def test_geometry_is_honoured_on_every_side(side):
    w, h = _strip_mm({"side": side, "width": 9, "height": 26})
    assert (w, h) == pytest.approx((9.0, 26.0), abs=1e-6), (
        f"side={side!r}: got {w:.2f} x {h:.2f}mm, expected 9.00 x 26.00mm"
    )


def test_geometry_is_honoured_inside_the_axes():
    """The ``inside=True`` path renders through a bare ``LegendBuilder``,
    not a band, and must honour mm geometry there too."""
    w, h = _strip_mm({"inside": True, "loc": "upper right",
                      "width": 30, "height": 8})
    assert (w, h) == pytest.approx((30.0, 8.0), abs=1e-6), (
        f"inside=True: got {w:.2f} x {h:.2f}mm, expected 30.00 x 8.00mm"
    )


def test_ticks_replace_the_derived_ones():
    """A tick list the default would never produce, so the assertion
    cannot pass by coincidence (the derived ticks are 0/0.5/1)."""
    fig, ax = _render({"side": "top", "ticks": [0.0, 0.25, 1.0]})
    assert list(_only_strip(ax).get_ticks()) == pytest.approx([0.0, 0.25, 1.0])


def test_heatmap_colorbar_honours_geometry():
    w, h = _strip_mm({"side": "top", "width": 30, "height": 8},
                     kind="heatmap")
    assert (w, h) == pytest.approx((30.0, 8.0), abs=1e-6)


# --- the constraint: a categorical legend must not see these keys ----------


_GEOMETRY_KEYS = frozenset({"height", "width", "ticks"})


def test_geometry_keys_never_reach_ax_legend():
    """Structural guard on the mirror image of #215.

    ``add_legend`` forwards ``_BUILDER_FORWARD_KEYS`` straight to
    ``ax.legend()``, which raises on any of these. Keeping the two sets
    disjoint is what makes the categorical path below safe.
    """
    overlap = _GEOMETRY_KEYS & _BUILDER_FORWARD_KEYS
    assert not overlap, (
        f"{sorted(overlap)} would be forwarded to ax.legend() and raise "
        "TypeError on every categorical legend (#231)"
    )


def test_colorbar_key_set_is_not_a_subset_of_the_legend_set():
    """The structural point of the fix: the colorbar path owns its own
    key set, applied to ``legend_kws`` directly. Re-subsetting it under
    ``_BUILDER_FORWARD_KEYS`` is what left no room for ``height``."""
    assert not _COLORBAR_FORWARD_KEYS <= _BUILDER_FORWARD_KEYS, (
        "_COLORBAR_FORWARD_KEYS became a subset of _BUILDER_FORWARD_KEYS "
        "again; a colorbar-only geometry key cannot survive there (#231)"
    )


def test_forwarded_key_set_is_exactly_the_agreed_four_plus_placement():
    """Pins the scope decision.

    ``vmin``/``vmax``/``center``/``cmap``/``label`` are each derived by
    the plot call from the data it just drew; forwarding them would put
    two sources of truth behind one strip (the failure class behind
    #221, #230 and #243). ``label`` would additionally collide with the
    ``label=entry.name`` the render path already passes.
    """
    assert _COLORBAR_FORWARD_KEYS == frozenset({
        "inside", "loc", "height", "width", "orientation", "ticks",
    })


@pytest.mark.parametrize("key", ["vmin", "vmax", "center", "cmap", "label"])
def test_data_derived_keys_are_not_forwarded(key):
    assert key not in _COLORBAR_FORWARD_KEYS, (
        f"{key!r} is derived from the data by the plot call; forwarding it "
        "from legend_kws invites a silent conflict (#231)"
    )


def test_categorical_hue_accepts_the_same_legend_kws():
    """The regression this fix must not cause: the identical
    ``legend_kws`` on a *categorical* hue has to keep working, because
    those keys must never reach ``ax.legend()``."""
    fig, ax = _render(
        {"side": "top", "width": 30, "height": 8,
         "orientation": "horizontal", "ticks": [0.0, 0.5, 1.0]},
        hue="g",
    )
    assert _strips(ax) == [], "a categorical hue must not render a colorbar"


def test_mixed_continuous_and_categorical_hue_on_one_axes():
    """One continuous and one categorical hue on the same axes: the
    strip takes the geometry, the categorical legend ignores it without
    raising."""
    fig, ax = pp.subplots(1, 1, axes_size=(60, 45))
    kws = {"side": "top", "width": 30, "height": 8,
           "orientation": "horizontal", "ticks": [0.0, 0.5, 1.0]}
    d = _df()
    pp.scatterplot(data=d, x="x", y="y", hue="c", ax=ax, legend_kws=dict(kws))
    pp.scatterplot(data=d, x="x", y="y", hue="g", ax=ax, legend_kws=dict(kws))
    fig.canvas.draw()
    fig.canvas.draw()
    cbar = _only_strip(ax)
    r = cbar.ax.get_window_extent()
    assert (_mm(fig, r.width), _mm(fig, r.height)) == pytest.approx(
        (30.0, 8.0), abs=1e-6)
    assert cbar.orientation == "horizontal"


# --- #213's per-side derivation must survive the forwarding ---------------


_DERIVED = {
    "top": ("horizontal", 15.0, 4.5),
    "bottom": ("horizontal", 15.0, 4.5),
    "left": ("vertical", 4.5, 15.0),
    "right": ("vertical", 4.5, 15.0),
}


@pytest.mark.parametrize("side", ["top", "bottom", "left", "right"])
def test_omitted_orientation_still_derives_from_the_side(side):
    """``orientation`` absent from ``legend_kws`` must stay absent from
    the ``add_colorbar`` call, so #213's derivation still runs."""
    expected_orientation, exp_w, exp_h = _DERIVED[side]
    fig, ax = _render({"side": side})
    cbar = _only_strip(ax)
    r = cbar.ax.get_window_extent()
    assert cbar.orientation == expected_orientation, (
        f"side={side!r}: derived {cbar.orientation!r}, "
        f"expected {expected_orientation!r} (#213)"
    )
    assert (_mm(fig, r.width), _mm(fig, r.height)) == pytest.approx(
        (exp_w, exp_h), abs=1e-6)
    assert "orientation" not in _colorbar_kwargs({"side": side})


def test_explicit_vertical_wins_on_a_top_band():
    fig, ax = _render({"side": "top", "orientation": "vertical"})
    cbar = _only_strip(ax)
    r = cbar.ax.get_window_extent()
    assert cbar.orientation == "vertical", (
        "an explicit orientation must beat the top band's derivation (#213)"
    )
    assert (_mm(fig, r.width), _mm(fig, r.height)) == pytest.approx(
        (4.5, 15.0), abs=1e-6)


def test_explicit_horizontal_wins_on_a_right_band():
    fig, ax = _render({"side": "right", "orientation": "horizontal"})
    cbar = _only_strip(ax)
    r = cbar.ax.get_window_extent()
    assert cbar.orientation == "horizontal", (
        "an explicit orientation must beat the right band's derivation (#213)"
    )
    assert (_mm(fig, r.width), _mm(fig, r.height)) == pytest.approx(
        (15.0, 4.5), abs=1e-6)


def test_orientation_reaches_the_inside_path():
    """The case that pins ``orientation`` in the forwarded set.

    On an outside band ``orientation`` also travels as a
    ``_GROUP_PLACEMENT_KEYS`` key, so the band's own orientation would
    carry it even if ``add_colorbar`` never saw it. ``inside=True``
    builds no group, so here the value can only arrive through
    ``_colorbar_kwargs``.
    """
    fig, ax = _render({"inside": True, "loc": "upper right",
                       "orientation": "horizontal"})
    assert _only_strip(ax).orientation == "horizontal", (
        "orientation must be forwarded to add_colorbar, not only to the "
        "per-axes group (#231)"
    )


def test_colorbar_kwargs_passes_orientation_through_verbatim():
    assert _colorbar_kwargs(
        {"side": "top", "orientation": "vertical"}
    )["orientation"] == "vertical"


# --- convergence -----------------------------------------------------------


_CONVERGENCE_CASES = [
    {"side": "top", "width": 30, "height": 8},
    {"side": "bottom", "width": 30, "height": 8},
    {"side": "left", "width": 8, "height": 30},
    {"side": "right", "width": 8, "height": 30},
    {"side": "top", "width": 30, "height": 8,
     "orientation": "horizontal", "ticks": [0.0, 0.5, 1.0]},
    {"inside": True, "loc": "upper right", "width": 30, "height": 8},
]


@pytest.mark.parametrize(
    "legend_kws", _CONVERGENCE_CASES,
    ids=lambda k: "-".join(f"{a}{b}" for a, b in sorted(k.items())),
)
def test_geometry_converges_and_saves(legend_kws, tmp_path, recwarn):
    """Repeated draws plus ``settle()`` must not move the figure or the
    strip, and neither must a save.

    Compared from the second draw on: a bottom or left band shifts the
    axes origin on the first draw, which is the reactor settling, not
    drift. Pre-existing — it happens with the default geometry too.
    """
    fig, ax = _render(legend_kws)

    def sample():
        r = _only_strip(ax).ax.get_window_extent()
        return (
            tuple(fig.get_size_inches()),
            (round(r.width, 6), round(r.height, 6)),
            (round(r.x0, 6), round(r.y0, 6)),
        )

    first = sample()
    for i in range(8):
        fig.canvas.draw()
        assert sample() == first, f"draw {i + 2} moved the layout"
    fig._publiplots_auto_layout.settle()
    fig.canvas.draw()
    assert sample() == first, "settle() moved the layout"

    plt.figure(fig.number)
    for ext in ("png", "pdf"):
        pp.savefig(str(tmp_path / f"strip.{ext}"))
        assert (tmp_path / f"strip.{ext}").stat().st_size > 0

    convergence = [
        w for w in recwarn.list
        if issubclass(w.category, pp.LayoutConvergenceWarning)
    ]
    assert not convergence, (
        f"{len(convergence)} convergence warning(s): "
        f"{[str(w.message) for w in convergence]}"
    )


# --- what adoption does NOT carry over -------------------------------------


def test_pp_legend_adoption_discards_the_plot_calls_geometry():
    """``pp.legend(ax)`` re-renders from the stash and drops ``legend_kws``.

    Pinning current behaviour, not asserting it is right. A claimed entry
    is re-rendered by ``MultiAxesLegendGroup._render_entry`` as
    ``add_colorbar(mappable=..., label=entry.name)`` with no geometry, so
    the strip this change sized is destroyed and replaced by a default.

    Deliberately NOT fixed here, because it is not specific to colorbars:
    the categorical branch of that same method passes no ``ncol`` either,
    and a legend built with ``legend_kws={'ncol': 3}`` comes back with
    ``ncol`` reset to the label count through the same call — measured 3
    before, 6 after. The adopt path discards every forwarded key for both
    entry kinds, so it is one general defect rather than a hole in this
    one, and it is filed separately.

    If it is ever fixed this test will fail on the second assertion. The
    right response then is to delete the test, not to restore the
    behaviour.
    """
    fig, ax = _render({"side": "top", "width": 30, "height": 8})
    r = _only_strip(ax).ax.get_window_extent()
    before = (_mm(fig, r.width), _mm(fig, r.height))
    assert before == pytest.approx((30.0, 8.0), abs=1e-6), (
        f"setup: the plot call should have sized the strip, got {before}"
    )

    pp.legend(ax, side="top")
    fig.canvas.draw()
    fig.canvas.draw()
    r = _only_strip(ax).ax.get_window_extent()
    after = (_mm(fig, r.width), _mm(fig, r.height))
    assert after == pytest.approx(_DEFAULT_HORIZONTAL, abs=1e-6), (
        "adoption is expected to drop the geometry and re-render a default "
        f"strip; got {after}. If this now preserves 30 x 8, the adopt path "
        "was fixed — delete this test rather than reverting that."
    )


def test_categorical_adoption_drops_its_kwargs_too():
    """The companion measurement that makes the above a general defect.

    Recorded because the asymmetry it rules out was the reason to consider
    fixing the adopt path inside this change: if only the colorbar branch
    lost its arguments, that would be a gap in #231. Both branches lose
    them, so it is not.
    """
    frame = _df(n=60)
    frame["g"] = np.tile(list("ABCDEF"), 10)
    fig, ax = pp.subplots(1, 1, axes_size=AXES_SIZE)
    pp.scatterplot(data=frame, x="x", y="y", hue="g", ax=ax,
                   legend_kws={"side": "top", "ncol": 3})
    fig.canvas.draw()
    legends = [c for a in fig.get_axes() for c in a.get_children()
               if type(c).__name__ == "Legend"]
    assert len(legends) == 1 and legends[0]._ncols == 3

    pp.legend(ax, side="top")
    fig.canvas.draw()
    legends = [c for a in fig.get_axes() for c in a.get_children()
               if type(c).__name__ == "Legend"]
    assert len(legends) == 1
    assert legends[0]._ncols != 3, (
        "if categorical adoption now preserves ncol, the two entry kinds no "
        "longer agree and the colorbar branch of _render_entry is a real gap"
    )
