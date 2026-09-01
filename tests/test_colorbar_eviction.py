"""Band eviction of already-rendered per-axes colorbars (#217).

``pp.legend(...)`` supports the "plot first, then legend" ordering by
evicting the per-axes legend artists it is about to render itself. That
sweep used to walk ``ax.get_children()`` for ``Legend`` instances only,
which a colorbar never is: an outside per-axes colorbar is a
``fig.add_axes`` strip plus a free-standing ``fig.text`` label, and an
``inside=True`` one is an ``ax.inset_axes`` child. So a continuous hue
claimed by the band rendered twice — once beside every panel, once in
the band — and the orphan strips kept their ``LayoutReactor``
registrations, which kept the panels' colorbar-shaped gap reserved.

The reference geometry throughout is the *legend-first* ordering
(``pp.legend(...)`` before the plot calls), which never had the bug.
"""
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


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    n = 60
    return pd.DataFrame({
        "x": rng.normal(size=n),
        "y": rng.normal(size=n),
        "c": rng.normal(size=n),          # continuous -> colorbar
        "d": rng.normal(size=n) * 3.0,    # a second, unrelated continuous
        "g": rng.choice(["A", "B"], n),   # categorical -> legend
    })


# Panels that get a plot; (1, 1) stays empty and hosts the band.
CONT_CELLS = [(0, 0), (0, 1), (1, 0)]


# ---------------------------------------------------------------- helpers

def _all_axes(fig):
    """Every Axes reachable from ``fig``, inset children included.

    An ``inside=True`` colorbar is created with ``ax.inset_axes``, which
    registers the strip in the parent's ``child_axes`` and NOT in
    ``fig.get_axes()`` — so a sweep over ``fig.get_axes()`` alone would
    silently miss exactly the artists these tests are about.
    """
    seen, out = set(), []

    def walk(ax):
        if id(ax) in seen:
            return
        seen.add(id(ax))
        out.append(ax)
        for child in getattr(ax, "child_axes", None) or ():
            walk(child)

    for ax in fig.get_axes():
        walk(ax)
    return out


def _colorbars(fig):
    """Every live Colorbar on the figure.

    ``Colorbar.__init__`` stamps itself onto its own strip axes as
    ``ax._colorbar``, so a strip that is still attached to the figure is
    exactly a reachable Axes carrying that attribute.
    """
    return [
        ax._colorbar for ax in _all_axes(fig)
        if getattr(ax, "_colorbar", None) is not None
    ]


def _floating_labels(fig):
    """Texts of the free-standing ``fig.text`` colorbar labels."""
    return sorted(t.get_text() for t in fig.texts)


def _legend_titles(fig):
    from matplotlib.legend import Legend
    return sorted(
        child.get_title().get_text()
        for ax in _all_axes(fig)
        for child in ax.get_children()
        if isinstance(child, Legend)
    )


def _registrations(fig):
    reactor = getattr(fig, "_publiplots_layout_reactor", None)
    return list(reactor._registrations) if reactor is not None else []


def _artist_is_attached(fig, artist):
    """True if a reactor-registered artist is still part of the figure."""
    from matplotlib.text import Text
    if getattr(artist, "ax", None) is not None and hasattr(artist, "mappable"):
        return any(artist.ax is ax for ax in _all_axes(fig))
    if isinstance(artist, Text):
        return any(artist is t for t in fig.texts)
    return any(
        any(artist is child for child in ax.get_children())
        for ax in _all_axes(fig)
    )


def _panels_mm(fig, axes):
    """Every panel's (x0, y0, w, h) rectangle in millimetres."""
    fig_w_mm, fig_h_mm = (v * 25.4 for v in fig.get_size_inches())
    out = []
    for row in axes:
        for ax in np.atleast_1d(row):
            pos = ax.get_position()
            out.append((
                round(pos.x0 * fig_w_mm, 3),
                round(pos.y0 * fig_h_mm, 3),
                round(pos.width * fig_w_mm, 3),
                round(pos.height * fig_h_mm, 3),
            ))
    return out


def _figure_mm(fig):
    return tuple(round(v * 25.4, 3) for v in fig.get_size_inches())


# ---------------------------------------------------------------- builders

def _build(df, *, legend_first, band_kws=None, cells=None, hue="c"):
    """2x2 grid, continuous hue on ``cells``, one band anchored at (1, 1)."""
    cells = CONT_CELLS if cells is None else cells
    fig, axes = pp.subplots(2, 2, axes_size=(35, 30))

    def plots():
        for r, c in cells:
            pp.scatterplot(data=df, x="x", y="y", hue=hue, ax=axes[r, c])

    def band():
        pp.legend(anchor=axes[1, 1], **(band_kws or {}))

    if legend_first:
        band()
        plots()
    else:
        plots()
        band()
    fig.canvas.draw()
    return fig, axes


# ------------------------------------------------------------------- tests

@pytest.mark.parametrize("band_kws", [
    pytest.param({"inside": True}, id="inside"),
    pytest.param({}, id="outside-right"),
    pytest.param({"side": "top"}, id="outside-top"),
    pytest.param({"side": "bottom"}, id="outside-bottom"),
])
def test_claimed_continuous_hue_renders_exactly_one_colorbar(df, band_kws):
    """Three panels + a band claiming their shared hue == one strip total."""
    fig, axes = _build(df, legend_first=False, band_kws=band_kws)
    assert len(_colorbars(fig)) == 1, (
        f"expected 1 colorbar, found {len(_colorbars(fig))} "
        "(the band's copy plus orphan per-axes strips)"
    )


@pytest.mark.parametrize("band_kws", [
    pytest.param({"inside": True}, id="inside"),
    pytest.param({}, id="outside-right"),
    pytest.param({"side": "top"}, id="outside-top"),
    pytest.param({"side": "bottom"}, id="outside-bottom"),
])
def test_no_orphan_colorbar_labels(df, band_kws):
    """The floating ``fig.text`` label goes with its evicted strip.

    A label with no strip is the same duplicate in a different costume,
    and it lives on the *figure*, so it never showed up in the
    ``ax.get_children()`` sweep either.
    """
    fig, axes = _build(df, legend_first=False, band_kws=band_kws)
    reference, _ = _build(df, legend_first=True, band_kws=band_kws)
    assert _floating_labels(fig) == _floating_labels(reference)


@pytest.mark.parametrize("band_kws", [
    pytest.param({"inside": True}, id="inside"),
    pytest.param({}, id="outside-right"),
    pytest.param({"side": "top"}, id="outside-top"),
    pytest.param({"side": "bottom"}, id="outside-bottom"),
])
def test_evicted_colorbars_leave_no_reactor_registrations(df, band_kws):
    """Eviction unregisters, it does not just detach.

    A surviving registration keeps ``LayoutReactor`` repositioning a
    removed artist every draw and keeps ``SubplotsAutoLayout`` reserving
    its footprint, so the count must land on the legend-first
    ordering's — which only ever registers the band's own elements.
    """
    fig, axes = _build(df, legend_first=False, band_kws=band_kws)
    reference, _ = _build(df, legend_first=True, band_kws=band_kws)
    assert len(_registrations(fig)) == len(_registrations(reference))


@pytest.mark.parametrize("band_kws", [
    pytest.param({"inside": True}, id="inside"),
    pytest.param({}, id="outside-right"),
    pytest.param({"side": "top"}, id="outside-top"),
    pytest.param({"side": "bottom"}, id="outside-bottom"),
])
def test_no_registration_points_at_a_removed_artist(df, band_kws):
    """Every reactor registration still resolves to a live artist."""
    fig, axes = _build(df, legend_first=False, band_kws=band_kws)
    stale = [
        reg for reg in _registrations(fig)
        if not _artist_is_attached(fig, reg.artist)
    ]
    assert stale == [], (
        f"{len(stale)} reactor registration(s) point at removed artists"
    )


@pytest.mark.parametrize("band_kws", [
    pytest.param({"inside": True}, id="inside"),
    pytest.param({}, id="outside-right"),
    pytest.param({"side": "top"}, id="outside-top"),
    pytest.param({"side": "bottom"}, id="outside-bottom"),
])
def test_eviction_reclaims_the_panel_reservation(df, band_kws):
    """Panel geometry must match the ordering that never had the bug.

    The evicted strips had grown their cells' per-column reservation via
    the reactor; dropping the artists without the registrations would
    leave every panel with a colorbar-shaped gap beside it.
    """
    fig, axes = _build(df, legend_first=False, band_kws=band_kws)
    reference, ref_axes = _build(df, legend_first=True, band_kws=band_kws)
    assert _figure_mm(fig) == _figure_mm(reference)
    assert _panels_mm(fig, axes) == _panels_mm(reference, ref_axes)


def test_mixed_continuous_and_categorical_panels(df):
    """One band claiming a colorbar entry and a legend entry renders one of each."""
    fig, axes = pp.subplots(2, 2, axes_size=(35, 30))
    pp.scatterplot(data=df, x="x", y="y", hue="c", ax=axes[0, 0])
    pp.scatterplot(data=df, x="x", y="y", hue="c", ax=axes[0, 1])
    pp.scatterplot(data=df, x="x", y="y", hue="g", ax=axes[1, 0])
    pp.legend(anchor=axes[1, 1])
    fig.canvas.draw()

    assert len(_colorbars(fig)) == 1
    assert _legend_titles(fig) == ["g"]

    reference, ref_axes = pp.subplots(2, 2, axes_size=(35, 30))
    pp.legend(anchor=ref_axes[1, 1])
    pp.scatterplot(data=df, x="x", y="y", hue="c", ax=ref_axes[0, 0])
    pp.scatterplot(data=df, x="x", y="y", hue="c", ax=ref_axes[0, 1])
    pp.scatterplot(data=df, x="x", y="y", hue="g", ax=ref_axes[1, 0])
    reference.canvas.draw()
    assert _panels_mm(fig, axes) == _panels_mm(reference, ref_axes)


def test_unclaimed_colorbar_survives(df):
    """``collect=['c']`` must not touch a panel's colorbar for ``'d'``."""
    fig, axes = pp.subplots(2, 2, axes_size=(35, 30))
    pp.scatterplot(data=df, x="x", y="y", hue="c", ax=axes[0, 0])
    pp.scatterplot(data=df, x="x", y="y", hue="c", ax=axes[0, 1])
    pp.scatterplot(data=df, x="x", y="y", hue="d", ax=axes[1, 0])
    pp.legend(anchor=axes[1, 1], collect=["c"])
    fig.canvas.draw()

    # The band's 'c' strip plus the untouched 'd' strip — and nothing else.
    assert len(_colorbars(fig)) == 2
    assert _floating_labels(fig) == ["c", "d"]
    # 'd' keeps its strip AND its label AND both registrations.
    assert len(_registrations(fig)) == 4


def test_out_of_scope_colorbar_survives(df):
    """A band scoped to the top row leaves the bottom-left panel alone."""
    fig, axes = pp.subplots(2, 2, axes_size=(35, 30))
    for r, c in CONT_CELLS:
        pp.scatterplot(data=df, x="x", y="y", hue="c", ax=axes[r, c])
    pp.legend(ax=[axes[0, 0], axes[0, 1]], anchor=axes[0, 1])
    fig.canvas.draw()

    # The scoped band's strip plus panel (1, 0)'s own, still in place.
    assert len(_colorbars(fig)) == 2
    assert _floating_labels(fig) == ["c", "c"]


def test_inside_per_axes_colorbars_are_evicted(df):
    """``legend_kws={'inside': True}`` strips are inset children, not Legends."""
    fig, axes = pp.subplots(2, 2, axes_size=(35, 30))
    for r, c in CONT_CELLS:
        pp.scatterplot(
            data=df, x="x", y="y", hue="c", ax=axes[r, c],
            legend_kws={"inside": True},
        )
    pp.legend(anchor=axes[1, 1])
    fig.canvas.draw()

    assert len(_colorbars(fig)) == 1


def test_legend_first_ordering_is_unchanged(df):
    """The ordering that never had the bug still renders exactly one of each.

    Nothing is evicted here — ``entry_is_in_group`` filters the claimed
    entry out before any per-axes colorbar is drawn — so this pins the
    eviction pass to being a no-op on that path.
    """
    fig, axes = _build(df, legend_first=True)
    assert len(_colorbars(fig)) == 1
    assert _floating_labels(fig) == ["c"]
    assert len(_registrations(fig)) == 2


def test_adopted_per_axes_colorbar_is_evicted(df):
    """A ``pp.legend(ax)`` adoption rebuilds the builder; eviction must follow.

    ``_reconfigure_for_adopt`` tears down the plot-created artists and
    swaps in a fresh builder. The axes-side handle has to move with it,
    or a later band inspects a drained builder and the adopted colorbar
    renders alongside the band's.
    """
    fig, axes = pp.subplots(1, 2, axes_size=(35, 30))
    pp.scatterplot(data=df, x="x", y="y", hue="c", ax=axes[0])
    pp.legend(axes[0], side="left")     # adopt the plot-created group
    pp.legend(anchor=axes[1])           # a band claiming the same entry
    fig.canvas.draw()

    assert len(_colorbars(fig)) == 1
    assert _floating_labels(fig) == ["c"]


def test_adoption_without_a_band_is_untouched(df):
    """The plain adopt path still renders its one colorbar."""
    fig, axes = pp.subplots(1, 2, axes_size=(35, 30))
    pp.scatterplot(data=df, x="x", y="y", hue="c", ax=axes[0])
    pp.legend(axes[0], side="left")
    fig.canvas.draw()

    assert len(_colorbars(fig)) == 1
    assert _floating_labels(fig) == ["c"]
    assert len(_registrations(fig)) == 2


def test_heatmap_colorbars_are_evicted(df):
    """Any plot that renders a per-axes colorbar is covered, not just scatter."""
    rng = np.random.default_rng(1)
    matrix = pd.DataFrame(rng.normal(size=(5, 5)))
    fig, axes = pp.subplots(1, 2, axes_size=(35, 30))
    for ax in axes:
        pp.heatmap(data=matrix, ax=ax)
    pp.legend(anchor=axes[1])
    fig.canvas.draw()

    assert len(_colorbars(fig)) == 1
