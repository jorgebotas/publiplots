"""Plots that draw an outline AND a data line must use two different knobs.

`edgewidth` is a stroke that outlines a shape; `lines.linewidth` is a
stroke that IS the data. regplot, residplot, kdeplot and the upset
membership matrix each drew both from one resolved value.
"""
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
    return pd.DataFrame({
        "x": rng.normal(size=120),
        "y": rng.normal(size=120),
        "cat": rng.choice(list("AB"), 120),
        "g": rng.choice(["one", "two"], 120),
    })


def _collection_widths(ax):
    """Marker-edge widths, i.e. the widths of the scatter collections.

    Restricted to ``PathCollection`` on purpose. ``regplot``'s CI band is
    a ``FillBetweenPolyCollection`` whose stroke comes from matplotlib's
    ``patch.linewidth``, not from any publiplots marker-edge knob, so it
    is not part of what this test measures.
    """
    from matplotlib.collections import PathCollection

    out = []
    for c in ax.collections:
        if not isinstance(c, PathCollection):
            continue
        out += [float(v) for v in np.atleast_1d(c.get_linewidths())]
    return out


def _line_widths(ax, min_points=3):
    """Widths of Line2D artists with enough points to be a curve, not a cap."""
    return [
        float(l.get_linewidth())
        for l in ax.lines
        if len(np.asarray(l.get_xdata())) >= min_points
    ]


@pytest.mark.parametrize("fn_name", ["regplot", "residplot"])
def test_scatter_edges_and_fit_line_use_different_knobs(df, fn_name):
    """regplot/residplot draw markers AND a line. The marker edge is an
    outline (edgewidth); the fit line is data (lines.linewidth). One
    parameter cannot mean both."""
    saved_ew = pp.rcParams["edgewidth"]
    saved_lw = pp.rcParams["lines.linewidth"]
    try:
        pp.rcParams["edgewidth"] = 2.5
        pp.rcParams["lines.linewidth"] = 0.5  # deliberately inverted
        fig, ax = pp.subplots(1, 1, axes_size=(40, 30))
        # regplot always draws its regression line; residplot only draws a
        # curve when a smoother is asked for (seaborn sets fit_reg=False
        # otherwise), so exercise its lowess path.
        extra = {"lowess": True} if fn_name == "residplot" else {}
        getattr(pp, fn_name)(data=df, x="x", y="y", ax=ax, **extra)

        edges = _collection_widths(ax)
        assert edges, "expected a scatter collection"
        assert all(w == pytest.approx(2.5) for w in edges), (
            f"{fn_name}: marker edges {edges} should follow edgewidth"
        )

        lines = _line_widths(ax)
        assert lines, "expected a fit line"
        assert all(w == pytest.approx(0.5) for w in lines), (
            f"{fn_name}: fit line {lines} should follow lines.linewidth"
        )
    finally:
        pp.rcParams["edgewidth"] = saved_ew
        pp.rcParams["lines.linewidth"] = saved_lw


def test_upset_matrix_dots_and_connector_use_different_knobs():
    """The membership matrix draws dot outlines (edgewidth) and a
    connector line between them (lines.linewidth).

    Introspection notes, verified against a real upsetplot figure: the
    matrix lives on the only axes carrying both lines and collections.
    Its connector Line2Ds are drawn in the active palette colour; the
    set-separator lines are drawn in the grid colour, so colour
    distinguishes them. Inactive dots are drawn with linewidths=0 by
    design, so only nonzero collection widths are the active dot edges.
    """
    from matplotlib.colors import to_rgba

    saved_ew = pp.rcParams["edgewidth"]
    saved_lw = pp.rcParams["lines.linewidth"]
    try:
        pp.rcParams["edgewidth"] = 2.5
        pp.rcParams["lines.linewidth"] = 0.5
        plt.close("all")
        pp.upsetplot(data={"A": {1, 2, 3, 4}, "B": {3, 4, 5}, "C": {4, 5, 6}})
        fig = plt.gcf()
        fig.canvas.draw()

        matrix_axes = [a for a in fig.get_axes() if a.lines and a.collections]
        assert len(matrix_axes) == 1, "expected exactly one membership matrix axes"
        ax = matrix_axes[0]

        active = to_rgba(pp.rcParams["color"])
        connectors = [
            l for l in ax.lines
            if to_rgba(l.get_color()) == active
            and len(np.asarray(l.get_xdata())) >= 2
        ]
        assert connectors, "expected connector lines in the active colour"
        for l in connectors:
            assert float(l.get_linewidth()) == pytest.approx(0.5), (
                "matrix connector is a data line -> lines.linewidth"
            )

        dot_edges = [w for w in _collection_widths(ax) if w > 0]
        assert dot_edges, "expected active dot edges with nonzero width"
        for w in dot_edges:
            assert w == pytest.approx(2.5), "dot outline -> edgewidth"
    finally:
        pp.rcParams["edgewidth"] = saved_ew
        pp.rcParams["lines.linewidth"] = saved_lw


@pytest.mark.parametrize("fn_name", ["lineplot", "kdeplot"])
def test_data_lines_ignore_edgewidth(df, fn_name):
    """Cranking edgewidth must not thicken a data line."""
    saved_ew = pp.rcParams["edgewidth"]
    try:
        pp.rcParams["edgewidth"] = 5.0
        fig, ax = pp.subplots(1, 1, axes_size=(40, 30))
        if fn_name == "lineplot":
            pp.lineplot(data=df.sort_values("x"), x="x", y="y", ax=ax)
        else:
            pp.kdeplot(data=df, x="y", ax=ax)
        lines = _line_widths(ax)
        assert lines, "expected a data line"
        assert all(w == pytest.approx(pp.rcParams["lines.linewidth"]) for w in lines)
    finally:
        pp.rcParams["edgewidth"] = saved_ew


def _fill_collection_widths(ax):
    """Widths of the filled-density polygons.

    ``kdeplot(fill=True)`` emits ``FillBetweenPolyCollection`` (a
    ``PolyCollection`` subclass); the 2D contour path emits
    ``QuadContourSet``, which is not one, so this picks out the 1D fills.
    """
    from matplotlib.collections import PolyCollection

    out = []
    for c in ax.collections:
        if not isinstance(c, PolyCollection):
            continue
        out += [float(v) for v in np.atleast_1d(c.get_linewidths())]
    return out


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fill": True},
        {"fill": True, "hue": "g"},
        {"hue": "g", "multiple": "stack"},
        {"hue": "g", "multiple": "fill"},
    ],
    ids=["fill", "fill+hue", "stack", "fill-multiple"],
)
def test_kdeplot_fill_outline_uses_edgewidth(df, kwargs):
    """A filled density's opaque edge outlines a shape -> edgewidth.

    This is the same classification histplot(element='step', fill=True)
    and violinplot already use for their KDE-shaped outlines; kdeplot
    was the last filled shape still riding lines.linewidth.
    ``multiple='stack'``/``'fill'`` are covered because seaborn fills
    implicitly there even with ``fill`` unset.
    """
    saved_ew = pp.rcParams["edgewidth"]
    saved_lw = pp.rcParams["lines.linewidth"]
    try:
        pp.rcParams["edgewidth"] = 2.5
        pp.rcParams["lines.linewidth"] = 0.5  # deliberately inverted
        fig, ax = pp.subplots(1, 1, axes_size=(40, 30))
        pp.kdeplot(data=df, x="x", ax=ax, **kwargs)

        fills = _fill_collection_widths(ax)
        assert fills, "expected a filled density collection"
        assert all(w == pytest.approx(2.5) for w in fills), (
            f"fill outline {fills} should follow edgewidth"
        )
    finally:
        pp.rcParams["edgewidth"] = saved_ew
        pp.rcParams["lines.linewidth"] = saved_lw


def test_kdeplot_curve_and_contours_ignore_edgewidth(df):
    """The 1D curve and the 2D contour isolines are data lines.

    Cranking edgewidth must not move either. (The 1D curve is also
    covered by ``test_data_lines_ignore_edgewidth`` above; this pins the
    2D contour path, whose stroke lives on a QuadContourSet rather than
    on Line2D artists.)
    """
    saved_ew = pp.rcParams["edgewidth"]
    saved_lw = pp.rcParams["lines.linewidth"]
    try:
        pp.rcParams["edgewidth"] = 2.5
        pp.rcParams["lines.linewidth"] = 0.5  # deliberately inverted
        fig, ax = pp.subplots(1, 1, axes_size=(40, 30))
        pp.kdeplot(data=df, x="x", fill=False, ax=ax)
        curves = _line_widths(ax)
        assert curves, "expected a density curve"
        assert all(w == pytest.approx(0.5) for w in curves), (
            f"density curve {curves} should follow lines.linewidth"
        )

        fig2, ax2 = pp.subplots(1, 1, axes_size=(40, 30))
        pp.kdeplot(data=df, x="x", y="y", ax=ax2)
        contour_widths = []
        for c in ax2.collections:
            contour_widths += [float(v) for v in np.atleast_1d(c.get_linewidths())]
        assert contour_widths, "expected a contour set"
        assert all(w == pytest.approx(0.5) for w in contour_widths), (
            f"contour isolines {contour_widths} should follow lines.linewidth"
        )
    finally:
        pp.rcParams["edgewidth"] = saved_ew
        pp.rcParams["lines.linewidth"] = saved_lw


# ---------------------------------------------------------------------------
# histplot(kde=True): the step/poly outline and the KDE curve (#205)
# ---------------------------------------------------------------------------

_HIST_BINS = 12
"""Explicit bin count, chosen far from the KDE's 200-point gridsize.

It lets these tests tell the outline from the curve by point count.
``publiplots`` itself must NOT rely on that: with ``bins=200`` and
``element='poly'`` the two are indistinguishable by point count *and* by
drawstyle, which is why ``hist._KDE_GID`` tags the curve at creation.
"""


def _hist_outline_and_curve_widths(ax):
    """Split a histplot's strokes into (outline widths, KDE curve widths).

    The outline is a ``Rectangle`` (``element='bars'``), a
    ``PolyCollection`` (step/poly with ``fill=True``) or a short
    ``Line2D`` (step/poly with ``fill=False``); the KDE curve is always
    the long ``Line2D``.
    """
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import Rectangle

    outline = [
        float(p.get_linewidth()) for p in ax.patches if isinstance(p, Rectangle)
    ]
    for c in ax.collections:
        if isinstance(c, PolyCollection):
            outline += [float(v) for v in np.atleast_1d(c.get_linewidths())]

    curve = []
    for line in ax.lines:
        n = len(np.asarray(line.get_xdata()))
        if n <= _HIST_BINS + 1:
            outline.append(float(line.get_linewidth()))
        else:
            curve.append(float(line.get_linewidth()))
    return outline, curve


@pytest.mark.parametrize("element", ["bars", "step", "poly"])
@pytest.mark.parametrize("fill", [True, False], ids=["fill", "nofill"])
@pytest.mark.parametrize(
    "extra",
    [
        {},
        {"hue": "g"},
        {"hue": "g", "multiple": "stack"},
        {"hue": "g", "multiple": "fill"},
    ],
    ids=["plain", "hue", "stack", "multiple-fill"],
)
def test_histplot_kde_curve_and_outline_use_different_knobs(df, element, fill, extra):
    """The histogram hull outlines a shape (edgewidth); the KDE curve laid
    over it is data (lines.linewidth). One knob cannot mean both.

    Before #205 this held only for ``element='bars'``: with step/poly the
    outline and the curve are both ``Line2D`` artists in ``ax.lines``, and
    both were painted with ``linewidth`` and then floored at
    ``lines.linewidth``.
    """
    saved_ew = pp.rcParams["edgewidth"]
    saved_lw = pp.rcParams["lines.linewidth"]
    try:
        pp.rcParams["edgewidth"] = 2.5
        pp.rcParams["lines.linewidth"] = 0.5  # deliberately inverted
        fig, ax = pp.subplots(1, 1, axes_size=(40, 30))
        pp.histplot(
            data=df, x="x", ax=ax, kde=True, element=element, fill=fill,
            bins=_HIST_BINS, **extra,
        )

        outline, curve = _hist_outline_and_curve_widths(ax)
        assert outline, "expected a histogram outline"
        assert curve, "expected a KDE curve"
        assert all(w == pytest.approx(2.5) for w in outline), (
            f"{element}/fill={fill}: outline {outline} should follow edgewidth"
        )
        assert all(w == pytest.approx(0.5) for w in curve), (
            f"{element}/fill={fill}: KDE curve {curve} should follow "
            "lines.linewidth"
        )
    finally:
        pp.rcParams["edgewidth"] = saved_ew
        pp.rcParams["lines.linewidth"] = saved_lw


@pytest.mark.parametrize("element", ["bars", "step", "poly"])
def test_histplot_line_kws_width_reaches_the_kde_curve(df, element):
    """``line_kws={'linewidth': ...}`` sets the KDE curve, not the outline.

    Row 1 of #205: under step/poly it was silently overwritten by the
    outline width and then floored at ``lines.linewidth``.
    """
    saved_ew = pp.rcParams["edgewidth"]
    saved_lw = pp.rcParams["lines.linewidth"]
    try:
        pp.rcParams["edgewidth"] = 2.5
        pp.rcParams["lines.linewidth"] = 0.5  # deliberately inverted
        fig, ax = pp.subplots(1, 1, axes_size=(40, 30))
        pp.histplot(
            data=df, x="x", ax=ax, kde=True, element=element, fill=False,
            bins=_HIST_BINS, line_kws={"linewidth": 3.0},
        )
        outline, curve = _hist_outline_and_curve_widths(ax)
        assert curve and all(w == pytest.approx(3.0) for w in curve), (
            f"{element}: line_kws linewidth lost -- curve is {curve}"
        )
        assert outline and all(w == pytest.approx(2.5) for w in outline), (
            f"{element}: line_kws must not reach the outline {outline}"
        )
    finally:
        pp.rcParams["edgewidth"] = saved_ew
        pp.rcParams["lines.linewidth"] = saved_lw


@pytest.mark.parametrize("element", ["bars", "step", "poly"])
def test_histplot_linewidth_does_not_leak_into_the_kde_curve(df, element):
    """The public ``linewidth=`` is the outline width only.

    Row 2 of #205: under step/poly, ``linewidth=2.0`` widened the curve too.
    """
    saved_ew = pp.rcParams["edgewidth"]
    saved_lw = pp.rcParams["lines.linewidth"]
    try:
        pp.rcParams["edgewidth"] = 2.5
        pp.rcParams["lines.linewidth"] = 0.5  # deliberately inverted
        fig, ax = pp.subplots(1, 1, axes_size=(40, 30))
        pp.histplot(
            data=df, x="x", ax=ax, kde=True, element=element, fill=False,
            bins=_HIST_BINS, linewidth=2.0,
        )
        outline, curve = _hist_outline_and_curve_widths(ax)
        assert outline and all(w == pytest.approx(2.0) for w in outline), (
            f"{element}: linewidth= should set the outline, got {outline}"
        )
        assert curve and all(w == pytest.approx(0.5) for w in curve), (
            f"{element}: linewidth= leaked into the KDE curve {curve}"
        )
    finally:
        pp.rcParams["edgewidth"] = saved_ew
        pp.rcParams["lines.linewidth"] = saved_lw


@pytest.mark.parametrize("element", ["bars", "step", "poly"])
def test_histplot_thin_outline_is_not_floored_by_the_kde_curve(df, element):
    """A ``linewidth`` below ``lines.linewidth`` stays where the caller put it.

    Row 3 of #205: under step/poly, ``fill=False, linewidth=0.4`` drew both
    strokes at ``lines.linewidth`` because the KDE curve's width floor was
    applied to the outline as well.
    """
    saved_ew = pp.rcParams["edgewidth"]
    saved_lw = pp.rcParams["lines.linewidth"]
    try:
        pp.rcParams["edgewidth"] = 2.5
        pp.rcParams["lines.linewidth"] = 0.5  # deliberately inverted
        fig, ax = pp.subplots(1, 1, axes_size=(40, 30))
        pp.histplot(
            data=df, x="x", ax=ax, kde=True, element=element, fill=False,
            bins=_HIST_BINS, linewidth=0.4,
        )
        outline, curve = _hist_outline_and_curve_widths(ax)
        assert outline and all(w == pytest.approx(0.4) for w in outline), (
            f"{element}: outline {outline} was floored at lines.linewidth"
        )
        assert curve and all(w == pytest.approx(0.5) for w in curve), (
            f"{element}: KDE curve {curve} should follow lines.linewidth"
        )
    finally:
        pp.rcParams["edgewidth"] = saved_ew
        pp.rcParams["lines.linewidth"] = saved_lw


def test_histplot_kde_tag_does_not_survive_onto_the_artists(df):
    """The KDE discriminator is internal scaffolding, not rendered output.

    A caller-supplied ``line_kws={'gid': ...}`` must come back intact, and
    an absent one must stay absent (duplicate gids would collide in SVG).
    """
    fig, ax = pp.subplots(1, 1, axes_size=(40, 30))
    pp.histplot(
        data=df, x="x", hue="g", ax=ax, kde=True, element="step", fill=False,
        bins=_HIST_BINS,
    )
    assert all(line.get_gid() is None for line in ax.lines)

    fig2, ax2 = pp.subplots(1, 1, axes_size=(40, 30))
    pp.histplot(
        data=df, x="x", hue="g", ax=ax2, kde=True, element="step", fill=False,
        bins=_HIST_BINS, line_kws={"gid": "mine"},
    )
    gids = [line.get_gid() for line in ax2.lines]
    assert gids.count("mine") == 2, f"caller gid not preserved: {gids}"
    assert gids.count(None) == 2, f"outline gid should stay unset: {gids}"
