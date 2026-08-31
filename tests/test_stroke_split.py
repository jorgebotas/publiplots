"""Plots that draw an outline AND a data line must use two different knobs.

`edgewidth` is a stroke that outlines a shape; `lines.linewidth` is a
stroke that IS the data. regplot, residplot and the upset membership
matrix each drew both from one resolved value.
"""
import numpy as np
import pandas as pd
import pytest

import publiplots as pp


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
    import matplotlib.pyplot as plt
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
