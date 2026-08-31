"""pp.rcParams['edgewidth'] drives every stroke that outlines a shape.

The legend assertions are the point of this file: a swatch that does not
match the stroke actually drawn is the regression that motivated the
edgewidth/lines.linewidth split. See
docs/superpowers/specs/2026-08-31-rcparams-polish-design.md section 4.
"""
import matplotlib
import numpy as np
import pandas as pd
import pytest

import publiplots as pp

PROBE = 2.5  # deliberately unlike any default, so a fallback is obvious


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "x": rng.normal(size=120),
        "y": rng.normal(size=120),
        "cat": rng.choice(list("AB"), 120),
        "g": rng.choice(["one", "two"], 120),
    })


@pytest.fixture
def edgewidth_probe():
    """Set edgewidth to PROBE for the duration of a test, then restore."""
    saved = pp.rcParams["edgewidth"]
    pp.rcParams["edgewidth"] = PROBE
    try:
        yield PROBE
    finally:
        pp.rcParams["edgewidth"] = saved


def _patch_widths(ax):
    return [float(p.get_linewidth()) for p in ax.patches]


def _collection_widths(ax):
    out = []
    for c in ax.collections:
        out += [float(v) for v in np.atleast_1d(c.get_linewidths())]
    return out


def _legend_swatch_widths(fig, ax):
    """Stroke width of every legend swatch, however it was drawn.

    Rectangle swatches carry it on linewidth; marker swatches carry it on
    markeredgewidth. Both must agree with the plotted stroke.
    """
    widths = []
    legends = [
        c for c in list(ax.get_children()) + list(fig.get_children())
        if isinstance(c, matplotlib.legend.Legend)
    ]
    assert legends, "expected at least one legend to be rendered"
    for leg in legends:
        for a in leg.findobj():
            if isinstance(a, matplotlib.patches.Rectangle) and a.get_width() < 50:
                lw = float(a.get_linewidth())
                if lw:  # the fill layer is linewidth=0 by design; skip it
                    widths.append(lw)
            elif isinstance(a, matplotlib.lines.Line2D):
                if a.get_marker() not in (None, "none", ""):
                    widths.append(float(a.get_markeredgewidth()))
    return widths


# ---- Patch-outline families -------------------------------------------------

@pytest.mark.parametrize("fn_name", ["barplot", "histplot"])
def test_patch_edges_follow_edgewidth(df, edgewidth_probe, fn_name):
    fig, ax = pp.subplots(1, 1, axes_size=(40, 30))
    fn = getattr(pp, fn_name)
    if fn_name == "histplot":
        fn(data=df, x="y", hue="g", ax=ax)
    else:
        fn(data=df, x="cat", y="y", hue="g", ax=ax)
    assert _patch_widths(ax), "expected patches"
    assert all(w == pytest.approx(PROBE) for w in _patch_widths(ax))


@pytest.mark.parametrize("fn_name", ["violinplot", "hexbinplot"])
def test_collection_edges_follow_edgewidth(df, edgewidth_probe, fn_name):
    fig, ax = pp.subplots(1, 1, axes_size=(40, 30))
    if fn_name == "hexbinplot":
        pp.hexbinplot(data=df, x="x", y="y", ax=ax)
    else:
        pp.violinplot(data=df, x="cat", y="y", hue="g", ax=ax)
    widths = _collection_widths(ax)
    assert widths, "expected collections"
    assert all(w == pytest.approx(PROBE) for w in widths)


# ---- Marker-edge families ---------------------------------------------------

@pytest.mark.parametrize("fn_name", ["scatterplot", "stripplot", "swarmplot"])
def test_marker_edges_follow_edgewidth(df, edgewidth_probe, fn_name):
    fig, ax = pp.subplots(1, 1, axes_size=(40, 30))
    fn = getattr(pp, fn_name)
    if fn_name == "scatterplot":
        fn(data=df, x="x", y="y", hue="g", ax=ax)
    else:
        fn(data=df, x="cat", y="y", hue="g", ax=ax)
    widths = _collection_widths(ax)
    assert widths, "expected marker collections"
    assert all(w == pytest.approx(PROBE) for w in widths)


# ---- The regression: legend swatch must match the drawn stroke -------------

# The three marker families are xfail until Task 4: they stash only
# `linewidth`, but HandlerMarker draws a marker swatch's outline from
# `markeredgewidth`, so the swatch silently reports the rcParams default.
# strict=True means Task 4 MUST remove these marks -- once the fix lands,
# an unexpected pass fails the suite. That keeps every commit green while
# still recording the known gap in the test file rather than in prose.
_MARKER_XFAIL = pytest.mark.xfail(
    strict=True,
    reason="fixed in Task 4: scatter/strip/swarm do not stash markeredgewidth",
)


@pytest.mark.parametrize("fn_name", [
    "barplot", "boxplot", "violinplot", "histplot",
    pytest.param("scatterplot", marks=_MARKER_XFAIL),
    pytest.param("stripplot", marks=_MARKER_XFAIL),
    pytest.param("swarmplot", marks=_MARKER_XFAIL),
])
def test_legend_swatch_matches_drawn_stroke(df, edgewidth_probe, fn_name):
    """A legend swatch drawn at a different width than the mark it labels
    is a lie about the figure. Regression: scatter/strip/swarm stashed
    only `linewidth`, so their marker swatches silently fell back to
    rcParams['lines.markeredgewidth']."""
    fig, ax = pp.subplots(1, 1, axes_size=(40, 30))
    fn = getattr(pp, fn_name)
    if fn_name == "histplot":
        fn(data=df, x="y", hue="g", ax=ax)
    elif fn_name == "scatterplot":
        fn(data=df, x="x", y="y", hue="g", ax=ax)
    else:
        fn(data=df, x="cat", y="y", hue="g", ax=ax)
    fig.canvas.draw()

    swatches = _legend_swatch_widths(fig, ax)
    assert swatches, f"{fn_name}: no legend swatch strokes found"
    assert all(w == pytest.approx(PROBE) for w in swatches), (
        f"{fn_name}: legend swatches {swatches} do not match "
        f"edgewidth={PROBE}"
    )
