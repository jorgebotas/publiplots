"""pp.rcParams['edgewidth'] drives the outlines publiplots draws itself.

(Seaborn-drawn confidence-band edges are the documented exception: they
follow matplotlib's ``patch.linewidth``. See ``test_stroke_split.py``.)

The legend assertions are the point of this file: a swatch that does not
match the stroke actually drawn is the regression that motivated the
edgewidth/lines.linewidth split. See
docs/superpowers/specs/2026-08-31-rcparams-polish-design.md section 4.
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import publiplots as pp


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


PROBE = 2.5  # deliberately unlike any default, so a fallback is obvious


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "x": rng.normal(size=120),
        "y": rng.normal(size=120),
        "cat": rng.choice(list("AB"), 120),
        "g": rng.choice(["one", "two"], 120),
        "s": rng.choice(["p", "q"], 120),
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


def _legends(fig, ax):
    legends = [
        c for c in list(ax.get_children()) + list(fig.get_children())
        if isinstance(c, matplotlib.legend.Legend)
    ]
    assert legends, "expected at least one legend to be rendered"
    return legends


def _draws_a_line(artist):
    """True if this artist's linewidth actually renders something.

    A marker-only artist (linestyle 'None') carries a linewidth that draws
    nothing, so reading it measures a value the figure never renders.
    """
    return str(artist.get_linestyle()) not in ("None", "none", " ", "")


def _legend_swatch_widths(fig, ax):
    """Stroke width of every legend swatch, however it was drawn.

    Rectangle swatches carry it on linewidth; marker swatches carry it on
    markeredgewidth. Both must agree with the plotted stroke.
    """
    widths = []
    for leg in _legends(fig, ax):
        for a in leg.findobj():
            if isinstance(a, matplotlib.patches.Rectangle) and a.get_width() < 50:
                # Identify the fill layer by its transparent edge, NOT by a
                # zero width -- skipping every zero-width stroke would also
                # hide a genuine outline that got zeroed out.
                if float(a.get_edgecolor()[3]) == 0:
                    continue
                widths.append(float(a.get_linewidth()))
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


def test_venn_circle_outlines_follow_edgewidth(edgewidth_probe):
    """Venn circles are shape outlines, so they must follow edgewidth.

    They previously rode matplotlib's ``patch.linewidth`` -- the same
    default value, so the appearance matched, but the wrong knob.
    """
    ax = pp.venn(sets={"A": {1, 2, 3}, "B": {2, 3, 4}})
    ellipses = [
        p for p in ax.patches if isinstance(p, matplotlib.patches.Ellipse)
    ]
    assert len(ellipses) == 2, "expected two venn circles"
    assert all(float(e.get_linewidth()) == pytest.approx(PROBE) for e in ellipses)


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

@pytest.mark.parametrize("fn_name", [
    "barplot", "boxplot", "violinplot", "histplot",
    "scatterplot", "stripplot", "swarmplot",
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
    # One swatch stroke per hue level -- pinned so that a *partial* failure
    # (one of the two swatches wrong, or silently dropped) cannot pass.
    assert len(swatches) == df["g"].nunique(), (
        f"{fn_name}: expected one swatch stroke per hue level, got {swatches}"
    )
    # Assert against the width actually drawn, not just against PROBE: the
    # swatch matching the figure is the property under test.
    drawn = sorted(set(_patch_widths(ax) + _collection_widths(ax)))
    assert drawn == [pytest.approx(PROBE)], (
        f"{fn_name}: expected a single drawn stroke at {PROBE}, got {drawn}"
    )
    assert all(w == pytest.approx(drawn[0]) for w in swatches), (
        f"{fn_name}: legend swatches {swatches} do not match the drawn "
        f"stroke {drawn[0]}"
    )


# ---- The same guarantee for the LINE half of a swatch ----------------------

# edgewidth and lines.linewidth are deliberately INVERTED here: an outline
# reads 2.5 while a data line reads 3.5. A site wired to the wrong knob is
# then caught outright, instead of passing by coincidence because both knobs
# happen to hold the same number.
LINE_PROBE = 3.5


@pytest.fixture
def inverted_stroke_probes():
    saved = (pp.rcParams["edgewidth"], pp.rcParams["lines.linewidth"])
    pp.rcParams["edgewidth"] = PROBE
    pp.rcParams["lines.linewidth"] = LINE_PROBE
    try:
        yield PROBE, LINE_PROBE
    finally:
        pp.rcParams["edgewidth"], pp.rcParams["lines.linewidth"] = saved


def _legend_line_widths(fig, ax):
    """linewidth of every legend artist that actually draws a line."""
    return [
        float(a.get_linewidth())
        for leg in _legends(fig, ax)
        for a in leg.findobj(matplotlib.lines.Line2D)
        if _draws_a_line(a)
    ]


@pytest.mark.parametrize("fn_name,kwargs", [
    ("pointplot", dict(x="cat", y="y", hue="g")),
    ("lineplot", dict(x="x", y="y", hue="g")),
    ("lineplot", dict(x="x", y="y", hue="g", style="s")),
])
def test_legend_line_swatch_matches_drawn_line(
    df, inverted_stroke_probes, fn_name, kwargs
):
    """The line half of a swatch must match the line actually plotted.

    Counterpart to test_legend_swatch_matches_drawn_stroke: that one covers
    strokes that OUTLINE a shape (edgewidth), this one covers strokes that
    ARE the data (lines.linewidth). Reading `linewidth` here, never
    `markeredgewidth`.
    """
    fig, ax = pp.subplots(1, 1, axes_size=(40, 30))
    getattr(pp, fn_name)(data=df, ax=ax, **kwargs)
    fig.canvas.draw()

    drawn = sorted({
        float(ln.get_linewidth()) for ln in ax.lines if _draws_a_line(ln)
    })
    assert drawn == [pytest.approx(LINE_PROBE)], (
        f"{fn_name}: expected plotted lines at lines.linewidth="
        f"{LINE_PROBE}, got {drawn}"
    )

    swatches = _legend_line_widths(fig, ax)
    assert swatches, f"{fn_name}: no legend line swatches found"
    assert all(w == pytest.approx(drawn[0]) for w in swatches), (
        f"{fn_name}: legend line swatches {swatches} do not match the "
        f"drawn line width {drawn[0]}"
    )


def test_create_legend_handles_splits_line_from_outline(inverted_stroke_probes):
    """A line swatch built without an explicit width falls back to
    lines.linewidth; an outline swatch falls back to edgewidth.

    This is the regression test for the defect that a single shared
    `resolve_param("edgewidth", linewidth)` in `create_legend_handles`
    handed the outline knob to the LinePatch/LineMarkerPatch branches too,
    rendering a 0.75 line swatch for lines drawn at 1.0. Every in-tree
    caller passes an explicit linewidth, so only a direct call to this
    public function pins the fallback.
    """
    line = pp.create_legend_handles(labels=["a"], style="line")[0]
    line_marker = pp.create_legend_handles(
        labels=["a"], markers=["o"], linestyles=["-"]
    )[0]
    rect = pp.create_legend_handles(labels=["a"], style="rectangle")[0]
    circle = pp.create_legend_handles(labels=["a"], style="circle")[0]

    # Line halves -> lines.linewidth
    assert float(line.get_linewidth()) == pytest.approx(LINE_PROBE)
    assert float(line_marker.get_linewidth()) == pytest.approx(LINE_PROBE)
    # Outlines -> edgewidth
    assert float(rect.get_linewidth()) == pytest.approx(PROBE)
    assert float(circle.get_markeredgewidth()) == pytest.approx(PROBE)
    assert float(line_marker.get_markeredgewidth()) == pytest.approx(PROBE)

    # An explicit linewidth still overrides both.
    explicit = pp.create_legend_handles(
        labels=["a"], style="line", linewidth=0.25
    )[0]
    assert float(explicit.get_linewidth()) == pytest.approx(0.25)
