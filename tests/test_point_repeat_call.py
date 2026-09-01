"""Repeated pp.pointplot / pp.lineplot calls on one Axes (issue #103).

The shape×color idiom — one ``pp.pointplot`` call per marker shape, sharing
a hue axis — used to leak matplotlib's tab10 defaults onto the second call's
markers: ``apply_double_layer_markers`` rescanned *every* line on the axes,
so it re-layered the previous call's marker copies. Those copies were drawn
by ``ax.plot`` without an explicit ``color=``, so they had picked up (and
advanced) the axes property cycle; re-layering them made those cycle colors
visible.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.colors import to_hex, to_rgba

import publiplots as pp


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


BACKBONE = {"None": "#6c7a89", "Trans": "#2471a3", "Mamba": "#7d3c98"}


def _two_pipeline_df():
    return pd.DataFrame({
        "Pool": ["ABMIL", "mean", "max"] * 6,
        "Backbone": (["None"] * 3 + ["Trans"] * 3 + ["Mamba"] * 3) * 2,
        "Value": [0.80, 0.82, 0.81, 0.83, 0.84, 0.82, 0.79, 0.80, 0.81] * 2,
        "Pipeline": ["Raw-CNN"] * 9 + ["VAE-prior"] * 9,
    })


def _draw_two_calls(ax, df):
    for pipeline, marker in [("Raw-CNN", "o"), ("VAE-prior", "D")]:
        sub = df[df["Pipeline"] == pipeline]
        pp.pointplot(
            data=sub, x="Value", y="Pool", hue="Backbone",
            palette=BACKBONE, hue_order=["None", "Trans", "Mamba"],
            markers=marker, linestyle="none", dodge=0.35,
            errorbar=None, ax=ax,
            legend=(pipeline == "VAE-prior"),
        )


def _marker_face_hexes(ax):
    """Base hue of every visible marker layer's fill, alpha channel dropped.

    The foreground layer's fill is ``to_rgba(color, alpha)`` — the RGB is
    the untouched base color with only the alpha channel lowered — so
    dropping alpha recovers the palette entry exactly.
    """
    hexes = set()
    for line in ax.lines:
        if line.get_marker() in (None, "None", ""):
            continue
        if not line.get_markersize():
            continue
        r, g, b, _ = to_rgba(line.get_markerfacecolor())
        hexes.add(to_hex((r, g, b)))
    return hexes


TAB10 = {
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
}


def test_second_pointplot_call_honors_palette():
    """No tab10 leak: every marker fill traces back to the explicit palette."""
    fig, ax = pp.subplots()
    _draw_two_calls(ax, _two_pipeline_df())

    faces = _marker_face_hexes(ax)
    assert not (faces & TAB10), f"tab10 colors leaked onto markers: {faces & TAB10}"


def test_second_pointplot_call_uses_only_palette_and_white():
    """Marker fills are exactly the palette colors plus the white backing."""
    fig, ax = pp.subplots()
    _draw_two_calls(ax, _two_pipeline_df())

    allowed = {to_hex(c) for c in BACKBONE.values()} | {"#ffffff"}
    assert _marker_face_hexes(ax) <= allowed


def test_no_line_carries_a_cycle_color():
    """Layer lines must declare their own color, not borrow the axes cycle.

    ``get_color()`` on a marker copy is read by other publiplots passes, so
    an implicit cycle color there is a lie about what the mark shows — and
    it silently advances the cycle for anything drawn later.
    """
    fig, ax = pp.subplots()
    _draw_two_calls(ax, _two_pipeline_df())

    allowed = {to_hex(c) for c in BACKBONE.values()} | {"#ffffff"}
    for line in ax.lines:
        assert to_hex(to_rgba(line.get_color())) in allowed


def test_repeated_calls_do_not_multiply_marker_layers():
    """Each call layers only the markers it just drew.

    3 hue levels per call → 3 seaborn originals + 3 white backings +
    3 foreground layers = 9 lines per call, 18 after two calls. The
    re-scan bug produced 36.
    """
    df = _two_pipeline_df()
    fig, ax = pp.subplots()

    pp.pointplot(
        data=df[df["Pipeline"] == "Raw-CNN"], x="Value", y="Pool",
        hue="Backbone", palette=BACKBONE, hue_order=["None", "Trans", "Mamba"],
        markers="o", linestyle="none", errorbar=None, ax=ax, legend=False,
    )
    assert len(ax.lines) == 9

    pp.pointplot(
        data=df[df["Pipeline"] == "VAE-prior"], x="Value", y="Pool",
        hue="Backbone", palette=BACKBONE, hue_order=["None", "Trans", "Mamba"],
        markers="D", linestyle="none", errorbar=None, ax=ax, legend=False,
    )
    assert len(ax.lines) == 18


def test_second_call_marker_shape_not_applied_to_first_call():
    """The two calls keep their own marker shapes, 9 lines each."""
    fig, ax = pp.subplots()
    _draw_two_calls(ax, _two_pipeline_df())

    shapes = [line.get_marker() for line in ax.lines
              if line.get_marker() not in (None, "None", "")]
    assert shapes.count("o") == 9
    assert shapes.count("D") == 9


def test_second_call_point_meta_matches_that_call_only():
    """``_publiplots_point_meta`` describes the latest call, not the axes.

    ``_iter_point_marker_series`` also rescanned the whole axes, so after a
    second call it saw six seaborn originals and paired series 3-5 against a
    three-level ``hue_order`` — dropping their hue to ``None``.
    """
    fig, ax = pp.subplots()
    _draw_two_calls(ax, _two_pipeline_df())

    meta = ax._publiplots_point_meta
    assert len(meta.points) == 9  # 3 pools x 3 backbones
    assert all(p.hue_value in BACKBONE for p in meta.points)


def test_lineplot_then_pointplot_does_not_relayer_the_lineplot():
    """Mixed plot kinds on one axes stay independent."""
    df = pd.DataFrame({
        "t": [0, 1, 2, 0, 1, 2],
        "y": [1.0, 2.0, 3.0, 2.0, 3.0, 4.0],
        "g": ["a", "a", "a", "b", "b", "b"],
    })
    fig, ax = pp.subplots()
    pp.lineplot(data=df, x="t", y="y", hue="g", style="g",
                palette={"a": "#6c7a89", "b": "#2471a3"},
                markers=True, ax=ax, legend=False)
    n_after_line = len(ax.lines)
    # lineplot layers its own markers: 2 originals + 2 copies each.
    assert n_after_line == 6

    pp.pointplot(data=df, x="t", y="y", hue="g",
                 palette={"a": "#6c7a89", "b": "#2471a3"},
                 markers="D", errorbar=None, ax=ax, legend=False)

    # Only the pointplot's own 2 series get layered (2 originals + 4 copies).
    assert len(ax.lines) == n_after_line + 6

    allowed = {"#6c7a89", "#2471a3", "#ffffff"}
    assert _marker_face_hexes(ax) <= allowed
