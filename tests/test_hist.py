"""Tests for pp.histplot."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes

import publiplots as pp
from publiplots.utils.legend_entries import get_entries, is_continuous_hue


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def hist_df():
    rng = np.random.default_rng(0)
    n = 2000
    value = np.concatenate([rng.normal(-2.0, 1.0, n // 2),
                            rng.normal(2.0, 1.0, n // 2)])
    group = rng.choice(["A", "B", "C"], size=n)
    pattern = rng.choice(["P", "Q"], size=n)
    positive = rng.lognormal(mean=0.0, sigma=1.0, size=n)
    return pd.DataFrame({
        "value": value,
        "group": group,
        "pattern": pattern,
        "positive": positive,
    })


# ---- Contract ----

def test_returns_axes(hist_df):
    ax = pp.histplot(data=hist_df, x="value")
    assert isinstance(ax, Axes)


def test_respects_ax(hist_df):
    fig, ax0 = pp.subplots(axes_size=(50, 30))
    ax1 = pp.histplot(data=hist_df, x="value", ax=ax0)
    assert ax1 is ax0


def test_rejects_figsize(hist_df):
    with pytest.raises(TypeError, match="figsize"):
        pp.histplot(data=hist_df, x="value", figsize=(4, 3))


def test_requires_x_or_y(hist_df):
    with pytest.raises(ValueError, match="At least one of"):
        pp.histplot(data=hist_df)


# ---- Bins ----

def test_bins_auto(hist_df):
    ax = pp.histplot(data=hist_df, x="value", bins="auto")
    assert len(ax.patches) > 0


def test_bins_int(hist_df):
    ax = pp.histplot(data=hist_df, x="value", bins=20)
    assert len(ax.patches) >= 20


def test_binwidth(hist_df):
    ax = pp.histplot(data=hist_df, x="value", binwidth=0.1)
    for patch in ax.patches:
        assert patch.get_width() == pytest.approx(0.1, abs=1e-3)


def test_binrange(hist_df):
    ax = pp.histplot(data=hist_df, x="value", binrange=(-1, 1))
    for patch in ax.patches:
        left = patch.get_x()
        right = left + patch.get_width()
        assert left >= -1 - 1e-9
        assert right <= 1 + 1e-9


# ---- Stats ----

def test_stat_count(hist_df):
    ax = pp.histplot(data=hist_df, x="value", stat="count")
    total = sum(p.get_height() for p in ax.patches)
    assert total == pytest.approx(len(hist_df), abs=1e-6)


def test_stat_density(hist_df):
    ax = pp.histplot(data=hist_df, x="value", stat="density")
    integral = sum(p.get_height() * p.get_width() for p in ax.patches)
    assert integral == pytest.approx(1.0, abs=0.01)


def test_stat_probability(hist_df):
    ax = pp.histplot(data=hist_df, x="value", stat="probability")
    total = sum(p.get_height() for p in ax.patches)
    assert total == pytest.approx(1.0, abs=0.01)


def test_stat_frequency(hist_df):
    ax = pp.histplot(data=hist_df, x="value", stat="frequency")
    heights = [p.get_height() for p in ax.patches]
    assert len(heights) > 0
    assert all(h > 0 for h in heights)


def test_stat_percent(hist_df):
    ax = pp.histplot(data=hist_df, x="value", stat="percent")
    total = sum(p.get_height() for p in ax.patches)
    assert total == pytest.approx(100.0, abs=0.01)


# ---- Hue paths ----

def test_no_hue_uses_color(hist_df):
    ax = pp.histplot(data=hist_df, x="value", color="red")
    expected = mcolors.to_rgba("red")[:3]
    assert ax.patches, "expected at least one patch"
    for patch in ax.patches:
        face = patch.get_facecolor()
        assert face[:3] == pytest.approx(expected, abs=1e-6)


def test_hue_string_palette(hist_df):
    ax = pp.histplot(data=hist_df, x="value", hue="group", palette="pastel")
    faces = {tuple(round(c, 6) for c in p.get_facecolor()[:3]) for p in ax.patches}
    assert len(faces) >= 2


def test_hue_dict_palette(hist_df):
    palette = {"A": "#ff0000", "B": "#00ff00", "C": "#0000ff"}
    ax = pp.histplot(data=hist_df, x="value", hue="group", palette=palette)
    faces = {tuple(round(c, 6) for c in p.get_facecolor()[:3]) for p in ax.patches}
    expected = {
        tuple(round(c, 6) for c in mcolors.to_rgba(v)[:3])
        for v in palette.values()
    }
    assert expected <= faces


def test_hue_order_reorders(hist_df):
    """hue_order controls the palette<->level mapping AND drawing order.

    Seaborn draws in reverse hue_order so the first-listed level renders
    on top; the *first* patch therefore matches the LAST entry in
    hue_order. This test pins that behaviour so we notice if seaborn
    changes it.
    """
    palette = {"A": "#ff0000", "B": "#00ff00", "C": "#0000ff"}
    ax = pp.histplot(
        data=hist_df, x="value",
        hue="group", hue_order=["C", "A", "B"], palette=palette,
    )
    first_face = tuple(round(c, 6) for c in ax.patches[0].get_facecolor()[:3])
    expected = tuple(round(c, 6) for c in mcolors.to_rgba(palette["B"])[:3])
    assert first_face == expected


# ---- Multiple modes ----

def test_multiple_layer(hist_df):
    ax = pp.histplot(data=hist_df, x="value", hue="group",
                     palette="pastel", multiple="layer")
    assert isinstance(ax, Axes)
    assert len(ax.patches) > 0


def test_multiple_dodge(hist_df):
    ax = pp.histplot(data=hist_df, x="value", hue="group",
                     palette="pastel", multiple="dodge", bins=10)
    # Bars at the same bin should be horizontally disjoint. Group
    # patches by rounded center y-bin index approximation by
    # clustering on near-equal bar widths — easier to check: every
    # unique (x, x+width) interval is distinct across hues within
    # each rendered bin, so there should be NO two patches with the
    # same start x.
    starts = [round(p.get_x(), 6) for p in ax.patches]
    assert len(starts) == len(set(starts))


def test_multiple_stack(hist_df):
    ax = pp.histplot(data=hist_df, x="value", hue="group",
                     palette="pastel", multiple="stack", bins=10)
    # With stacking, patches at the same (x, width) stack vertically;
    # sum of heights at a given x = total count in that bin.
    from collections import defaultdict
    by_x = defaultdict(float)
    for p in ax.patches:
        by_x[round(p.get_x(), 6)] += p.get_height()
    # Each x should have a finite total >= max individual height there.
    assert len(by_x) > 0
    assert all(v > 0 for v in by_x.values())


def test_multiple_fill(hist_df):
    ax = pp.histplot(data=hist_df, x="value", hue="group",
                     palette="pastel", multiple="fill", bins=10)
    from collections import defaultdict
    by_x = defaultdict(float)
    for p in ax.patches:
        by_x[round(p.get_x(), 6)] += p.get_height()
    assert len(by_x) > 0
    for total in by_x.values():
        assert total == pytest.approx(1.0, abs=1e-6)


# ---- Element modes ----

def test_element_bars(hist_df):
    ax = pp.histplot(data=hist_df, x="value", element="bars")
    assert len(ax.patches) > 0


def test_element_step(hist_df):
    # With fill=False, seaborn emits a Line2D outline per hue level
    # (one here) and no patches / no filled collections.
    ax = pp.histplot(data=hist_df, x="value",
                     element="step", fill=False)
    assert len(ax.lines) > 0
    assert len(ax.patches) == 0


def test_element_poly(hist_df):
    ax = pp.histplot(data=hist_df, x="value",
                     element="poly", fill=False)
    assert len(ax.lines) > 0
    assert len(ax.patches) == 0


def test_element_step_filled(hist_df):
    ax = pp.histplot(data=hist_df, x="value",
                     element="step", fill=True)
    assert len(ax.collections) > 0


# ---- KDE ----

def test_kde_adds_lines(hist_df):
    ax = pp.histplot(data=hist_df, x="value", hue="pattern",
                     palette="pastel", kde=True)
    # bars element keeps ax.lines empty; KDE adds exactly one line per
    # hue level.
    assert len(ax.lines) == 2


def test_kde_single_group(hist_df):
    ax = pp.histplot(data=hist_df, x="value", kde=True)
    assert len(ax.lines) == 1


# ---- Hatch ----

def test_hatch_applied(hist_df):
    # Provide a hatch_map with non-empty values on both levels so every
    # bar ends up with a visible hatch string (the default hatch map
    # uses '' for the first level, which would defeat "non-empty").
    ax = pp.histplot(data=hist_df, x="value",
                     hatch="pattern",
                     hatch_map={"P": "///", "Q": "..."})
    assert ax.patches, "expected bar patches"
    for patch in ax.patches:
        h = patch.get_hatch()
        assert isinstance(h, str) and h != ""


def test_hatch_map_dict(hist_df):
    # With no hue, the v1 implementation applies a single hatch (the
    # first in the map) to every bar. To exercise BOTH hatch patterns
    # we need hue == hatch so per-patch hue resolution picks the
    # per-level hatch.
    ax = pp.histplot(data=hist_df, x="value",
                     hue="pattern", palette="pastel",
                     hatch="pattern",
                     hatch_map={"P": "///", "Q": "..."})
    hatches = {p.get_hatch() for p in ax.patches}
    assert "///" in hatches
    assert "..." in hatches


def test_hatch_with_dodge_raises(hist_df):
    with pytest.raises(NotImplementedError):
        pp.histplot(data=hist_df, x="value",
                    hue="group", palette="pastel",
                    hatch="pattern", multiple="dodge")


# ---- Legend stash ----

def test_stash_contract_hue(hist_df):
    ax = pp.histplot(data=hist_df, x="value",
                     hue="group", palette="pastel")
    kinds = [e.kind for e in get_entries(ax)]
    assert "hue" in kinds


def test_stash_contract_no_hue(hist_df):
    ax = pp.histplot(data=hist_df, x="value")
    entries = get_entries(ax)
    assert entries == [] or all(e.kind != "hue" for e in entries)


def test_legend_false_no_render(hist_df):
    ax = pp.histplot(data=hist_df, x="value",
                     hue="group", palette="pastel", legend=False)
    assert ax.get_legend() is None


def test_legend_dict_hue_false(hist_df):
    ax = pp.histplot(data=hist_df, x="value",
                     hue="group", palette="pastel",
                     legend={"hue": False})
    # No "hue" legend rendered even though the entry may be stashed.
    legend = ax.get_legend()
    if legend is not None:
        titles = [t.get_text() for t in legend.get_texts()]
        assert "group" not in titles


# ---- Annotate ----

def test_annotate_true_adds_text(hist_df):
    ax = pp.histplot(data=hist_df, x="value",
                     bins=10, element="bars", annotate=True)
    nonempty = [t for t in ax.texts if t.get_text()]
    assert len(nonempty) > 0


def test_annotate_with_step_raises(hist_df):
    with pytest.raises(NotImplementedError):
        pp.histplot(data=hist_df, x="value",
                    element="step", annotate=True)


# ---- rcParams respect ----

def test_rcparams_color(hist_df):
    saved = pp.rcParams["color"]
    try:
        pp.rcParams["color"] = "#00ffff"
        ax = pp.histplot(data=hist_df, x="value")
        expected = mcolors.to_rgba("#00ffff")[:3]
        assert ax.patches
        for patch in ax.patches:
            face = patch.get_facecolor()
            assert face[:3] == pytest.approx(expected, abs=1e-6)
    finally:
        pp.rcParams["color"] = saved


def test_rcparams_alpha(hist_df):
    saved = pp.rcParams["alpha"]
    try:
        pp.rcParams["alpha"] = 0.42
        ax = pp.histplot(data=hist_df, x="value")
        assert ax.patches
        for patch in ax.patches:
            assert patch.get_facecolor()[3] == pytest.approx(0.42, abs=1e-6)
    finally:
        pp.rcParams["alpha"] = saved


# ---- Log scale ----

def test_log_scale_x(hist_df):
    ax = pp.histplot(data=hist_df, x="positive", log_scale=True)
    assert ax.get_xscale() == "log"


# ---- Orientation ----

def test_horizontal_y(hist_df):
    ax = pp.histplot(data=hist_df, y="value")
    assert len(ax.patches) > 0
    # Sensibility: y-axis now carries the binned variable name
    # (seaborn sets the ylabel when `y=` is used, publiplots overwrites
    # it to "" by default — but the xlabel should be whatever the stat
    # name resolved to, i.e. not the binned variable).
    assert isinstance(ax.get_ylabel(), str)


# ---- 2D mode ----

@pytest.fixture
def df_2d():
    rng = np.random.default_rng(0)
    n = 1000
    return pd.DataFrame({
        "x": rng.normal(0, 1, n),
        "y": rng.normal(0, 1, n),
        "g": rng.choice(["a", "b"], size=n),
    })


def test_2d_returns_quadmesh(df_2d):
    from matplotlib.collections import QuadMesh
    ax = pp.histplot(data=df_2d, x="x", y="y")
    assert any(isinstance(c, QuadMesh) for c in ax.collections)


def test_2d_default_cmap_tracks_pp_color(df_2d):
    import publiplots as _pp
    from matplotlib.collections import QuadMesh
    original = _pp.rcParams["color"]
    try:
        _pp.rcParams["color"] = "#ff0000"
        ax = pp.histplot(data=df_2d, x="x", y="y")
    finally:
        _pp.rcParams["color"] = original
    mesh = next(c for c in ax.collections if isinstance(c, QuadMesh))
    high_end = mesh.get_cmap()(1.0)
    assert high_end[0] > 0.9 and high_end[1] < 0.2 and high_end[2] < 0.2


def test_2d_explicit_cmap(df_2d):
    from matplotlib.collections import QuadMesh
    ax = pp.histplot(data=df_2d, x="x", y="y", cmap="rocket")
    mesh = next(c for c in ax.collections if isinstance(c, QuadMesh))
    assert mesh.get_cmap().name == "rocket"


def test_2d_vmin_vmax(df_2d):
    from matplotlib.collections import QuadMesh
    ax = pp.histplot(data=df_2d, x="x", y="y", vmin=0, vmax=10)
    mesh = next(c for c in ax.collections if isinstance(c, QuadMesh))
    assert mesh.norm.vmin == 0
    assert mesh.norm.vmax == 10


def test_2d_legend_stashes_continuous_hue_colorbar(df_2d):
    ax = pp.histplot(data=df_2d, x="x", y="y", stat="density")
    entries = get_entries(ax)
    assert len(entries) == 1
    entry = entries[0]
    assert entry.kind == "hue"
    assert is_continuous_hue(entry.handles)
    assert entry.name == "density"


def test_2d_with_hue_stashes_categorical_rectangles(df_2d):
    ax = pp.histplot(data=df_2d, x="x", y="y", hue="g")
    entries = get_entries(ax)
    assert len(entries) == 1
    entry = entries[0]
    assert entry.kind == "hue"
    assert not is_continuous_hue(entry.handles)
    assert len(entry.handles) == 2  # two hue levels


def test_2d_annotate_raises(df_2d):
    with pytest.raises(NotImplementedError, match="annotate"):
        pp.histplot(data=df_2d, x="x", y="y", annotate=True)


def test_2d_element_step_raises(df_2d):
    with pytest.raises(NotImplementedError, match="element"):
        pp.histplot(data=df_2d, x="x", y="y", element="step")


def test_2d_alpha_default_is_one(df_2d):
    from matplotlib.collections import QuadMesh
    ax = pp.histplot(data=df_2d, x="x", y="y")
    mesh = next(c for c in ax.collections if isinstance(c, QuadMesh))
    a = mesh.get_alpha()
    assert a is None or a == 1.0


def test_neither_x_nor_y_raises(df_2d):
    with pytest.raises(ValueError, match="At least one of"):
        pp.histplot(data=df_2d)
