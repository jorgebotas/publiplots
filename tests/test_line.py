"""Tests for pp.lineplot."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
import numpy as np
import pandas as pd
import pytest

import publiplots as pp
from publiplots.utils.legend_entries import get_entries


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def line_df():
    rng = np.random.default_rng(0)
    n = 40
    return pd.DataFrame({
        "t": np.tile(np.linspace(0, 10, 20), 2),
        "y": rng.normal(size=n),
        "g": np.repeat(["A", "B"], 20),
        "m": np.tile(np.repeat(["raw", "smoothed"], 10), 2),
        "s": rng.uniform(1, 5, n),
    })


# ---- Contract ----

def test_lineplot_returns_axes(line_df):
    ax = pp.lineplot(data=line_df, x="t", y="y")
    assert isinstance(ax, Axes)


def test_lineplot_respects_ax(line_df):
    fig, ax0 = pp.subplots(axes_size=(50, 30))
    ax1 = pp.lineplot(data=line_df, x="t", y="y", ax=ax0)
    assert ax1 is ax0


def test_lineplot_no_hue_stashes_nothing(line_df):
    ax = pp.lineplot(data=line_df, x="t", y="y")
    assert get_entries(ax) == []


def test_lineplot_with_hue_stashes_hue_entry(line_df):
    ax = pp.lineplot(data=line_df, x="t", y="y", hue="g", palette="pastel")
    names_kinds = [(e.name, e.kind) for e in get_entries(ax)]
    assert ("g", "hue") in names_kinds


def test_lineplot_with_hue_size_style(line_df):
    ax = pp.lineplot(data=line_df, x="t", y="y",
                     hue="g", size="s", style="m",
                     palette="pastel", markers=True)
    kinds = {e.kind for e in get_entries(ax)}
    assert {"hue", "size", "style"} <= kinds


def test_lineplot_sort_false(line_df):
    ax = pp.lineplot(data=line_df, x="t", y="y", sort=False)
    assert isinstance(ax, Axes)


def test_lineplot_err_style_bars(line_df):
    ax = pp.lineplot(data=line_df, x="t", y="y", hue="g",
                     palette="pastel", err_style="bars")
    assert isinstance(ax, Axes)


def test_lineplot_continuous_hue_colorbar(line_df):
    df = line_df.assign(score=np.linspace(0, 100, len(line_df)))
    ax = pp.lineplot(data=df, x="t", y="y", hue="score",
                     palette="viridis", hue_norm=(0, 100))
    entries = [e for e in get_entries(ax) if e.kind == "hue"]
    assert entries and isinstance(entries[0].handles[0], ScalarMappable)


def test_lineplot_categorical_hue_uses_line_patch(line_df):
    """Categorical-hue handles must be LinePatch (colored line swatch),
    not RectanglePatch — the swatch is a solid colored line matching
    the rendered plot."""
    from publiplots.utils.legend import LinePatch
    ax = pp.lineplot(data=line_df, x="t", y="y", hue="g", palette="pastel")
    hue_entry = next(e for e in get_entries(ax) if e.kind == "hue")
    assert all(isinstance(h, LinePatch) for h in hue_entry.handles)


def test_lineplot_style_dashes_uses_line_patch_with_dash_linestyle(line_df):
    """Style-via-dashes handles must be LinePatch with the user's dash
    pattern embedded in each handle's linestyle."""
    from publiplots.utils.legend import LinePatch
    ax = pp.lineplot(
        data=line_df, x="t", y="y",
        style="g", palette="pastel",
        dashes={"A": (4, 2), "B": (1, 1)},
    )
    style_entry = next(e for e in get_entries(ax) if e.kind == "style")
    assert all(isinstance(h, LinePatch) for h in style_entry.handles)
    linestyles = [h.get_linestyle() for h in style_entry.handles]
    # User's tuples preserved on the handle (normalization to (offset, seq)
    # happens at render time in HandlerLine, not at stash time).
    assert (4, 2) in linestyles
    assert (1, 1) in linestyles


def test_lineplot_hue_style_dashes_markers_renders(line_df):
    """Regression for issue #99: hue + style + dashes={...(on, off)...} + markers
    used to crash in matplotlib's ``_get_dash_pattern`` during legend render
    because ``HandlerLineMarker`` forwarded the raw on-off tuple to a fresh
    ``Line2D``. Rendering must not raise."""
    fig, ax = pp.subplots(axes_size=(50, 40))
    pp.lineplot(
        data=line_df, x="t", y="y",
        hue="g", style="m",
        palette={"A": "#43adaa", "B": "#6565eb"},
        dashes={"raw": "", "smoothed": (4, 2)},
        markers=True, ax=ax, sort=False, errorbar=None, legend=True,
    )
    fig.canvas.draw()


def test_lineplot_hue_equals_style_dashes_markers_renders(line_df):
    """Regression for issue #99, merge_hue_style branch: when hue == style the
    handle is a merged ``LineMarkerPatch`` carrying the raw dash tuple. It must
    render without crashing too."""
    fig, ax = pp.subplots(axes_size=(50, 40))
    pp.lineplot(
        data=line_df, x="t", y="y",
        hue="g", style="g",
        palette={"A": "#43adaa", "B": "#6565eb"},
        dashes={"A": (4, 2), "B": (1, 1)},
        markers=True, ax=ax, sort=False, errorbar=None, legend=True,
    )
    fig.canvas.draw()


# ---- Legend stash ----

def test_lineplot_legend_false_stashes_nothing(line_df):
    ax = pp.lineplot(data=line_df, x="t", y="y", hue="g",
                     palette="pastel", legend=False)
    assert get_entries(ax) == []


def test_lineplot_legend_dict_suppresses_hue(line_df):
    ax = pp.lineplot(data=line_df, x="t", y="y",
                     hue="g", size="s", palette="pastel",
                     legend={"hue": False})
    names = [e.name for e in get_entries(ax)]
    assert "g" not in names
    assert "s" in names


def test_lineplot_in_group_suppresses_per_axis_render(line_df):
    from matplotlib.legend import Legend
    fig, axes = pp.subplots(1, 2, axes_size=(50, 40))
    pp.legend_group(anchor=axes[-1])
    pp.lineplot(data=line_df, x="t", y="y", hue="g",
                palette="pastel", ax=axes[0])
    fig.canvas.draw()
    assert [c for c in axes[0].get_children() if isinstance(c, Legend)] == []


# ---- Same-variable dedup ----

def test_lineplot_hue_equals_style_single_entry(line_df):
    """When hue and style reference the same categorical column, stash one
    merged LegendEntry instead of duplicate hue + style entries."""
    ax = pp.lineplot(
        data=line_df, x="t", y="y",
        hue="g", style="g",
        palette="pastel",
        dashes={"A": (4, 2), "B": (1, 1)},
    )
    entries = get_entries(ax)
    assert len(entries) == 1
    entry = entries[0]
    assert entry.name == "g"
    assert entry.kind == "hue"
    # The merged handle must carry the style's linestyle so the swatch
    # shows the dash pattern alongside the color.
    linestyles = [h.get_linestyle() for h in entry.handles]
    assert (4, 2) in linestyles and (1, 1) in linestyles


def test_lineplot_hue_equals_style_different_columns_kept_separate(line_df):
    """When hue and style reference *different* columns, keep them as two
    separate entries (the intentional two-legend case)."""
    ax = pp.lineplot(
        data=line_df, x="t", y="y",
        hue="g", style="m",
        palette="pastel",
    )
    kinds = {e.kind for e in get_entries(ax)}
    assert {"hue", "style"} <= kinds


def test_lineplot_continuous_hue_equals_style_not_merged(line_df):
    """Continuous hue (colorbar) cannot composite with a style swatch —
    keep them as separate entries. ``is_categorical`` returning True for
    a small-int numeric column would defeat this test, so use a truly
    continuous score column."""
    df = line_df.assign(score=np.linspace(0.0, 100.0, len(line_df)))
    ax = pp.lineplot(
        data=df, x="t", y="y",
        hue="score", style="score",
        palette="viridis", hue_norm=(0, 100),
        dashes=True,
    )
    kinds = [e.kind for e in get_entries(ax)]
    # hue stays as continuous hue (ScalarMappable), style stays separate.
    assert kinds.count("hue") == 1
    assert "style" in kinds


# ---- Categorical size ----

@pytest.fixture(scope="module")
def cat_size_df():
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "t": np.tile(np.linspace(0, 10, 20), 3),
        "y": rng.normal(size=60),
        "tier": np.repeat(["low", "med", "high"], 20),
    })


def test_lineplot_categorical_size_does_not_crash(cat_size_df):
    """Regression: previously crashed in get_size_ticks via np.isnan(str)."""
    ax = pp.lineplot(data=cat_size_df, x="t", y="y", size="tier")
    assert isinstance(ax, Axes)


def test_lineplot_categorical_size_stashes_one_handle_per_category(cat_size_df):
    ax = pp.lineplot(data=cat_size_df, x="t", y="y", size="tier",
                     sizes={"low": 0.5, "med": 2.0, "high": 5.0})
    [entry] = [e for e in get_entries(ax) if e.kind == "size"]
    assert entry.name == "tier"
    by_label = dict(zip(entry.labels, entry.handles))
    assert by_label["low"].get_linewidth() == 0.5
    assert by_label["med"].get_linewidth() == 2.0
    assert by_label["high"].get_linewidth() == 5.0


def test_lineplot_categorical_size_default_uses_tuple_interpolation(cat_size_df):
    """With no ``sizes`` kwarg, publiplots falls back to (1.0, 4.0)
    linearly across the categories."""
    ax = pp.lineplot(data=cat_size_df, x="t", y="y", size="tier")
    [entry] = [e for e in get_entries(ax) if e.kind == "size"]
    widths = [h.get_linewidth() for h in entry.handles]
    # Three categories → (1.0, 2.5, 4.0).
    assert widths == [1.0, 2.5, 4.0]


# ---- Double-layer marker styling ----

def _foreground_marker_lines(ax):
    """Pick only the double-layer foreground marker lines.

    publiplots' ``apply_double_layer_markers`` draws *three* artists
    per group: the original connecting line (ms=0, markers stripped);
    a white-background marker layer; and a foreground marker layer
    with alpha fill + opaque ring. We assert the contract on the
    foreground layer, detected by its transparent markerfacecolor.
    """
    from matplotlib.colors import to_rgba
    fg = []
    for line in ax.lines:
        if line.get_markersize() in (None, 0, 0.0):
            continue
        mfc = to_rgba(line.get_markerfacecolor())
        if mfc[3] < 1.0:  # alpha-filled layer
            fg.append(line)
    return fg


def test_lineplot_markers_use_double_layer_style(line_df):
    """Markers must follow publiplots' double-layer convention: face in
    the line's color at rcParams['alpha'] (pale fill), edge in the
    line's color at full opacity (solid ring). Seaborn's default is a
    white edge on an opaque pastel face — this test guards against
    regression back to that styling."""
    from matplotlib.colors import to_rgba
    from publiplots.themes.rcparams import resolve_param
    alpha_rc = resolve_param("alpha", None)

    ax = pp.lineplot(
        data=line_df, x="t", y="y",
        hue="g", style="m",
        palette="pastel", markers=True,
    )
    foreground = _foreground_marker_lines(ax)
    assert foreground, "no foreground marker layer produced"
    for line in foreground:
        mfc = to_rgba(line.get_markerfacecolor())
        mec = to_rgba(line.get_markeredgecolor())
        # Face RGB equals edge RGB; alphas differ (face dim, edge opaque).
        assert mfc[:3] == mec[:3]
        assert abs(mfc[3] - alpha_rc) < 1e-6
        assert mec[3] == 1.0


def test_lineplot_edgecolor_override_applies_to_markers(line_df):
    """``edgecolor=`` overrides the ring color while the face stays
    in the line's own color."""
    from matplotlib.colors import to_rgba
    ax = pp.lineplot(
        data=line_df, x="t", y="y",
        hue="g", style="m",
        palette="pastel", markers=True,
        edgecolor="black",
    )
    black = to_rgba("black")
    foreground = _foreground_marker_lines(ax)
    assert foreground
    for line in foreground:
        mec = to_rgba(line.get_markeredgecolor())
        assert mec[:3] == black[:3]


# ---- Reject ----

def test_lineplot_rejects_figsize(line_df):
    with pytest.raises(TypeError, match="figsize"):
        pp.lineplot(data=line_df, x="t", y="y", figsize=(4, 3))
