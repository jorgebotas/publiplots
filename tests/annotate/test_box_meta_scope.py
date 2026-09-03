"""Box/violin meta must pair against the artists its own call drew (#241).

Last member of the whole-axes-scan family behind #103, #199 and #236.
``build_from_boxplot_call`` filtered all of ``ax.patches`` for
``(PathPatch, _RoundedBarPatch)`` and ``build_from_violinplot_call`` filtered
all of ``ax.collections`` for ``PolyCollection``, so anything else on the
axes matching those types took a box's slot in the positional pairing.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest

import publiplots as pp


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _ab():
    return pd.DataFrame({"cat": ["A"] * 6 + ["B"] * 6, "v": [1, 2, 3, 4, 5, 6] * 2})


def _cd():
    return pd.DataFrame({"cat": ["C"] * 6 + ["D"] * 6,
                         "v": [10, 20, 30, 40, 50, 60] * 2})


def _summary(ax):
    """Categories plus rounded geometry — rounded because seaborn's
    categorical positions carry float noise (2.9999999999999996 for 3.0)."""
    return [(b.category, round(b.center_pos, 6), round(b.cat_half_width, 4))
            for b in ax._publiplots_box_meta.boxes]


# ---- boxplot ----

def test_boxplot_meta_ignores_a_rounded_barplots_patches():
    """The reported case: rounded bars are ``_RoundedBarPatch``, so they
    matched the box builder's filter and supplied the box geometry.

    ``pp.barplot`` draws at width 0.72, ``pp.boxplot`` at 0.8, so the
    half-width is the tell.
    """
    df = _ab()
    fig, ax = pp.subplots()
    pp.barplot(data=df, x="cat", y="v", border_radius=1.5, legend=False, ax=ax)
    pp.boxplot(data=df, x="cat", y="v", legend=False, ax=ax)

    assert _summary(ax) == [("A", -0.0, 0.4), ("B", 1.0, 0.4)]


def test_boxplot_meta_unaffected_by_an_unrounded_barplot():
    """Control: plain ``Rectangle`` bars never matched the filter."""
    df = _ab()
    fig, ax = pp.subplots()
    pp.barplot(data=df, x="cat", y="v", legend=False, ax=ax)
    pp.boxplot(data=df, x="cat", y="v", legend=False, ax=ax)

    assert _summary(ax) == [("A", -0.0, 0.4), ("B", 1.0, 0.4)]


def test_second_boxplot_call_pairs_with_its_own_boxes():
    """Two ``pp.boxplot`` calls on one axes, with distinct categories.

    Categorical units accumulate, so the second call's boxes are drawn at
    2.0 and 3.0. Scanning the whole axes paired ``C``/``D`` against the
    first call's boxes at 0.0 and 1.0 — every label on the wrong box.
    """
    fig, ax = pp.subplots()
    pp.boxplot(data=_ab(), x="cat", y="v", legend=False, ax=ax)
    pp.boxplot(data=_cd(), x="cat", y="v", legend=False, ax=ax)

    centres = [round(p.get_path().get_extents().x0
                     + p.get_path().get_extents().width / 2, 3)
               for p in ax.patches]
    assert centres == [0.0, 1.0, 2.0, 3.0], "precondition: 4 boxes, accumulating"

    boxes = ax._publiplots_box_meta.boxes
    assert [b.category for b in boxes] == ["C", "D"]
    assert [b.center_pos for b in boxes] == pytest.approx([2.0, 3.0], abs=1e-9)


def test_boxplot_meta_survives_a_foreign_pathpatch():
    """A user-drawn PathPatch must not take a box's slot."""
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch

    fig, ax = pp.subplots()
    ax.add_patch(PathPatch(Path([(0, 0), (1, 0), (1, 1), (0, 0)]),
                           facecolor="red"))
    pp.boxplot(data=_ab(), x="cat", y="v", legend=False, ax=ax)

    assert _summary(ax) == [("A", -0.0, 0.4), ("B", 1.0, 0.4)]


# ---- violinplot ----

def test_violin_meta_ignores_a_filled_kdeplots_collection():
    """A filled kdeplot emits a ``FillBetweenPolyCollection``.

    That is a ``PolyCollection`` subclass, so it matched the violin
    builder's filter and took category A's slot — reporting a centre of 3.5
    and a half-extent of 5.7555 where A should be -0.0 and 0.4. B was
    displaced onto A's violin at the same time, and the second violin's
    record was dropped outright by the length-truncating zip, so the whole
    summary is asserted rather than just A's row.
    """
    df = _ab()
    fig, ax = pp.subplots()
    pp.kdeplot(data=df, x="v", fill=True, legend=False, ax=ax)
    pp.violinplot(data=df, x="cat", y="v", legend=False, ax=ax)

    assert _summary(ax) == [("A", -0.0, 0.4), ("B", 1.0, 0.4)]


def test_second_violin_call_pairs_with_its_own_violins():
    """Same accumulating-position problem on the violin path."""
    fig, ax = pp.subplots()
    pp.violinplot(data=_ab(), x="cat", y="v", legend=False, ax=ax)
    pp.violinplot(data=_cd(), x="cat", y="v", legend=False, ax=ax)

    boxes = ax._publiplots_box_meta.boxes
    assert [b.category for b in boxes] == ["C", "D"]
    assert [b.center_pos for b in boxes] == pytest.approx([2.0, 3.0], abs=1e-9)


# ---- guards: the scoping must not break the ordinary paths ----

def test_boxplot_alone_is_unchanged():
    fig, ax = pp.subplots()
    pp.boxplot(data=_ab(), x="cat", y="v", legend=False, ax=ax)
    assert _summary(ax) == [("A", -0.0, 0.4), ("B", 1.0, 0.4)]


def test_boxplot_with_border_radius_alone_is_unchanged():
    """Rounding swaps in `_RoundedBarPatch`; the tracker must still find them."""
    fig, ax = pp.subplots()
    pp.boxplot(data=_ab(), x="cat", y="v", border_radius=1.5,
               legend=False, ax=ax)
    assert _summary(ax) == [("A", -0.0, 0.4), ("B", 1.0, 0.4)]


def test_boxplot_hue_dodge_is_unchanged():
    df = pd.DataFrame({
        "cat": list("AABB") * 5,
        "h": list("xy") * 10,
        "v": [1, 5, 2, 6, 3, 7, 4, 8, 5, 9, 6, 10, 7, 11, 8, 12, 9, 13, 10, 14],
    })
    fig, ax = pp.subplots()
    pp.boxplot(data=df, x="cat", y="v", hue="h", legend=False, ax=ax)

    got = [(b.category, b.hue_value) for b in ax._publiplots_box_meta.boxes]
    assert got == [("A", "x"), ("B", "x"), ("A", "y"), ("B", "y")]


def test_violinplot_alone_is_unchanged():
    fig, ax = pp.subplots()
    pp.violinplot(data=_ab(), x="cat", y="v", legend=False, ax=ax)
    assert _summary(ax) == [("A", -0.0, 0.4), ("B", 1.0, 0.4)]


def test_raincloud_still_builds_its_meta():
    """raincloud composes violin + box + strip on one axes.

    The composition is exactly the shape this scoping change touches, so
    pin that it still produces a meta rather than an empty one.
    """
    fig, ax = pp.subplots()
    pp.raincloudplot(data=_ab(), x="cat", y="v", legend=False, ax=ax)
    assert len(ax._publiplots_box_meta.boxes) == 2


# ---- histplot: same family, found while reviewing this change ----

def test_histplot_meta_ignores_another_plots_bars():
    """A ``pp.barplot`` on the axes must not have its bars labelled as bins.

    ``build_from_histplot_call`` enumerates rather than pairing positionally,
    so a stray patch adds a spurious label rather than shifting the others —
    but the bars still got labelled with their own values alongside the bin
    counts.
    """
    bars = pd.DataFrame({"cat": list("AB"), "v": [3.5, 4.0]})
    fig, ax = pp.subplots()
    pp.barplot(data=bars, x="cat", y="v", legend=False, ax=ax)
    pp.histplot(data=pd.DataFrame({"v": [1, 2, 3, 4, 5, 6]}), x="v", bins=3,
                legend=False, ax=ax, annotate=True)

    assert len(ax._publiplots_bar_meta.bars) == 3, "3 bins, not 3 bins + 2 bars"
    assert [t.get_text() for t in ax.texts] == ["2.00", "2.00", "2.00"]


def test_histplot_alone_is_unchanged():
    """Guard: the ordinary single-call path still labels every bin."""
    fig, ax = pp.subplots()
    pp.histplot(data=pd.DataFrame({"v": [1, 2, 3, 4, 5, 6]}), x="v", bins=3,
                legend=False, ax=ax, annotate=True)
    assert [t.get_text() for t in ax.texts] == ["2.00", "2.00", "2.00"]


# ---- the drawn_artists=None fallback ----

def test_drawn_artists_none_still_scans_the_whole_axes():
    """Pin the fallback branch, so it cannot rot uncovered.

    Every caller passes ``drawn_artists`` now. Without a test, a future
    caller that forgets the argument would silently reinstate the bug this
    module is about and nothing would fail.
    """
    from publiplots.annotate._builders import build_from_boxplot_call

    df = _ab()
    fig, ax = pp.subplots()
    pp.barplot(data=df, x="cat", y="v", border_radius=1.5, legend=False, ax=ax)
    pp.boxplot(data=df, x="cat", y="v", legend=False, ax=ax)

    # Omitting drawn_artists reproduces the pre-fix contamination: the
    # rounded bars are picked up and supply the geometry.
    meta = build_from_boxplot_call(
        ax=ax, data=df, x="cat", y="v", hue=None, categorical_axis="cat",
        palette=None, whis=1.5, source_frame=df,
    )
    assert [round(b.cat_half_width, 4) for b in meta.boxes] == [0.36, 0.36]

    # Passing them explicitly is what the plot function does, and is correct.
    from publiplots.utils.rounding import _RoundedBarPatch
    from matplotlib.patches import PathPatch
    own = [p for p in ax.patches
           if isinstance(p, PathPatch) and not isinstance(p, _RoundedBarPatch)]
    scoped = build_from_boxplot_call(
        ax=ax, data=df, x="cat", y="v", hue=None, categorical_axis="cat",
        palette=None, whis=1.5, source_frame=df, drawn_artists=own,
    )
    assert [round(b.cat_half_width, 4) for b in scoped.boxes] == [0.4, 0.4]
