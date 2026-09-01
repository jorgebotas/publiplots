"""Exact-zero bar values must still produce a paired BarRecord (issue #199).

A group whose aggregate is exactly 0 is a *real* group that happens to be
zero, and matplotlib emits a Rectangle for it — degenerate only on the
value axis. Dropping it desynced the positional pairing between drawn
rects and aggregated group keys, shifting every label by one and silently
truncating the last category.
"""
import contextlib
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest

import publiplots as pp


@contextlib.contextmanager
def _no_mismatch_warning():
    """Fail if the bar/group count-mismatch warning is emitted."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield
    offenders = [str(w.message) for w in caught
                 if "does not match the number of data groups" in str(w.message)]
    assert not offenders, offenders


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _zero_df():
    return pd.DataFrame({"cat": list("ABCD"), "val": [0.0, 2.0, 3.0, 4.0]})


@pytest.mark.parametrize("zero_at", [0, 1, 3])
def test_zero_bar_keeps_pairing_horizontal(zero_at):
    """Horizontal orient: a zero-width rect is a real bar, not an empty group."""
    vals = [1.0, 2.0, 3.0, 4.0]
    vals[zero_at] = 0.0
    df = pd.DataFrame({"cat": list("ABCD"), "val": vals})

    fig, ax = pp.subplots(1, 1)
    pp.barplot(data=df, y="cat", x="val", order=list("ABCD"), legend=False, ax=ax)

    meta = ax._publiplots_bar_meta
    assert [b.category for b in meta.bars] == list("ABCD")
    assert [b.value for b in meta.bars] == vals
    # Each record must point at the patch drawn for its own category.
    assert [b.patch.get_width() for b in meta.bars] == vals


@pytest.mark.parametrize("zero_at", [0, 1, 3])
def test_zero_bar_keeps_pairing_vertical(zero_at):
    """Vertical orient: same invariant on the height axis."""
    vals = [1.0, 2.0, 3.0, 4.0]
    vals[zero_at] = 0.0
    df = pd.DataFrame({"cat": list("ABCD"), "val": vals})

    fig, ax = pp.subplots(1, 1)
    pp.barplot(data=df, x="cat", y="val", order=list("ABCD"), legend=False, ax=ax)

    meta = ax._publiplots_bar_meta
    assert [b.category for b in meta.bars] == list("ABCD")
    assert [b.value for b in meta.bars] == vals
    assert [b.patch.get_height() for b in meta.bars] == vals


def test_zero_bar_keeps_pairing_with_hue():
    """Zero values under a hue split keep hue/category pairing intact."""
    df = pd.DataFrame({
        "cat": ["A", "A", "B", "B"],
        "grp": ["x", "y", "x", "y"],
        "val": [0.0, 2.0, 3.0, 4.0],
    })
    fig, ax = pp.subplots(1, 1)
    pp.barplot(data=df, x="cat", y="val", hue="grp",
               order=["A", "B"], hue_order=["x", "y"], legend=False, ax=ax)

    meta = ax._publiplots_bar_meta
    # Draw order is hue-outer, category-inner.
    assert [(b.category, b.hue_value, b.value) for b in meta.bars] == [
        ("A", "x", 0.0),
        ("B", "x", 3.0),
        ("A", "y", 2.0),
        ("B", "y", 4.0),
    ]


def test_missing_hue_combo_pairs_only_real_groups():
    """A (cat, hue) combo with no observations yields no record.

    Note this does NOT exercise the both-extents-zero branch of
    `_is_bar_rect`: ``pp.barplot`` emits 4 patches here, none degenerate,
    because its category preparation never hands seaborn an empty combo.
    The filter's premise is exercised against raw ``sns.barplot`` in
    ``tests/annotate/test_pairing_scope.py``. What this pins is that the
    group aggregation skips the empty combos, so pairing stays 1:1.
    """
    df = pd.DataFrame({
        "cat": ["A", "A", "B", "C"],
        "grp": ["x", "y", "x", "y"],
        "val": [1.0, 2.0, 3.0, 4.0],
    })
    fig, ax = pp.subplots(1, 1)
    pp.barplot(data=df, x="cat", y="val", hue="grp", legend=False, ax=ax)

    meta = ax._publiplots_bar_meta
    assert len(ax.patches) == 4, "premise: publiplots emits no degenerate rects"
    assert len(meta.bars) == 4
    assert [(b.category, b.hue_value, b.value) for b in meta.bars] == [
        ("A", "x", 1.0),
        ("B", "x", 3.0),
        ("A", "y", 2.0),
        ("C", "y", 4.0),
    ]


def test_zero_bar_annotate_labels_land_on_own_bar():
    """End-to-end: the rendered label for a zero bar sits at that bar."""
    df = _zero_df()
    fig, ax = pp.subplots(1, 1)
    pp.barplot(data=df, x="cat", y="val", order=list("ABCD"),
               legend=False, ax=ax, annotate={"fmt": ".1f"})

    labels = [t.get_text() for t in ax.texts]
    assert labels == ["0.0", "2.0", "3.0", "4.0"]

    # Label i must be horizontally centred on category i's bar.
    for i, text in enumerate(ax.texts):
        rect = ax.patches[i]
        centre = rect.get_x() + rect.get_width() / 2.0
        assert text.get_position()[0] == pytest.approx(centre, abs=1e-6)


def test_rect_count_mismatch_warns():
    """Defense in depth: a desync between rects and group keys must be loud.

    Issue #199 shipped a mislabelled panel because ``zip`` truncated to the
    shorter list in silence. No supported ``pp.barplot`` call should reach
    this state — the guard exists so that a future geometry-filter change
    that reintroduces one fails loudly instead of mispairing labels — so
    the desync is constructed here by handing the builder a frame with more
    groups than the axes has bars.
    """
    from publiplots.annotate._builders import build_from_barplot_call

    two_cats = pd.DataFrame({"cat": list("AB"), "val": [1.0, 2.0]})
    fig, ax = pp.subplots(1, 1)
    pp.barplot(data=two_cats, x="cat", y="val", legend=False, ax=ax)

    with pytest.warns(UserWarning, match="does not match"):
        build_from_barplot_call(
            ax=ax, data=_zero_df(), x="cat", y="val", hue=None,
            categorical_axis="cat", palette=None, errorbar=None,
            source_frame=_zero_df(),
        )


def test_order_subset_does_not_warn():
    """``order=`` to a subset is a supported call and must stay warning-free.

    ``pp.barplot`` filters the frame to ``order`` before aggregating, so
    rects and group keys still match; the guard must not cry wolf here.
    """
    df = _zero_df()
    fig, ax = pp.subplots(1, 1)
    with _no_mismatch_warning():
        pp.barplot(data=df, x="cat", y="val", order=["A", "B"],
                   legend=False, ax=ax)

    meta = ax._publiplots_bar_meta
    assert [(b.category, b.value) for b in meta.bars] == [("A", 0.0), ("B", 2.0)]


def test_foreign_patch_on_axes_does_not_shift_labels():
    """A patch publiplots did not draw must not be mistaken for a bar.

    Same root cause as the zero-value desync: the builder scanned the whole
    axes instead of the patches this call produced. A user-drawn highlight
    Rectangle then took bar 0's slot and pushed every label one bar over.
    """
    import matplotlib.pyplot as _plt
    from matplotlib.patches import Rectangle

    fig, ax = _plt.subplots()
    ax.add_patch(Rectangle((0, 5), 0.5, 1, facecolor="red"))
    df = pd.DataFrame({"cat": ["A", "B"], "val": [1.0, 2.0]})
    pp.barplot(data=df, x="cat", y="val", legend=False, ax=ax)

    meta = ax._publiplots_bar_meta
    assert [(b.category, b.value) for b in meta.bars] == [("A", 1.0), ("B", 2.0)]


def test_foreign_patch_on_axes_does_not_shift_stacked_labels():
    """Same guarantee on the stacked path."""
    import matplotlib.pyplot as _plt
    from matplotlib.patches import Rectangle

    fig, ax = _plt.subplots()
    ax.add_patch(Rectangle((0, 5), 0.5, 1, facecolor="red"))
    df = pd.DataFrame({
        "cat": ["A", "A", "B", "B"],
        "grp": ["x", "y", "x", "y"],
        "val": [1.0, 2.0, 3.0, 4.0],
    })
    pp.barplot(data=df, x="cat", y="val", hue="grp", multiple="stack",
               legend=False, ax=ax)

    meta = ax._publiplots_bar_meta
    assert [(b.category, b.hue_value, b.value) for b in meta.bars] == [
        ("A", "x", 1.0),
        ("B", "x", 3.0),
        ("A", "y", 2.0),
        ("B", "y", 4.0),
    ]
