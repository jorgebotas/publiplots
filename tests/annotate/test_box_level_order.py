"""Box/violin level order must follow seaborn's, not first-occurrence (#262).

#254 fixed the *nesting* order (cat-outer for violin, hue-outer for box).
This is the *level* order within each dimension. Seaborn's
``categorical_order`` honours an explicit ``order=`` / ``hue_order=`` and
**sorts numeric levels**; ``_splits._categories_in_draw_order`` does
neither, so the aggregated groups came out in a different sequence from the
drawn artists and the index-based pairing put every label on the wrong mark.

Each test identifies the drawn artists independently: every ``(cat, hue)``
group is given a **disjoint value band**, so an artist's value extent says
which group it belongs to without reference to colour or to the meta.

``pp.barplot`` is unaffected, and two tests (four cases) guard that it
stays so. Its immunity has two sources: with an explicit order,
``_prepare_split_data`` rewrites the frame as an ordered Categorical; by
default, ``pp.barplot`` rejects a non-categorical-dtype axis outright and
runs ``as_categorical`` over the categorical axis and the hue. So it never
reaches first-occurrence order for a numeric level the way box and violin
did.
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


def _banded(cats, hues):
    """Frame where each (cat, hue) occupies its own disjoint value band."""
    rows, band = [], {}
    for ci, c in enumerate(cats):
        for hi, h in enumerate(hues):
            base = 100 * (ci * len(hues) + hi)
            band[(c, h)] = base
            rows += [{"cat": c, "h": h, "v": base + k} for k in range(5)]
    return pd.DataFrame(rows), band


def _owner(band, y0):
    """Which group owns the artist whose values start at ``y0``."""
    return min(band, key=lambda g: abs(band[g] - y0))


def _mispairs(kind, df, band, **kw):
    """Records whose position belongs to a different group. Empty == correct."""
    fig, ax = pp.subplots()
    getattr(pp, kind)(data=df, x="cat", y="v", hue="h", legend=False, ax=ax, **kw)

    if kind == "boxplot":
        extents = [p.get_path().get_extents() for p in ax.patches]
    else:
        extents = [c.get_paths()[0].get_extents() for c in ax.collections]
    drawn = {round(e.x0 + e.width / 2, 4): _owner(band, e.y0) for e in extents}

    bad = []
    for b in ax._publiplots_box_meta.boxes:
        key = round(b.center_pos, 4)
        if drawn.get(key) != (b.category, b.hue_value):
            bad.append(f"{b.category}/{b.hue_value}@{key} is really {drawn.get(key)}")
    return bad, len(extents), len(ax._publiplots_box_meta.boxes)


KINDS = ["boxplot", "violinplot"]


@pytest.mark.parametrize("kind", KINDS)
def test_default_level_order_is_unchanged(kind):
    """Control: the default ordering already worked and must keep working."""
    df, band = _banded(list("AB"), list("xy"))
    bad, n_art, n_rec = _mispairs(kind, df, band)
    assert not bad and n_art == n_rec == 4


@pytest.mark.parametrize("kind", KINDS)
def test_reversed_order(kind):
    df, band = _banded(list("AB"), list("xy"))
    bad, _, _ = _mispairs(kind, df, band, order=["B", "A"])
    assert not bad, bad


@pytest.mark.parametrize("kind", KINDS)
def test_reversed_hue_order(kind):
    df, band = _banded(list("AB"), list("xy"))
    bad, _, _ = _mispairs(kind, df, band, hue_order=["y", "x"])
    assert not bad, bad


@pytest.mark.parametrize("kind", KINDS)
def test_order_subset_filters_rather_than_truncates(kind):
    """``order=`` naming a subset must drop the other categories' records.

    Previously the aggregation produced six rows against four drawn
    artists and the builder truncated to the first four, so the meta
    described ``A`` and ``B`` while the axes showed ``B`` and ``C``.
    """
    df, band = _banded(list("ABC"), list("xy"))
    bad, n_art, n_rec = _mispairs(kind, df, band, order=["B", "C"])
    assert not bad, bad
    assert n_art == n_rec == 4
    fig, ax = pp.subplots()
    getattr(pp, kind)(data=df, x="cat", y="v", hue="h", order=["B", "C"],
                      legend=False, ax=ax)
    assert {b.category for b in ax._publiplots_box_meta.boxes} == {"B", "C"}


@pytest.mark.parametrize("kind", KINDS)
def test_three_level_hue_order(kind):
    """Three levels reordered: the classic off-by-one shows in the middle."""
    df, band = _banded(list("AB"), list("xyz"))
    bad, n_art, n_rec = _mispairs(kind, df, band, hue_order=["z", "x", "y"])
    assert not bad, bad
    assert n_art == n_rec == 6


@pytest.mark.parametrize("kind", KINDS)
def test_numeric_categories_are_sorted_like_seaborn(kind):
    """Numeric levels need no arguments at all — just unsorted rows.

    Seaborn sorts numeric levels; first-occurrence order does not, so a
    frame whose rows happen to start at 2 mispaired everything.
    """
    df, band = _banded([2, 1], list("xy"))
    bad, _, _ = _mispairs(kind, df, band)
    assert not bad, bad


@pytest.mark.parametrize("kind", KINDS)
def test_numeric_hue_levels_are_sorted_like_seaborn(kind):
    df, band = _banded(list("AB"), [2, 1])
    bad, _, _ = _mispairs(kind, df, band)
    assert not bad, bad


@pytest.mark.parametrize("kind", KINDS)
def test_explicit_order_beats_a_categorical_dtype(kind):
    """An explicit ``order=`` wins over the column's declared categories.

    That is seaborn's rule, and the pre-existing code took `.cat.categories`
    unconditionally.
    """
    df, band = _banded(list("AB"), list("xy"))
    df["cat"] = pd.Categorical(df["cat"], categories=["A", "B"])
    bad, _, _ = _mispairs(kind, df, band, order=["B", "A"])
    assert not bad, bad


# ---- pp.barplot already handled this; keep it that way ----

def _bar_frame(cats, hues):
    """Bars draw the aggregate, so a unique mean identifies each group."""
    rows, band = [], {}
    for ci, c in enumerate(cats):
        for hi, h in enumerate(hues):
            val = float(100 * (ci * len(hues) + hi) + 7)
            band[(c, h)] = val
            rows += [{"cat": c, "h": h, "v": val}] * 3
    return pd.DataFrame(rows), band


@pytest.mark.parametrize("kw", [
    {},
    {"order": ["B", "A"]},
    {"hue_order": ["y", "x"]},
])
def test_barplot_level_order_stays_correct(kw):
    """Guard: ``_prepare_split_data`` already rewrites the frame as an
    ordered Categorical, so the bar path was never affected."""
    df, band = _bar_frame(list("AB"), list("xy"))
    fig, ax = pp.subplots()
    pp.barplot(data=df, x="cat", y="v", hue="h", legend=False, ax=ax, **kw)

    for b in ax._publiplots_bar_meta.bars:
        assert b.value == pytest.approx(band[(b.category, b.hue_value)])


def test_barplot_order_subset_stays_correct():
    df, band = _bar_frame(list("ABC"), list("xy"))
    fig, ax = pp.subplots()
    pp.barplot(data=df, x="cat", y="v", hue="h", order=["B", "C"],
               legend=False, ax=ax)

    bars = ax._publiplots_bar_meta.bars
    assert {b.category for b in bars} == {"B", "C"}
    for b in bars:
        assert b.value == pytest.approx(band[(b.category, b.hue_value)])


# ---- object-dtype numerics and bools: seaborn calls these numeric too ----

@pytest.mark.parametrize("kind", KINDS)
def test_object_dtype_numeric_categories_are_sorted(kind):
    """Seaborn's numeric test is broader than the pandas dtype test.

    ``variable_type``'s ``all_numeric`` fallback treats an object-dtype
    column of ``Number`` entries as numeric and sorts it, where
    ``is_numeric_dtype`` returns False. Found by review after the first
    version of this fix, which still mispaired all six records here.
    """
    df, band = _banded([3, 1, 2], list("xy"))
    df["cat"] = df["cat"].astype(object)
    bad, n_art, n_rec = _mispairs(kind, df, band)
    assert not bad, bad
    assert n_art == n_rec == 6


@pytest.mark.parametrize("kind", KINDS)
def test_object_dtype_numeric_hue_levels_are_sorted(kind):
    df, band = _banded(list("AB"), [2, 1])
    df["h"] = df["h"].astype(object)
    bad, _, _ = _mispairs(kind, df, band)
    assert not bad, bad


@pytest.mark.parametrize("kind", KINDS)
def test_decimal_categories_are_sorted(kind):
    """``Decimal`` is a ``Number`` but not a numeric dtype."""
    from decimal import Decimal
    df, band = _banded([Decimal(3), Decimal(1), Decimal(2)], list("xy"))
    bad, _, _ = _mispairs(kind, df, band)
    assert not bad, bad


@pytest.mark.parametrize("kind", KINDS)
def test_bool_hue_levels_are_sorted(kind):
    """Bools are ``Number`` subclasses, so seaborn sorts them: False, True.

    First-occurrence order gave ``[True, False]`` for a frame starting at
    True, which mispaired both records.
    """
    df, band = _banded(list("AB"), [True, False])
    bad, _, _ = _mispairs(kind, df, band)
    assert not bad, bad


def test_resolver_matches_seaborns_across_dtypes():
    """Pin the equivalence directly, not only through the plotters.

    The whole fix rests on `_categorical_order` agreeing with seaborn's
    `categorical_order`; assert that against the real function so a
    seaborn change surfaces here rather than as a mispaired figure.
    """
    from decimal import Decimal
    from seaborn._base import categorical_order as sns_order
    from publiplots.annotate._builders import _categorical_order

    cases = [
        pd.Series([3, 1, 2]),
        pd.Series([3, 1, 2], dtype=object),
        pd.Series([Decimal(3), Decimal(1), Decimal(2)]),
        pd.Series([3.5, -1.0, 2.25]),
        pd.Series([True, False]),
        pd.Series(["b", "a", "c"]),
        pd.Series(["10", "9", "2"]),
        pd.Series([2.0, None, 1.0]),
        pd.Series(pd.Categorical(list("ba"), categories=["b", "a"])),
        pd.Series(pd.Categorical(list("ab"), categories=["a", "b", "z"])),
        pd.Series(pd.to_datetime(["2026-02-01", "2026-01-01"])),
    ]
    for ser in cases:
        assert list(_categorical_order(ser)) == list(sns_order(ser)), (
            f"diverged for {ser.dtype} {list(ser)!r}"
        )

    # An explicit order wins verbatim in both, including a superset.
    for ser in (pd.Series(list("ab")), pd.Series([1, 2])):
        for order in ([*ser.tolist()][::-1], [*ser.tolist(), "zz"]):
            assert list(_categorical_order(ser, order)) == list(sns_order(ser, order))
