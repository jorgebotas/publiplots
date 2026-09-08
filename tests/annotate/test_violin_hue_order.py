"""Violin hue-split records must take the position of their own violin (#254).

Seaborn's violinplot draws **cat-outer, hue-inner**; its boxplot is
effectively **hue-outer, cat-inner** (one ``bxp`` call per hue level).
``_aggregate_box_stats`` used the boxplot nesting for both, and
``_build_box_stats_meta`` zips groups to artists by index, so violin
records between the first and the last took a neighbour's position. Stats
were unaffected — computed from the group key, not the artist — so only
``center_pos`` / ``cat_half_width`` were wrong, which is what places the
label.

Scope: this file covers the **nesting** order only. The *level* order
within each dimension is a separate, still-open bug (#262) affecting both
plotters — ``_categories_in_draw_order`` ignores explicit ``order=`` /
``hue_order=`` and does not sort numeric levels the way seaborn does — so
every test here uses default level ordering with string levels.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_hex

import publiplots as pp


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


PAL2 = {"x": "#1f77b4", "y": "#d62728"}
PAL3 = {"x": "#1f77b4", "y": "#d62728", "z": "#2ca02c"}


def _df2():
    return pd.DataFrame({"cat": list("AABB") * 5, "h": list("xy") * 10,
                         "v": list(range(20))})


def _df3():
    rows = []
    for ci, c in enumerate("AB"):
        for hi, h in enumerate("xyz"):
            for k in range(5):
                rows.append({"cat": c, "h": h, "v": 10 * ci + hi + k})
    return pd.DataFrame(rows)


def _medians(df):
    return {(c, h): float(np.median(g["v"]))
            for (c, h), g in df.groupby(["cat", "h"], observed=True)}


def _hue_of(rgba, palette):
    """Recover a hue level from a drawn artist's facecolor."""
    hx = to_hex(rgba[:3] if len(rgba) > 3 else rgba)
    px = lambda s: [int(s[i:i + 2], 16) for i in (1, 3, 5)]
    return min(palette, key=lambda k: sum((a - b) ** 2
                                          for a, b in zip(px(hx), px(palette[k]))))


def _drawn_violins(ax, palette):
    """(centre, hue) per drawn violin, hue recovered from its own colour."""
    from matplotlib.collections import PolyCollection
    out = []
    for coll in ax.collections:
        if not isinstance(coll, PolyCollection) or not coll.get_paths():
            continue
        ext = coll.get_paths()[0].get_extents()
        out.append((ext.x0 + ext.width / 2.0,
                    _hue_of(coll.get_facecolor()[0], palette)))
    return out


def test_violin_hue_records_take_their_own_violins_position():
    """Each record's centre must be the centre of the violin for its group.

    The drawn violins are identified independently — position plus the hue
    recovered from each artist's own facecolor — rather than trusting the
    order the builder produced.
    """
    df = _df2()
    fig, ax = pp.subplots()
    pp.violinplot(data=df, x="cat", y="v", hue="h", palette=PAL2,
                  legend=False, ax=ax)

    drawn = _drawn_violins(ax, PAL2)
    cats = ["A", "B"]
    # centre -> (category, hue), read off the drawing, not the meta
    by_centre = {round(c, 6): (cats[int(round(c))], h) for c, h in drawn}
    assert len(by_centre) == 4, "precondition: four distinct violins"

    for b in ax._publiplots_box_meta.boxes:
        key = round(b.center_pos, 6)
        assert key in by_centre, f"{b.category}/{b.hue_value} at unknown centre {key}"
        assert by_centre[key] == (b.category, b.hue_value), (
            f"record {b.category}/{b.hue_value} sits at {key}, which is the "
            f"{by_centre[key][0]}/{by_centre[key][1]} violin"
        )


def test_violin_hue_stats_still_match_their_group():
    """Guard: stats were always right; the fix must not disturb them."""
    df = _df2()
    truth = _medians(df)
    fig, ax = pp.subplots()
    pp.violinplot(data=df, x="cat", y="v", hue="h", palette=PAL2,
                  legend=False, ax=ax)

    for b in ax._publiplots_box_meta.boxes:
        assert b.stats["median"] == pytest.approx(
            truth[(b.category, b.hue_value)])


def test_violin_hue_record_order_is_cat_outer():
    """Pin the order explicitly, since it is what the zip depends on."""
    fig, ax = pp.subplots()
    pp.violinplot(data=_df2(), x="cat", y="v", hue="h", palette=PAL2,
                  legend=False, ax=ax)

    got = [(b.category, b.hue_value) for b in ax._publiplots_box_meta.boxes]
    assert got == [("A", "x"), ("A", "y"), ("B", "x"), ("B", "y")]

    centres = [b.center_pos for b in ax._publiplots_box_meta.boxes]
    assert centres == pytest.approx([-0.2, 0.2, 0.8, 1.2], abs=1e-6)


def test_violin_hue_split_takes_its_own_position():
    """``split=True`` draws half-violins in the same cat-outer order.

    A split half's centre is not its group's nominal centre — the halves sit
    inboard — so the position is checked against the drawn artists rather
    than against hardcoded offsets.
    """
    df = _df2()
    fig, ax = pp.subplots()
    pp.violinplot(data=df, x="cat", y="v", hue="h", split=True, palette=PAL2,
                  legend=False, ax=ax)

    got = [(b.category, b.hue_value) for b in ax._publiplots_box_meta.boxes]
    assert got == [("A", "x"), ("A", "y"), ("B", "x"), ("B", "y")]

    drawn = _drawn_violins(ax, PAL2)
    cats = ["A", "B"]
    by_centre = {round(c, 6): (cats[int(round(c))], h) for c, h in drawn}
    assert len(by_centre) == 4, "precondition: four distinct half-violins"
    for b in ax._publiplots_box_meta.boxes:
        assert by_centre[round(b.center_pos, 6)] == (b.category, b.hue_value)


def test_violin_three_hue_levels():
    """Three levels: the middle ones are where an off-by-order shows up."""
    df = _df3()
    fig, ax = pp.subplots()
    pp.violinplot(data=df, x="cat", y="v", hue="h", palette=PAL3,
                  legend=False, ax=ax)

    drawn = _drawn_violins(ax, PAL3)
    cats = ["A", "B"]
    by_centre = {round(c, 6): (cats[int(round(c))], h) for c, h in drawn}
    assert len(by_centre) == 6

    for b in ax._publiplots_box_meta.boxes:
        assert by_centre[round(b.center_pos, 6)] == (b.category, b.hue_value)


def test_violin_hue_horizontal_orient():
    """The categorical axis being y must not change the pairing."""
    df = _df2()
    fig, ax = pp.subplots()
    pp.violinplot(data=df, y="cat", x="v", hue="h", palette=PAL2,
                  legend=False, ax=ax)

    got = [(b.category, b.hue_value) for b in ax._publiplots_box_meta.boxes]
    assert got == [("A", "x"), ("A", "y"), ("B", "x"), ("B", "y")]
    centres = [b.center_pos for b in ax._publiplots_box_meta.boxes]
    assert centres == pytest.approx([-0.2, 0.2, 0.8, 1.2], abs=1e-6)


def test_violin_custom_labels_land_on_their_own_violin():
    """End-to-end: the rendered label must sit at its group's violin."""
    df = pd.DataFrame({
        "cat": pd.Categorical(["A"] * 10 + ["B"] * 10, categories=["A", "B"]),
        "h": pd.Categorical((["x"] * 5 + ["y"] * 5) * 2, categories=["x", "y"]),
        "v": list(range(1, 21)),
        "n": [11] * 5 + [22] * 5 + [33] * 5 + [44] * 5,
    })
    fig, ax = pp.subplots()
    pp.violinplot(data=df, x="cat", y="v", hue="h", palette=PAL2,
                  legend=False, ax=ax)
    texts = pp.annotate(ax, kind="violin_custom", labels="n")

    # n identifies the group: 11=A/x, 22=A/y, 33=B/x, 44=B/y.
    expected_x = {"11": -0.2, "22": 0.2, "33": 0.8, "44": 1.2}
    for t in texts:
        assert t.get_position()[0] == pytest.approx(
            expected_x[t.get_text()], abs=1e-6), (
            f"label {t.get_text()!r} drawn at x={t.get_position()[0]:.3f}, "
            f"expected {expected_x[t.get_text()]}"
        )


# ---- boxplot must keep its own (different) order ----

def test_boxplot_hue_order_stays_hue_outer():
    """matplotlib's ``bxp`` really does draw hue-outer — do not "fix" it."""
    df = _df2()
    fig, ax = pp.subplots()
    pp.boxplot(data=df, x="cat", y="v", hue="h", palette=PAL2,
               legend=False, ax=ax)

    got = [(b.category, b.hue_value) for b in ax._publiplots_box_meta.boxes]
    assert got == [("A", "x"), ("B", "x"), ("A", "y"), ("B", "y")]

    centres = [b.center_pos for b in ax._publiplots_box_meta.boxes]
    assert centres == pytest.approx([-0.2, 0.8, 0.2, 1.2], abs=1e-6)


def test_boxplot_hue_records_take_their_own_boxes_position():
    """Same independent check as for violin, on the box path."""
    df = _df2()
    fig, ax = pp.subplots()
    pp.boxplot(data=df, x="cat", y="v", hue="h", palette=PAL2,
               legend=False, ax=ax)

    cats = ["A", "B"]
    by_centre = {}
    for p in ax.patches:
        ext = p.get_path().get_extents()
        c = ext.x0 + ext.width / 2.0
        by_centre[round(c, 6)] = (cats[int(round(c))],
                                  _hue_of(p.get_facecolor(), PAL2))
    assert len(by_centre) == 4

    for b in ax._publiplots_box_meta.boxes:
        assert by_centre[round(b.center_pos, 6)] == (b.category, b.hue_value)


def test_violin_without_hue_is_unchanged():
    """Guard: no hue means no ordering question."""
    df = _df2()
    fig, ax = pp.subplots()
    pp.violinplot(data=df, x="cat", y="v", legend=False, ax=ax)

    got = [(b.category, round(b.center_pos, 6))
           for b in ax._publiplots_box_meta.boxes]
    assert got == [("A", -0.0), ("B", 1.0)]
