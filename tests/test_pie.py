"""Tests for pp.pieplot."""
from __future__ import annotations

import warnings

import matplotlib
matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.patches import Wedge

import publiplots as pp


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


@pytest.fixture
def df():
    return pd.DataFrame({
        "kind": ["symptom", "admin", "lay_dx", "none"],
        "n": [52.2, 41.8, 6.0, 4.0],
    })


def test_draws_one_wedge_per_category(df):
    fig, ax = pp.subplots()
    pp.pieplot(data=df, x="kind", y="n", legend=False, ax=ax)
    assert len([p for p in ax.patches if isinstance(p, Wedge)]) == 4


def test_every_record_holds_an_artist_that_is_on_the_axes(df):
    """The property that makes the index-pairing bug family unreachable."""
    fig, ax = pp.subplots()
    pp.pieplot(data=df, x="kind", y="n", legend=False, ax=ax)
    meta = ax._publiplots_wedge_meta
    drawn = [p for p in ax.patches if isinstance(p, Wedge)]
    assert len(meta.wedges) == len(drawn)
    for record in meta.wedges:
        assert any(record.patch is p for p in drawn)


def test_records_describe_their_own_artist_after_a_second_call(df):
    """Two pies on one axes: records must not pair against the first call."""
    other = pd.DataFrame({"kind": ["p", "q"], "n": [1.0, 3.0]})
    fig, ax = pp.subplots()
    pp.pieplot(data=df, x="kind", y="n", legend=False, ax=ax)
    pp.pieplot(data=other, x="kind", y="n", legend=False, ax=ax)
    meta = ax._publiplots_wedge_meta
    assert [r.path[0] for r in meta.wedges] == ["p", "q"]
    assert [r.fraction_of_total for r in meta.wedges] == pytest.approx(
        [0.25, 0.75])
    for record in meta.wedges:
        assert record.patch.theta2 - record.patch.theta1 == pytest.approx(
            record.theta1 - record.theta0)


def test_value_omitted_counts_rows():
    raw = pd.DataFrame({"kind": list("AABBB")})
    fig, ax = pp.subplots()
    pp.pieplot(data=raw, x="kind", legend=False, ax=ax)
    meta = ax._publiplots_wedge_meta
    assert [r.value for r in meta.wedges] == pytest.approx([2.0, 3.0])


def test_shares_a_palette_dict_with_barplot(df):
    """#268's motivating case: a pie beside a bar panel, same colours."""
    palette = {"symptom": "#75b375", "admin": "#8e8ec1",
               "lay_dx": "#eeaa58", "none": "#e67e7e"}
    fig, axes = pp.subplots(ncols=2)
    pp.barplot(data=df, x="kind", y="n", palette=palette, legend=False,
               ax=axes[0])
    pp.pieplot(data=df, x="kind", y="n", palette=palette, legend=False,
               ax=axes[1])

    import matplotlib.colors as mcolors
    pie_colors = {r.path[0]: tuple(r.hue_color)
                  for r in axes[1]._publiplots_wedge_meta.wedges}
    for category, spec in palette.items():
        assert pie_colors[category] == pytest.approx(mcolors.to_rgb(spec))


def test_wedges_inherit_the_theme_edge(df):
    fig, ax = pp.subplots()
    pp.pieplot(data=df, x="kind", y="n", legend=False, ax=ax)
    wedge = next(p for p in ax.patches if isinstance(p, Wedge))
    assert wedge.get_linewidth() == pytest.approx(pp.rcParams["edgewidth"])
    assert wedge.get_edgecolor()[3] > 0, "edge must not be fully transparent"


def test_spans_sum_to_a_full_circle(df):
    fig, ax = pp.subplots()
    pp.pieplot(data=df, x="kind", y="n", legend=False, ax=ax)
    spans = [r.theta1 - r.theta0 for r in ax._publiplots_wedge_meta.wedges]
    assert sum(spans) == pytest.approx(360.0)


def test_starts_at_twelve_oclock_clockwise_by_default(df):
    fig, ax = pp.subplots()
    pp.pieplot(data=df, x="kind", y="n", legend=False, ax=ax)
    first = ax._publiplots_wedge_meta.wedges[0]
    assert first.theta1 == pytest.approx(90.0)


def test_donut_leaves_a_hole(df):
    fig, ax = pp.subplots()
    pp.pieplot(data=df, x="kind", y="n", donut=0.5, legend=False, ax=ax)
    meta = ax._publiplots_wedge_meta
    assert meta.donut == pytest.approx(0.5)
    for record in meta.wedges:
        assert record.r_inner == pytest.approx(0.5)


def test_center_label_is_drawn_only_with_a_donut(df):
    fig, ax = pp.subplots()
    pp.pieplot(data=df, x="kind", y="n", donut=0.5,
               center_label="{:.0f}".format, legend=False, ax=ax)
    assert [t.get_text() for t in ax.texts] == ["104"]

    fig2, ax2 = pp.subplots()
    pp.pieplot(data=df, x="kind", y="n", center_label="{:.0f}".format,
               legend=False, ax=ax2)
    assert list(ax2.texts) == [], "no hole, nowhere to put the label"


def test_center_label_accepts_a_plain_string(df):
    fig, ax = pp.subplots()
    pp.pieplot(data=df, x="kind", y="n", donut=0.5, center_label="total",
               legend=False, ax=ax)
    assert [t.get_text() for t in ax.texts] == ["total"]


def test_disc_is_round_on_a_non_square_cell(df):
    """Pins the #101 avoidance: circular, and no allocated width lost."""
    fig, ax = pp.subplots(axes_size=(70, 50))
    pp.pieplot(data=df, x="kind", y="n", legend=False, ax=ax)
    fig.canvas.draw()
    (x0, y0), (x1, y1) = ax.transData.transform([(0.0, 0.0), (1.0, 1.0)])
    assert (x1 - x0) == pytest.approx(y1 - y0, rel=1e-6)
    assert ax.get_position().width == pytest.approx(
        ax.get_position(original=True).width, rel=1e-9)


def test_aspect_equal_is_available(df):
    fig, ax = pp.subplots(axes_size=(70, 50))
    pp.pieplot(data=df, x="kind", y="n", aspect="equal", legend=False, ax=ax)
    assert ax.get_aspect() == 1.0


def test_order_wins(df):
    fig, ax = pp.subplots()
    with pytest.warns(UserWarning, match="renormalise"):
        pp.pieplot(data=df, x="kind", y="n", order=["none", "admin"],
                   legend=False, ax=ax)
    assert [r.path[0] for r in ax._publiplots_wedge_meta.wedges] == [
        "none", "admin"]


def test_order_that_drops_a_category_warns_and_names_it(df):
    """order= alone renormalises the pie to a false 100%: `none` is 4 of
    104 in the frame but 8.7% of the two-wedge chart, and annotate=True
    would print that number. The branch refuses order= + min_share for
    exactly this hazard, so order= alone must at least say so."""
    fig, ax = pp.subplots()
    with pytest.warns(UserWarning) as rec:
        pp.pieplot(data=df, x="kind", y="n", order=["none", "admin"],
                   legend=False, ax=ax)
    message = str(rec[0].message)
    assert "symptom" in message and "lay_dx" in message
    assert "renormalise" in message
    shares = [r.fraction_of_total for r in ax._publiplots_wedge_meta.wedges]
    assert sum(shares) == pytest.approx(1.0)
    assert shares[0] == pytest.approx(4.0 / 45.8)      # not 4.0 / 104.0


def test_an_order_naming_every_category_does_not_warn(df):
    fig, ax = pp.subplots()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        pp.pieplot(data=df, x="kind", y="n",
                   order=["none", "admin", "lay_dx", "symptom"],
                   legend=False, ax=ax)


@pytest.fixture
def unsorted():
    """Deliberately not in value order -- the fixture ``df`` already is,
    which would let sort= pass while doing nothing."""
    return pd.DataFrame({
        "kind": ["lay_dx", "symptom", "none", "admin"],
        "n": [6.0, 52.2, 4.0, 41.8],
    })


def test_sort_orders_by_value(unsorted):
    fig, ax = pp.subplots()
    pp.pieplot(data=unsorted, x="kind", y="n", sort=True, legend=False, ax=ax)
    records = ax._publiplots_wedge_meta.wedges
    assert [r.path[0] for r in records] == [
        "symptom", "admin", "lay_dx", "none"]
    assert [r.value for r in records] == sorted(
        [r.value for r in records], reverse=True)


def test_unsorted_by_default_keeps_the_frames_order(unsorted):
    """A pie beside a bar panel must not silently reorder its categories."""
    fig, ax = pp.subplots()
    pp.pieplot(data=unsorted, x="kind", y="n", legend=False, ax=ax)
    assert [r.path[0] for r in ax._publiplots_wedge_meta.wedges] == [
        "lay_dx", "symptom", "none", "admin"]


def test_pull_offsets_a_named_category(df):
    fig, ax = pp.subplots()
    pp.pieplot(data=df, x="kind", y="n", pull={"admin": 0.2}, legend=False,
               ax=ax)
    by_path = {r.path[0]: r for r in ax._publiplots_wedge_meta.wedges}
    assert by_path["admin"].pull == pytest.approx(0.2)
    assert by_path["symptom"].pull == pytest.approx(0.0)
    assert by_path["admin"].patch.center != (0.0, 0.0)


def test_pull_as_a_float_moves_every_wedge(df):
    fig, ax = pp.subplots()
    pp.pieplot(data=df, x="kind", y="n", pull=0.1, legend=False, ax=ax)
    assert all(r.pull == pytest.approx(0.1)
               for r in ax._publiplots_wedge_meta.wedges)


def test_min_share_buckets_small_categories(df):
    fig, ax = pp.subplots()
    pp.pieplot(data=df, x="kind", y="n", min_share=0.1, legend=False, ax=ax)
    paths = [r.path[0] for r in ax._publiplots_wedge_meta.wedges]
    assert paths == ["symptom", "admin", "Other"]
    other = ax._publiplots_wedge_meta.wedges[-1]
    assert other.value == pytest.approx(10.0)      # 6.0 + 4.0


def test_min_share_leaves_a_lone_small_category_named():
    """Bucketing one category renames it without merging anything."""
    raw = pd.DataFrame({"kind": list("abc"), "n": [60.0, 38.0, 2.0]})
    fig, ax = pp.subplots()
    pp.pieplot(data=raw, x="kind", y="n", min_share=0.1, legend=False, ax=ax)
    assert [r.path[0] for r in ax._publiplots_wedge_meta.wedges] == [
        "a", "b", "c"]


def test_the_other_bucket_sorts_last_whatever_the_frame_order():
    raw = pd.DataFrame({
        "kind": ["lay_dx", "none", "symptom", "admin"],
        "n": [6.0, 4.0, 52.2, 41.8],
    })
    fig, ax = pp.subplots()
    pp.pieplot(data=raw, x="kind", y="n", min_share=0.1, legend=False, ax=ax)
    assert [r.path[0] for r in ax._publiplots_wedge_meta.wedges] == [
        "symptom", "admin", "Other"]


def test_min_share_keeps_a_categoricals_declared_order():
    """The correspondence with a bar panel is the point of the feature.

    Bucketing used to cast the column to object, which threw the declared
    category order away: barplot drew e, d, c, b, a and the pie drew
    a, b, c, Other.
    """
    raw = pd.DataFrame({
        "kind": pd.Categorical(list("abcde") * 4,
                               categories=list("edcba"), ordered=True),
        "n": [10.0, 9.0, 8.0, 0.4, 0.3] * 4,
    })
    fig, ax = pp.subplots()
    pp.barplot(data=raw, x="kind", y="n", legend=False, ax=ax)
    bar_order = [t.get_text() for t in ax.get_xticklabels()]
    assert bar_order == ["e", "d", "c", "b", "a"]

    fig2, ax2 = pp.subplots()
    pp.pieplot(data=raw, x="kind", y="n", min_share=0.05, legend=False,
               ax=ax2)
    paths = [r.path[0] for r in ax2._publiplots_wedge_meta.wedges]
    # d and e bucket; the survivors keep the declared order, not the
    # alphabetical one an object cast would have produced.
    assert paths == ["c", "b", "a", "Other"]


def test_a_hatch_column_typo_names_hatch_not_hue(df):
    """hatch= resolves through the same code as hue=; the message must
    still name the keyword the caller actually passed."""
    fig, ax = pp.subplots()
    with pytest.raises(ValueError, match=r"hatch='zzz'"):
        pp.pieplot(data=df, x="kind", y="n", hatch="zzz", legend=False, ax=ax)


def test_a_non_constant_hatch_column_names_hatch_not_hue():
    raw = pd.DataFrame({
        "kind": list("aabb"),
        "n": [1.0, 2.0, 3.0, 4.0],
        "g": list("wxyz"),
    })
    fig, ax = pp.subplots()
    with pytest.raises(ValueError, match=r"hatch='g' is not constant"):
        pp.pieplot(data=raw, x="kind", y="n", hatch="g", legend=False, ax=ax)


def test_min_share_does_not_make_an_outside_hue_column_look_ambiguous():
    """The user's column IS constant within every real category; it was
    min_share that merged them, so the bucket must not be blamed on it."""
    raw = pd.DataFrame({
        "kind": list("abcd"),
        "n": [52.0, 42.0, 6.0, 4.0],
        "g": list("wxyz"),
    })
    fig, ax = pp.subplots()
    pp.pieplot(data=raw, x="kind", y="n", hue="g", min_share=0.1,
               legend=False, ax=ax)
    records = ax._publiplots_wedge_meta.wedges
    assert [r.path[0] for r in records] == ["a", "b", "Other"]
    # The bucket takes its own neutral entry, not a real category's colour.
    assert records[-1].hue_color == pytest.approx(
        mcolors.to_rgb("#b4b4b4"))


def test_min_share_does_not_make_an_outside_hatch_column_look_ambiguous():
    raw = pd.DataFrame({
        "kind": list("abcd"),
        "n": [52.0, 42.0, 6.0, 4.0],
        "g": list("wxyz"),
    })
    fig, ax = pp.subplots()
    pp.pieplot(data=raw, x="kind", y="n", hatch="g", min_share=0.1,
               legend=False, ax=ax)
    assert len(ax._publiplots_wedge_meta.wedges) == 3


def test_min_share_leaves_a_plain_object_column_alone(df):
    """The Categorical branch must not change the non-categorical path."""
    fig, ax = pp.subplots()
    pp.pieplot(data=df, x="kind", y="n", min_share=0.1, legend=False, ax=ax)
    assert [r.path[0] for r in ax._publiplots_wedge_meta.wedges] == [
        "symptom", "admin", "Other"]


def test_meta_holds_the_callers_own_frame(df):
    """Annotate builders look rows up in it, so it must not be the
    internally bucketed copy."""
    fig, ax = pp.subplots()
    pp.pieplot(data=df, x="kind", y="n", min_share=0.1, legend=False, ax=ax)
    assert ax._publiplots_wedge_meta.source_frame is df


def test_min_share_bucket_takes_a_palette_override(df):
    import matplotlib.colors as mcolors
    fig, ax = pp.subplots()
    pp.pieplot(data=df, x="kind", y="n", min_share=0.1,
               palette={"Other": "#123456"}, legend=False, ax=ax)
    records = ax._publiplots_wedge_meta.wedges
    assert records[-1].hue_color == pytest.approx(mcolors.to_rgb("#123456"))


def test_a_partial_palette_dict_leaves_the_rest_on_the_default_palette(df):
    """resolve_palette_map returns a dict palette verbatim, so naming one
    category must not strip the colours off every other wedge."""
    import matplotlib.colors as mcolors
    grey = mcolors.to_rgb("#cccccc")
    fig, ax = pp.subplots()
    pp.pieplot(data=df, x="kind", y="n", palette={"admin": "#123456"},
               legend=False, ax=ax)
    by_path = {r.path[0]: r.hue_color
               for r in ax._publiplots_wedge_meta.wedges}
    assert by_path["admin"] == pytest.approx(mcolors.to_rgb("#123456"))
    for other in ("symptom", "lay_dx", "none"):
        assert by_path[other] != pytest.approx(grey), (
            f"{other} fell through to the missing-entry default"
        )
    assert len({tuple(c) for c in by_path.values()}) == 4


def test_multi_column_x_is_refused(df):
    """One ring only -- and the message must not name a function that
    does not exist, which is where a reader would go looking next."""
    fig, ax = pp.subplots()
    with pytest.raises(ValueError, match="single column"):
        pp.pieplot(data=df, x=["kind", "kind"], y="n", ax=ax)


def test_negative_values_raise():
    bad = pd.DataFrame({"kind": list("AB"), "n": [1.0, -1.0]})
    fig, ax = pp.subplots()
    with pytest.raises(ValueError, match="negative"):
        pp.pieplot(data=bad, x="kind", y="n", ax=ax)


def test_legend_stashes_the_categories(df):
    from publiplots.utils.legend_entries import get_entries
    fig, ax = pp.subplots()
    pp.pieplot(data=df, x="kind", y="n", legend=False, ax=ax)
    entries = get_entries(ax)
    assert len(entries) == 1
    assert list(entries[0].labels) == ["symptom", "admin", "lay_dx", "none"]


def test_hue_may_colour_by_another_column():
    import matplotlib.colors as mcolors
    raw = pd.DataFrame({
        "dept": ["a", "b", "c"],
        "region": ["north", "north", "south"],
        "n": [1.0, 1.0, 2.0],
    })
    fig, ax = pp.subplots()
    pp.pieplot(data=raw, x="dept", y="n", hue="region",
               palette={"north": "#75b375", "south": "#8e8ec1"},
               legend=False, ax=ax)
    colors = {r.path[0]: r.hue_color for r in ax._publiplots_wedge_meta.wedges}
    assert colors["a"] == pytest.approx(mcolors.to_rgb("#75b375"))
    assert colors["c"] == pytest.approx(mcolors.to_rgb("#8e8ec1"))


def test_hue_that_varies_within_a_wedge_raises():
    raw = pd.DataFrame({
        "dept": ["a", "a"], "region": ["north", "south"], "n": [1.0, 1.0],
    })
    fig, ax = pp.subplots()
    with pytest.raises(ValueError, match="not constant within the wedge"):
        pp.pieplot(data=raw, x="dept", y="n", hue="region", ax=ax)


def test_hatch_is_applied_per_category(df):
    fig, ax = pp.subplots()
    pp.pieplot(data=df, x="kind", y="n", hatch="kind", legend=False, ax=ax)
    patterns = [r.patch.get_hatch()
                for r in ax._publiplots_wedge_meta.wedges]
    assert patterns[0] == "", (
        "the house hatch cycle leaves the first level plain")
    assert all(p for p in patterns[1:]), "every later wedge carries a pattern"
    assert len(set(patterns)) == len(patterns), "patterns are distinct"


def test_hatch_keys_off_its_own_column_not_the_hue():
    """hatch= and hue= are independent; hatching must follow hatch=."""
    raw = pd.DataFrame({
        "dept": list("abcd"),
        "team": ["x", "x", "y", "y"],
        "n": [1.0, 1.0, 1.0, 1.0],
    })
    fig, ax = pp.subplots()
    pp.pieplot(data=raw, x="dept", y="n", hatch="team", legend=False, ax=ax)
    patterns = [r.patch.get_hatch()
                for r in ax._publiplots_wedge_meta.wedges]
    assert patterns[0] == patterns[1], "one team, one pattern"
    assert patterns[2] == patterns[3], "one team, one pattern"
    assert patterns[0] != patterns[2], "different teams, different patterns"


def test_fill_is_tinted_and_the_outline_is_opaque(df):
    """The house style: alpha tints the fill, the edge stays full strength
    and defaults to the face colour."""
    fig, ax = pp.subplots()
    pp.pieplot(data=df, x="kind", y="n", legend=False, ax=ax)
    wedge = ax._publiplots_wedge_meta.wedges[0].patch
    assert wedge.get_facecolor()[3] == pytest.approx(pp.rcParams["alpha"])
    assert wedge.get_edgecolor()[3] == pytest.approx(1.0)
    assert wedge.get_edgecolor()[:3] == pytest.approx(
        wedge.get_facecolor()[:3])


def test_edgecolor_overrides_the_face_derived_outline(df):
    fig, ax = pp.subplots()
    pp.pieplot(data=df, x="kind", y="n", edgecolor="#123456", legend=False,
               ax=ax)
    import matplotlib.colors as mcolors
    for record in ax._publiplots_wedge_meta.wedges:
        assert record.patch.get_edgecolor()[:3] == pytest.approx(
            mcolors.to_rgb("#123456"))


def test_no_hatch_by_default(df):
    fig, ax = pp.subplots()
    pp.pieplot(data=df, x="kind", y="n", legend=False, ax=ax)
    assert all(r.patch.get_hatch() is None
               for r in ax._publiplots_wedge_meta.wedges)


def test_numeric_hue_stashes_a_colorbar_not_categorical_entries():
    from publiplots.utils.legend_entries import get_entries, is_continuous_hue
    raw = pd.DataFrame({
        "dept": list("abcd"),
        "score": [1.0, 2.0, 3.0, 4.0],
        "n": [1.0, 1.0, 1.0, 1.0],
    })
    fig, ax = pp.subplots()
    pp.pieplot(data=raw, x="dept", y="n", hue="score", legend=False, ax=ax)
    entries = get_entries(ax)
    assert len(entries) == 1
    assert is_continuous_hue(entries[0].handles)
    assert ax._publiplots_wedge_meta.hue_continuous is True


def test_numeric_hue_orders_colours_by_value():
    raw = pd.DataFrame({
        "dept": list("abcd"),
        "score": [4.0, 1.0, 3.0, 2.0],
        "n": [1.0, 1.0, 1.0, 1.0],
    })
    fig, ax = pp.subplots()
    pp.pieplot(data=raw, x="dept", y="n", hue="score", legend=False, ax=ax)
    colors = {r.path[0]: r.hue_color
              for r in ax._publiplots_wedge_meta.wedges}
    # The lowest score and the highest must not land on the same colour.
    assert colors["b"] != colors["a"]


def test_figsize_is_rejected(df):
    with pytest.raises(TypeError, match="axes_size"):
        pp.pieplot(data=df, x="kind", y="n", figsize=(4, 3))


# --- review round 1 ------------------------------------------------------


def test_order_with_min_share_is_refused(df):
    """A bucketed category cannot be named in an explicit order, so the
    combination used to drop it and renormalise the pie to a false 100%."""
    fig, ax = pp.subplots()
    with pytest.raises(ValueError, match="min_share"):
        pp.pieplot(data=df, x="kind", y="n", min_share=0.1,
                   order=["symptom", "admin", "lay_dx", "none"], ax=ax)
    with pytest.raises(ValueError, match="order"):
        pp.pieplot(data=df, x="kind", y="n", min_share=0.1,
                   order=["symptom", "admin", "lay_dx", "none"], ax=ax)


def test_order_and_min_share_are_each_fine_alone(df):
    """The refusal above must not have made either parameter unusable."""
    fig, axes = pp.subplots(ncols=2)
    with pytest.warns(UserWarning, match="renormalise"):
        pp.pieplot(data=df, x="kind", y="n", order=["none", "admin"],
                   legend=False, ax=axes[0])
    pp.pieplot(data=df, x="kind", y="n", min_share=0.1, legend=False,
               ax=axes[1])
    assert [r.path[0] for r in axes[0]._publiplots_wedge_meta.wedges] == [
        "none", "admin"]
    assert [r.path[0] for r in axes[1]._publiplots_wedge_meta.wedges] == [
        "symptom", "admin", "Other"]


def test_hue_order_reorders_the_legend_without_dropping_wedges(df):
    """hue_order narrows the legend's running order, never the palette:
    an omitted level must keep a real colour, not the grey fallback."""
    from publiplots.utils.legend_entries import get_entries
    fig, ax = pp.subplots()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        pp.pieplot(data=df, x="kind", y="n", hue_order=["none", "admin"],
                   legend=False, ax=ax)

    records = ax._publiplots_wedge_meta.wedges
    assert [r.path[0] for r in records] == [
        "symptom", "admin", "lay_dx", "none"], "wedges are unaffected"

    grey = mcolors.to_rgb("#cccccc")
    for record in records:
        assert record.hue_color != pytest.approx(grey), (
            f"{record.path[0]} fell through to the missing-entry default"
        )
    assert len({tuple(r.hue_color) for r in records}) == 4

    # Listed levels lead; the rest keep their place behind them.
    assert list(get_entries(ax)[0].labels) == [
        "none", "admin", "symptom", "lay_dx"]


def test_an_independent_hatch_column_gets_its_own_legend_entry():
    """Folding hatch patterns into the hue swatches looks them up in the
    wrong key space, so the drawn patterns end up with no key at all."""
    from publiplots.utils.legend_entries import get_entries
    raw = pd.DataFrame({
        "dept": list("abcd"),
        "team": ["x", "x", "y", "y"],
        "n": [1.0, 1.0, 1.0, 1.0],
    })
    fig, ax = pp.subplots()
    pp.pieplot(data=raw, x="dept", y="n", hatch="team", legend=False, ax=ax)

    entries = {e.kind: e for e in get_entries(ax)}
    assert set(entries) == {"hue", "hatch"}
    assert entries["hatch"].name == "team"
    assert list(entries["hatch"].labels) == ["x", "y"]

    drawn = {r.patch.get_hatch()
             for r in ax._publiplots_wedge_meta.wedges}
    keyed = {h.get_hatch() for h in entries["hatch"].handles}
    assert keyed == drawn, "every drawn pattern is in the key"

    # The colour swatches stand for the hue alone.
    assert all(not h.get_hatch() for h in entries["hue"].handles)


def test_a_colliding_hatch_value_does_not_leak_into_a_hue_swatch():
    """The two columns have separate value spaces. Looking a hatch map up
    with hue keys can still *hit* when the spaces overlap, hatching a
    colour swatch for a pattern its wedge does not carry."""
    from publiplots.utils.legend_entries import get_entries
    raw = pd.DataFrame({
        "dept": ["x", "y", "z"],
        "team": ["z", "z", "x"],
        "n": [1.0, 1.0, 1.0],
    })
    fig, ax = pp.subplots()
    pp.pieplot(data=raw, x="dept", y="n", hatch="team", legend=False, ax=ax)
    entries = {e.kind: e for e in get_entries(ax)}
    assert all(not h.get_hatch() for h in entries["hue"].handles), (
        "a colour swatch must never borrow a pattern from the hatch map"
    )
    assert [h.get_hatch() for h in entries["hatch"].handles] == ["", "//"]


def test_a_continuous_hue_on_the_hatch_column_still_gets_a_hatch_key():
    """A numeric hue renders as a colorbar, which folds in no patterns.
    Sharing the column with hatch= must therefore still stash a hatch
    entry, or the drawn patterns have no key at all."""
    from publiplots.utils.legend_entries import get_entries, is_continuous_hue
    raw = pd.DataFrame({
        "dept": list("abc"),
        "score": [1.0, 2.0, 3.0],
        "n": [1.0, 1.0, 1.0],
    })
    fig, ax = pp.subplots()
    pp.pieplot(data=raw, x="dept", y="n", hue="score", hatch="score",
               legend=False, ax=ax)

    entries = {e.kind: e for e in get_entries(ax)}
    assert set(entries) == {"hue", "hatch"}
    assert is_continuous_hue(entries["hue"].handles)

    drawn = {r.patch.get_hatch()
             for r in ax._publiplots_wedge_meta.wedges}
    assert len(drawn) == 3, "three distinct patterns are drawn"
    keyed = {h.get_hatch() for h in entries["hatch"].handles}
    assert keyed == drawn, "every drawn pattern is in the key"


def test_a_shared_hatch_column_keeps_one_combined_entry(df):
    from publiplots.utils.legend_entries import get_entries
    fig, ax = pp.subplots()
    pp.pieplot(data=df, x="kind", y="n", hatch="kind", legend=False, ax=ax)
    entries = get_entries(ax)
    assert [e.kind for e in entries] == ["hue"]
    assert [h.get_hatch() for h in entries[0].handles] == [
        r.patch.get_hatch() for r in ax._publiplots_wedge_meta.wedges]


def test_hatch_order_leads_without_dropping_levels(df):
    fig, ax = pp.subplots()
    pp.pieplot(data=df, x="kind", y="n", hatch="kind",
               hatch_order=["none", "lay_dx"], legend=False, ax=ax)
    patterns = {r.path[0]: r.patch.get_hatch()
                for r in ax._publiplots_wedge_meta.wedges}
    # The house cycle starts plain, so the first listed level is the plain
    # one and the unlisted levels take the later patterns.
    assert patterns["none"] == ""
    assert patterns["lay_dx"] == "//"
    assert len(set(patterns.values())) == 4, "no level lost its pattern"


def test_min_share_is_strictly_below_the_threshold():
    """The docstring says "below this fraction": a category sitting
    exactly on the threshold is kept."""
    raw = pd.DataFrame({"kind": list("abcd"), "n": [82.0, 10.0, 5.0, 3.0]})
    fig, ax = pp.subplots()
    pp.pieplot(data=raw, x="kind", y="n", min_share=0.1, legend=False, ax=ax)
    records = ax._publiplots_wedge_meta.wedges
    assert [r.path[0] for r in records] == ["a", "b", "Other"], (
        "b is exactly 10% of 100 and must survive min_share=0.1"
    )
    assert records[-1].value == pytest.approx(8.0)   # 5.0 + 3.0


def test_the_min_share_value_selects_which_categories_bucket():
    raw = pd.DataFrame({"kind": list("abcd"), "n": [82.0, 10.0, 5.0, 3.0]})
    fig, ax = pp.subplots()
    pp.pieplot(data=raw, x="kind", y="n", min_share=0.2, legend=False, ax=ax)
    records = ax._publiplots_wedge_meta.wedges
    assert [r.path[0] for r in records] == ["a", "Other"]
    assert records[-1].value == pytest.approx(18.0)  # 10.0 + 5.0 + 3.0


def test_legend_swatch_matches_the_mark_it_stands_for(df):
    """The house rule that a legend swatch carries the same width the
    mark is drawn with: the key must match the wedge's stroke and fill."""
    from publiplots.utils.legend_entries import get_entries
    fig, ax = pp.subplots()
    pp.pieplot(data=df, x="kind", y="n", legend=False, ax=ax)
    entry = get_entries(ax)[0]
    records = ax._publiplots_wedge_meta.wedges
    assert list(entry.labels) == [r.path[0] for r in records]
    for handle, record in zip(entry.handles, records):
        assert handle.get_linewidth() == pytest.approx(
            record.patch.get_linewidth())
        assert handle.get_linewidth() == pytest.approx(
            pp.rcParams["edgewidth"])
        assert tuple(handle.get_facecolor()) == pytest.approx(
            tuple(record.patch.get_facecolor()))


def test_a_deferred_pp_legend_renders_the_categories(df):
    """#268: pp.legend(ax) on a pie ran without error and drew nothing.
    The entry is stashed even under legend=False so the later call finds
    it, and the group materialises the artist at draw time."""
    from matplotlib.legend import Legend
    fig, ax = pp.subplots()
    pp.pieplot(data=df, x="kind", y="n", legend=False, ax=ax)
    assert not [a for a in fig.get_children() if isinstance(a, Legend)]

    pp.legend(ax)
    fig.canvas.draw()

    legends = [a for a in list(fig.get_children()) + list(ax.get_children())
               if isinstance(a, Legend)]
    assert len(legends) == 1, "exactly one legend artist"
    assert [t.get_text() for t in legends[0].get_texts()] == [
        "symptom", "admin", "lay_dx", "none"]


def test_the_default_legend_true_path_renders_a_legend(df):
    """Every other legend test passes legend=False, so the default path
    -- resolve_legend_flags(legend) with legend=True -- was never run end
    to end: replacing it with resolve_legend_flags(False) survived."""
    from matplotlib.legend import Legend
    fig, ax = pp.subplots()
    pp.pieplot(data=df, x="kind", y="n", ax=ax)          # no legend=
    fig.canvas.draw()
    legends = [a for a in list(fig.get_children()) + list(ax.get_children())
               if isinstance(a, Legend)]
    assert len(legends) == 1, "exactly one legend artist"
    assert [t.get_text() for t in legends[0].get_texts()] == [
        "symptom", "admin", "lay_dx", "none"]


def test_start_angle_and_direction_reach_the_layout():
    """Both were hardcoded at the layout() call; unwiring either from the
    public signature changed nothing that any test looked at."""
    raw = pd.DataFrame({"kind": list("ab"), "n": [3.0, 1.0]})

    fig, ax = pp.subplots()
    pp.pieplot(data=raw, x="kind", y="n", legend=False, ax=ax)
    first = ax._publiplots_wedge_meta.wedges[0]
    # Default: 90 degrees, clockwise -- the first wedge runs from twelve
    # o'clock backwards through 3/4 of the circle to 90 - 270 = -180.
    assert (first.theta1, first.theta0) == pytest.approx((90.0, -180.0))

    fig2, ax2 = pp.subplots()
    pp.pieplot(data=raw, x="kind", y="n", start_angle=0.0,
               direction="counterclockwise", legend=False, ax=ax2)
    first2 = ax2._publiplots_wedge_meta.wedges[0]
    assert (first2.theta0, first2.theta1) == pytest.approx((0.0, 270.0))
    second2 = ax2._publiplots_wedge_meta.wedges[1]
    assert (second2.theta0, second2.theta1) == pytest.approx((270.0, 360.0))


def test_the_framing_makes_room_for_the_pulled_extent():
    """radius = 1.0 + max_pull + 0.05: dropping max_pull survived,
    because nothing asserted the pulled rim was inside the limits."""
    raw = pd.DataFrame({"kind": list("ab"), "n": [3.0, 1.0]})
    fig, ax = pp.subplots(axes_size=(60, 60))
    pp.pieplot(data=raw, x="kind", y="n", pull=0.3, legend=False, ax=ax)

    records = ax._publiplots_wedge_meta.wedges
    radial = max((r.patch.center[0] ** 2 + r.patch.center[1] ** 2) ** 0.5
                 + r.r_outer for r in records)
    assert radial == pytest.approx(1.30, abs=1e-9), "pull 0.3 past rim 1.0"

    # The square cell frames both axes at the same radius, so the
    # reconciliation is visible directly in the limits.
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    assert (x1, y1) == pytest.approx((1.35, 1.35))
    per_axis = max(max(abs(r.patch.center[0]), abs(r.patch.center[1]))
                   + r.r_outer for r in records)
    assert per_axis < min(x1, y1)
    assert -per_axis > max(x0, y0)


def test_the_entry_is_named_for_the_column_pp_legend_collects_on(df):
    """pp.legend(collect=[...]) selects on the entry name."""
    from publiplots.utils.legend_entries import get_entries
    fig, ax = pp.subplots()
    pp.pieplot(data=df, x="kind", y="n", legend=False, ax=ax)
    assert get_entries(ax)[0].name == "kind", "defaults to the x column"

    raw = pd.DataFrame({"dept": list("abc"), "region": ["n", "n", "s"],
                        "n": [1.0, 1.0, 2.0]})
    fig2, ax2 = pp.subplots()
    pp.pieplot(data=raw, x="dept", y="n", hue="region", legend=False, ax=ax2)
    assert get_entries(ax2)[0].name == "region", "an explicit hue wins"


def test_title_is_set(df):
    fig, ax = pp.subplots()
    pp.pieplot(data=df, x="kind", y="n", title="Referral source",
               legend=False, ax=ax)
    assert ax.get_title() == "Referral source"


def test_an_empty_title_clears_a_pre_existing_one(df):
    """The house form is `title is not None` (bar.py:639, box.py): ""
    means "no title", and only None means "leave whatever is there"."""
    fig, ax = pp.subplots()
    ax.set_title("previously set")
    pp.pieplot(data=df, x="kind", y="n", title="", legend=False, ax=ax)
    assert ax.get_title() == ""


def test_title_none_leaves_a_pre_existing_one_alone(df):
    fig, ax = pp.subplots()
    ax.set_title("previously set")
    pp.pieplot(data=df, x="kind", y="n", title=None, legend=False, ax=ax)
    assert ax.get_title() == "previously set"


def test_an_all_nan_numeric_hue_names_the_column_instead_of_min():
    """min()/max() over an empty level list raised a bare "min() iterable
    argument is empty" that named neither the column nor the cause."""
    import numpy as np
    raw = pd.DataFrame({
        "kind": list("abc"),
        "n": [3.0, 2.0, 1.0],
        "score": [np.nan, np.nan, np.nan],
    })
    fig, ax = pp.subplots()
    with pytest.warns(UserWarning) as rec:
        pp.pieplot(data=raw, x="kind", y="n", hue="score", legend=False,
                   ax=ax)
    messages = [str(r.message) for r in rec]
    assert any("score" in m and "no non-NaN value" in m for m in messages)
    # Degrades the way the categorical path does: grey wedges, no crash.
    assert len(ax._publiplots_wedge_meta.wedges) == 3


def test_an_unknown_keyword_is_refused(df):
    """Nothing is forwarded on, so a leftover keyword is a typo."""
    fig, ax = pp.subplots()
    with pytest.raises(TypeError, match="donu"):
        pp.pieplot(data=df, x="kind", y="n", donu=0.5, ax=ax)


# --- hatch strokes follow linewidth=, on the wedge and on its swatch ------

_HAS_HATCH_LINEWIDTH = hasattr(Wedge, "set_hatch_linewidth")
_needs_hatch_linewidth = pytest.mark.skipif(
    not _HAS_HATCH_LINEWIDTH,
    reason="Patch.set_hatch_linewidth needs matplotlib>=3.10")


def _hatched_wedges(ax):
    return [r.patch for r in ax._publiplots_wedge_meta.wedges
            if r.patch.get_hatch()]


def _hatched_legend_swatches(fig):
    """The stroked swatch rectangles a rendered legend actually draws."""
    from matplotlib.legend import Legend
    from matplotlib.patches import Rectangle
    fig.canvas.draw()
    return [r for leg in fig.findobj(Legend) for r in leg.findobj(Rectangle)
            if r.get_hatch()]


@_needs_hatch_linewidth
def test_hatch_strokes_default_to_the_themed_edgewidth(df):
    fig, ax = pp.subplots(axes_size=(50, 40))
    pp.pieplot(data=df, x="kind", y="n", hatch="kind", legend=False, ax=ax)
    hatched = _hatched_wedges(ax)
    assert hatched, "the hatch cycle should pattern at least one wedge"
    expected = pp.rcParams["edgewidth"]
    assert expected == pytest.approx(0.75)
    for patch in hatched:
        assert patch.get_hatch_linewidth() == pytest.approx(expected)
        assert patch.get_linewidth() == pytest.approx(expected)


@_needs_hatch_linewidth
def test_hatch_strokes_follow_an_explicit_linewidth(df):
    fig, ax = pp.subplots(axes_size=(50, 40))
    pp.pieplot(data=df, x="kind", y="n", hatch="kind", linewidth=2.0,
               legend=False, ax=ax)
    hatched = _hatched_wedges(ax)
    assert hatched
    for patch in hatched:
        assert patch.get_hatch_linewidth() == pytest.approx(2.0)
        assert patch.get_linewidth() == pytest.approx(2.0)


@_needs_hatch_linewidth
@pytest.mark.parametrize("linewidth", [None, 2.0])
def test_a_hatched_hue_swatch_carries_the_wedges_hatch_width(df, linewidth):
    """hatch=x folds the patterns into the hue swatches; the swatch's hatch
    must be drawn at the width the wedge's hatch is."""
    fig, ax = pp.subplots(axes_size=(50, 40))
    pp.pieplot(data=df, x="kind", y="n", hatch="kind", linewidth=linewidth,
               ax=ax)
    expected = pp.rcParams["edgewidth"] if linewidth is None else linewidth
    hatched = _hatched_wedges(ax)
    assert hatched
    assert all(p.get_hatch_linewidth() == pytest.approx(expected)
               for p in hatched)
    swatches = _hatched_legend_swatches(fig)
    assert swatches, "the legend should draw hatched swatches"
    # Without the fix the swatch sits at rcParams["hatch.linewidth"], 1.0,
    # which neither 0.75 nor 2.0 is.
    assert all(s.get_hatch_linewidth() == pytest.approx(expected)
               for s in swatches)


@_needs_hatch_linewidth
def test_an_independent_hatch_swatch_carries_the_wedges_hatch_width():
    """The second, hatch-only key is built separately from the hue key and
    needs the same width."""
    raw = pd.DataFrame({
        "dept": list("abcd"),
        "team": ["x", "y", "x", "y"],
        "n": [1.0, 2.0, 3.0, 4.0],
    })
    fig, ax = pp.subplots(axes_size=(50, 40))
    pp.pieplot(data=raw, x="dept", y="n", hatch="team", linewidth=1.6, ax=ax)
    hatched = _hatched_wedges(ax)
    assert hatched
    assert all(p.get_hatch_linewidth() == pytest.approx(1.6)
               for p in hatched)
    swatches = _hatched_legend_swatches(fig)
    assert swatches
    assert all(s.get_hatch_linewidth() == pytest.approx(1.6)
               for s in swatches)


# --- a numeric hue honours palette= ---------------------------------------

@pytest.fixture
def scored():
    return pd.DataFrame({
        "dept": list("abcd"),
        "score": [4.0, 1.0, 3.0, 2.0],
        "n": [1.0, 2.0, 3.0, 4.0],
    })


def _colorbar_mappable(ax):
    from publiplots.utils.legend_entries import get_entries, is_continuous_hue
    entries = get_entries(ax)
    assert len(entries) == 1
    assert is_continuous_hue(entries[0].handles)
    return entries[0].handles[0]


def _assert_wedges_follow(ax, raw, cmap):
    """Every wedge's fill RGB is cmap(norm(score)), and the stashed colorbar
    maps each score to that same RGB. Fills are drawn at the themed alpha
    (``_split_face_and_edge_alpha``), so only the RGB channels compare."""
    from matplotlib.colors import Normalize
    norm = Normalize(vmin=raw["score"].min(), vmax=raw["score"].max())
    score = dict(zip(raw["dept"], raw["score"]))
    mappable = _colorbar_mappable(ax)
    for record in ax._publiplots_wedge_meta.wedges:
        value = score[record.path[0]]
        expected = cmap(norm(value))[:3]
        assert tuple(record.patch.get_facecolor()[:3]) == pytest.approx(
            expected)
        assert tuple(record.hue_color) == pytest.approx(expected)
        assert tuple(mappable.to_rgba(value)[:3]) == pytest.approx(expected)
    return mappable


def test_a_numeric_hue_maps_through_a_named_palette(scored):
    from matplotlib import colormaps
    fig, ax = pp.subplots(axes_size=(50, 40))
    pp.pieplot(data=scored, x="dept", y="n", hue="score", palette="viridis",
               legend=False, ax=ax)
    mappable = _assert_wedges_follow(ax, scored, colormaps["viridis"])
    assert mappable.get_cmap().name == "viridis"


def test_a_numeric_hue_maps_through_a_colormap_instance(scored):
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("red_to_blue", ["red", "blue"])
    fig, ax = pp.subplots(axes_size=(50, 40))
    pp.pieplot(data=scored, x="dept", y="n", hue="score", palette=cmap,
               legend=False, ax=ax)
    mappable = _assert_wedges_follow(ax, scored, cmap)
    assert mappable.get_cmap().name == "red_to_blue"


def test_a_numeric_hue_without_a_palette_keeps_the_light_ramp(scored):
    """palette=None is today's behaviour: a light ramp on rcParams color."""
    from publiplots.themes.colors import resolve_continuous_cmap
    ramp = resolve_continuous_cmap(None, color=pp.rcParams["color"])
    fig, ax = pp.subplots(axes_size=(50, 40))
    pp.pieplot(data=scored, x="dept", y="n", hue="score", legend=False, ax=ax)
    mappable = _assert_wedges_follow(ax, scored, ramp)
    assert mappable.get_cmap().name == ramp.name


def test_a_numeric_hue_without_a_palette_anchors_on_color(scored):
    from publiplots.themes.colors import resolve_continuous_cmap
    ramp = resolve_continuous_cmap(None, color="#c0392b")
    fig, ax = pp.subplots(axes_size=(50, 40))
    pp.pieplot(data=scored, x="dept", y="n", hue="score", color="#c0392b",
               legend=False, ax=ax)
    _assert_wedges_follow(ax, scored, ramp)


@pytest.mark.parametrize("palette", [
    {1.0: "red", 2.0: "blue", 3.0: "green", 4.0: "black"},
    ["red", "blue"],
])
def test_a_numeric_hue_refuses_a_discrete_palette(scored, palette):
    """scatterplot raises TypeError on both; this does too, but names the
    parameter."""
    fig, ax = pp.subplots(axes_size=(50, 40))
    with pytest.raises(TypeError, match=r"hue='score' is numeric"):
        pp.pieplot(data=scored, x="dept", y="n", hue="score",
                   palette=palette, legend=False, ax=ax)


def test_a_numeric_hue_refuses_an_unknown_colormap_name(scored):
    fig, ax = pp.subplots(axes_size=(50, 40))
    with pytest.raises(ValueError, match=r"'not_a_cmap' is not one"):
        pp.pieplot(data=scored, x="dept", y="n", hue="score",
                   palette="not_a_cmap", legend=False, ax=ax)
