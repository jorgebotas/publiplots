"""Tests for pp.pieplot."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

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
    pp.pieplot(data=df, x="kind", y="n", order=["none", "admin"],
               legend=False, ax=ax)
    assert [r.path[0] for r in ax._publiplots_wedge_meta.wedges] == [
        "none", "admin"]


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


def test_multi_column_x_points_at_sunburstplot(df):
    fig, ax = pp.subplots()
    with pytest.raises(ValueError, match="sunburstplot"):
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
    with pytest.raises(TypeError):
        pp.pieplot(data=df, x="kind", y="n", figsize=(4, 3))
