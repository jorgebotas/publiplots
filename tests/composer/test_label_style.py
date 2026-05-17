"""Tests for canvas-wide label_style + per-panel override."""

import pytest

import publiplots as pp


def test_canvas_label_style_setter_updates_canvas_state():
    canvas = pp.Canvas("cell-2col")
    canvas.label_style(loc="ur", size=11)
    assert canvas._label_style["loc"] == "ur"
    assert canvas._label_style["size"] == 11


def test_canvas_label_style_partial_update_preserves_other_keys():
    """Calling label_style(loc='ur') doesn't reset weight/size."""
    canvas = pp.Canvas("cell-2col")
    initial_weight = canvas._label_style["weight"]
    canvas.label_style(loc="ur")
    assert canvas._label_style["weight"] == initial_weight
    assert canvas._label_style["loc"] == "ur"


def test_canvas_label_style_default_for_cell():
    """Cell preset: loc='ul', size=9 (preset default), weight='bold'."""
    canvas = pp.Canvas("cell-2col")
    assert canvas._label_style["loc"] == "ul"
    assert canvas._label_style["size"] == 9
    assert canvas._label_style["weight"] == "bold"


def test_canvas_label_style_default_for_nature():
    canvas = pp.Canvas("nature-2col")
    assert canvas._label_style["size"] == 8


def test_canvas_label_style_default_for_science():
    canvas = pp.Canvas("science-2col")
    assert canvas._label_style["size"] == 10


def test_canvas_label_style_loc_ultraplot_vocab():
    """The 8 ultraplot loc values: ul, ur, ll, lr, uc, lc, cl, cr."""
    valid_locs = {"ul", "ur", "ll", "lr", "uc", "lc", "cl", "cr"}
    for loc in valid_locs:
        canvas = pp.Canvas("cell-2col")
        canvas.label_style(loc=loc)
        assert canvas._label_style["loc"] == loc


def test_canvas_label_style_invalid_loc_raises():
    canvas = pp.Canvas("cell-2col")
    with pytest.raises(ValueError, match="loc"):
        canvas.label_style(loc="middle")


def test_per_panel_label_style_overrides_canvas():
    """Per-panel label_style merges INTO the canvas-wide style:
    panel keys win, others fall through."""
    canvas = pp.Canvas("cell-2col")  # canvas: loc='ul', size=9
    canvas.add_row(
        pp.PanelAxes(label="A", size=(60.0, 40.0)),
        pp.PanelAxes(label="B", size=(60.0, 40.0),
                     label_style={"loc": "ur", "size": 11}),
    )
    # Resolved per-panel styles should be available on the Panel object.
    a_style = canvas["A"].resolved_label_style
    b_style = canvas["B"].resolved_label_style
    assert a_style["loc"] == "ul"   # canvas default
    assert a_style["size"] == 9
    assert b_style["loc"] == "ur"   # per-panel override
    assert b_style["size"] == 11    # per-panel override
    assert b_style["weight"] == "bold"  # canvas default falls through


def test_per_panel_label_style_partial_override():
    """Per-panel can override just one key; others fall through."""
    canvas = pp.Canvas("cell-2col")  # canvas: weight='bold', size=9
    canvas.add_row(
        pp.PanelAxes(label="A", size=(60.0, 40.0),
                     label_style={"size": 14}),  # only size override
    )
    style = canvas["A"].resolved_label_style
    assert style["size"] == 14
    assert style["weight"] == "bold"  # canvas default falls through


def test_label_renders_in_figure_after_add_row():
    """Smoke test: after add_row, the panel's axes has a Text artist
    placed at the resolved loc."""
    canvas = pp.Canvas("cell-2col")
    canvas.add_row(pp.PanelAxes(label="A", size=(60.0, 40.0)))

    # The label is added as an ax.text artist; find it by its text content.
    ax = canvas["A"].ax
    label_artists = [t for t in ax.texts if t.get_text() == "A"]
    assert len(label_artists) == 1, (
        f"expected exactly 1 'A' text artist, got {len(label_artists)}"
    )


def test_label_false_renders_no_text_artist():
    canvas = pp.Canvas("cell-2col")
    canvas.add_row(pp.PanelAxes(label=False, size=(60.0, 40.0)))
    ax = list(canvas._panels.values())[0].ax if canvas._panels else canvas._panels_ordered[0].ax
    # No label-text artist with non-empty content.
    label_texts = [t.get_text() for t in ax.texts]
    assert all(t == "" or t is None for t in label_texts)


def test_abc_false_renders_no_text_for_none_panels():
    canvas = pp.Canvas("cell-2col", abc=False)
    canvas.add_row(pp.PanelAxes(label=None, size=(60.0, 40.0)))
    ax = canvas._panels_ordered[0].ax
    label_texts = [t.get_text() for t in ax.texts]
    assert all(t == "" or t is None for t in label_texts)


# ---------------------------------------------------------------------------
# loc-table outward-grow contract — abc text grows AWAY from data into the
# canvas-reserved decoration margin (not INTO the axes-data box).
#
# For corner anchors ('ul'/'ur'/'ll'/'lr') the y_frac sits ON the spine and
# the va flips so the glyph's vertical extent lies OUTSIDE the spine. For
# edge centers ('uc'/'lc') va flips similarly. For 'cl'/'cr' the ha flips so
# the glyph extends sideways outside the spine.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("loc,expected_ha,expected_va", [
    ("ul", "left",   "bottom"),
    ("ur", "right",  "bottom"),
    ("ll", "left",   "top"),
    ("lr", "right",  "top"),
    ("uc", "center", "bottom"),
    ("lc", "center", "top"),
    ("cl", "right",  "center"),
    ("cr", "left",   "center"),
])
def test_abc_label_loc_anchors_grow_outside_axes(loc, expected_ha, expected_va):
    """Each loc anchor's text artist has the va/ha that grows the glyph
    AWAY from the data box (outward-grow contract, PR 6c addendum)."""
    canvas = pp.Canvas("cell-2col")
    canvas.label_style(loc=loc)
    canvas.add_row(pp.PanelAxes(label="A", size=(60.0, 40.0)))

    ax = canvas["A"].ax
    label_artists = [t for t in ax.texts if t.get_text() == "A"]
    assert len(label_artists) == 1, (
        f"expected exactly 1 'A' text artist for loc={loc!r}, got {len(label_artists)}"
    )
    artist = label_artists[0]
    assert artist.get_ha() == expected_ha, (
        f"loc={loc!r}: expected ha={expected_ha!r}, got {artist.get_ha()!r}"
    )
    assert artist.get_va() == expected_va, (
        f"loc={loc!r}: expected va={expected_va!r}, got {artist.get_va()!r}"
    )
