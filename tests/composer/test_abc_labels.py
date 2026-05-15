"""Tests for abc auto-letter sequencing + render-time label resolution."""

import pytest

import publiplots as pp
from publiplots.composer.abc_labels import resolve_labels


# ---------------------------------------------------------------------------
# resolve_labels — pure-Python sequencer
# ---------------------------------------------------------------------------

def test_resolve_labels_upper_simple():
    out = resolve_labels(
        panel_labels=[None, None, None],
        abc="upper",
    )
    assert out == ["A", "B", "C"]


def test_resolve_labels_lower_simple():
    out = resolve_labels(
        panel_labels=[None, None, None],
        abc="lower",
    )
    assert out == ["a", "b", "c"]


def test_resolve_labels_template_dot():
    out = resolve_labels(
        panel_labels=[None, None, None],
        abc="A.",
    )
    assert out == ["A.", "B.", "C."]


def test_resolve_labels_template_lower_dot():
    out = resolve_labels(
        panel_labels=[None, None, None],
        abc="a.",
    )
    assert out == ["a.", "b.", "c."]


def test_resolve_labels_explicit_list():
    out = resolve_labels(
        panel_labels=[None, None, None, None],
        abc=["i", "ii", "iii", "iv"],
    )
    assert out == ["i", "ii", "iii", "iv"]


def test_resolve_labels_false_disables_all():
    """abc=False suppresses every auto-letter; explicit str labels stay."""
    out = resolve_labels(
        panel_labels=[None, "X", None],
        abc=False,
    )
    assert out == [None, "X", None]


def test_resolve_labels_explicit_str_passes_through_and_consumes_slot():
    """A panel with label='X' stays as 'X' AND consumes a slot in the
    sequence: subsequent None panels still increment correctly."""
    out = resolve_labels(
        panel_labels=[None, "Z", None],
        abc="upper",
    )
    # Slot 0 → 'A'; slot 1 → 'Z' (verbatim, but consumes the 'B' slot);
    # slot 2 → 'C' (next after the consumed 'B').
    assert out == ["A", "Z", "C"]


def test_resolve_labels_false_panel_skips_sequence_slot():
    """label=False panels are SKIPPED from the sequence — the next auto
    panel uses the next letter."""
    out = resolve_labels(
        panel_labels=[None, False, None],
        abc="upper",
    )
    # Slot 0 → 'A'; slot 1 → False (no label, skip from sequence);
    # slot 2 → 'B' (NOT 'C' — the False didn't consume a slot).
    assert out == ["A", False, "B"]


def test_resolve_labels_explicit_list_too_short_raises():
    with pytest.raises(ValueError, match="abc list"):
        resolve_labels(
            panel_labels=[None, None, None, None],
            abc=["i", "ii"],
        )


def test_resolve_labels_unknown_mode_raises():
    with pytest.raises(ValueError, match="abc"):
        resolve_labels(panel_labels=[None], abc="upper-roman")


def test_resolve_labels_after_z_uses_aa_continuation():
    """Past 26 panels, abc='upper' continues 'AA','AB','AC',...
    Important: we don't expect anyone to actually have 27 panels, but
    it's a low-risk fallback."""
    panels = [None] * 28
    out = resolve_labels(panel_labels=panels, abc="upper")
    assert out[0] == "A"
    assert out[25] == "Z"
    assert out[26] == "AA"
    assert out[27] == "AB"


# ---------------------------------------------------------------------------
# Canvas integration — abc kwarg + per-panel label resolution
# ---------------------------------------------------------------------------

def test_canvas_abc_default_upper_for_cell():
    canvas = pp.Canvas("cell-2col")
    canvas.add_row(
        pp.PanelAxes(label=None, size=(60.0, 40.0)),
        pp.PanelAxes(label=None, size=(60.0, 40.0)),
    )
    # PR 2: canvas exposes resolved labels on each Panel.
    assert canvas[0].label == "A"
    assert canvas[1].label == "B"


def test_canvas_abc_default_lower_for_nature():
    canvas = pp.Canvas("nature-2col")
    canvas.add_row(
        pp.PanelAxes(label=None, size=(60.0, 40.0)),
        pp.PanelAxes(label=None, size=(60.0, 40.0)),
    )
    assert canvas[0].label == "a"
    assert canvas[1].label == "b"


def test_canvas_abc_explicit_kwarg_overrides_preset_default():
    canvas = pp.Canvas("nature-2col", abc="upper")  # override Nature's lower default
    canvas.add_row(
        pp.PanelAxes(label=None, size=(60.0, 40.0)),
        pp.PanelAxes(label=None, size=(60.0, 40.0)),
    )
    assert canvas[0].label == "A"
    assert canvas[1].label == "B"


def test_canvas_abc_false_disables_labels():
    canvas = pp.Canvas("cell-2col", abc=False)
    canvas.add_row(
        pp.PanelAxes(label=None, size=(60.0, 40.0)),
        pp.PanelAxes(label=None, size=(60.0, 40.0)),
    )
    # When abc=False AND label=None, the resolved label is None → no
    # render. canvas[0].label stays None to signal "no label".
    assert canvas[0].label is None
    assert canvas[1].label is None


def test_canvas_indexing_by_resolved_label():
    """canvas['A'] should resolve to the panel even if the input had
    label=None — the resolved letter is the lookup key."""
    canvas = pp.Canvas("cell-2col")
    canvas.add_row(
        pp.PanelAxes(label=None, size=(60.0, 40.0)),
        pp.PanelAxes(label=None, size=(60.0, 40.0)),
    )
    a = canvas["A"]
    b = canvas["B"]
    assert a.bbox_mm[0] < b.bbox_mm[0]
