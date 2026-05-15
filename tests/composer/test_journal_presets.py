"""Verified-dimension tests for the 12 journal presets.

Source verification (2026-05-15):
- Cell: cell.com/figureguidelines (verified via web.archive.org snapshot 2023-12-30)
- Nature: nature.com/nature/for-authors/final-submission (live)
- Nature Methods: nature.com/nmeth/submission-guidelines/aip-and-formatting (live)
- Science: science.org/...instructions-preparing-initial-manuscript (verified via
  web.archive.org snapshot 2024-04-21)

Three values are DERIVED rather than direct quotes (flagged in presets.py):
- Cell max-height 225mm (page-fit derivation)
- Science max-height 240mm (trim derivation)
- Nature Methods 1-col / 1.5-col widths (inherited from Nature family)
"""

import pytest

import publiplots as pp
from publiplots.composer.presets import PRESETS, resolve_preset


# ---------------------------------------------------------------------------
# Preset registry — count + naming
# ---------------------------------------------------------------------------

EXPECTED_PRESETS = {
    "custom",
    "cell-1col", "cell-1.5col", "cell-2col",
    "nature-1col", "nature-1.5col", "nature-2col",
    "nature-methods-1col", "nature-methods-1.5col", "nature-methods-2col",
    "science-1col", "science-1.5col", "science-2col",
}


def test_all_journal_presets_registered():
    assert set(PRESETS.keys()) == EXPECTED_PRESETS


# ---------------------------------------------------------------------------
# Cell — 85 / 114 / 174 mm (verified)
# ---------------------------------------------------------------------------

def test_cell_1col_width():
    p = resolve_preset("cell-1col", width=None)
    assert p["width_mm"] == 85.0


def test_cell_1_5col_width():
    p = resolve_preset("cell-1.5col", width=None)
    assert p["width_mm"] == 114.0


def test_cell_2col_width():
    p = resolve_preset("cell-2col", width=None)
    assert p["width_mm"] == 174.0


def test_cell_max_height_is_225_derived():
    """Cell only specifies "fit on a single page". 225mm is the
    page-fit derivation; flagged DERIVED in presets.py."""
    p = resolve_preset("cell-2col", width=None)
    assert p["max_height_mm"] == 225.0


# ---------------------------------------------------------------------------
# Nature — 89 / 120-136 / 183 mm + 247 mm max (verified live)
# ---------------------------------------------------------------------------

def test_nature_1col_width():
    p = resolve_preset("nature-1col", width=None)
    assert p["width_mm"] == 89.0


def test_nature_1_5col_width():
    """Nature publishes 120-136 mm range; we commit the lower bound 120
    to leave room for the user; they can override via width=."""
    p = resolve_preset("nature-1.5col", width=None)
    assert p["width_mm"] == 120.0


def test_nature_2col_width():
    p = resolve_preset("nature-2col", width=None)
    assert p["width_mm"] == 183.0


def test_nature_max_height_is_247():
    p = resolve_preset("nature-2col", width=None)
    assert p["max_height_mm"] == 247.0


# ---------------------------------------------------------------------------
# Nature Methods — inherits Nature columns BUT caps double-col at 180 mm
# ---------------------------------------------------------------------------

def test_nature_methods_1col_inherits_from_nature():
    p = resolve_preset("nature-methods-1col", width=None)
    assert p["width_mm"] == 89.0


def test_nature_methods_1_5col_inherits_from_nature():
    p = resolve_preset("nature-methods-1.5col", width=None)
    assert p["width_mm"] == 120.0


def test_nature_methods_2col_caps_at_180_not_183():
    """Nature Methods AIP page quotes 180mm max; differs from Nature's 183."""
    p = resolve_preset("nature-methods-2col", width=None)
    assert p["width_mm"] == 180.0


def test_nature_methods_max_height_inherits():
    p = resolve_preset("nature-methods-2col", width=None)
    assert p["max_height_mm"] == 247.0


# ---------------------------------------------------------------------------
# Science — 57 / 121 / 184 mm + 240 mm max-height (derived)
# ---------------------------------------------------------------------------

def test_science_1col_width():
    """Verified 57mm (corrects ultraplot's rounded 55mm)."""
    p = resolve_preset("science-1col", width=None)
    assert p["width_mm"] == 57.0


def test_science_1_5col_width():
    p = resolve_preset("science-1.5col", width=None)
    assert p["width_mm"] == 121.0


def test_science_2col_width():
    p = resolve_preset("science-2col", width=None)
    assert p["width_mm"] == 184.0


def test_science_max_height_is_240_derived():
    """Science doesn't quote a max figure height; ~240 mm is the trim
    derivation. Flagged DERIVED in presets.py."""
    p = resolve_preset("science-2col", width=None)
    assert p["max_height_mm"] == 240.0


# ---------------------------------------------------------------------------
# Width override — user can pass width= to override the preset default
# ---------------------------------------------------------------------------

def test_user_can_override_preset_width():
    """User wants Nature-2col but with custom 175mm width — they pass width=."""
    p = resolve_preset("nature-2col", width=175.0)
    assert p["width_mm"] == 175.0


def test_custom_preset_unchanged_from_pr1():
    """The 'custom' escape hatch is unchanged: requires explicit width."""
    with pytest.raises(ValueError, match="width"):
        resolve_preset("custom", width=None)


# ---------------------------------------------------------------------------
# abc default per preset (for Task 5)
# ---------------------------------------------------------------------------

def test_cell_default_abc_is_upper():
    p = resolve_preset("cell-2col", width=None)
    assert p["abc_default"] == "upper"


def test_nature_default_abc_is_lower():
    p = resolve_preset("nature-2col", width=None)
    assert p["abc_default"] == "lower"


def test_science_default_abc_is_upper():
    p = resolve_preset("science-2col", width=None)
    assert p["abc_default"] == "upper"


def test_custom_default_abc_is_upper():
    p = resolve_preset("custom", width=174.0)
    assert p["abc_default"] == "upper"


# ---------------------------------------------------------------------------
# Default label font size per preset (for Task 5)
# ---------------------------------------------------------------------------

def test_cell_default_label_size_is_9pt():
    p = resolve_preset("cell-2col", width=None)
    assert p["label_size_pt"] == 9


def test_nature_default_label_size_is_8pt():
    p = resolve_preset("nature-2col", width=None)
    assert p["label_size_pt"] == 8


def test_science_default_label_size_is_10pt():
    p = resolve_preset("science-2col", width=None)
    assert p["label_size_pt"] == 10


# ---------------------------------------------------------------------------
# Canvas construction with journal presets (top-level smoke)
# ---------------------------------------------------------------------------

def test_canvas_cell_2col_no_width_arg_works():
    """Cell-2col supplies its own default width; user need not pass width=."""
    canvas = pp.Canvas("cell-2col")
    assert canvas.width_mm == 174.0


def test_canvas_nature_2col_no_width_arg_works():
    canvas = pp.Canvas("nature-2col")
    assert canvas.width_mm == 183.0


def test_canvas_science_2col_no_width_arg_works():
    canvas = pp.Canvas("science-2col")
    assert canvas.width_mm == 184.0
