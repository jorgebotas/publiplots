"""The publication defaults contract.

Every value here is a deliberate choice from
docs/superpowers/specs/2026-08-31-rcparams-polish-design.md. If you are
changing one, change it here too and say why in the changelog.
"""
import matplotlib.pyplot as plt
import pytest

import publiplots as pp


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ---- Type: flat 7pt ---------------------------------------------------------

@pytest.mark.parametrize("key", [
    "font.size",
    "axes.labelsize",
    "axes.titlesize",
    "figure.titlesize",
    "xtick.labelsize",
    "ytick.labelsize",
    "legend.fontsize",
])
def test_every_font_size_is_7pt(key):
    """Flat hierarchy: journals that cap body text at 7pt reject anything above."""
    assert plt.rcParams[key] == 7


# ---- Strokes: 0.75 outlines, 1.0 data lines ---------------------------------

@pytest.mark.parametrize("key", [
    "axes.linewidth",
    "patch.linewidth",
    "lines.markeredgewidth",
    "grid.linewidth",
    "xtick.major.width",
    "ytick.major.width",
])
def test_outline_strokes_are_075(key):
    assert plt.rcParams[key] == 0.75


def test_edgewidth_is_075():
    """The publiplots global for every stroke that outlines a shape."""
    assert pp.rcParams["edgewidth"] == 0.75


def test_data_line_width_is_1():
    """1.33x the 0.75 frame: data outweighs furniture without crowding."""
    assert plt.rcParams["lines.linewidth"] == 1.0


def test_data_lines_are_heavier_than_the_frame():
    """The invariant behind the split -- if this ever inverts, the design broke."""
    assert plt.rcParams["lines.linewidth"] > pp.rcParams["edgewidth"]


# ---- Furniture colour: black ink, alpha carries the dimming -----------------

def test_spines_are_black():
    assert plt.rcParams["axes.edgecolor"] == "black"


def test_gridlines_are_black_dimmed_by_alpha():
    """Colour is the ink, alpha is the dimmer -- one job each, no double-dip."""
    assert plt.rcParams["grid.color"] == "black"
    assert plt.rcParams["grid.alpha"] == 0.15


def test_grid_renders_as_light_gray():
    """black at alpha 0.15 on white == 0.85 gray, matching the previous
    0.8-gray-at-0.8-alpha appearance (0.840). This is a mechanism change,
    not a darkness change."""
    effective = 1.0 - plt.rcParams["grid.alpha"]
    assert effective == pytest.approx(0.85, abs=0.01)


# ---- Layout ----------------------------------------------------------------

def test_default_axes_size_is_40mm_square():
    assert pp.rcParams["subplots.axes_size"] == (40.0, 40.0)


# ---- edgewidth is a real, settable publiplots key --------------------------

def test_edgewidth_is_settable_and_restorable():
    saved = pp.rcParams["edgewidth"]
    try:
        pp.rcParams["edgewidth"] = 2.0
        assert pp.rcParams["edgewidth"] == 2.0
    finally:
        pp.rcParams["edgewidth"] = saved
    assert pp.rcParams["edgewidth"] == saved
