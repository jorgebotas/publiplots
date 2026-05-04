"""Unit tests for LegendLayout (pure mm-based geometry, no matplotlib)."""
import pytest
from publiplots.utils.legend_layout import LegendLayout


def test_fresh_layout_starts_at_y_offset():
    layout = LegendLayout(x_offset=2, gap=2, column_spacing=5, vpad=5)
    layout.reset_to(edge_length_mm=80.0)
    assert layout.current_outward == 2
    assert layout.current_along == 80.0 - 5  # edge_length - vpad


def test_fresh_layout_uses_explicit_y_offset():
    layout = LegendLayout(x_offset=2, y_offset=50.0, gap=2)
    layout.reset_to(edge_length_mm=80.0)
    # With explicit y_offset, reset_to should honor it rather than (length - vpad)
    assert layout.current_along == 50.0


def test_advance_along_subtracts_element_size_plus_gap():
    layout = LegendLayout(gap=2)
    layout.reset_to(edge_length_mm=80.0)
    start = layout.current_along
    layout.advance_along(element_along=10.0)
    assert layout.current_along == start - 10.0 - 2


def test_update_width_is_monotonic_max():
    layout = LegendLayout()
    layout.reset_to(edge_length_mm=80.0)
    layout.update_width(15)
    assert layout.current_band_width == 15
    layout.update_width(10)  # smaller - should NOT shrink
    assert layout.current_band_width == 15
    layout.update_width(20)
    assert layout.current_band_width == 20


def test_check_overflow_returns_true_when_required_exceeds_current_along():
    layout = LegendLayout(vpad=5)
    layout.reset_to(edge_length_mm=20.0)
    # current_along = 15, required = 20 -> overflow
    assert layout.check_overflow(required_along=20.0) is True
    # required = 10 -> fits
    assert layout.check_overflow(required_along=10.0) is False


def test_start_new_band_records_width_and_shifts_outward():
    layout = LegendLayout(x_offset=2, column_spacing=5)
    layout.reset_to(edge_length_mm=80.0)
    layout.update_width(12)
    start = layout.current_along
    layout.advance_along(10)  # move down the edge
    layout.start_new_band()
    assert layout.bands == [12]
    assert layout.current_outward == 2 + 12 + 5  # x_offset + band_width + spacing
    assert layout.current_along == start  # along reset to band top
    assert layout.current_band_width == 0  # reset


def test_multiple_new_bands_accumulate():
    layout = LegendLayout(x_offset=2, column_spacing=5)
    layout.reset_to(edge_length_mm=80.0)
    layout.update_width(10)
    layout.start_new_band()
    layout.update_width(8)
    layout.start_new_band()
    assert layout.bands == [10, 8]
    # Second band shifted by band1 width + spacing;
    # third band shifted by band1 + band2 widths + 2*spacing.
    assert layout.current_outward == 2 + 10 + 5 + 8 + 5


def test_reset_to_clears_state():
    layout = LegendLayout(x_offset=2, vpad=5)
    layout.reset_to(edge_length_mm=80.0)
    layout.update_width(10)
    layout.advance_along(5)
    layout.start_new_band()
    # Now reset
    layout.reset_to(edge_length_mm=60.0)
    assert layout.current_outward == 2
    assert layout.current_along == 60.0 - 5
    assert layout.bands == []
    assert layout.current_band_width == 0


def test_zero_size_elements_do_not_corrupt_state():
    layout = LegendLayout()
    layout.reset_to(edge_length_mm=80.0)
    layout.update_width(0)
    assert layout.current_band_width == 0
    layout.advance_along(0)
    layout.start_new_band()
    assert layout.bands == [0]


def test_along_from_start_begins_at_vpad():
    layout = LegendLayout(vpad=5, gap=2)
    layout.reset_to(edge_length_mm=80.0)
    assert layout.along_from_start == 5


def test_along_from_start_advances_correctly():
    layout = LegendLayout(vpad=5, gap=2)
    layout.reset_to(edge_length_mm=80.0)
    layout.advance_along(10.0)
    assert layout.along_from_start == 5 + 10 + 2  # vpad + size + gap
    layout.advance_along(3.0)
    assert layout.along_from_start == 5 + 10 + 2 + 3 + 2


def test_along_from_start_is_independent_of_edge_length():
    """y_from_top / along_from_start must not change when anchor size changes."""
    layout1 = LegendLayout(vpad=5)
    layout1.reset_to(edge_length_mm=80.0)
    layout1.advance_along(10.0)
    y1 = layout1.along_from_start

    layout2 = LegendLayout(vpad=5)
    layout2.reset_to(edge_length_mm=40.0)  # half the length
    layout2.advance_along(10.0)
    y2 = layout2.along_from_start
    assert y1 == y2


def test_along_from_start_resets_on_new_band():
    layout = LegendLayout(vpad=5, gap=2)
    layout.reset_to(edge_length_mm=80.0)
    layout.advance_along(10.0)
    assert layout.along_from_start > 5
    layout.start_new_band()
    assert layout.along_from_start == 5
