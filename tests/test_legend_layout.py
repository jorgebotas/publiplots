"""Unit tests for LegendLayout (pure mm-based geometry, no matplotlib)."""
import pytest
from publiplots.utils.legend_layout import LegendLayout


def test_fresh_layout_starts_at_y_offset():
    layout = LegendLayout(x_offset=2, gap=2, column_spacing=5, vpad=5)
    layout.reset_to(axes_height_mm=80.0)
    assert layout.current_x == 2
    assert layout.current_y == 80.0 - 5  # axes_height - vpad


def test_fresh_layout_uses_explicit_y_offset():
    layout = LegendLayout(x_offset=2, y_offset=50.0, gap=2)
    layout.reset_to(axes_height_mm=80.0)
    # With explicit y_offset, reset_to should honor it rather than (height - vpad)
    assert layout.current_y == 50.0


def test_advance_y_subtracts_element_height_plus_gap():
    layout = LegendLayout(gap=2)
    layout.reset_to(axes_height_mm=80.0)
    start_y = layout.current_y
    layout.advance_y(element_height=10.0)
    assert layout.current_y == start_y - 10.0 - 2


def test_update_width_is_monotonic_max():
    layout = LegendLayout()
    layout.reset_to(axes_height_mm=80.0)
    layout.update_width(15)
    assert layout.current_column_width == 15
    layout.update_width(10)  # smaller - should NOT shrink
    assert layout.current_column_width == 15
    layout.update_width(20)
    assert layout.current_column_width == 20


def test_check_overflow_returns_true_when_required_exceeds_current_y():
    layout = LegendLayout(vpad=5)
    layout.reset_to(axes_height_mm=20.0)
    # current_y = 15, required = 20 -> overflow
    assert layout.check_overflow(required_height=20.0) is True
    # required = 10 -> fits
    assert layout.check_overflow(required_height=10.0) is False


def test_start_new_column_records_width_and_shifts_x():
    layout = LegendLayout(x_offset=2, column_spacing=5)
    layout.reset_to(axes_height_mm=80.0)
    layout.update_width(12)
    start_y = layout.current_y
    layout.advance_y(10)  # move down
    layout.start_new_column()
    assert layout.columns == [12]
    assert layout.current_x == 2 + 12 + 5  # x_offset + col_width + spacing
    assert layout.current_y == start_y  # y reset to column top
    assert layout.current_column_width == 0  # reset


def test_multiple_new_columns_accumulate():
    layout = LegendLayout(x_offset=2, column_spacing=5)
    layout.reset_to(axes_height_mm=80.0)
    layout.update_width(10)
    layout.start_new_column()
    layout.update_width(8)
    layout.start_new_column()
    assert layout.columns == [10, 8]
    # Second column shifted by col1 width + spacing;
    # third column shifted by col1 + col2 widths + 2*spacing.
    assert layout.current_x == 2 + 10 + 5 + 8 + 5


def test_reset_to_clears_state():
    layout = LegendLayout(x_offset=2, vpad=5)
    layout.reset_to(axes_height_mm=80.0)
    layout.update_width(10)
    layout.advance_y(5)
    layout.start_new_column()
    # Now reset
    layout.reset_to(axes_height_mm=60.0)
    assert layout.current_x == 2
    assert layout.current_y == 60.0 - 5
    assert layout.columns == []
    assert layout.current_column_width == 0


def test_zero_width_elements_do_not_corrupt_state():
    layout = LegendLayout()
    layout.reset_to(axes_height_mm=80.0)
    layout.update_width(0)
    assert layout.current_column_width == 0
    layout.advance_y(0)
    layout.start_new_column()
    assert layout.columns == [0]