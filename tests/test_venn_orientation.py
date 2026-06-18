"""Tests for the orientation parameter on the 2-way Venn diagram."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import publiplots as pp
from publiplots.plot.venn.geometry import compute_2way_geometry, get_geometry


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def test_vertical_stacks_first_set_on_top():
    circles, labels, set_labels = compute_2way_geometry(orientation="vertical")
    # Both circle centers lie on the vertical axis (x == 0)
    assert circles[0].x_offset == pytest.approx(0.0)
    assert circles[1].x_offset == pytest.approx(0.0)
    # Set A (index 0) sits above Set B (index 1)
    assert circles[0].y_offset > circles[1].y_offset


def test_vertical_intersection_labels():
    _, labels, _ = compute_2way_geometry(orientation="vertical")
    # "10" (only A) above origin, "01" (only B) below, "11" (both) at origin
    assert labels["10"][0] == pytest.approx(0.0)
    assert labels["01"][0] == pytest.approx(0.0)
    assert labels["10"][1] > 0
    assert labels["01"][1] < 0
    assert labels["11"] == pytest.approx((0.0, 0.0))


def test_vertical_set_labels_at_outer_ends():
    circles, _, set_labels = compute_2way_geometry(orientation="vertical")
    # Set A label above the top circle, Set B label below the bottom circle
    assert set_labels[0][0] == pytest.approx(0.0)
    assert set_labels[1][0] == pytest.approx(0.0)
    assert set_labels[0][1] > circles[0].y_offset
    assert set_labels[1][1] < circles[1].y_offset


def test_vertical_is_rotation_of_horizontal():
    h_circles, h_labels, _ = compute_2way_geometry(orientation="horizontal")
    v_circles, v_labels, _ = compute_2way_geometry(orientation="vertical")
    # 90-deg clockwise rotation: (x, y) -> (y, -x)
    for hc, vc in zip(h_circles, v_circles):
        assert vc.x_offset == pytest.approx(hc.y_offset)
        assert vc.y_offset == pytest.approx(-hc.x_offset)
        assert vc.radius_a == pytest.approx(hc.radius_a)
        assert vc.radius_b == pytest.approx(hc.radius_b)
    for key in h_labels:
        hx, hy = h_labels[key]
        vx, vy = v_labels[key]
        assert (vx, vy) == pytest.approx((hy, -hx))


def test_horizontal_default_unchanged():
    # Default arg and explicit "horizontal" must be identical
    default = compute_2way_geometry()
    explicit = compute_2way_geometry(orientation="horizontal")
    assert [c.x_offset for c in default[0]] == [c.x_offset for c in explicit[0]]
    assert default[1] == explicit[1]
    assert default[2] == explicit[2]
    # And matches the known horizontal centers
    circles, _, _ = default
    assert circles[0].x_offset < 0  # Set A on the left
    assert circles[1].x_offset > 0  # Set B on the right
    assert circles[0].y_offset == pytest.approx(0.0)


def test_get_geometry_threads_orientation_for_2way():
    direct = compute_2way_geometry(orientation="vertical")
    via = get_geometry(2, orientation="vertical")
    # Same circle centers
    assert [c.y_offset for c in via[0]] == [c.y_offset for c in direct[0]]
    assert via[1] == direct[1]


def test_get_geometry_3way_unaffected_by_default_orientation():
    # New signature must not change 3/4/5-way output for the default
    circles, labels, set_labels = get_geometry(3)
    assert len(circles) == 3
    assert "111" in labels


def test_venn_vertical_returns_axes_with_stacked_circles():
    from matplotlib.patches import Ellipse
    ax = pp.venn(sets=[{1, 2, 3}, {3, 4, 5}], orientation="vertical")
    ellipses = [p for p in ax.patches if isinstance(p, Ellipse)]
    assert len(ellipses) == 2
    cy = [e.center[1] for e in ellipses]
    cx = [e.center[0] for e in ellipses]
    # Both on the vertical axis, first set above second
    assert cx[0] == pytest.approx(0.0)
    assert cx[1] == pytest.approx(0.0)
    assert cy[0] > cy[1]


def test_venn_vertical_with_three_sets_raises():
    with pytest.raises(ValueError, match="2-way"):
        pp.venn(sets=[{1}, {2}, {3}], orientation="vertical")


def test_venn_invalid_orientation_raises():
    with pytest.raises(ValueError, match="horizontal.*vertical|orientation"):
        pp.venn(sets=[{1, 2}, {2, 3}], orientation="diagonal")


def test_venn_horizontal_default_unchanged():
    from matplotlib.patches import Ellipse
    ax = pp.venn(sets=[{1, 2, 3}, {3, 4, 5}])
    ellipses = [p for p in ax.patches if isinstance(p, Ellipse)]
    cy = [e.center[1] for e in ellipses]
    # Horizontal: both centers on the y == 0 axis
    assert cy[0] == pytest.approx(0.0)
    assert cy[1] == pytest.approx(0.0)
