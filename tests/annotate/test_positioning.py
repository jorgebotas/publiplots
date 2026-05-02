"""Pure-math tests for anchor resolution and fit check."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from publiplots.annotate._cache import BarRecord
from publiplots.annotate._positioning import (
    fit_check,
    mm_to_data,
    resolve_anchor,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _vbar(ax, x=0.5, width=0.8, height=10.0, bottom=0.0):
    return ax.bar([x], [height], width=width, bottom=bottom)[0]


def _hbar(ax, y=0.5, height=0.8, width=10.0, left=0.0):
    return ax.barh([y], [width], height=height, left=left)[0]


def _make_record(patch, value, err_low=None, err_high=None):
    return BarRecord(patch=patch, value=value, err_low=err_low,
                     err_high=err_high, hue_color=None)


# ---------- resolve_anchor: vertical + positive ----------

def test_vertical_positive_outside_no_errorbar():
    fig, ax = plt.subplots()
    p = _vbar(ax, x=1.0, width=0.8, height=5.0)
    bar = _make_record(p, value=5.0)
    x, y, ha, va = resolve_anchor(bar, anchor="outside", orient="v",
                                  offset_mm=0.0, ax=ax)
    assert ha == "center"
    assert va == "bottom"
    assert x == pytest.approx(1.0)
    assert y == pytest.approx(5.0)  # top of bar


def test_vertical_positive_outside_with_errorbar():
    fig, ax = plt.subplots()
    p = _vbar(ax, x=1.0, width=0.8, height=5.0)
    bar = _make_record(p, value=5.0, err_low=4.5, err_high=5.5)
    x, y, ha, va = resolve_anchor(bar, anchor="outside", orient="v",
                                  offset_mm=0.0, ax=ax)
    assert y == pytest.approx(5.5)  # past the upper cap


def test_vertical_positive_inside():
    fig, ax = plt.subplots()
    p = _vbar(ax, x=1.0, width=0.8, height=5.0)
    bar = _make_record(p, value=5.0)
    x, y, ha, va = resolve_anchor(bar, anchor="inside", orient="v",
                                  offset_mm=0.0, ax=ax)
    assert ha == "center"
    assert va == "top"
    assert y == pytest.approx(5.0)  # offset_mm=0, inner_pad=0 for this parametrization


def test_vertical_positive_base():
    fig, ax = plt.subplots()
    p = _vbar(ax, x=1.0, width=0.8, height=5.0)
    bar = _make_record(p, value=5.0)
    x, y, ha, va = resolve_anchor(bar, anchor="base", orient="v",
                                  offset_mm=0.0, ax=ax)
    assert ha == "center"
    assert va == "bottom"
    assert y == pytest.approx(0.0)


def test_vertical_positive_center():
    fig, ax = plt.subplots()
    p = _vbar(ax, x=1.0, width=0.8, height=5.0)
    bar = _make_record(p, value=5.0)
    x, y, ha, va = resolve_anchor(bar, anchor="center", orient="v",
                                  offset_mm=0.0, ax=ax)
    assert va == "center"
    assert y == pytest.approx(2.5)


# ---------- resolve_anchor: vertical + negative ----------

def test_vertical_negative_outside_no_errorbar():
    fig, ax = plt.subplots()
    p = _vbar(ax, x=1.0, width=0.8, height=-3.0)
    bar = _make_record(p, value=-3.0)
    x, y, ha, va = resolve_anchor(bar, anchor="outside", orient="v",
                                  offset_mm=0.0, ax=ax)
    assert ha == "center"
    assert va == "top"
    assert y == pytest.approx(-3.0)


def test_vertical_negative_outside_with_errorbar():
    fig, ax = plt.subplots()
    p = _vbar(ax, x=1.0, width=0.8, height=-3.0)
    bar = _make_record(p, value=-3.0, err_low=-3.5, err_high=-2.5)
    x, y, ha, va = resolve_anchor(bar, anchor="outside", orient="v",
                                  offset_mm=0.0, ax=ax)
    assert y == pytest.approx(-3.5)


def test_vertical_negative_base():
    fig, ax = plt.subplots()
    p = _vbar(ax, x=1.0, width=0.8, height=-3.0)
    bar = _make_record(p, value=-3.0)
    x, y, ha, va = resolve_anchor(bar, anchor="base", orient="v",
                                  offset_mm=0.0, ax=ax)
    assert ha == "center"
    assert va == "top"
    assert y == pytest.approx(0.0)


# ---------- resolve_anchor: horizontal ----------

def test_horizontal_positive_outside():
    fig, ax = plt.subplots()
    p = _hbar(ax, y=1.0, height=0.8, width=5.0)
    bar = _make_record(p, value=5.0)
    x, y, ha, va = resolve_anchor(bar, anchor="outside", orient="h",
                                  offset_mm=0.0, ax=ax)
    assert va == "center"
    assert ha == "left"
    assert x == pytest.approx(5.0)


def test_horizontal_negative_outside():
    fig, ax = plt.subplots()
    p = _hbar(ax, y=1.0, height=0.8, width=-3.0)
    bar = _make_record(p, value=-3.0)
    x, y, ha, va = resolve_anchor(bar, anchor="outside", orient="h",
                                  offset_mm=0.0, ax=ax)
    assert ha == "right"
    assert x == pytest.approx(-3.0)


# ---------- log scale + base ----------

def test_log_scale_base_uses_ylim_lower():
    fig, ax = plt.subplots()
    p = _vbar(ax, x=1.0, width=0.8, height=5.0)
    ax.set_yscale("log")
    ax.set_ylim(0.1, 10.0)
    bar = _make_record(p, value=5.0)
    x, y, ha, va = resolve_anchor(bar, anchor="base", orient="v",
                                  offset_mm=0.0, ax=ax)
    assert y == pytest.approx(0.1)


# ---------- offset effect ----------

def test_positive_offset_moves_outward_vertical():
    fig, ax = plt.subplots()
    p = _vbar(ax, x=1.0, width=0.8, height=5.0)
    ax.set_ylim(0, 10)
    fig.canvas.draw()
    bar = _make_record(p, value=5.0)
    _, y0, _, _ = resolve_anchor(bar, anchor="outside", orient="v",
                                 offset_mm=0.0, ax=ax)
    _, y1, _, _ = resolve_anchor(bar, anchor="outside", orient="v",
                                 offset_mm=5.0, ax=ax)
    assert y1 > y0


# ---------- mm_to_data ----------

def test_mm_to_data_roundtrip_y():
    fig, ax = plt.subplots()
    ax.set_ylim(0, 10)
    fig.canvas.draw()
    # 10 mm in y should yield a positive data delta.
    delta = mm_to_data(10.0, ax, axis="y")
    assert delta > 0


def test_mm_to_data_zero_is_zero():
    fig, ax = plt.subplots()
    ax.set_ylim(0, 10)
    fig.canvas.draw()
    assert mm_to_data(0.0, ax, axis="y") == 0.0


# ---------- fit_check ----------

def test_fit_check_outside_always_fits():
    fig, ax = plt.subplots()
    ax.bar([0], [1.0])
    t = ax.text(0, 0, "hello")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = ax.patches[0].get_window_extent(renderer)
    assert fit_check(t, bbox, orient="v", anchor="outside", renderer=renderer) == "fits"


def test_fit_check_short_bar_inside_reanchors():
    fig, ax = plt.subplots()
    ax.bar([0], [0.001])  # extremely short
    ax.set_ylim(0, 100)  # keep the bar short in display space too (without this, autoscale makes it fill the axes)
    t = ax.text(0, 0.0005, "huge label", fontsize=20)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = ax.patches[0].get_window_extent(renderer)
    assert fit_check(t, bbox, orient="v", anchor="inside", renderer=renderer) == "reanchor_outside"


def test_fit_check_tall_bar_inside_fits():
    fig, ax = plt.subplots()
    ax.bar([0], [100.0])
    ax.set_ylim(0, 110)
    t = ax.text(0, 50, "x", fontsize=8)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = ax.patches[0].get_window_extent(renderer)
    assert fit_check(t, bbox, orient="v", anchor="inside", renderer=renderer) == "fits"
