# `pp.annotate()` In-Plot Value Labels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `pp.annotate(ax, kind="bar_values", ...)` as the primitive for in-plot value labels on barplots, with `pp.barplot(..., annotate=True | dict)` as sugar. Orientation/errorbar/sign-aware positioning; `anchor` selectable (outside/inside/base/center); contrast-aware coloring with compositing for translucent bars; auto axis-limit expansion gated on autoscale state.

**Architecture:** New top-level package `src/publiplots/annotate/` with five files: a `_dispatcher` with a strategy registry, pure-math `_positioning` and `_color` modules, a `_cache` dataclass contract, and one `bar_values` strategy that orchestrates them. A `annotate` kwarg on `barplot()` attaches `BarValueMeta` to the axes and calls the primitive. Introspection fallback walks `ax.patches`/`ax.lines` for foreign axes.

**Tech Stack:** Python 3.9+, matplotlib, seaborn, numpy, pytest (Agg backend for draw tests).

**Worktree:** `.claude/worktrees/pp-annotate` on branch `worktree-pp-annotate`. Baseline: 120 tests passing + 1 pre-existing unrelated failure (`test_subplots_axes_size_none_uses_rcparams_default`).

**Spec:** `docs/superpowers/specs/2026-05-01-pp-annotate-design.md` (authoritative — re-read before starting).

---

## File structure

**Create:**
- `src/publiplots/annotate/__init__.py` — re-exports `annotate`
- `src/publiplots/annotate/_cache.py` — `BarRecord`, `BarValueMeta`, `_introspect(ax)`
- `src/publiplots/annotate/_positioning.py` — `resolve_anchor(...)`, `fit_check(...)`, `mm_to_data(...)`
- `src/publiplots/annotate/_color.py` — `resolve_color(...)`
- `src/publiplots/annotate/bar_values.py` — `_bar_values_strategy(ax, ...)`
- `src/publiplots/annotate/_dispatcher.py` — `annotate(ax, kind=..., ...)`, `_STRATEGIES` registry
- `tests/annotate/__init__.py` — empty marker
- `tests/annotate/test_positioning.py`
- `tests/annotate/test_color.py`
- `tests/annotate/test_cache.py`
- `tests/annotate/test_dispatcher.py`
- `tests/annotate/test_bar_values.py`
- `examples/plots/plot_15_annotate.py` — gallery example

**Modify:**
- `src/publiplots/__init__.py` — add `from publiplots.annotate import annotate`
- `src/publiplots/plot/bar.py` — add `annotate` kwarg to `barplot()`; build `BarValueMeta`; call `annotate(ax, ...)` at the end

---

## Rules of the road

- **TDD only.** Write the failing test first, verify it fails, then the minimum code to make it pass.
- **No placeholder comments.** If code changes, show the exact code.
- **mm for user-facing offsets.** Conversion to data coordinates happens once, at the transform boundary.
- **Run the full suite (`uv run pytest tests/ -q`) after each task.** Regressions in existing tests fail the task. The known failure `test_subplots_axes_size_none_uses_rcparams_default` is pre-existing; ignore if unchanged.
- **Commit after every task** with a conventional-commits message (`feat:`, `test:`, `docs:`, `refactor:`).
- **Always include the Agg backend header** in new test files:
  ```python
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  ```
- **Imports from `publiplots.annotate`** never reach into `publiplots.plot.*`. `plot/bar.py` imports from `publiplots.annotate`, not the other way round.

---

## Task 1: Create `_cache.py` — `BarRecord`, `BarValueMeta`, and `_introspect`

**Files:**
- Create: `src/publiplots/annotate/__init__.py`
- Create: `src/publiplots/annotate/_cache.py`
- Create: `tests/annotate/__init__.py`
- Create: `tests/annotate/test_cache.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/annotate/__init__.py` with empty content (marker file).

Create `tests/annotate/test_cache.py`:

```python
"""Tests for the BarValueMeta cache contract and axes introspection."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from publiplots.annotate._cache import BarRecord, BarValueMeta, _introspect


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_bar_value_meta_minimal_construct():
    meta = BarValueMeta(orient="v", bars=[], errorbar_kind=None,
                        hue_active=False, owner_is_publiplots=False)
    assert meta.orient == "v"
    assert meta.bars == []
    assert meta.hue_active is False
    assert meta.owner_is_publiplots is False


def test_bar_record_fields():
    fig, ax = plt.subplots()
    rect = ax.bar([0], [1.0])[0]
    rec = BarRecord(patch=rect, value=1.0, err_low=None, err_high=None, hue_color=None)
    assert rec.value == 1.0
    assert rec.err_low is None
    assert rec.err_high is None


def test_introspect_vertical_bars_no_errorbars():
    fig, ax = plt.subplots()
    ax.bar([0, 1, 2], [1.0, 2.0, 3.0], width=0.8)
    meta = _introspect(ax)
    assert meta.orient == "v"
    assert len(meta.bars) == 3
    assert [b.value for b in meta.bars] == pytest.approx([1.0, 2.0, 3.0])
    assert all(b.err_low is None and b.err_high is None for b in meta.bars)
    assert meta.owner_is_publiplots is False
    assert meta.hue_active is False


def test_introspect_horizontal_bars_no_errorbars():
    fig, ax = plt.subplots()
    ax.barh([0, 1, 2], [1.0, 2.0, 3.0], height=0.8)
    meta = _introspect(ax)
    assert meta.orient == "h"
    assert [b.value for b in meta.bars] == pytest.approx([1.0, 2.0, 3.0])


def test_introspect_empty_axes():
    fig, ax = plt.subplots()
    meta = _introspect(ax)
    assert meta.bars == []


def test_introspect_vertical_bars_with_errorbars():
    fig, ax = plt.subplots()
    ax.bar([0, 1, 2], [1.0, 2.0, 3.0], yerr=[0.1, 0.2, 0.3], width=0.8)
    meta = _introspect(ax)
    assert len(meta.bars) == 3
    # Matplotlib's errorbar plots vertical segments whose endpoints match bar ± yerr.
    for bar, exp_low, exp_high, val in zip(
        meta.bars, [0.9, 1.8, 2.7], [1.1, 2.2, 3.3], [1.0, 2.0, 3.0]
    ):
        # err_low/err_high are full-extent y values (including the bar tip direction).
        # For positive bars, err_high is the top-cap y, err_low is the bottom-cap y.
        assert bar.err_high == pytest.approx(exp_high, rel=1e-2)
        assert bar.err_low == pytest.approx(exp_low, rel=1e-2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/annotate/test_cache.py -v`
Expected: ModuleNotFoundError or ImportError — `publiplots.annotate` does not exist yet.

- [ ] **Step 3: Write the minimal package skeleton and `_cache.py`**

Create `src/publiplots/annotate/__init__.py`:

```python
"""pp.annotate: in-plot value labels.

Public surface is the `annotate` function. All other modules are internal
and subject to change; do not import from them directly.
"""
from publiplots.annotate._dispatcher import annotate

__all__ = ["annotate"]
```

Create `src/publiplots/annotate/_cache.py`:

```python
"""Private cache contract between plot functions and annotation strategies.

`BarValueMeta` is attached to an Axes as `ax._publiplots_bar_meta` by
`pp.barplot(..., annotate=...)`. When absent (foreign axes), `_introspect`
reconstructs an equivalent meta by walking `ax.patches` and `ax.lines`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple

from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


RGBA = Tuple[float, float, float, float]


@dataclass
class BarRecord:
    patch: Rectangle
    value: float
    err_low: Optional[float]
    err_high: Optional[float]
    hue_color: Optional[RGBA]


@dataclass
class BarValueMeta:
    orient: Literal["v", "h"]
    bars: List[BarRecord]
    errorbar_kind: Optional[str]
    hue_active: bool
    owner_is_publiplots: bool


def _is_bar_rect(p) -> bool:
    if not isinstance(p, Rectangle):
        return False
    w, h = p.get_width(), p.get_height()
    if w <= 0 or h <= 0:
        return False
    # Matplotlib's axes frame and legend patches are Rectangles too; filter by
    # checking the patch is attached to ax.patches (caller iterates that list)
    # and that the bar is anchored on the data axis. Frame rectangles live in
    # ax.spines, not ax.patches, so iterating ax.patches already excludes them.
    return True


def _match_errorbars(
    lines: List[Line2D],
    rects: List[Rectangle],
    orient: Literal["v", "h"],
) -> List[Tuple[Optional[float], Optional[float]]]:
    """For each rect, find an errorbar line whose midpoint aligns with the bar's center.

    Matplotlib/seaborn errorbars are drawn as short Line2D segments — either
    whole vertical/horizontal bars or individual cap segments. We match by
    proximity of the segment's midpoint to each rect's center on the
    categorical axis.
    """
    if not rects:
        return []
    if orient == "v":
        widths = [r.get_width() for r in rects]
        tol = 0.5 * min(widths) if widths else 0.0
    else:
        heights = [r.get_height() for r in rects]
        tol = 0.5 * min(heights) if heights else 0.0

    result: List[Tuple[Optional[float], Optional[float]]] = []
    for r in rects:
        if orient == "v":
            center = r.get_x() + r.get_width() / 2.0
            # Find vertical line whose x is near bar center (single vertical bar,
            # not the caps, which are horizontal).
            low = high = None
            for ln in lines:
                xs, ys = ln.get_xdata(), ln.get_ydata()
                if len(xs) < 2:
                    continue
                # Vertical segment: all xs roughly equal, and that x near center.
                if max(xs) - min(xs) < 1e-6 and abs(xs[0] - center) <= tol:
                    low = min(ys)
                    high = max(ys)
                    break
            result.append((low, high))
        else:
            center = r.get_y() + r.get_height() / 2.0
            low = high = None
            for ln in lines:
                xs, ys = ln.get_xdata(), ln.get_ydata()
                if len(xs) < 2:
                    continue
                if max(ys) - min(ys) < 1e-6 and abs(ys[0] - center) <= tol:
                    low = min(xs)
                    high = max(xs)
                    break
            result.append((low, high))
    return result


def _infer_orient(rects: List[Rectangle]) -> Literal["v", "h"]:
    """All bars share width → vertical; all share height → horizontal."""
    if not rects:
        return "v"
    widths = [r.get_width() for r in rects]
    heights = [r.get_height() for r in rects]
    w_spread = max(widths) - min(widths)
    h_spread = max(heights) - min(heights)
    return "v" if w_spread <= h_spread else "h"


def _introspect(ax: Axes) -> BarValueMeta:
    """Build a BarValueMeta from an already-drawn Axes."""
    rects = [p for p in ax.patches if _is_bar_rect(p)]
    orient = _infer_orient(rects)
    err_by_bar = _match_errorbars(list(ax.lines), rects, orient)
    bars: List[BarRecord] = []
    for r, (err_low, err_high) in zip(rects, err_by_bar):
        value = r.get_height() if orient == "v" else r.get_width()
        bars.append(BarRecord(
            patch=r,
            value=float(value),
            err_low=err_low,
            err_high=err_high,
            hue_color=tuple(r.get_facecolor()),
        ))
    return BarValueMeta(
        orient=orient,
        bars=bars,
        errorbar_kind=None,
        hue_active=False,
        owner_is_publiplots=False,
    )
```

Also create a temporary stub for the dispatcher so `__init__.py` imports cleanly — we'll flesh it out in Task 5. Create `src/publiplots/annotate/_dispatcher.py`:

```python
"""Dispatcher stub. Real implementation lands in Task 5."""
from typing import List

from matplotlib.axes import Axes
from matplotlib.text import Text


_STRATEGIES: dict = {}


def annotate(ax: Axes, kind: str = "bar_values", **kwargs) -> List[Text]:
    raise NotImplementedError("annotate() implementation lands in Task 5")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/annotate/test_cache.py -v`
Expected: all 6 tests pass.

Run: `uv run pytest tests/ -q`
Expected: all previously-passing tests still pass; new 6 tests pass; 1 pre-existing failure unchanged.

- [ ] **Step 5: Commit**

```bash
git add src/publiplots/annotate/__init__.py \
        src/publiplots/annotate/_cache.py \
        src/publiplots/annotate/_dispatcher.py \
        tests/annotate/__init__.py \
        tests/annotate/test_cache.py
git commit -m "$(cat <<'EOF'
feat(annotate): BarValueMeta cache contract and introspection fallback

New private contract between plot functions and annotation strategies:
BarRecord/BarValueMeta dataclasses plus _introspect() that reconstructs
meta by walking ax.patches and ax.lines for foreign axes.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Create `_positioning.py` — `resolve_anchor` and `fit_check`

**Files:**
- Create: `src/publiplots/annotate/_positioning.py`
- Create: `tests/annotate/test_positioning.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/annotate/test_positioning.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/annotate/test_positioning.py -v`
Expected: ModuleNotFoundError — `_positioning` does not exist.

- [ ] **Step 3: Write `_positioning.py`**

Create `src/publiplots/annotate/_positioning.py`:

```python
"""Anchor resolution and fit check. Pure math, no drawing.

`resolve_anchor` composes orientation × sign × anchor kind × errorbar
presence into a single (x, y, ha, va) tuple in data coordinates.
`fit_check` decides whether an already-drawn Text artist fits inside
its bar's bbox; callers use its verdict to re-anchor if needed.
"""
from __future__ import annotations

from typing import Literal, Tuple

from matplotlib.axes import Axes
from matplotlib.text import Text
from matplotlib.transforms import Bbox

from publiplots.annotate._cache import BarRecord


MM_TO_INCH = 1.0 / 25.4
INNER_PAD_MM = 0.75


def mm_to_data(mm: float, ax: Axes, axis: Literal["x", "y"]) -> float:
    """Convert a millimeter displacement to a data-coord delta along `axis`.

    Uses a two-point transform: (0, 0) → (dpi * mm/25.4, 0) in display space
    is mapped back to data space; the difference is the data-coord delta.
    """
    if mm == 0.0:
        return 0.0
    fig = ax.figure
    px = mm * fig.dpi * MM_TO_INCH
    inv = ax.transData.inverted()
    if axis == "y":
        (x0, y0) = inv.transform((0, 0))
        (x1, y1) = inv.transform((0, px))
        return y1 - y0
    else:
        (x0, y0) = inv.transform((0, 0))
        (x1, y1) = inv.transform((px, 0))
        return x1 - x0


def _bar_extents(bar: BarRecord, orient: Literal["v", "h"]):
    """Return (left, right, bottom, top) of the bar in data coords."""
    r = bar.patch
    return (r.get_x(), r.get_x() + r.get_width(),
            r.get_y(), r.get_y() + r.get_height())


def resolve_anchor(
    bar: BarRecord,
    anchor: Literal["outside", "inside", "base", "center"],
    orient: Literal["v", "h"],
    offset_mm: float,
    ax: Axes,
) -> Tuple[float, float, str, str]:
    """Return (x, y, ha, va) for a label anchor on `bar`."""
    left, right, bottom, top = _bar_extents(bar, orient)
    is_positive = bar.value >= 0

    if orient == "v":
        x_center = (left + right) / 2.0
        ha = "center"
        offset_data = mm_to_data(offset_mm, ax, axis="y")
        inner_pad_data = mm_to_data(INNER_PAD_MM, ax, axis="y")

        if anchor == "outside":
            if is_positive:
                edge = bar.err_high if bar.err_high is not None else top
                return x_center, edge + offset_data, ha, "bottom"
            else:
                edge = bar.err_low if bar.err_low is not None else bottom
                return x_center, edge - offset_data, ha, "top"
        if anchor == "inside":
            if is_positive:
                return x_center, top - inner_pad_data, ha, "top"
            else:
                return x_center, bottom + inner_pad_data, ha, "bottom"
        if anchor == "base":
            if ax.get_yscale() == "log":
                base = ax.get_ylim()[0]
            else:
                base = 0.0
            if is_positive:
                return x_center, base + offset_data, ha, "bottom"
            else:
                return x_center, base - offset_data, ha, "top"
        if anchor == "center":
            return x_center, (top + bottom) / 2.0, ha, "center"
        raise ValueError(f"unknown anchor: {anchor!r}")

    # orient == "h"
    y_center = (bottom + top) / 2.0
    va = "center"
    offset_data = mm_to_data(offset_mm, ax, axis="x")
    inner_pad_data = mm_to_data(INNER_PAD_MM, ax, axis="x")

    if anchor == "outside":
        if is_positive:
            edge = bar.err_high if bar.err_high is not None else right
            return edge + offset_data, y_center, "left", va
        else:
            edge = bar.err_low if bar.err_low is not None else left
            return edge - offset_data, y_center, "right", va
    if anchor == "inside":
        if is_positive:
            return right - inner_pad_data, y_center, "right", va
        else:
            return left + inner_pad_data, y_center, "left", va
    if anchor == "base":
        if ax.get_xscale() == "log":
            base = ax.get_xlim()[0]
        else:
            base = 0.0
        if is_positive:
            return base + offset_data, y_center, "left", va
        else:
            return base - offset_data, y_center, "right", va
    if anchor == "center":
        return (left + right) / 2.0, y_center, "center", va
    raise ValueError(f"unknown anchor: {anchor!r}")


def fit_check(
    text_artist: Text,
    bar_bbox_display: Bbox,
    orient: Literal["v", "h"],
    anchor: str,
    renderer,
) -> Literal["fits", "reanchor_outside"]:
    """Return 'fits' if text fits inside bar; 'reanchor_outside' otherwise.

    `anchor="outside"` always fits by construction.
    """
    if anchor == "outside":
        return "fits"
    tbb = text_artist.get_window_extent(renderer)
    margin_px = 1.0
    if orient == "v":
        return "fits" if tbb.height + 2 * margin_px <= bar_bbox_display.height else "reanchor_outside"
    else:
        return "fits" if tbb.width + 2 * margin_px <= bar_bbox_display.width else "reanchor_outside"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/annotate/test_positioning.py -v`
Expected: all tests pass.

Run: `uv run pytest tests/ -q`
Expected: full suite still passes (+ new positioning tests).

- [ ] **Step 5: Commit**

```bash
git add src/publiplots/annotate/_positioning.py tests/annotate/test_positioning.py
git commit -m "$(cat <<'EOF'
feat(annotate): resolve_anchor and fit_check positioning math

Pure-math module composing orient × sign × anchor × errorbar into a
single (x, y, ha, va) tuple in data coords. Log-scale awareness for
the 'base' anchor. fit_check reports whether an inside-anchored label
actually fits; callers re-anchor to outside on 'reanchor_outside'.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Create `_color.py` — `resolve_color`

**Files:**
- Create: `src/publiplots/annotate/_color.py`
- Create: `tests/annotate/test_color.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/annotate/test_color.py`:

```python
"""Tests for the color resolution module."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgba

from publiplots.annotate._cache import BarRecord
from publiplots.annotate._color import resolve_color


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _bar(ax, facecolor="#000000", alpha=1.0, edgecolor="#000000"):
    r = ax.bar([0], [1.0])[0]
    r.set_facecolor(to_rgba(facecolor, alpha=alpha))
    r.set_edgecolor(to_rgba(edgecolor))
    return r


def test_auto_dark_fill_inside_returns_light():
    fig, ax = plt.subplots()
    r = _bar(ax, facecolor="#000000")
    bar = BarRecord(patch=r, value=1.0, err_low=None, err_high=None, hue_color=None)
    rgba = resolve_color(bar, color="auto", anchor="inside", ax=ax)
    assert rgba == to_rgba("#ffffff")


def test_auto_light_fill_inside_returns_dark():
    fig, ax = plt.subplots()
    r = _bar(ax, facecolor="#ffffff")
    bar = BarRecord(patch=r, value=1.0, err_low=None, err_high=None, hue_color=None)
    rgba = resolve_color(bar, color="auto", anchor="inside", ax=ax)
    # text.color rcParam is dark by default; compare via rgba match
    assert rgba != to_rgba("#ffffff")


def test_auto_translucent_on_white_bg_returns_dark():
    """A saturated palette color with alpha=0.1 composited onto white should be near-white → dark text."""
    fig, ax = plt.subplots()
    ax.set_facecolor("white")
    r = _bar(ax, facecolor="#2a5ea6", alpha=0.1)  # vivid blue, mostly transparent
    bar = BarRecord(patch=r, value=1.0, err_low=None, err_high=None, hue_color=None)
    rgba = resolve_color(bar, color="auto", anchor="center", ax=ax)
    assert rgba != to_rgba("#ffffff")


def test_auto_outside_ignores_fill_uses_rcparam():
    fig, ax = plt.subplots()
    r = _bar(ax, facecolor="#000000")
    bar = BarRecord(patch=r, value=1.0, err_low=None, err_high=None, hue_color=None)
    expected = to_rgba(plt.rcParams["text.color"])
    assert resolve_color(bar, color="auto", anchor="outside", ax=ax) == expected


def test_hue_returns_hue_color_when_set():
    fig, ax = plt.subplots()
    r = _bar(ax, facecolor="#112233")
    hue = to_rgba("#ff8800")
    bar = BarRecord(patch=r, value=1.0, err_low=None, err_high=None, hue_color=hue)
    assert resolve_color(bar, color="hue", anchor="outside", ax=ax) == hue


def test_literal_color_passes_through():
    fig, ax = plt.subplots()
    r = _bar(ax, facecolor="#ffffff")
    bar = BarRecord(patch=r, value=1.0, err_low=None, err_high=None, hue_color=None)
    assert resolve_color(bar, color="#ff0000", anchor="inside", ax=ax) == to_rgba("#ff0000")
    assert resolve_color(bar, color=(0, 1, 0), anchor="inside", ax=ax) == to_rgba((0, 1, 0))


def test_auto_on_translucent_dark_fill_returns_light():
    """A very dark fill at alpha=1 should still yield light text."""
    fig, ax = plt.subplots()
    ax.set_facecolor("white")
    r = _bar(ax, facecolor="#111111", alpha=1.0)
    bar = BarRecord(patch=r, value=1.0, err_low=None, err_high=None, hue_color=None)
    assert resolve_color(bar, color="auto", anchor="inside", ax=ax) == to_rgba("#ffffff")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/annotate/test_color.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Write `_color.py`**

Create `src/publiplots/annotate/_color.py`:

```python
"""Text color resolution: auto (contrast), hue (palette), or literal."""
from __future__ import annotations

import colorsys
from typing import Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba

from publiplots.annotate._cache import BarRecord


RGBA = Tuple[float, float, float, float]
LUMINANCE_THRESHOLD = 0.408  # W3C / seaborn heatmap convention


def _effective_rgb(face_rgba: RGBA, bg_rgba: RGBA) -> Tuple[float, float, float]:
    """Composite a translucent face over an opaque background using its alpha."""
    fr, fg, fb, fa = face_rgba
    br, bg_, bb, ba = bg_rgba
    # Assume background is opaque (common case); if bg alpha < 1, recurse to fig bg later.
    r = fr * fa + br * (1 - fa)
    g = fg * fa + bg_ * (1 - fa)
    b = fb * fa + bb * (1 - fa)
    return r, g, b


def _background_rgba(ax: Axes) -> RGBA:
    bg = to_rgba(ax.get_facecolor())
    if bg[3] >= 1.0:
        return bg
    fig_bg = to_rgba(ax.figure.get_facecolor())
    return fig_bg


def resolve_color(
    bar: BarRecord,
    color: Union[str, tuple],
    anchor: str,
    ax: Axes,
) -> RGBA:
    """Return RGBA for the label text."""
    if isinstance(color, str) and color == "auto":
        if anchor in ("inside", "center", "base"):
            face = to_rgba(bar.patch.get_facecolor())
            bg = _background_rgba(ax)
            r, g, b = _effective_rgb(face, bg)
            _, lightness, _ = colorsys.rgb_to_hls(r, g, b)
            if lightness < LUMINANCE_THRESHOLD:
                return to_rgba("#ffffff")
            return to_rgba(plt.rcParams["text.color"])
        # outside — nothing's under the label
        return to_rgba(plt.rcParams["text.color"])

    if isinstance(color, str) and color == "hue":
        if bar.hue_color is not None:
            return to_rgba(bar.hue_color)
        edge = to_rgba(bar.patch.get_edgecolor())
        if edge[3] > 0:
            return edge
        return to_rgba(plt.rcParams["text.color"])

    # Literal color
    return to_rgba(color)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/annotate/test_color.py -v`
Expected: all 7 tests pass.

Run: `uv run pytest tests/ -q`
Expected: full suite still passes.

- [ ] **Step 5: Commit**

```bash
git add src/publiplots/annotate/_color.py tests/annotate/test_color.py
git commit -m "$(cat <<'EOF'
feat(annotate): resolve_color with luminance-aware auto mode

auto + inside/center/base: composite fill onto axes background, HLS
threshold at 0.408 → dark or light text. auto + outside: rcParams
text color (fill isn't under the label). hue: palette color from the
record. Literal colors pass through to_rgba.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Create `bar_values.py` — the strategy

**Files:**
- Create: `src/publiplots/annotate/bar_values.py`
- Create: `tests/annotate/test_bar_values.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/annotate/test_bar_values.py`:

```python
"""Integration tests for the bar_values strategy on real axes."""
import math
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from publiplots.annotate._cache import BarRecord, BarValueMeta
from publiplots.annotate.bar_values import _bar_values_strategy


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# -----------------------------------------------------------------------------
# Introspection path (foreign axes, no cache)
# -----------------------------------------------------------------------------

def test_foreign_vertical_bars_produces_labels():
    fig, ax = plt.subplots()
    ax.bar([0, 1, 2], [1.0, 2.0, 3.0])
    fig.canvas.draw()
    texts = _bar_values_strategy(ax, fmt=".1f", anchor="outside", offset=0.0,
                                 color="auto", pad=0.0)
    assert len(texts) == 3
    labels = [t.get_text() for t in texts]
    assert labels == ["1.0", "2.0", "3.0"]


def test_foreign_horizontal_bars_produces_labels():
    fig, ax = plt.subplots()
    ax.barh([0, 1, 2], [1.0, 2.0, 3.0])
    fig.canvas.draw()
    texts = _bar_values_strategy(ax, fmt=".1f", anchor="outside", offset=0.0,
                                 color="auto", pad=0.0)
    assert [t.get_text() for t in texts] == ["1.0", "2.0", "3.0"]


def test_empty_axes_returns_empty_and_warns():
    fig, ax = plt.subplots()
    with pytest.warns(UserWarning, match="no bars found"):
        texts = _bar_values_strategy(ax, fmt=".1f", anchor="outside", offset=0.0,
                                     color="auto", pad=0.0)
    assert texts == []


def test_nan_values_are_skipped():
    """If a bar's height is NaN, no label is drawn for it."""
    fig, ax = plt.subplots()
    ax.bar([0, 1, 2], [1.0, 2.0, 3.0])
    # Manually install a meta with one NaN value
    rects = [p for p in ax.patches]
    bars = [
        BarRecord(patch=rects[0], value=1.0, err_low=None, err_high=None, hue_color=None),
        BarRecord(patch=rects[1], value=float("nan"), err_low=None, err_high=None, hue_color=None),
        BarRecord(patch=rects[2], value=3.0, err_low=None, err_high=None, hue_color=None),
    ]
    ax._publiplots_bar_meta = BarValueMeta(
        orient="v", bars=bars, errorbar_kind=None,
        hue_active=False, owner_is_publiplots=True,
    )
    fig.canvas.draw()
    texts = _bar_values_strategy(ax, fmt=".1f", anchor="outside", offset=0.0,
                                 color="auto", pad=0.0)
    assert len(texts) == 2
    assert [t.get_text() for t in texts] == ["1.0", "3.0"]


def test_fmt_format_string_with_braces():
    fig, ax = plt.subplots()
    ax.bar([0], [3.14159])
    fig.canvas.draw()
    texts = _bar_values_strategy(ax, fmt="{:.3f}", anchor="outside", offset=0.0,
                                 color="auto", pad=0.0)
    assert texts[0].get_text() == "3.142"


def test_fmt_bare_spec():
    fig, ax = plt.subplots()
    ax.bar([0], [3.14159])
    fig.canvas.draw()
    texts = _bar_values_strategy(ax, fmt=".3f", anchor="outside", offset=0.0,
                                 color="auto", pad=0.0)
    assert texts[0].get_text() == "3.142"


# -----------------------------------------------------------------------------
# Fit-check fallback
# -----------------------------------------------------------------------------

def test_inside_fallback_to_outside_when_text_too_big():
    fig, ax = plt.subplots()
    ax.bar([0], [0.001])  # extremely short bar
    ax.set_ylim(0, 0.01)
    fig.canvas.draw()
    texts = _bar_values_strategy(
        ax, fmt=".3f", anchor="inside", offset=0.0, color="auto", pad=0.0,
        fontsize=20,
    )
    assert len(texts) == 1
    # Label should sit at or above the bar top, not inside the 0.001 height band.
    t = texts[0]
    _, y = t.get_position()
    assert t.get_va() == "bottom"
    assert y >= 0.001


# -----------------------------------------------------------------------------
# Auto limit expansion
# -----------------------------------------------------------------------------

def test_publiplots_owned_expands_limits_past_seaborn_default():
    """owner_is_publiplots=True always expands regardless of autoscale state."""
    fig, ax = plt.subplots()
    rects = ax.bar([0, 1], [10.0, 10.0])
    ax.set_ylim(0, 10)  # disables autoscale (like seaborn does)
    fig.canvas.draw()
    bars = [BarRecord(patch=r, value=10.0, err_low=None, err_high=None, hue_color=None)
            for r in rects]
    ax._publiplots_bar_meta = BarValueMeta(
        orient="v", bars=bars, errorbar_kind=None,
        hue_active=False, owner_is_publiplots=True,
    )
    _bar_values_strategy(ax, fmt=".1f", anchor="outside", offset=1.5,
                         color="auto", pad=1.0)
    _, top = ax.get_ylim()
    assert top > 10.0


def test_foreign_locked_limits_warn_on_clip():
    """Autoscale off on foreign axes → warn if labels would clip."""
    fig, ax = plt.subplots()
    ax.bar([0], [10.0])
    ax.set_ylim(0, 10)  # disables autoscale
    fig.canvas.draw()
    # Introspection → owner_is_publiplots=False
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        _bar_values_strategy(ax, fmt=".1f", anchor="outside", offset=1.5,
                             color="auto", pad=1.0)
    clip_warnings = [w for w in rec if "clipped" in str(w.message)]
    assert clip_warnings
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/annotate/test_bar_values.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Write `bar_values.py`**

Create `src/publiplots/annotate/bar_values.py`:

```python
"""Value-label strategy for barplots."""
from __future__ import annotations

import logging
import math
import warnings
from typing import List

from matplotlib.axes import Axes
from matplotlib.text import Text

from publiplots.annotate._cache import BarValueMeta, _introspect
from publiplots.annotate._color import resolve_color
from publiplots.annotate._positioning import (
    fit_check,
    mm_to_data,
    resolve_anchor,
)


logger = logging.getLogger(__name__)


def _get_or_introspect(ax: Axes) -> BarValueMeta:
    meta = getattr(ax, "_publiplots_bar_meta", None)
    if isinstance(meta, BarValueMeta):
        return meta
    return _introspect(ax)


def _format_value(value: float, fmt: str) -> str:
    if "{" in fmt and "}" in fmt:
        return fmt.format(value)
    return format(value, fmt)


def _ensure_renderer(ax: Axes):
    canvas = ax.figure.canvas
    renderer = canvas.get_renderer() if hasattr(canvas, "get_renderer") else None
    if renderer is None:
        canvas.draw()
        renderer = canvas.get_renderer()
    return renderer


def _bar_values_strategy(
    ax: Axes,
    *,
    fmt: str,
    anchor: str,
    offset: float,
    color,
    pad: float,
    **text_kws,
) -> List[Text]:
    meta = _get_or_introspect(ax)
    if not meta.bars:
        warnings.warn("pp.annotate: no bars found on axes", UserWarning, stacklevel=3)
        return []

    renderer = _ensure_renderer(ax)
    texts: List[Text] = []

    for bar in meta.bars:
        if math.isnan(bar.value):
            continue
        x, y, ha, va = resolve_anchor(bar, anchor, meta.orient, offset, ax)
        rgba = resolve_color(bar, color, anchor, ax)
        label = _format_value(bar.value, fmt)
        t = ax.text(x, y, label, ha=ha, va=va, color=rgba, **text_kws)

        if anchor != "outside":
            bbox = bar.patch.get_window_extent(renderer)
            if fit_check(t, bbox, meta.orient, anchor, renderer) == "reanchor_outside":
                x2, y2, ha2, va2 = resolve_anchor(bar, "outside", meta.orient, offset, ax)
                rgba2 = resolve_color(bar, color, "outside", ax)
                t.set_position((x2, y2))
                t.set_ha(ha2)
                t.set_va(va2)
                t.set_color(rgba2)
                logger.debug(
                    "pp.annotate: bar value=%s label re-anchored to 'outside'",
                    bar.value,
                )
        texts.append(t)

    _maybe_expand_limits(ax, texts, meta.orient, pad_mm=pad,
                        owner_is_publiplots=meta.owner_is_publiplots)
    return texts


def _maybe_expand_limits(
    ax: Axes,
    texts: List[Text],
    orient: str,
    pad_mm: float,
    owner_is_publiplots: bool,
) -> None:
    if not texts:
        return

    value_axis = "y" if orient == "v" else "x"
    autoscale_on = (ax.get_autoscaley_on() if value_axis == "y"
                    else ax.get_autoscalex_on())
    should_expand = owner_is_publiplots or autoscale_on

    renderer = _ensure_renderer(ax)
    inv = ax.transData.inverted()
    extents = [t.get_window_extent(renderer).transformed(inv) for t in texts]
    if value_axis == "y":
        need_min = min(e.y0 for e in extents)
        need_max = max(e.y1 for e in extents)
        get_lim, set_lim = ax.get_ylim, ax.set_ylim
    else:
        need_min = min(e.x0 for e in extents)
        need_max = max(e.x1 for e in extents)
        get_lim, set_lim = ax.get_xlim, ax.set_xlim

    cur_lo, cur_hi = get_lim()
    pad_data = mm_to_data(pad_mm, ax, axis=value_axis)

    if should_expand:
        set_lim(min(cur_lo, need_min - pad_data),
                max(cur_hi, need_max + pad_data))
    else:
        if need_min < cur_lo or need_max > cur_hi:
            warnings.warn(
                "pp.annotate: labels clipped; autoscale is off on this axis",
                UserWarning,
                stacklevel=3,
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/annotate/test_bar_values.py -v`
Expected: all tests pass.

Run: `uv run pytest tests/ -q`
Expected: full suite still passes.

- [ ] **Step 5: Commit**

```bash
git add src/publiplots/annotate/bar_values.py tests/annotate/test_bar_values.py
git commit -m "$(cat <<'EOF'
feat(annotate): bar_values strategy with fit-check and limit expansion

Orchestrates cache lookup → positioning → color → text draw → fit
check with outside fallback → limit expansion. owner_is_publiplots
axes always expand (seaborn disables autoscale); foreign axes respect
autoscale state and warn on clip. NaN values are skipped silently.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Flesh out `_dispatcher.py` — validation and registry

**Files:**
- Modify: `src/publiplots/annotate/_dispatcher.py`
- Create: `tests/annotate/test_dispatcher.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/annotate/test_dispatcher.py`:

```python
"""Tests for the annotate() dispatcher: validation and routing."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from publiplots.annotate import annotate


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_unknown_kind_raises_valueerror_with_known_list():
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="unknown kind"):
        annotate(ax, kind="not_a_kind")


def test_invalid_anchor_raises():
    fig, ax = plt.subplots()
    ax.bar([0], [1.0])
    with pytest.raises(ValueError, match="anchor"):
        annotate(ax, kind="bar_values", anchor="sideways")


def test_negative_offset_raises():
    fig, ax = plt.subplots()
    ax.bar([0], [1.0])
    with pytest.raises(ValueError, match=">= 0"):
        annotate(ax, kind="bar_values", offset=-1)


def test_negative_pad_raises():
    fig, ax = plt.subplots()
    ax.bar([0], [1.0])
    with pytest.raises(ValueError, match=">= 0"):
        annotate(ax, kind="bar_values", pad=-0.5)


def test_default_kind_is_bar_values():
    fig, ax = plt.subplots()
    ax.bar([0, 1], [1.0, 2.0])
    fig.canvas.draw()
    texts = annotate(ax)  # no kind arg
    assert len(texts) == 2


def test_registry_has_only_bar_values_in_v1():
    from publiplots.annotate._dispatcher import _STRATEGIES
    assert set(_STRATEGIES.keys()) == {"bar_values"}


def test_text_kws_forwarded():
    fig, ax = plt.subplots()
    ax.bar([0], [1.0])
    fig.canvas.draw()
    texts = annotate(ax, kind="bar_values", fontsize=14, fontweight="bold")
    assert texts[0].get_fontsize() == 14
    assert texts[0].get_fontweight() == "bold"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/annotate/test_dispatcher.py -v`
Expected: most tests fail with `NotImplementedError` from the Task 1 stub.

- [ ] **Step 3: Replace the stub with the real dispatcher**

Replace contents of `src/publiplots/annotate/_dispatcher.py`:

```python
"""Public entry point: annotate(ax, kind=..., ...).

Validates inputs and dispatches to a registered strategy. Adding a new
strategy means: write the `_<kind>_strategy` function in its own file and
register it in `_STRATEGIES` below.
"""
from __future__ import annotations

from typing import Callable, List, Union

from matplotlib.axes import Axes
from matplotlib.text import Text

from publiplots.annotate.bar_values import _bar_values_strategy


_STRATEGIES: dict[str, Callable] = {
    "bar_values": _bar_values_strategy,
}

_VALID_ANCHORS = {"outside", "inside", "base", "center"}


def annotate(
    ax: Axes,
    kind: str = "bar_values",
    *,
    fmt: str = ".2f",
    anchor: str = "outside",
    offset: float = 1.5,
    color: Union[str, tuple] = "auto",
    pad: float = 1.0,
    **text_kws,
) -> List[Text]:
    """Add value labels to plot marks on `ax`.

    Parameters
    ----------
    ax : Axes
        The axes to annotate. Must already have marks drawn on it.
    kind : str, default="bar_values"
        Which strategy to use. v1 ships 'bar_values'.
    fmt : str, default=".2f"
        Either a bare format spec (e.g. ".2f") or a format-string template
        containing {} (e.g. "{:,.1f}%").
    anchor : {"outside", "inside", "base", "center"}, default="outside"
        Where to place the label relative to the bar.
    offset : float, default=1.5
        Additional offset in millimeters, applied in the outward direction
        of the anchor.
    color : str or tuple, default="auto"
        "auto" = contrast-aware (compositing for translucent fills); "hue" =
        use the bar's palette color; any matplotlib color passes through.
    pad : float, default=1.0
        Extra padding in millimeters when auto-expanding axis limits.
    **text_kws
        Forwarded to `ax.text` (fontsize, fontweight, etc.).

    Returns
    -------
    list of Text
        The Text artists created, in bar order.
    """
    if kind not in _STRATEGIES:
        raise ValueError(
            f"unknown kind={kind!r}; known: {sorted(_STRATEGIES)}"
        )
    if anchor not in _VALID_ANCHORS:
        raise ValueError(
            f"anchor must be one of {sorted(_VALID_ANCHORS)}; got {anchor!r}"
        )
    if offset < 0 or pad < 0:
        raise ValueError("offset and pad must be >= 0")

    return _STRATEGIES[kind](
        ax, fmt=fmt, anchor=anchor, offset=offset, color=color, pad=pad,
        **text_kws,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/annotate/test_dispatcher.py -v`
Expected: all tests pass.

Run: `uv run pytest tests/ -q`
Expected: full suite still passes.

- [ ] **Step 5: Commit**

```bash
git add src/publiplots/annotate/_dispatcher.py tests/annotate/test_dispatcher.py
git commit -m "$(cat <<'EOF'
feat(annotate): dispatcher with input validation and registry

pp.annotate(ax, kind='bar_values', ...) validates kind, anchor, and
non-negative offset/pad, then routes to the strategy. Default kind is
'bar_values'; text_kws are forwarded verbatim to ax.text.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Export `annotate` at package top level

**Files:**
- Modify: `src/publiplots/__init__.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/annotate/test_dispatcher.py`:

```python
def test_annotate_is_exposed_on_pp():
    import publiplots as pp
    assert pp.annotate is annotate
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/annotate/test_dispatcher.py::test_annotate_is_exposed_on_pp -v`
Expected: AttributeError: module 'publiplots' has no attribute 'annotate'.

- [ ] **Step 3: Add the import**

In `src/publiplots/__init__.py`, find the "# Utilities" block (near the existing `from publiplots.utils.io import savefig, ...` line). Add below the utils imports:

```python
# Annotations
from publiplots.annotate import annotate
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/annotate/ -v`
Expected: all annotate tests pass.

Run: `uv run pytest tests/ -q`
Expected: full suite still passes.

- [ ] **Step 5: Commit**

```bash
git add src/publiplots/__init__.py tests/annotate/test_dispatcher.py
git commit -m "$(cat <<'EOF'
feat(annotate): expose pp.annotate at package top level

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Wire `annotate=` kwarg into `pp.barplot`

**Files:**
- Modify: `src/publiplots/plot/bar.py`
- Create: `tests/annotate/test_barplot_integration.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/annotate/test_barplot_integration.py`:

```python
"""Integration: pp.barplot(..., annotate=...) cache-building + end-to-end."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import publiplots as pp
from publiplots.annotate._cache import BarValueMeta


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _simple_df():
    return pd.DataFrame({
        "category": pd.Categorical(["A", "B", "C"]),
        "value": [1.0, 2.0, 3.0],
    })


def _grouped_df():
    rng = np.random.default_rng(0)
    rows = []
    for group in ("A", "B"):
        for cond in ("ctrl", "trt"):
            for v in rng.normal(loc=(1 if group == "A" else 2) + (0 if cond == "ctrl" else 1),
                                scale=0.1, size=8):
                rows.append({"group": group, "cond": cond, "y": float(v)})
    df = pd.DataFrame(rows)
    df["group"] = df["group"].astype("category")
    df["cond"] = df["cond"].astype("category")
    return df


def test_barplot_annotate_false_attaches_no_meta():
    fig, ax = pp.barplot(data=_simple_df(), x="category", y="value")
    assert not hasattr(ax, "_publiplots_bar_meta") or ax._publiplots_bar_meta is None


def test_barplot_annotate_true_attaches_meta_and_draws_labels():
    fig, ax = pp.barplot(data=_simple_df(), x="category", y="value", annotate=True)
    assert isinstance(ax._publiplots_bar_meta, BarValueMeta)
    assert ax._publiplots_bar_meta.owner_is_publiplots is True
    texts = [t for t in ax.texts]
    assert len(texts) == 3


def test_barplot_annotate_dict_forwarded():
    fig, ax = pp.barplot(data=_simple_df(), x="category", y="value",
                         annotate={"fmt": ".3f"})
    labels = [t.get_text() for t in ax.texts]
    assert labels == ["1.000", "2.000", "3.000"]


def test_barplot_annotate_with_hue_has_hue_active():
    fig, ax = pp.barplot(data=_grouped_df(), x="group", y="y", hue="cond",
                         annotate=True)
    meta = ax._publiplots_bar_meta
    assert meta.hue_active is True
    assert all(b.hue_color is not None for b in meta.bars)


def test_barplot_annotate_no_hue_hue_active_false():
    fig, ax = pp.barplot(data=_simple_df(), x="category", y="value", annotate=True)
    assert ax._publiplots_bar_meta.hue_active is False


def test_barplot_annotate_expands_ylim():
    df = pd.DataFrame({
        "category": pd.Categorical(["A", "B"]),
        "value": [10.0, 10.0],
    })
    fig, ax = pp.barplot(data=df, x="category", y="value", annotate=True)
    top = ax.get_ylim()[1]
    # seaborn's default would stop at ~10 + a small pad; with annotate the
    # label must fit above that, so top should be noticeably above 10.
    assert top > 10.0


def test_barplot_annotate_with_errorbars_anchors_past_cap():
    """With errorbar='se' and annotate=True, label y should sit above the cap."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "group": pd.Categorical(np.repeat(["A", "B"], 20)),
        "y": np.concatenate([
            rng.normal(1.0, 0.5, 20),
            rng.normal(2.0, 0.5, 20),
        ]),
    })
    fig, ax = pp.barplot(data=df, x="group", y="y", errorbar="se", annotate=True)
    meta = ax._publiplots_bar_meta
    # Each bar has an err_high > value (standard error extends above mean).
    for bar in meta.bars:
        assert bar.err_high is not None
        assert bar.err_high > bar.value
    # Corresponding label y coordinates should be at or above err_high.
    for bar, text in zip(meta.bars, ax.texts):
        _, y = text.get_position()
        assert y >= bar.err_high
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/annotate/test_barplot_integration.py -v`
Expected: TypeError — `barplot()` does not accept `annotate` kwarg.

- [ ] **Step 3: Add the kwarg, helper, and call site in `bar.py`**

In `src/publiplots/plot/bar.py`:

**3a. Extend imports.** Near the top, after the existing `from publiplots.utils.transparency import ArtistTracker` line, add:

```python
from publiplots.annotate._cache import BarRecord, BarValueMeta
```

**3b. Add the `annotate` parameter to the `barplot` signature.** Find the block `legend: bool = True, legend_kws: Optional[Dict] = None,` in the signature and add right after it (same indentation):

```python
    annotate: Union[bool, Dict, None] = None,
```

(The `Union` and `Dict` types are already imported at the top of the file.)

**3c. Document the parameter.** In the docstring's Parameters section, after the `legend_kws` entry and before `errorbar`, add:

```
    annotate : bool or dict, optional
        If True, label each bar with its aggregated value. Pass a dict to
        forward options to pp.annotate (e.g. {"fmt": ".3f", "anchor": "inside"}).
```

**3d. Build the cache and call `annotate()` at the end of `barplot()`.** Find `return fig, ax` (should be the very last line of the function body, currently at ~line 321). Insert immediately before it:

```python
    if annotate:
        meta = _build_bar_value_meta(
            ax=ax, data=data, x=x, y=y, hue=hue,
            categorical_axis=categorical_axis,
            palette=palette, errorbar=errorbar,
        )
        ax._publiplots_bar_meta = meta
        from publiplots.annotate import annotate as _annotate_fn
        opts = annotate if isinstance(annotate, dict) else {}
        _annotate_fn(ax, kind="bar_values", **opts)

```

**3e. Add the helper at the end of the file** (after `_apply_hatches_and_override_colors` — outside the `barplot` function):

```python
def _build_bar_value_meta(
    ax: Axes,
    data: pd.DataFrame,
    x: str,
    y: str,
    hue: Optional[str],
    categorical_axis: str,
    palette: Optional[Dict],
    errorbar: Optional[str],
) -> BarValueMeta:
    """Compute aggregated values + errorbar extents aligned with the drawn bars.

    We re-run the same aggregation seaborn uses. The resulting records are
    paired with the Rectangles on the axes in draw order. Hue colors come
    from the resolved palette map.
    """
    value_col = y if categorical_axis == x else x
    orient: str = "v" if categorical_axis == x else "h"

    # Aggregate and compute errorbar half-widths with the same spec seaborn used.
    agg = _aggregate_for_annotate(data, x=x, y=y, hue=hue,
                                  categorical_axis=categorical_axis,
                                  errorbar=errorbar)

    rects = [p for p in ax.patches
             if isinstance(p, Rectangle) and p.get_width() > 0 and p.get_height() > 0]

    # Pair each rectangle with an aggregation row, in draw order. Seaborn draws
    # bars in the same order it iterates the hue × categorical axis groupby,
    # which matches the order we produce from _aggregate_for_annotate.
    bars: List[BarRecord] = []
    for rect, row in zip(rects, agg):
        value = float(row["mean"])
        err_low = row.get("err_low")
        err_high = row.get("err_high")
        hue_color = None
        if hue is not None and palette is not None:
            key = row.get("hue_value")
            if key is not None and key in palette:
                hue_color = to_rgba(palette[key])
        if hue_color is None:
            hue_color = tuple(rect.get_facecolor())
        bars.append(BarRecord(
            patch=rect,
            value=value,
            err_low=err_low,
            err_high=err_high,
            hue_color=hue_color,
        ))
    return BarValueMeta(
        orient=orient,
        bars=bars,
        errorbar_kind=errorbar,
        hue_active=hue is not None,
        owner_is_publiplots=True,
    )


def _aggregate_for_annotate(
    data: pd.DataFrame,
    x: str,
    y: str,
    hue: Optional[str],
    categorical_axis: str,
    errorbar: Optional[str],
) -> List[Dict]:
    """Group by (categorical_axis [, hue]) and return mean plus err_low/err_high.

    Returned row ordering: categorical_axis outer, hue inner — matches
    seaborn's draw order (dodge-grouped bars within each x category).
    """
    import numpy as _np

    value_col = y if categorical_axis == x else x
    cat_categories = list(data[categorical_axis].cat.categories)

    def _aggregate_group(values: _np.ndarray) -> Dict:
        mean = float(_np.mean(values))
        n = len(values)
        if errorbar is None or n < 2:
            return {"mean": mean, "err_low": None, "err_high": None}
        if errorbar == "se":
            half = float(_np.std(values, ddof=1) / _np.sqrt(n))
        elif errorbar == "sd":
            half = float(_np.std(values, ddof=1))
        elif errorbar == "ci":
            # 95% CI using normal approx — consistent with seaborn's default
            half = float(1.96 * _np.std(values, ddof=1) / _np.sqrt(n))
        else:
            half = 0.0
        return {"mean": mean, "err_low": mean - half, "err_high": mean + half}

    rows: List[Dict] = []
    if hue is None:
        for cat in cat_categories:
            mask = data[categorical_axis] == cat
            agg = _aggregate_group(data.loc[mask, value_col].to_numpy())
            rows.append(agg)
        return rows

    hue_categories = list(data[hue].cat.categories)
    for cat in cat_categories:
        for h in hue_categories:
            mask = (data[categorical_axis] == cat) & (data[hue] == h)
            vals = data.loc[mask, value_col].to_numpy()
            if len(vals) == 0:
                continue
            agg = _aggregate_group(vals)
            agg["hue_value"] = h
            rows.append(agg)
    return rows
```

**3f. Add the missing top-level imports `Rectangle` and `to_rgba`** to `bar.py`. Find the existing `from matplotlib.axes import Axes` import near the top and add directly after it:

```python
from matplotlib.patches import Rectangle
from matplotlib.colors import to_rgba
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/annotate/test_barplot_integration.py -v`
Expected: all tests pass.

Run: `uv run pytest tests/ -q`
Expected: full suite still passes.

- [ ] **Step 5: Commit**

```bash
git add src/publiplots/plot/bar.py tests/annotate/test_barplot_integration.py
git commit -m "$(cat <<'EOF'
feat(barplot): annotate= kwarg for in-plot value labels

pp.barplot(..., annotate=True | dict) attaches BarValueMeta to the axes
(pairing seaborn-drawn Rectangles with re-aggregated means and errorbar
extents from the same errorbar spec seaborn used) and calls pp.annotate.
owner_is_publiplots=True so labels expand ylim past seaborn's set_ylim.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Gallery example

**Files:**
- Create: `examples/plots/plot_15_annotate.py`

- [ ] **Step 1: Write the example**

Create `examples/plots/plot_15_annotate.py`:

```python
"""
Value Label Annotations
=======================

This example demonstrates pp.annotate, the helper for labeling plot
marks with their aggregated values. v1 supports barplots; the module is
designed so point_values, box_medians, and other strategies can slot in.

Two entry points:

- ``pp.annotate(ax, kind="bar_values", ...)`` — post-hoc on any axes
- ``pp.barplot(..., annotate=True | dict)`` — the sugar call
"""

import publiplots as pp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

# %%
# Simple: labels outside the bars (default)
# -----------------------------------------
df = pd.DataFrame({
    "category": pd.Categorical(["A", "B", "C", "D"]),
    "value": [1.2, 2.8, 2.1, 3.6],
})

fig, ax = pp.barplot(data=df, x="category", y="value", annotate=True)
ax.set_title("annotate=True")
plt.show()

# %%
# Custom format string
# --------------------
fig, ax = pp.barplot(
    data=df, x="category", y="value",
    annotate={"fmt": "{:.1f}%"},
)
ax.set_title("annotate={'fmt': '{:.1f}%'}")
plt.show()

# %%
# Anchor positions
# ----------------
fig, axes = plt.subplots(1, 4, figsize=(12, 3))
for ax, anchor in zip(axes, ("outside", "inside", "base", "center")):
    pp.barplot(data=df, x="category", y="value", ax=ax,
               annotate={"anchor": anchor})
    ax.set_title(f"anchor='{anchor}'")
plt.show()

# %%
# Hue-colored labels
# ------------------
rows = []
for group in ("A", "B", "C"):
    for cond in ("ctrl", "trt"):
        base = 2 if group == "A" else (3 if group == "B" else 4)
        bump = 0 if cond == "ctrl" else 0.8
        for v in rng.normal(loc=base + bump, scale=0.2, size=10):
            rows.append({"group": group, "cond": cond, "y": float(v)})
grouped = pd.DataFrame(rows)
grouped["group"] = grouped["group"].astype("category")
grouped["cond"] = grouped["cond"].astype("category")

fig, ax = pp.barplot(
    data=grouped, x="group", y="y", hue="cond", errorbar="se",
    annotate={"color": "hue"},
)
ax.set_title('annotate={"color": "hue"}')
plt.show()

# %%
# Horizontal orientation
# ----------------------
df_h = df.rename(columns={"category": "c", "value": "v"})
fig, ax = pp.barplot(data=df_h, x="v", y="c", annotate=True)
ax.set_title("horizontal")
plt.show()

# %%
# Post-hoc on a foreign axes
# --------------------------
# pp.annotate works on any Axes with bars on it, not just pp.barplot output.
fig, ax = plt.subplots()
ax.bar([0, 1, 2], [1.0, 2.4, 0.7])
pp.annotate(ax, kind="bar_values", fmt=".1f")
ax.set_title("foreign ax + pp.annotate")
plt.show()
```

- [ ] **Step 2: Verify the example runs without errors**

Run: `uv run python examples/plots/plot_15_annotate.py`
Expected: six matplotlib figures show (or are generated silently in Agg backend). No exceptions, no warnings about bars not found.

Because Agg is the default in tests but the example may be run in a non-interactive environment, a headless smoke test is sufficient:
```bash
uv run python -c "
import matplotlib; matplotlib.use('Agg')
import runpy
runpy.run_path('examples/plots/plot_15_annotate.py', run_name='__main__')
print('OK')
"
```
Expected output: `OK` with no tracebacks.

- [ ] **Step 3: Commit**

```bash
git add examples/plots/plot_15_annotate.py
git commit -m "$(cat <<'EOF'
docs(examples): plot_15_annotate gallery example

Demonstrates pp.annotate via pp.barplot(annotate=True) sugar and the
standalone pp.annotate(ax) primitive: default outside anchor, fmt
templates, all four anchors side-by-side, color='hue' with grouped
bars, horizontal orientation, and foreign-axes introspection path.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: README and CHANGELOG entry

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Check existing CHANGELOG style**

Run: `head -30 CHANGELOG.md`
Note the versioning convention (e.g., unreleased section, release tag format).

- [ ] **Step 2: Add the entry**

At the top of `CHANGELOG.md`, under the unreleased/next-version section (create one if missing), add an entry under a `### Added` subsection:

```markdown
- `pp.annotate(ax, kind="bar_values", ...)` for in-plot value labels on
  barplots. Orientation-aware, errorbar-aware, sign-aware; anchors
  `outside`/`inside`/`base`/`center`; contrast-aware coloring via
  `color="auto"` (with compositing for translucent fills) or palette
  colors via `color="hue"`. `pp.barplot(..., annotate=True | dict)`
  wires the primitive into the barplot API.
```

Do **not** bump the version number — the user handles releases.

- [ ] **Step 3: Run full suite one last time**

Run: `uv run pytest tests/ -q`
Expected: all previously passing tests + all new tests pass (the one pre-existing unrelated failure remains if it was present at baseline).

Run: `uv run pytest tests/annotate/ -v`
Expected: ~40+ annotate tests, all pass.

- [ ] **Step 4: Commit**

```bash
git add CHANGELOG.md
git commit -m "$(cat <<'EOF'
docs(changelog): pp.annotate() and barplot annotate= kwarg

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Post-implementation self-check

Before declaring the feature done, run through this checklist:

- [ ] `uv run pytest tests/ -q` passes (modulo the pre-existing unrelated failure)
- [ ] `uv run python -c "import publiplots as pp; help(pp.annotate)"` shows the docstring
- [ ] `uv run python examples/plots/plot_15_annotate.py` runs without error under Agg
- [ ] `git log --oneline <first-annotate-commit>..HEAD` shows 9 commits, one per task
- [ ] `tree src/publiplots/annotate` shows exactly 6 files: `__init__.py`, `_cache.py`, `_color.py`, `_dispatcher.py`, `_positioning.py`, `bar_values.py`
- [ ] `grep -r "TODO\|FIXME\|TBD" src/publiplots/annotate/` returns nothing
- [ ] `pp.barplot(..., annotate=True)` on a grouped+hue+errorbar dataset draws one label per bar, above the cap
- [ ] `pp.annotate(ax)` on a foreign `ax.bar(...)` axes draws labels via the introspection path
- [ ] Negative-bar smoke test: `pp.barplot(df_with_negatives, annotate=True)` shows labels below negative bars
