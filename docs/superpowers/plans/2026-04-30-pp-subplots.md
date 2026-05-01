# pp.subplots() Fixed-Axes Helper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `pp.subplots(nrows, ncols, axes_size=(w, h))` where `axes_size` in mm is inviolate and the figure grows to fit auto-measured decorations (titles, axis labels, legend column).

**Architecture:** Three units — (1) `FigureLayout` pure-geometry dataclass (mm math, no matplotlib); (2) `SubplotsAutoLayout` draw-event hook that measures `tightbbox` and resizes the figure; (3) `pp.subplots()` public API that wires them together. New `publiplots.rcParams` keys under `subplots.*` drive defaults per style.

**Tech Stack:** Python 3.11, matplotlib, numpy, pytest (Agg backend for draw tests).

**Worktree:** `.worktrees/pp-subplots` on branch `feat/pp-subplots`. Baseline: 67 tests passing.

**Spec:** `docs/superpowers/specs/2026-04-30-pp-subplots-design.md` (authoritative — re-read before starting).

---

## File structure

**Create:**
- `src/publiplots/layout/__init__.py` — re-exports `subplots`, `FigureLayout`, `SubplotsAutoLayout`
- `src/publiplots/layout/figure_layout.py` — `FigureLayout` dataclass (pure math)
- `src/publiplots/layout/auto_layout.py` — `SubplotsAutoLayout` draw-event hook
- `src/publiplots/layout/subplots.py` — `pp.subplots()` public API
- `tests/test_subplots.py` — all three layers' tests (split by section within one file)

**Modify:**
- `src/publiplots/__init__.py` — add `subplots` export
- `src/publiplots/themes/rcparams.py` — add 7 `subplots.*` keys (publication-grade defaults)
- `src/publiplots/themes/styles.py` — add notebook-grade overrides in `NOTEBOOK_STYLE`
- `examples/plots/plot_14_edgecolor_control.py` — migrate 2 subplot sections

---

## Rules of the road

- **TDD only.** Write the failing test first, verify it fails, then the minimum code to make it pass.
- **No placeholder comments.** If code changes, show the exact code.
- **All values in mm.** matplotlib's inches live only at `plt.figure(figsize=...)` and `fig.set_size_inches(...)` call sites. Everywhere else is mm. `MM2INCH = 1/25.4`.
- **Run the full suite (`uv run pytest tests/ -q`) after each task**, not just the one new test. Regressions in existing tests fail the task.
- **Commit after every task** with a conventional-commits message (`feat:`, `test:`, `docs:`, `refactor:`).
- **Do not skip the matplotlib backend header** in new test files:
  ```python
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  ```

---

## Task 1: Add `subplots.*` rcParams keys

**Files:**
- Modify: `src/publiplots/themes/rcparams.py:107-125` (extend `PUBLIPLOTS_RCPARAMS`)
- Modify: `src/publiplots/themes/styles.py:29-47` (extend `NOTEBOOK_STYLE` overrides)
- Test: `tests/test_subplots.py` (new file)

**Baseline defaults (publication-grade)** go in `PUBLIPLOTS_RCPARAMS`; notebook overrides go in `NOTEBOOK_STYLE`. This mirrors the existing pattern (e.g., `alpha=0.1` in `PUBLIPLOTS_RCPARAMS` with no override in `PUBLICATION_STYLE`).

| Key | `PUBLIPLOTS_RCPARAMS` (baseline = publication) | `NOTEBOOK_STYLE` override |
|---|---|---|
| `subplots.title_space` | 5 | 8 |
| `subplots.xlabel_space` | 8 | 12 |
| `subplots.ylabel_space` | 10 | 14 |
| `subplots.right` | 2 | 2 (no override needed) |
| `subplots.hspace` | 8 | 12 |
| `subplots.wspace` | 10 | 14 |
| `subplots.outer_pad` | 2 | 3 |

- [ ] **Step 1: Write the failing tests**

Create `tests/test_subplots.py`:

```python
"""Tests for pp.subplots() and its supporting components."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

import publiplots as pp


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# rcParams
# ---------------------------------------------------------------------------

SUBPLOT_KEYS = [
    "subplots.title_space",
    "subplots.xlabel_space",
    "subplots.ylabel_space",
    "subplots.right",
    "subplots.hspace",
    "subplots.wspace",
    "subplots.outer_pad",
]


def test_subplots_rcparams_keys_exist():
    for key in SUBPLOT_KEYS:
        assert key in pp.rcParams, f"missing rcParam: {key}"


def test_subplots_rcparams_publication_defaults():
    pp.set_publication_style()
    try:
        assert pp.rcParams["subplots.title_space"] == 5
        assert pp.rcParams["subplots.xlabel_space"] == 8
        assert pp.rcParams["subplots.ylabel_space"] == 10
        assert pp.rcParams["subplots.right"] == 2
        assert pp.rcParams["subplots.hspace"] == 8
        assert pp.rcParams["subplots.wspace"] == 10
        assert pp.rcParams["subplots.outer_pad"] == 2
    finally:
        pp.reset_style()


def test_subplots_rcparams_notebook_defaults():
    pp.set_notebook_style()
    try:
        assert pp.rcParams["subplots.title_space"] == 8
        assert pp.rcParams["subplots.xlabel_space"] == 12
        assert pp.rcParams["subplots.ylabel_space"] == 14
        assert pp.rcParams["subplots.right"] == 2
        assert pp.rcParams["subplots.hspace"] == 12
        assert pp.rcParams["subplots.wspace"] == 14
        assert pp.rcParams["subplots.outer_pad"] == 3
    finally:
        pp.reset_style()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_subplots.py -v`
Expected: 3 FAILED with `KeyError` or assertion failures on the new keys.

- [ ] **Step 3: Add keys to `PUBLIPLOTS_RCPARAMS`**

In `src/publiplots/themes/rcparams.py`, extend the `PUBLIPLOTS_RCPARAMS` dict (after line 124, before the closing `}`):

```python
PUBLIPLOTS_RCPARAMS: Dict[str, Any] = {
    # Color and transparency
    "color": "#5d83c3",  # Default blue
    "alpha": 0.1,  # Default transparency for bars
    "edgecolor": None,  # Global edge color for patches and marker outlines; None = each plot's default auto behavior

    # Error bars
    "capsize": 0.0,  # Error bar cap size

    # Color palettes
    "palette": "pastel",  # Default color palette

    # Hatch patterns
    "hatch_mode": 2,  # Default hatch density mode (2=medium)

    # Scatter plot sizes
    "scatter.size_min": 50,  # Minimum marker size for size mapping
    "scatter.size_max": 1000,  # Maximum marker size for size mapping

    # Subplots layout (mm) — baseline is publication-grade; notebook style
    # overrides these in themes/styles.py.
    "subplots.title_space": 5,    # reserved above each row
    "subplots.xlabel_space": 8,   # reserved below each row
    "subplots.ylabel_space": 10,  # reserved left of each col
    "subplots.right": 2,          # reserved right of each col
    "subplots.hspace": 8,         # vertical gap between rows
    "subplots.wspace": 10,        # horizontal gap between cols
    "subplots.outer_pad": 2,      # figure outer margin (all sides)
}
```

- [ ] **Step 4: Add notebook overrides in `NOTEBOOK_STYLE`**

In `src/publiplots/themes/styles.py`, extend `NOTEBOOK_STYLE` (after `"patch.linewidth": 2.0,` on line 46, before the closing `}`):

```python
NOTEBOOK_STYLE = {
    **MATPLOTLIB_RCPARAMS,
    **PUBLIPLOTS_RCPARAMS,
    # Overrides for notebook/interactive work
    "figure.figsize": [6.0, 4.0],
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "axes.linewidth": 1.0,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "lines.linewidth": 2.0,
    "lines.markersize": 6,
    "lines.markeredgewidth": 2.0,
    "patch.linewidth": 2.0,
    # Notebook overrides for subplot layout reservations (mm)
    "subplots.title_space": 8,
    "subplots.xlabel_space": 12,
    "subplots.ylabel_space": 14,
    "subplots.hspace": 12,
    "subplots.wspace": 14,
    "subplots.outer_pad": 3,
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_subplots.py -v`
Expected: 3 PASSED.

- [ ] **Step 6: Run the full suite**

Run: `uv run pytest tests/ -q`
Expected: 70 passed (67 baseline + 3 new).

- [ ] **Step 7: Commit**

```bash
git add src/publiplots/themes/rcparams.py src/publiplots/themes/styles.py tests/test_subplots.py
git commit -m "feat(rcparams): add subplots.* layout keys"
```

---

## Task 2: `FigureLayout` pure-geometry dataclass

**Files:**
- Create: `src/publiplots/layout/__init__.py`
- Create: `src/publiplots/layout/figure_layout.py`
- Test: `tests/test_subplots.py` (append)

Pure math, no matplotlib imports. Dataclass with `figure_size()`, `axes_position(row, col)`, and `with_updated_reservations(**overrides)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_subplots.py`:

```python
# ---------------------------------------------------------------------------
# FigureLayout — pure geometry
# ---------------------------------------------------------------------------
from publiplots.layout.figure_layout import FigureLayout


def _make_layout(nrows=1, ncols=1, **overrides):
    defaults = dict(
        nrows=nrows, ncols=ncols,
        axes_size=(50.0, 30.0),
        title_space=5.0, xlabel_space=8.0, ylabel_space=10.0, right=2.0,
        hspace=8.0, wspace=10.0, outer_pad=2.0, legend_column=0.0,
    )
    defaults.update(overrides)
    return FigureLayout(**defaults)


def test_figure_layout_single_cell_size():
    layout = _make_layout()
    W, H = layout.figure_size()
    # W = 2 + (10 + 50 + 2) + 0 + 2 = 66
    # H = 2 + (5 + 30 + 8) + 2 = 47
    assert W == pytest.approx(66.0)
    assert H == pytest.approx(47.0)


def test_figure_layout_2x3_size_matches_formula():
    layout = _make_layout(nrows=2, ncols=3)
    W, H = layout.figure_size()
    # W = 2 + 3*(10+50+2) + 2*10 + 0 + 2 = 2 + 186 + 20 + 2 = 210
    # H = 2 + 2*(5+30+8) + 1*8 + 2 = 2 + 86 + 8 + 2 = 98
    assert W == pytest.approx(210.0)
    assert H == pytest.approx(98.0)


def test_figure_layout_legend_column_adds_width_only():
    base = _make_layout(nrows=2, ncols=3)
    wide = _make_layout(nrows=2, ncols=3, legend_column=30.0)
    W0, H0 = base.figure_size()
    W1, H1 = wide.figure_size()
    assert W1 == pytest.approx(W0 + 30.0)
    assert H1 == pytest.approx(H0)


def test_figure_layout_axes_position_is_deterministic():
    layout = _make_layout(nrows=2, ncols=3)
    p_first = layout.axes_position(0, 0)
    p_again = layout.axes_position(0, 0)
    assert p_first == p_again


def test_figure_layout_axes_positions_dont_overlap():
    layout = _make_layout(nrows=2, ncols=3)
    rects = [layout.axes_position(r, c) for r in range(2) for c in range(3)]
    # Check pairwise non-overlap (rectangles as (x0, y0, w, h) in figure fractions)
    for i, (x0a, y0a, wa, ha) in enumerate(rects):
        for j, (x0b, y0b, wb, hb) in enumerate(rects):
            if i == j:
                continue
            x1a, y1a = x0a + wa, y0a + ha
            x1b, y1b = x0b + wb, y0b + hb
            overlaps = not (x1a <= x0b or x1b <= x0a or y1a <= y0b or y1b <= y0a)
            assert not overlaps, f"cells {i} and {j} overlap"


def test_figure_layout_with_updated_reservations_preserves_axes_size():
    layout = _make_layout()
    updated = layout.with_updated_reservations(title_space=20.0, xlabel_space=15.0)
    assert updated.axes_size == layout.axes_size
    assert updated.title_space == 20.0
    assert updated.xlabel_space == 15.0
    # Untouched fields unchanged
    assert updated.ylabel_space == layout.ylabel_space


def test_figure_layout_row_zero_is_top():
    """Row 0 should have the HIGHEST y0 in figure fractions (matplotlib convention)."""
    layout = _make_layout(nrows=3, ncols=1)
    y0_top = layout.axes_position(0, 0)[1]
    y0_mid = layout.axes_position(1, 0)[1]
    y0_bot = layout.axes_position(2, 0)[1]
    assert y0_top > y0_mid > y0_bot
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_subplots.py -v -k "figure_layout"`
Expected: 7 FAILED with `ImportError` / `ModuleNotFoundError: publiplots.layout.figure_layout`.

- [ ] **Step 3: Create `src/publiplots/layout/__init__.py`**

```python
"""publiplots layout engine — fixed-axes, flexible-canvas helpers."""

from publiplots.layout.figure_layout import FigureLayout

__all__ = ["FigureLayout"]
```

- [ ] **Step 4: Create `src/publiplots/layout/figure_layout.py`**

```python
"""
Pure-geometry layout for publiplots subplot grids.

FigureLayout computes figure size and axes positions from declared mm
dimensions. No matplotlib imports — this module is pure math and is
testable in isolation.
"""

from dataclasses import dataclass, replace
from typing import Tuple


@dataclass
class FigureLayout:
    """
    Millimeter-based grid geometry for a uniform subplot layout.

    Every cell has the same ``axes_size`` and the same per-side
    reservations. Gaps between cells are ``hspace`` (vertical) and
    ``wspace`` (horizontal). An optional ``legend_column`` reserves
    space on the far right of the whole figure, outside the grid.

    All values are in millimeters.

    Parameters
    ----------
    nrows, ncols : int
        Grid shape (>= 1).
    axes_size : tuple of (width, height) in mm
        The declared axes bbox for every cell. Inviolate after
        construction — never changes.
    title_space : float
        Space reserved above each row for titles.
    xlabel_space : float
        Space reserved below each row for x-axis labels and tick labels.
    ylabel_space : float
        Space reserved left of each column for y-axis labels and tick labels.
    right : float
        Space reserved right of each column (breathing room past the spine).
    hspace : float
        Vertical gap between rows.
    wspace : float
        Horizontal gap between columns.
    outer_pad : float
        Figure outer margin (same on all four sides).
    legend_column : float
        Extra width reserved on the far right of the figure for a
        legend_group anchored outside the grid. Never auto-measured.
    """

    nrows: int
    ncols: int
    axes_size: Tuple[float, float]
    title_space: float
    xlabel_space: float
    ylabel_space: float
    right: float
    hspace: float
    wspace: float
    outer_pad: float
    legend_column: float

    def figure_size(self) -> Tuple[float, float]:
        """Total figure size in mm as (width, height)."""
        w_ax, h_ax = self.axes_size
        W = (
            self.outer_pad
            + self.ncols * (self.ylabel_space + w_ax + self.right)
            + max(self.ncols - 1, 0) * self.wspace
            + self.legend_column
            + self.outer_pad
        )
        H = (
            self.outer_pad
            + self.nrows * (self.title_space + h_ax + self.xlabel_space)
            + max(self.nrows - 1, 0) * self.hspace
            + self.outer_pad
        )
        return W, H

    def axes_position(self, row: int, col: int) -> Tuple[float, float, float, float]:
        """
        Figure-fraction (x0, y0, w, h) of the axes at (row, col).

        Row 0 is the top row. y is measured from the bottom, matching
        matplotlib's convention.
        """
        W, H = self.figure_size()
        w_ax, h_ax = self.axes_size
        x0_mm = (
            self.outer_pad
            + col * (self.ylabel_space + w_ax + self.right + self.wspace)
            + self.ylabel_space
        )
        rows_below = self.nrows - 1 - row
        y0_mm = (
            self.outer_pad
            + rows_below * (self.title_space + h_ax + self.xlabel_space + self.hspace)
            + self.xlabel_space
        )
        return (x0_mm / W, y0_mm / H, w_ax / W, h_ax / H)

    def with_updated_reservations(self, **overrides) -> "FigureLayout":
        """Return a copy with the given fields updated. ``axes_size`` must not change."""
        if "axes_size" in overrides:
            raise ValueError("axes_size is inviolate; cannot be overridden")
        return replace(self, **overrides)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_subplots.py -v -k "figure_layout"`
Expected: 7 PASSED.

- [ ] **Step 6: Run the full suite**

Run: `uv run pytest tests/ -q`
Expected: 77 passed.

- [ ] **Step 7: Commit**

```bash
git add src/publiplots/layout/__init__.py src/publiplots/layout/figure_layout.py tests/test_subplots.py
git commit -m "feat(layout): FigureLayout pure-geometry dataclass"
```

---

## Task 3: `SubplotsAutoLayout` draw-event hook

**Files:**
- Create: `src/publiplots/layout/auto_layout.py`
- Modify: `src/publiplots/layout/__init__.py` (add export)
- Test: `tests/test_subplots.py` (append)

Measures decoration sizes on every `draw_event`, resizes the figure, repositions all axes. Re-entrance guarded. Skips updates smaller than 0.1 mm.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_subplots.py`:

```python
# ---------------------------------------------------------------------------
# SubplotsAutoLayout — draw-event hook
# ---------------------------------------------------------------------------
from publiplots.layout.auto_layout import SubplotsAutoLayout

MM2INCH = 1 / 25.4


def _make_fig_with_layout(layout, ncells=None):
    """Build a matplotlib figure with axes placed per the layout. Returns (fig, axes_matrix)."""
    W, H = layout.figure_size()
    fig = plt.figure(figsize=(W * MM2INCH, H * MM2INCH), layout=None)
    axes = []
    for r in range(layout.nrows):
        row = []
        for c in range(layout.ncols):
            ax = fig.add_axes(layout.axes_position(r, c))
            row.append(ax)
        axes.append(row)
    return fig, axes


def test_auto_layout_resizes_figure_for_title():
    layout = _make_layout(nrows=1, ncols=1, title_space=1.0)  # deliberately too small
    fig, axes = _make_fig_with_layout(layout)
    ax = axes[0][0]
    ax.set_title("A title that needs more vertical room than 1 mm")
    reactor = SubplotsAutoLayout(fig, layout, locked=set())
    initial_h_mm = fig.get_figheight() / MM2INCH
    fig.canvas.draw()
    final_h_mm = fig.get_figheight() / MM2INCH
    assert final_h_mm > initial_h_mm, (
        f"figure should have grown; initial {initial_h_mm:.2f} mm, final {final_h_mm:.2f} mm"
    )


def test_auto_layout_preserves_axes_size_after_resize():
    declared_w, declared_h = 50.0, 30.0
    layout = _make_layout(axes_size=(declared_w, declared_h))
    fig, axes = _make_fig_with_layout(layout)
    ax = axes[0][0]
    ax.set_title("A title")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    SubplotsAutoLayout(fig, layout, locked=set())
    fig.canvas.draw()

    # Actual axes bbox in mm
    pos = ax.get_position()
    fig_w_in, fig_h_in = fig.get_size_inches()
    ax_w_mm = pos.width * fig_w_in / MM2INCH
    ax_h_mm = pos.height * fig_h_in / MM2INCH
    assert ax_w_mm == pytest.approx(declared_w, abs=0.5)
    assert ax_h_mm == pytest.approx(declared_h, abs=0.5)


def test_auto_layout_locked_side_not_remeasured():
    layout = _make_layout(title_space=25.0)  # locked oversize reservation
    fig, axes = _make_fig_with_layout(layout)
    axes[0][0].set_title("tiny")  # way less than 25 mm
    initial_h_mm = fig.get_figheight() / MM2INCH
    SubplotsAutoLayout(fig, layout, locked={"title_space"})
    fig.canvas.draw()
    final_h_mm = fig.get_figheight() / MM2INCH
    # Locked → height should not shrink (only allow growth from auto-measured sides)
    assert final_h_mm >= initial_h_mm - 0.5, (
        f"locked side should not shrink figure; {initial_h_mm:.2f} -> {final_h_mm:.2f} mm"
    )


def test_auto_layout_no_hook_when_all_sides_locked():
    layout = _make_layout()
    fig, _ = _make_fig_with_layout(layout)
    all_sides = {"title_space", "xlabel_space", "ylabel_space", "right"}
    reactor = SubplotsAutoLayout(fig, layout, locked=all_sides)
    # No draw-event callback should be connected
    assert reactor._cid is None


def test_auto_layout_second_draw_no_change_within_threshold():
    layout = _make_layout()
    fig, axes = _make_fig_with_layout(layout)
    axes[0][0].set_title("stable")
    SubplotsAutoLayout(fig, layout, locked=set())
    fig.canvas.draw()
    size_after_first = fig.get_size_inches().copy()
    fig.canvas.draw()
    size_after_second = fig.get_size_inches()
    assert np.allclose(size_after_first, size_after_second, atol=1e-4)


def test_auto_layout_attaches_layout_to_figure():
    layout = _make_layout()
    fig, _ = _make_fig_with_layout(layout)
    SubplotsAutoLayout(fig, layout, locked=set())
    assert getattr(fig, "_publiplots_layout", None) is layout
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_subplots.py -v -k "auto_layout"`
Expected: 6 FAILED with `ModuleNotFoundError: publiplots.layout.auto_layout`.

- [ ] **Step 3: Create `src/publiplots/layout/auto_layout.py`**

```python
"""
Draw-event hook that keeps declared axes sizes fixed while the figure
grows to fit auto-measured decorations (titles, axis labels, tick labels).

Only the four per-cell reservations are auto-measured:
title_space, xlabel_space, ylabel_space, right.

Gaps (hspace, wspace) and outer_pad are never remeasured — users set
them explicitly or inherit rcParams defaults.

Cooperates with LayoutReactor (utils/layout_reactor.py): both react to
draw_event, but SubplotsAutoLayout is registered first (during
pp.subplots()) and therefore fires first, so LayoutReactor sees the
repositioned axes and re-anchors legends correctly.
"""

from typing import Set

from publiplots.layout.figure_layout import FigureLayout


_MM2INCH = 1 / 25.4
_UPDATE_THRESHOLD_MM = 0.1
_ALL_SIDES = {"title_space", "xlabel_space", "ylabel_space", "right"}


class SubplotsAutoLayout:
    """Per-figure draw-event listener that resizes the figure to fit decorations."""

    def __init__(self, fig, layout: FigureLayout, locked: Set[str]):
        self._fig = fig
        self._layout = layout
        self._locked = set(locked)
        self._updating = False

        # Expose the layout on the figure for introspection and the
        # future composer PR. Public-ish but underscore-prefixed until
        # the composer API ships.
        fig._publiplots_layout = layout
        fig._publiplots_auto_layout = self

        # If every auto-measurable side is locked, no hook is needed —
        # figure size is fully deterministic.
        if _ALL_SIDES.issubset(self._locked):
            self._cid = None
        else:
            self._cid = fig.canvas.mpl_connect("draw_event", self._on_draw)

    def _on_draw(self, event):
        if self._updating:
            return
        self._updating = True
        try:
            new = self._measure()
            if self._needs_update(new):
                self._apply(new)
        finally:
            self._updating = False

    def _measure(self) -> dict:
        """Measure max decoration size per side across all axes, in mm."""
        fig = self._fig
        dpi = fig.dpi
        if dpi <= 0:
            return {}
        # Collect axes in grid order
        axes_matrix = self._axes_matrix()
        measured: dict = {}

        def _mm(px: float) -> float:
            return px / dpi * 25.4

        if "title_space" not in self._locked:
            max_top_px = 0.0
            for row in axes_matrix:
                for ax in row:
                    ax_bbox = ax.get_window_extent()
                    tight = ax.get_tightbbox()
                    if tight is None:
                        continue
                    max_top_px = max(max_top_px, tight.y1 - ax_bbox.y1)
            measured["title_space"] = max(_mm(max_top_px), 0.0)

        if "xlabel_space" not in self._locked:
            max_bot_px = 0.0
            for row in axes_matrix:
                for ax in row:
                    ax_bbox = ax.get_window_extent()
                    tight = ax.get_tightbbox()
                    if tight is None:
                        continue
                    max_bot_px = max(max_bot_px, ax_bbox.y0 - tight.y0)
            measured["xlabel_space"] = max(_mm(max_bot_px), 0.0)

        if "ylabel_space" not in self._locked:
            max_left_px = 0.0
            for row in axes_matrix:
                for ax in row:
                    ax_bbox = ax.get_window_extent()
                    tight = ax.get_tightbbox()
                    if tight is None:
                        continue
                    max_left_px = max(max_left_px, ax_bbox.x0 - tight.x0)
            measured["ylabel_space"] = max(_mm(max_left_px), 0.0)

        if "right" not in self._locked:
            max_right_px = 0.0
            for row in axes_matrix:
                for ax in row:
                    ax_bbox = ax.get_window_extent()
                    tight = ax.get_tightbbox()
                    if tight is None:
                        continue
                    max_right_px = max(max_right_px, tight.x1 - ax_bbox.x1)
            measured["right"] = max(_mm(max_right_px), 0.0)

        return measured

    def _needs_update(self, measured: dict) -> bool:
        for side, new_val in measured.items():
            current = getattr(self._layout, side)
            if abs(new_val - current) >= _UPDATE_THRESHOLD_MM:
                return True
        return False

    def _apply(self, measured: dict) -> None:
        new_layout = self._layout.with_updated_reservations(**measured)
        self._layout = new_layout
        self._fig._publiplots_layout = new_layout

        W, H = new_layout.figure_size()
        self._fig.set_size_inches(W * _MM2INCH, H * _MM2INCH, forward=False)

        for r, row in enumerate(self._axes_matrix()):
            for c, ax in enumerate(row):
                ax.set_position(new_layout.axes_position(r, c))

    def _axes_matrix(self):
        """
        Return the axes in row-major order.

        Stored by pp.subplots() as ``fig._publiplots_axes`` (list of lists).
        For figures built manually in tests, this falls back to walking
        ``fig.axes`` in insertion order.
        """
        stored = getattr(self._fig, "_publiplots_axes", None)
        if stored is not None:
            return stored
        # Fallback: reshape fig.axes to (nrows, ncols)
        flat = list(self._fig.axes)
        nrows, ncols = self._layout.nrows, self._layout.ncols
        if len(flat) < nrows * ncols:
            return [[]]
        return [flat[r * ncols:(r + 1) * ncols] for r in range(nrows)]
```

- [ ] **Step 4: Re-export from `src/publiplots/layout/__init__.py`**

Replace contents with:

```python
"""publiplots layout engine — fixed-axes, flexible-canvas helpers."""

from publiplots.layout.figure_layout import FigureLayout
from publiplots.layout.auto_layout import SubplotsAutoLayout

__all__ = ["FigureLayout", "SubplotsAutoLayout"]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_subplots.py -v -k "auto_layout"`
Expected: 6 PASSED.

- [ ] **Step 6: Run the full suite**

Run: `uv run pytest tests/ -q`
Expected: 83 passed.

- [ ] **Step 7: Commit**

```bash
git add src/publiplots/layout/auto_layout.py src/publiplots/layout/__init__.py tests/test_subplots.py
git commit -m "feat(layout): SubplotsAutoLayout draw-event hook"
```

---

## Task 4: `pp.subplots()` public API

**Files:**
- Create: `src/publiplots/layout/subplots.py`
- Modify: `src/publiplots/layout/__init__.py` (add export)
- Modify: `src/publiplots/__init__.py` (add top-level export)
- Test: `tests/test_subplots.py` (append)

Wires `FigureLayout` + `SubplotsAutoLayout` behind a function that looks like `plt.subplots` but takes `axes_size` (mm) and per-side reservations.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_subplots.py`:

```python
# ---------------------------------------------------------------------------
# pp.subplots() — public API
# ---------------------------------------------------------------------------
from matplotlib.axes import Axes as _Axes
from matplotlib.figure import Figure as _Figure


def test_subplots_scalar_axes_size_coerced_to_tuple():
    fig, ax = pp.subplots(axes_size=30)
    assert fig._publiplots_layout.axes_size == (30.0, 30.0)


def test_subplots_rejects_figsize_kwarg():
    with pytest.raises(TypeError, match="axes_size"):
        pp.subplots(axes_size=(50, 30), figsize=(5, 3))


def test_subplots_warns_on_layout_engine_kwarg():
    with pytest.warns(UserWarning, match="publiplots manages layout"):
        fig, ax = pp.subplots(axes_size=(50, 30), constrained_layout=True)
    assert fig.get_layout_engine() is None


def test_subplots_disables_layout_engine():
    fig, ax = pp.subplots(axes_size=(50, 30))
    assert fig.get_layout_engine() is None


def test_subplots_squeeze_returns_scalar_for_1x1():
    fig, ax = pp.subplots(axes_size=(50, 30))
    assert isinstance(ax, _Axes)


def test_subplots_returns_1d_array_for_single_row():
    fig, axes = pp.subplots(1, 3, axes_size=(50, 30))
    assert axes.shape == (3,)


def test_subplots_returns_1d_array_for_single_col():
    fig, axes = pp.subplots(3, 1, axes_size=(50, 30))
    assert axes.shape == (3,)


def test_subplots_returns_2d_array_for_grid():
    fig, axes = pp.subplots(2, 3, axes_size=(50, 30))
    assert axes.shape == (2, 3)


def test_subplots_attaches_figure_layout_to_fig():
    fig, _ = pp.subplots(2, 3, axes_size=(50, 30))
    from publiplots.layout.figure_layout import FigureLayout
    assert isinstance(fig._publiplots_layout, FigureLayout)
    assert fig._publiplots_layout.nrows == 2
    assert fig._publiplots_layout.ncols == 3


def test_subplots_validates_nrows():
    with pytest.raises(ValueError, match="nrows"):
        pp.subplots(nrows=0, ncols=1, axes_size=(50, 30))


def test_subplots_validates_ncols():
    with pytest.raises(ValueError, match="ncols"):
        pp.subplots(nrows=1, ncols=0, axes_size=(50, 30))


def test_subplots_validates_axes_size_scalar():
    with pytest.raises(ValueError, match="axes_size"):
        pp.subplots(axes_size=-5)


def test_subplots_validates_axes_size_tuple():
    with pytest.raises(ValueError, match="axes_size"):
        pp.subplots(axes_size=(50, 0))


def test_subplots_validates_negative_reservation():
    with pytest.raises(ValueError, match="title_space"):
        pp.subplots(axes_size=(50, 30), title_space=-1.0)


def test_subplots_legend_column_reserves_extra_width():
    fig_no_col, _ = pp.subplots(axes_size=(50, 30), legend_column=0,
                                title_space=5, xlabel_space=8,
                                ylabel_space=10, right=2,
                                hspace=8, wspace=10, outer_pad=2)
    fig_with_col, _ = pp.subplots(axes_size=(50, 30), legend_column=30,
                                  title_space=5, xlabel_space=8,
                                  ylabel_space=10, right=2,
                                  hspace=8, wspace=10, outer_pad=2)
    w_no = fig_no_col.get_figwidth()
    w_with = fig_with_col.get_figwidth()
    assert w_with == pytest.approx(w_no + 30 * MM2INCH, abs=1e-3)


def test_subplots_sharex_true_shares_all():
    fig, axes = pp.subplots(2, 3, axes_size=(50, 30), sharex=True)
    # sharex=True -> every axes shares with (0,0)
    shared = axes[0, 0].get_shared_x_axes()
    for r in range(2):
        for c in range(3):
            assert shared.joined(axes[0, 0], axes[r, c])


def test_subplots_sharex_row_shares_within_row_only():
    fig, axes = pp.subplots(2, 3, axes_size=(50, 30), sharex="row")
    shared_top = axes[0, 0].get_shared_x_axes()
    # Same row is shared
    assert shared_top.joined(axes[0, 0], axes[0, 2])
    # Different rows are NOT shared
    assert not shared_top.joined(axes[0, 0], axes[1, 0])


def test_subplots_all_locked_skips_hook():
    fig, ax = pp.subplots(
        axes_size=(50, 30),
        title_space=5, xlabel_space=8, ylabel_space=10, right=2,
    )
    assert fig._publiplots_auto_layout._cid is None


def test_subplots_any_auto_side_attaches_hook():
    fig, ax = pp.subplots(
        axes_size=(50, 30),
        title_space=5, xlabel_space=8, ylabel_space=10,
        # right left as auto
    )
    assert fig._publiplots_auto_layout._cid is not None


def test_subplots_works_with_legend_builder_after_auto_resize():
    """pp.legend() on a pp.subplots axes should follow the auto-layout resize."""
    from publiplots.utils.legend import create_legend_handles
    fig, ax = pp.subplots(axes_size=(60, 40))
    ax.set_title("A")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    handles = create_legend_handles(labels=["A"], colors=["#5d83c3"],
                                    alpha=0.2, linewidth=1.0)
    builder = pp.legend(ax, auto=False)
    builder.add_legend(handles=handles, label="group")
    fig.canvas.draw()
    # Simple sanity: axes bbox in mm matches declared size within tolerance
    pos = ax.get_position()
    fig_w_in, fig_h_in = fig.get_size_inches()
    ax_w_mm = pos.width * fig_w_in / MM2INCH
    ax_h_mm = pos.height * fig_h_in / MM2INCH
    assert ax_w_mm == pytest.approx(60.0, abs=0.5)
    assert ax_h_mm == pytest.approx(40.0, abs=0.5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_subplots.py -v -k "test_subplots_ or test_subplots.rejects or test_subplots.disables or test_subplots.returns or test_subplots.squeeze or test_subplots.attaches or test_subplots.validates or test_subplots.legend_column or test_subplots.sharex or test_subplots.all_locked or test_subplots.any_auto or test_subplots.works"`

Simpler: `uv run pytest tests/test_subplots.py -v` — expect the 19 new tests to FAIL with `AttributeError: module 'publiplots' has no attribute 'subplots'`.

- [ ] **Step 3: Create `src/publiplots/layout/subplots.py`**

```python
"""
Public API: pp.subplots() — fixed-axes, flexible-canvas subplot factory.

Declared axes dimensions (in mm) are inviolate; the figure grows to
accommodate auto-measured decorations. Follow-up PR(s) will add
legend-width awareness and a Composer for cross-figure page layout.
"""

import warnings
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from publiplots.themes.rcparams import resolve_param
from publiplots.layout.figure_layout import FigureLayout
from publiplots.layout.auto_layout import SubplotsAutoLayout


_MM2INCH = 1 / 25.4
_RESERVATION_KEYS = (
    "title_space", "xlabel_space", "ylabel_space", "right",
    "hspace", "wspace", "outer_pad",
)
_LAYOUT_ENGINE_KWARGS = ("layout", "constrained_layout", "tight_layout")


def subplots(
    nrows: int = 1,
    ncols: int = 1,
    *,
    axes_size: Union[Tuple[float, float], float],
    sharex: Union[bool, str] = False,
    sharey: Union[bool, str] = False,
    title_space: Optional[float] = None,
    xlabel_space: Optional[float] = None,
    ylabel_space: Optional[float] = None,
    right: Optional[float] = None,
    hspace: Optional[float] = None,
    wspace: Optional[float] = None,
    outer_pad: Optional[float] = None,
    legend_column: float = 0.0,
    **fig_kw,
):
    """
    Create a figure and a grid of axes with deterministic axes dimensions.

    Every axes in the grid has exactly ``axes_size`` mm as its spine
    bounding box. The figure size is computed to accommodate decorations
    (titles, axis labels, tick labels) which are auto-measured on first
    draw. Any per-side reservation passed explicitly is locked and never
    remeasured.

    Parameters
    ----------
    nrows, ncols : int, default 1
        Grid shape (must be >= 1).
    axes_size : (float, float) or float, in mm
        Declared axes bbox. Scalar is coerced to ``(s, s)``.
    sharex, sharey : bool or {"all", "row", "col", "none"}
        Axis-sharing semantics, matching ``plt.subplots``.
    title_space, xlabel_space, ylabel_space, right : float, optional
        Per-cell reservations in mm. ``None`` means: initial value from
        rcParams, then auto-measured on first draw. A float locks the
        value.
    hspace, wspace, outer_pad : float, optional
        Gaps and outer margin in mm. ``None`` falls back to rcParams.
        Never auto-measured.
    legend_column : float, default 0
        Extra width reserved on the far right (e.g., for
        ``pp.legend_group``). Never auto-measured — opt-in only.
    **fig_kw
        Forwarded to ``plt.figure``. ``figsize`` is rejected; layout-
        engine kwargs are ignored with a warning.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : matplotlib.axes.Axes or numpy.ndarray of Axes
        Shape matches ``plt.subplots(squeeze=True)``.
    """
    # --- validation ---------------------------------------------------------
    if nrows < 1:
        raise ValueError(f"nrows must be >= 1, got {nrows}")
    if ncols < 1:
        raise ValueError(f"ncols must be >= 1, got {ncols}")

    if isinstance(axes_size, (int, float)):
        if axes_size <= 0:
            raise ValueError(f"axes_size must be positive, got {axes_size}")
        axes_size_t: Tuple[float, float] = (float(axes_size), float(axes_size))
    else:
        try:
            w_ax, h_ax = axes_size
        except (TypeError, ValueError):
            raise ValueError(
                f"axes_size must be a positive scalar or (width, height) tuple, got {axes_size!r}"
            )
        if w_ax <= 0 or h_ax <= 0:
            raise ValueError(f"axes_size must be positive, got {axes_size}")
        axes_size_t = (float(w_ax), float(h_ax))

    if "figsize" in fig_kw:
        raise TypeError(
            "pp.subplots() does not accept figsize; use axes_size (mm). "
            "Figure size is computed from axes_size + reservations."
        )
    for k in _LAYOUT_ENGINE_KWARGS:
        if k in fig_kw:
            warnings.warn(
                f"publiplots manages layout; ignoring {k}={fig_kw[k]!r}",
                UserWarning,
                stacklevel=2,
            )
            fig_kw.pop(k)

    # --- resolve reservation defaults & track locked sides ----------------
    user_values = dict(
        title_space=title_space, xlabel_space=xlabel_space,
        ylabel_space=ylabel_space, right=right,
        hspace=hspace, wspace=wspace, outer_pad=outer_pad,
    )
    locked = {k for k, v in user_values.items() if v is not None}
    resolved = {}
    for k, v in user_values.items():
        val = resolve_param(f"subplots.{k}", v)
        if val < 0:
            raise ValueError(f"{k} must be non-negative, got {val}")
        resolved[k] = float(val)

    if legend_column < 0:
        raise ValueError(f"legend_column must be non-negative, got {legend_column}")

    # --- build layout & figure --------------------------------------------
    layout = FigureLayout(
        nrows=nrows, ncols=ncols,
        axes_size=axes_size_t,
        legend_column=float(legend_column),
        **resolved,
    )
    W, H = layout.figure_size()
    fig = plt.figure(figsize=(W * _MM2INCH, H * _MM2INCH), layout=None, **fig_kw)

    # --- create axes with sharing semantics -------------------------------
    axes_matrix = _build_axes(fig, layout, sharex, sharey)
    fig._publiplots_axes = axes_matrix

    # --- attach auto-layout hook (skipped internally if all sides locked) -
    # Only lock the auto-measurable sides; hspace/wspace/outer_pad are not
    # auto-measured regardless.
    auto_locked = locked & {"title_space", "xlabel_space", "ylabel_space", "right"}
    # If every auto-measurable side is user-locked, SubplotsAutoLayout
    # skips the draw-event connection (see auto_layout.py).
    SubplotsAutoLayout(fig, layout, locked=auto_locked)

    # --- squeeze & return --------------------------------------------------
    arr = np.empty((nrows, ncols), dtype=object)
    for r in range(nrows):
        for c in range(ncols):
            arr[r, c] = axes_matrix[r][c]
    return fig, _squeeze(arr, nrows, ncols)


def _build_axes(fig, layout, sharex, sharey):
    """Create axes in row-major order with plt.subplots-style sharing."""
    matrix = [[None] * layout.ncols for _ in range(layout.nrows)]
    for r in range(layout.nrows):
        for c in range(layout.ncols):
            share_x = _resolve_shared(sharex, matrix, r, c, axis="x")
            share_y = _resolve_shared(sharey, matrix, r, c, axis="y")
            kwargs = {}
            if share_x is not None:
                kwargs["sharex"] = share_x
            if share_y is not None:
                kwargs["sharey"] = share_y
            ax = fig.add_axes(layout.axes_position(r, c), **kwargs)
            matrix[r][c] = ax
    return matrix


def _resolve_shared(share, matrix, r, c, axis):
    """Return the axes to share with, or None."""
    if share in (False, "none"):
        return None
    if r == 0 and c == 0:
        return None
    if share in (True, "all"):
        return matrix[0][0]
    if share == "row":
        return matrix[r][0] if c > 0 else None
    if share == "col":
        return matrix[0][c] if r > 0 else None
    raise ValueError(f"share{axis} must be bool or one of 'all'/'row'/'col'/'none', got {share!r}")


def _squeeze(arr, nrows, ncols):
    if nrows == 1 and ncols == 1:
        return arr[0, 0]
    if nrows == 1:
        return arr[0, :]
    if ncols == 1:
        return arr[:, 0]
    return arr
```

- [ ] **Step 4: Extend `src/publiplots/layout/__init__.py`**

Replace with:

```python
"""publiplots layout engine — fixed-axes, flexible-canvas helpers."""

from publiplots.layout.figure_layout import FigureLayout
from publiplots.layout.auto_layout import SubplotsAutoLayout
from publiplots.layout.subplots import subplots

__all__ = ["FigureLayout", "SubplotsAutoLayout", "subplots"]
```

- [ ] **Step 5: Wire top-level export**

In `src/publiplots/__init__.py`, after the existing `from publiplots.utils.legend_group import ...` line (near the `MultiAxesLegendGroup` import), add:

```python
from publiplots.layout import subplots
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_subplots.py -v`
Expected: all 36 tests PASSED (3 rcParams + 7 FigureLayout + 6 auto-layout + 19 subplots + 1 legend-builder integration = match the counter — adjust if off by one after running).

- [ ] **Step 7: Run the full suite**

Run: `uv run pytest tests/ -q`
Expected: baseline 67 + all the new tests. No regressions in existing tests.

- [ ] **Step 8: Commit**

```bash
git add src/publiplots/layout/subplots.py src/publiplots/layout/__init__.py src/publiplots/__init__.py tests/test_subplots.py
git commit -m "feat(layout): pp.subplots() public API"
```

---

## Task 5: Migrate `plot_14_edgecolor_control.py` to `pp.subplots()`

**Files:**
- Modify: `examples/plots/plot_14_edgecolor_control.py` (two sections)

The existing file uses `plt.subplots + fig.subplots_adjust` in two places. Replace with `pp.subplots(legend_column=...)`. The `pp.legend_group(anchor=axes[-1])` lines stay unchanged.

- [ ] **Step 1: Read the current `plot_14` sections**

Open `examples/plots/plot_14_edgecolor_control.py`. The two sections to migrate are:
- "Uniform Edges Across Plot Types" (1×3, lines ~94-126 currently)
- "Composite Plots: Raincloud" (1×2, lines ~142-195 currently)

- [ ] **Step 2: Migrate the 1×3 section**

Replace:

```python
# Wider figure with subplots_adjust to reserve ~12% on the right for the
# unified legend column. publiplots leaves the matplotlib layout engine off
# by default, so manual subplots_adjust works as expected.
fig, axes = plt.subplots(1, 3, figsize=(14, 3))
fig.subplots_adjust(left=0.05, right=0.88, wspace=0.25)
```

with:

```python
# pp.subplots declares axes dimensions in mm and extends the figure to
# accommodate decorations. legend_column=30 reserves a 30 mm strip on
# the right for the unified legend; pp.legend_group lands in that strip.
fig, axes = pp.subplots(1, 3, axes_size=(45, 30), legend_column=30)
```

- [ ] **Step 3: Migrate the 1×2 raincloud section**

Replace:

```python
fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
fig.subplots_adjust(left=0.06, right=0.85, wspace=0.1)
```

with:

```python
fig, axes = pp.subplots(1, 2, axes_size=(55, 50), sharey=True, legend_column=30)
```

- [ ] **Step 4: Regenerate gallery locally**

Run: `uv run sphinx-build -b html docs/source docs/_build/html 2>&1 | tail -30`
Expected: no errors, examples build successfully. (If the docs command differs, check `docs/Makefile` or `.readthedocs.yaml` for the exact invocation — the worktree shares config with main.)

- [ ] **Step 5: Run the full suite to ensure the example didn't break anything**

Run: `uv run pytest tests/ -q`
Expected: still all passing.

- [ ] **Step 6: Commit**

```bash
git add examples/plots/plot_14_edgecolor_control.py
git commit -m "docs(gallery): migrate plot_14 subplot sections to pp.subplots"
```

---

## Task 6: Final verification & PR prep

**Files:** none modified; verification only.

- [ ] **Step 1: Run the full test suite one more time**

Run: `uv run pytest tests/ -q`
Expected: clean pass.

- [ ] **Step 2: Review `git log --oneline` on the branch**

Run: `git log --oneline origin/main..HEAD`
Expected: 6-7 commits in logical order:
1. `docs(spec): pp.subplots() fixed-axes helper design (PR #79)` (already on branch)
2. `feat(rcparams): add subplots.* layout keys`
3. `feat(layout): FigureLayout pure-geometry dataclass`
4. `feat(layout): SubplotsAutoLayout draw-event hook`
5. `feat(layout): pp.subplots() public API`
6. `docs(gallery): migrate plot_14 subplot sections to pp.subplots`

- [ ] **Step 3: Push the branch**

Run: `git push -u origin feat/pp-subplots`

- [ ] **Step 4: Open a PR via `gh pr create`**

The PR description should cite the spec (`docs/superpowers/specs/2026-04-30-pp-subplots-design.md`), list the new public surface (`pp.subplots`), and explicitly note the out-of-scope items deferred to follow-up PRs (legend-width awareness, `plot_15_legends.py`, Composer).

---

## Self-review (pre-execution check)

**Spec coverage:**
- Goal ("axes_size inviolate, figure grows for decorations") → covered by Task 4 (API) + Task 3 (auto-layout measurement).
- `FigureLayout` pure math → Task 2.
- `SubplotsAutoLayout` draw-event hook → Task 3.
- `pp.subplots()` signature + validation + sharex/sharey → Task 4.
- rcParams `subplots.*` keys + notebook/publication defaults → Task 1.
- Exports → Task 4 (steps 4-5).
- Gallery migration of plot_14 → Task 5.
- All tests listed in spec (FigureLayout, API, SubplotsAutoLayout) → Tasks 2/3/4.
- Out-of-scope (`pp.figure()`, GridSpec, legend tutorial, collision warning, `plot_15`) → correctly absent from every task.

**Placeholder scan:** No TBD / TODO / "similar to" references. Every step has exact code or exact commands.

**Type consistency:**
- `FigureLayout` field names (`title_space`, `xlabel_space`, `ylabel_space`, `right`, `hspace`, `wspace`, `outer_pad`, `legend_column`) match between Task 2 (definition), Task 3 (measurement dict keys), Task 4 (API kwargs), and Task 5 (gallery call-site). Consistent.
- `SubplotsAutoLayout(fig, layout, locked: set[str])` signature identical across Task 3 (definition) and Task 4 (call site).
- `fig._publiplots_layout`, `fig._publiplots_axes`, `fig._publiplots_auto_layout` attribute names consistent across Tasks 3 & 4.
- `_MM2INCH = 1 / 25.4` used identically in auto_layout and subplots.

All checks pass — plan is ready for execution.
