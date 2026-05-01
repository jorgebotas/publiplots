# Per-Cell Reservations Implementation Plan (PR #79 addendum)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans.

**Goal:** Make `SubplotsAutoLayout` measure reservations per-row and per-column instead of grid-wide, and exclude `LayoutReactor`-managed external overlays (`legend_group`, external colorbars) from the tightbbox measurement — while keeping per-axis legends visible so their width reflects in the column's `right`.

**Architecture:** Tuple-valued fields on `FigureLayout` for the 4 per-side reservations (length `nrows` for row-axis sides, length `ncols` for col-axis sides). Scalar-to-tuple broadcast in `pp.subplots()`. `SubplotsAutoLayout._measure()` iterates rows/cols and builds a tuple per side. Legend-group registrations gain an `external_to_axis=True` flag consulted during tightbbox filtering.

**Tech Stack:** Python 3.11, matplotlib, numpy, pytest (Agg backend).

**Worktree:** `.worktrees/pp-subplots` on branch `feat/pp-subplots`. Continues from commit `cb11028`.

**Spec:** `docs/superpowers/specs/2026-05-01-per-cell-reservations-amendment.md` (authoritative — re-read before starting).

---

## File structure

**Modify:**
- `src/publiplots/layout/figure_layout.py` — tuple fields, updated formulas, `__post_init__` length validation.
- `src/publiplots/layout/auto_layout.py` — per-row/per-col `_measure`, tightbbox exclusion for reactor-managed artists.
- `src/publiplots/layout/subplots.py` — scalar/tuple coercion, wrong-length rejection.
- `src/publiplots/utils/layout_reactor.py` — `_Registration.external_to_axis` flag; default False.
- `src/publiplots/utils/legend_group.py` — pass `external_to_axis=True` when registering the unified legend with the reactor.
- `tests/test_subplots.py` — new tests for tuple behavior, existing tests updated for tuple assertions.

**No files created.**

---

## Rules of the road

- TDD strictly: failing test first → implement → run → verify pass.
- After each task, run the FULL suite. Do not proceed on a regression.
- Commit after each task with conventional-commits style.
- Do not amend previous commits. Always new commits.
- Do NOT modify any public API surface beyond what this plan specifies.
- `MM2INCH = 1/25.4` already defined; reuse.

---

## Task 1: `FigureLayout` — tuple-valued reservations

**Files:**
- Modify: `src/publiplots/layout/figure_layout.py`
- Modify: `tests/test_subplots.py` (append tuple tests + update affected existing tests)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_subplots.py` (after the existing FigureLayout tests, before the SubplotsAutoLayout section):

```python
# ---------------------------------------------------------------------------
# FigureLayout — per-row / per-column tuples
# ---------------------------------------------------------------------------


def _make_tuple_layout(nrows=1, ncols=1, **overrides):
    """Like _make_layout but builds tuple-valued reservations."""
    defaults = dict(
        nrows=nrows, ncols=ncols,
        axes_size=(50.0, 30.0),
        title_space=tuple([5.0] * nrows),
        xlabel_space=tuple([8.0] * nrows),
        ylabel_space=tuple([10.0] * ncols),
        right=tuple([2.0] * ncols),
        hspace=4.0, wspace=4.0, outer_pad=2.0, legend_column=0.0,
    )
    defaults.update(overrides)
    return FigureLayout(**defaults)


def test_figure_layout_per_row_title_space_changes_position():
    small = _make_tuple_layout(nrows=2, ncols=1,
                                title_space=(5.0, 5.0),
                                xlabel_space=(8.0, 8.0))
    big = _make_tuple_layout(nrows=2, ncols=1,
                              title_space=(20.0, 5.0),
                              xlabel_space=(8.0, 8.0))
    y0_top_small = small.axes_position(0, 0)[1]
    y0_top_big = big.axes_position(0, 0)[1]
    # Bigger top-row title pushes row 0's axes DOWN (lower y0 in figure fractions).
    assert y0_top_big < y0_top_small


def test_figure_layout_per_col_right_is_cumulative():
    uniform = _make_tuple_layout(nrows=1, ncols=3,
                                  right=(2.0, 2.0, 2.0))
    asymmetric = _make_tuple_layout(nrows=1, ncols=3,
                                     right=(2.0, 2.0, 30.0))
    w_uniform, _ = uniform.figure_size()
    w_asym, _ = asymmetric.figure_size()
    # Asymmetric adds 28 mm to ONE column's right, not 3 * 28.
    assert w_asym == pytest.approx(w_uniform + 28.0)


def test_figure_layout_scalar_broadcast_equivalence():
    a = _make_tuple_layout(nrows=2, ncols=3,
                            title_space=(8.0, 8.0),
                            xlabel_space=(12.0, 12.0),
                            ylabel_space=(14.0, 14.0, 14.0),
                            right=(2.0, 2.0, 2.0))
    b = _make_tuple_layout(nrows=2, ncols=3,
                            title_space=(8.0, 8.0),
                            xlabel_space=(12.0, 12.0),
                            ylabel_space=(14.0, 14.0, 14.0),
                            right=(2.0, 2.0, 2.0))
    assert a.figure_size() == b.figure_size()
    for r in range(2):
        for c in range(3):
            assert a.axes_position(r, c) == b.axes_position(r, c)


def test_figure_layout_with_updated_reservations_accepts_tuples():
    layout = _make_tuple_layout(nrows=2, ncols=1)
    updated = layout.with_updated_reservations(title_space=(12.0, 6.0))
    assert updated.title_space == (12.0, 6.0)
    assert updated.xlabel_space == layout.xlabel_space
    assert updated.axes_size == layout.axes_size


def test_figure_layout_wrong_length_title_space_rejected():
    with pytest.raises(ValueError, match="title_space"):
        FigureLayout(
            nrows=2, ncols=1,
            axes_size=(50.0, 30.0),
            title_space=(5.0, 5.0, 5.0),  # wrong length
            xlabel_space=(8.0, 8.0),
            ylabel_space=(10.0,), right=(2.0,),
            hspace=4.0, wspace=4.0, outer_pad=2.0, legend_column=0.0,
        )


def test_figure_layout_wrong_length_ylabel_space_rejected():
    with pytest.raises(ValueError, match="ylabel_space"):
        FigureLayout(
            nrows=1, ncols=2,
            axes_size=(50.0, 30.0),
            title_space=(5.0,), xlabel_space=(8.0,),
            ylabel_space=(10.0, 10.0, 10.0),  # wrong length
            right=(2.0, 2.0),
            hspace=4.0, wspace=4.0, outer_pad=2.0, legend_column=0.0,
        )


def test_figure_layout_rejects_scalar_reservations():
    """FigureLayout takes tuples only; broadcast happens at the pp.subplots boundary."""
    with pytest.raises((ValueError, TypeError)):
        FigureLayout(
            nrows=2, ncols=1,
            axes_size=(50.0, 30.0),
            title_space=5.0,  # scalar, not tuple
            xlabel_space=(8.0, 8.0),
            ylabel_space=(10.0,), right=(2.0,),
            hspace=4.0, wspace=4.0, outer_pad=2.0, legend_column=0.0,
        )
```

Also update EXISTING tests that read scalar reservation fields to read tuple elements:

- `test_figure_layout_single_cell_size` and `test_figure_layout_2x3_size_matches_formula` and `test_figure_layout_legend_column_adds_width_only` and `test_figure_layout_axes_position_is_deterministic` and `test_figure_layout_axes_positions_dont_overlap` and `test_figure_layout_row_zero_is_top`: these all go through `_make_layout(...)` which currently passes scalar reservations to `FigureLayout`. Update `_make_layout` to broadcast its scalar inputs to tuples:

```python
def _make_layout(nrows=1, ncols=1, **overrides):
    defaults = dict(
        nrows=nrows, ncols=ncols,
        axes_size=(50.0, 30.0),
        title_space=5.0, xlabel_space=8.0, ylabel_space=10.0, right=2.0,
        hspace=8.0, wspace=10.0, outer_pad=2.0, legend_column=0.0,
    )
    defaults.update(overrides)
    # Broadcast scalar reservations to the expected tuple length.
    for side in ("title_space", "xlabel_space"):
        v = defaults[side]
        if not isinstance(v, tuple):
            defaults[side] = (float(v),) * defaults["nrows"]
    for side in ("ylabel_space", "right"):
        v = defaults[side]
        if not isinstance(v, tuple):
            defaults[side] = (float(v),) * defaults["ncols"]
    return FigureLayout(**defaults)
```

- `test_figure_layout_with_updated_reservations_preserves_axes_size` — update the assertions:

```python
def test_figure_layout_with_updated_reservations_preserves_axes_size():
    layout = _make_layout()
    updated = layout.with_updated_reservations(title_space=(20.0,), xlabel_space=(15.0,))
    assert updated.axes_size == layout.axes_size
    assert updated.title_space == (20.0,)
    assert updated.xlabel_space == (15.0,)
    assert updated.ylabel_space == layout.ylabel_space
```

Formula expected values stay correct because after broadcasting a scalar `v` to a tuple of length `n`, `sum(tuple) == n * v` — same as the old formula.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_subplots.py -v -k "figure_layout"`
Expected: new tuple tests FAIL with TypeError/AssertionError because `FigureLayout` still has scalar fields.

- [ ] **Step 3: Rewrite `src/publiplots/layout/figure_layout.py`**

Replace the file contents with:

```python
"""
Pure-geometry layout for publiplots subplot grids.

FigureLayout computes figure size and axes positions from declared mm
dimensions. No matplotlib imports — this module is pure math.

Per-side reservations are TUPLES indexed by position (length nrows for
title_space / xlabel_space; length ncols for ylabel_space / right).
Scalar-to-tuple broadcast happens at the pp.subplots() boundary.
"""

from dataclasses import dataclass, field, replace
from typing import Tuple


@dataclass
class FigureLayout:
    """
    Millimeter-based grid geometry with per-row / per-column reservations.

    Parameters
    ----------
    nrows, ncols : int
        Grid shape (>= 1).
    axes_size : tuple of (width, height) in mm
        The declared axes bbox for every cell. Inviolate after construction.
    title_space : tuple[float, ...] of length nrows
        Space reserved above each row for titles.
    xlabel_space : tuple[float, ...] of length nrows
        Space reserved below each row for x-axis labels and tick labels.
    ylabel_space : tuple[float, ...] of length ncols
        Space reserved left of each column for y-axis labels and tick labels.
    right : tuple[float, ...] of length ncols
        Space reserved right of each column.
    hspace, wspace : float
        Inter-row and inter-column gaps (scalar; gaps are global).
    outer_pad : float
        Figure outer margin (same on all four sides).
    legend_column : float
        Extra width on the far right, outside the grid. Never auto-measured.
    """

    nrows: int
    ncols: int
    axes_size: Tuple[float, float]
    title_space: Tuple[float, ...]
    xlabel_space: Tuple[float, ...]
    ylabel_space: Tuple[float, ...]
    right: Tuple[float, ...]
    hspace: float
    wspace: float
    outer_pad: float
    legend_column: float

    def __post_init__(self) -> None:
        # Enforce tuple type + non-scalar (fail loud on accidental scalars).
        for side in ("title_space", "xlabel_space", "ylabel_space", "right"):
            val = getattr(self, side)
            if not isinstance(val, tuple):
                raise TypeError(
                    f"{side} must be a tuple of floats; got {type(val).__name__}. "
                    f"pp.subplots() broadcasts scalars — construct FigureLayout with tuples."
                )
        # Enforce length invariants.
        for side in ("title_space", "xlabel_space"):
            val = getattr(self, side)
            if len(val) != self.nrows:
                raise ValueError(
                    f"{side} must have length nrows={self.nrows}, got length {len(val)}"
                )
        for side in ("ylabel_space", "right"):
            val = getattr(self, side)
            if len(val) != self.ncols:
                raise ValueError(
                    f"{side} must have length ncols={self.ncols}, got length {len(val)}"
                )
        # Enforce non-negativity.
        for side in ("title_space", "xlabel_space", "ylabel_space", "right"):
            for i, v in enumerate(getattr(self, side)):
                if v < 0:
                    raise ValueError(f"{side}[{i}] must be non-negative, got {v}")

    def figure_size(self) -> Tuple[float, float]:
        """Total figure size in mm as (width, height)."""
        w_ax, h_ax = self.axes_size
        W = (
            self.outer_pad
            + sum(self.ylabel_space)
            + self.ncols * w_ax
            + sum(self.right)
            + max(self.ncols - 1, 0) * self.wspace
            + self.legend_column
            + self.outer_pad
        )
        H = (
            self.outer_pad
            + sum(self.title_space)
            + self.nrows * h_ax
            + sum(self.xlabel_space)
            + max(self.nrows - 1, 0) * self.hspace
            + self.outer_pad
        )
        return W, H

    def axes_position(self, row: int, col: int) -> Tuple[float, float, float, float]:
        """Figure-fraction (x0, y0, w, h) for the cell at (row, col)."""
        W, H = self.figure_size()
        w_ax, h_ax = self.axes_size

        # x: accumulate offsets of preceding columns.
        x0_mm = self.outer_pad
        for c in range(col):
            x0_mm += self.ylabel_space[c] + w_ax + self.right[c] + self.wspace
        x0_mm += self.ylabel_space[col]

        # y: accumulate from the bottom of the figure upward. Rows below
        # (r > row) contribute their xlabel + axes + title + hspace.
        y0_mm = self.outer_pad
        for r in range(self.nrows - 1, row, -1):
            y0_mm += self.xlabel_space[r] + h_ax + self.title_space[r] + self.hspace
        y0_mm += self.xlabel_space[row]

        return (x0_mm / W, y0_mm / H, w_ax / W, h_ax / H)

    def with_updated_reservations(self, **overrides) -> "FigureLayout":
        """Return a copy with the given fields updated. ``axes_size`` must not change."""
        if "axes_size" in overrides:
            raise ValueError("axes_size is inviolate; cannot be overridden")
        return replace(self, **overrides)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_subplots.py -v -k "figure_layout"`
Expected: all FigureLayout tests PASS (existing + 7 new tuple tests).

- [ ] **Step 5: Commit**

```bash
git add src/publiplots/layout/figure_layout.py tests/test_subplots.py
git commit -m "feat(layout): tuple-valued per-row/per-col reservations in FigureLayout"
```

Other tests (SubplotsAutoLayout, pp.subplots) will break temporarily after this commit because they still pass scalars — fix in the next tasks.

---

## Task 2: `pp.subplots()` — scalar/tuple coercion

**Files:**
- Modify: `src/publiplots/layout/subplots.py`
- Modify: `tests/test_subplots.py` (append coercion tests)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_subplots.py` (after the existing pp.subplots tests):

```python
# ---------------------------------------------------------------------------
# pp.subplots() — scalar/tuple coercion
# ---------------------------------------------------------------------------


def test_subplots_scalar_reservation_broadcasts_to_nrows():
    fig, _ = pp.subplots(2, 3, axes_size=(50, 30), title_space=8)
    assert fig._publiplots_layout.title_space == (8.0, 8.0)


def test_subplots_scalar_reservation_broadcasts_to_ncols():
    fig, _ = pp.subplots(2, 3, axes_size=(50, 30), right=5)
    assert fig._publiplots_layout.right == (5.0, 5.0, 5.0)


def test_subplots_tuple_reservation_preserved():
    fig, _ = pp.subplots(2, 3, axes_size=(50, 30), title_space=(12, 6))
    assert fig._publiplots_layout.title_space == (12.0, 6.0)


def test_subplots_wrong_length_title_space_raises():
    with pytest.raises(ValueError, match="title_space"):
        pp.subplots(2, 3, axes_size=(50, 30), title_space=(12, 6, 3))


def test_subplots_wrong_length_ylabel_space_raises():
    with pytest.raises(ValueError, match="ylabel_space"):
        pp.subplots(2, 3, axes_size=(50, 30), ylabel_space=(10, 10))


def test_subplots_default_reservations_broadcast_to_tuple():
    fig, _ = pp.subplots(2, 3, axes_size=(50, 30))
    layout = fig._publiplots_layout
    assert len(layout.title_space) == 2
    assert len(layout.xlabel_space) == 2
    assert len(layout.ylabel_space) == 3
    assert len(layout.right) == 3


def test_subplots_negative_tuple_element_raises():
    with pytest.raises(ValueError, match="title_space"):
        pp.subplots(2, 1, axes_size=(50, 30), title_space=(5, -1))
```

Also, these existing tests will need to update — their assertions on scalar layout fields become tuple assertions. Update them when writing code in Step 3 (they shouldn't break after the coercion lands):

- `test_subplots_scalar_axes_size_coerced_to_tuple` — still passes (reads `axes_size` which is already a tuple).
- `test_subplots_legend_column_reserves_extra_width` — still passes (reads figwidth).
- `test_subplots_all_locked_skips_hook` and `test_subplots_any_auto_side_attaches_hook` — STILL pass scalar reservations as args; coercion handles them.
- `test_subplots_validates_negative_reservation` — still passes (scalar negative is caught at the coercion boundary).

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_subplots.py -v`
Expected: the 7 new tests FAIL (missing coercion); some existing tests also FAIL because `FigureLayout` now demands tuples.

- [ ] **Step 3: Update `src/publiplots/layout/subplots.py`**

Find the validation / resolution block (around the `user_values = dict(...)` and `resolved = {}` section) and replace with:

```python
    # --- resolve reservation defaults & track locked sides ----------------
    # Per-row and per-column reservation kinds.
    _ROW_SIDES = ("title_space", "xlabel_space")
    _COL_SIDES = ("ylabel_space", "right")
    _SCALAR_SIDES = ("hspace", "wspace", "outer_pad")

    user_values = dict(
        title_space=title_space, xlabel_space=xlabel_space,
        ylabel_space=ylabel_space, right=right,
        hspace=hspace, wspace=wspace, outer_pad=outer_pad,
    )
    locked = {k for k, v in user_values.items() if v is not None}

    resolved = {}
    for side in _ROW_SIDES + _COL_SIDES:
        user_val = user_values[side]
        length = nrows if side in _ROW_SIDES else ncols
        if user_val is None:
            # Broadcast rcParams default to the correct length.
            default_scalar = resolve_param(f"subplots.{side}", None)
            tup = (float(default_scalar),) * length
        elif isinstance(user_val, (int, float)):
            if user_val < 0:
                raise ValueError(f"{side} must be non-negative, got {user_val}")
            tup = (float(user_val),) * length
        else:
            # Treat as sequence.
            try:
                tup = tuple(float(x) for x in user_val)
            except (TypeError, ValueError):
                raise ValueError(
                    f"{side} must be a scalar or sequence of numbers, got {user_val!r}"
                )
            if len(tup) != length:
                raise ValueError(
                    f"{side} must have length {length} for nrows={nrows} ncols={ncols}, "
                    f"got length {len(tup)}"
                )
            for i, v in enumerate(tup):
                if v < 0:
                    raise ValueError(f"{side}[{i}] must be non-negative, got {v}")
        resolved[side] = tup

    for side in _SCALAR_SIDES:
        val = resolve_param(f"subplots.{side}", user_values[side])
        if val < 0:
            raise ValueError(f"{side} must be non-negative, got {val}")
        resolved[side] = float(val)

    if legend_column < 0:
        raise ValueError(f"legend_column must be non-negative, got {legend_column}")
```

The `FigureLayout(...)` call below it now correctly receives tuples for the 4 per-side fields and scalars for the gaps/outer_pad.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_subplots.py -v`
Expected: all `test_subplots_*` PASS. (SubplotsAutoLayout tests may still fail — that's Task 3.)

Run the full suite: `uv run pytest tests/ -q`. SubplotsAutoLayout tests likely still fail because their asserts on layout fields assume scalars. Next task fixes that.

- [ ] **Step 5: Commit**

```bash
git add src/publiplots/layout/subplots.py tests/test_subplots.py
git commit -m "feat(layout): scalar/tuple coercion in pp.subplots() reservations"
```

---

## Task 3: `SubplotsAutoLayout` — per-position measurement + test updates

**Files:**
- Modify: `src/publiplots/layout/auto_layout.py`
- Modify: `tests/test_subplots.py` (update existing SubplotsAutoLayout tests)

- [ ] **Step 1: Update failing tests**

Four existing tests read scalar layout fields and need to index into tuples now:

```python
# test_auto_layout_grows_title_space_for_title
def test_auto_layout_grows_title_space_for_title():
    layout = _make_layout(nrows=1, ncols=1, title_space=1.0)
    fig, axes = _make_fig_with_layout(layout)
    ax = axes[0][0]
    ax.set_title("A title that needs more vertical room than 1 mm")
    reactor = SubplotsAutoLayout(fig, layout, locked=set())
    fig.canvas.draw()
    assert reactor._layout.title_space[0] > 1.0, (
        f"title_space[0] should have grown past 1.0 mm, got {reactor._layout.title_space[0]:.2f}"
    )


# test_auto_layout_locked_side_not_remeasured
def test_auto_layout_locked_side_not_remeasured():
    layout = _make_layout(title_space=25.0)
    fig, axes = _make_fig_with_layout(layout)
    axes[0][0].set_title("tiny")
    reactor = SubplotsAutoLayout(fig, layout, locked={"title_space"})
    fig.canvas.draw()
    assert reactor._layout.title_space == (25.0,), (
        f"locked title_space should remain (25.0,), got {reactor._layout.title_space}"
    )
```

The other two SubplotsAutoLayout tests (`test_auto_layout_preserves_axes_size_after_resize`, `test_auto_layout_second_draw_no_change_within_threshold`, `test_auto_layout_no_hook_when_all_sides_locked`, `test_auto_layout_attaches_layout_to_figure`) don't read layout fields directly — they should keep passing. Verify after implementation.

Append new tests for per-position behavior:

```python
def test_auto_layout_right_is_per_column():
    """Adding a wide artist to one column should only grow that column's right."""
    layout = _make_layout(nrows=1, ncols=3, right=2.0)
    fig, axes = _make_fig_with_layout(layout)
    # Attach a wide text artist to the rightmost axes only — it inflates
    # that axes' tightbbox beyond the spine.
    axes[0][2].text(
        1.3, 0.5, "hanging text", transform=axes[0][2].transAxes,
    )
    reactor = SubplotsAutoLayout(fig, layout, locked=set())
    fig.canvas.draw()
    # Column 2 grows; columns 0 and 1 stay at baseline (~ 2 mm).
    assert reactor._layout.right[2] > reactor._layout.right[0] + 5.0, (
        f"right[2] should exceed right[0] by > 5 mm after the text, got "
        f"right[0]={reactor._layout.right[0]:.1f}, right[2]={reactor._layout.right[2]:.1f}"
    )


def test_auto_layout_title_space_is_per_row():
    """Set a title on only one row; only that row's title_space grows."""
    layout = _make_layout(nrows=2, ncols=1)
    fig, axes = _make_fig_with_layout(layout)
    axes[0][0].set_title("Top row title")
    # axes[1][0] has no title.
    reactor = SubplotsAutoLayout(fig, layout, locked=set())
    fig.canvas.draw()
    assert reactor._layout.title_space[0] > reactor._layout.title_space[1] + 2.0, (
        f"title_space[0] should exceed title_space[1] by > 2 mm, got "
        f"{reactor._layout.title_space[0]:.1f} vs {reactor._layout.title_space[1]:.1f}"
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_subplots.py -v -k "auto_layout"`
Expected: the 4 updated existing tests PASS (indexing into a tuple works even on scalar-broadcast layouts) but the 2 new per-position tests FAIL because `_measure()` is still grid-wide.

- [ ] **Step 3: Rewrite `src/publiplots/layout/auto_layout.py`**

Replace the file contents with:

```python
"""
Draw-event hook that keeps declared axes sizes fixed while the figure
grows to fit auto-measured decorations.

Reservations are per-row (title_space, xlabel_space) or per-column
(ylabel_space, right). Measurement excludes LayoutReactor-managed
artists that are flagged as external_to_axis (e.g., pp.legend_group) —
those are handled by the reactor's own anchoring geometry plus the
user's legend_column reservation, not by axis-level tightbbox.
"""

from typing import Dict, Set, Tuple

from publiplots.layout.figure_layout import FigureLayout


_MM2INCH = 1 / 25.4
_UPDATE_THRESHOLD_MM = 0.1
_ALL_SIDES = {"title_space", "xlabel_space", "ylabel_space", "right"}

# side_name -> (axis_kind, bbox_fn)
#   axis_kind: "row" (result length == nrows) or "col" (length == ncols)
#   bbox_fn:   float (ax_bbox, tight_bbox) -> px
_SIDE_CALCULATORS = {
    "title_space":  ("row", lambda ax_bb, t: t.y1 - ax_bb.y1),
    "xlabel_space": ("row", lambda ax_bb, t: ax_bb.y0 - t.y0),
    "ylabel_space": ("col", lambda ax_bb, t: ax_bb.x0 - t.x0),
    "right":        ("col", lambda ax_bb, t: t.x1 - ax_bb.x1),
}


class SubplotsAutoLayout:
    """Per-figure draw-event listener that resizes the figure to fit decorations."""

    def __init__(self, fig, layout: FigureLayout, locked: Set[str]):
        self._fig = fig
        self._layout = layout
        self._locked = set(locked)
        self._updating = False

        fig._publiplots_layout = layout
        fig._publiplots_auto_layout = self

        if _ALL_SIDES.issubset(self._locked):
            self._cid = None
        else:
            self._cid = fig.canvas.mpl_connect("draw_event", self._on_draw)

    def _on_draw(self, event) -> None:
        if self._updating:
            return
        self._updating = True
        try:
            new = self._measure()
            if self._needs_update(new):
                self._apply(new)
        finally:
            self._updating = False

    def _measure(self) -> Dict[str, Tuple[float, ...]]:
        fig = self._fig
        dpi = fig.dpi
        if dpi <= 0:
            return {}
        axes_matrix = self._axes_matrix()
        if not axes_matrix or not axes_matrix[0]:
            return {}

        managed = self._externally_managed_artist_ids()
        measured: Dict[str, Tuple[float, ...]] = {}

        for side, (axis_kind, calc) in _SIDE_CALCULATORS.items():
            if side in self._locked:
                continue
            if axis_kind == "row":
                per = []
                for row in axes_matrix:
                    max_px = 0.0
                    for ax in row:
                        max_px = max(max_px, self._side_extent(ax, calc, managed))
                    per.append(max(max_px / dpi * 25.4, 0.0))
                measured[side] = tuple(per)
            else:  # "col"
                ncols = len(axes_matrix[0])
                per = []
                for c in range(ncols):
                    max_px = 0.0
                    for row in axes_matrix:
                        ax = row[c]
                        max_px = max(max_px, self._side_extent(ax, calc, managed))
                    per.append(max(max_px / dpi * 25.4, 0.0))
                measured[side] = tuple(per)
        return measured

    def _side_extent(self, ax, calc, managed_artist_ids) -> float:
        """Measure ax's tight-vs-window extent for one side, excluding managed overlays."""
        ax_bbox = ax.get_window_extent()
        # Temporarily drop managed artists from layout consideration.
        toggled = []
        for child in ax.get_children():
            if id(child) in managed_artist_ids and child.get_in_layout():
                child.set_in_layout(False)
                toggled.append(child)
        try:
            tight = ax.get_tightbbox()
        finally:
            for child in toggled:
                child.set_in_layout(True)
        if tight is None:
            return 0.0
        return calc(ax_bbox, tight)

    def _externally_managed_artist_ids(self) -> set:
        """IDs of LayoutReactor registrations flagged external_to_axis=True."""
        reactor = getattr(self._fig, "_publiplots_layout_reactor", None)
        if reactor is None:
            return set()
        return {
            id(reg.artist)
            for reg in reactor._registrations
            if getattr(reg, "external_to_axis", False)
        }

    def _needs_update(self, measured: Dict[str, Tuple[float, ...]]) -> bool:
        for side, new_tuple in measured.items():
            current = getattr(self._layout, side)
            if len(new_tuple) != len(current):
                return True
            for new_v, cur_v in zip(new_tuple, current):
                if abs(new_v - cur_v) >= _UPDATE_THRESHOLD_MM:
                    return True
        return False

    def _apply(self, measured: Dict[str, Tuple[float, ...]]) -> None:
        new_layout = self._layout.with_updated_reservations(**measured)
        self._layout = new_layout
        self._fig._publiplots_layout = new_layout

        W, H = new_layout.figure_size()
        self._fig.set_size_inches(W * _MM2INCH, H * _MM2INCH, forward=False)

        for r, row in enumerate(self._axes_matrix()):
            for c, ax in enumerate(row):
                ax.set_position(new_layout.axes_position(r, c))

    def _axes_matrix(self):
        stored = getattr(self._fig, "_publiplots_axes", None)
        if stored is not None:
            return stored
        flat = list(self._fig.axes)
        nrows, ncols = self._layout.nrows, self._layout.ncols
        if len(flat) < nrows * ncols:
            return [[]]
        return [flat[r * ncols:(r + 1) * ncols] for r in range(nrows)]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_subplots.py -v -k "auto_layout"`
Expected: all (6 existing + 2 new) PASS.

Run the full suite: `uv run pytest tests/ -q`. All tests should pass.

- [ ] **Step 5: Commit**

```bash
git add src/publiplots/layout/auto_layout.py tests/test_subplots.py
git commit -m "feat(layout): per-row/per-column measurement in SubplotsAutoLayout"
```

---

## Task 4: `LayoutReactor` + `legend_group` — `external_to_axis` flag

**Files:**
- Modify: `src/publiplots/utils/layout_reactor.py` (add `external_to_axis` field to `_Registration`, plumb through `register`)
- Modify: `src/publiplots/utils/legend_group.py` (pass `external_to_axis=True` when registering)
- Modify: `tests/test_subplots.py` (add the legend-exclusion integration test)

- [ ] **Step 1: Write failing test**

Append to `tests/test_subplots.py`:

```python
def test_auto_layout_excludes_legend_group_from_tightbbox():
    """pp.legend_group anchors a legend external to the axes — its width
    is handled by legend_column, not the column's `right` reservation."""
    from publiplots.utils.legend import create_legend_handles
    fig, axes = pp.subplots(1, 3, axes_size=(45, 30), legend_column=30)
    # Draw something on each axes to establish real tick labels.
    for ax in axes:
        ax.plot([0, 1, 2], [0, 1, 0])
    group = pp.legend_group(anchor=axes[-1])
    group.add_legend(
        handles=create_legend_handles(
            labels=["A", "B", "C"],
            colors=list(pp.color_palette("pastel", 3)),
            alpha=0.2, linewidth=1.0,
        ),
        label="group",
    )
    fig.canvas.draw()
    layout = fig._publiplots_layout
    # The rightmost column's `right` should be close to baseline (< 5 mm),
    # NOT inflated to the legend's width.
    assert layout.right[-1] < 5.0, (
        f"right[-1] should be ~ baseline (legend excluded from tightbbox); "
        f"got {layout.right[-1]:.1f} mm"
    )


def test_auto_layout_per_axis_pp_legend_is_counted():
    """Per-axis pp.legend() is part of the axes' visual footprint —
    should inflate the column's `right`, unlike legend_group."""
    from publiplots.utils.legend import create_legend_handles
    fig, axes = pp.subplots(1, 2, axes_size=(45, 30))
    for ax in axes:
        ax.plot([0, 1, 2], [0, 1, 0])
    builder = pp.legend(axes[0], auto=False)
    builder.add_legend(
        handles=create_legend_handles(
            labels=["A", "B"],
            colors=["#5d83c3", "#c0392b"],
            alpha=0.2, linewidth=1.0,
        ),
        label="group",
    )
    fig.canvas.draw()
    layout = fig._publiplots_layout
    # Column 0 has the per-axis legend → its `right` should exceed column 1's.
    assert layout.right[0] > layout.right[1] + 5.0, (
        f"right[0] (has pp.legend) should exceed right[1] by > 5 mm, got "
        f"right[0]={layout.right[0]:.1f}, right[1]={layout.right[1]:.1f}"
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_subplots.py -v -k "excludes_legend_group or per_axis_pp_legend"`
Expected: `test_auto_layout_excludes_legend_group_from_tightbbox` FAILS because the `external_to_axis` flag isn't set yet. `test_auto_layout_per_axis_pp_legend_is_counted` may pass or fail depending on current behavior — capture it in the output.

- [ ] **Step 3: Add `external_to_axis` to `_Registration`**

In `src/publiplots/utils/layout_reactor.py`, update the `_Registration` dataclass:

```python
@dataclass
class _Registration:
    ax: Axes
    artist: object  # Legend or Colorbar; duck-typed via set_bbox_to_anchor
    mm_x_from_right: float
    mm_y_from_top: float
    mm_width: Optional[float] = None
    mm_height: Optional[float] = None
    # External overlays (legend_group anchored to an axes, standalone
    # colorbar) should NOT count toward the axes' tightbbox — their
    # geometry is owned by the reactor + user's legend_column reservation.
    # Per-axis legends (pp.legend(ax)) keep this False.
    external_to_axis: bool = False
```

Update the `register` method signature to accept `external_to_axis: bool = False` and store it:

```python
def register(
    self,
    *,
    ax: Axes,
    artist: object,
    mm_x_from_right: float,
    mm_y_from_top: float,
    mm_width: Optional[float] = None,
    mm_height: Optional[float] = None,
    external_to_axis: bool = False,
) -> None:
    self._registrations.append(_Registration(
        ax=ax,
        artist=artist,
        mm_x_from_right=mm_x_from_right,
        mm_y_from_top=mm_y_from_top,
        mm_width=mm_width,
        mm_height=mm_height,
        external_to_axis=external_to_axis,
    ))
```

Read the existing `register` method first to make sure the signature is extended correctly and no callers break.

- [ ] **Step 4: Pass `external_to_axis=True` from `legend_group`**

In `src/publiplots/utils/legend_group.py`, find every `LayoutReactor.get(...).register(...)` or equivalent `self._reactor.register(...)` call made by the legend-group machinery. For each one, add `external_to_axis=True` to the call.

If `legend_group` delegates to `LegendBuilder.add_legend(...)` (which is what the spec said), the registration call lives inside `LegendBuilder`. In that case, we need a way for `legend_group` to signal "external" to the builder. Approach: give `LegendBuilder` an optional `external_to_axis` constructor arg (default False) which it passes to every `reactor.register(...)` it makes. Then `MultiAxesLegendGroup` constructs its builder with `external_to_axis=True`.

Read `src/publiplots/utils/legend.py` and `src/publiplots/utils/legend_group.py` to confirm the call chain, then pipe the flag through:

- `LegendBuilder.__init__(self, ax, *, auto=True, anchor_ax=None, external_to_axis=False, ...)` — store as `self._external_to_axis = external_to_axis`.
- Inside `LegendBuilder.add_legend`, wherever it calls `self._reactor.register(...)`, add `external_to_axis=self._external_to_axis`.
- Same for `add_colorbar` if applicable.
- `MultiAxesLegendGroup.__init__` — when it constructs its internal `LegendBuilder`, pass `external_to_axis=True`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_subplots.py -v`
Expected: all tests PASS including the two new legend-exclusion tests.

Run full suite: `uv run pytest tests/ -q`. No regressions in PR #78 tests.

- [ ] **Step 6: Commit**

```bash
git add src/publiplots/utils/layout_reactor.py src/publiplots/utils/legend.py src/publiplots/utils/legend_group.py tests/test_subplots.py
git commit -m "feat(legend): external_to_axis flag excludes legend_group from tightbbox"
```

---

## Task 5: Smoke-test gallery & final verification

**Files:** none modified; verification only.

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest tests/ -q`
Expected: all tests pass.

- [ ] **Step 2: Smoke-run the `plot_14` gallery example**

Run: `uv run python examples/plots/plot_14_edgecolor_control.py 2>&1 | tail -10`
Expected: exits clean, no traceback.

- [ ] **Step 3: Visual check via a diagnostic script**

Run this from the worktree to confirm the legend-group fix:

```bash
uv run python - <<'EOF'
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import publiplots as pp

pp.set_notebook_style()
pp.rcParams['edgecolor'] = 'black'

np.random.seed(11)
df = pd.DataFrame({
    'group': np.repeat(['A','B','C'], 30),
    'value': np.concatenate([np.random.normal(m, s, 30)
                             for m, s in [(50, 8), (65, 9), (80, 10)]]),
})
fig, axes = pp.subplots(1, 3, axes_size=(45, 30), legend_column=30)
pp.barplot(data=df, x='group', y='value', hue='group', palette='pastel',
           errorbar='se', title='Bar', ax=axes[0], legend=False)
pp.boxplot(data=df, x='group', y='value', hue='group', palette='pastel',
           title='Box', ax=axes[1], legend=False)
pp.scatterplot(data=df, x='group', y='value', hue='group', palette='pastel',
               title='Scatter', ax=axes[2], legend=False)

group = pp.legend_group(anchor=axes[-1])
group.add_legend(
    handles=pp.create_legend_handles(
        labels=['A','B','C'],
        colors=list(pp.color_palette('pastel', 3)),
        alpha=pp.rcParams['alpha'],
        linewidth=pp.rcParams['lines.linewidth'],
        edgecolors=pp.rcParams['edgecolor']),
    label='group',
)
fig.canvas.draw()
layout = fig._publiplots_layout
print("title_space  =", layout.title_space)
print("xlabel_space =", layout.xlabel_space)
print("ylabel_space =", layout.ylabel_space)
print("right        =", layout.right)
print("figure_size  =", layout.figure_size())
EOF
```

Expected: `right` is a tuple `(~2, ~2, ~2)` (not `(33, 33, 33)`); figure width ~220 mm (not 303 mm).

- [ ] **Step 4: Push and update PR**

Run: `git push`
No new PR — this amendment lands on the same `feat/pp-subplots` branch that's already open as PR #79.

---

## Self-review

- Spec coverage: the amendment calls for (a) tuple reservations, (b) legend-group tightbbox exclusion. Task 1 → tuples in FigureLayout. Task 2 → coercion in pp.subplots. Task 3 → per-row/per-col measurement. Task 4 → external_to_axis plumbing. Task 5 → verification. All covered.
- Placeholders: none — every step has exact code.
- Type consistency: `title_space: Tuple[float, ...]` used consistently in FigureLayout, SubplotsAutoLayout, and pp.subplots. `external_to_axis: bool = False` used consistently across LegendBuilder, MultiAxesLegendGroup, and SubplotsAutoLayout's `_externally_managed_artist_ids`.

Ready for execution.
