# Design — `pp.subplots()` Fixed-Axes Helper (PR #79)

**Status:** approved, awaiting implementation plan.
**Worktree:** `.worktrees/pp-subplots` on `feat/pp-subplots`.
**Follows:** PR #78 (bulletproof LegendBuilder, fixed-axes/flexible-canvas philosophy).
**Blocks:** legend-aware follow-up PR (collision warning, `plot_15_legends.py` tutorial).

---

## Goal

Ship `pp.subplots(nrows, ncols, axes_size=(w, h), ...)` so users can declare axes dimensions in millimeters and have publiplots compute a figure size that accommodates titles, labels, and an optional right-hand legend column. Reservations are **auto-measured on first draw** by default, with explicit overrides for reproducible figures.

The contract: **`axes_size` is inviolate; the figure grows to fit decorations.** This is the inverse of matplotlib's `constrained_layout` / `tight_layout` (which shrink axes to fit a fixed figure).

## Non-goals

- `pp.figure()` — deferred (YAGNI; `pp.subplots()` covers the common case).
- GridSpec / heterogeneous axes sizes / per-row-col reservation overrides — deferred to the composer PR (#80+), where panel heterogeneity lives at the page level.
- Legend-width awareness (`legend_column` auto-sizing, collision warning, tutorial) — follow-up PR that depends on this one.
- Two-pass rendering with oscillation handling — not needed. Decoration size is independent of axes size, so one measurement pass converges.

## Philosophy alignment

Publiplots is a publication-first library. Journals specify panel dimensions in mm. Illustrator defaults to mm/pt. The rest of publiplots (`LegendLayout`, `LayoutReactor`, `x_offset`, `vpad`) already uses mm. `pp.subplots()` continues that: mm in, mm out, matplotlib's inches live only at the boundary.

---

## Architecture

Three units, isolated:

```
pp.subplots(...) ──► FigureLayout (pure math, mm) ──► plt.figure + ax.set_position
                           │
                           └─► SubplotsAutoLayout (draw-event hook)
                                  │
                                  └─► measures decorations → updates FigureLayout
                                                             → fig.set_size_inches
                                                             → ax.set_position (all)
                                                             → LayoutReactor (PR #78)
                                                                repositions legends
```

**Why three units.** `FigureLayout` is pure geometry — testable without matplotlib. `pp.subplots()` is the API surface. `SubplotsAutoLayout` is the only piece that touches the canvas, keeping the matplotlib-dependent surface small.

---

## Component 1 — `FigureLayout` (pure geometry)

**File:** `src/publiplots/layout/figure_layout.py`.

Dataclass mirroring the `LegendLayout` pattern from PR #78. All values in mm. No matplotlib imports.

```python
@dataclass
class FigureLayout:
    nrows: int
    ncols: int
    axes_size: tuple[float, float]    # (w, h) — the declared axes bbox
    title_space: float                # reserved above each row
    xlabel_space: float               # reserved below each row
    ylabel_space: float               # reserved left of each col
    right: float                      # reserved right of each col
    hspace: float                     # gap between rows
    wspace: float                     # gap between cols
    outer_pad: float                  # figure outer margin (all sides)
    legend_column: float              # extra on far right, outside the grid

    def figure_size(self) -> tuple[float, float]:
        w_ax, h_ax = self.axes_size
        W = (
            self.outer_pad
            + self.ncols * (self.ylabel_space + w_ax + self.right)
            + (self.ncols - 1) * self.wspace
            + self.legend_column
            + self.outer_pad
        )
        H = (
            self.outer_pad
            + self.nrows * (self.title_space + h_ax + self.xlabel_space)
            + (self.nrows - 1) * self.hspace
            + self.outer_pad
        )
        return W, H

    def axes_position(self, row: int, col: int) -> tuple[float, float, float, float]:
        """(x0, y0, w, h) in figure fractions. Row 0 is the top row."""
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
        from dataclasses import replace
        return replace(self, **overrides)
```

**Cell convention.** Each grid cell reserves `ylabel_space` on its left, `right` on its right, `title_space` above, `xlabel_space` below. Gaps between cells: `hspace` (vertical), `wspace` (horizontal). This keeps every cell identical — no special-case for edge rows/cols.

**Invariant.** `axes_size` never mutates post-construction. Reservations and gaps may (auto-measure), but declared axes dimensions stay constant across the figure's life.

---

## Component 2 — `pp.subplots()` public API

**File:** `src/publiplots/layout/subplots.py`.

```python
def subplots(
    nrows: int = 1,
    ncols: int = 1,
    *,
    axes_size: tuple[float, float] | float,   # required; scalar → square
    sharex: bool | str = False,
    sharey: bool | str = False,
    title_space: float | None = None,
    xlabel_space: float | None = None,
    ylabel_space: float | None = None,
    right: float | None = None,
    hspace: float | None = None,
    wspace: float | None = None,
    outer_pad: float | None = None,
    legend_column: float = 0,
    **fig_kw,
) -> tuple[Figure, np.ndarray | Axes]:
    ...
```

**Input handling.**
- `axes_size` is required. Scalar → `(s, s)`. Non-positive → `ValueError`.
- `nrows`, `ncols` must be ≥ 1 → else `ValueError`.
- User-specified reservations must be ≥ 0 → else `ValueError`.
- `figsize` in `fig_kw` → `TypeError("use axes_size; figure size is computed from axes + reservations")`.
- `layout` / `constrained_layout` / `tight_layout` in `fig_kw` → ignored with `UserWarning`; we always set `layout=None`.
- Each `*_space` / gap arg that is `None` → initial value from `pp.rcParams`, then auto-measured on first draw.
- Each arg given as a float → **locked**; never remeasured.
- `legend_column` defaults to 0. It is never auto-measured — users opt in explicitly by passing a value.

**Construction flow.**
1. Resolve defaults from `pp.rcParams` for any `None` arg.
2. Track which args were user-provided (lock set).
3. Build `FigureLayout`.
4. `fig = plt.figure(figsize=(W/25.4, H/25.4), layout=None, **fig_kw)`.
5. Create axes via `fig.add_axes(layout.axes_position(r, c))` row-by-row, col-by-col.
6. Wire `sharex` / `sharey` semantics matching `plt.subplots` (`True`/`"all"` → share with (0,0); `"row"` → share within row; `"col"` → share within col; `False`/`"none"` → independent).
7. If the lock set doesn't cover every side, attach `SubplotsAutoLayout(fig, layout, locked)`. If all sides locked, no hook.
8. Attach `fig._publiplots_layout = layout` for introspection and future composer consumption.
9. Pack axes into ndarray with `plt.subplots` squeeze semantics; return `(fig, axes)`.

**Returned axes shape.** Matches `plt.subplots(squeeze=True)`:
- `1×1` → bare `Axes`.
- `1×N` or `N×1` → 1-D ndarray.
- `N×M` → 2-D ndarray.

---

## Component 3 — `SubplotsAutoLayout` (draw-event hook)

**File:** `src/publiplots/layout/auto_layout.py`.

Per-figure draw-event listener. Measures decoration sizes from `ax.get_tightbbox()` and resizes the figure so declared `axes_size` stays exact. Re-runs on every draw so late `ax.set_title(...)` calls still work.

```python
class SubplotsAutoLayout:
    def __init__(self, fig, layout: FigureLayout, locked: set[str]):
        self._fig = fig
        self._layout = layout
        self._locked = locked
        self._updating = False
        self._cid = fig.canvas.mpl_connect("draw_event", self._on_draw)
        fig._publiplots_layout = layout
        fig._publiplots_auto_layout = self   # keeps ref alive, enables teardown
```

**Measurement.** For each side NOT in `_locked`, measure across all axes in the grid (not just outer edges — inner rows may have tall titles too). All measurements in display pixels, converted to mm via `px / fig.dpi * 25.4`:

- `title_space` ← max over every axes of `(tightbbox.y1 - axes_bbox.y1)`.
- `xlabel_space` ← max of `(axes_bbox.y0 - tightbbox.y0)`.
- `ylabel_space` ← max of `(axes_bbox.x0 - tightbbox.x0)`.
- `right` ← max of `(tightbbox.x1 - axes_bbox.x1)`.
- `hspace`, `wspace`, `outer_pad` — never remeasured (they're gaps / aesthetic breathing room, not decoration sizes).

**Update threshold.** A side only updates if `|new - current| >= 0.1 mm`. Prevents micro-oscillation from sub-pixel rounding on interactive redraws.

**Apply.**
1. Compute updated `FigureLayout` via `with_updated_reservations(**new_sides)`.
2. `fig.set_size_inches(W/25.4, H/25.4, forward=False)` — `forward=False` prevents a GUI resize loop on interactive backends.
3. For each axes in the grid: `ax.set_position(updated_layout.axes_position(r, c))`.
4. Do NOT call `fig.canvas.draw_idle()`. We're inside a `draw_event` already; matplotlib continues rendering with the updated positions.

**Re-entrance guard.** `self._updating = True` around the update block; bail out if already true. Matches the pattern in `LayoutReactor._updating` from PR #78.

**Interaction with `LayoutReactor`.** Both listen to `draw_event`. `SubplotsAutoLayout` must fire first (moves axes) so `LayoutReactor` re-anchors legends to the new positions. `pp.subplots()` registers `SubplotsAutoLayout` during construction — before any user code attaches a legend — so registration order gives correct firing order. No explicit priority system needed.

**Overhead.** One `get_tightbbox()` call per axes per draw. matplotlib already runs text layout during rendering, so the incremental cost is negligible. The threshold check keeps pan/zoom redraws from thrashing the figure size.

---

## rcParams

**File:** `src/publiplots/themes/rcparams.py` (add to `PUBLIPLOTS_RCPARAMS`).
**File:** `src/publiplots/themes/styles.py` (add to `NOTEBOOK_STYLE` and `PUBLICATION_STYLE`).

Seven new keys under `subplots.*`:

| Key | Notebook | Publication | Notes |
|---|---|---|---|
| `subplots.title_space` | 8 | 5 | 1-line title + gap (15pt ≈ 5.3 mm; 10pt ≈ 3.5 mm) |
| `subplots.xlabel_space` | 12 | 8 | tick labels (~4 mm) + axis label (~4 mm) + padding |
| `subplots.ylabel_space` | 14 | 10 | numeric ticks (~5 mm) + rotated label (~4 mm) + padding |
| `subplots.right` | 2 | 2 | breathing room past the right spine |
| `subplots.hspace` | 12 | 8 | between-row gap |
| `subplots.wspace` | 14 | 10 | between-col gap |
| `subplots.outer_pad` | 3 | 2 | figure outer margin |

These are *initial* values. Auto-measure refines them to what was actually drawn unless the user locked them at the call site.

---

## Exports

**File:** `src/publiplots/__init__.py`.

Add below the existing `legend_group` import:

```python
from publiplots.layout import subplots
```

`src/publiplots/layout/__init__.py` re-exports `subplots`, `FigureLayout`, and `SubplotsAutoLayout`.

---

## Testing

**File:** `tests/test_subplots.py`.

### FigureLayout (pure math, no matplotlib)

- `test_figure_layout_single_cell_size` — 1×1, known inputs → known `(W, H)`.
- `test_figure_layout_2x3_size_matches_formula` — exercises the full sum expression.
- `test_figure_layout_legend_column_adds_width_only` — `legend_column=N` only changes W, not H, not axes positions in mm.
- `test_figure_layout_axes_position_is_deterministic` — same inputs twice → identical tuples.
- `test_figure_layout_axes_positions_dont_overlap` — for a 2×3 grid, no two cell rectangles intersect.
- `test_figure_layout_with_updated_reservations_preserves_axes_size` — update `title_space` → `axes_size` field unchanged in returned copy.

### `pp.subplots()` API

- `test_subplots_scalar_axes_size_coerced_to_tuple` — `axes_size=30` → `(30, 30)`.
- `test_subplots_rejects_figsize_kwarg` — `TypeError` raised, message mentions `axes_size`.
- `test_subplots_disables_layout_engine` — returned fig has `layout_engine is None`.
- `test_subplots_squeeze_returns_scalar_for_1x1` — `nrows=ncols=1` → bare `Axes`.
- `test_subplots_returns_1d_array_for_single_row_or_col` — `1×3` → shape `(3,)`; `3×1` → shape `(3,)`.
- `test_subplots_returns_2d_array_for_grid` — `2×3` → shape `(2, 3)`.
- `test_subplots_attaches_figure_layout_to_fig` — `fig._publiplots_layout` is a `FigureLayout` instance.
- `test_subplots_validates_nrows_ncols_and_axes_size` — `nrows=0`, `axes_size=(0, 30)`, `axes_size=-5`, negative reservation → `ValueError`.
- `test_subplots_sharex_sharey_forwarded` — on a 2×3 grid, `sharex=True` makes every axes share x with (0,0); `sharex='row'` shares within each row but not across rows; `sharex='col'` shares within each column.

### `SubplotsAutoLayout` (integration, Agg backend)

- `test_auto_layout_resizes_figure_for_title` — small figure, add `suptitle`-sized `ax.set_title(...)`, draw, assert fig height grew.
- `test_auto_layout_preserves_axes_size_after_resize` — declared 50×30 mm, draw with a title, verify final `ax.get_position()` × fig size in mm ≈ 50×30 (tolerance 0.5 mm).
- `test_auto_layout_locked_side_not_remeasured` — pass explicit `title_space=20`, add a tiny title, draw, assert figure height didn't shrink.
- `test_auto_layout_no_hook_when_all_sides_locked` — pass explicit values for every side, assert no `draw_event` callback registered.
- `test_auto_layout_second_draw_no_change_within_threshold` — draw twice with same content, assert figure size stable (no oscillation).
- `test_auto_layout_works_with_legend_builder` — `pp.legend(ax)` on a `pp.subplots` axes, draw, assert legend anchor tracks the axes after the auto-resize.

Target: ~18 tests, ~1-2 s total runtime.

---

## Gallery migration

**File:** `examples/plots/plot_14_edgecolor_control.py`.

Two sections use `plt.subplots + fig.subplots_adjust`. Migrate both to `pp.subplots(legend_column=...)`.

**"Uniform Edges Across Plot Types" (1×3 grid):** replace
```python
fig, axes = plt.subplots(1, 3, figsize=(14, 3))
fig.subplots_adjust(left=0.05, right=0.88, wspace=0.25)
```
with
```python
fig, axes = pp.subplots(1, 3, axes_size=(45, 30), legend_column=30)
```

**"Raincloud" (1×2 grid, shared y):** replace
```python
fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
fig.subplots_adjust(left=0.06, right=0.85, wspace=0.1)
```
with
```python
fig, axes = pp.subplots(1, 2, axes_size=(55, 50), sharey=True, legend_column=30)
```

The `pp.legend_group(anchor=axes[-1])` call stays unchanged — nothing about the legend code path moves, the legend just lands in a pre-reserved column instead of whatever `subplots_adjust` left behind.

Acceptance: rendered gallery images (sphinx-gallery PNGs) visually equivalent to the pre-migration versions.

---

## Error handling

| Condition | Behavior |
|---|---|
| `axes_size` missing | `TypeError` (standard Python: required kwarg) |
| `axes_size` scalar negative/zero | `ValueError("axes_size must be positive")` |
| `axes_size` tuple with non-positive | `ValueError("axes_size must be positive")` |
| `nrows < 1` or `ncols < 1` | `ValueError("nrows and ncols must be >= 1")` |
| User-specified reservation < 0 | `ValueError("{name} must be non-negative")` |
| `figsize` in `fig_kw` | `TypeError("use axes_size; figure size is computed from axes + reservations")` |
| `layout` / `constrained_layout` / `tight_layout` in `fig_kw` | `UserWarning("publiplots manages layout; ignoring {kwarg}")`, forced to `layout=None` |
| Decorations exceed locked reservation | No exception, no auto-grow. Figure renders with clipped/overlapping decoration. (User locked it; we trust them.) |

---

## Interactions with existing code

- **`LegendBuilder` / `LayoutReactor` (PR #78).** No changes. Both continue to work as-is on `pp.subplots()` figures. `LayoutReactor` will re-anchor legends after `SubplotsAutoLayout` resizes axes on first draw.
- **`pp.legend_group`.** No changes. `legend_column > 0` on `pp.subplots` reserves right-side space; user still explicitly calls `pp.legend_group(anchor=axes[-1, -1])` or equivalent to attach a unified legend.
- **`set_notebook_style` / `set_publication_style`.** Unchanged contract — still don't force `constrained_layout`. The new `subplots.*` rcParams are added to both style dicts.

---

## Decisions already made — do NOT re-litigate during implementation

- mm is the unit. `figsize` is rejected. Conversion to inches happens only at the matplotlib boundary.
- No `_mm` suffix on field names (except `*_space` names which avoid conflict with matplotlib's `title` / `xlabel` / `ylabel` string-valued kwargs).
- Auto-measurement on every draw. `None` reservations auto-measure; float reservations lock.
- No `pp.figure()` in this PR.
- No GridSpec / heterogeneous cells in this PR (composer's job, PR #80+).
- No legend collision warning, no `plot_15_legends.py`, no `legend_column` auto-sizing in this PR (next follow-up PR).
- `legend_column` is opt-in (explicit value), never auto-measured.
- `pp.subplots()` always sets `layout=None`; layout-engine kwargs in `fig_kw` are ignored with warning.
- Every axes in the grid has the same `axes_size`. No per-cell overrides.
- `fig._publiplots_layout` is attached for composer consumption (public-ish but underscore-prefixed until composer ships).

---

## Open questions for implementation

None. All API decisions are locked in this spec.

---

## Files touched

**Create:**
- `src/publiplots/layout/__init__.py`
- `src/publiplots/layout/figure_layout.py`
- `src/publiplots/layout/auto_layout.py`
- `src/publiplots/layout/subplots.py`
- `tests/test_subplots.py`

**Modify:**
- `src/publiplots/__init__.py` (add `subplots` export)
- `src/publiplots/themes/rcparams.py` (add 7 `subplots.*` keys)
- `src/publiplots/themes/styles.py` (add 7 defaults to both style dicts)
- `examples/plots/plot_14_edgecolor_control.py` (migrate 2 sections)

---

## Handoff to the composer PR (#80+)

This PR attaches `fig._publiplots_layout = layout` on every `pp.subplots()` figure. The composer will:
- Consume that attribute to know each panel's axes rectangles in figure-fractional coordinates.
- Align axes rectangles across panels on a shared page grid (same "fixed axes, flexible canvas" idea, one level up).
- Grow the page (letter / A4 / custom) as needed to fit the declared panel sizes.
- Handle heterogeneity at the page level (panel A is 50×30, panel B is 80×50, etc.).

GridSpec-style heterogeneous cells inside a single `pp.subplots()` call are explicitly the composer's territory, not `pp.subplots()`'s.
