# Spec Amendment — Per-row/Per-column Reservations (PR #79 addendum)

**Status:** approved, implementation pending.
**Amends:** `docs/superpowers/specs/2026-04-30-pp-subplots-design.md`.
**Motivated by:** gallery output bug — `axes_size=(45, 30)` 1×3 grid with a `legend_group` on the rightmost axes caused every column to reserve 33 mm of `right` space because `SubplotsAutoLayout._measure()` takes a grid-wide `max`, and the legend attached to column 2 propagated its width onto columns 0 and 1. Empty phantom gutters; wrong figure.

---

## What changes

Two coupled changes, shipped together:

1. **Reservations become per-row or per-column** instead of grid-wide.
2. **Legend/colorbar overlays managed by `LayoutReactor` are excluded from tightbbox measurement.**

Together they produce correct geometry for heterogeneous decorations (the normal case in publication figures) without double-counting external overlays like `pp.legend_group`.

## Scope

- `src/publiplots/layout/figure_layout.py` — `FigureLayout` fields become tuples for the 4 per-side reservations. `figure_size()` and `axes_position()` updated to per-row/per-column math.
- `src/publiplots/layout/auto_layout.py` — `SubplotsAutoLayout._measure()` returns position-indexed values; `_apply()` passes tuples to `with_updated_reservations`. Also excludes `LayoutReactor`-managed artists from tightbbox.
- `src/publiplots/layout/subplots.py` — user reservation kwargs accept scalar OR tuple; scalar is broadcast to the correct length.
- `tests/test_subplots.py` — new tests for per-row/per-column behavior and for the legend-exclusion path. Existing tests continue to pass (scalar kwargs still supported).

Out of scope: no changes to `LegendBuilder` public API, no changes to `LegendLayout`, no new rcParams.

---

## Data model

### `FigureLayout` fields

Per-side reservations become tuples indexed by position:

```python
@dataclass
class FigureLayout:
    nrows: int
    ncols: int
    axes_size: tuple[float, float]

    # Per-position reservations (mm). Tuple lengths: nrows for row-axis
    # reservations, ncols for col-axis reservations.
    title_space:  tuple[float, ...]   # length nrows — above each row
    xlabel_space: tuple[float, ...]   # length nrows — below each row
    ylabel_space: tuple[float, ...]   # length ncols — left of each col
    right:        tuple[float, ...]   # length ncols — right of each col

    # Gaps and outer margin stay scalar.
    hspace: float
    wspace: float
    outer_pad: float
    legend_column: float
```

Invariants:
- `len(title_space) == len(xlabel_space) == nrows`.
- `len(ylabel_space) == len(right) == ncols`.
- `axes_size` still inviolate; `with_updated_reservations(axes_size=...)` still raises.
- All tuple elements non-negative.

### `figure_size()` formula

```python
def figure_size(self) -> tuple[float, float]:
    w_ax, h_ax = self.axes_size
    W = (
        self.outer_pad
        + sum(self.ylabel_space) + self.ncols * w_ax + sum(self.right)
        + max(self.ncols - 1, 0) * self.wspace
        + self.legend_column
        + self.outer_pad
    )
    H = (
        self.outer_pad
        + sum(self.title_space) + self.nrows * h_ax + sum(self.xlabel_space)
        + max(self.nrows - 1, 0) * self.hspace
        + self.outer_pad
    )
    return W, H
```

When every element of a tuple is equal to `v`, the per-row/per-column formula collapses to the scalar formula from the original spec (`ncols * v` ≡ `sum([v]*ncols)`). Backwards-compatible for uniform reservations.

### `axes_position(row, col)` formula

```python
def axes_position(self, row: int, col: int) -> tuple[float, float, float, float]:
    W, H = self.figure_size()
    w_ax, h_ax = self.axes_size

    # Accumulate offsets of preceding columns (each contributes
    # ylabel_space[c] + w_ax + right[c] + wspace).
    x0_mm = self.outer_pad
    for c in range(col):
        x0_mm += self.ylabel_space[c] + w_ax + self.right[c] + self.wspace
    x0_mm += self.ylabel_space[col]

    # Accumulate offsets from bottom upward. rows_below = nrows - 1 - row.
    y0_mm = self.outer_pad
    for r in range(self.nrows - 1, row, -1):
        y0_mm += self.xlabel_space[r] + h_ax + self.title_space[r] + self.hspace
    y0_mm += self.xlabel_space[row]

    return (x0_mm / W, y0_mm / H, w_ax / W, h_ax / H)
```

Row 0 is still the top row (matches matplotlib axes-array convention).

### `with_updated_reservations`

Accepts the same 4 tuple-valued fields plus scalar gaps. Rejects `axes_size` override (unchanged). No scalar→tuple broadcasting here — callers pass tuples; the coercion happens at the public API boundary.

---

## `pp.subplots()` API — scalar/tuple coercion

User-facing kwargs stay ergonomic. Each per-side reservation accepts:
- `None` → auto-measure (broadcast to rcParams default as initial value).
- scalar `float` → broadcast to the correct length, locks all positions.
- `tuple[float, ...]` → per-position, must match `nrows` (for title/xlabel) or `ncols` (for ylabel/right). Locks.

Validation:
- Reject scalar negatives and any negative tuple element (`ValueError` mentions the field name).
- Reject wrong-length tuples (`ValueError` mentions the field name and expected length).

Example:
```python
# scalar → uniform
pp.subplots(2, 3, axes_size=(50, 30), title_space=8)

# per-row explicit
pp.subplots(2, 3, axes_size=(50, 30), title_space=(12, 6))

# mixed: title per-row, xlabel auto-measured
pp.subplots(2, 3, axes_size=(50, 30), title_space=(12, 6))
```

The `locked` set tracked in `pp.subplots()` becomes `locked_positions`: a dict `{side: set_of_locked_positions}` rather than a flat set of side names. `SubplotsAutoLayout` uses this to skip remeasurement only for the specific `(side, position)` pairs the user locked — while still remeasuring unlocked positions on the same side.

Simpler alternative: track `locked_sides_fully` (only when user gave a scalar or complete tuple) and leave partially-locked side remeasurement as a future refinement. My pick: **fully-locked sides only.** If a user wants per-position lock-some-measure-others, they can pass explicit rcParams-default values for the ones they want to keep. Avoids a combinatorial API.

Final API contract:
- Scalar reservation → "lock all positions on this side at this value."
- Tuple reservation → "lock all positions on this side at these values."
- `None` → "auto-measure this side on every draw; initialize from rcParams default (broadcast)."

---

## `SubplotsAutoLayout._measure()` changes

Per side, measure across each row or column INDEPENDENTLY instead of grid-wide max.

```python
def _measure(self) -> dict:
    fig = self._fig
    dpi = fig.dpi
    if dpi <= 0:
        return {}
    axes_matrix = self._axes_matrix()

    # Exclude LayoutReactor-managed artists from tightbbox. These are
    # external overlays (legend_group, standalone colorbars attached via
    # pp.legend(ax)) whose geometry is tracked independently by the
    # reactor, not by axis-level reservations. Preserves the semantic
    # line between "axis decoration" and "external overlay."
    managed_artists = self._reactor_managed_artists()
    # Temporarily remove these from layout consideration for the tightbbox
    # pass, then restore. (Using set_in_layout(False)/set_in_layout(True).)
    # See _tight_bbox_excluding_managed below.

    measured: dict = {}

    if "title_space" not in self._locked_sides:
        per_row = []
        for row in axes_matrix:
            max_px = 0.0
            for ax in row:
                ax_bbox = ax.get_window_extent()
                tight = self._tight_bbox_excluding_managed(ax, managed_artists)
                if tight is None:
                    continue
                max_px = max(max_px, tight.y1 - ax_bbox.y1)
            per_row.append(max(max_px / dpi * 25.4, 0.0))
        measured["title_space"] = tuple(per_row)

    # xlabel_space — per row, max over axes in that row, bottom edge.
    # ylabel_space — per column, max over axes in that column, left edge.
    # right — per column, max over axes in that column, right edge.
    # Same shape as above, just different bbox arithmetic and axis iteration.

    return measured

def _reactor_managed_artists(self) -> set:
    """IDs of artists registered with LayoutReactor on this figure."""
    reactor = getattr(self._fig, "_publiplots_layout_reactor", None)
    if reactor is None:
        return set()
    return {id(reg.artist) for reg in reactor._registrations}

def _tight_bbox_excluding_managed(self, ax, managed_artist_ids):
    """Return ax.get_tightbbox() with reactor-managed children excluded.

    Implementation: toggle set_in_layout(False) on matching children for
    the duration of the call, then restore. If no managed artists are
    present on this axes, delegate directly to get_tightbbox().
    """
    excluded = []
    for child in ax.get_children():
        if id(child) in managed_artist_ids and child.get_in_layout():
            child.set_in_layout(False)
            excluded.append(child)
    try:
        return ax.get_tightbbox()
    finally:
        for child in excluded:
            child.set_in_layout(True)
```

Note: the `_SIDE_CALCULATORS` lookup table from the Task 3 refactor is retained but each lambda's output is now a per-row or per-column list, not a single scalar. The DRY refactor becomes:

```python
# side_name -> (axis_kind, bbox_fn)
#   axis_kind: "row" (iterate rows, max over cells in row) or
#              "col" (iterate cols, max over cells in col)
_SIDE_CALCULATORS = {
    "title_space":  ("row", lambda ax_bb, t: t.y1 - ax_bb.y1),
    "xlabel_space": ("row", lambda ax_bb, t: ax_bb.y0 - t.y0),
    "ylabel_space": ("col", lambda ax_bb, t: ax_bb.x0 - t.x0),
    "right":        ("col", lambda ax_bb, t: t.x1 - ax_bb.x1),
}
```

### Update threshold

Unchanged logic: `abs(new - current) >= _UPDATE_THRESHOLD_MM`. Applied ELEMENT-WISE across the tuple; any element exceeding the threshold triggers an update. The whole tuple gets replaced atomically (no partial updates — simplicity).

---

## rcParams

No new keys. The existing 7 keys stay as the scalar default that gets broadcast to a tuple in `pp.subplots()`:

```python
# in pp.subplots(), when resolving a None reservation:
initial_scalar = pp.rcParams[f"subplots.{name}"]
axis_kind = "row" if name in ("title_space", "xlabel_space") else "col"
length = nrows if axis_kind == "row" else ncols
initial_tuple = (float(initial_scalar),) * length
```

---

## Backwards compatibility within PR #79

All 36 subplots tests added on the branch pass scalar reservations. Scalars still work via broadcast. A handful of the tests read individual scalar fields (`layout.title_space == 5`); those need one-line updates to read `layout.title_space[0]` or check the tuple.

---

## Testing

### New tests in `tests/test_subplots.py`

#### FigureLayout tuple math (pure geometry)

- `test_figure_layout_per_row_title_space_changes_position`
  Build a 2×1 layout with `title_space=(20.0, 5.0)`. Assert row 0's y0 is HIGHER than if `title_space=(5.0, 5.0)` was used — the tall top-row title pushes its axes down.

- `test_figure_layout_per_col_right_is_cumulative`
  Build a 1×3 layout with `right=(2.0, 2.0, 30.0)`. Figure width should equal the 1×3 uniform-`right=2` width plus 28 mm (the extra on the last column), not plus 84 mm (3×28 as grid-wide max would imply).

- `test_figure_layout_scalar_broadcast_equivalence`
  For `title_space=(8.0, 8.0)` vs `title_space=8.0` broadcast at the API boundary, assert `figure_size()` and `axes_position(r, c)` match exactly for all (r, c).

- `test_figure_layout_with_updated_reservations_accepts_tuples`
  Build a layout, call `with_updated_reservations(title_space=(12.0, 6.0))`, assert the new layout has the right tuple and scalar fields unchanged.

- `test_figure_layout_wrong_length_tuple_rejected`
  `FigureLayout(nrows=2, ..., title_space=(5.0, 5.0, 5.0))` → `ValueError` mentioning "title_space" and length mismatch. (Validation happens in `__post_init__` since `FigureLayout` now has structural constraints.)

#### SubplotsAutoLayout — per-position measurement

- `test_auto_layout_right_is_per_column_not_grid_wide`
  1×3 layout with a small legend-like artist attached via `ax.add_artist` on the third axes only. After `SubplotsAutoLayout` resize, `layout.right[0]` and `layout.right[1]` stay near baseline; `layout.right[2]` reflects the added artist.

- `test_auto_layout_title_space_is_per_row`
  2×1 layout. Set a 3-line title on row 0's axes, no title on row 1. After resize, `layout.title_space[0] > layout.title_space[1]` by ~2 line-heights.

- `test_auto_layout_excludes_reactor_managed_artists`
  Build a `pp.subplots()` + attach a `pp.legend_group(anchor=axes[-1])`. Draw. Assert `layout.right[-1]` is close to the baseline `right` (e.g., < 5 mm), NOT inflated to the legend's pixel width. The `legend_column` reservation is the one handling legend width.

- `test_auto_layout_per_axis_pp_legend_IS_counted`
  Build a `pp.subplots(1, 2, ...)` and attach a per-axis `pp.legend(ax=axes[0])` (NOT `legend_group`). Draw. Assert `layout.right[0]` grew to include the legend's width — per-axis legends are semantically part of their axes' footprint.

  **Note:** this requires the implementation to distinguish `pp.legend` (keep in tightbbox) from `pp.legend_group` (exclude from tightbbox). The discriminator: the `LayoutReactor._Registration` struct needs a flag (e.g., `external_to_axis: bool`) set to True only by `legend_group`. `pp.legend(ax)` registrations keep it False.

#### `pp.subplots()` API — tuple coercion & validation

- `test_subplots_scalar_reservation_locks_all_rows`
  `pp.subplots(2, 3, axes_size=(50, 30), title_space=8)` → `fig._publiplots_layout.title_space == (8.0, 8.0)`.

- `test_subplots_tuple_reservation_preserved`
  `pp.subplots(2, 3, axes_size=(50, 30), title_space=(12, 6))` → `fig._publiplots_layout.title_space == (12.0, 6.0)`.

- `test_subplots_wrong_length_tuple_raises`
  `pp.subplots(2, 3, axes_size=(50, 30), title_space=(12, 6, 3))` → `ValueError` mentioning "title_space" and "length".

- `test_subplots_ylabel_space_tuple_must_match_ncols`
  `pp.subplots(2, 3, axes_size=(50, 30), ylabel_space=(10, 10))` → `ValueError`.

- `test_subplots_default_rcparams_broadcast_to_tuple`
  No user-provided reservations → `fig._publiplots_layout.title_space == (default,) * nrows`.

### Existing test updates

Two existing tests read scalar reservation fields; update them to read element 0 of the tuple (or equivalent assertion on the tuple):

- `test_subplots_rcparams_publication_defaults` — no change needed (reads from `pp.rcParams`, not the layout object).
- `test_figure_layout_with_updated_reservations_preserves_axes_size` — update `updated.title_space == 20.0` to `updated.title_space == (20.0,)` (nrows=1) and similar for xlabel_space.
- `test_auto_layout_grows_title_space_for_title` — update `reactor._layout.title_space > 1.0` to `reactor._layout.title_space[0] > 1.0`.
- `test_auto_layout_locked_side_not_remeasured` — update `reactor._layout.title_space == 25.0` to `reactor._layout.title_space == (25.0,)`.

Everything else on the branch stays unchanged because `pp.subplots` API still accepts scalars.

---

## Risks

- **`set_in_layout(False)` side effects.** Toggling this during measurement could race with matplotlib internals if a subsequent draw hooks in before we restore. Mitigated by the `self._updating` re-entrance guard — we're already inside one draw when we do this, and matplotlib doesn't issue nested draw_event callbacks.
- **`LayoutReactor._registrations` is a private API.** We read it from `SubplotsAutoLayout._measure()`. Acceptable coupling — both modules live in publiplots and both are internal. Add a comment and consider exposing `LayoutReactor.managed_artist_ids()` as a stable internal API in a later tidy.
- **Per-axis `pp.legend` vs `legend_group` discriminator.** Requires a small addition to `_Registration` and the `legend_group` factory. Low risk; PR #78 already owns this file.

---

## What stays decided from the original spec

- `axes_size` is inviolate; figure grows around it.
- Auto-measure on every draw; user-locked sides skip remeasurement.
- Bidirectional updates (abs threshold 0.1 mm).
- `hspace` / `wspace` / `outer_pad` stay scalar; never auto-measured.
- `legend_column` stays scalar, opt-in, never auto-measured.
- No `pp.figure()`.
- No GridSpec / heterogeneous `axes_size` across cells — still deferred to composer PR.
