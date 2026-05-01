# Design — Legend Auto-Collection and Width-Aware Sizing (PR #81)

**Status:** approved, awaiting implementation plan.
**Worktree:** `.worktrees/legend-positioning` on `feat/legend-positioning`.
**Follows:** PR #79 (`pp.subplots`), PR #80 (drop notebook style).
**Blocks:** Composer PR (axes-rectangle snapping, noted in
`docs/superpowers/handoff/2026-04-30-fixed-axes-pp-subplots.md`).

---

## Goal

Two orthogonal pieces shipped together:

1. **Auto-collect.** `pp.legend_group(anchor=...)` walks the grid, collects
   per-axes legend entries, dedups, and renders a single unified legend.
   Users no longer construct `handles=[...]` manually or call
   `group.add_legend(handles=...)` for the common case.
2. **Width-awareness.** `SubplotsAutoLayout` measures the rendered group
   width and grows the figure's `legend_column` reservation to fit. The
   `legend_column=` kwarg is dropped from `pp.subplots()` — the reservation
   is always automatic.

After this PR the common shared-legend flow becomes:

```python
fig, axes = pp.subplots(1, 3)
pp.scatterplot(..., ax=axes[0])
pp.scatterplot(..., ax=axes[1])
pp.scatterplot(..., ax=axes[2])
pp.legend_group(anchor=axes[-1])
```

No handle construction, no `legend_column` guess, no `legend=False` on
individual plots — the group handles everything.

## Philosophy

publiplots already treats `pp.rcParams["subplots.*"]` values as the single
baseline and auto-measures the four per-side reservations on draw.
`legend_column` was the one remaining hand-tuned mm value. This PR
eliminates the last guess.

The `legend=` kwarg on plot functions grows from `bool` to `bool | dict`
for per-kind control (hue / size / style / marker). Existing `True` / `False`
semantics are unchanged; the dict form is additive.

## Non-goals

- Inside-the-axes legend placement (e.g., `loc="upper right"` inside the
  spines). Filed as a potential follow-up — users with that need can call
  matplotlib's `ax.legend()` directly today.
- Additional legend kinds beyond `hue`, `size`, `style`, `marker`.
- Composer / cross-figure axes-rectangle snapping (captured in the
  PR #79 handoff).
- Changing `pp.legend(ax)` behavior. `pp.legend(ax)` stays as today: reads
  stashed entries from `ax`, renders on the right side with
  `external_to_axis=False`, grows the column's `right` reservation.
- Auto-detection that would turn any figure with multiple legends into
  a `legend_group`. Explicit opt-in only — user calls
  `pp.legend_group(...)`.

---

## Architecture

Three units:

1. **`LegendEntry` + per-axes storage** (new `legend_entries.py`). A
   dataclass that every plot function writes to `ax._publiplots_legend_entries`.
   Records `(name, kind, handles, labels, signature)`.
2. **`MultiAxesLegendGroup` auto-collect** (extend existing). Walks
   `fig._publiplots_axes`, collects stashed entries, dedups by
   `(name, kind)` + signature, filters by optional `collect=[...]`,
   renders via the existing `LegendBuilder` dispatch.
3. **`SubplotsAutoLayout` width-awareness** (extend existing). Measures
   the legend_group's rendered right-extent in mm on each draw, updates
   `FigureLayout.legend_column` via the existing
   `with_updated_reservations` path.

```
plot function (pp.barplot / pp.scatterplot / ...)
  ├─► resolve legend=True/False/dict → per-kind include map
  ├─► for each enabled kind: stash_entry(ax, LegendEntry.build(...))
  └─► for each enabled kind: if not entry_is_in_group(fig, entry):
                                render per-axis via pp.legend(ax)

pp.legend_group(anchor=..., collect=None|[names])
  ├─► attaches to fig._publiplots_legend_group
  ├─► on first draw: _materialize() walks grid, dedups, filters, renders
  └─► subsequent draws: _materialized flag prevents re-collecting

SubplotsAutoLayout._measure()
  ├─► existing: four per-side tightbbox measurements
  └─► NEW: _measure_legend_column() returns group right-extent mm
       (calls group._materialize() to ensure artists exist)

SubplotsAutoLayout._apply() → FigureLayout.with_updated_reservations(
    **per_side_tuples, legend_column=<new_mm>
) — existing API
```

Both pieces compose but are independent. Width-awareness works even
without auto-collect (user calls manual `group.add_legend(...)`).
Auto-collect works without width-awareness (group renders inside the
`right` reservation — ugly but functional).

---

## Component 1 — `LegendEntry` + storage

**File:** `src/publiplots/utils/legend_entries.py` (new).

```python
from dataclasses import dataclass
import hashlib


_LEGEND_KINDS = ("hue", "size", "style", "marker")


@dataclass(frozen=True)
class LegendEntry:
    """A single stashed legend entry on an axes.

    Plot functions write these to ``ax._publiplots_legend_entries`` so
    pp.legend(ax) and pp.legend_group(anchor=ax) can render them.
    """
    name: str                  # variable name, e.g. 'treatment'
    kind: str                  # one of _LEGEND_KINDS
    handles: tuple             # matplotlib-compatible handles
    labels: tuple              # tuple[str, ...]
    signature: str             # for dedup in legend_group

    @classmethod
    def build(cls, name, kind, handles, labels) -> "LegendEntry":
        return cls(
            name=name, kind=kind,
            handles=tuple(handles),
            labels=tuple(labels),
            signature=_hash_handles(handles, labels),
        )


def _hash_handles(handles, labels) -> str:
    parts = []
    for h, lab in zip(handles, labels):
        parts.append(type(h).__name__)
        parts.append(str(lab))
        for attr in ("get_facecolor", "get_marker", "get_markersize",
                     "get_linewidth"):
            fn = getattr(h, attr, None)
            if fn is not None:
                try:
                    parts.append(repr(fn()))
                except Exception:
                    pass
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:12]


def stash_entry(ax, entry: LegendEntry) -> None:
    """Append an entry to ax._publiplots_legend_entries.

    Creates the list attribute on first call. Order is preserved —
    later calls append."""
    existing = getattr(ax, "_publiplots_legend_entries", None)
    if existing is None:
        existing = []
        ax._publiplots_legend_entries = existing
    existing.append(entry)


def get_entries(ax) -> list[LegendEntry]:
    """Return the ordered list of entries stashed on an axes."""
    return list(getattr(ax, "_publiplots_legend_entries", []))


def resolve_legend_flags(legend) -> dict[str, bool]:
    """Convert legend= (bool | dict) to a per-kind include map.

    - True  → {k: True for k in kinds}
    - False → {k: False for k in kinds}
    - dict  → as given; missing keys default to True
    """
    if legend is True:
        return {k: True for k in _LEGEND_KINDS}
    if legend is False:
        return {k: False for k in _LEGEND_KINDS}
    if isinstance(legend, dict):
        return {k: bool(legend.get(k, True)) for k in _LEGEND_KINDS}
    raise TypeError(
        f"legend must be bool or dict[str, bool], got {type(legend).__name__}"
    )


def entry_is_in_group(fig, entry: LegendEntry) -> bool:
    """True if the figure's legend_group (if any) claims this entry."""
    group = getattr(fig, "_publiplots_legend_group", None)
    if group is None:
        return False
    return group.claims(entry.name)
```

**Invariants:**
- `LegendEntry.build(...)` produces deterministic signatures for identical
  inputs.
- `stash_entry` is idempotent-safe only in the sense that it appends;
  duplicate calls add duplicate entries. Dedup happens at collection time
  in the group.
- Handle signatures include color / marker / size / linewidth so
  palette differences across axes are detectable.

---

## Component 2 — `MultiAxesLegendGroup` auto-collect

**File:** `src/publiplots/utils/legend_group.py` (modify).

### API surface

```python
def __init__(
    self,
    anchor: Axes,
    collect: list[str] | None = None,   # NEW
    x_offset: float = 2,
    y_offset: Optional[float] = None,
    gap: float = 2,
    column_spacing: float = 5,
    vpad: float = 5,
    max_width: Optional[float] = None,
):
```

- `collect=None` → collect every `(name, kind)` pair across all grid
  axes (dedup by signature, first-wins on mismatch).
- `collect=['treatment', 'category']` → filter to those names only,
  render in the listed order.

Existing `add_legend(handles, label, ax=...)` and `add_colorbar(...)`
stay. Manual entries render BEFORE auto-collected ones.

### Constructor validation

`collect` accepts `None` or a sequence (list/tuple) of strings. A bare
string like `collect='treatment'` is rejected with `TypeError` to
prevent a silent bug where `'x' in 'treatment'` would accidentally
match substrings. Validation goes in `__init__`:

```python
if collect is not None:
    if isinstance(collect, str) or not hasattr(collect, "__iter__"):
        raise TypeError(
            "collect must be None or a list/tuple of names; "
            "got a bare string. Wrap in a list: collect=['name']"
        )
    collect = list(collect)
self._collect = collect
```

### Figure registration

`pp.legend_group(...)` now sets `fig._publiplots_legend_group = group`.
Plot functions consult this during rendering decisions.

### Claims check

```python
def claims(self, name: str) -> bool:
    """True if the group will render an entry with this name."""
    if self._collect is None:
        return True
    return name in self._collect
```

### `_materialize()` — the collection pass

Runs on first draw (called from `SubplotsAutoLayout._measure_legend_column`):

```python
def _materialize(self) -> None:
    if self._materialized:
        return
    self._materialized = True

    fig = self.anchor.get_figure()
    axes_matrix = getattr(fig, "_publiplots_axes", None)
    if axes_matrix is None:
        return   # not a pp.subplots figure — nothing to auto-collect

    seen: dict[tuple[str, str], LegendEntry] = {}
    order: list[tuple[str, str]] = []
    for row in axes_matrix:
        for ax in row:
            for entry in get_entries(ax):
                if self._collect is not None and entry.name not in self._collect:
                    continue
                key = (entry.name, entry.kind)
                if key in seen:
                    if seen[key].signature != entry.signature and not self._warned_mismatch:
                        warnings.warn(
                            f"legend entry {entry.name!r} ({entry.kind}) "
                            "differs between axes; group uses first occurrence",
                            UserWarning, stacklevel=2,
                        )
                        self._warned_mismatch = True
                    continue
                seen[key] = entry
                order.append(key)

    if self._collect is not None:
        order.sort(key=lambda k: (self._collect.index(k[0]), k[1]))

    for key in order:
        self._render_entry(seen[key])


def _render_entry(self, entry: LegendEntry) -> None:
    """Route to add_legend (categorical) or add_colorbar (continuous)."""
    if entry.kind == "hue" and _is_continuous(entry.handles):
        mappable = entry.handles[0]
        self.add_colorbar(mappable=mappable, label=entry.name)
    else:
        self.add_legend(
            handles=list(entry.handles),
            label=entry.name,
            labels=list(entry.labels),
        )


def _is_continuous(handles) -> bool:
    """Continuous hue stashes a ScalarMappable as the first handle."""
    from matplotlib.cm import ScalarMappable
    return len(handles) >= 1 and isinstance(handles[0], ScalarMappable)
```

### Idempotence

`_materialized` flag guards against duplicate collection on repeated
draws. `SubplotsAutoLayout`'s settlement loop may call `_materialize`
multiple times; only the first has an effect.

### Manual + auto interaction

If the user calls `group.add_legend(handles=..., label=...)` explicitly
before any draw, those entries land in `group._builder.elements` first.
`_materialize()` runs after and appends auto-collected entries. No
dedup between manual and auto (different signatures by construction —
the user built different handles).

---

## Component 3 — `SubplotsAutoLayout` width-awareness

**File:** `src/publiplots/layout/auto_layout.py` (modify).

### `_ALL_SIDES` extension

```python
_ALL_SIDES = {
    "title_space", "xlabel_space", "ylabel_space", "right",
    "legend_column",   # NEW
}
```

`legend_column` can be added to the locked set if user provides a value
(we don't expose that kwarg, but the lock-set machinery handles it
uniformly).

### `_measure_legend_column()`

```python
def _measure_legend_column(self) -> float:
    """mm width past the anchor axes' right edge, plus 1 mm padding."""
    group = getattr(self._fig, "_publiplots_legend_group", None)
    if group is None:
        return 0.0

    # Force collection so artists exist to measure.
    group._materialize()
    if not group._builder.elements:
        return 0.0

    dpi = self._fig.dpi
    anchor_bb = group.anchor.get_window_extent()
    max_x1 = anchor_bb.x1
    for kind, obj in group._builder.elements:
        extent = _artist_window_extent(obj)
        if extent is None:
            continue
        max_x1 = max(max_x1, extent.x1)
    overhang_px = max_x1 - anchor_bb.x1
    if overhang_px <= 0:
        return 0.0
    return overhang_px / dpi * 25.4 + 1.0


def _artist_window_extent(obj):
    if hasattr(obj, "get_window_extent"):
        return obj.get_window_extent()
    if hasattr(obj, "ax"):   # Colorbar stores on .ax
        return obj.ax.get_window_extent()
    return None
```

### `_measure()` integration

```python
def _measure(self) -> Dict[str, ...]:
    # ... existing per-side tuple measurements ...

    if "legend_column" not in self._locked:
        measured["legend_column"] = self._measure_legend_column()
    return measured
```

### `_needs_update()` — handle scalar + tuples

```python
def _needs_update(self, measured) -> bool:
    for side, new_val in measured.items():
        current = getattr(self._layout, side)
        if side == "legend_column":
            if abs(new_val - current) >= _UPDATE_THRESHOLD_MM:
                return True
        else:
            # tuple comparison — existing code path
            if len(new_val) != len(current):
                return True
            for nv, cv in zip(new_val, current):
                if abs(nv - cv) >= _UPDATE_THRESHOLD_MM:
                    return True
    return False
```

### `_apply()` — unchanged

`FigureLayout.with_updated_reservations` already accepts arbitrary
field names. Passing `legend_column=X` alongside tuple reservations
works without code change.

### Convergence

Settle loop (existing `settle()` wrapper from PR #80) handles this
automatically. Typical sequence:

1. Draw 1: no artists yet; `_measure_legend_column` calls
   `_materialize` which creates artists. Artists positioned at old
   `legend_column=0` (falling off the right edge). Measured overhang =
   full group width.
2. `_apply` grows `FigureLayout.legend_column` → `fig.set_size_inches`.
3. Draw 2: artists re-position (LayoutReactor re-anchors on draw),
   measurement matches current → converged.

`_MAX_CONVERGENCE_ITERS=5` cap accommodates the two-pass settlement.

### `pp.subplots()` kwarg drop

```python
# REMOVED from signature:
legend_column: float = 0.0,
```

If user passes `legend_column`:

```python
if "legend_column" in fig_kw:
    raise TypeError(
        "pp.subplots() no longer accepts legend_column. "
        "Attach a pp.legend_group(anchor=...) to the figure; the column "
        "is auto-sized."
    )
```

Gallery migration removes the two `legend_column=30` call sites in
`plot_14_edgecolor_control.py`.

---

## Component 4 — plot function integration

Each of nine plot files (`bar.py`, `box.py`, `point.py`, `scatter.py`,
`strip.py`, `swarm.py`, `violin.py`, `raincloud.py`, `heatmap.py`) adopts
the standardized stash+render pattern.

### The pattern

```python
from publiplots.utils.legend_entries import (
    resolve_legend_flags, stash_entry, get_entries, LegendEntry,
    entry_is_in_group,
)

def barplot(..., legend=True, ...):
    # ... plotting logic ...

    flags = resolve_legend_flags(legend)

    # Stash per kind — only for kinds the plot actually used
    if hue is not None and flags["hue"]:
        handles, labels = _build_hue_handles(...)
        stash_entry(ax, LegendEntry.build(
            name=hue, kind="hue",
            handles=handles, labels=labels,
        ))
    if size is not None and flags["size"]:
        # similar
        ...

    # Render per-axis — skip entries claimed by a legend_group
    fig = ax.get_figure()
    if any(flags.values()):
        _render_stashed_on_axes(ax, fig, flags)


def _render_stashed_on_axes(ax, fig, flags):
    from publiplots.utils.legend import legend as pp_legend
    entries_to_render = [
        e for e in get_entries(ax)
        if flags[e.kind] and not entry_is_in_group(fig, e)
    ]
    if not entries_to_render:
        return
    builder = pp_legend(ax=ax, auto=False)
    for entry in entries_to_render:
        if entry.kind == "hue" and _is_continuous(entry.handles):
            builder.add_colorbar(mappable=entry.handles[0], label=entry.name)
        else:
            builder.add_legend(
                handles=list(entry.handles),
                label=entry.name,
                labels=list(entry.labels),
            )
```

### Old ad-hoc stash removal

Plots currently write to various attributes:

- `ax.collections[0]._legend_data = ...` (scatter)
- `ax._hue_handles = ...` / similar (bar, box)
- per-plot dict on `ax`

These are removed. `pp.legend(ax)` auto-mode reads from
`ax._publiplots_legend_entries` only. A migration grep confirms no
other internal code reads the old attributes.

### Per-plot specifics

- **Categorical hue** (most plots): handles are Rectangle/MarkerPatch
  with palette colors; labels are category strings.
- **Continuous hue** (scatter with numeric column + `hue_norm`):
  handles are `[ScalarMappable]` (length 1); labels are `[]`. Router
  in `_materialize` / `_render_stashed_on_axes` detects this via
  `_is_continuous()`.
- **Size** (scatter, heatmap bubble): handles are MarkerPatches with
  varying `markersize`; labels are numeric buckets.
- **Style / marker** (scatter): handles are MarkerPatches with
  different markers; labels are level names.

The existing per-plot code that builds these handles is reused — we
only change where they land (stash vs. direct render).

---

## rcParams

No new keys. The existing `subplots.*` reservations continue to drive
defaults for the four per-side tuples. `legend_column` is always
auto-measured.

---

## Exports

`publiplots.__init__.py` exports unchanged. `LegendEntry` stays as an
internal helper (no user-facing need). `resolve_legend_flags` stays
internal.

---

## Testing

### New test files

**`tests/test_legend_entries.py` (new):**

- `test_resolve_legend_flags_bool_true` — `True` → all kinds True.
- `test_resolve_legend_flags_bool_false` — `False` → all kinds False.
- `test_resolve_legend_flags_dict_partial` — `dict(hue=False)` → hue
  False, others True.
- `test_resolve_legend_flags_dict_all` — every kind listed explicitly.
- `test_resolve_legend_flags_rejects_bad_type` — `legend=1` raises
  `TypeError`.
- `test_legend_entry_build_is_deterministic` — same inputs → same
  signature.
- `test_legend_entry_different_palettes_differ` — same name+kind,
  different palette → different signatures.
- `test_stash_entry_preserves_order` — multiple stashes append in
  order.
- `test_get_entries_empty_ax` — no stashes → empty list.

**`tests/test_legend_group_auto_collect.py` (new):**

- `test_auto_collect_dedups_identical_entries` — 1×3 grid, same hue on
  each axes → group has 1 entry.
- `test_auto_collect_preserves_order_across_axes` — unique entries on
  each axes → group renders in first-seen order.
- `test_collect_filter_picks_only_listed_names` — `collect=['x']`
  drops other entries; they render per-axis.
- `test_collect_ordering_follows_list_order` — `collect=['b', 'a']`
  renders b before a.
- `test_mismatched_signature_warns_once` — same name, different
  handles → 1 warning (not 2).
- `test_no_publiplots_axes_noops` — group on a `plt.subplots` figure
  doesn't crash; just doesn't auto-collect.
- `test_manual_and_auto_coexist` — `group.add_legend(...)` manually,
  then draw → both manual and auto entries render.

**`tests/test_subplots.py` (extend):**

- `test_legend_column_auto_sizes_to_group_width` — attach group, draw,
  `layout.legend_column > 0` and within 2 mm of measured group width.
- `test_legend_column_is_zero_without_group` — no group → `layout.legend_column == 0.0`.
- `test_legend_column_grows_when_entries_added` — add one entry, draw;
  add another, draw; width grows.
- `test_pp_subplots_rejects_legend_column_kwarg` —
  `pp.subplots(legend_column=30)` raises `TypeError`.

**`tests/test_scatter_legend_stash.py` (new):**

- `test_scatterplot_stashes_hue_entry` — `hue='x', legend=True` →
  `get_entries(ax)` has a hue entry with correct name.
- `test_scatterplot_stashes_size_entry` — `size='y', legend=True` →
  size entry stashed.
- `test_scatterplot_legend_dict_suppresses_hue` —
  `legend=dict(hue=False)` → no hue entry stashed; size still stashed.
- `test_scatterplot_legend_false_stashes_nothing` — `legend=False` →
  no entries.
- `test_scatterplot_legend_group_suppresses_per_axis_render` — group
  on fig + scatterplot → entries stashed but no Legend child on the
  axes.

**`tests/test_bar_legend_stash.py` (new):**

- `test_barplot_stashes_hue_entry` — basic stash path works.
- `test_barplot_legend_dict_hue_false` — dict form suppresses hue.

### Existing test updates

- `tests/test_subplots.py::test_subplots_legend_column_reserves_extra_width`
  — DROP this test. It tested an API that no longer exists.
- Any test that called `pp.subplots(legend_column=N)` — update to drop
  the kwarg; auto-measurement covers it.

### Gallery smoke tests

- `examples/plots/plot_14_edgecolor_control.py` — both `pp.subplots(...)`
  calls lose `legend_column=30`. The explicit `group.add_legend(handles=...)`
  calls are replaced with bare `pp.legend_group(anchor=axes[-1])` — auto
  collect handles it.
- `examples/plots/plot_02_scatter_plots.py` — bubble plot section
  continues to render correctly (visual regression check).

### Baseline

Full suite target: **121 baseline + ~25 new** = ~146 passing.

---

## Error handling

| Condition | Behavior |
|---|---|
| `pp.subplots(legend_column=30)` | `TypeError` with message pointing at `pp.legend_group` |
| `legend=1` (int) on plot | `TypeError` from `resolve_legend_flags` |
| `collect='treatment'` (str, not list) | Not caught explicitly; `in` check on the string returns True for any substring → buggy. **Mitigation**: `collect` validation in `MultiAxesLegendGroup.__init__`: if not None and not list/tuple → `TypeError`. |
| `legend_group` on non-`pp.subplots` figure | No-op auto-collect (warn is NOT emitted — user may have explicit `add_legend` planned). |
| Mismatched signatures | One-shot `UserWarning` per figure. |
| Same `(name, kind)` appears in `collect=` multiple times | Dedup to first occurrence. No warning. |

---

## Interactions with existing code

- **`LegendBuilder`** (PR #78): unchanged public API. Still used by
  `pp.legend(ax)` and by `MultiAxesLegendGroup._render_entry`. The
  `external_to_axis` flag from PR #79 amendment continues to route
  legend_group artists away from per-axis tightbbox measurement.
- **`LayoutReactor`** (PR #78): unchanged. Still repositions the
  legend artists on every draw based on registration mm-offsets.
- **`SubplotsAutoLayout`** (PR #79): gains one new measurement
  (`legend_column`). Settle loop already handles convergence.
- **`pp.legend(ax)`** (PR #78): auto mode rewritten to read from
  `ax._publiplots_legend_entries` instead of scattered ad-hoc
  attributes. Public API unchanged.

---

## Decisions locked — do NOT re-litigate

- `legend_column` kwarg dropped entirely (breaking change; no deprecation).
- `legend=True/False` preserved; `legend=dict(kind=bool)` added; missing
  dict keys default to True.
- Per-axes storage uses a single standardized attribute
  (`_publiplots_legend_entries`). Ad-hoc stashes removed.
- Dedup by `(name, kind)` with signature-based mismatch detection.
  First-wins + one-shot warn.
- `collect=None` = collect all; `collect=[names]` = filter + order.
- Suppression rule: entry claimed by group → per-axis skips. Entry
  not claimed (or no group) → per-axis renders if its `kind` flag is
  True.
- Explicit `add_legend` calls on the group still work; render before
  auto-collected entries.
- `pp.legend(ax)` semantics unchanged — no magic auto-group detection.
- No inside-the-axes legend placement (deferred).
- No new rcParams.

---

## Files touched

**Create:**
- `src/publiplots/utils/legend_entries.py`
- `tests/test_legend_entries.py`
- `tests/test_legend_group_auto_collect.py`
- `tests/test_scatter_legend_stash.py`
- `tests/test_bar_legend_stash.py`

**Modify:**
- `src/publiplots/utils/legend_group.py` (collect, _materialize,
  claims, register on fig)
- `src/publiplots/layout/auto_layout.py` (_measure_legend_column,
  extend _ALL_SIDES, extend _needs_update)
- `src/publiplots/layout/subplots.py` (drop legend_column kwarg,
  raise on use)
- `src/publiplots/utils/legend.py` (rewrite auto-mode to read the
  standardized store)
- `src/publiplots/plot/bar.py` (adopt stash pattern)
- `src/publiplots/plot/box.py`
- `src/publiplots/plot/point.py`
- `src/publiplots/plot/scatter.py`
- `src/publiplots/plot/strip.py`
- `src/publiplots/plot/swarm.py`
- `src/publiplots/plot/violin.py`
- `src/publiplots/plot/raincloud.py`
- `src/publiplots/plot/heatmap.py`
- `examples/plots/plot_14_edgecolor_control.py` (drop `legend_column=30`,
  drop explicit `group.add_legend(handles=...)` calls)
- `tests/test_subplots.py` (drop legend_column test; add
  width-awareness tests)

---

## Open questions for implementation

None. All API decisions are locked above.
