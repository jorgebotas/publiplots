# Bulletproof LegendBuilder — Design

**Status:** Approved
**Date:** 2026-04-30
**Scope:** `src/publiplots/utils/legend.py` + `src/publiplots/themes/styles.py` + gallery cleanup + new tests

## Problem

The current `LegendBuilder` converts mm offsets to figure coordinates **once at `add_legend()` time** and passes that snapshot as `bbox_to_anchor` with `bbox_transform=fig.transFigure`. Any subsequent axes-position change leaves the legend anchored to the old axes corner while the axes moves:

```
axes x0=0.125 → 0.058   (tight_layout grew the axes)
axes x1=0.900 → 0.945
legend bbox_to_anchor=(0.916, 0.88)   ← unchanged; now points into the axes interior
```

Triggers: `plt.tight_layout()`, `fig.subplots_adjust(...)`, `constrained_layout=True` passes that run on every draw, and complex-subplot builders (e.g. `complex_heatmap` margin plots) that reposition the anchor axes after the legend is built.

## Goal

A legend builder that stays visually correct across every layout engine publiplots supports or might adopt, plus a coordinator for multi-subplot figures so users can compose one unified legend column across a subplot grid. Ship `constrained_layout=True` as the default layout engine for publiplots-styled figures so users don't need `tight_layout()` in the first place.

## Design

### Single global `constrained_layout` default

`set_notebook_style()` and `set_publication_style()` flip on `figure.constrained_layout.use=True` via rcParams. Gallery examples drop all `plt.tight_layout()` calls; `constrained_layout` handles them automatically. Users who don't call a style function are unaffected — bare `import publiplots as pp` keeps matplotlib's default layout engine (`None`).

### Reactive anchor via draw-event hook

The builder stores **mm offsets relative to the axes** (not figure coordinates). A new `LayoutReactor` registers a single `draw_event` callback per figure; on each draw, it re-reads `ax.get_position()` and updates every anchored element's `bbox_to_anchor`. This covers `tight_layout`, `subplots_adjust`, `constrained_layout`, and deferred axes repositioning by other builders.

Same hook detects "axes moved since last draw but no constrained-layout engine is active" and warns the user once. No monkey-patching of `tight_layout` or any matplotlib function.

### Component split

Keep the public API (`pp.legend(ax)` → `LegendBuilder`) unchanged. Split the 700-line `LegendBuilder` internals into:

| Component | Role | LOC |
|---|---|---|
| `LegendLayout` | pure geometry (mm bookkeeping, overflow) | ~150 |
| `LegendBuilder` | orchestrates `LegendLayout` + creates legend/colorbar artists | ~250 |
| `LayoutReactor` | per-figure singleton that refreshes anchors on every draw; emits layout warnings | ~100 |
| `MultiAxesLegendGroup` | **new public API**: one shared mm layout across multiple subplots, anchored to one chosen axes | ~80 |

### Why mm-offsets, not `transAxes`

`transAxes`-based anchoring (`bbox_transform=ax.transAxes` + an offset transform) would give free reactivity, but it can't express:
- Multi-column packing where column 2 needs to know column 1's rendered width.
- Colorbar widths/heights specified in mm (not axes-fraction).
- `MultiAxesLegendGroup` anchoring to a different axes than the legend's own.

Storing mm offsets + a per-figure draw hook gives us all three at the cost of one callback per figure.

## Files Touched

### Modify
- `src/publiplots/utils/legend.py` — extract `LegendLayout`, `LayoutReactor`; slim `LegendBuilder`; add `MultiAxesLegendGroup`. Public API (`pp.legend`, `create_legend_handles`, handler classes) untouched.
- `src/publiplots/utils/__init__.py` — export `MultiAxesLegendGroup` and a thin helper `legend_group(...)`.
- `src/publiplots/themes/styles.py` — add `"figure.constrained_layout.use": True` (plus `h_pad`/`w_pad`) to both `set_notebook_style()` and `set_publication_style()`.
- Every `examples/plots/plot_*.py` that calls `plt.tight_layout()` — remove the call (~40 sites across 14 files).

### Create
- `tests/test_legend_layout.py` — unit tests for `LegendLayout` (pure geometry, no figures needed).
- `tests/test_layout_reactor.py` — draw-event hook + warning tests.
- `tests/test_legend_group.py` — `MultiAxesLegendGroup` behavior.

### Not touched
- `create_legend_handles()`, `HandlerRectangle`, `HandlerMarker`, `HandlerLineMarker`, `LineMarkerPatch`, `MarkerPatch`, `RectanglePatch`.
- `pp.legend()` function signature.
- `LegendBuilder` public API (`add_legend`, `add_colorbar`, `add_title`).
- `complex_heatmap` / `ComplexHeatmapBuilder` internals — wiring to `MultiAxesLegendGroup` is a follow-up PR.
- `set_hatch_mode` and other style knobs.

## Component interfaces

### `LegendLayout` (pure geometry)

```python
@dataclass
class LegendLayout:
    x_offset: float = 2
    y_offset: float | None = None
    gap: float = 2
    column_spacing: float = 5
    vpad: float = 5
    max_width: float | None = None

    # Mutable state
    current_x: float = field(init=False)
    current_y: float = field(init=False)
    columns: list[float] = field(default_factory=list)
    current_column_width: float = 0

    def reset_to(self, axes_height_mm: float) -> None: ...
    def check_overflow(self, required_height: float) -> bool: ...
    def start_new_column(self) -> None: ...
    def advance_y(self, element_height: float) -> None: ...
    def update_width(self, element_width: float) -> None: ...
```

No matplotlib imports. Fully unit-testable with synthetic mm values.

### `LayoutReactor` (per-figure singleton)

```python
class LayoutReactor:
    """Keeps anchored elements aligned across layout changes."""

    @classmethod
    def get(cls, fig: Figure) -> "LayoutReactor":
        """Return (creating if needed) the reactor attached to this figure."""

    def register(
        self,
        ax: Axes,
        artist: Legend | Colorbar,
        mm_x_from_right: float,
        mm_y_from_top: float,
    ) -> None:
        """Store the mm offsets; artist's bbox will be recomputed on every draw."""

    def _on_draw(self, event) -> None:
        """
        For each registered element:
          1. Read ax.get_position() now.
          2. Convert stored mm offsets to figure coordinates.
          3. If differs from current bbox_to_anchor > 0.5 pixels, update it.
          4. On first detected displacement without an active constrained layout
             engine, emit a single UserWarning pointing at tight_layout /
             subplots_adjust as the likely cause.
        """
```

Implementation notes: use `weakref.WeakSet` for registered elements so that figure garbage collection cleans up callbacks. The reactor stores itself on `fig._publiplots_layout_reactor` to make `get()` idempotent.

### `LegendBuilder` (existing public API)

Unchanged signatures for `__init__`, `add_legend`, `add_colorbar`, `add_title`. Internals:

1. Construct a `LegendLayout` with the init parameters.
2. On each `add_*` call, ask the layout for mm positions, convert once to figure coords for initial draw, create the matplotlib artist, then hand the `(ax, artist, mm_x, mm_y)` tuple to `LayoutReactor.get(fig).register(...)`.
3. The reactor handles all subsequent repositioning. The builder doesn't need to track anything beyond what it already does for column overflow.

### `MultiAxesLegendGroup` (new)

```python
def legend_group(
    anchor: Axes,
    *,
    x_offset: float = 2,
    y_offset: float | None = None,
    gap: float = 2,
    column_spacing: float = 5,
    vpad: float = 5,
    max_width: float | None = None,
) -> MultiAxesLegendGroup: ...


class MultiAxesLegendGroup:
    """
    Shared mm-based legend column anchored to one chosen axes.

    All add_* calls go into one LegendLayout; each individual legend or
    colorbar is attached to its own axes but positioned in the shared column.
    """

    def add_legend(
        self, handles: list, label: str = "", *,
        ax: Axes | None = None,  # defaults to anchor
        **kwargs,
    ) -> Legend: ...

    def add_colorbar(
        self, mappable, *, ax: Axes | None = None, **kwargs,
    ) -> Colorbar: ...

    def add_title(
        self, text: str, *, ax: Axes | None = None, **kwargs,
    ) -> Text: ...
```

Anchor is the axes whose right edge defines `x=0` for the mm layout. Each element can be attached to a different axes (its own `ax` arg), but its anchor for `bbox_to_anchor` is still the group's anchor axes. This gives one unified column to the right of a subplot grid.

## Warning contract

Emitted exactly once per figure when `LayoutReactor._on_draw` detects:
- The axes moved since the previous draw (position delta > 0.5 pixel).
- AND the figure's layout engine is not `constrained_layout`.

Message (exact text):
```
A LegendBuilder element was displaced by a layout change (likely plt.tight_layout()
or fig.subplots_adjust). publiplots enables constrained_layout in
set_notebook_style() and set_publication_style() — using those avoids this
issue. The element was re-anchored automatically; rendered output is correct.
```

Category: `UserWarning`. Emitted via `warnings.warn(..., stacklevel=2)` so it points at the user's code (not ours). Emitted once per figure via a `_warned_figures` set on the reactor.

## Testing Strategy

### `tests/test_legend_layout.py` (~10 tests, pure geometry)
- Fresh layout starts at `(x_offset, axes_height - vpad)`.
- `advance_y` moves the cursor down by `element_height + gap`.
- `check_overflow` returns True when `current_y < required_height`.
- `start_new_column` records current width and resets y.
- Multiple columns: second column starts at `x_offset + col1_width + column_spacing`.
- `update_width` is max-monotonic (grows, never shrinks).
- Zero handles, single handle, very wide labels — no crashes.

### `tests/test_layout_reactor.py` (~6 tests)
- Registering an artist attaches a `draw_event` callback.
- After `ax.set_position(new_bbox)` + `fig.canvas.draw()`, artist's `_bbox_to_anchor` matches the new axes edge within 1 pixel.
- `plt.tight_layout()` followed by `fig.canvas.draw()` updates the anchor AND emits the `UserWarning`.
- With `fig = plt.figure(constrained_layout=True)`, the warning is NOT emitted (constrained is the intended path).
- Warning emitted exactly once per figure, even after many draws.
- Reactor survives figure GC (weakref cleanup test — create figure, register, delete figure, assert reactor released).

### `tests/test_legend_builder.py` (new test, file exists)
- After `add_legend`, calling `plt.tight_layout()` + `fig.canvas.draw()` leaves the legend within 1 px of the new axes right edge.

### `tests/test_legend_group.py` (~5 tests, new)
- `legend_group(axes[0]).add_legend(...)` places legend to the right of `axes[0]`.
- Adding legends anchored to `axes[1]` and `axes[2]` in the same group produces a single mm-aligned column (x positions equal within 1 px).
- Overflow within the group creates a second column still anchored to `axes[0]`.
- After `tight_layout` / axes repositioning, every element stays aligned to the (possibly-new) anchor edge.

### Full suite
- Current count: 39 passing. Target: ~60 after new tests.
- Gallery smoke test in CI: `make html` must build 14/14 (integration guard after dropping `tight_layout()` calls).

## Migration / compatibility

- **Breaking change to style functions:** `set_notebook_style()` and `set_publication_style()` now enable `constrained_layout`. Users with scripts that call `tight_layout()` after these styles will see a warning (informational; rendering is still correct because the reactor re-anchors).
- **Gallery examples:** all `plt.tight_layout()` calls removed. Smoke-tested that constrained_layout produces acceptable output for each.
- **No removed / renamed public API.** `pp.legend`, `create_legend_handles`, handler classes, patch classes, `LegendBuilder.add_legend`/`add_colorbar`/`add_title` — all identical.
- **New public symbol:** `pp.legend_group(anchor=...)` → `MultiAxesLegendGroup`. Exported from `publiplots.utils`.

## Out of Scope / YAGNI Fences

- Wiring `complex_heatmap` / `ComplexHeatmapBuilder` to accept a `MultiAxesLegendGroup`. Follow-up PR after this design lands.
- Interactive GUI-backend resize support beyond what `constrained_layout` already provides.
- Any refactor of `create_legend_handles`, `HandlerRectangle`, `HandlerMarker`, `HandlerLineMarker`, or the custom Patch classes.
- Any change to the `pp.legend()` function signature.
- `savefig(bbox_inches='tight')` interaction audit — if it surfaces a problem in the gallery, fix separately.
- Removing the `PT2MM` / `MM2INCH` constants. They become module-level so `LegendLayout` and `LegendBuilder` share them, but the numeric values don't change.
- Monkey-patching `tight_layout` or any stdlib function.
- Changing the default value of any existing rcParam other than the two `figure.constrained_layout.*` additions.

## Review Checklist (self-review, completed)

- **Placeholder scan**: none.
- **Internal consistency**: component responsibilities match the data flow diagram; the "mm offsets stored, figure coords derived per draw" invariant is named consistently in architecture, components, and testing sections.
- **Scope**: single implementation plan; follow-up items explicitly fenced off.
- **Ambiguity check**: warning text and trigger conditions are exact; `constrained_layout` detection uses `fig.get_layout_engine()` idiom (matplotlib ≥3.6 — verify in plan).
