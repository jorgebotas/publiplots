# Unified `edgecolor` Support Across Plot Types

**Date:** 2026-04-02
**Branch:** `fix/barplot-hue-equals-x`
**Status:** Design approved

## Problem

Publiplots' double-layer style (transparent face + opaque edge) assumes face and edge share the same base color. When a user wants a different edge color (e.g., `edgecolor="black"`), three things break:

1. **barplot:** Derives facecolor from edgecolor (`patch.set_facecolor(patch.get_edgecolor())`), so custom edgecolor poisons the face.
2. **boxplot:** `linecolor` controls all lines including outlier markers. When set, the recoloring block is skipped entirely, so outlier markers lose their palette-matched face colors.
3. **legend:** `create_legend_handles` always sets `edgecolor=col` (same as face). No mechanism to pass a distinct edgecolor, so legends never reflect edge overrides.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Parameter name | `edgecolor` everywhere | Unified API over seaborn's inconsistent `linecolor`/`edgecolor` |
| Legend behavior | Faithfully mirrors plot (face=palette, edge=override) | Legend should be a visual representation of the actual artists |
| Value type | Single color string only | Per-category edge mapping is over-engineering; covers 95% of use cases |
| Scope | bar, box, violin, scatter, strip, swarm, point, raincloud + legend | Full sweep for consistency. Venn/UpSet/heatmap excluded. |

## Color Model

Every publiplots artist follows this rule:

```
facecolor  = palette color (semantic identity)
edgecolor  = edgecolor param if provided, else facecolor
face_alpha = rcParams["alpha"] (transparent fill)
edge_alpha = 1.0 (opaque edge)
```

## Per-Function Changes

### barplot (`bar.py`)

- Add `edgecolor: Optional[str] = None` parameter.
- Fix the color flow after seaborn draws with `fill=False`:
  1. Set facecolor from the color seaborn put on the edge (this is the palette color).
  2. If `edgecolor` is provided, override patch edgecolor. Otherwise leave as-is (matches face).
  3. `apply_transparency` handles alpha only.
- Error bar lines: use `edgecolor` when set.
- `hue == categorical_axis` recolor block: same treatment.
- `_apply_hatches_and_override_colors`: respect `edgecolor` override.
- `_legend()`: pass `edgecolor` through to legend handles.

### boxplot (`box.py`)

- Add `edgecolor: Optional[str] = None` parameter.
- Deprecate `linecolor`: keep it functional for backward compatibility, but if both `edgecolor` and `linecolor` are provided, `edgecolor` takes precedence and a `FutureWarning` is emitted via `warnings.warn()`.
- Split the line recoloring into two categories:
  - **Structural lines** (whiskers, caps, medians, box edges): use `edgecolor` if provided, else palette facecolor.
  - **Outlier markers**: face = palette color + alpha, edge = `edgecolor` if provided, else palette color.
- Pass `edgecolor` to legend handles.

### violinplot (`violin.py`)

- Add `edgecolor: Optional[str] = None` parameter.
- When `edgecolor` is provided, pass it as seaborn's `linecolor` parameter.
- Keep existing `linecolor` parameter for backward compat; `edgecolor` takes precedence.
- Pass `edgecolor` to legend handles.

### scatterplot (`scatter.py`)

- Already has `edgecolor` parameter.
- Only change: pass resolved edgecolor through to `_legend()` and into `create_legend_handles`.

### stripplot (`strip.py`)

- Add `edgecolor: Optional[str] = None` parameter.
- After seaborn draws, set edge colors on the `PathCollection`.
- Pass `edgecolor` to legend handles.

### swarmplot (`swarm.py`)

- Add `edgecolor: Optional[str] = None` parameter.
- Same pattern as stripplot.
- Pass `edgecolor` to legend handles.

### pointplot (`point.py`)

- Add `edgecolor: Optional[str] = None` parameter.
- Controls marker edge color. Line color stays palette-derived.
- Pass `edgecolor` to legend handles.

### raincloudplot (`raincloud.py`)

- Add `edgecolor: Optional[str] = None` parameter.
- Pass through to `violinplot()`, `boxplot()`, and `stripplot()`/`swarmplot()` calls.

## Legend System Changes

### `create_legend_handles` (`legend.py`)

Add `edgecolors` parameter:

```python
def create_legend_handles(
    labels, colors=None, edgecolors=None, ...
):
```

- `edgecolors=None`: current behavior (edge = face color).
- `edgecolors="black"` (single string): broadcast to all handles.
- `edgecolors=["black", "navy", ...]` (list): zip with labels/colors.

Each handle (`RectanglePatch`, `MarkerPatch`, `LineMarkerPatch`) is created with `facecolor=col, edgecolor=edge_col`.

### Handler rendering

`HandlerRectangle._extract_properties` already reads `orig_handle.get_edgecolor()` from the handle object. Once handles are created with distinct edgecolors, handlers render them correctly. No handler changes needed.

### Per-function legend integration

Each plot function's legend code passes the resolved edgecolor:

```python
handle_kwargs = dict(
    alpha=alpha,
    linewidth=linewidth,
    color=color,
    edgecolors=[edgecolor] * n if edgecolor else None,
    style="rectangle",
)
```

## What Does NOT Change

- **`apply_transparency`** — only manages alpha. Color resolution happens upstream.
- **`ArtistTracker`** — no new responsibilities.
- **`rcParams`** — no new `edgecolor` default. This is an explicit override, not a theme setting.
- **`resolve_palette_map` / color system** — edgecolor is orthogonal to palette resolution.
- **Heatmap** — cell borders are a different concept.
- **Venn / UpSet** — out of scope, different artist model. Can be tackled independently later.
