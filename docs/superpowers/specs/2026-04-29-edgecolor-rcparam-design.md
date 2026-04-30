# Global `edgecolor` via rcParams — Design

**Status:** Approved
**Date:** 2026-04-29
**Scope:** `src/publiplots/themes/rcparams.py` + 8 plot modules + new test file

## Problem

Today every plot function (`barplot`, `boxplot`, `violinplot`, `scatterplot`, `stripplot`, `swarmplot`, `pointplot`, `raincloudplot`, `heatmap`) accepts `edgecolor: Optional[str] = None`, where `None` means "auto — use the face/palette color as the edge." Setting the same edgecolor across a figure therefore requires passing `edgecolor="black"` to every call. Other styling knobs (`alpha`, `color`, `capsize`, `palette`) are already exposed through `pp.rcParams` and resolved via `resolve_param(...)`; `edgecolor` should follow the same pattern.

## Goal

Let users set a global edge color once:

```python
pp.rcParams["edgecolor"] = "black"
pp.barplot(...)     # black edges
pp.boxplot(...)     # black edges
pp.scatterplot(...) # black marker edges
```

…and still override per-call:

```python
pp.barplot(..., edgecolor="red")  # red edges, ignores rcParam
```

…while preserving today's "auto" behavior when neither a function argument nor an rcParam override is set.

## Design

### Single global knob with `None` as "auto" sentinel

- Add `"edgecolor": None` to `PUBLIPLOTS_RCPARAMS` in `src/publiplots/themes/rcparams.py`.
- Every plot function that currently has an `edgecolor` parameter adds one line at the top of the function body:
  ```python
  edgecolor = resolve_param("edgecolor", edgecolor)
  ```
- All existing downstream logic (e.g. `edgecolor if edgecolor else face_color`, `if edgecolor is not None: ...`) is untouched. When `rcParams["edgecolor"]` is at its default `None` and the user doesn't pass `edgecolor`, `resolve_param` returns `None` and today's auto behavior runs unchanged.

### Resolution precedence

| User call | `rcParams["edgecolor"]` | Resolved value |
|---|---|---|
| `plot(edgecolor="red")` | any | `"red"` (user value wins) |
| `plot()` | `"black"` | `"black"` (rcParam) |
| `plot()` | `None` (default) | `None` → "auto" downstream |

### Why `None` instead of an `"auto"` sentinel string

- Zero downstream changes. Every existing `if edgecolor is not None` and `edgecolor if edgecolor else X` site keeps working.
- No chance of one plot function misinterpreting the string `"auto"` as a literal matplotlib color.
- Consistent with how Python libraries typically signal "unset" for Optional parameters.

### Known tradeoff (accepted)

If a user sets `rcParams["edgecolor"] = "black"` globally and wants a single plot to fall back to "auto," they can't express that via `edgecolor=None` (it's indistinguishable from "not provided"). Workarounds: context-manager the rcParam, or pass the palette color explicitly. Not worth designing a second sentinel for a niche case (YAGNI).

## Files Touched

### Modify
- `src/publiplots/themes/rcparams.py` — add `"edgecolor": None` to `PUBLIPLOTS_RCPARAMS`.
- `src/publiplots/plot/bar.py` — one-line `resolve_param` insertion at top of `barplot`.
- `src/publiplots/plot/box.py` — one-line insertion at top of `boxplot` (leave deprecated `linecolor` backcompat path alone).
- `src/publiplots/plot/violin.py` — one-line insertion at top of `violinplot`.
- `src/publiplots/plot/scatter.py` — one-line insertion at top of public `scatterplot` only.
- `src/publiplots/plot/strip.py` — one-line insertion at top of `stripplot`.
- `src/publiplots/plot/swarm.py` — one-line insertion at top of `swarmplot`.
- `src/publiplots/plot/point.py` — one-line insertion at top of `pointplot`. Marker edges only; the connecting line remains tied to the series color (matches seaborn and matplotlib conventions — `edgecolor` is for closed-shape edges and marker outlines, not line strokes).
- `src/publiplots/plot/raincloud.py` — one-line insertion at top of `raincloudplot`.
- `src/publiplots/plot/heatmap.py` — one-line insertion at top of both `heatmap()` and `complex_heatmap()` entry points.

### Create
- `tests/test_edgecolor_rcparam.py` — unit tests for the resolution semantics across representative plot types.

### Not touched
- No helper functions added.
- No change to `utils/validation.py` or `utils/__init__.py`.
- No change to downstream artist-manipulation logic in any plot module.
- `box.py`'s deprecated `linecolor` backcompat path is left as-is.
- No per-plot-family rcParams (`bar.edgecolor`, etc.). Single global knob.

## Testing Strategy

Four focused unit tests in `tests/test_edgecolor_rcparam.py`, plus smoke tests across plot types:

1. **Default is `None`** — `pp.rcParams["edgecolor"]` returns `None` immediately after import (protects against accidental default changes by future contributors).
2. **rcParam applies when arg omitted (barplot)** — set `rcParams["edgecolor"] = "red"`, call `pp.barplot(...)` without `edgecolor`, assert a bar patch has red edge.
3. **Explicit arg wins over rcParam (barplot)** — set `rcParams["edgecolor"] = "red"`, call `pp.barplot(edgecolor="blue")`, assert edge is blue.
4. **Passthrough preserves auto (barplot)** — default rcParam (None), no arg, assert the bar edge matches the face color (current behavior unchanged).
5. **Smoke test: scatterplot** — one test confirming rcParam-set edge reaches marker collection edges.
6. **Smoke test: boxplot** — one test confirming rcParam-set edge reaches box patch edges.

All tests use an autouse fixture that snapshots and restores `pp.rcParams["edgecolor"]` to prevent cross-test pollution, and an autouse fixture that closes figures.

Full `pytest tests/` must remain green (32 → 38+ passing).

## Out of Scope / YAGNI Fences

- No per-plot-family overrides (`bar.edgecolor`, `scatter.edgecolor`).
- No `"auto"` sentinel string.
- No changes to downstream ternaries / `if edgecolor is not None` sites.
- No pointplot line-color change.
- No `linecolor` deprecation cleanup in `box.py`.
- No edge-alpha parameter. Edge alpha stays at 1.0 (current behavior).
- No global context manager helper (`with pp.rc_context(edgecolor="black"):`). Matplotlib already ships `plt.rc_context`; if we need a publiplots-aware version later that's a separate change.

## Review Checklist (self-review, completed)

- **Placeholder scan**: none.
- **Internal consistency**: resolution rule, file list, and test plan match. `None` semantics are consistent everywhere.
- **Scope**: single implementation plan, no decomposition needed.
- **Ambiguity**: "auto" defined explicitly as "whatever each plot does today for `edgecolor=None`"; pointplot line color explicitly excluded from the definition.
