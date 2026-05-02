# UpSet layout rework — follow-up after PR C

**Status:** deferred after PR C (figsize migration).
**Filed:** 2026-05-02.

## Problem

`pp.upsetplot(...)` clips its decorations on `plt.show()` and non-tight
`savefig`. Specifically:

- The intersection bar-top annotations (`ax.text(..., va="bottom")` at
  `src/publiplots/plot/upset/draw.py:82-88`) extend above the figure top.
- The set-size xlabel extends below the figure bottom.
- Any `title=` / `intersection_label=` / `set_label=` pushes further
  outside the canvas.

The root cause: `setup_upset_axes` (`src/publiplots/plot/upset/draw.py:282`)
builds a `GridSpec` with `top=1, bottom=0, left=0, right=1` — zero
margins. `fig.set_figheight((colw * (n_sets + 2)) / render_ratio)` sizes
the panels exactly; nothing reserves space for decorations drawn
outside the panel rectangles.

This bug is **pre-existing** (was already present on `main` before PR C).

## What we already tried

A stopgap in commit `5a009b9` reserved a **fixed** `1.2 × font-size`
top padding and `2.5 × font-size` bottom padding, growing the figure
by that amount and shifting the GridSpec bounds inward by the same
fraction. Result: fixed the no-title/no-label case, but a call with
`title` + `intersection_label` + `set_label` still overflowed by
`0.050 in` top + `0.046 in` bottom. Reverted in commit `4b8553c`.

**Conclusion:** fixed padding is the wrong abstraction. UpSet needs
either measurement-driven sizing or a composer-friendly redesign.

## Two viable paths

### Path A: measurement-driven sizing (minimal)

Same pattern as `SubplotsAutoLayout.settle()`: draw once, measure
`fig.get_tightbbox() - fig.bbox_inches` overflow, grow the figure by
the measured overflow, shift the GridSpec bounds, re-draw. Loop up
to `_MAX_CONVERGENCE_ITERS` times. This doesn't refactor UpSet's
API; it just makes the figure-sizing settle like the rest of
publiplots.

Pros:
- Narrow change, self-contained in `setup_upset_axes` (or a new
  `_settle_upset_layout` helper called at the end of `upsetplot`).
- Matches the existing `SubplotsAutoLayout` mental model.

Cons:
- UpSet still owns its own figure, can't be composed inside a
  larger `pp.subplots` grid.
- Redraw-then-measure is an extra draw pass.

### Path B: composer-friendly UpSet (architectural)

Refactor `pp.upsetplot` to:
1. Optionally accept `axes` (three axes: intersections, matrix, sets)
   created by a higher-level composer rather than building its own
   `plt.figure()` + GridSpec.
2. Internally, when `axes is None`, use `pp.subplots` with a
   multi-axes layout (3 axes in a 2×2 cell arrangement) so
   `SubplotsAutoLayout` handles measurement automatically.
3. Return the same `fig, axes_dict` API.

This unlocks aligned sub-plots (e.g., add a heatmap above the
intersection bars, share the intersection x-axis across panels) —
the use case the user mentioned.

Pros:
- Correct long-term design for publiplots' composer vision.
- UpSet becomes a first-class primitive that `pp.subplots` can host.
- Tight-bbox measurement is automatic via `SubplotsAutoLayout`.

Cons:
- Larger refactor. Requires: (a) a way for `pp.subplots` to accept
  a non-rectangular grid (3 axes in 2×2 cell space), (b) custom
  `width_ratios`/`height_ratios` driven by `elementsize` and
  `set_names` text width, (c) migration of the bar-width calculation
  in `setup_upset_axes:419-466` into the composer.
- Probably needs its own brainstorm + spec + plan cycle.

## Recommendation

Start with Path A for a quick ship. Ship Path B as a separate
follow-up PR after the return-types redesign (PR D) and the
ComplexHeatmap migration (also still in "inches-internally" mode).

## Files to touch (Path A)

- `src/publiplots/plot/upset/draw.py:setup_upset_axes` — add a
  post-draw measurement + settlement loop.
- Consider moving the settlement primitive out of
  `SubplotsAutoLayout.settle()` into a shared helper so UpSet can
  reuse it.

## Files to touch (Path B)

- `src/publiplots/plot/upset/diagram.py:upsetplot` — accept `axes`
  triple or create via `pp.subplots`.
- `src/publiplots/plot/upset/draw.py` — decouple `setup_upset_axes`
  from figure creation; make it operate on already-created axes.
- `src/publiplots/layout/subplots.py` — probably gain a
  `mosaic=`-style API to accept irregular layouts, or add a
  separate helper.
- Tests: port existing UpSet gallery examples; verify the
  aligned-subplot use case.

## Related

- ComplexHeatmap has similar pre-existing crop (~0.08 in bottom),
  deferred per user decision as "still WIP". The composer
  redesign (Path B) would also apply to it — both plots are
  composite GridSpec plots that bypass `pp.subplots`.
