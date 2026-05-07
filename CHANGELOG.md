# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.10.1] - 2026-05-07

### Added

- Claude Code plugin shipped in-repo at `.claude-plugin/` with two
  skills: `publiplots-guide` (core conventions + full `pp.*` API
  surface, canonical idioms, common gotchas) and `legend-placement`
  (detailed `pp.legend` scoping guide — per-axes, row/column bands,
  figure-level bands, and the `pp.legend(ax)` vs
  `pp.legend(anchor=ax)` asymmetry). Install via
  `/plugin marketplace add jorgebotas/publiplots` then
  `/plugin install publiplots@publiplots`. See README for details.

[0.10.1]: https://github.com/jorgebotas/publiplots/releases/tag/v0.10.1

## [0.10.0] - 2026-05-07

### BREAKING
- `pp.legend_group` has been removed. Use `pp.legend(...)` instead —
  the API is identical (all kwargs match), it is purely a rename:
  - Before: `pp.legend_group(side='right')` →
    After: `pp.legend(side='right')`
  - Before: `pp.legend_group(anchor=axes[-1])` →
    After: `pp.legend(anchor=axes[-1])`
  - Before: `pp.legend_group(axes=top_row, side='top')` →
    After: `pp.legend(axes=top_row, side='top')`
- The old `pp.legend(ax, auto=False, ...)` per-axes API has been
  removed. Use `pp.legend(ax)` (auto-collects stashed entries) or
  `pp.legend(ax, collect=[])` (suppress auto-collection for manual
  `.add_legend(...)` calls).

### Added
- Unified `pp.legend(axes=None, collect=None, *, side='right',
  anchor=None, ...)` — single public legend API that handles
  per-axes, multi-axes bands, and figure-level bands through one
  mental model.
- Single-axes scope (`pp.legend(ax)`) flips `external_to_axis=False`
  so the legend is measured by `ax.get_tightbbox()`, matching
  pre-0.10 `pp.legend(ax)` behavior.
- Internal `_ScopeAnchor` abstraction for anchor geometry
  (scaffolding — not yet user-facing).
- `_get_or_create_per_axes_group(ax)` helper — `render_entries` now
  routes all non-inside legends through a cached per-axes
  `MultiAxesLegendGroup`, preserving cursor state across successive
  plot calls.

### Fixed
- `_measure_one_group` early-returns for single-axes groups
  (`external_to_axis=False`) to prevent double-counting against the
  per-cell reservation (the axes tightbbox already counts the
  legend).

### Migration
```
sed -i 's/pp\.legend_group(/pp.legend(/g' your_file.py
```
No other changes needed — all kwargs (`side`, `anchor`, `axes`,
`collect`, `orientation`, `align`, `x_offset`, etc.) are identical
between the old `legend_group` and the new `legend`.

### Note on `pp.legend(ax)` vs `pp.legend(anchor=ax)`
These two forms now have subtly different meanings:
- `pp.legend(ax)` (positional) — **internal** per-axes legend
  (like pre-0.10 `pp.legend(ax)`), measured by axes tightbbox.
- `pp.legend(anchor=ax)` (kwarg) — **external** band pinned to
  that axes' edge (like pre-0.10 `pp.legend_group(anchor=ax)`),
  measured as an overhang.

The asymmetry preserves pre-0.10 `pp.legend_group(anchor=ax)`
semantics across the mechanical rename. A future minor release
may consolidate these.

[0.10.0]: https://github.com/jorgebotas/publiplots/releases/tag/v0.10.0

## [0.9.3] - 2026-05-06

### Added
- `pp.legend_group(axes=...)` scopes collection to a subset of the
  figure's axes. Multiple scoped groups can coexist on the same
  figure — each collects stashed entries only from its listed axes
  and evicts per-axis legends only from those axes. Enables
  independent bands for different subplot regions (e.g. a
  `side='top'` band for the top row and a `side='bottom'` band for
  the bottom row, each rendering its own entries). Both
  figure-anchored and axes-anchored multi-group configurations are
  supported; a `UserWarning` fires when two groups' scopes and
  `collect` filters overlap. Gallery sections 10 and 10b in
  `plot_17_legend_placement` showcase the two patterns (PR #120).

### Fixed
- `pp.legend_group(anchor=axes[r,c], side='top'|'bottom'|'left')`
  no longer places the band on top of the anchor axes'
  title / xlabel / ylabel. The band steps past the decoration
  via an ``mm_outward_decoration_offset`` baked onto each reactor
  registration during `SubplotsAutoLayout._measure_one_group` —
  no `get_tightbbox` / `set_in_layout` calls inside the reactor's
  per-draw refresh, so sphinx-gallery builds stay deterministic.
  `side='right'` is byte-identical to prior behavior. Both
  group-before-plots and group-after-plots orderings work (PR #120).
- `pp.savefig` used to force `bbox_inches='tight'`, which
  re-cropped the figure to the union of artist bboxes. For figures
  with figure-anchored `pp.legend_group` bands on
  `side='top'`/`'bottom'`/`'left'`, the crop could place the legend
  off-canvas and save a blank PNG. The default is now
  `bbox_inches=None`, letting publiplots' mm-precise `FigureLayout`
  control the canvas. Pass `bbox_inches='tight'` explicitly to opt
  back in. The rcParam default `savefig.bbox` was also changed to
  `'standard'` for consistency (PR #120).

[0.9.3]: https://github.com/jorgebotas/publiplots/releases/tag/v0.9.3

## [0.9.2] - 2026-05-05

### Fixed
- `pp.lineplot` and `pp.pointplot` legend markers now honor the
  `edgecolor` override (both the explicit argument and
  `pp.rcParams['edgecolor']`). `HandlerLineMarker.create_artists`
  extracted the handle's edgecolor but reused the face color for the
  marker ring, so `edgecolor='black'` had no visible effect on the
  legend swatch. `pp.scatterplot` was unaffected (it goes through
  `HandlerMarker`, which already routed `edgecolor` correctly)
  (PR #118).

[0.9.2]: https://github.com/jorgebotas/publiplots/releases/tag/v0.9.2

## [0.9.1] - 2026-05-05

### Added
- `pp.legend_group` can now be attached **after** the plot calls with
  the same end-state as attaching it before. On construction the group
  walks every axes in the grid, finds per-axis Legend artists whose
  titles match entries it will claim, removes them, and unregisters
  their `LayoutReactor` entries. `SubplotsAutoLayout` shrinks the
  per-column / per-row reservation on the next draw — no explicit
  redraw needed. Legends for entries the group does NOT claim
  (different kind, or excluded via ``collect=``) survive the eviction
  (PR #116).

### Fixed
- `pp.legend_group(figure=fig, side='bottom')` crashed with
  `AttributeError: 'FigureCanvasPdf' object has no attribute
  'get_renderer'` when saving to PDF, SVG, or PS.
  `LegendBuilder._measure_object_dimensions` called
  `self.fig.canvas.get_renderer()`, a method only present on
  `FigureCanvasAgg`. It now calls `get_window_extent()` without a
  renderer argument, falling back to matplotlib's cached renderer
  populated by the preceding `canvas.draw()`. Closes #115 (PR #116).

### Docs
- README gains a "Matplotlib backends" section clarifying publiplots
  is backend-agnostic and won't call `matplotlib.use(...)` implicitly
  (PR #116).

[0.9.1]: https://github.com/jorgebotas/publiplots/releases/tag/v0.9.1

## [0.9.0] - 2026-05-05

### Added
- Per-axis inside legend placement via `legend_kws={"inside": True,
  "loc": "upper right"}`. Drops the legend inside the axes using
  matplotlib's native corner-based placement instead of the default
  outside-right column; reactor skips registration so the legend
  doesn't snap back outward on redraw. Works across every plot
  function that forwards `legend_kws` (bar, scatter, line, point,
  strip, swarm, heatmap, ...) (PR #112).
- `pp.legend_group(side=...)` — the shared-legend band now supports
  all four sides: `'right'` (default, unchanged), `'bottom'`,
  `'left'`, `'top'`. `FigureLayout` gains `legend_band_bottom`,
  `legend_band_top`, `legend_band_left` scalars alongside the
  existing `legend_column` so the figure grows on the correct side
  to accommodate the legend (PR #113).
- `pp.legend_group(anchor=None)` — when no anchor is passed the group
  is **figure-anchored** and spans the full subplot grid on the
  chosen side. Passing `anchor=axes[r, c]` pins the band to a single
  cell and its reservation absorbs into that column's/row's
  per-cell tuple instead of the figure-level band (PR #113).
- `pp.legend_group(orientation=, align=)` — horizontal-layout mode
  for top/bottom bands with per-side auto defaults: bottom/top pick
  `orientation='horizontal'` + `align='center'`; right/left keep
  `'vertical'` + `'start'`. Defaults to `ncol=len(handles)` on
  horizontal legends (user-passed `ncol=` still wins). Alignment
  supports `'start' | 'center' | 'end'` (PR #113).

### Changed
- `pp.legend_group` now **merges** label sets across axes when a
  single name/kind entry appears on multiple panels. Previously it
  kept first-occurrence only, which meant a hue with levels
  distributed across panels (e.g., Panel A: `low`, Panel B: `mid`,
  Panel C: `high`) rendered as a legend with just the first panel's
  subset. The merged legend now shows the union of labels in
  first-seen order; a single `UserWarning` fires announcing the
  merge. When two axes disagree on the *handle* for the same label
  (different color, marker, etc.), first occurrence wins and a
  different `UserWarning` calls out the conflict. Continuous-hue
  (ScalarMappable) entries still fall back to first-occurrence
  because merging colormaps is ill-defined (PR #114).
- `pp.legend_group(anchor=axes[-1])` now grows that column's per-cell
  `right[-1]` instead of the figure-level `legend_column`. Visually
  identical on a 1×N grid; cleaner semantics on 2×2+ grids where a
  single-cell anchor shouldn't grow the figure's whole right side
  (PR #113).
- `LegendLayout` cursor fields renamed to orientation-neutral names
  (`current_outward`, `current_along`, `advance_along`,
  `start_new_band`, ...) so horizontal and vertical share one code
  path. Internal API, no user-visible change unless you were calling
  `LegendLayout` directly (PR #113).

### Fixed
- `pp.barplot(color=<hex>, hatch=<col>, hatch_map=...)` with no `hue=`
  no longer renders black bars. Rewrote the face/hatch/edge paint flow
  as a single deterministic pass driven by `BarSplitSpec.iter_draw_order`
  instead of the prior four-pass "seaborn `fill=False` puts palette on
  edge, copy edge to face later" chain that drifted across matplotlib
  versions. Closes #105 (PR #108).
- Top padding of legends attached to a real Axes (per-axis
  `pp.scatterplot(...)` default and axes-anchored `pp.legend_group(...)`)
  used to sit 5 mm below the axes rectangle top because `vpad` always
  defaulted to 5 regardless of anchor kind. `LegendBuilder` now
  resolves `vpad=None` to 0 for real-Axes anchors (legend top flush
  with axes) and to 5 for `_GridAnchor` (figure-anchored groups,
  unchanged). User-passed `vpad=...` still wins (PR #113).
- Figure-anchored `pp.legend_group(side='bottom'|'left'|'top')` used
  to crowd axis decorations because `_GridAnchor.get_position()`
  returned the raw axes-rectangle union. Now returns the
  **decorated-grid** bbox (axes rectangles plus their per-axis
  `xlabel_space`/`ylabel_space`/`title_space` reservations), so
  bottom/left/top bands start past every panel's tick labels and
  titles (PR #113).
- `pp.legend_group` now preserves pre-existing Legend children on its
  anchor axes when `_materialize` runs. Previously, if an inside
  legend was attached to the same axes as a `legend_group`'s anchor,
  one of the two legends would be evicted by matplotlib's
  `ax.legend()` call. Both now survive (PR #112).

### Refactor
- 5 plot modules (`scatter`, `line`, `point`, `strip`, `swarm`) each
  inlined the same seven-line render loop over stashed legend
  entries. Consolidated into `render_entries(ax, flags=,
  legend_kws=)` with a `_BUILDER_FORWARD_KEYS` allowlist. Net −60 LOC
  and `legend_kws` now actually flows through to `LegendBuilder`
  (PR #112).

### Docs
- New `examples/plots/plot_17_legend_placement.py` — seven sections
  walking through every placement mode: default outside-right,
  per-axis inside, figure-anchored on each of the four sides,
  axes-anchored single-cell, and the inside + group coexistence
  pattern (PR #113).
- `examples/plots/plot_01_bar_plots.py` gains three new panels
  covering `color=` + `hatch=` combinations missed by the previous
  gallery: fixed color with hatch on a separate column (the exact
  case from #105), `hue == hatch` merged legend, and `edgecolor=`
  override alongside `hatch=` (PR #108).

[0.9.0]: https://github.com/jorgebotas/publiplots/releases/tag/v0.9.0

## [0.8.4] - 2026-05-04

### Added
- `pp.scatterplot` gains `background_marker: bool | str = False`. When
  truthy, each point gets a solid background-colored twin (white by
  default; any color string overrides it) so overlap is hidden —
  useful for publication panels, small-multiples, and categorical
  bubble plots where each point should read independently. Off by
  default because duplicating every point doubles artist count and
  overlap is often informative. The real work is foreground
  pre-compositing: face colors are blended over `background_color`
  at the user's `alpha` and drawn at full opacity, so overlapping
  points last-draw-wins instead of alpha-blending into muddy
  patches. A lower-zorder `PathCollection` twin is also emitted so
  the effect survives non-white axes patches (PR #110).
- Gallery panel "Hiding Overlap with `background_marker`" in
  `plot_02_scatter_plots.py` showing the three modes side by side
  (PR #110).

### Internal
- New `composite_facecolors_over` and `apply_background_markers`
  helpers in `publiplots.utils.transparency`, reusing the same
  abstraction boundary as `apply_double_layer_markers` (Line2D
  counterpart used by `pp.pointplot` / `pp.lineplot`) (PR #110).

[0.8.4]: https://github.com/jorgebotas/publiplots/releases/tag/v0.8.4

## [0.8.3] - 2026-05-04

### Fixed
- `pp.barplot(color=<hex>, hatch=<col>, hatch_map=...)` rendered black
  bars regardless of `color=` or `palette=` under matplotlib 3.10.8,
  and was inconsistent across matplotlib versions. The face-color flow
  round-tripped through `patch.get_edgecolor()` after several in-place
  rewrites, picking up a stale sentinel. Replaced the four sequential
  paint passes with a single `_paint_bars` helper driven by
  `BarSplitSpec.iter_draw_order` that sets face → hatch → edge in one
  pass. Closes #105 (PR #108).

### Added
- Gallery panels in `plot_01_bar_plots.py` covering the previously
  undocumented combinations: fixed `color=` + `hatch=` on a separate
  column, `hue == hatch` on the same column (combined legend), and
  `edgecolor=` override paired with `hatch=` (PR #108).

[0.8.3]: https://github.com/jorgebotas/publiplots/releases/tag/v0.8.3

## [0.8.2] - 2026-05-04

### Fixed
- `pp.lineplot` markers now follow the publiplots double-layer
  convention (pale fill + solid colored ring), matching `pp.pointplot`.
  Seaborn's default was a white edge on an opaque pastel face, which
  clashed visually. Lifts pointplot's two-pass marker renderer into a
  shared `apply_double_layer_markers` helper in
  `utils/transparency.py` so both plots stay in sync (PR #106).
- `pp.lineplot(hue=..., style=..., dashes={'X': (on, off)}, markers=True)`
  crashed in matplotlib's `Line2D._get_dash_pattern` because
  `HandlerLineMarker` forwarded the raw on-off tuple to a fresh
  `Line2D`. Extracted a `_normalize_dash_linestyle` helper and
  routed both `HandlerLine` and `HandlerLineMarker` through it so
  they can't drift again. Closes #99 (PR #104).

### Added
- Gallery panel "Markers on Aggregated Points" in
  `plot_04_lineplot.py` demonstrating the double-layer marker
  styling on a lineplot (PR #106).

### Internal
- Shared marker renderer `apply_double_layer_markers` now lives in
  `publiplots.utils.transparency`; `pp.pointplot` and `pp.lineplot`
  both delegate to it. Removes ~75 lines of duplicated logic from
  `plot/point.py` (PR #106).

[0.8.2]: https://github.com/jorgebotas/publiplots/releases/tag/v0.8.2

## [0.8.1] - 2026-05-04

### Added
- `pp.heatmap` dot heatmap mode now accepts `square=True` (force
  equal aspect) and `edgecolor=` (override bubble edges), delegating
  to `pp.scatterplot`'s own support for both.

### Fixed
- Dot heatmap bubble edges rendered in steel blue (matplotlib's
  default C0) regardless of the cmap.  ``ax.scatter(edgecolors="face")``
  recorded a sentinel that ``apply_transparency`` snapshotted before
  the figure drew, locking the wrong color onto every bubble.
  ``_draw_dot_heatmap`` now delegates to `pp.scatterplot`, which sets
  edge colors explicitly from the resolved facecolors in the right
  order. ``edgecolor=`` kwarg also now overrides correctly.
- Colorbar tick labels clipped outside the figure bounds under
  sphinx-gallery's scraper (``savefig.bbox='standard'``,
  ``pad_inches=0``) when the colorbar had no title. Root cause:
  ``SubplotsAutoLayout._artist_window_extent`` returned the bare
  colorbar rectangle instead of the tight bbox, so the reactor
  didn't reserve space for tick labels. Colorbars with a title
  happened to work because the title is a separate ``fig.text``
  artist that the reactor already measured. Switched to
  ``get_tightbbox()`` which includes all decorations.
- Dot heatmap axis limits inflated to absurd margins for small
  categorical grids (3 columns → xlim spanned 4 units). Per-axis
  margin is now ``0.5 / max(n - 1, 1)`` — half a cell of padding
  regardless of grid size.

### Changed
- Dot heatmap default ``sizes`` changed from ``(50, 500)`` to
  ``(20, 200)`` (the shared scatter default). User-supplied
  ``sizes=`` is unchanged. Existing figures with an explicit
  ``sizes=`` keyword continue to render identically.
- Dot heatmap spines and minor grid now share a single neutral gray
  (``#b0b0b0``) for a coherent cell-matrix outline. Previous
  ``#e0e0e0`` grid was too faint against a white background.

### Internal
- ``_draw_dot_heatmap`` delegates to ``pp.scatterplot`` instead of
  calling ``ax.scatter`` + reimplementing transparency, edge
  handling, size legends, and colorbar stashing. Removes 175 lines
  of duplicated logic; centralizes future fixes in one place.
- Deletes ``_create_size_legend`` helper (scatter's size legend
  path covers every case).

[0.8.1]: https://github.com/jorgebotas/publiplots/releases/tag/v0.8.1

## [0.8.0] - 2026-05-03

### Added
- `pp.lineplot` — full seaborn-parity line plot with publiplots
  conventions. Supports continuous/categorical `hue`, `size`, and
  `style`; `err_style="band"`/`"bars"`; `estimator`/`errorbar`/`n_boot`/
  `seed`/`orient`/`sort`; and the same `legend` / `legend_kws` /
  `legend_group` pipeline as the rest of the library. Gallery example
  at `plot_04_lineplot.py`; scripts `plot_05`–`plot_16` shifted up by
  one slot.
- `publiplots.utils.legend.LinePatch` + `HandlerLine` — line-only
  legend swatch (horizontal colored line with optional dash pattern).
  Used for lineplot hue and style legends.
- Categorical `size=` for `scatterplot` and `lineplot`. Passing a
  categorical column (or a ``pd.Categorical``) to ``size=`` resolves
  to one handle per category via the new ``resolve_size_map`` helper;
  accepts an explicit ``sizes={category: width}`` dict, a list, a
  ``(min, max)`` tuple, or a default ``(1.0, 4.0)`` interpolation.
- Shared plot-legend helpers in `publiplots.utils.plot_legend`:
  `get_size_ticks`, `stash_continuous_hue`, `resolve_style_maps`,
  `resolve_size_map`, `merge_categorical_entries`. Lifted from the
  scatter/point/strip/swarm modules and reused across scatter + line.

### Fixed
- When ``hue`` and ``style`` reference the **same categorical column**,
  `scatterplot` and `lineplot` now stash one merged `LegendEntry` whose
  swatches encode both dimensions (colored shaped marker for scatter;
  colored dashed line for lineplot) instead of two legend columns with
  identical row labels. Continuous hue (colorbar) and continuous size
  (size-tick swatches) stay separate — no sensible composite artist.
- ``pp.barplot(hue=x_column)`` without a ``hatch`` column previously
  stashed an empty ``LegendEntry(name=None, kind="hatch")`` that
  rendered as a blank legend frame on the axes. Guarded so nothing is
  stashed when ``hatch_map`` is empty.
- `LegendBuilder` row overlap with oversized markers. New helper
  `compute_min_labelspacing` scales `labelspacing` so size-encoded
  scatter legends no longer stack their circles on top of each other.
  Caller-supplied `labelspacing` always wins; text-only legends keep
  the matplotlib default.

### Changed
- Default scatter `sizes` tightened from `(50, 500)` to `(20, 200)`
  (points²). Produces legibly-sized markers without overflowing the
  plot area or forcing oversized legend swatches. Only applies when
  `sizes=None`; explicit user ranges are unchanged.

[0.8.0]: https://github.com/jorgebotas/publiplots/releases/tag/v0.8.0

## [0.7.1] - 2026-05-02

### Added
- `pp.annotate(ax, kind="bar_values", ...)` for in-plot value labels on
  barplots. Orientation-aware, errorbar-aware, sign-aware; anchors
  `outside`/`inside`/`base`/`center`; contrast-aware coloring via
  `color="auto"` (with compositing for translucent fills) or palette
  colors via `color="hue"`. `pp.barplot(..., annotate=True | dict)`
  wires the primitive into the barplot API.
- `pp.annotate(ax, kind="point_values", ...)` for pointplots. Directional
  anchors (`top`/`bottom`/`left`/`right`/`center`), errorbar-cap aware,
  hue-aware. `pp.pointplot(..., annotate=True | dict)` wires it in.
- `pp.annotate(ax, kind="box_stats", ...)` for boxplots and violinplots.
  Labels the median by default; pass `stats=["median", "q1", "q3", ...]`
  to label multiple stats per box. Shared strategy across
  `pp.boxplot(..., annotate=...)` and `pp.violinplot(..., annotate=...)`.
- New gallery example `plot_16_annotate.py` demonstrates `pp.annotate`
  across bar, point, box, and violin plots. Each of `plot_01`, `plot_03`,
  `plot_07`, `plot_08` gains a short annotated section that
  cross-references the dedicated gallery page.
- `BarSplitSpec` (`publiplots.annotate._splits`): single source of truth
  for barplot dodge semantics (which of `hue` / `hatch` cause splitting
  given collapse rules for matching-column cases). Shared between
  `barplot` and the `bar_values` annotation builder so they can't drift.

### Fixed
- `barplot` hatch-legend color for the `hue == categorical_axis` + separate
  `hatch` column case. The hatch swatches now render in gray (matching
  the double-split convention where hue and hatch are distinct columns);
  previously they took the publication color, which visually suggested
  the swatch color was meaningful even though the bar coloring came from
  the hue legend.

### Internal
- Anchor validation moved from the dispatcher to each strategy — each
  one defines its own anchor vocabulary and default. `bar_values` keeps
  `{outside, inside, base, center}`; `point_values` and `box_stats`
  take directional `{top, bottom, left, right, center}`.

[0.7.1]: https://github.com/jorgebotas/publiplots/releases/tag/v0.7.1

## [0.7.0] - 2026-05-02

### Breaking changes
- Plot functions return only the axes they drew into, not `(fig, ax)`. The `fig` handle is accessible via `ax.get_figure()` when needed. `pp.subplots` is unchanged and still returns `(fig, ax)` — it creates the figure, so the user needs the handle.
- `upsetplot` returns a dict `{"intersections": Axes, "matrix": Axes, "sets": Axes}` instead of `(fig, (ax_i, ax_m, ax_s))`. Switch from positional unpacking to named-key access.
- `complex_heatmap().build()` returns the axes dict directly instead of `(fig, axes_dict)`.

### Migration
```python
# before
fig, ax = pp.barplot(data=df, x="x", y="y")
fig.savefig("out.png")

# after
ax = pp.barplot(data=df, x="x", y="y")
ax.get_figure().savefig("out.png")   # or plt.savefig("out.png") / pp.savefig("out.png")
```

For `upsetplot`:
```python
# before
fig, (ax_i, ax_m, ax_s) = pp.upsetplot(sets)

# after
axes = pp.upsetplot(sets)
axes["intersections"]; axes["matrix"]; axes["sets"]
```

### Why
Every gallery example did `fig, ax = pp.<plot>(...)` and then ignored `fig`. `pp.savefig` already operated on the current figure (the old `pp.savefig(fig, ...)` examples were stale and would have errored). Removing `fig` from the return aligns with seaborn's convention and removes a dead variable.

### Fixed
- Stale docstring examples across the codebase that showed `pp.savefig(fig, 'output.png')` (an invalid call — `pp.savefig` takes only a filepath) are now correct.

### Internal
- Removed `docs/COMPLEX_HEATMAP_PLAN.md` (stale pre-implementation design doc for a feature that shipped long ago).
- Untracked `docs/superpowers/` from the repo — it's a local planning workspace (specs / plans / handoff notes) used by the brainstorming + writing-plans workflow. Added to `.gitignore`; previously-tracked files preserved on disk.

[0.7.0]: https://github.com/jorgebotas/publiplots/releases/tag/v0.7.0

## [0.6.0] - 2026-05-02

### Breaking changes
- `figsize=` is no longer accepted by `barplot`, `boxplot`, `violinplot`, `raincloudplot`, `scatterplot`, `stripplot`, `swarmplot`, `pointplot`, `heatmap` (both categorical and dot paths), `complex_heatmap`, `venn`, or `upsetplot`. Passing `figsize=` raises `TypeError` with a clear migration message. To customize axes dimensions, compose with `pp.subplots(axes_size=(w_mm, h_mm))` and pass `ax=` into the plot function.
- `complex_heatmap` replaces `figsize=(w_in, h_in)` with `axes_size=(w_mm, h_mm)` (millimeters, matching the rest of the publiplots API). Internally converts to inches for the existing gridspec math. Default is `(80, 60)` mm.
- `publiplots` no longer overrides matplotlib's `figure.figsize` rcParam. Figure dimensions come from `subplots.axes_size` via `pp.subplots`. Users reading `pp.rcParams['figure.figsize']` now see matplotlib's own default.

### Migration
Before:
```python
pp.barplot(data=df, x="x", y="y", figsize=(4, 3))
pp.complex_heatmap(data=df, figsize=(5, 5))
```
After:
```python
fig, ax = pp.subplots(axes_size=(80, 50))  # mm, not inches
pp.barplot(data=df, x="x", y="y", ax=ax)

pp.complex_heatmap(data=df, axes_size=(90, 90))  # mm, not inches
```
Or omit for auto-layout defaults.

### Why
When a plot function received `figsize=`, it took a `plt.subplots(figsize=...)` branch that did not install `SubplotsAutoLayout`. Legends, titles, and colorbar overflow were not reserved for — figures cropped on save (see the 0.5.0 horizontal raincloud known issue). `pp.subplots` is now the single source of truth for figure + axes dimensions.

### Fixed
- `barplot` `IndexError` in the `hue-only` legend path (when `hatch == categorical_axis` and `len(hue) < len(categorical_axis)`). The crashing `bar_color` computation is now gated behind the `if not (double_split or hatch == categorical_axis)` check that already guards its only consumer.
- `venn` internally used `plt.subplots` when `ax is None`, bypassing `SubplotsAutoLayout`. Now routes through `pp.subplots` consistently.

### Known issues
- `upsetplot` decorations (bar-top annotations, xlabel, title) can clip against the canvas on `plt.show()` and non-tight `savefig`. The bug is pre-existing (present since 0.5.0 and earlier). Tracked in `docs/superpowers/handoff/2026-05-02-upset-layout-followup.md`; a dedicated layout PR will address it.
- `complex_heatmap` has minor decoration overflow (~0.08 in bottom) in some configurations; part of the same composite-layout rework.

[0.6.0]: https://github.com/jorgebotas/publiplots/releases/tag/v0.6.0

## [0.5.0] - 2026-05-01

### Added
- `pp.subplots(nrows, ncols, axes_size=(w_mm, h_mm), ...)` — fixed-axes-size helper that grows the figure to fit auto-measured decorations (titles, labels, legends).
- `pp.legend_group(anchor=ax)` auto-collects `LegendEntry` objects stashed by plotting functions; renders one unified legend on the right of the anchor axes with auto-measured column width. Works across scatter, strip, swarm, point, box, violin, raincloud, bar, and heatmap.
- `"hatch"` as a new `_LEGEND_KINDS` entry — enables `legend={"hatch": False}` for per-kind suppression on bar plots' secondary split dimension.
- `legend: bool | dict[kind, bool]` accepted on every plot function for per-kind legend control (e.g. `legend={"size": False}` on a dot heatmap keeps the colorbar, hides the size legend).
- `LayoutReactor` — mm-based draw-event anchoring for colorbars and legend groups; recalibrates on figure resize without per-plot manual positioning.
- `rcParams["edgecolor"]` as a global default for plot edges (was per-call only).
- `external_to_axis` flag on reactor registrations so `legend_group` artists don't inflate per-axis tight-bbox measurements.
- New gallery example `plot_14_edgecolor_control.py` and shared-legend demo panels in `plot_01_bar_plots.py`, `plot_02_scatter_plots.py`.

### Changed
- Plot functions now use `pp.subplots()` when `figsize` is not provided, installing `SubplotsAutoLayout` for auto-sizing. Users who pass `figsize=` still get the legacy `plt.subplots` path (see [Known issues]).
- Legend auto-mode reads the new `LegendEntry` store on each axes first, falling back to the legacy per-artist `_legend_data` attribute for backward compatibility.
- Publication-first defaults unified — dropped the separate "notebook" style; all plots render at paper-ready dimensions by default.
- Bulletproof legend builder: mm-based positioning with automatic column overflow and a per-axes singleton pattern.

### Fixed
- Legend auto-layout measurement now unions reactor-pinned figure-level text artists (e.g. colorbar titles) into the side-reservation calculation, so titles no longer crop on `savefig`.
- `savefig` now settles layout before rendering — legend/title overflow is sized into the saved figure, not cropped post-hoc.
- `raincloudplot` no longer injects `rcParams["figure.figsize"]` as a default, letting the inner `violinplot` take the `pp.subplots` path and auto-reserve legend space.
- `violinplot` applies `edgecolor` to cloud `PolyCollection`s (previously seaborn's `linecolor` only touched inner stat lines).
- Colorbar title now sits above the colorbar, not overlapping.
- Gallery build preserves publiplots styling across sphinx-gallery runs.

### Known issues
- Plots called with explicit `figsize=(w_in, h_in)` still route through `plt.subplots(figsize=...)` and bypass `SubplotsAutoLayout`, which can crop legends and titles (e.g. the horizontal raincloud example in `plot_08`). Tracked for a dedicated cleanup PR that replaces `figsize=` with `axes_size=` across all plot functions.
- `barplot` raises `IndexError` in the `hue-only` legend path (`hatch == categorical_axis`) when `len(hue) < len(categorical_axis)`. Workaround: match cardinalities in that specific layout. Tracked for a dedicated fix PR.

[0.5.0]: https://github.com/jorgebotas/publiplots/releases/tag/v0.5.0

## [0.4.7] - 2026-04-29

### Added
- `heatmap()` and `complex_heatmap()` builder with dot-heatmap and margin-plot support
- Unified `edgecolor` parameter across `barplot`, `boxplot`, `violinplot`, `scatterplot`, `stripplot`, `swarmplot`, `pointplot`, and `raincloudplot`
- `edgecolors` parameter on `create_legend_handles` (decouples edge from face in legend entries)
- `as_categorical()` helper in `publiplots.utils` that guarantees `category` dtype
- Optional `ax` parameter on `pp.legend()` (defaults to `plt.gca()`)
- `rotate()` helper for axis-label rotation
- Scatterplot colorbar rendering test coverage

### Changed
- `LegendBuilder` now uses mm-based positioning with automatic column overflow and a singleton per-axes pattern
- Default text color set to black
- Barplot skips seaborn `hue` when `hue == categorical_axis` (avoids unwanted dodge)

### Fixed
- `AttributeError: Can only use .cat accessor with a 'category' dtype` in `barplot` when the categorical axis column was object-dtype strings
- Face/edge color bug in `barplot` when recoloring from palette
- Legend overlap bug caused by inaccurate title-height measurement
- Colorbar positioning bugs (vpad subtraction, xmargin/ymargin, remaining-space math)

### Dependencies
- Added `scipy>=1.10.0` (required by `heatmap` clustering)

[0.4.7]: https://github.com/jorgebotas/publiplots/releases/tag/v0.4.7

## [0.4.6] - 2025-12-04

### Added
- Diverging palettes --> sample from extremes
- Custom RdBu and RbGyBu, useful for "significance" related palettes (e.g. up/down regulation)
- Support for reversed custom palettes

[0.4.6]: https://github.com/jorgebotas/publiplots/releases/tag/v0.4.6

## [0.4.5] - 2025-11-26

### Added
- `pointplot()` - Point plots for visualizing trends and comparisons with error bars
- Forest plot examples (using `pointplot`) for meta-analysis visualization

[0.4.5]: https://github.com/jorgebotas/publiplots/releases/tag/v0.4.5

## [0.4.4] - 2025-11-24

### Added
- `stripplot()` - Strip plots for visualizing individual data points
- Support for jitter and dodging in strip plots

[0.4.4]: https://github.com/jorgebotas/publiplots/releases/tag/v0.4.4

## [0.4.3] - 2025-11-21

### Added
- `raincloudplot()` - RainCloud plots supported! see [documentation](https://jorgebotas.github.io/publiplots/) for further details
- `side` parameter support in `pp.violinplot` to generate one-sided violin plots
  
[0.4.3]: https://github.com/jorgebotas/publiplots/releases/tag/v0.4.3

## [0.4.2] - 2025-11-21

### Added
- `violinplot()` - Violin plots with transparent fill and opaque edges
- `ArtistTracker` class for selectively applying transparency when overlaying plots
- Combined violin + swarm plot examples

### Changed
- Reorganized codebase: moved plotting functions from `base/` and `advanced/` to single `plot/` directory
- Updated `boxplot()`, `violinplot()`, and `swarmplot()` to use `ArtistTracker` for proper overlay support
- Default `fill=False` for `violinplot()` to match publication style
- Reordered examples: swarm plots now appear before box and violin plots

### Fixed
- Transparency override issue when overlaying multiple plots (e.g., violin + swarm)

[0.4.2]: https://github.com/jorgebotas/publiplots/releases/tag/v0.4.2


## [0.4.1] - 2025-11-20

### Added
- `boxplot()` - Box plots with transparent fill and opaque edges
- `swarmplot()` - Swarm plots showing individual data points with transparency
- Combined box + swarm plot examples for richer visualizations
- Reorganized example gallery with flat structure (removed base/advanced separation)

[0.4.1]: https://github.com/jorgebotas/publiplots/releases/tag/v0.4.1



## [0.4.0] - 2025-11-20

### Added
- Changed approach to transparency application
- Avoid duplicating artists
- Included comprehensive documentation through Sphinx and Github Pages

[0.4.0]: https://github.com/jorgebotas/publiplots/releases/tag/v0.4.0



## [0.3.0] - 2025-11-17

### Added

- Dynamic Venn diagrams (not precomputed) based on **[ggvenn](https://github.com/yanlinlin82/ggvenn)** geometry

- UpSet plots based on **[UpSetPlot](https://github.com/jnothman/UpSetPlot)**
 

[0.3.0]: https://github.com/jorgebotas/publiplots/releases/tag/v0.3.0




## [0.2.0] - 2025-11-06

### Added
- Venn diagram support for 2 to upto 5 sets!
- Thanks to yanlinlin82/ggvenn for their amazing work on Venn diagram geometry
- Removed matplolib-venn dependencies

[0.2.0]: https://github.com/jorgebotas/publiplots/releases/tag/v0.2.0



## [0.1.0] - 2025-11-05

### Added
- Initial release
- Base plotting functions:
  - `barplot()` - Bar plots with error bars and grouping
  - `scatterplot()` - Scatter plots with flexible styling
- Advanced plotting functions:
  - `venn()` - 2-way and 3-way Venn diagrams
- Theme system:
  - Pastel color palettes optimized for publications
  - Publication, minimal, and poster style presets
  - Customizable marker and hatch patterns
- Utilities:
  - Legend builders with custom handlers
  - File I/O with `savefig()`
  - Axes manipulation utilities
- Comprehensive documentation and examples

[0.1.0]: https://github.com/jorgebotas/publiplots/releases/tag/v0.1.0
