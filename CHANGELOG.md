# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
