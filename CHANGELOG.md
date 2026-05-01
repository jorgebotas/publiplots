# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
