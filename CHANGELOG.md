# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
