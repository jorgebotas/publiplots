# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.15.3] - 2026-07-21

### Fixed

- `pp.pointplot(..., annotate=...)` now labels each point at the value it is
  actually drawn at for any `estimator`, and places the label on the drawn
  marker under `dodge=`. Previously both the label text and its anchor were
  recomputed from the group **mean** at the integer category position, so a
  non-mean `estimator` (median or a custom callable) mislabeled every point,
  and dodged markers had their labels float off-center. Positions and values
  now come from the drawn marker artists, mirroring the `pp.barplot` fix in
  0.15.2 (#194).

## [0.15.2] - 2026-07-21

### Fixed

- `pp.barplot(..., annotate=...)` now labels the value the bar is actually
  drawn at for any `estimator`. Previously the annotation always showed the
  group **mean**, even when `estimator="median"` (or a custom callable) drew
  the bars at a different height, so the label and its bar silently disagreed.
  Bar values now come from the drawn patch (`get_height()` / `get_width()`),
  matching the reconstruct-from-patches path, so `bar_values` and
  `bar_custom`'s default `value_source` stay consistent with the plot (#194).

## [0.15.1] - 2026-06-18

### Added

- Vertical orientation for 2-way Venn diagrams. `pp.venn(...)` accepts a new
  `orientation` argument (`"horizontal"`, the default, or `"vertical"`).
  Vertical stacks the two circles with the first set on top, the intersection
  in the middle, and the set-name labels at the outer ends. Computed as a 90°
  rotation of the existing horizontal layout, so the two orientations stay in
  lockstep. Supported for 2 sets only — `orientation="vertical"` with 3 or more
  sets raises `ValueError`.

## [0.15.0] - 2026-06-15

### Added

- Outer-only axis labels for shared-axes grids, matching seaborn's
  `FacetGrid`. When `pp.subplots(...)` is called with `sharex` / `sharey`,
  interior tick labels, offset text, and axis labels are now hidden by
  default — leaving labels only on the bottom row (x) and left column (y)
  — and the grid tightens as the freed label reservations collapse. A new
  `label_outer` kwarg on `pp.subplots` (default `True`; pass `False` or
  `"all"` to draw every label) controls this. The standalone
  `pp.label_outer(axes, *, sharex=True, sharey=True)` helper mirrors
  matplotlib's `ax.label_outer()` for re-applying outer-only labeling
  after late `set_xlabel` / `set_ylabel` calls, and works on foreign
  (`plt.subplots`) grids too. Because publiplots builds its grid with
  `fig.add_axes` (no `SubplotSpec`), the standard `ax.label_outer()` is
  unavailable; the outer edge is computed from publiplots' own axes
  matrix instead. New gallery example
  `examples/plots/plot_25_shared_axes_labels.py` (#188).

### Fixed

- `LegendBuilder.add_colorbar(cmap=...)` no longer raises `ImportError`
  on matplotlib 3.9+. `matplotlib.cm.get_cmap` was removed in 3.9; the
  call now routes through `matplotlib.pyplot.get_cmap` (accepts a name or
  a `Colormap` instance). This also unblocked the documentation gallery
  build on modern matplotlib (#189).

## [0.14.2] - 2026-06-11

### Added

- `legend.handletextpad` and `legend.borderpad` matplotlib rcParams are
  now honoured as defaults for publiplots legends (previously hard-coded
  in `LegendBuilder.add_legend` and its size estimators, so the rcParams
  had no effect). Set them globally via `pp.rcParams[...]`; an explicit
  per-call `legend_kws={...}` still overrides. The width-estimation path
  also reads `legend.handlelength` so the reserved layout space stays
  consistent with what matplotlib draws. `legend.columnspacing` already
  worked; `legend.labelspacing` is unchanged (it keeps its no-overlap
  floor) (#186).

### Changed

- The default `legend.columnspacing` is now `1.0` (down from
  matplotlib's `2.0`). At publiplots' small `legend.fontsize` the
  font-size-relative `2.0` reads as loose, so multi-entry horizontal
  (`side='top'`/`'bottom'`) legends drifted apart; `1.0` keeps them
  compact. This changes default output for multi-entry horizontal
  legends; override per-call or via `pp.rcParams['legend.columnspacing']`
  (#186).

## [0.14.1] - 2026-06-10

### Fixed

- `pp.barplot`/`pp.pointplot` value labels (`annotate=True`/`pp.annotate`)
  no longer overlap the error bar when `capsize > 0`. With a non-zero
  capsize, seaborn packs the whole errorbar (bottom cap + vertical stem +
  top cap) into a single `Line2D` whose data is `nan`-separated between
  sub-segments; `_iter_error_segments` checked finiteness over the whole
  array and discarded any line containing a `nan`, so `err_low/err_high`
  came back `None` and labels anchored on the bar top instead of clearing
  the cap. The default `capsize=0` (a clean 2-point `Line2D`) was
  unaffected, which is why it slipped past the existing anchor test.
  Errorbar lines are now split on `nan` into finite contiguous runs so
  the vertical-stem run survives the matcher; all-`nan` lines
  (single-sample groups) still yield nothing. Fixes both orientations and
  is sign-aware (negative bars label below the bottom cap) (#184).

## [0.14.0] - 2026-06-09

### Added

- `legend_kws` on plot functions now forwards legend **placement** keys
  (`side`, `orientation`, `align`, `x_offset`, `y_offset`, `gap`), so a
  per-axes legend can be positioned in a single call —
  e.g. `pp.scatterplot(..., legend_kws={"side": "left"})` (#182).
- Per-axes legends (`pp.legend(ax, side=...)`) support all four sides
  (`top`/`bottom`/`left`/`right`). For `side="top"` the axes title is
  lifted above the legend band (stacking: axes → legend → title); for
  `side="left"` the legend is offset past the y-tick labels
  dynamically (#182).

### Changed

- `pp.legend(ax)` now **adopts** the per-axes legend group that the
  plot call already created (cached on `ax._legend_group`) instead of
  building a second competing group. The requested `side=` is honoured,
  the spurious `scope overlaps with an existing group` warning on the
  `pp.scatterplot(...); pp.legend(ax)` flow is gone, and the redundant
  double-render/per-draw hook is removed (~7× → ~1.2× on multi-panel
  figures). Repeated `pp.legend(ax)` calls re-adopt the same group (#182).
- `__version__` is now resolved from the installed package metadata
  (`importlib.metadata`) rather than hard-coded in `__init__.py`, so the
  version lives in a single place (`pyproject.toml`) and can no longer
  drift (#182).

### Fixed

- `side="top"`/`"bottom"` center-aligned legends no longer force a full
  `fig.canvas.draw()` per element on every draw (the per-draw alignment
  hook reuses the current renderer cache), eliminating an O(panels)
  slowdown (#182).
- The `side="top"` title lift uses the title offset transform instead of
  `ax.set_title()`, so it no longer resets the title's color/fontsize/
  weight, and a user-set title pad larger than the band lift is
  preserved (#182).

## [0.13.1] - 2026-05-28

### Fixed

- `pp.legend(anchor=ax, inside=True)` no longer shifts the layout
  of sibling axes (#180). Three connected regressions in the
  inside-mode rollout: (a) `MultiAxesLegendGroup.add_legend(...)`
  did not auto-inject `inside=True` + `loc=` for inside groups, so
  manual handle-add calls (e.g. `band.add_legend(handles=...,
  ncol=2)` after `collect=[]`) fell through to the band path with
  reactor registration; (b) even when the inside path did fire,
  the resulting matplotlib `Legend` artist counted in
  `axes.tightbbox`, so `SubplotsAutoLayout`'s per-cell measurement
  pass grew the anchor cell and pushed siblings inward; (c)
  `clear_anchor=True` was gated behind a successful collection in
  `_materialize`, so `collect=[]` skipped both the render AND the
  blank, leaving the cell's frame intact. Fix: route auto-injection
  through the public `add_legend`; call `legend.set_in_layout(False)`
  in `LegendBuilder`'s inside branch (matches matplotlib's
  `loc=...` semantics); move `set_axis_off()` to fire at group
  construction so it's independent of the collection path.

## [0.13.0] - 2026-05-19

### Added

- `pp.legend(anchor=ax, inside=True)` — render a shared legend
  **inside a dedicated grid cell** rather than overhanging the
  figure's edge. Common use case: a 2x2 grid where 3 cells host
  plots and the 4th is reserved for the legend; previously this
  required either wasting the empty cell on a band overhang or
  cluttering one of the plotted cells with a per-axes legend.
  The new mode renders the collected legend inside the cell at
  `anchor=ax`, auto-blanks the cell so it reads as a clean tile
  (opt out with `clear_anchor=False`), and walks the existing
  collection scope (`axes=` for explicit, full figure by default).
  Default placement `side='left', align='start'` maps to matplotlib
  `loc='upper left'` — visually a continuation of the band-mode
  recipe `pp.legend(side='right', align='start')` for the
  empty-cell-on-the-right case. The full `side` x `align` matrix
  (13 reachable matplotlib `loc` strings, including the new
  `side='center'` sentinel) lets callers tuck the legend into any
  corner of the cell. See PR #179 and the `legend-placement` skill
  for usage examples.

## [0.12.1] - 2026-05-19

### Fixed

- `pp.barplot(border_radius=...)` and `pp.boxplot(border_radius=...)`
  now round the **free end** of horizontal bars and boxes (the
  end opposite the baseline / zero line) instead of the long top
  and bottom edges. Previously, `_RoundedBarPatch` always applied
  `top_mm` to the bbox's y-extreme corners and `bottom_mm` to the
  y-baseline corners — correct for vertical orientations (and
  sign-aware via matplotlib's signed `height`), but visually wrong
  for horizontal bars/boxes where the user expected end-cap
  rounding. `apply_border_radius` now accepts `orient="v" | "h"`
  and rotates the corner mapping 90° for horizontal plots, so
  `border_radius=(top_mm, bottom_mm)` consistently means
  (free-end, base-end) regardless of orientation or sign. The
  signed-width path also correctly handles negative-valued
  horizontal bars (free end = visually-left). `pp.raincloudplot`
  inherits the fix via its internal `pp.boxplot` call. (Fixes
  the case where `border_radius` on `seaborn.barplot(orient='h')`
  bars rounded the long sides instead of the end caps.)

## [0.12.0] - 2026-05-18

### Added

- `pp.legend(rows=, cols=, span=, ax=)` — grid-scope kwargs for
  figure legends. `rows`/`cols` resolve over the `pp.subplots` axes
  matrix to a single index or inclusive `(start, end)` range.
  `span='fig'` is sugar for the full-figure default;
  `span='row'`/`'col'` with a positional Axes anchor expands to that
  anchor's row or column. `ax=[a1, a2, ...]` collects from an
  explicit list with handle dedupe by label. All four are
  mutually-exclusive addressing modes; existing call signatures
  unchanged. (PR #166)

### Fixed

- `pp.annotate(color="hue")` now renders opaque label text when the
  bar fill is translucent (e.g. the default `alpha=0.1` outline
  style). Previously, when palette lookup fell back to the bar's
  facecolor — most notably when `hue == x` collapses `hue_value` to
  `None` in seaborn — the bar's alpha leaked into the label color,
  producing 10%-opacity text. The three bar-record builders
  (`build_from_barplot_call`, `build_from_stacked_barplot_call`,
  `build_from_histplot_call`) now pin the fallback alpha to 1.0.
  (PR #172, fixes #171)
- `pp.annotate(kind='bar_values' | 'bar_custom' | 'box_stats')` now
  renders labels when `pp.rcParams['bar.border_radius']` (or
  `box.border_radius`) is non-zero. `apply_border_radius` swaps
  matplotlib's `Rectangle` / `PathPatch` artists for publiplots'
  custom `_RoundedBarPatch` class, and the annotate subsystem's
  isinstance gates in `_cache._is_bar_rect`, the
  stacked-barplot patch walker, and the boxplot patch walker
  previously rejected those rounded patches — silently emitting
  zero labels. The gates are widened to admit `_RoundedBarPatch`
  alongside the original types. (PR #175, fixes #173)

## [0.11.3] - 2026-05-13

### Added

- `pp.histplot` 2D mode — calling `pp.histplot(data=df, x=..., y=...)`
  with both axes set now renders a heatmap-like bivariate histogram
  via a single `QuadMesh` (or N stacked `QuadMesh`es when `hue=` is
  set, one per hue level). 1D paths are unchanged. Three new 2D-only
  kwargs land alongside: `cmap=`, `vmin=`, `vmax=` (silently ignored
  in 1D). The default cmap is a light sequential gradient built from
  `pp.rcParams["color"]` (or per-call `color=`); pass any matplotlib
  or seaborn cmap name to override (PR #159).
- `pp.jointplot(kind="hist")` is now valid (was deferred in 0.11.0
  with a `NotImplementedError`). The joint panel renders a 2D
  histogram, the marginals render 1D histograms. `_KIND_REGISTRY`
  wires `'hist'` → `(pp.histplot, pp.histplot)`; the kind-enumeration
  docstring lists `'hist'` alongside `'scatter'`, `'hex'`, `'kde'`,
  `'reg'`, `'resid'`.
- `publiplots.themes.colors.resolve_continuous_cmap(cmap=None,
  color=None)` — new helper shared between 2D distribution primitives
  (`pp.histplot` 2D + `pp.hexbinplot`). When `cmap` is `None`, builds
  a light sequential gradient from `color` (or `pp.rcParams["color"]`
  when `color` is also `None`); when `cmap` is a string or `Colormap`
  instance, it round-trips through `matplotlib.colormaps`.
- Gallery sections demonstrating the new 2D mode:
  - `examples/plots/plot_02_histogram.py` — "2D Histogram
    (Bivariate)" using a Gaussian mixture, with a default-cmap panel,
    a `cmap="magma"` override panel, and a `hue=` panel showing the
    per-level stacked colorbars.
  - `examples/plots/plot_09_jointgrid.py` — "Histogram Jointplot"
    section using `pp.jointplot(kind="hist")`.

### Changed

- `pp.hexbinplot`'s default cmap now also derives from
  `pp.rcParams["color"]` (via the new
  `resolve_continuous_cmap` helper) instead of
  `rcParams["image.cmap"]`. This unifies the themed look across
  publiplots' 2D distribution primitives — `pp.histplot(x=, y=)`
  and `pp.hexbinplot(...)` both pick up the same palette-derived
  gradient when `cmap=` is omitted.
- `pp.histplot` 2D + `hue=` now stashes one continuous-hue colorbar
  per hue level (e.g. `count [A]` / `count [B]`) instead of the
  categorical rectangles used in 1D. Each colorbar reflects that
  level's own count magnitude, preserving per-subgroup density
  information that a single shared norm would flatten.

### Fixed

- `pp.histplot(x=, y=, kde=True)` previously silently no-op'd —
  seaborn's `histplot` does not draw 2D KDE contours from `kde=True`
  (that's a `kdeplot` feature), so the kwarg was forwarded but
  produced no contour artist. The 2D path now raises
  `NotImplementedError` with a pointer to `pp.kdeplot` for 2D KDE
  contours, matching the existing 2D guards on `annotate=` and
  `element != "bars"`. The 1D `kde=True` path is unchanged
  (PR #160).

## [0.11.2] - 2026-05-13

### Fixed

- `pp.boxplot` and `pp.violinplot` now accept univariate calls
  (only `x=` or only `y=`). Previously these raised `KeyError: None`
  from the annotate-meta cache builder when the missing axis was
  looked up by `None`. Both functions now synthesize a constant
  categorical column for the missing axis so the existing 2D code
  path handles 1D as a degenerate single-category case — `annotate`,
  `hue`, `dodge`, side-clip, and the legend stash all keep working
  in 1D unchanged. The synthetic-side ticks and tick labels are
  dropped post-draw while the spine is preserved so the standalone
  1D frame stays closed (PR #157).

### Added

- `pp.JointGrid.plot_marginals(pp.boxplot)` and
  `pp.JointGrid.plot_marginals(pp.violinplot)` are now first-class
  compositions, unblocked by the 1D-mode fix above. Both forward
  `data` + `x`/`y` + `ax=` to the marginal axes via the existing
  `JointGrid` plumbing. New gallery sections demonstrate the
  combinations:
  - `examples/plots/plot_14_box_plots.py` — "Univariate (1D) Box
    Plot" section.
  - `examples/plots/plot_15_violin_plots.py` — "Univariate (1D)
    Violin Plot" section.
  - `examples/plots/plot_09_jointgrid.py` — "Box Marginals" and
    "Violin Marginals" sections; the "Custom Composition" docstring
    now lists `pp.boxplot` and `pp.violinplot` alongside `pp.histplot`
    and `pp.kdeplot` as 1D-capable marginal functions.
  - Note: `pp.jointplot(kind=)` does not yet expose `kind="box"` /
    `kind="violin"` aliases — only the explicit
    `JointGrid.plot_marginals(pp.boxplot)` /
    `JointGrid.plot_marginals(pp.violinplot)` form works in 0.11.2.

## [0.11.1] - 2026-05-13

### Added

- `pp.errorbarplot(data, x, y, *, xerr=, yerr=, hue=, palette=, ...)`
  — publication-ready scatter-with-uncertainty primitive. Supports x
  and/or y errorbars (1D or 2D), categorical and continuous `hue=`,
  and asymmetric errors via tuple-of-columns `(lo_col, hi_col)`. Marker
  rendering is delegated to `pp.scatterplot`, so the publiplots
  double-layer alpha convention (face transparent, edge opaque),
  palette resolution, legend stash, and continuous-hue colorbar all
  apply for free.
  - **xerr / yerr semantics** mirror matplotlib: `None`, a column name
    (symmetric), a scalar/sequence (raw values), or a 2-tuple of
    column names `(lo, hi)` (asymmetric).
  - **Stem color** matches the marker:
    - Categorical `hue=` → one `ax.errorbar` call per level with
      `ecolor=palette[level]`.
    - Continuous numeric `hue=` → one `ax.errorbar` call per point
      with `ecolor` resolved from the cmap and norm.
    - No `hue=` → single call with `ecolor=rcParams['edgecolor']`.
    - `errorbar_kws={'ecolor': ...}` always wins (e.g. force a
      uniform neutral stem color on any hue plot).
  - **`background_marker=True`** by default: an opaque background twin
    is drawn behind each marker so the stem does not show through the
    alpha-faded marker face. Pass `background_marker=False` (or via
    `scatter_kws`) for the see-through look.
  - **`capsize`** falls through to `rcParams['capsize']` (currently
    `0.0`); pass `capsize=2` per call for a publication look.
  - Layered below the scatter at `zorder=1`; markers stay on top.
  - Use cases: cross-experiment measurements with x and y uncertainty,
    calibration curves, bootstrap CI plots with asymmetric bounds.
  - Gallery: `examples/plots/plot_06_errorbarplot.py` (8 worked
    sections).

### Changed

- **Gallery renumbering.** `examples/plots/plot_06_errorbarplot.py`
  inserts at slot 06, immediately after `plot_05_scatter_plots.py`
  (errorbarplot is "scatter + uncertainty"). All slots from 06 onward
  shift by +1: `regplot` 06→07, `residplot` 07→08, `jointgrid`
  08→09, `pointplot` 09→10, `lineplot` 10→11, …, `legend_placement`
  23→24. Intra-gallery `:doc:` cross-references updated to the new
  slot numbers; skill-file references in `skills/publiplots-guide/`
  and `skills/legend-placement/` bumped to match.

- **Plugin skill — `skills/publiplots-guide/SKILL.md`** —
  `errorbarplot` added to the Plots inventory and the Relational
  family grouping. Gallery references updated to the new slot numbers
  (per the persistent rule that release PRs must keep the Claude Code
  plugin discovery surface in sync with the public API).

## [0.11.0] - 2026-05-13

### Added

- `pp.JointGrid` and `pp.jointplot` — bivariate + marginal-distribution
  composition (PR #145). Three-axes layout: large bivariate main panel
  with thin top and right marginals, hidden top-right corner, shared
  axes, and symmetric joint↔marginal gaps at any height.
  - `pp.JointGrid(data, *, x, y, height=80, ratio=5, space=None)` —
    the primitive. Exposes `fig`, `ax_joint`, `ax_marg_x`, `ax_marg_y`.
    Methods `.plot_joint(fn, **kw)` / `.plot_marginals(fn, **kw)` /
    `.plot(joint_fn, marginal_fn, **kw)` forward `data`+`x`/`y`+`ax=`
    to any compatible `pp.*` plot function. Chainable.
  - `pp.jointplot(data, *, x, y, kind='scatter', ...)` — convenience
    wrapper. `kind` in `{'scatter', 'hex', 'kde', 'reg', 'resid'}`
    maps to (joint_fn, marginal_fn). Returns the constructed
    `JointGrid` so per-axes tweaks remain possible. `'hist'` is
    deferred until publiplots ships a 2D-histogram primitive.
  - **Sizing.** `height` is the total grid budget in mm (square);
    matches seaborn's `JointGrid(height=)` modulo the unit. `ratio`
    is the marginal-to-main split (default `5`, matching seaborn's
    default). `space` is the inter-panel gap in mm — `None` (default)
    auto-scales as `height * 0.025` so the gap reads well across
    sizes; an explicit value is taken as absolute mm for cross-figure
    consistency.
  - **Joint-panel legend routing.** Continuous-hue colorbar entries
    stashed by `pp.hexbinplot` (or any 2D plot with a colorbar) on the
    joint panel are auto-routed to a figure-level band anchored on the
    right marginal — keeping the joint↔right-marginal pair flush.
    Zero layout cost when no colorbar is stashed.

- `pp.subplots` per-position lock/auto reservations via tuples-with-None.
  Tuples passed to `title_space`, `xlabel_space`, `ylabel_space`, or
  `right` may now contain `None` entries to opt that position into
  auto-measurement while locking the others (PR #145).

      pp.subplots(2, 2, ...,
          xlabel_space=(0.0, None),    # row 0 locked to 0; row 1 auto
          ylabel_space=(None, 0.0),    # col 0 auto; col 1 locked to 0
      )

  Each `None` resolves to the rcParams default at the `pp.subplots`
  boundary; `FigureLayout` still receives a tuple of floats and its
  data invariants are unchanged. Locked indices are tracked in
  `SubplotsAutoLayout` so the reactor preserves the layout's current
  value at locked positions instead of remeasuring on draw.

  Used by `pp.JointGrid` to lock the four inside-facing reservations
  to 0 mm so the joint↔marginal gaps stay symmetric (the auto-measured
  `tightbbox` baseline padding from corner-most count-axis tick labels
  was previously inflating one gap by ~0.7 mm relative to the other).
  Available to any user composing asymmetric grids that need a flush
  edge somewhere.

### Changed

- **Bumped to 0.11.0 (minor).** The new `JointGrid` / `jointplot`
  surfaces and the `tuples-with-None` extension to `pp.subplots`
  collectively justify a minor bump under semver. Backward
  compatibility on existing `pp.subplots` and all per-side reservation
  forms (scalar / `None` / full-floats-tuple) is preserved bit-for-bit.

## [0.10.12] - 2026-05-13

### Changed

- **Default marker diameter dropped from 6pt to 4pt** across every
  marker-bearing plot (PR #152). 4pt ≈ 1.4mm at 600 dpi — small
  enough that density patterns read through, large enough to land
  as a discrete data point at publication-mm sizes. The new default
  is driven by ``pp.rcParams["lines.markersize"]``, so a single
  rcParam override now propagates to ``pp.scatterplot``,
  ``pp.regplot``, ``pp.residplot``, ``pp.swarmplot``,
  ``pp.stripplot``, ``pp.pointplot``, and ``pp.lineplot``.
  - Previously every plot picked its own size: 5pt (swarm/strip
    hardcode), 6pt (point/line via rcParam), 6pt (scatter — but its
    own ``sizes=(100, 100)`` literal was silently dropped because
    seaborn ignores ``sizes=`` when ``size=`` is None, so seaborn's
    36 pt² fallback won). Two latent bugs fixed as part of the
    rewiring (see "Fixed" below).
  - The scatter family squares the rcParam internally to get
    matplotlib's points² area; everything else uses it as a
    diameter directly.
  - User-supplied ``size=`` (swarm/strip), ``scatter_kws['s']=``
    (regplot/residplot), or ``s=`` (scatterplot) still wins via
    ``setdefault``.

### Fixed

- ``pp.scatterplot``'s default size now actually reaches the canvas
  (PR #152). The previous ``sizes=(100, 100)`` literal at
  ``scatter.py:250`` was dropped silently because seaborn ignores
  ``sizes=`` when ``size=`` is None — seaborn's 36 pt² fallback won
  unnoticed. Replaced with a top-level ``s=`` kwarg that seaborn
  honors.

- ``pp.swarmplot`` / ``pp.stripplot`` defaults now read from
  ``pp.rcParams["lines.markersize"]`` instead of being hardcoded
  to 5 (PR #152). Signature default changed from ``size: float = 5``
  to ``size: Optional[float] = None``.

- ``pp.regplot(x_estimator=, x_bins=)`` errorbar linewidth and
  binned-point area no longer follow seaborn's hardcoded
  ``rcParams['lines.linewidth'] * 1.75`` (errorbar) / ``s=50``
  (point) values — both are now driven by publiplots conventions
  (PR #152). Errorbars are reset to ``rcParams['lines.linewidth']``
  and binned points to ``rcParams['lines.markersize']²`` post-draw,
  with user overrides via ``line_kws['linewidth']`` and
  ``scatter_kws['s']`` preserved.

## [0.10.11] - 2026-05-12

### Added

- `pp.kdeplot` — univariate and bivariate kernel-density plot wrapping
  `seaborn.kdeplot` with publiplots styling and legend integration
  (PR #146).
  - **1D mode** (exactly one of `x` / `y`): density curve with optional
    `hue`-grouped overlays. `multiple="layer" | "stack" | "fill"`
    controls overlap handling. Legend stashes one categorical-hue entry
    whose handle shape follows the drawn geometry — `Rectangle` when
    `fill=True`, `Line2D` when `fill=False`.
  - **2D mode** (both `x` and `y`): filled or line contours. `cbar=True`
    auto-fills contour bands with the resolved cmap (defaults to
    `matplotlib.rcParams["image.cmap"]`) and stashes a continuous-hue
    `ScalarMappable` through the shared legend reactor, so
    `pp.legend(side=...)`, figure-level bands, and
    `legend_kws={"inside": True}` all work without plot-specific code
    (matching `pp.hexbinplot`). Categorical `hue` stashes a line-handle
    entry per group.
  - New `cmap=` kwarg (2D, no-hue) overrides the rcParams default.
  - Preserves seaborn's `fill=None` tri-state (seaborn derives `fill`
    from `multiple` when unset); `bw_method`, `bw_adjust`, `gridsize`,
    `cut`, `clip`, `common_norm`, `common_grid`, `cumulative`,
    `weights`, `log_scale`, `levels`, `thresh`, and `warn_singular` are
    all forwarded verbatim.

- `pp.regplot` — linear / polynomial / LOWESS / robust regression plot
  wrapping `seaborn.regplot`, extended with native `hue=` support
  (PR #146).
  - `sns.regplot` itself has no `hue` parameter (users are expected to
    reach for `sns.lmplot` + FacetGrid). `pp.regplot` resolves a
    categorical palette via `resolve_palette_map` and loops over hue
    levels, calling `sns.regplot` once per group with the group's
    `color`. One `LegendEntry(kind="hue")` is stashed with marker-style
    swatches — the regression line color matches the scatter swatch.
  - `scatter_kws` / `line_kws` are forwarded via `setdefault` so user
    values win over publiplots defaults for `linewidths`, `alpha`, and
    `edgecolor`. `marker` is routed as a top-level kwarg (seaborn
    hard-overwrites `scatter_kws["marker"]`).
  - Full seaborn surface: `order`, `logistic`, `lowess`, `robust`,
    `logx`, `x_estimator`, `x_bins`, `x_ci`, `x_partial`, `y_partial`,
    `truncate`, `x_jitter`, `y_jitter`, `ci`, `n_boot`, `seed`.
  - Continuous (numeric) hue emits `UserWarning` and falls back to a
    single-color regression.

- `pp.residplot` — residuals-vs-fitted plot wrapping `seaborn.residplot`
  with the same hue-loop extension as `pp.regplot` (PR #146).
  - The `y=0` dotted reference line drawn by seaborn is preserved; with
    `hue` each group overdraws an identical reference line (visually
    indistinguishable from a single draw).
  - `lowess`, `order`, `robust`, `dropna`, `x_partial`, `y_partial`
    forwarded verbatim. Legend stash matches `pp.regplot` — one
    marker-style hue entry per group, no entry when `hue` is None.
  - Continuous hue warns + falls back to single-color residuals.

- New optional dependency group `regression` (`pip install
  publiplots[regression]`) pulls in `statsmodels>=0.14.0`. Required at
  call time when passing `lowess=True`, `robust=True`, or
  `logistic=True` to `pp.regplot` / `pp.residplot` (matches seaborn's
  optional-statsmodels stance — the underlying primitives raise
  `RuntimeError` from `_check_statsmodels` when missing). Gallery
  sections that use `lowess=True` are guarded with a `try: import
  statsmodels` block so the docs build whether or not the extra is
  installed (PR #146).

### Fixed

- `pp.kdeplot(x=, y=, cbar=True)` (no hue) now actually colors contour
  bands by density. Previously the plot drew unfilled contour lines
  all in a single color while the colorbar advertised a density
  gradient — the colorbar promised a mapping the contours did not
  encode (the same issue exists in seaborn's native `cbar=True`).
  publiplots now defaults to `fill=True` in this case (overridable
  with explicit `fill=False`) and uses the resolved continuous cmap
  for the colorbar mappable, not the discrete `light_palette` seaborn
  builds on the contour set (PR #146).

- `pp.hexbinplot` is now listed under Distribution in the API
  reference. The function shipped in v0.10.10 but
  `docs/source/api/plotting.rst` was not updated alongside the
  implementation, so the auto-generated reference index didn't
  surface it (PR #147).

### Changed

- Gallery reorganized to keep thematically related plots adjacent:
  `plot_03_kdeplot.py` inserted right after `plot_02_histogram.py` (1D
  density sits with the histogram, 2D sits with `plot_04_hexbinplot.py`);
  `plot_06_regplot.py` and `plot_07_residplot.py` inserted right after
  `plot_05_scatter_plots.py` (regression is "fit a line through the
  cloud"). Existing `plot_03` through `plot_19` shifted to `plot_04`
  through `plot_22`. Four intra-gallery cross-references to
  `plot_18_annotate` updated to `plot_21_annotate` (PR #146).

- `docs/source/sg_execution_times.rst` (sphinx-gallery's per-build
  timing table) is now gitignored. The file regenerates on every
  `make html` with fresh per-machine values, so committing it created
  spurious diffs. `docs/source/auto_examples/` (sphinx-gallery's
  generated rST output) is also gitignored as a defensive measure
  (PR #148).

## [0.10.10] - 2026-05-12

### Added

- `pp.hexbinplot` — new bivariate 2D-density plot. Aggregates an
  `(x, y)` point cloud into a hexagonal grid and colors each cell by
  count (default) or by a user-supplied reduction of a third column
  (`C=` + `reduce_C_function=`). Supports `bins="log"` for heavy-tailed
  densities and `mincnt=` to hide sparse cells (PR #141).
  - Sits in the plot surface as the 2D sibling of `pp.histplot`. Reach
    for it when scatter is too dense to read as points.
  - Legend integrates with the shared reactor: one continuous-hue
    `ScalarMappable` stashed via `stash_continuous_hue`, so
    `pp.legend(side=...)`, `legend_kws={"inside": True}`, and figure-
    level bands all work without plot-specific legend code.
  - `alpha` defaults to `1.0` (literal, not `rcParams["alpha"]=0.1`):
    hex cells are solid density patches, not layered markers, so the
    scatter-tuned rcParams default would wash them out.
  - `gridsize=30` default (matplotlib's 100 is too fine at publiplots'
    mm-sized axes); `cmap=None` falls back to
    `matplotlib.rcParams["image.cmap"]`; `edgecolor=None` coerces to
    `"none"` (stroking thousands of cells rarely reads well).

### Changed

- Gallery reorganized: `plot_03_hexbinplot.py` inserted right after
  `plot_02_histogram.py` so 1D and 2D density sit together. Existing
  `plot_03` through `plot_18` renumbered to `plot_04` through
  `plot_19`. Four intra-gallery cross-references to
  `plot_17_annotate` updated to `plot_18_annotate`.

## [0.10.9] - 2026-05-11

### Added

- `pp.barplot(multiple="stack"|"fill", stack_by=...)` — stacked and
  100%-stacked bar plots with publiplots styling, palette/hatch
  handling, legend stash, and per-segment annotate labels.
  - Default `multiple="dodge"` preserves current seaborn-backed
    behavior exactly. `"stack"` and `"fill"` take a parallel code path
    that draws with raw `ax.bar` / `ax.barh` at integer category
    positions with cumulative `bottom=` (or `left=` for horizontal)
    — seaborn's `barplot` has no stacked mode.
  - **Single-dimension stacking**: driven by whichever of `hue` / `hatch`
    is set and distinct from the categorical axis. `hue == hatch`
    (patterns overlaid on colored swatches) is allowed.
  - **Dual-dimension stacking** (`hue` + `hatch` as distinct columns):
    pass `stack_by="hue"` or `stack_by="hatch"` to pick the stack
    dimension — the other is dodged side-by-side within each category.
    Each category then shows `N_dodge` sub-stacks of `N_stack` segments.
    `stack_by` is required in this case; missing it raises `ValueError`.
  - Errorbars are dropped with a `UserWarning` — per-segment errors
    aren't additive without covariance info.
  - `annotate=True` defaults to `anchor="inside"` and produces one
    label per drawn segment (including the dual-dim case, which emits
    `N_cat × N_hue × N_hatch` labels). Override with
    `annotate={"anchor": "outside"}` for per-segment top-edge labels.
  - Legend stashing reuses the dodge path's four-case dispatcher: one
    hue entry (single-dim hue or `hue == hatch`), one hatch entry
    (hatch-only), or both (dual-dim + `stack_by`).
  - `hue_order` / `hatch_order` now correctly drive **both** the stack
    order (bottom-to-top) *and* the legend entry order on both the
    dodge and stacked paths — previously the legend ignored the
    ordering kwargs in favor of palette-resolution order.

- `pp.barplot(multiple="gain")` — pairwise comparison bars for exactly
  two levels. Per category: bottom segment = `min` of the two values
  (loser color), top segment = `max - min` (winner color). Bar top
  = `max`. Annotate shows absolute values; ties render as a single
  bar; missing-level-at-a-category raises `ValueError`. Works
  side-by-side with a second non-cat column via `stack_by=`. Useful
  for summary metrics (AUC, accuracy, F1) where additive stacking is
  semantically wrong.
  - `annotate=True` defaults to **mixed per-segment anchoring** on
    gain: winner label floats above the bar top (`anchor="outside"`),
    loser label sits inside the base segment (`anchor="inside"`), tie
    label sits inside the single rect. Pass `annotate={"anchor": ...}`
    to override uniformly. Stack/fill defaults stay `"inside"` for
    all segments.
  - New `anchor_override` field on `BarRecord` lets annotate strategies
    resolve anchor per-bar (consumed by the gain-mode meta builder;
    transparent to existing callers).

## [0.10.8] - 2026-05-10

### Fixed

- `pp.annotate` label positions now stay stable when the axis limits
  change after annotation — e.g. an explicit `ax.set_xlim/ylim`, or the
  implicit relim triggered by `pp.subplots(..., sharex=True)` /
  `sharey=True` when a neighboring axes has a wider value range.
  - Previously, the `offset_mm` was converted to a data-coord delta at
    annotate time using the *current* transform and baked into the
    text's (x, y). When the limits later expanded (commonly via a bigger
    sharey neighbor), the visual pixel gap between label and bar edge
    shrank — on tall neighbors the shorter pane's labels could drift so
    close that they touched the bar tops.
  - Labels are now placed at the bar edge in data coords and the mm
    offset is applied through a `ScaledTranslation(dx_mm, dy_mm)`
    display-space transform. The gap is constant in physical units
    across any downstream transform change (including dpi).
  - `resolve_anchor` / `_resolve_box_anchor` / `_resolve_point_anchor`
    now return `(x, y, dx_mm, dy_mm, ha, va)` instead of `(x, y, ha, va)`
    — callers use the new `make_offset_transform(ax, dx_mm, dy_mm)`
    helper to build the text's transform.

### Added

- `pp.annotate(rotation=...)` — first-class text rotation on bar-value,
  box-stat, and point-value annotations.
  - `rotation` is a promoted kwarg on `pp.annotate(...)` alongside
    `fmt`, `anchor`, `offset`, `color`, `pad`. Validated via
    `math.isfinite`. Default `0.0` — fully backward compatible.
  - matplotlib applies `(ha, va)` to the **post-rotation** bbox, so the
    alignment that `resolve_anchor` returns for a given anchor kind
    already positions the rotated text correctly (`va='bottom'` at
    rotation=90° still anchors the text at its bbox's lower edge in
    screen space). No alignment remap is performed.
  - Under rotation, the limit-expansion helpers in `bar_values`,
    `box_stats`, and `point_values` now expand **both** the value and
    categorical axes (previously only the value axis). Prevents
    rotated labels from clipping neighbors on narrow bars or from
    overflowing the frame.
  - `_expand_axis` now handles matplotlib's inverted-axis convention
    (seaborn draws horizontal barplots with `ylim[0] > ylim[1]`) so
    expansion preserves the original axis direction instead of
    flipping it.
  - Primary use case: narrow vertical bars where the horizontal label
    doesn't fit. `pp.barplot(..., annotate={"rotation": 90})` now
    renders clean vertical labels without manual `ha`/`va` overrides.

## [0.10.7] - 2026-05-08

### Added

- `pp.histplot` — publication-ready histograms with seaborn-parity API
  and the full publiplots styling pipeline.
  - Univariate `x=` or `y=`; counts, densities, probability, percent,
    and frequency stats; `bins=`, `binwidth=`, `binrange=`, `discrete=`,
    `cumulative=`, `log_scale=`; full `multiple=` support (`layer`,
    `dodge`, `stack`, `fill`); `element=` for `bars`, `step`, or `poly`;
    `kde=True` overlay with per-hue colors; `weights=` for weighted
    histograms.
  - Palette resolution via `pp.color_palette` / dict / hue_order,
    matching `barplot` / `lineplot` conventions.
  - Optional `hatch=` + `hatch_map=` for B&W-print-friendly overlays
    under `multiple="layer"`. Raises `NotImplementedError` when `hatch`
    is combined with `hue` + `multiple="dodge"` (v1 scope).
  - Full legend-stash integration: `pp.legend_group` can claim the
    legend entries across axes exactly as it does for `barplot` and
    `lineplot`. `legend=False` and `legend={"hue": False}` supported.
  - `annotate=True` (or a dict of options) adds count/height labels
    per bar via the shared `annotate(ax, kind="bar_values", ...)`
    strategy. `element="bars"` only in v1.
  - Returns `Axes`, matching every other `pp.*plot` function.

## [0.10.6] - 2026-05-08

### Added

- `pp.boxplot(border_radius=...)` — rounded corners for the IQR box.
  - Same shape as `pp.barplot(border_radius=...)` from 0.10.5: a
    scalar rounds all four corners symmetrically; a
    `(top_mm, bottom_mm)` tuple rounds top and bottom independently
    (e.g. `border_radius=(1.5, 0)` keeps the Q1 edge square — useful
    when the box is visually paired with a density cloud).
  - Radii are specified in **millimeters**, print-consistent and
    independent of the data-axis range. Horizontal-orient boxplots
    round cleanly too.
  - Global control via the new `box.border_radius` rcParam.
  - Whiskers, caps, medians, and outlier markers are untouched —
    only the IQR box body is rounded.
- `pp.raincloudplot` — rounded inner-box via auto-propagation. The
  raincloud's box component is drawn by `pp.boxplot` internally, so
  setting `pp.rcParams['box.border_radius'] = 1.5` (or passing
  `box_kws={'border_radius': ...}` on a single call) rounds the
  raincloud's IQR box with no additional wiring.

### Changed

- `publiplots.utils.rounding.apply_border_radius` now accepts
  `PathPatch` inputs in addition to `Rectangle`. Seaborn 0.13+ draws
  boxplot IQR boxes as `PathPatch`, so the helper dispatches on patch
  type and derives `(x, y, w, h)` from the path's axis-aligned
  bounding box. Degenerate patches (non-positive width/height) are
  skipped. `pp.barplot` behavior is unchanged.

### Notes

- `pp.violinplot(inner='box')` stays flat. A dedicated
  `violin.inner_box.border_radius` knob is a future-work candidate.
- Notched boxplots are not rounded specially — the path's bounding
  box is used, which effectively ignores the notch geometry. Flat
  notches remain the recommended combination.

## [0.10.5] - 2026-05-08

### Added

- `pp.barplot(border_radius=...)` — rounded corners for bar plots.
  - Scalar `float` rounds all four corners symmetrically; a
    `(top_mm, bottom_mm)` tuple rounds top and bottom independently
    (e.g. `border_radius=(1.5, 0)` for infographic-style bars with
    rounded tops and a flat baseline).
  - Radii are specified in **millimeters** — print-consistent and
    independent of the y-axis data range. Draw-time conversion to
    data coordinates handles non-square aspect ratios cleanly, so
    corners stay visually circular on any plot.
  - Global control via the new `bar.border_radius` rcParam — set
    `pp.rcParams['bar.border_radius'] = 1.5` once and every
    subsequent `pp.barplot` call picks it up (passing
    `border_radius=0` on an individual call opts out).
  - Preserves face color, edge color, hatch, hatch linewidth, alpha,
    zorder, and label on the swap from `Rectangle` to the internal
    `_RoundedBarPatch`.
  - Gallery: see the new "Rounded bars" section in
    `plot_01_bar_plots.py`.
- `publiplots.utils.rounding` — new module with
  `apply_border_radius(patches, radius_mm, ax)` and
  `normalize_border_radius(value)`, designed to generalize to other
  `Rectangle`-based plot primitives in future releases.

## [0.10.4] - 2026-05-08

### Added

- `pp.suptitle` now reserves vertical space via the auto-layout engine:
  the figure grows to fit the title instead of overlapping the top row
  of axes. New `FigureLayout.suptitle_space` scalar sits between
  `outer_pad` and `legend_band_top` in the figure H formula; auto-measured
  from the Text's tight bbox on `draw_event`. Repeated `pp.suptitle`
  calls replace the prior title rather than stacking. A `side="top"`
  figure-anchored `pp.legend` band now sits below the reserved suptitle
  band — both coexist cleanly (PR #128).

### Changed

- `figure.titlesize` default bumped to `11` (was matplotlib's `"large"`,
  which resolves to 9.6pt at publiplots' 8pt base — smaller than the
  10pt panel titles, a flipped hierarchy). `figure.titleweight` stays
  `"normal"`. The figure-level suptitle now reads as the outermost
  heading in the type stack (PR #128).
- Dropped the now-stale `y=1.02` nudge from the three `pp.suptitle`
  calls in `examples/plots/plot_12_heatmap.py` — auto-layout positions
  the title correctly regardless (PR #128).

[0.11.3]: https://github.com/jorgebotas/publiplots/releases/tag/v0.11.3
[0.11.2]: https://github.com/jorgebotas/publiplots/releases/tag/v0.11.2
[0.11.1]: https://github.com/jorgebotas/publiplots/releases/tag/v0.11.1
[0.11.0]: https://github.com/jorgebotas/publiplots/releases/tag/v0.11.0
[0.10.12]: https://github.com/jorgebotas/publiplots/releases/tag/v0.10.12
[0.10.11]: https://github.com/jorgebotas/publiplots/releases/tag/v0.10.11
[0.10.10]: https://github.com/jorgebotas/publiplots/releases/tag/v0.10.10
[0.10.9]: https://github.com/jorgebotas/publiplots/releases/tag/v0.10.9
[0.10.8]: https://github.com/jorgebotas/publiplots/releases/tag/v0.10.8
[0.10.7]: https://github.com/jorgebotas/publiplots/releases/tag/v0.10.7
[0.10.6]: https://github.com/jorgebotas/publiplots/releases/tag/v0.10.6
[0.10.5]: https://github.com/jorgebotas/publiplots/releases/tag/v0.10.5
[0.10.4]: https://github.com/jorgebotas/publiplots/releases/tag/v0.10.4

## [0.10.3] - 2026-05-08

### Changed

- Docstring overhaul across every public `pp.*` function and the
  Sphinx API reference:
  - Completed the API-reference toctrees — every name in
    `publiplots.__all__` now has an entry (previously missing:
    `lineplot`, `pointplot`, `violinplot`, `raincloudplot`, `heatmap`,
    `complex_heatmap`, `dendrogram`, `subplots`, `show`, `suptitle`,
    `rotate`, `invert_axis`, `MultiAxesLegendGroup`,
    `HandlerLineMarker`, `LineMarkerPatch`, `STANDARD_MARKERS`,
    `HATCH_PATTERNS`).
  - Documented publiplots-specific features that differentiate from
    seaborn: `errorbar=('custom', (lo, hi))` on `pp.lineplot` /
    `pp.pointplot`, `pp.subplots` mm-units + `reject_figsize`,
    `pp.savefig` `bbox_inches=None` default (0.9.3+),
    `pp.scatterplot` `background_marker`, `pp.raincloudplot`
    composite (half-violin + box + strip), `pp.heatmap` /
    `pp.complex_heatmap` clustering + dot-heatmap mode.
  - Added a **Notes** section to `pp.legend` documenting the
    flagship 0.10 asymmetry: `pp.legend(ax)` positional (internal
    per-axes legend) vs `pp.legend(anchor=ax)` keyword (external
    band overhang).
  - Rewrote `pp.barplot` `hatch` / `hatch_map` params with the
    "second categorical via texture" mental model; cross-references
    to `pp.set_hatch_mode`, `pp.list_hatch_patterns`,
    `pp.HATCH_PATTERNS`.
  - Aligned docstring defaults with actual signatures (drift fixes
    in `pp.raincloudplot`, `pp.venn`, `pp.scatterplot`, `pp.savefig`
    `transparent`, DPI).
  - Fixed broken `See Also` blocks (e.g. removed phantom
    `barplot_enrichment` reference).
  - Removed stale references to 0.10-removed API (`pp.legend_group`,
    `pp.legend(ax, auto=False)`).

### Fixed

- Several Sphinx-rendering issues in docstrings:
  `pp.pointplot` `errorbar` nested-bullet indentation,
  `pp.barplot` `hatch_map` malformed escape example, single-backtick
  inline literals throughout.

## [0.10.2] - 2026-05-07

### Added

- `pp.lineplot` now accepts `errorbar=('custom', (lower_col,
  upper_col))` for rendering precomputed CI bands (e.g. from a LOESS
  bootstrap, a GAM fit, or a Bayesian posterior) directly as
  `err_style='band'` — no manual `fill_between` loop per group
  required. Matches the existing `pp.pointplot` API. See
  `examples/plots/plot_04_lineplot.py` for a full "raw scatter +
  smooth + CI band" composition.
- Shared `publiplots.utils.errorbar.format_for_custom_errorbar`
  helper backing both `pp.lineplot` and `pp.pointplot`.

### Fixed

- `pp.pointplot` with `errorbar=('custom', ...)` no longer crashes
  when both x and y are numeric and `orient` is left at the default
  `None`. Previously the internal `_format_for_custom_errorbar`
  helper called `orient.isin(...)` on a `str`/`None`, which only
  avoided an `AttributeError` when one of the axes happened to be
  categorical.

[0.10.2]: https://github.com/jorgebotas/publiplots/releases/tag/v0.10.2

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
