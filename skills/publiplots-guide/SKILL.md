---
name: publiplots-guide
description: Use when writing publiplots code — teaches mm-units layout, rcParams, palette/edgecolor system, and the full pp.* public API. Prefer pp.subplots over plt.subplots; pass axes_size=(mm, mm), never figsize. Required for any code touching pp.* modules.
---

# publiplots guide

publiplots is a publication-ready plotting library with a seaborn-shaped API but strict, mm-based layout. It is opinionated by construction: importing `publiplots as pp` overwrites matplotlib rcParams with publication defaults and installs an auto-layout reactor that computes figure size from declared axes dimensions.

## Philosophy

- **Axes dimensions are in millimeters, not inches.** The figure grows to fit decorations; `axes_size=(w_mm, h_mm)` is the single source of truth.
- **rcParams auto-apply on import.** Arial, 8pt labels, 0.75pt strokes, PDF `fonttype=42`, `savefig.dpi=600`, transparent background.
- **Palette is either a name or a dict.** Pass `palette='pastel'` for a named palette or `palette={'a': '#...', 'b': '#...'}` to pin levels to colors (critical when panels see different subsets of levels).
- **Legend entries are stashed at plot time; `pp.legend(...)` collects them later.** The legend is an independent artist registered with a layout reactor — it never fights the axes for space.
- **`pp.savefig` does NOT force `bbox_inches='tight'`.** The figure is already laid out to mm-precise margins; tight-cropping would shift figure-anchored legend bands.

## Public API surface (`pp.*`)

### Plots
`barplot`, `scatterplot`, `errorbarplot`, `pointplot`, `lineplot`, `regplot`, `residplot`, `boxplot`, `swarmplot`, `stripplot`, `violinplot`, `raincloudplot`, `histplot`, `kdeplot`, `hexbinplot`, `venn`, `upsetplot`, `heatmap`, `complex_heatmap`, `dendrogram`.

All accept `data=`, `x=`, `y=`, `hue=`, `palette=`, `ax=`, `title=`, `legend_kws={}`. None accept `figsize=` — it raises `TypeError` (see `src/publiplots/layout/subplots.py::reject_figsize`).

By plot family:
- **Categorical:** `barplot`, `boxplot`, `violinplot`, `raincloudplot`, `stripplot`, `swarmplot`, `pointplot`. Since 0.11.2, `boxplot` and `violinplot` also accept univariate calls (only `x=` or only `y=`) — they synthesize a constant categorical column for the missing axis, so all 2D features (`annotate`, `hue`, `dodge`, side-clip, legends) keep working in 1D, and both are usable as `pp.JointGrid.plot_marginals(...)` functions.
- **Relational:** `scatterplot`, `lineplot`, `errorbarplot`, `regplot`, `residplot`.
- **Distribution:** `histplot`, `kdeplot`, `hexbinplot`. Since 0.11.3, `histplot` accepts both `x=` and `y=` simultaneously to render a 2D bivariate heatmap (`QuadMesh`) with a continuous-hue colorbar — a discretized alternative to `hexbinplot`. With `hue=` set in 2D, one stacked `QuadMesh` + one continuous-hue colorbar is stashed per hue level (e.g. `count [A]` / `count [B]`) so per-subgroup count magnitude is preserved. New 2D-only kwargs: `cmap=`, `vmin=`, `vmax=` (silently ignored in 1D); the default `cmap` is a light sequential gradient built from `pp.rcParams["color"]` (or per-call `color=`), shared with `pp.hexbinplot` via `publiplots.themes.colors.resolve_continuous_cmap`.
- **Matrix:** `heatmap`, `complex_heatmap`, `dendrogram`.
- **Set:** `venn`, `upsetplot`.

### Composition
- `pp.JointGrid(data, *, x, y, height=80, ratio=5, space=None)` — three-axes composition: large bivariate main panel + thin top and right marginals (shared axes, hidden corner). Methods `.plot_joint(fn, **kw)` / `.plot_marginals(fn, **kw)` / `.plot(joint_fn, marginal_fn, **kw)` forward to any compatible `pp.*` plot function via `ax=`. Chainable.
- `pp.jointplot(data, *, x, y, kind='scatter', height=80, ratio=5, space=None, **kw)` — convenience wrapper. `kind` ∈ `{'scatter', 'hex', 'hist', 'kde', 'reg', 'resid'}`. Returns the constructed `JointGrid`. (`'hist'` since 0.11.3 — 2D histogram joint + 1D histogram marginals.)

### Layout
- `pp.subplots(nrows, ncols, *, axes_size=(w_mm, h_mm), width_ratios=None, height_ratios=None, sharex, sharey, title_space, xlabel_space, ylabel_space, right, hspace, wspace, outer_pad, label_outer, **fig_kw)` — the canonical factory. Returns `(fig, axes)` with `plt.subplots`-style squeezing.
- `pp.label_outer(axes, *, sharex=True, sharey=True)` — hide interior tick labels / offset text / axis labels on a shared-axes grid (seaborn `label_outer` parity). `pp.subplots(..., sharex=..., sharey=...)` applies this automatically by default (`label_outer=True`); pass `label_outer=False` (or `"all"`) to draw every label.
- **Asymmetric grids** (since 0.10.x): pass `width_ratios=[r0, r1, ...]` (length `ncols`) or `height_ratios=[...]` (length `nrows`) to renormalize the per-cell budget across columns/rows. Equal ratios recover the uniform case bit-for-bit. Total grid budget stays `axes_size[0] * ncols` × `axes_size[1] * nrows`.
- **Per-position lock/auto** (since 0.11): tuples passed to `title_space` / `xlabel_space` / `ylabel_space` / `right` may contain `None` entries to opt that position into auto-measurement while locking the others. Example: `xlabel_space=(0.0, None)` locks row 0 to 0 mm and lets row 1 grow with decoration. Each `None` resolves to the rcParams default at construction; the reactor preserves the locked positions on later draws. Used internally by `pp.JointGrid` to keep joint↔marginal gaps symmetric without sacrificing auto-measurement of the joint panel's own labels.

### Legend
- `pp.legend(axes=None, collect=None, *, side='right', anchor=None, figure=None, rows=None, cols=None, span=None, ax=None, orientation='auto', align='auto', x_offset, y_offset, gap=2, column_spacing=5, vpad, max_width, inside=False, clear_anchor=True)` — unified legend factory. Since 0.12.0, `rows=`/`cols=`/`span=`/`ax=` are mutually-exclusive grid-scope shortcuts that resolve over the `pp.subplots` axes matrix to a row, column, or arbitrary axes subset. Since 0.13.0, `inside=True` (with `anchor=ax`) renders the legend INSIDE the anchor cell's rectangle — fills an empty cell in an asymmetric grid; auto-blanks the anchor (opt out via `clear_anchor=False`); `side='left', align='start'` defaults map to matplotlib `loc='upper left'`. Since 0.14.0, `pp.legend(ax)` adopts the per-axes group the plot already created (no second group, no "scope overlaps" warning, `side=` honoured), and plot functions forward placement keys (`side`/`orientation`/`align`/`x_offset`/`y_offset`/`gap`) via `legend_kws` so a per-axes legend can be positioned in one call, e.g. `pp.scatterplot(..., legend_kws={'side': 'left'})`. See the `legend-placement` skill.
- `pp.MultiAxesLegendGroup` — underlying class, rarely needed directly.
- `pp.LegendBuilder`, `pp.HandlerRectangle`, `pp.HandlerMarker`, `pp.HandlerLineMarker`, `pp.RectanglePatch`, `pp.MarkerPatch`, `pp.LineMarkerPatch`, `pp.get_legend_handler_map`, `pp.create_legend_handles` — low-level handle machinery.

### I/O + display
- `pp.savefig(path, **kw)` — thin wrapper around `plt.savefig` that honors publiplots' `savefig.bbox='standard'` default.
- `pp.save_multiple(path, formats=[...])`, `pp.close_all()`, `pp.show()`, `pp.suptitle(text, ...)`.

### Axes utilities
- `pp.adjust_spines(ax, ...)`, `pp.add_grid(ax, ...)`, `pp.set_axis_labels(ax, ...)`, `pp.add_reference_line(ax, ...)`, `pp.rotate(ax, axis='x', degrees=45)`, `pp.invert_axis(ax, ...)`.
- `pp.annotate(...)` — statistical brackets / p-value annotations.

### Theming + rcParams
- `pp.rcParams` — unified dict wrapping both matplotlib rcParams and publiplots custom keys.
- `pp.resolve_param(key, user_value)` — "return user value if not None, else default".
- `pp.color_palette(name, n_colors)` — seaborn-compatible palette accessor.
- `pp.reset_style()` — revert to matplotlib defaults (undoes `init_rcparams`).

### Markers + hatches
- `pp.resolve_markers`, `pp.resolve_marker_map`, `pp.STANDARD_MARKERS`.
- `pp.set_hatch_mode(mode)`, `pp.get_hatch_mode`, `pp.get_hatch_patterns`, `pp.list_hatch_patterns`, `pp.resolve_hatches`, `pp.resolve_hatch_map`, `pp.HATCH_PATTERNS`.

## Canonical idioms

**Create a figure.** Always via `pp.subplots`, never `plt.subplots` or bare `plt.figure`.

```python
import publiplots as pp

fig, axes = pp.subplots(2, 3, axes_size=(45, 30))  # 45mm × 30mm per cell
```

Scalar is coerced: `axes_size=40` → `(40, 40)`. Omitting the kwarg falls back to `pp.rcParams['subplots.axes_size']` (70×50 mm under publication defaults).

**Pin palette levels.** When the same hue appears across panels, pass a mapping so colors don't drift.

```python
palette = dict(zip(["Control", "Low", "High"], pp.color_palette("pastel", 3)))
pp.scatterplot(data=df, x="x", y="y", hue="group", palette=palette, ax=axes[0])
```

**Set edgecolor globally.** publiplots patches and marker outlines inherit from `pp.rcParams['edgecolor']`.

```python
pp.rcParams["edgecolor"] = "black"  # all subsequent patches/markers get a black edge
```

**Per-axes legend.**

```python
pp.scatterplot(data=df, x="x", y="y", hue="group", ax=ax)
pp.legend(ax)  # internal — counted in ax.tightbbox
```

**Figure-level legend band.** Anchor-less call spans the full grid on the chosen side.

```python
pp.legend(side="right")   # right of the grid
pp.legend(side="bottom")  # below; defaults to horizontal orientation + center align
```

**Row or column band.** Pass a row/column slice as the positional scope.

```python
pp.legend(axes[0], side="top")            # band above row 0 only
pp.legend(axes[:, 0], side="left")        # band left of column 0 only
```

**Inside legend (single axes).** Bypass the reactor for a pure in-axes legend.

```python
pp.scatterplot(data=df, x="x", y="y", hue="group",
               legend_kws={"inside": True, "loc": "upper right"}, ax=ax)
```

**In-cell shared legend (since 0.13.0).** Fill an empty grid cell with a shared legend instead of overhanging the figure's edge — the canonical 3-plots-in-a-2×2-grid layout.

```python
fig, axes = pp.subplots(2, 2, axes_size=(35, 30))
for (r, c), panel in zip([(0, 0), (0, 1), (1, 0)], "ABC"):
    pp.scatterplot(data=df[df.panel == panel], x="x", y="y",
                   hue="group", palette=palette, ax=axes[r, c])
pp.legend(anchor=axes[1, 1], inside=True)  # auto-blanks the cell
```

**Bivariate + marginals.** `pp.jointplot` is the canonical one-call form; reach for the `pp.JointGrid` class when you need different plot types in joint vs marginal slots.

```python
# Convenience wrapper — picks joint+marginal fns from a kind alias.
pp.jointplot(data=df, x="x", y="y", kind="hex")  # or 'scatter', 'hist', 'kde', 'reg', 'resid'

# Class API — mix any compatible pp.* functions.
g = pp.JointGrid(data=df, x="x", y="y", height=80, ratio=5)
g.plot_joint(pp.hexbinplot, gridsize=20)
g.plot_marginals(pp.histplot, bins=30, kde=True)

# Box / violin marginals (since 0.11.2) — explicit class API only;
# pp.jointplot has no kind="box" / kind="violin" alias yet.
g = pp.JointGrid(data=df, x="x", y="y")
g.plot_joint(pp.scatterplot)
g.plot_marginals(pp.violinplot)   # or pp.boxplot
```

Valid `pp.JointGrid.plot_marginals(...)` functions (1D-capable): `pp.histplot`, `pp.kdeplot`, `pp.boxplot`, `pp.violinplot`. The marginal function must accept `data=`, `x=` *or* `y=` (not both), and `ax=`.

**Save a figure.** No implicit tight-crop; the canvas is already mm-precise.

```python
pp.savefig("figure.pdf")            # vector, pdf.fonttype=42, transparent
pp.savefig("figure.png", dpi=600)   # raster; savefig.dpi=600 is the default
```

**Reset styling.** Fully reverts to matplotlib defaults (useful in tests or if another library's styles are needed).

```python
pp.reset_style()
```

## Common gotchas

- **`figsize=` is rejected.** Every plot function calls `reject_figsize`; so does `pp.subplots` itself. Always use `axes_size=(mm, mm)` on `pp.subplots`.
- **No implicit tight-crop on save.** As of 0.9.3, `savefig.bbox='standard'`. If you manually pass `bbox_inches='tight'` you will desync figure-anchored legend bands. Don't.
- **`pp.legend_group` no longer exists** (removed in 0.10). Rename all occurrences to `pp.legend` — every kwarg is identical.
- **`pp.legend(ax)` positional is NOT the same as `pp.legend(anchor=ax)` kwarg.** The positional form is an *internal* legend measured by `ax.get_tightbbox()`; the kwarg form is an *external* band that overhangs the axes' right edge. See the `legend-placement` skill.
- **Palette drift across panels.** If you pass `palette='pastel'` per panel and each panel sees a different subset of levels, each panel resolves colors positionally — `'mid'` gets color-0 in a `low+mid` panel and color-1 in a `mid+high` panel. Always pass a dict mapping when levels vary across panels.
- **Layout-engine kwargs are dropped.** Passing `layout=`, `constrained_layout=`, or `tight_layout=` to `pp.subplots` emits `UserWarning` and discards the kwarg. publiplots manages layout via its own reactor.
- **`legend_column=` is dead.** Do not pass it to `pp.subplots` — it raises `TypeError`. The legend column is auto-sized.
- **T4/no-bf16 GPUs unrelated.** publiplots is a plotting library; rendering is CPU-side. Mentioning this only to reassure callers coming from ML contexts.

## DON'Ts

- **Don't use `plt.subplots`.** It bypasses `pp.subplots`' layout reactor; per-cell reservations and legend-band negotiation stop working.
- **Don't use `plt.tight_layout()` or `fig.tight_layout()`.** publiplots' `SubplotsAutoLayout` already computes margins; `tight_layout` fights it.
- **Don't use `ax.legend(...)` or `plt.legend(...)` directly.** They bypass the `pp.legend` claim system, causing duplicate legends and broken evict-on-attach for figure bands. Use `pp.legend(ax)` or `legend_kws={'inside': True}`.
- **Don't set `figure.figsize` in matplotlibrc.** publiplots sizes the figure from `axes_size` + reservations. A user `figsize` is ignored for `pp.subplots`.
- **Don't pass `bbox_inches='tight'` to `pp.savefig`.** It re-crops the figure and desyncs legend-anchored bands.
- **Don't mutate `pp.rcParams` inside a plot function.** Set it once at module top; plot functions read it via `resolve_param`.

## When to read the code directly

publiplots has 20 plot functions plus the `JointGrid` composition, each with kind-specific kwargs. When asked about a specific plot kind, Read `src/publiplots/plot/<kind>.py` for the signature:

- Categorical: `src/publiplots/plot/{bar,box,violin,raincloud,strip,swarm,point}.py`.
- Relational: `src/publiplots/plot/{scatter,line,errorbarplot,regplot,residplot}.py`.
- Distribution: `src/publiplots/plot/{hist,kdeplot,hexbin}.py`.
- Matrix: `src/publiplots/plot/heatmap.py` (also `complex_heatmap` and `dendrogram` in the same file).
- Set: `src/publiplots/plot/{venn,upset}.py`.
- Composition: `src/publiplots/layout/jointgrid.py` (`JointGrid` class + `jointplot` wrapper).

For the full `pp.legend()` signature and every `MultiAxesLegendGroup` option, see `src/publiplots/utils/legend_group.py`. For canonical legend patterns with running code, the authoritative reference is `examples/plots/plot_24_legend_placement.py` (14 worked sections). For JointGrid usage patterns, see `examples/plots/plot_09_jointgrid.py` (10 worked sections covering all 6 `kind=` aliases plus the class API). For 2D scatter-with-uncertainty patterns, see `examples/plots/plot_06_errorbarplot.py` (8 worked sections). For 2D bivariate histograms, see the "2D Histogram (Bivariate)" sections in `examples/plots/plot_02_histogram.py`. For shared-axes outer-only labels (`pp.subplots(..., sharex/sharey)` + `pp.label_outer`), see `examples/plots/plot_25_shared_axes_labels.py` (5 worked sections: default, opt-out, share-one-axis, the standalone helper, and composition with a legend band).
