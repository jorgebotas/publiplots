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
`barplot`, `scatterplot`, `pointplot`, `lineplot`, `boxplot`, `swarmplot`, `stripplot`, `violinplot`, `raincloudplot`, `venn`, `upsetplot`, `heatmap`, `complex_heatmap`, `dendrogram`.

All accept `data=`, `x=`, `y=`, `hue=`, `palette=`, `ax=`, `title=`, `legend_kws={}`. None accept `figsize=` — it raises `TypeError` (see `src/publiplots/layout/subplots.py::reject_figsize`).

### Layout
- `pp.subplots(nrows, ncols, *, axes_size=(w_mm, h_mm), sharex, sharey, title_space, xlabel_space, ylabel_space, right, hspace, wspace, outer_pad, **fig_kw)` — the canonical factory. Returns `(fig, axes)` with `plt.subplots`-style squeezing.

### Legend
- `pp.legend(axes=None, collect=None, *, side='right', anchor=None, figure=None, orientation='auto', align='auto', x_offset, y_offset, gap=2, column_spacing=5, vpad, max_width)` — unified legend factory. See the `legend-placement` skill.
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

**Inside legend.** Bypass the reactor for a pure in-axes legend.

```python
pp.scatterplot(data=df, x="x", y="y", hue="group",
               legend_kws={"inside": True, "loc": "upper right"}, ax=ax)
```

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

publiplots has ~15 plot functions, each with plot-specific kwargs. When asked about a specific plot kind, Read `src/publiplots/plot/<kind>.py` for the signature:

- Heatmaps: `src/publiplots/plot/heatmap.py` (plus `complex_heatmap` and `dendrogram` in the same file).
- Distributions: `src/publiplots/plot/{violin,box,swarm,strip,raincloud}.py`.
- Set diagrams: `src/publiplots/plot/{venn,upset}.py`.
- Relational: `src/publiplots/plot/{scatter,line,point,bar}.py`.

For the full `pp.legend()` signature and every `MultiAxesLegendGroup` option, see `src/publiplots/utils/legend_group.py`. For canonical legend patterns with running code, the authoritative reference is `examples/plots/plot_17_legend_placement.py` (14 worked sections).
