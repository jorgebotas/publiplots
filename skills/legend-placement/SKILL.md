---
name: legend-placement
description: Use when deciding where/how to place legends in publiplots — per-axes inside/outside, multi-axes row/column bands, figure-level bands. Covers pp.legend(ax) vs pp.legend(anchor=ax) asymmetry, axes= scope rules, and before/after-plot ordering.
---

# Legend placement in publiplots

`pp.legend` is the single public API for every legend configuration in publiplots 0.10+. The first positional argument is the **scope** (which axes contribute entries); `side=` picks the edge; `anchor=` is a rarely-needed geometric override.

## Decision tree

```
Single axes, legend sits inside the axes frame?
    -> legend_kws={'inside': True, 'loc': 'upper right'}
       on the plot call (bypasses the reactor)

Single axes, legend next to the axes?
    -> pp.legend(ax)               # INTERNAL — counted in ax.tightbbox
    -> pp.legend(anchor=ax)        # EXTERNAL band pinned to ax's edge

Subset of the grid (one row, one column) shares a legend?
    -> pp.legend(axes[0], side='top')         # row band
    -> pp.legend(axes[:, 0], side='left')     # column band

Whole figure shares a legend?
    -> pp.legend(side='right')     # (or 'bottom' / 'top' / 'left')

Empty cell in the grid you want to fill with the shared legend?
    -> pp.legend(anchor=empty_ax, inside=True)             # whole-figure collection
    -> pp.legend(axes=[a,b,c], anchor=empty_ax, inside=True)  # explicit scope

Two independent legends on the same figure?
    -> pp.legend(side='top',    collect=['treatment'])
       pp.legend(side='bottom', collect=['method'])
```

## The five scope modes

| Intent | Call | Position semantics |
|---|---|---|
| Per-axes internal | `pp.legend(ax)` | Counted against `ax.get_tightbbox()`; behaves like a tick label. |
| Per-axes external band | `pp.legend(anchor=ax)` | Overhang past `ax`'s edge; absorbs the cell's `right` / `xlabel_space` reservation. |
| Row/column band | `pp.legend(axes[r], side='top')` / `pp.legend(axes[:, c], side='left')` | Pinned to that slice's bounding rect; scoped collection and eviction. |
| Full-figure band | `pp.legend(side='right')` / `'bottom'` / `'top'` / `'left'` | Spans the whole grid on the chosen side via `_GridAnchor`. |
| In-cell render | `pp.legend(anchor=empty_ax, inside=True)` | Renders inside `empty_ax`'s rectangle via matplotlib `loc=`. Auto-blanks the anchor; no reactor reservation; collection still walks the full figure (or the explicit `axes=`). |

Default orientation + alignment:
- `side='right'` / `'left'` → vertical, `align='start'` (top).
- `side='top'` / `'bottom'` → horizontal, `align='center'`.

Override with `orientation='vertical'|'horizontal'` and `align='start'|'center'|'end'` when needed.

## Canonical patterns

**1. Per-axes internal legend on a 1×1 figure.**

```python
fig, ax = pp.subplots(axes_size=(50, 35))
pp.scatterplot(data=df, x="x", y="y", hue="group", ax=ax)
pp.legend(ax)  # after the plot — auto-collects stashed entries
```

**2. 2×2 grid with a shared right-side band.**

```python
fig, axes = pp.subplots(2, 2, axes_size=(35, 30))
pp.legend(side="right")  # claim the band BEFORE plotting
for (r, c), panel in zip([(0, 0), (0, 1), (1, 0), (1, 1)], "ABCD"):
    pp.scatterplot(data=df[df.panel == panel], x="x", y="y",
                   hue="group", palette=palette, ax=axes[r, c])
```

**3. 2×3 grid with a band above each row (the inter-row case).**

```python
fig, axes = pp.subplots(2, 3, axes_size=(35, 25))
pp.legend(axes[0], side="top", collect=["group"])
pp.legend(axes[1], side="top", collect=["group"])  # sits BETWEEN rows
for r, row in enumerate(axes):
    for c, ax in enumerate(row):
        pp.scatterplot(data=df, x="x", y="y", hue="group", ax=ax)
```

**4. 3×2 grid with a column-0 left band.**

```python
fig, axes = pp.subplots(3, 2, axes_size=(35, 25))
pp.legend(axes[:, 0], side="left", collect=["group"])
for r, row in enumerate(axes):
    for c, ax in enumerate(row):
        pp.scatterplot(data=df, x="x", y="y", hue="group", ax=ax)
```

**5. Advanced: explicit anchor override (collection scope ≠ geometric pin).**

```python
fig, axes = pp.subplots(2, 3, axes_size=(35, 25))
top_row = list(axes[0])
# Collect from the whole top row, but pin the band above the top-right corner only.
pp.legend(axes=top_row, anchor=axes[0, -1], side="top", collect=["group"])
```

**6. In-cell shared legend (fill an empty grid cell with the legend).**

```python
fig, axes = pp.subplots(2, 2, axes_size=(35, 30))
for (r, c), panel in zip([(0, 0), (0, 1), (1, 0)], "ABC"):
    pp.scatterplot(data=df[df.panel == panel], x="x", y="y",
                   hue="group", palette=palette, ax=axes[r, c])
# The bottom-right cell is empty — fill it with the shared legend.
pp.legend(anchor=axes[1, 1], inside=True)
```

The anchor cell is auto-blanked (no frame, no ticks). Pass `clear_anchor=False` to keep its content (e.g. a logo or annotation already drawn there). Default placement is upper-left of the tile — the visual continuation of band-mode `side='right', align='start'`. Override with `side` + `align`:

| `side`     | `align='start'`     | `align='center'`    | `align='end'`       |
|------------|---------------------|---------------------|---------------------|
| `left` *(default)* | `upper left` *(default)*  | `center left`       | `lower left`        |
| `right`    | `upper right`       | `center right`      | `lower right`       |
| `top`      | `upper left`        | `upper center`      | `upper right`       |
| `bottom`   | `lower left`        | `lower center`      | `lower right`       |
| `center`   | `center` *(align ignored)* | `center`     | `center`            |

Common variations:
- Empty cell on the **left** of the plots → `pp.legend(anchor=ax, inside=True, side='right', align='start')` (legend hugs inner-right edge, against the divide).
- Centered tile look → `pp.legend(anchor=ax, inside=True, side='center')`.
- Narrow collection scope → `pp.legend(axes=[a, b], anchor=empty, inside=True)` (collect from a, b only; render in `empty`).

## Ordering: before vs after plots

**Both orderings work** for figure-level bands, row/column bands, and in-cell mode (`inside=True`).

- **Before** (preferred, marginally faster): the band is registered first; each plot call sees it and stashes entries instead of rendering its own per-axis legend.
- **After** (seamless): the band walks every axes in scope on construction, evicts per-axis legend artists whose titles match entries it will claim, and renders the shared band. Inside legends for entries the band does NOT claim (e.g. via `collect=[...]`) survive the eviction.

For per-axes legends, call `pp.legend(ax)` **after** the plot — it auto-collects stashed entries. Before would force you to pass `collect=[]` and add handles manually.

## The 0.10 asymmetry (important)

`pp.legend(ax)` and `pp.legend(anchor=ax)` look similar; their layout semantics are opposite:

```python
fig, axes = pp.subplots(1, 2, axes_size=(45, 35))
pp.legend(axes[0])              # INTERNAL — counts in axes[0].tightbbox
pp.legend(anchor=axes[1])       # EXTERNAL — overhangs past axes[1]'s right edge
```

Why: the asymmetry preserves pre-0.10 semantics across the `pp.legend_group` → `pp.legend` rename. `pp.legend(ax)` matches old `pp.legend(ax)`; `pp.legend(anchor=ax)` matches old `pp.legend_group(anchor=ax)`.

Migration: a plain `sed -i 's/pp\.legend_group(/pp.legend(/g'` over your code is sufficient — every kwarg carries over and semantics are preserved.

## The `collect=` filter

`collect=` narrows which stashed legend names the band will claim. Useful when a plot exposes multiple orthogonal kinds (e.g., `hue=` + `style=` in `pp.lineplot`) and you want each on its own band.

```python
fig, axes = pp.subplots(2, 2, axes_size=(45, 30))
pp.legend(side="top",    collect=["treatment"])   # color band
pp.legend(side="bottom", collect=["method"])      # dash-style band
for r, row in enumerate(axes):
    for c, ax in enumerate(row):
        pp.lineplot(data=line_df, x="time", y="value",
                    hue="treatment", style="method",
                    palette=treatment_palette, ax=ax)
```

Edge cases:
- `collect=None` (default) → claim every stashed entry in scope.
- `collect=[]` (empty list) → **skip auto-collection entirely**; use when adding handles manually via `group.add_legend(...)`.
- Passing a bare string raises `TypeError` — always wrap in a list: `collect=['name']`.

## Two `inside=True` flavors

Same word, two distinct mechanisms. Pick by where the legend should end up.

| Form | Where the legend renders | When to use |
|------|--------------------------|-------------|
| `legend_kws={'inside': True, 'loc': '...'}` on a **plot call** | Inside the SAME axes the plot lives in (matplotlib corner placement). | Single-axes plots; quick in-frame legend that doesn't claim layout space. |
| `pp.legend(anchor=ax, inside=True)` on a **legend factory call** | Inside a SEPARATE axes you point at via `anchor=`. Collection still walks the full figure (or `axes=`). | Asymmetric grids where one cell is reserved for the legend; auto-blanks the anchor cell so the cell reads as a clean tile. |

Combine with a figure-level `pp.legend(collect=[...])` to split legend kinds: collected names go in the figure band, everything else can go inside via the plot-call form.

```python
fig, axes = pp.subplots(2, 2, axes_size=(35, 30))
pp.legend(side="bottom", collect=["group"])
for (r, c), panel in zip([(0, 0), (0, 1), (1, 0), (1, 1)], "ABCD"):
    pp.scatterplot(data=df[df.panel == panel], x="x", y="y",
                   hue="group", style="replicate", palette="pastel",
                   legend_kws={"inside": True, "loc": "upper right"},
                   ax=axes[r, c])
```

## Grid scoping (PR 4 / v0.12)

For figures built by `pp.subplots` or `pp.Canvas`, `pp.legend` accepts
four kwargs that resolve to a sub-rect of the grid:

```python
# Row-scoped band over row 0 (e.g., a top-row group legend)
pp.legend(rows=0, side='top')

# Inclusive row range × specific column
pp.legend(rows=(1, 3), cols=2, side='right')

# Sugar: full-figure band (alternative to default kwargs)
pp.legend(span='fig', side='bottom')

# Sugar: full-row band keyed off a positional Axes anchor
pp.legend(axes[0, 1], span='row', side='top')

# Explicit list with handle dedupe
pp.legend(ax=[ax_a, ax_b, ax_c], side='top')
```

**Mutual exclusivity:** `rows=`/`cols=`/`span=`/`ax=` are alternative
addressing modes. Mixing them (other than `span='row'`/`'col'` with a
positional anchor) raises `ValueError`. The legacy positional `axes=`
arg is also exclusive with the new kwargs except for the
positional-anchor `span` form.

**Out-of-range indices** raise with the actual `_publiplots_axes`
shape so the message points at the cause: e.g.,
`pp.legend(rows=5)` on a 2×3 grid → `rows=5 out of range for shape (2, 3)`.

**Raw matplotlib figures:** `rows=`/`cols=` need the publiplots-built
matrix; on raw `plt.subplots()` figures use `ax=[ax1, ax2, ...]`
instead.

## Further reading

For the full factory signature (every kwarg: `orientation`, `align`, `x_offset`, `y_offset`, `gap`, `column_spacing`, `vpad`, `max_width`, `figure`), see `src/publiplots/utils/legend_group.py`. For 14 worked examples covering every scope mode, see `examples/plots/plot_24_legend_placement.py`.
