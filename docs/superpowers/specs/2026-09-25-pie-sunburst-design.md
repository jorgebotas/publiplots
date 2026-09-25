# Pie, donut and sunburst charts

Design for `pp.pieplot` and `pp.sunburstplot`, closing #268.

Status: approved design, not yet implemented.
Target: two public functions over one shared internal renderer.

## Motivation

publiplots has no part-of-whole plotter, and `matplotlib.axes.Axes.pie` does
not inherit the theme. Measured on 0.18.2 (#268): wedges take matplotlib's
`tab10` cycle rather than the publiplots palette, `rcParams["edgecolor"]` is
ignored so wedges have no outline, and `pp.legend(ax)` runs without error and
produces no legend artist.

The legend is the part that cannot be worked around locally. Wedges are never
stashed, so a pie cannot join a shared band with the other panels of its
figure — which is what a composition panel is usually for. The motivating case
in #268 is a four-way composition beside an existing `pp.barplot` panel keyed
on the same four categories in the same palette: the one situation where the
two panels *must* share colours to be read together.

## Scope

This document specifies both functions, because the sunburst is the consumer
that tests whether the shared renderer's boundary is drawn in the right place.
Implementation still lands in two stages — `pieplot` first, which closes #268
on its own.

A pie is a one-level sunburst. That equivalence is expressed in the signature,
not only in the implementation.

## Data model

Both functions take the library's standard tidy-frame idiom.

```python
pp.pieplot(data=df, x="kind", y="n")
pp.sunburstplot(data=df, x=["kind", "subkind", "leaf"], y="n")
```

**`x`** — the nesting. A column name, or a list of column names, one per ring,
root first. `pieplot` accepts a scalar or a one-element list and rejects
anything longer, naming `sunburstplot` in the error. `sunburstplot` accepts
either and treats a scalar as a one-ring chart.

**`y`** — the value. Aggregated by sum within each leaf group. When omitted,
rows are counted, which makes `pp.pieplot(data=df, x="kind")` a composition of
row counts without a prior groupby.

**A parent's value is always the sum of its children.** With tidy input this is
the only coherent model, and it removes the need for plotly's `branchvalues`
parameter entirely. Internal nodes are never read from the frame.

**`hue`** — what determines colour, defaulting to `x[0]`. This is the one place
the idiom diverges from the Cartesian plotters, and deliberately: in `barplot`,
`hue=` conflates how marks group with how they are coloured, because grouping
has no other parameter. Here the structure already lives in `x`, so `hue` is
free to mean only what its name says.

Consequences:

- With one level, `hue` defaults to `x` itself — "colour each wedge by its
  category", which is what every pie does. The flat case falls out of the
  hierarchical rule rather than being special-cased.
- `hue` may name a column outside the path: colour each department wedge by its
  region, or by a numeric column.
- A categorical `hue` **must be functionally dependent on the path** — constant
  within a wedge. Otherwise the wedge's colour is ambiguous. This raises a
  `ValueError` naming the offending node and the conflicting values, rather
  than silently taking the first.
- A numeric `hue` aggregates by mean within each node and routes to the
  existing continuous-colour path, stashing a colorbar rather than categorical
  entries. The depth and sibling colour knobs below do not apply; colour comes
  from the data.

### Value handling

- **NaN** values are dropped, matching `_aggregate_group_keys`.
- **Negative** values raise `ValueError`. A negative wedge has no meaning and
  silently taking the absolute value would misreport a share.
- **Zero-valued** and **absent** categories are dropped: they draw no artist
  and produce no record. A category named in `order=` but not present in the
  data is likewise dropped rather than drawn as an empty wedge. This matches
  the filtering behaviour `_iter_box_group_keys` adopted in #262.

## Geometry and layout

### Roundness without losing the cell

Measured on a `pp.subplots` cell of 70x50mm:

```
(a) set_aspect("equal") : px/unit (206.7, 147.6)  width drawn/alloc 0.6008 / 0.8411
(b) mm-matched limits   : px/unit (147.6, 147.6)  width drawn/alloc 0.8589 / 0.8589
    rendered disc bbox  : 295.28 x 295.28 px -> round
```

`set_aspect("equal")` — what `plot/venn/draw.py:164` does — shrinks the drawn
box to 71% of its allocated width and abandons the rest as whitespace. That is
#101, reproduced in one line.

Because the renderer draws its own wedges and `ax.get_position()` returns exact
millimetres *before any draw*, the pie can match the cell instead of fighting
it: leave the aspect unconstrained and set the data limits to the cell's own mm
ratio. For a 70x50 cell, `ylim=(-1, 1)` and `xlim=(-1.4, 1.4)`.

This is measurably better on both axes: the disc is circular to within a
rendering pixel, and no allocated width is lost. The pie sizes to the cell
height, leaving the surplus width where outside labels and leader lines want to
be.

**Tradeoff, stated because it is real:** the limits are computed once at draw
time, so an interactive figure resize distorts the circle, where matplotlib's
aspect re-applies on every draw. This library pins figures in millimetres and
`savefig.bbox` is `"standard"`, so resize is not a normal workflow.

`aspect="cell"` (default) selects the mm-matched behaviour; `aspect="equal"`
selects `set_aspect("equal")` for anyone the tradeoff bites.

### Ordering

`order=` wins when given. Otherwise levels resolve through
`_categorical_order` — the resolver widened in #262 — so explicit ordering,
Categorical categories and numeric sorting all behave as they do everywhere
else, rather than growing a second rule.

**Shape of `order=` with more than one level.** `pieplot` takes a flat
sequence, as every other plotter does. `sunburstplot` takes a **dict keyed by
column name**, `{"kind": [...], "subkind": [...]}`, and a column left out of
the dict resolves through `_categorical_order` as usual. A list of lists was
rejected: it would be positional against `x`, and positional alignment between
a user sequence and a structure is the fragility this design avoids
everywhere else. For a one-element `x`, `sunburstplot` also accepts a flat
sequence, so the pie form keeps working when a caller switches functions.

`hue_order=` orders the stashed legend entries and takes a flat sequence in
both functions, since `hue` is always a single column. It does not affect
wedge order; `order=` does.

`sort=False` by default. `sort=True` orders wedges by descending value —
**within each parent**, not globally, so a sunburst's children stay under the
parent they belong to. On a pie, which has one parent, that reduces to the
ordinary meaning. `sort=True` and `order=` together raise rather than silently
picking a winner.

This breaks with plotly, which sorts by default. The motivating case is a pie
beside a bar panel keyed on the same categories; a pie that silently re-sorts
destroys exactly the correspondence #268 asks for. Largest-first stays one
keyword away.

### Angles

`start_angle=90` (twelve o'clock) and `direction="clockwise"`. matplotlib
defaults to 0 degrees counterclockwise; diverging from matplotlib's defaults is
what this library is for, and twelve-o'clock-clockwise is what readers expect.

### Radii

Rings divide the radial span evenly between `donut` and 1.0. `donut=0.0` is a
pie; `donut=0.5` leaves a hole of half the radius. The hole is where a total
belongs, which is often why the ring variant is the better choice.

### Pull

`pull=` accepts a float applied to every wedge, or a dict keyed by category —
by path tuple for a sunburst. It is a fraction of the radius, applied along the
wedge's mid-angle.

It is deliberately **not** a positional list, which is matplotlib's `explode=`
interface. Positional alignment between a user-supplied sequence and drawn
artists is the fragility behind #103, #199, #236, #241, #254 and #262; the
library should stop offering it.

Pulling a sunburst node translates its **entire subtree** by the same vector.
Pulling a parent and leaving its children in place would tear the chart apart.

### Small slices

`min_share=0.0` (off). Above zero, categories below that fraction of the total
merge into a single `"Other"` wedge in a neutral grey, overridable with a
`palette` entry keyed `"Other"`. The merged wedge sorts last regardless of
`order=`.

**`pieplot` only.** Bucketing inside a sunburst requires deciding whether the
threshold applies per-parent or globally, and either answer can produce a chart
with six sibling wedges all labelled "Other". That deserves its own design
rather than a guess bundled here. `sunburstplot` rejects `min_share`.

## Colour

### One model, two knobs

Colour derives from the `hue` level, with descendants shaded from their parent.
Two scalars control it, both expressed as a **total span** rather than a
per-ring step:

- **`sunburst.depth_range`** (default `0.45`) — the outermost ring lands this
  far toward white; intermediate rings interpolate.
- **`sunburst.sibling_range`** (default `0.14`) — the spread across siblings
  within one parent, interpolated over however many there are.

Positive blends toward white, negative toward black, which gives the dark-rim
variant without a separate direction parameter.

Spans, not steps, because a step degrades silently as the tree deepens: an
additive 0.18 per ring is fine at depth 3 but puts the rim 72% toward white at
depth 5, which renders unreadable. Tree depth is not something the caller
controls. A total span is depth-independent by construction and needs no
retuning when a level is added.

Both knobs are inert for a one-ring chart, so the flat case pays nothing.

### Rejected: per-ring palettes as a named mode

A mode that gives each ring its own palette was prototyped and rejected. It is
the only variant where the reader cannot tell which outer wedge belongs to
which root — a blue outer ring above a green centre. It does not trade
hierarchy for discrimination; it destroys the structure the chart exists to
show. Anyone who wants it can pass a `palette` dict keyed per node. Available,
not suggested.

### Exposure

The dotted namespace matches `scatter.size_min` and `subplots.axes_size`.
Namespaced to `sunburst` and not `pie` because the knobs genuinely do nothing
on one ring, and a parameter that does nothing is a lie in the signature.

```python
pp.rcParams["sunburst.depth_range"]   = 0.45
pp.rcParams["sunburst.sibling_range"] = 0.14
```

`sunburstplot` also takes `depth_range=` and `sibling_range=` as
`Optional[float] = None`, resolved through `resolve_param`. This follows
`subplots()`, where every `subplots.*` rcParam has a matching parameter, rather
than `scatter`, where `size_min`/`size_max` are rcParams only. The distinction:
scatter's size bounds are a house style set once, whereas which end of a
hierarchy carries the emphasis is an editorial choice that can differ between
two panels of one figure.

`pieplot` takes neither parameter.

## Hatch

`hatch=`, `hatch_map=` and `hatch_order=` behave as they do in `barplot`,
resolved through `resolve_hatch_map` (`themes/hatches.py:363`). Hatch keys off
the same value as `hue` unless given its own column.

For a sunburst, hatch applies at the ring named by the `hatch` column and is
inherited by descendants, mirroring colour.

## Annotation

### Why the wedge meta cannot repeat the pairing family

Every annotate bug fixed in 0.17.0 through 0.18.3 — #103, #199, #236, #241,
#254, #262 — is one failure mode. publiplots does not draw seaborn's artists,
so the builder must **recover** them afterwards by scanning the axes and
zipping an aggregated group list against them by index. Every one of those bugs
lived in that zip: a foreign patch taking a slot, a zero-extent bar filtered
out, a rounding pass permuting `ax.patches`, two plotters nesting their dodge
differently, seaborn sorting levels the resolver did not.

`BarRecord` already holds `patch:`. The record shape was never the problem; the
recovery was.

A wedge has no recovery step. The renderer constructs every `Wedge` itself, so
each record is built at draw time with its artist in hand. No scan, no zip, no
order dependence, and no count-mismatch guard needed.

This rules out that family, not bugs in general. Wrong parent sums, wrong angle
arithmetic and wrong colour derivation all remain available and are covered by
tests below.

### Records

```python
@dataclass(frozen=True)
class WedgeRecord:
    patch: Wedge                     # held from construction, never re-found
    path: Tuple[str, ...]            # root to this node
    value: float
    fraction_of_total: float
    fraction_of_parent: float
    theta0: float                    # degrees
    theta1: float
    r_inner: float
    r_outer: float
    depth: int                       # 0 = innermost ring
    color: tuple
    hatch: Optional[str]
    pull: float
```

```python
@dataclass
class WedgeMeta:
    wedges: List[WedgeRecord]
    total: float
    donut: float
    center: Tuple[float, float]
    hue_active: bool
    hue_continuous: bool
    levels: Tuple[str, ...]          # the x columns, root first
    source_frame: Optional[Any]
```

Stashed as `ax._publiplots_wedge_meta`, matching the existing
`_publiplots_bar_meta` / `_box_meta` / `_point_meta` convention.

### No introspection fallback

`pp.annotate(ax, "wedge_values")` on an axes publiplots did not draw raises,
naming `pp.pieplot`.

The bar path supports annotating a foreign `ax.bar()` by reverse-engineering
patches, and that machinery — `_is_bar_rect`, `_infer_orient` — is where #199
lived and where a further regression surfaced while fixing it. Declining to
support `ax.pie()` costs a user one function-name change and forecloses the
same class of defect.

### The `wedge_values` strategy

A new `kind`, alongside `bar_values`, `bar_custom`, `box_stats` and
`point_values`.

**Default label is the percentage**, not the value — a deliberate divergence
from `bar_values`' `.2f`. A part-of-whole chart exists to show shares.
`fmt=",.0f"` returns the raw value.

**`basis="total"`** (default) or `"parent"` selects which fraction the
percentage is of. The record carries both. For a pie they are the same number,
so the parameter stays invisible until a second ring exists.

**`anchor`** takes `{"inside", "outside", "auto"}`, default `"auto"`: inside
when the text fits within the wedge, outside with a leader line when it does
not. There is precedent — `BarRecord.anchor_override` exists so stacked bars
can pin inside-vs-outside per mark. Leader lines draw at `edgewidth` in
`edgecolor`.

**`rotation`** widens from `float` to also accept `"tangential"` and
`"radial"`. Additive; no existing call changes.

`offset` keeps its existing millimetre semantics, applied radially.

### The hole

`center_label=` on the plot function: a string, or a callable receiving the
total. Not an annotate kind — it is part of drawing the donut, and an annotate
kind producing exactly one label regardless of the data would be a strategy in
name only.

## Legend

Wedges stash entries through `stash_entry` (`utils/legend_entries.py:75`),
keyed by the `hue` levels — by default the root ring's categories.

Derived shades on outer rings are **not** stashed. They are not independent
categories, and stashing them would put twenty near-identical greens in a
shared band.

A continuous `hue` stashes a colorbar instead, through the existing path.

## Module layout

Mirrors `plot/venn/`, which is the existing precedent for a plotter that draws
its own artists.

```
src/publiplots/plot/wedge/
    __init__.py
    geometry.py   # frame -> tree -> (theta, radius) layout; pull offsets
    colour.py     # hue resolution, depth/sibling derivation
    draw.py       # Wedge construction, meta emission, legend stashing
    pie.py        # pieplot
    sunburst.py   # sunburstplot
src/publiplots/annotate/
    _wedge.py     # the wedge_values strategy
    _cache.py     # + WedgeRecord, WedgeMeta
```

`geometry.py` holds no matplotlib artists and `draw.py` holds no aggregation —
the boundary that lets the tree layout be tested as data.

## Testing

Per the project convention, every figure-touching test file carries an autouse
`plt.close("all")` fixture. There is no `tests/conftest.py`.

```
tests/test_pie.py
tests/test_sunburst.py
tests/annotate/test_wedge_values.py
```

Cases that must exist:

1. **Artist identity** — every `WedgeRecord.patch` is an object actually in
   `ax.patches`, and the sets have equal length. This is the property that
   makes the pairing family unreachable; it should fail loudly if a future
   change reintroduces a recovery step.
2. **Palette sharing** — a `pp.barplot` and a `pp.pieplot` over the same four
   categories with the same `palette` dict produce identical face colours per
   category. This is #268's motivating case.
3. **Parent sums** — an internal node's value equals the sum of its children,
   over a ragged tree where branches differ in depth.
4. **Angle closure** — wedge spans sum to 360 degrees within floating-point
   tolerance, at each ring independently.
5. **Roundness** — the rendered disc's bounding box is square within a pixel on
   a non-square cell, and the drawn axes width equals the allocated width.
   Pins the #101 avoidance.
6. **Ordering** — `order=` wins; numeric levels sort; a category in `order=`
   but absent from the data is dropped; a `sunburstplot` `order=` dict orders
   only the columns it names and leaves the rest to `_categorical_order`;
   `sort=True` orders by value within each parent rather than globally; and
   `sort=True` with `order=` raises.
7. **Hue dependence** — a `hue` column that varies within a wedge raises, and
   the message names the node.
8. **Legend** — a hue-keyed pie stashes exactly the root-ring categories, and a
   three-ring sunburst stashes the same number, not one per wedge.
9. **Pull** — pulling a sunburst parent moves its whole subtree by the same
   vector.
10. **Rejection** — `pp.annotate(ax, "wedge_values")` on a foreign axes raises
    and names `pp.pieplot`; `pieplot` with a multi-column `x` raises and names
    `sunburstplot`; `sunburstplot` with `min_share` raises.

Floating-point comparisons use `pytest.approx`. Seaborn's categorical positions
carry float noise and exact equality has produced three false failures in this
codebase already.

## Documentation

- `examples/plots/plot_25_pie_and_sunburst.py` — a numbered sphinx-gallery
  page. Sections: a flat pie beside a bar panel sharing a palette dict (the
  #268 case); the donut with `center_label`; a three-ring sunburst; the colour
  knobs shown flat (`depth_range=0`), default, and inverted (negative); pull;
  `min_share`.
- `docs/source/api/configuration.rst` — the two rcParams.
- `docs/source/api/plotting.rst` — both functions.
- Function docstrings follow the numpydoc shape used across `plot/`, with the
  `hue`-means-colour divergence called out in a `Notes` block.

## Deferred

Recorded so they are choices rather than oversights.

- `max_slices=N` (keep the top N, merge the rest). `min_share` covers the
  common case; add it if asked for.
- Sunburst small-slice bucketing, for the reason given above.
- Interactive or click-to-zoom behaviour. Out of scope for a static
  publication library.
- `ax.pie()` introspection, for the reason given above.
- Rounded wedge corners. `apply_border_radius` operates on rectangles, and the
  analogue for an annular sector is a different construction.
