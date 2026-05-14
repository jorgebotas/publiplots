# Design — `pp.Canvas` Composer for paper-grade multi-panel figures

**Status:** awaiting implementation plan.
**Scope:** new `publiplots.composer` subpackage providing a programmatic, mm-precise canvas for assembling multi-panel paper figures (Cell, Nature, Nature Methods, Science) with vector-preserving external schematic insertion, panel labels, alignment, and direct journal-ready export. Plus a strictly-additive upgrade to `pp.legend` (`rows=/cols=/span=/ax=` scoping) and a new `composer-guide` Claude Code skill.
**Non-breaking.** All existing publiplots APIs (`pp.subplots`, `pp.legend`, all plot functions) keep their current signatures.

---

## Context

Paper figures live in two worlds: matplotlib generates the data panels, then the author opens Adobe Illustrator (or Inkscape) to drop in schematics, align panels, set abc labels, fix typography, and export at the journal's exact mm dimensions. **The Illustrator step is where most of the time and most of the regressions happen.** It also breaks reproducibility — figures can't be regenerated end-to-end from code.

publiplots already eliminates one half of this: `pp.subplots` produces mm-precise figures with consistent typography. What's missing is the *composition* layer — the part that knows about journal canvas presets, multi-panel layout with mixed content kinds (axes, sub-grids, schematics), figure-level legend bands, panel labels, and vector-preserving output.

Existing tools cover fragments of this gap but not the whole:

| Tool | Has | Lacks |
|---|---|---|
| matplotlib `subplot_mosaic` / `subfigures` / `constrained_layout` | grid primitives | journal presets, abc labels, schematic slots, mm guarantees |
| seaborn `FacetGrid` / `objects` | single-dataset facets | multi-panel composition |
| **proplot** (halted 2023) | journal sizes, abc, mm-ish, legends | external image slots, round-trip code, **no longer maintained** |
| **ultraplot** (proplot fork, alive) | proplot's surface, modern fixes | external image slots, embed-existing-figure, mm-deterministic output |
| **figurefirst** (Inkscape-bound) | SVG schematic slots, mm | journal presets, dead-ish, requires Inkscape ≤ 1.0 |
| **svgutils** (post-export SVG composition) | vector composition | works in SVG-space outside matplotlib, no panel labels |
| **pylustrator** (GUI editor) | round-trip-to-source | GUI-bound, no presets, no SVG slots |
| **patchworklib** (R `patchwork` clone) | operator algebra | no journal presets, no SVG, no abc |

No existing tool covers all six of: journal-specific canvas presets, external SVG/PDF schematic slots, abc panel labels with consistent typography, figure-level legend bands, mm-precise output, and ergonomic Python API. This design occupies that gap.

## Goal

1. Add `pp.Canvas(preset, ...)` plus panel kinds (`PanelAxes`, `PanelGrid`, `PanelImage`, `PanelText`) — an imperative, mm-precise composer for paper-grade multi-panel figures.
2. Vector-preserving compositing pipeline: `canvas.savefig('fig.pdf')` produces a real vector PDF with embedded vector schematics via pypdf + cairosvg, no Illustrator step.
3. `canvas.embed_figure(panel='B', fig=existing_fig)` drops a `pp.subplots`-built figure into a panel via the same compositing path — vectors preserved, source figure stays alive.
4. Journal canvas presets: Cell (1col/1.5col/2col), Nature (1col/1.5col/2col), Nature Methods, Science (1col/2col), plus a `Canvas('custom', width=N)` escape hatch.
5. Auto-letter abc panel labels with per-canvas + per-panel overrides. Adopts ultraplot's *concepts* (location codes `'ul'`/`'ur'`/`'ll'`/...; `border` outline; `bbox` background; arbitrary text-prop overrides) but exposes them through a unified `label_style={'loc': ..., 'border': ..., 'bbox': ..., ...}` dict rather than ultraplot's flat `abcloc=`/`abcborder=` kwargs — keeps `pp.Canvas`'s constructor surface small.
6. Default alignment from grid topology (axes-bbox); explicit `canvas.align(panels, edge, mode)` escape hatch with `'shift content within slot, never move slot boundaries'` semantics.
7. Strictly-additive upgrade to `pp.legend`: `rows=`, `cols=`, `span=`, `ax=[list]` for grid-scope figure legends. Independent of the Composer; works on `pp.subplots` figures too.
8. New `skills/composer-guide/SKILL.md` and a `canvas.inspect()` introspection helper so Claude Code can reason about layout without rendering.

## Non-goals

- **MCP server / live-canvas tool service.** A separate follow-up project (`publiplots-mcp`) will wrap the stable Composer API in MCP tools. Out of scope here.
- **Recursive panel nesting.** Panels are caption-addressable units only (one level deep — flat sub-grids inside a panel, but no panels-inside-panels). If you need to address it as `Fig 2A.iii` you split into two labeled panels.
- **Constraint solver.** Grid topology + explicit `canvas.align()` covers paper figures. A Cassowary-style solver is over-engineered for this domain.
- **Auto-grow figure dimensions.** Composer does NOT silently resize the figure to make panels fit. Overflow → raises with a suggested per-row scaling factor; the author decides.
- **Replacement for `pp.subplots`.** Single-grid figures keep using `pp.subplots`. Composer is for multi-panel work with mixed content kinds.
- **Operator algebra (`p1 | p2 / p3`).** Explicit `add_row` is clearer for paper figures than overloaded operators; we leave that pattern to patchworklib.
- **`share=` / `align=` axis-sharing modes at canvas level.** Each `PanelGrid` delegates to the existing `pp.subplots` `sharex`/`sharey` mechanism; no canvas-wide axis sharing.
- **Pinned canvas height in v1.** Default is auto-grow from declared panel mm. Cross-figure consistency (all main figs at 174×180mm, etc.) is the author's responsibility for v1; height-pinning + `vfill` policy is a known follow-up if user feedback demands it.

## Philosophy alignment

- **Pure additive.** No existing publiplots API changes signature. The `pp.legend` upgrade is strictly new kwargs.
- **mm-precise, single-pass.** All geometry computed once at `add_row`/`add_panel` time; no draw-event resize. Same invariant `pp.subplots` already enforces — extended to non-grid panel rectangles.
- **One layout engine.** Composer extends publiplots' existing `FigureLayout` / `SubplotsAutoLayout` rather than introducing matplotlib `subfigures` (which would mix two layout engines, the trap proplot fell into).
- **Plotting stays where it is.** Composer composes; existing publiplots functions plot. `pp.scatterplot(..., ax=canvas['B'].ax)` is the canonical pattern.
- **Vector compositing is post-savefig.** Matplotlib renders an empty-slot canvas; pypdf/cairosvg/in-tree-SVG composer stamp schematics into the slots. No live mpl-object surgery.

---

## Architecture

```
pp.Canvas(preset='cell-2col', abc='upper')          # presets.py + abc_labels.py
  │
  ├── canvas.add_row(*panels, hpad=2, ...)
  │     └─ stages PanelAxes / PanelGrid / PanelImage / PanelText records
  │
  ├── canvas.align(panels, edge, mode='axes')
  │     └─ post-row alignment graph; shifts content WITHIN slots, slot
  │        boundaries are inviolate (no constraint solver needed)
  │
  ├── canvas.embed_figure(panel='B', fig=existing_fig)
  │     └─ marks the panel slot for compositing-time stamping
  │
  ├── canvas.label_style(...) / per-panel label_style=
  │     └─ ultraplot-vocab abc rendering: abc/abcloc/abcborder/abcbbox/abc_kw
  │
  └── canvas.savefig(path)
        └─ dispatch by extension (compositing/dispatch.py):
            ├── '.pdf'           → compositing/pdf.py      (pypdf-based vector)
            ├── '.svg'           → compositing/svg.py      (in-tree SVG composer)
            ├── '.png'/'.jpg'    → compositing/raster.py   (PIL + figimage)
            └── '.tiff'          → compositing/raster.py   (TIFF, optional CMYK)

Internal layers:
  composer/layout_engine.py  — pure-geometry CanvasLayout extending FigureLayout
  composer/alignment.py      — canvas.align() resolver, edge/mode/anchor logic
  composer/abc_labels.py     — auto-letter sequencer, label_style merger
  composer/inspect.py        — canvas.inspect() schema builder
  composer/exceptions.py     — ComposerError, ComposerOverflowError, ComposerVectorError
```

Layering invariants:

- **Pure geometry** (`layout_engine.py`, `alignment.py`) — no matplotlib imports. Pure-math mm computations. Mirrors how `FigureLayout` is structured today.
- **matplotlib glue** (`canvas.py`, `panels.py`) — creates `Figure`, calls `fig.add_axes(rect)`, attaches `SubplotsAutoLayout`. Reads from pure-geometry layer.
- **Compositing** (`compositing/`) — runs *after* matplotlib has produced the base file. Operates on bytes/pages, never on live mpl objects.

Inviolate invariants (each tested):

1. `canvas.figure_size_mm` is computed once at `add_row` / `add_panel` time and never changes during draw. No draw-event resize. (The trap ultraplot's two-pass `auto_layout` falls into; we explicitly avoid it.)
2. `canvas.savefig(p)` is byte-for-byte deterministic given identical canvas state and seed (modulo PDF `/CreationDate`, which we pin).
3. `canvas.figure` is a normal matplotlib `Figure`. Users can `fig.savefig(p)` directly to bypass compositing — slots will appear as empty rectangles. Documented as the escape hatch.
4. `canvas.align()` shifts CONTENT within slots; slot mm-rects themselves are inviolate. (Eliminates need for a constraint solver and keeps alignment local + cheap.)
5. Embedded figures' `SubplotsAutoLayout` runs exactly once at render-to-buffer time and never fires again. Cross-figure layout-reactor crossing is structurally impossible.

---

## Public API

### `pp.Canvas`

```python
def Canvas(
    preset: str,                                    # 'cell-1col'|'cell-2col'|...|'custom'
    *,
    width: Optional[float] = None,                  # mm; required ONLY for preset='custom'
    abc: Union[bool, str, Sequence[str]] = 'auto',  # 'upper'|'lower'|'A.'|False|['i','ii',...]
    label_style: Optional[Mapping[str, Any]] = None,  # {'loc':'ul','size':9,'weight':'bold',...}
    dpi: float = 600.0,                             # raster fallback DPI
    strict_vectors: bool = False,                   # raise on schematic vector failure
    **rc_overrides: Any,                            # local rcParams overrides
) -> "Canvas": ...
```

`abc='auto'` resolves to the preset's default (`'upper'` for Cell, `'lower'` for Nature, `'A.'` for some templates). `abc=False` disables all auto-labels.

### Panel constructors

```python
PanelAxes(
    label: Optional[Union[str, bool]] = None,           # None=auto, False=no label, str=verbatim
    *,
    size: Tuple[Union[float, str], Union[float, str]],  # ('flex'|w_mm, h_mm|'match')
    label_style: Optional[Mapping[str, Any]] = None,
    **subplots_kw: Any,                                  # forwarded to underlying axes
)

PanelGrid(
    label: Optional[Union[str, bool]] = None,
    *,
    shape: Tuple[int, int],                              # (nrows, ncols) of inner cells
    axes_size: Tuple[float, float],                      # (w_mm, h_mm) per inner cell
    sharex: bool = False,
    sharey: bool = False,
    hspace: float = 2.0,                                 # mm between inner cells
    wspace: float = 2.0,
    label_style: Optional[Mapping[str, Any]] = None,
    **subplots_kw: Any,
)

PanelImage(
    label: Optional[Union[str, bool]] = None,
    *,
    path: Union[str, Path],                              # PDF / SVG / PNG / JPG / TIFF
    size: Tuple[Union[float, str], Union[float, str]],
    align: str = 'center',                               # 8 anchors + 'center'
    clip: str = 'fit',                                   # 'fit'|'fill'|'stretch'
    label_style: Optional[Mapping[str, Any]] = None,
)

PanelText(
    label: Optional[Union[str, bool]] = None,
    *,
    text: str,                                           # supports basic mathtext
    size: Tuple[Union[float, str], Union[float, str]],
    text_kw: Optional[Mapping[str, Any]] = None,         # ax.text kwargs
    label_style: Optional[Mapping[str, Any]] = None,
)
```

### `Canvas` methods

```python
canvas.add_row(
    *panels: Panel,
    hpad: float = 4.0,        # mm between panels in this row
    vpad: float = 4.0,        # mm above this row (0 for first row)
    justify: str = 'start',   # 'start'|'end'|'center'|'space-between'|'space-around'
    valign: str = 'top',      # 'top'|'bottom'|'center'|'baseline' within row
) -> Row

canvas.add_column(
    side: str,                # 'left'|'right'  — for vertical legend strips, not the main flow
    *panels: Panel,
    vpad: float = 4.0,
    hpad: float = 4.0,
    align: str = 'top',
) -> Column

canvas.align(
    panels: Sequence[str],                # list of panel labels
    *,
    edge: str,                            # 'left'|'right'|'top'|'bottom'|'center_x'|'center_y'|'baseline'
    mode: str = 'axes',                   # 'axes' (data rect) | 'tight' (incl. ticklabels)
    anchor: Optional[str] = None,         # which panel's edge wins; default = first
) -> None

canvas.embed_figure(
    panel: str,
    fig: matplotlib.figure.Figure,
    *,
    fit: str = 'fit',                     # 'fit'|'fill'|'stretch'|'exact'
    align: str = 'center',
) -> None

canvas.label_style(**kwargs: Any) -> None  # canvas-wide override

canvas.inspect() -> dict                   # see "canvas.inspect schema" below

canvas.savefig(path: Union[str, Path], **kwargs: Any) -> Path

canvas.save_multiple(stem: str, formats: Sequence[str]) -> Sequence[Path]

canvas[label]      -> Panel                # e.g. canvas['B'].ax
canvas.row(i)      -> Row                  # canvas.row(0).axes for pp.legend(...)
canvas.col(i)      -> Column
canvas.figure      -> matplotlib.figure.Figure
```

### Panel object (returned by `canvas[label]`)

```python
panel.label         : str                                  # 'A'
panel.kind          : Literal['axes','axesgrid','image','text']
panel.ax            : matplotlib.axes.Axes                  # raises for non-axes panels
panel.axes          : numpy.ndarray[matplotlib.axes.Axes]   # axesgrid only; raises otherwise
panel.bbox_mm       : Tuple[float, float, float, float]     # (x, y, w, h) in canvas mm
panel.size_mm       : Tuple[float, float]
panel.row, panel.col : int                                  # grid coords
```

### Size grammar

- `size=(w_mm, h_mm)` — both pinned
- `size=('flex', h_mm)` — width grows to fill row leftover space; multiple flex panels in a row share equally
- `size=(w_mm, 'match')` — height matches sibling panels in the same row (default for non-flex when row has explicit height-bearing panels)
- Row width policy: `sum(non_flex_widths) + n_flex * leftover_per_flex + (n−1) * hpad ≤ row_available`; else raise `ComposerOverflowError` with suggested per-row scaling factor.

### Image panel `align` × `clip` matrix

`align` ∈ {`'top-left'`, `'top'`, `'top-right'`, `'left'`, `'center'`, `'right'`, `'bottom-left'`, `'bottom'`, `'bottom-right'`}; mirrors CSS `object-position`.

`clip` ∈ {`'fit'`, `'fill'`, `'stretch'`}; mirrors CSS `object-fit`.

### `pp.legend` upgrade (additive; ships in same project as Composer)

```python
def legend(
    *axes: matplotlib.axes.Axes,            # existing positional
    anchor: Optional[matplotlib.axes.Axes] = None,
    side: str = 'right',
    # NEW kwargs (all backward-compatible additions):
    rows: Optional[Union[int, Tuple[int, int]]] = None,
    cols: Optional[Union[int, Tuple[int, int]]] = None,
    span: Optional[str] = None,             # 'row'|'col'|'fig' — sugar
    ax: Optional[Sequence[matplotlib.axes.Axes]] = None,  # explicit list, dedupe handles
    # ...all existing kwargs (collect, orientation, align, gap, vpad, etc.) ...
) -> Legend: ...
```

New behaviors:

- `pp.legend(rows=0, side='top')` — figure-level band scoped to row 0 of the active figure's grid (resolved via `fig._publiplots_layout` — works on both `pp.subplots` and `Canvas` figures).
- `pp.legend(rows=(1, 3), cols=2, side='right')` — band spanning rows 1-3 in column 2 only.
- `pp.legend(ax=[ax1, ax2, ax3], side='top')` — collect handles from those axes, dedupe by label, render as a single band over their union bbox.
- `pp.legend(span='fig', side='bottom')` — full-figure-width band along the bottom (sugar for "ignore grid scoping").

Backward compat: every existing call signature unchanged. New kwargs are additive.

---

## Worked example

```python
import publiplots as pp

canvas = pp.Canvas('cell-2col')   # 174 mm wide, abc='upper' default

canvas.add_row(
    pp.PanelImage('A', path='schematic.svg', size=(80, 40)),
    pp.PanelAxes ('B', size=('flex', 40)),
)
canvas.add_row(
    pp.PanelGrid('C', shape=(1, 3), axes_size=(50, 30), sharey=True),
)
canvas.add_row(
    pp.PanelAxes('D', size=(80, 35)),
    pp.PanelText('E', text='n = 1,234\nP < 0.001', size=('flex', 35)),
)

# Plot into the panels:
pp.scatterplot(data=df, x='x', y='y', hue='cond', ax=canvas['B'].ax)
for i, ax in enumerate(canvas['C'].axes.flat):
    pp.lineplot(data=df_c[i], x='t', y='v', ax=ax)
pp.barplot(data=df_d, x='cat', y='val', ax=canvas['D'].ax)

# Legends use existing pp.legend; no new API needed:
pp.legend(canvas['B'].ax, side='right')           # panel-local
pp.legend(canvas['C'].axes, side='top')           # row-of-axes band
# Or, post-upgrade, by row scope:
pp.legend(rows=0, side='top')                     # whatever's in row 0

# Optional explicit alignment:
canvas.align(['A', 'D'], edge='left', mode='axes')

# Save: vector PDF with embedded SVG schematic.
canvas.savefig('fig2.pdf')
canvas.savefig('fig2.png', dpi=600)
canvas.save_multiple('fig2', formats=['pdf', 'png', 'tiff'])
```

---

## Vector-preserving compositing pipeline

Single sentence: matplotlib's PDF backend can't host external vector PDFs as embedded objects, so we let matplotlib render the canvas with reserved white-space slots, then **post-composite** schematics into those slots with `pypdf` (PDF source) or `cairosvg → PDF → pypdf` (SVG source) — vectors preserved end-to-end.

### PDF pipeline

```
1. RESERVE   — for each PanelImage / embed_figure panel, the underlying
               matplotlib axes is rendered as nothing:
                 ax.set_axis_off()
                 ax.patch.set_visible(False)
               The panel's mm-rect is recorded in canvas._slots.

2. RENDER    — fig.savefig(buffer, format='pdf', metadata={...})
               Pure matplotlib PDF, with empty rectangles where schematics
               belong. Vectors fully preserved.

3. STAMP     — for each slot:
                 if slot.path.endswith('.pdf'):  load with pypdf
                 if slot.path.endswith('.svg'):  cairosvg.svg2pdf → load with pypdf
                 if slot.path.endswith(('.png','.jpg','.tif','.tiff')):
                       wrap raster in 1-page PDF (PIL → reportlab or hand-rolled)
                 # for canvas.embed_figure(fig=…):
                 fig.savefig(buf, format='pdf')  → load with pypdf

               Compute mm→pt transform (1 pt = 1/72 in, 1 in = 25.4 mm).
               pypdf merge_transformed_page() places the schematic at
               (x_pt, y_pt) with scale + translate.

4. WRITE     — pypdf writes the composited PDF to the user's path.
               Existing PDF metadata (fonttype=42, transparent bg) survives
               because step 2's PDF is the canvas; step 3 only ADDS
               content streams, never re-encodes existing vectors.
```

Failure modes:

- pypdf can't open the schematic (corrupt, encrypted) → raise `ComposerVectorError(panel='B', cause=...)` with a hint to convert via `pdftk` or re-export as SVG.
- cairosvg can't parse the SVG (foreign object, unsupported feature) → fallback: rasterize at the canvas DPI, emit `UserWarning`, keep going. User can flip to fail-hard mode with `Canvas(..., strict_vectors=True)`.
- Schematic natural aspect doesn't match slot → apply `align`/`clip` rules.

### SVG pipeline

Same shape. Merge tool is different: matplotlib SVG output is parseable XML; we use a minimal in-tree SVG composer (a few hundred lines, no `svgutils` dep) to insert each schematic's `<svg>` subtree at the slot's mm-position. Image schematics get embedded as `<image href="data:..."/>` for self-contained output.

### Raster pipeline

Schematics are decoded → resampled at the requested DPI → composited with `fig.figimage` at the slot's pixel position before `fig.savefig`. This is also what powers `canvas.savefig('fig.png')`.

### Embedded `pp.subplots` figures

`canvas.embed_figure(panel='B', fig=existing_fig)` is the same flow as `PanelImage` — the existing figure is rendered to PDF/SVG in memory, then stamped into the slot. The embedded figure's own `SubplotsAutoLayout` runs once at render-to-buffer time, then the buffer is frozen.

`fit='exact'` is special: instead of scaling the rendered buffer, the source figure has `set_size_inches(...)` called to match the slot, then re-rendered. Useful for pixel-perfect mm scaling.

### Dependencies

New deps in a `[composer]` install extra:

- `pypdf >=4.0` — PDF read/transform/write. Pure Python, ~600 KB. **No system deps.**
- `cairosvg >=2.7` — SVG → PDF. **System dep on libcairo2.** Widely available (Debian/Ubuntu/Conda/Homebrew); worth flagging in install docs.
- `pillow >=10.0` — already pulled in via matplotlib; used for raster compositing.
- `reportlab >=4.0` (optional) — used to wrap raster→PDF page. Could be replaced with a hand-rolled writer if we want zero-dep.

Without `[composer]` extra, Composer still works but PDF/SVG savefig dispatches through the raster pipeline with a `UserWarning` on first PDF/SVG export pointing to the install hint. Keeps publiplots' core install lean.

---

## abc panel labels (ultraplot vocabulary)

```python
canvas = pp.Canvas('cell-2col', abc='upper')      # 'A','B','C',...   (Cell default)
canvas = pp.Canvas('nature-2col', abc='lower')    # 'a','b','c',...   (Nature default)
canvas = pp.Canvas('cell-2col', abc='A.')         # template: 'A.','B.','C.'
canvas = pp.Canvas('cell-2col', abc=['i','ii','iii','iv'])  # explicit list
canvas = pp.Canvas('cell-2col', abc=False)        # no labels
```

Label style controls (canvas-wide and per-panel):

```python
canvas.label_style(
    weight='bold',           # default 'bold'
    size=9,                  # pt; preset default — Cell=9, Nature=8, Science=10
    family=None,             # falls back to rcParams['font.family']
    loc='ul',                # 'ul'|'ur'|'ll'|'lr'|'uc'|'lc'|'cl'|'cr' (ultraplot vocab)
    pad_mm=(0.0, 0.0),       # offset INSIDE the slot from the chosen corner
    border=False,            # white outline (ultraplot abcborder)
    bbox=False,              # white background box (ultraplot abcbbox)
)
```

Per-panel override at panel construction:

```python
pp.PanelAxes(label='B', label_style={'loc': 'ur', 'size': 11})
pp.PanelAxes(label='C.i')   # custom letter, follows abc template otherwise
pp.PanelAxes(label=False)   # NO label
```

Auto-letter:

- Panels with `label=None` are auto-assigned by the `abc=` template/sequence in row-major order.
- Panels with explicit `label='X'` use that string verbatim AND consume a position in the auto sequence (so the next auto panel still increments correctly — no skipping).
- Panels with `label=False` are skipped from the sequence (the next auto panel uses the next letter).

---

## Alignment

### Default (no `align()` calls)

- Panels in the same row auto-align by **top edge of the axes-bbox** (the data rectangle, not the tightbbox).
- Adjacent rows are stacked top-to-bottom; column auto-alignment falls out of canvas-wide `outer_pad + ylabel_space` reservations matching across rows.
- This is the "axes-bbox" mode: data rectangles align; ticklabels may overhang slightly.

### Explicit overrides

```python
canvas.align(panels, *, edge, mode='axes', anchor=None)
```

- `panels`: list of panel labels.
- `edge`: `'left' | 'right' | 'top' | 'bottom' | 'center_x' | 'center_y' | 'baseline'`.
  - `'baseline'` = bottom of axes-bbox excluding xlabel space (rare; for "first text-line baseline" alignment, common in ggpubr).
- `mode`: `'axes'` (align inner data rectangles) or `'tight'` (align outer tightbbox edges, ticklabels included).
- `anchor=<label>`: optional. If given, that panel's edge is the reference and the others are shifted to match. If omitted, the leftmost/topmost panel's edge wins by convention.

### Resolution

1. Compute initial panel mm-rects from grid topology + flex sizing (one pass, deterministic).
2. Apply each `canvas.align()` call in order. Each shifts panels **inside their grid slots** — slot boundaries themselves never move.
3. If a shift would push content outside its slot (slot too narrow for the alignment), raise with a helpful message. Author chooses: enlarge slot, or drop the alignment.

This rule (slot boundaries are inviolate) eliminates the constraint-solver requirement and keeps alignment local + cheap.

---

## `canvas.inspect()` schema

```python
{
    'preset': 'cell-2col',
    'figure_size_mm': (174.0, 87.5),
    'panels': {
        'A': {
            'kind': 'image',
            'path': 'schematic.svg',
            'pos_mm':  (5.0, 47.5),                # bottom-left in canvas mm
            'size_mm': (80.0, 40.0),
            'row': 0, 'col': 0,
            'natural_size_mm': (95.2, 47.6),       # source aspect
            'fit_strategy': 'fit',
            'align_strategy': 'center',
            'label': 'A',
            'label_style': {'loc': 'ul', 'size': 9, 'weight': 'bold', ...},
        },
        'B': {
            'kind': 'axes',
            'pos_mm':  (89.0, 47.5),
            'size_mm': (80.0, 40.0),
            'row': 0, 'col': 1,
            'has_legend': True,
            'legend_loc': 'right',
            'label': 'B', 'label_style': {...},
        },
        'C': {
            'kind': 'axesgrid',
            'pos_mm':  (5.0, 5.0),
            'size_mm': (164.0, 30.0),
            'shape': (1, 3),
            'row': 1, 'col': 0,
            'sharex': False, 'sharey': True,
            'label': 'C', 'label_style': {...},
        },
        # ...
    },
    'rows': [
        {'index': 0, 'panels': ['A', 'B'], 'height_mm': 40.0, 'hpad_mm': 4.0, 'valign': 'top'},
        {'index': 1, 'panels': ['C'],      'height_mm': 30.0, 'hpad_mm': 4.0, 'valign': 'top'},
    ],
    'alignment': [
        {'panels': ['A', 'C'], 'edge': 'left', 'mode': 'axes', 'anchor': 'A'},
    ],
    'overflow': [],          # row indices that overflow + suggested per-row scaling factor
    'warnings': [],          # e.g. 'panel B and panel D have axes-bbox left
                             #       edges 1.2mm apart; consider canvas.align'
    'embedded_figures': [],  # list of {panel, figure_id, fit, align}
    'pending_savefig': None, # set if savefig is in flight
    'composer_version': '0.12.0',
}
```

Used by Claude (via `composer-guide` skill) to reason about layout without rendering and reading PNGs.

---

## Module layout

```
src/publiplots/
├── composer/                        ← NEW subpackage
│   ├── __init__.py                  # public re-exports
│   ├── canvas.py                    # Canvas class
│   ├── panels.py                    # PanelAxes/PanelGrid/PanelImage/PanelText dataclasses
│   ├── presets.py                   # JOURNAL_PRESETS dict (Cell, Nature, Nature Methods, Science, custom)
│   ├── layout_engine.py             # CanvasLayout (extends FigureLayout)
│   ├── alignment.py                 # canvas.align resolver
│   ├── abc_labels.py                # auto-letter sequencer + label rendering
│   ├── inspect.py                   # canvas.inspect() schema builder
│   ├── compositing/
│   │   ├── __init__.py
│   │   ├── pdf.py                   # pypdf-based stamping
│   │   ├── svg.py                   # in-tree SVG composer
│   │   ├── raster.py                # PIL/figimage path
│   │   ├── embed.py                 # embed_figure helpers
│   │   └── dispatch.py              # extension → pipeline router
│   └── exceptions.py                # ComposerError, ComposerOverflowError, ComposerVectorError
├── layout/
│   ├── figure_layout.py             # EXISTING — extended for non-grid panel rects
│   ├── auto_layout.py               # EXISTING — light extension for Composer reservations
│   └── ...
├── utils/
│   └── legend_group.py              # EXISTING — extended with rows=/cols=/span=/ax=
└── ...

skills/
├── publiplots-guide/                # EXISTING (transitional sections during rollout)
├── legend-placement/                # EXISTING (minor update for new rows=/cols= patterns)
└── composer-guide/                  # NEW

examples/composer/                   # NEW
├── cell_1col_simple.py
├── cell_2col_with_schematic.py
├── nature_2col_aging.py
├── science_2col_dense.py
├── custom_width_supplementary.py
├── kitchen_sink_all_features.py
└── README.md

tests/composer/                      # NEW
├── conftest.py
├── test_canvas_basic.py
├── test_panels.py
├── test_layout.py
├── test_alignment.py
├── test_abc_labels.py
├── test_inspect.py
├── test_presets.py
├── test_save_raster.py
├── test_save_vector_pdf.py
├── test_save_vector_svg.py
├── test_embed_figure.py
├── test_legend_integration.py
├── test_overflow_advisory.py
├── test_strict_vectors.py
└── golden/                          # committed PDF/PNG fixtures
```

---

## Testing strategy

### Layers

1. **Unit tests** — geometry, panel registry, alignment resolution, abc sequencing, inspect schema, overflow advisor, error paths. Run on every commit.
2. **mm-precision tests** — assert `figure_size_mm` and panel `pos_mm`/`size_mm` to 0.01 mm tolerance for representative compositions. Stops stealth regressions from rounding changes. Snapshot JSON files committed in `tests/composer/golden/mm_snapshots/`.
3. **Vector-pipeline integration tests** — render PDF, parse with pypdf, assert:
   - page mediabox dims match `figure_size_mm` (1pt = 1/72in tolerance)
   - one content stream per matplotlib axes/text artist (sanity)
   - one external XObject per stamped schematic
   - schematic content-stream byte-checksum matches the source PDF's content stream (proves no re-encoding)
4. **Visual regression tests** — PNG fixtures rendered at 600 DPI, compared with `mpl.testing.compare_images`, `tol=10`. Forgiving — catches gross regressions, not pixel-identical output.
5. **No-deps fallback tests** — pytest fixture monkeypatches `import pypdf` to fail; asserts raster fallback runs and emits the expected `UserWarning`.
6. **Skill-trigger smoke test** — manual gate: open a clean Claude Code session in a worktree, send a representative composer prompt, verify `composer-guide` (or transitional `publiplots-guide` section) auto-triggers and Claude calls `canvas.inspect()` first. Enforced before each release-tagging PR.

### Pre-merge eval gates per PR

Before any PR is human-reviewable:

- Test suite green with the contract-required coverage from the test-designer agent.
- mm-precision regression diff: ANY diff > 0.01 mm fails the gate.
- Golden-file vector diff (PR 5+): byte-checksum match for unchanged paths.
- Visual regression: PNG renders match fixtures at `tol=10`.
- Reviewer agent sign-off (independent opus agent, adversarial pass).
- Skill-trigger smoke test passes (release PRs only).

---

## Multi-PR roadmap

Eight PRs + one spike-validation PR. Vertical slicing — each PR ships a useful artifact. Implementation runs through the agent team specified below.

### Pre-PR-1 spike — `spike(composer): vector compositing pipeline derisk`

1-2 day spike on a separate branch; validates the pypdf + cairosvg flow on toy inputs BEFORE PR 1 commits to the architecture. Three parallel opus agents:

- pypdf path: simple `pp.subplots` PDF + external PDF schematic → composited PDF, validated in Acrobat + Inkscape + Preview.app.
- cairosvg path: SVG schematic → PDF → composited, including font handling (cairosvg handles `<text>` differently from inkscape-exported SVGs).
- in-tree SVG composer path: matplotlib SVG output + raw SVG schematic → merged SVG, opened in Inkscape + Firefox.

Output: a small PR with test fixtures + a `composer-spike.md` design-validation doc. If any path fundamentally fails, we revisit architecture before PR 1.

### PR 1 — `feat(composer): canvas + axes panels (single-row)`

`Canvas`, `PanelAxes`, `add_row`, auto-grow height, custom preset, `canvas['<label>'].ax`, raster `savefig`. No alignment, no abc, no images, no flex sizing yet.
Result: usable end-to-end for axes-only multi-panel figures with raster output. ~6 files, ~600 LOC + tests.

### PR 2 — `feat(composer): journal presets + flex sizing + abc labels`

`presets.py` with Cell / Nature / Nature Methods / Science presets (and the `'custom'` escape hatch), flex-size resolution, overflow advisor, abc auto-letter + `label_style`, per-panel label overrides.
Result: actual paper-grade composition for axes-only figures. ~4 files, ~500 LOC + tests + 2 examples.

### PR 3 — `feat(composer): PanelGrid + PanelText + canvas.align`

`PanelGrid` (sub-grid of axes inside one panel), `PanelText`, `canvas.align()` resolver (edges, modes, anchor, "shift within slot" semantics).
Result: full axes/grid/text composition with alignment. ~4 files, ~500 LOC + tests + 2 examples.

### PR 4 — `feat(legend): pp.legend rows/cols/span/ax kwargs`

Extend `legend_group.py` with grid-scope kwargs. Independent of Composer (works on `pp.subplots` figures too). Update `legend-placement` skill.
Result: better `pp.legend` for everyone, prerequisite for composer figure legends. ~2 files, ~200 LOC + tests.

### PR 4.5 — `test(composer): mm-precision + golden-output test infrastructure`

Snapshot framework, golden-file comparison helpers, per-preset fixture generation. Lands BEFORE PR 5 needs it so PR 5 reviewers don't face "new feature + new test infra" in one diff.
Result: full pre-merge gate machinery in place. ~3 files, ~400 LOC.

### PR 5 — `feat(composer): vector-PDF compositing pipeline`

`compositing/pdf.py` with pypdf integration, `[composer]` install extra, `PanelImage` (PDF/SVG/PNG inputs), `strict_vectors` flag, mm→pt math, golden-PDF tests.
Result: vector-preserving PDF output with embedded schematics. The headline feature. ~5 files, ~800 LOC + tests + 1 example.

### PR 6 — `feat(composer): vector-SVG + embed_figure + raster polish`

`compositing/svg.py` (in-tree), `canvas.embed_figure(panel=, fig=)`, raster pipeline polish (TIFF / optional CMYK), `save_multiple` integration.
Result: full save dispatch matrix; `pp.subplots → embedded panel` workflow. ~3 files, ~600 LOC + tests + kitchen-sink example.

### PR 7 — `feat(composer): canvas.inspect() + composer-guide skill`

Full `inspect()` schema, skill content, integration with `legend-placement` updates, ReadTheDocs gallery integration.
Result: Claude becomes Composer-aware. ~3 files + skill markdown, ~400 LOC + tests + RTD config.

### Sequencing constraints

- PR 4 (legend kwargs) is independent of all others; can ship in parallel any time.
- PR 5 (vector PDF) needs PRs 1-3 to have something to composite.
- The pre-PR-1 spike must succeed before PR 1 starts.
- PR 7 (inspect + skill) lands last — needs all behavior in place.

### Total scope

8 PRs + 1 spike = ~5000 LOC code + ~5000 LOC tests/fixtures + 7-10 examples + 1 new skill. Roughly 5-8 weeks of careful work.

---

## Per-PR agent team

Each non-trivial PR runs through a five-agent team (all opus). Token usage is not a constraint; quality is.

1. **`code-architect`** — drafts module interfaces (dataclass shapes, method signatures, invariants) BEFORE any code. Writes a 1-page contract doc per PR. Stops the "implementation choices that look fine in isolation but conflict at the boundary" failure mode.
2. **`test-designer`** (general-purpose, opus) — designs the test matrix from the contract: unit, mm-precision, golden-file, error-path, no-deps-fallback. Writes test scaffolds with descriptive names + docstrings, leaves bodies as `assert False, "TODO"`. Forces the test surface to be considered before the implementation.
3. **`ml-implementer` / general-purpose implementer** (opus) — fills in the implementation against the contract + tests. Fans out to parallel implementer agents on opus where modules are independent (e.g. PR 5: pypdf integration + cairosvg integration + raster fallback).
4. **`reviewer`** (general-purpose, opus, explicit code-review prompt) — independent adversarial review BEFORE PR open. Checks contract conformance, edge cases, missing invariants, doc quality.
5. **`debugger`** — invoked on any test failure, layout regression, or vector-pipeline anomaly during implementation.

---

## Plugin release coordination

Per the project convention, every release PR refreshes the Claude Code plugin's skill files. The Composer rollout adds a wrinkle: PRs 1-3 ship behavior but no `composer-guide` skill yet. Transitional plan:

- PRs 1-3: each release adds a brief composer-aware section to **`publiplots-guide`** (gated to "if Canvas exists in code") so Claude doesn't fly blind.
- PR 7: the brief section is removed from `publiplots-guide` and the standalone `composer-guide` skill takes over.

This avoids a "Composer ships but Claude doesn't know" gap during the rollout window.

---

## Open items requiring verification before PR 5

- **Cell figure dimensions**: research-agent reports cited Cell figure guidelines but the source page was Cloudflare-blocked. Verify against the official Cell Press author PDF before committing presets to `presets.py`. Same for Science.
- **Nature / Nature Methods dimensions**: Nature confirmed via nature.com directly during research; values are 89 / 120-136 / 183 mm. Nature Methods follows Nature's artwork policy but verify the current Nat Methods artwork guide separately. Spot-check both before PR 2.
- **pypdf merge_transformed_page** behavior for matplotlib-generated PDFs with `fonttype=42`: validate during the spike. If pypdf re-encodes embedded fonts or strips transparency, we need a different vector compositor.
- **cairosvg font-handling for inkscape-exported vs illustrator-exported SVGs**: validate during the spike with both source kinds.

These are explicit gates on PR 5 readiness, not blockers for PRs 1-4.

---

## Risks & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| pypdf can't preserve `fonttype=42` metadata cleanly | Medium | Spike validates before PR 1; fallback is rasterizing schematics |
| cairosvg fails on common Illustrator-exported SVG features | Medium | Spike validates; `strict_vectors=False` default falls back to raster with warning |
| Composer's extended FigureLayout breaks `pp.subplots` regression | Low | PR 4.5 lands mm-precision regression tests against `pp.subplots` AND Canvas |
| Claude misuses Canvas without the skill (transitional gap) | Medium | Brief Composer section in `publiplots-guide` during PRs 1-6 |
| Library install bloat from `[composer]` extra | Low | Extra is opt-in; core install untouched |
| Cross-figure consistency complaints from no-height-pinning v1 | Medium | Documented as known v1 limitation; height-pinning + vfill is a clear follow-up addition |
| MCP server scope creep into this project | Medium | Explicit non-goal; deferred to standalone `publiplots-mcp` project |

---

## Definition of done (v1)

- All 8 PRs + spike merged.
- `pip install publiplots[composer]` installs cleanly on Ubuntu, macOS, Windows.
- All preset compositions render to PDF, SVG, PNG, TIFF without warnings on the canonical examples.
- `tests/composer/` green with mm-precision, golden-file, and visual regression gates.
- `composer-guide` skill auto-triggers in Claude Code on a clean session for the canonical "make a Cell 2-col figure with a schematic and a 2×3 grid" prompt.
- ReadTheDocs gallery includes the 7 composer examples.
- CHANGELOG entries for each release in the rollout.
- One real LemurFlow paper figure rebuilt end-to-end with Composer (no Illustrator step) as the dogfooding acceptance test.
