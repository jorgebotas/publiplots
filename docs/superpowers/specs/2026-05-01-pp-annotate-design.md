# Design — `pp.annotate()` In-Plot Value Labels

**Status:** approved, awaiting implementation plan.
**Worktree:** `.claude/worktrees/pp-annotate` on `worktree-pp-annotate`.
**Scope:** new top-level package `src/publiplots/annotate/`; one hook in `src/publiplots/plot/bar.py`.

---

## Goal

Ship `pp.annotate(ax, kind="bar_values", ...)` as the primitive for in-plot value labels, with `pp.barplot(..., annotate=True | dict)` as sugar. v1 ships the `bar_values` strategy only; the module is shaped as a dispatcher so `point_values`, `box_medians`, `stacked_segments`, etc. slot in cleanly later.

Labels are orientation-aware (vertical vs horizontal bars), errorbar-aware (anchor past the cap when present), sign-aware (flip for negative bars), anchor-controllable (`outside` / `inside` / `base` / `center`), and contrast-aware in color (compositing the fill onto the axes background before picking dark/light text). Axis limits auto-expand to fit labels when autoscale is on, and stay put when the user or `LayoutReactor` has locked them.

## Non-goals

- Strategies other than `bar_values`. Slots are there, bodies are not.
- Arbitrary user text annotations. That's `ax.annotate`; publiplots is not wrapping it.
- Collision resolution between labels and other artists (legends, other labels). `LayoutReactor` already handles the legend case; label-to-label collision is YAGNI.
- Integration with the complex-heatmap `label()` arrow-style external annotator — different concept, different module.
- Negative `offset` (moving labels *toward* the bar). Semantics unclear; defer.

## Philosophy alignment

Publiplots is a publication-first library with a clear module shape: `plot/` for user-facing plot functions, `layout/` / `themes/` / `fonts/` as top-level packages for cross-cutting machinery, `utils/` for flat helper files. Annotations are cross-cutting infrastructure with a public verb, so they live at the top level as `annotate/`, parallel to `layout/`.

Units are kept small and well-bounded: a pure-math positioning module, a pure-math color module, a cache contract, a dispatcher, and one strategy file. Each testable in isolation, each ~50–150 lines.

mm for all user-facing offsets and pads, consistent with `LegendLayout` / `FigureLayout` / `LayoutReactor`. Matplotlib's display coordinates live only at the boundary.

---

## Architecture

```
pp.barplot(data, x, y, annotate=True | dict)
       │
       ├──► draws bars via seaborn
       ├──► attaches BarValueMeta to ax (bar records, orient, errorbar spec)
       └──► calls annotate(ax, kind="bar_values", **annotate_dict)

pp.annotate(ax, kind="bar_values", anchor="outside", color="auto", ...)
       │
       ├──► _dispatcher picks strategy from a registry
       └──► bar_values strategy:
              ├─ read ax._publiplots_bar_meta  ─► if None, _introspect(ax)
              ├─ for each bar:
              │    ├─ _positioning.resolve_anchor(...)  ─► (x, y, ha, va)
              │    ├─ _color.resolve_color(...)          ─► RGBA
              │    ├─ ax.text(...)
              │    └─ if anchor != "outside": fit_check; re-anchor to "outside" if needed
              └─ _maybe_expand_limits(ax, texts, orient, pad)
```

Five files, each with one responsibility:

```
src/publiplots/annotate/
  ├─ __init__.py        # re-exports: annotate, BarValueMeta
  ├─ _dispatcher.py     # annotate(ax, kind=..., **kws); strategy registry
  ├─ _positioning.py    # resolve_anchor(...), fit_check(...)     — pure math
  ├─ _color.py          # resolve_color(...)                       — pure math
  ├─ _cache.py          # BarValueMeta, BarRecord dataclasses
  └─ bar_values.py      # v1 strategy: orchestrates the above
```

Why five files rather than one: positioning and color are shared infrastructure that the future `point_values` / `box_medians` strategies will reuse. Co-locating them with `bar_values` in a single file would suggest ownership they don't have. The cache is a contract between plot functions and the dispatcher. The dispatcher is trivial but deserves its own file so the `kind=` registry is the single place you look when adding a strategy.

Public API surface, exported from `publiplots/__init__.py`:

```python
from publiplots.annotate import annotate
```

---

## Component 1 — `_cache.py`

Two frozen-ish dataclasses. No matplotlib imports other than the `Rectangle` type annotation.

```python
@dataclass
class BarRecord:
    patch: Rectangle                              # the matplotlib artist
    value: float                                  # aggregated mean for this bar
    err_low: float | None                         # lower errorbar extent in data coords
    err_high: float | None                        # upper errorbar extent in data coords
    hue_color: tuple[float, float, float, float] | None   # palette-assigned RGBA


@dataclass
class BarValueMeta:
    orient: Literal["v", "h"]                     # "v" = vertical bars
    bars: list[BarRecord]                         # in draw order
    errorbar_kind: str | None                     # "se" | "sd" | "ci" | None (informational)
    owner_is_publiplots: bool                     # True iff built by pp.barplot, not introspection
```

Attached to axes as `ax._publiplots_bar_meta`. Stable internal name, same prefix convention as `ArtistTracker`.

**`owner_is_publiplots`** is the flag that distinguishes the limit-expansion behavior between "we control the whole plot" (expand freely) and "user handed us a foreign axes" (respect their `set_ylim`). See *Auto-limit expansion* below.

### Introspection fallback

For foreign axes (no cache):

```python
def _introspect(ax) -> BarValueMeta:
    rects = [p for p in ax.patches if isinstance(p, Rectangle) and p.get_width() > 0 and p.get_height() > 0]
    # orient inference: all bars share width → vertical; all share height → horizontal
    orient = "v" if _all_close([r.get_width() for r in rects]) else "h"
    # errorbar matching: walk ax.lines, match by x-coord (vertical) or y-coord (horizontal)
    err_by_bar = _match_errorbars(ax.lines, rects, orient)
    bars = [
        BarRecord(
            patch=r,
            value=r.get_height() if orient == "v" else r.get_width(),
            err_low=err_by_bar[i][0],
            err_high=err_by_bar[i][1],
            hue_color=r.get_facecolor(),
        )
        for i, r in enumerate(rects)
    ]
    return BarValueMeta(orient=orient, bars=bars, errorbar_kind=None, owner_is_publiplots=False)
```

Errorbar matching tolerance is `0.5 * min_bar_width` (vertical) or `0.5 * min_bar_height` (horizontal). A line whose midpoint falls within this distance of a bar's center is that bar's errorbar. Caps show up as short horizontal/vertical segments; the endpoints of the vertical (or horizontal) line give err_low/err_high. No errorbar found for a bar → `err_low=err_high=None`.

---

## Component 2 — `_positioning.py`

Two pure functions, no drawing. All inputs/outputs are plain numbers or strings.

```python
def resolve_anchor(
    bar: BarRecord,
    anchor: Literal["outside", "inside", "base", "center"],
    orient: Literal["v", "h"],
    offset_mm: float,
    ax: Axes,                                     # needed for transData + get_yscale
) -> tuple[float, float, str, str]:
    """Returns (x, y, ha, va) in data coordinates."""
```

Logic composition (vertical bars shown; horizontal swaps axes and ha/va):

| anchor    | positive value                                       | negative value                                      |
| --------- | ---------------------------------------------------- | --------------------------------------------------- |
| `outside` | y = top + (err_high or 0) + mm→data(offset); va=bottom | y = bottom - (err_low or 0) - mm→data(offset); va=top |
| `inside`  | y = top - mm→data(inner_pad); va=top                 | y = bottom + mm→data(inner_pad); va=bottom          |
| `base`    | y = ylim[0]_if_log_else_0 + mm→data(offset); va=bottom | y = 0 - mm→data(offset); va=top                     |
| `center`  | y = (top + bottom) / 2; va=center                    | y = (top + bottom) / 2; va=center                   |

`ha="center"` in all cases. `x = (left + right) / 2`.

`inner_pad` is a fixed 0.75 mm. `offset_mm` defaults to 1.5 mm, user-configurable. Both convert to data coordinates via `ax.transData.inverted()` applied to a display-coord displacement of `mm * ax.figure.dpi / 25.4`.

**Log-scale on the value axis + `anchor="base"`:** zero doesn't exist in log space. Detect via `ax.get_yscale() == "log"` (or xscale for horizontal); use `ax.get_ylim()[0]` as the "base" reference instead of zero. One-line branch.

```python
def fit_check(
    text_artist: Text,
    bar_bbox_display: Bbox,
    orient: Literal["v", "h"],
    anchor: str,
    renderer,
) -> Literal["fits", "reanchor_outside"]:
    """After provisional draw. Only inside/center/base can fail; outside always fits (by definition)."""
```

For vertical bars + inside/center/base: compare `text_artist.get_window_extent(renderer).height` to bar height in display pixels. For horizontal bars: compare text width to bar width. Margin of 1 px on either side. Returns `"reanchor_outside"` when the text exceeds the bar's bbox in the constraining direction.

Callers that receive `"reanchor_outside"` mutate the existing `Text` artist via `set_position` / `set_ha` / `set_va` / `set_color` rather than removing and redrawing. One artist per bar, no canvas flush in between.

---

## Component 3 — `_color.py`

One pure function. No drawing.

```python
def resolve_color(
    bar: BarRecord,
    color: str | tuple,
    anchor: str,
    ax: Axes,
) -> tuple[float, float, float, float]:
    """Returns RGBA for the text."""
```

Branches:

- **`color="auto"` + anchor in `{"inside", "center", "base"}`:**
  1. Get bar face: `face_rgba = bar.patch.get_facecolor()` (already an RGBA tuple).
  2. Get axes background: `bg_rgba = ax.get_facecolor()`; if transparent, recurse to `fig.get_facecolor()`.
  3. Composite: `effective_rgb = face_rgba[:3] * face_rgba[3] + bg_rgba[:3] * (1 - face_rgba[3])`.
  4. Convert to HLS: `h, l, s = colorsys.rgb_to_hls(*effective_rgb)`.
  5. Luminance threshold 0.408 (W3C / seaborn heatmap convention): `l < 0.408` → light text, else dark text.
  6. Return `matplotlib.colors.to_rgba("#ffffff")` or `matplotlib.colors.to_rgba(rcParams["text.color"])`.

- **`color="auto"` + `anchor="outside"`:** return `to_rgba(rcParams["text.color"])`. The bar isn't underneath the text; there's nothing to contrast against.

- **`color="hue"`:** return `bar.hue_color` if present and non-transparent. Else fall back to `bar.patch.get_edgecolor()` (bars with no hue still have an edge color). Else `rcParams["text.color"]`, and warn once per `annotate()` call: `"pp.annotate: color='hue' requested but no hue colors found; using 'auto'"`.

- **Literal color** (string name, hex, RGB/RGBA tuple): pass through `matplotlib.colors.to_rgba(color)`.

The compositing step matters specifically because publiplots bars are often `alpha=0.1–0.3`: a near-white composited fill needs a dark label, but the raw palette color is saturated and would tell us to use white. Tested in `test_color.py`.

---

## Component 4 — `_dispatcher.py`

```python
_STRATEGIES: dict[str, Callable] = {
    "bar_values": _bar_values_strategy,
}


def annotate(
    ax: Axes,
    kind: Literal["bar_values"] = "bar_values",
    *,
    fmt: str = ".2f",
    anchor: Literal["outside", "inside", "base", "center"] = "outside",
    offset: float = 1.5,                          # mm
    color: str | tuple = "auto",
    pad: float = 1.0,                             # mm, used when auto-expanding limits
    **text_kws,                                   # forwarded to ax.text
) -> list[Text]:
    """Add value labels to plot marks on `ax`. Returns the Text artists created."""
    if kind not in _STRATEGIES:
        raise ValueError(f"unknown kind={kind!r}; known: {sorted(_STRATEGIES)}")
    if anchor not in {"outside", "inside", "base", "center"}:
        raise ValueError(f"anchor must be one of outside/inside/base/center; got {anchor!r}")
    if offset < 0 or pad < 0:
        raise ValueError("offset and pad must be >= 0")
    return _STRATEGIES[kind](ax, fmt=fmt, anchor=anchor, offset=offset, color=color, pad=pad, **text_kws)
```

Input validation is the dispatcher's only job beyond routing. Strategies assume validated input.

`fmt` is a format spec passed to `format(value, fmt)`. Default `.2f`. Users who want `f"{v:,.1f}%"`-style formatting pass a pre-formatted string via `fmt="{:,.1f}%"`; the strategy detects the `{}` substring and uses `fmt.format(value)` instead of `format(value, fmt)`.

---

## Component 5 — `bar_values.py`

The v1 strategy. Orchestrates cache lookup → positioning → color → draw → fit check → limit expansion.

```python
def _bar_values_strategy(ax, *, fmt, anchor, offset, color, pad, **text_kws) -> list[Text]:
    meta = _get_or_introspect(ax)
    if not meta.bars:
        warnings.warn("pp.annotate: no bars found on axes", stacklevel=3)
        return []

    if ax.figure.canvas.get_renderer() is None:
        ax.figure.canvas.draw()                   # force renderer for fit_check
    renderer = ax.figure.canvas.get_renderer()

    texts: list[Text] = []
    for bar in meta.bars:
        if math.isnan(bar.value):
            continue
        x, y, ha, va = resolve_anchor(bar, anchor, meta.orient, offset, ax)
        rgba = resolve_color(bar, color, anchor, ax)
        label = _format_value(bar.value, fmt)
        t = ax.text(x, y, label, ha=ha, va=va, color=rgba, **text_kws)

        if anchor != "outside":
            if fit_check(t, bar.patch.get_window_extent(renderer), meta.orient, anchor, renderer) == "reanchor_outside":
                x2, y2, ha2, va2 = resolve_anchor(bar, "outside", meta.orient, offset, ax)
                rgba2 = resolve_color(bar, color, "outside", ax)
                t.set_position((x2, y2)); t.set_ha(ha2); t.set_va(va2); t.set_color(rgba2)
                logger.debug(f"pp.annotate: bar {bar} label re-anchored to 'outside' (did not fit)")
        texts.append(t)

    _maybe_expand_limits(ax, texts, meta.orient, pad_mm=pad, owner_is_publiplots=meta.owner_is_publiplots)
    return texts
```

### Auto-limit expansion

```python
def _maybe_expand_limits(ax, texts, orient, pad_mm, owner_is_publiplots):
    value_axis = "y" if orient == "v" else "x"
    autoscale_on = ax.get_autoscaley_on() if value_axis == "y" else ax.get_autoscalex_on()

    # publiplots-owned axes: always expand (seaborn disables autoscale via set_ylim; we override).
    # foreign axes: respect user's autoscale state.
    should_expand = owner_is_publiplots or autoscale_on

    text_extents_data = [
        t.get_window_extent(ax.figure.canvas.get_renderer())
         .transformed(ax.transData.inverted())
        for t in texts
    ]
    if value_axis == "y":
        need_min = min(e.y0 for e in text_extents_data)
        need_max = max(e.y1 for e in text_extents_data)
        get_lim, set_lim = ax.get_ylim, ax.set_ylim
    else:
        need_min = min(e.x0 for e in text_extents_data)
        need_max = max(e.x1 for e in text_extents_data)
        get_lim, set_lim = ax.get_xlim, ax.set_xlim

    cur_lo, cur_hi = get_lim()
    pad_data = _mm_to_data(pad_mm, ax, axis=value_axis)

    if should_expand:
        set_lim(min(cur_lo, need_min - pad_data), max(cur_hi, need_max + pad_data))
    else:
        if need_min < cur_lo or need_max > cur_hi:
            warnings.warn("pp.annotate: labels clipped; autoscale is off on this axis", stacklevel=3)
```

The `owner_is_publiplots` branch is what makes `pp.barplot(..., annotate=True)` work without the caller manually re-enabling autoscale — seaborn's `set_ylim(0, ...)` call is fine, we just expand past its upper bound anyway. Calling `pp.annotate(ax)` a second time on the same axes won't re-expand wastefully: the bars and labels haven't moved, so `need_max` is already within the current limits.

### The `pp.barplot` sugar hook

In `src/publiplots/plot/bar.py`, at the very end of `barplot()`, after seaborn draw and all post-processing:

```python
if annotate:
    from publiplots.annotate import annotate as _annotate
    from publiplots.annotate._cache import BarValueMeta, BarRecord
    ax._publiplots_bar_meta = _build_bar_meta_from_seaborn_call(
        ax, data, x, y, hue, errorbar, palette_map, orient,
    )
    opts = annotate if isinstance(annotate, dict) else {}
    _annotate(ax, kind="bar_values", **opts)
```

`_build_bar_meta_from_seaborn_call` is a small private helper in `bar.py` (~30 lines). It re-aggregates the df with the same groupby used by seaborn, pairs each aggregate with the corresponding `Rectangle` in draw order, and records `owner_is_publiplots=True`. The errorbar extents are computed directly from the df using the same `errorbar` spec seaborn used — more reliable than parsing `ax.lines`, and we already have all the inputs.

The `annotate` parameter is added to `barplot()`'s signature as `annotate: Union[bool, Dict, None] = None` in the same keyword-argument block as `legend` / `legend_kws`.

---

## Data flow (worked example)

Input:
```python
pp.barplot(
    data=df, x="group", y="value", hue="cond",
    errorbar="se", annotate={"anchor": "outside", "fmt": ".2f"},
)
```

Sequence:
1. Seaborn draws 4 bars (2 groups × 2 conditions), with se errorbars.
2. `bar.py` builds `BarValueMeta` with 4 `BarRecord`s: each has `patch`, `value` (aggregated mean), `err_low`/`err_high` (SE extents), `hue_color` (from palette_map).
3. `ax._publiplots_bar_meta = meta`; `meta.owner_is_publiplots = True`.
4. `annotate(ax, kind="bar_values", anchor="outside", fmt=".2f")` runs.
5. For each bar: `resolve_anchor` → `(x_c, top + err_high + offset_data)`, `va=bottom`; `resolve_color` → `rcParams["text.color"]` (outside anchor); `ax.text(...)`.
6. No fit-check needed (anchor=outside).
7. `_maybe_expand_limits`: `owner_is_publiplots=True` → always expand. Measures 4 text bboxes, pushes `ylim[1]` up by `max(text.y1) - cur_hi + pad_data`.

Return value: list of 4 `Text` artists from the strategy; user-facing return of `pp.barplot` unchanged (still `(fig, ax)`).

---

## Error handling

| Condition | Response |
| --------- | -------- |
| `kind` not in registry | `ValueError("unknown kind='foo'; known: ['bar_values']")` |
| `anchor` not in valid set | `ValueError(...)` listing valid choices |
| `offset < 0` or `pad < 0` | `ValueError("offset and pad must be >= 0")` |
| `color="hue"` but no hue colors present | Warn once, fall back to `color="auto"` |
| Empty axes (no bars) | `UserWarning("pp.annotate: no bars found on axes")`; return `[]` |
| NaN value bars | Skip silently (don't count as errors); non-NaN bars still labelled |
| Cache attribute exists but wrong type (stale) | `isinstance` check fails → ignore, fall through to introspection |
| Renderer not yet available | Call `fig.canvas.draw()` once before `fit_check`; same trick as `LayoutReactor` |
| Log-scale value axis + `anchor="base"` + value=0 | Anchor to `ylim[0]` (or `xlim[0]`) instead of zero |
| Autoscale off (foreign axes) + labels exceed limits | `UserWarning("labels clipped; autoscale is off on this axis")` |

All warnings go through `warnings.warn(..., stacklevel=3)` so the warning location points to the user's `pp.annotate(...)` call, not publiplots internals. Debug-level re-anchoring messages go through the standard `logging` module (no per-call warning spam).

---

## Testing

Tests live under `tests/annotate/` mirroring the module layout. Four files, pytest fixtures in the existing `tests/conftest.py`.

### `tests/annotate/test_positioning.py` — pure math
Table-driven over `orient × sign × anchor × errorbar-present` = 2×2×4×2 = 32 cases, plus a handful of special cases:
- log-scale value axis + `anchor="base"` → anchor at `ylim[0]`, not zero
- offset=0 vs offset=5mm → positions differ by the expected data-coord delta at fixed dpi
- `fit_check` with tall text + short bar → `"reanchor_outside"`
- `fit_check` with outside anchor → always `"fits"`

### `tests/annotate/test_color.py` — pure math
- auto + dark fill + inside → light text
- auto + light fill + inside → dark text
- auto + `alpha=0.1` on white background → composited → dark text (Q5b regression test)
- auto + anchor=outside → returns `rcParams["text.color"]` regardless of fill
- hue + bar has hue_color → returns it
- hue + bar missing hue_color → falls back to edgecolor, warns once
- literal "#ff0000" / `(1, 0, 0)` / RGBA tuple → all pass through to RGBA

### `tests/annotate/test_bar_values.py` — integration with a real axes
- `pp.barplot(..., annotate=True)` produces N text artists on ax
- `annotate={"fmt": ".3f"}` produces 3-decimal strings
- foreign axes (plain `ax.bar(...)`) → introspection path works, same visual result as our plot
- re-anchor fallback: tiny bars + `anchor="inside"` → label y-positions end up outside the bar
- autoscale expansion (publiplots-owned): bar values span [0, 10], labels push `ylim[1]` beyond 10
- locked limits on foreign axes warn on clip
- NaN value bars are skipped, non-NaN bars still labelled
- hue-colored labels: bars with hue → `color="hue"` paints labels in palette colors
- horizontal orientation: ha/va and x/y swap correctly

### `tests/annotate/test_dispatcher.py`
- unknown kind raises ValueError, message lists known kinds
- invalid anchor raises ValueError
- negative offset/pad raises ValueError
- empty axes returns `[]` and warns
- registry is single-entry in v1 (guards against accidental kind additions without corresponding tests)

### Visual regression
One new example in `examples/plots/plot_XX_annotate.py` demonstrating:
- vertical + outside (default)
- vertical + inside with `color="auto"`
- vertical + `color="hue"`
- horizontal orientation
- negative values with outside anchor
- the fit-check fallback on a mixed-size bar panel

Baked into the gallery build following the same pattern as existing `plot_XX_*.py` examples.

---

## File & API summary

**New files:**
- `src/publiplots/annotate/__init__.py` (~15 lines, re-exports)
- `src/publiplots/annotate/_dispatcher.py` (~40 lines)
- `src/publiplots/annotate/_positioning.py` (~120 lines)
- `src/publiplots/annotate/_color.py` (~80 lines)
- `src/publiplots/annotate/_cache.py` (~100 lines — dataclasses + `_introspect`)
- `src/publiplots/annotate/bar_values.py` (~130 lines)
- `tests/annotate/test_positioning.py`, `test_color.py`, `test_bar_values.py`, `test_dispatcher.py`
- `examples/plots/plot_XX_annotate.py`

**Edited files:**
- `src/publiplots/__init__.py` — add `from publiplots.annotate import annotate` to the public API block
- `src/publiplots/plot/bar.py` — add `annotate` parameter to `barplot()`, add ~40 lines for the cache-building hook + the `annotate(ax, ...)` call at the end

**Public API additions:**
```python
pp.annotate(ax, kind="bar_values", *, fmt=".2f", anchor="outside",
            offset=1.5, color="auto", pad=1.0, **text_kws) -> list[Text]

pp.barplot(..., annotate: bool | dict | None = None)    # new kwarg
```

No breaking changes to existing APIs.

---

## Future work (out of v1 scope, but module is shaped to accept)

- `kind="point_values"` — labels on `pointplot` means
- `kind="box_medians"` — median values on `boxplot`
- `kind="violin_medians"` / `kind="violin_peaks"`
- `kind="stacked_bar_segments"` — value per segment in a stacked bar
- `kind="scatter_labels"` — per-point text from a column (`label_col=`)
- Per-bar conditional formatting (e.g. only label significant bars) via a `mask` parameter
- Label-to-label collision resolution (if it ever matters in practice)

Each would live in its own `<kind>.py` file alongside `bar_values.py`, registered in `_STRATEGIES`, reusing `_positioning` and `_color` unchanged.
