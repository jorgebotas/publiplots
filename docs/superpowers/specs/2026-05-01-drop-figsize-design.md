# PR C — drop `figsize=` from simple plot functions

**Status:** draft
**Date:** 2026-05-01
**Depends on:** PR #79 (`pp.subplots`), PR #81 (`LayoutReactor` + `SubplotsAutoLayout`), PR #83 + #84 (legend auto-collection).

## Goal

Remove the `figsize=` kwarg from every "simple" publiplots plot function (bar, box, violin, raincloud, scatter, strip, swarm, point, heatmap categorical + dot). All figure creation now routes through `pp.subplots()`, which installs `SubplotsAutoLayout` for auto-sizing. Users who want custom axes dimensions compose: call `pp.subplots(axes_size=...)` first, pass the returned `ax=` into the plot.

## Motivation

When a plot function received `figsize=(w_in, h_in)`, it took a `plt.subplots(figsize=...)` branch that **does not install `SubplotsAutoLayout`**. Legends, titles, and colorbar overflow were not reserved for — figures cropped on save (see `examples/plots/plot_08_raincloud_plots.py` "Horizontal Raincloud Plot" known issue in 0.5.0). publiplots is mm-based and axes-size-first; `figsize` was an orphan from the pre-`pp.subplots` era.

Hard-remove (not deprecation). The user has decided: clean API, no legacy path, let users migrate to the one documented way.

## Scope

### In scope — "simple" plot functions

Every function below loses `figsize`. Each function's figure-creation block collapses to:

```python
if ax is None:
    from publiplots.layout.subplots import subplots as _pp_subplots
    fig, ax = _pp_subplots()
else:
    fig = ax.get_figure()
```

| Plot | File | Signature line |
|---|---|---|
| bar | `src/publiplots/plot/bar.py` | line 44 |
| box | `src/publiplots/plot/box.py` | line 44 |
| violin | `src/publiplots/plot/violin.py` | line 55 |
| raincloud | `src/publiplots/plot/raincloud.py` | line 57 (also forwards `figsize=` to violin at line 210 — drop both) |
| scatter | `src/publiplots/plot/scatter.py` | line 50 |
| strip | `src/publiplots/plot/strip.py` | line 49 |
| swarm | `src/publiplots/plot/swarm.py` | line 48 |
| point | `src/publiplots/plot/point.py` | line 61 |
| heatmap (public) | `src/publiplots/plot/heatmap.py` | line 59 |
| heatmap (dot public) | `src/publiplots/plot/heatmap.py` | line 631 (also drops forwarded `figsize=figsize` at line 779) |

### In scope — rcParam cleanup

Drop `"figure.figsize": [3, 1.8]` from `src/publiplots/themes/rcparams.py:25` and the gallery print statement from `plot_13_configuration.py`. Two internal composites still call `resolve_param("figure.figsize")` today — see below.

### In scope — narrow internal fix for rcParam consumers

After dropping the rcParam, two composites would `KeyError`:

- `src/publiplots/plot/upset/diagram.py:271` — `resolve_param("figure.figsize")` → replace with a hardcoded `(4.0, 3.0)` default (matches matplotlib's own default, preserves existing UpSet geometry).
- `src/publiplots/plot/heatmap.py:885` — `self._figsize = figsize or resolve_param("figure.figsize")` in `ComplexHeatmap.__init__` → replace with `self._figsize = figsize or (4.0, 3.0)`.

These composites still accept `figsize=` for now (they're in OUT-OF-SCOPE composite section below). We're only removing the rcParam dependency, not the `figsize=` kwarg itself on those two.

### In scope — gallery updates

- `examples/plots/plot_08_raincloud_plots.py:85` — drop `figsize=(4, 7)` from the horizontal raincloud example. Rely on auto-layout. If the visual needs to stay wide, use `fig, ax = pp.subplots(axes_size=(70, 70)); pp.raincloudplot(..., ax=ax)`.
- `examples/plots/plot_13_configuration.py:30, 53, 79` — replace `rcParams['figure.figsize']` references with `rcParams['subplots.axes_size']` (the real publiplots sizing knob).

### Out of scope (deferred to a later PR)

- **`ComplexHeatmap`** at `heatmap.py:831` — composite layout with dynamic sizing. `figsize=` stays on its signature; only its rcParam fallback is patched.
- **Venn** (`src/publiplots/plot/venn/*`) — `figsize=` has a different semantic there (aspect-ratio for the square Venn area). Keep as-is.
- **UpSet** (`src/publiplots/plot/upset/*`) — composite. `figsize=` stays on its signature; only its rcParam fallback is patched.
- **Gallery** — `plot_09` (venn) and `plot_11` (complex heatmap) pass `figsize=` to out-of-scope plots; untouched.

## Non-goals

- No new `axes_size=` parameter on plot functions (composition-only — `pp.subplots(axes_size=...)` + `ax=` is the single documented path).
- No deprecation warning — hard `TypeError: unexpected keyword argument 'figsize'`. Users upgrade in one step.
- No changes to `pp.subplots` itself.

## Migration for users

Before:
```python
pp.barplot(data=df, x="x", y="y", figsize=(4, 3))
```

After:
```python
fig, ax = pp.subplots(axes_size=(80, 50))  # mm, not inches
pp.barplot(data=df, x="x", y="y", ax=ax)
```

Or omit for auto:
```python
pp.barplot(data=df, x="x", y="y")  # auto-layout picks defaults
```

## Testing

### Contract tests (new `tests/test_drop_figsize.py`)

For every migrated function, verify:
1. `pp.<plot>(..., figsize=(4, 3))` raises `TypeError` with `figsize` in the message.
2. `fig, ax = pp.<plot>(...)` returns a figure with `SubplotsAutoLayout` installed (`hasattr(fig, "_publiplots_auto_layout")`).
3. `fig, ax = pp.subplots(axes_size=(50, 30)); pp.<plot>(..., ax=ax)` uses the passed axes (no new figure created).

Parametrized across the 10 migrated functions.

### Regression

- Existing 200 tests on main continue to pass after the migration.
- Gallery smoke: all 14 `examples/plots/*.py` render without error.

## Risks

| Risk | Mitigation |
|---|---|
| Internal modules still pass `figsize` to a migrated plot | Exhaustive grep; `raincloud.py:210` is the only known one. |
| `resolve_param("figure.figsize")` called from a place I missed | Full-repo grep for the literal `"figure.figsize"` string before committing. |
| Users on 0.5.0 break silently when upgrading | Documented as BREAKING in CHANGELOG + PR body; release goes out as 0.5.1 (we'd bump to 0.6.0 if this were user-facing long-term, but per user decision it's a minor bump). |
| Gallery `plot_11` (complex heatmap) accidentally affected | complex_heatmap retains `figsize`; confirm plot_11 still renders after the rcParam change. |

## Success criteria

1. `figsize` accepted as a kwarg on zero of the 10 migrated functions.
2. `grep -rn "plt.subplots" src/publiplots/` returns matches only inside `upset/`, `venn/`, and `ComplexHeatmap` (deferred).
3. `"figure.figsize"` string appears in source only within the two rcParam-fallback shims (upset + ComplexHeatmap) — nowhere else in `src/publiplots/`.
4. All tests + gallery green.
