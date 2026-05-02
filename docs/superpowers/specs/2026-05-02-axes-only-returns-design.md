# PR D — axes-only returns from plot functions

**Status:** draft
**Date:** 2026-05-02
**Depends on:** 0.6.0 release (post-PR C).

## Goal

Simplify publiplots plot function return types to match seaborn's convention: plot functions return only the axes they draw into (single `Axes` for simple plots, `dict[str, Axes]` for composite plots). `pp.subplots` — which creates the figure — is unchanged and still returns `(fig, ax)`.

## Motivation

Every publiplots plot function today returns `(fig, ax)`. In practice, `fig` is dead weight:

- Zero `fig.<something>` calls in any gallery example.
- `pp.savefig(filepath, ...)` doesn't take a figure argument — it operates on the current figure via `plt.savefig`. The docstring examples showing `pp.savefig(fig, 'output.png')` are stale and would error today.
- The only genuine consumer of `fig` is `fig.savefig(...)` directly on matplotlib, which is easily recovered via `ax.get_figure().savefig(...)` or `plt.savefig()`.

Dropping `fig` from the return:
- Matches seaborn (`ax = sns.barplot(...)`), the mental model users already have.
- Eliminates the `fig, ax = pp.barplot(...)` → ignore `fig` pattern seen 62 times in the gallery.
- Frees the `fig` variable name when composing: `fig, ax = pp.subplots(); pp.barplot(..., ax=ax)` reads cleanly because `pp.subplots` still owns figure creation.
- One-liner call: `pp.barplot(data=df, x="x", y="y")` works without discarding the `fig` half of the return.

This is a **breaking API change**. 0.x semver is loose, but we'll bump to 0.7.0 in a separate release PR.

## Scope

### In scope — 12 functions

| Function | Before | After |
|---|---|---|
| `barplot` | `(fig, ax)` | `ax` |
| `boxplot` | `(fig, ax)` | `ax` |
| `violinplot` | `(fig, ax)` | `ax` |
| `raincloudplot` | `(fig, ax)` | `ax` |
| `scatterplot` | `(fig, ax)` | `ax` |
| `stripplot` | `(fig, ax)` | `ax` |
| `swarmplot` | `(fig, ax)` | `ax` |
| `pointplot` | `(fig, ax)` | `ax` |
| `heatmap` (categorical + dot) | `(fig, ax)` | `ax` |
| `venn` | `(fig, ax)` | `ax` |
| `upsetplot` | `(fig, (ax_i, ax_m, ax_s))` | `{'intersections', 'matrix', 'sets'}` dict |
| `complex_heatmap().build()` | `(fig, axes_dict)` | `axes_dict` |

### Out of scope

- `pp.subplots(nrows, ncols, ...)` — keeps `(fig, ax)` / `(fig, axes)` return. The user is creating the figure here, so needs the handle.
- `pp.legend_group`, `pp.create_legend_handles`, `pp.savefig` — unrelated surfaces.
- UpSet / ComplexHeatmap composite-layout rework (tracked in `docs/superpowers/handoff/2026-05-02-upset-layout-followup.md`).
- Internal `_legend`, `_stash_*` helpers — never returned figures anyway.
- Version bump to 0.7.0 — separate release PR after this lands.

## Non-goals

- No deprecation warning. Hard break — same strategy as PR C's figsize removal. Python's natural `TypeError: cannot unpack non-iterable AxesSubplot` on `fig, ax = pp.<plot>(...)` is the user's signal to migrate.
- No compatibility wrapper, no `rcParam['return_axes_only']` toggle.
- No change to the `ax=` input parameter on plot functions.

## Migration for users

Before:
```python
fig, ax = pp.barplot(data=df, x="x", y="y")
fig.savefig("out.png")

fig, axes = pp.upsetplot(sets)
axes[0].set_title("Intersections")  # positional

fig, axes = pp.complex_heatmap(df).build()
axes['main'].set_title("Heatmap")
```

After:
```python
ax = pp.barplot(data=df, x="x", y="y")
ax.get_figure().savefig("out.png")   # or just plt.savefig("out.png") / pp.savefig("out.png")

axes = pp.upsetplot(sets)
axes['intersections'].set_title("Intersections")  # named

axes = pp.complex_heatmap(df).build()
axes['main'].set_title("Heatmap")
```

Pattern: add `ax.get_figure()` / `plt.gcf()` only where `fig` was actually used. Most migrations are "delete the `fig,` prefix".

For `upsetplot`, positional unpacking (`ax_i, ax_m, ax_s = pp.upsetplot(sets)`) no longer works because the return is a dict. Migration: use named keys (`axes['intersections']`, `axes['matrix']`, `axes['sets']`).

## Testing

### Contract tests (new `tests/test_axes_only_returns.py`)

Parametrized across the 12 plot functions:

1. **Return type:**
   - Simple plots (10): `isinstance(result, matplotlib.axes.Axes)`.
   - `upsetplot`: `isinstance(result, dict)` and `set(result.keys()) == {"intersections", "matrix", "sets"}`, all values are `Axes`.
   - `complex_heatmap().build()`: `isinstance(result, dict)`, has `"main"` key, value is `Axes`.

2. **Figure accessible via `.get_figure()`:**
   - Simple plots: `result.get_figure() is not None`.
   - Composites: `result['main'].get_figure() is not None` (complex_heatmap) / `result['intersections'].get_figure() is not None` (upsetplot).

3. **`ax=` kwarg still works (simple plots only):**
   - `fig, ax = pp.subplots(axes_size=(50, 30)); result = pp.<plot>(..., ax=ax)`.
   - `assert result is ax` (identity, not equality).

4. **Tuple unpacking fails (regression guard):**
   - `with pytest.raises(TypeError): fig, a = pp.<plot>(...)` — verifies the natural Python error path.

### Existing tests — update mechanically

Every test that does `fig, ax = pp.<plot>(...)` (non-`pp.subplots`) needs to become `ax = pp.<plot>(...)`. Affected files (from grep of existing tests):

- `tests/test_box_legend_stash.py`
- `tests/test_violin_legend_stash.py`
- `tests/test_scatter_legend_stash.py`
- `tests/test_bar_legend_stash.py`
- `tests/test_heatmap_legend_stash.py`
- `tests/test_raincloud_legend_group.py`
- `tests/test_edgecolor.py`
- `tests/test_edgecolor_rcparam.py`
- `tests/test_drop_figsize.py` (contract tests that call each plot)
- `tests/test_plot_legend.py`
- `tests/test_subplots.py` (only the plot-calling sections; `pp.subplots` returns unchanged)
- `tests/test_bar_hatch_eq_categorical.py`

The change is always: `fig, ax = pp.<plot>(...)` → `ax = pp.<plot>(...)`. Drop any subsequent `fig.<something>` if present; replace with `ax.get_figure().<something>` only when the `fig` was actually used (most tests don't use it).

### Gallery + docs

- **Gallery** (`examples/plots/plot_*.py`): 62 lines of `fig, ax = pp.<plot>(...)` become `ax = pp.<plot>(...)`. Plus upsetplot + complex_heatmap adjustments (positional→dict for upset).
- **Docstrings** in `src/publiplots/__init__.py` (lines 10–11) — update the stale `pp.savefig(fig, 'output.png')` example.
- **Sphinx docs** `docs/source/index.rst:70`, `docs/source/quickstart.rst:176-179` — same update.

### Full suite target

- Baseline: 235 passed.
- New contract tests: ~48 (12 plots × 4 assertions).
- Net after updates + additions: ~283 total passed.
- All 14 gallery examples render.

## Risks

| Risk | Mitigation |
|---|---|
| Users upgrading from 0.6.0 get `TypeError: cannot unpack non-iterable AxesSubplot` | Prominent CHANGELOG entry with migration recipe; clear error message from Python itself |
| Internal `return fig, ax` occurrences I miss | Grep audit in Task 1: `grep -rn "return fig, ax\|return fig," src/publiplots/plot/` — expected to match exactly 12 occurrences (one per migrated function) |
| `upsetplot` users with positional unpack (`a, b, c = pp.upsetplot(...)`) | Break is deliberate; dict keys are the canonical UpSet vocabulary. Migration is trivial |
| `complex_heatmap` builder's `.build()` has downstream callers | Update both the class method and any `return fig, ...` call sites |
| Tuple-return appears in any type annotation that also lives in a public module (importable) | Grep for `Tuple[plt.Figure, Axes]` etc. and swap to `Axes` / `Dict[str, Axes]` |

## Dependencies and ordering

Single PR. All 12 plot function migrations + test updates + gallery + docstring updates land together. No partial state: the type system and the imports need to be consistent.

## Success criteria

1. `grep -rn "return fig, ax\|return fig," src/publiplots/plot/` returns only internal helper occurrences (if any survive) — no public plot function returns a tuple.
2. `grep -rn "fig, ax = pp\." examples/ docs/ tests/` returns zero hits for non-`pp.subplots` calls.
3. New `tests/test_axes_only_returns.py` passes all 48 parametrized assertions.
4. All 14 gallery examples render without error.
5. `pp.subplots` return signature unchanged — confirmed by its own tests continuing to pass.
