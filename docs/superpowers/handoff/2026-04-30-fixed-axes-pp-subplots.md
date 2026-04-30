# Handoff — Fixed-Axes `pp.subplots()` Helper (PR #79)

**Context**: shipped PR #78 (bulletproof LegendBuilder) and reverted `constrained_layout` from default styles. Publiplots now has a clear philosophy: **fixed axes, flexible canvas** — declared figsize represents the axes rectangle, decorations extend the figure.

**Next feature**: `pp.subplots()` / `pp.figure()` helper that computes figsize from declared axes dimensions + decoration reservations, and positions axes manually. This is the on-ramp to a future Illustrator-composer workflow where every panel has deterministic axes dimensions.

---

## Why this matters

Core user pain: `figsize=(5, 3)` today means "5×3 inches including decorations". Add a legend → axes shrink. Remove the title → axes grow. Compose two figures side-by-side in Illustrator → their axes rectangles don't align. Publication workflows want:

- Declare axes dimensions (e.g. "data panel is 50×30 mm").
- publiplots computes total figure size to accommodate titles, labels, legends, colorbars.
- Axes rectangle stays at the same size and same relative position across every figure.
- Saved SVG has a known, deterministic axes box that an Illustrator script (or future publiplots `Composer`) can align against.

## Current state after PR #78

- `LayoutReactor` keeps legend positions correct across any axes repositioning (tight_layout, subplots_adjust, constrained_layout, or manual `.set_position`).
- `LegendLayout` owns mm-based cursor state, stable across layout changes.
- `pp.legend_group(anchor=...)` for unified legends across subplot grids.
- `set_notebook_style` / `set_publication_style` do NOT force constrained_layout anymore — users declare their layout explicitly.
- Gallery `plot_14` demonstrates the manual pattern: `figsize` wider than axes, `subplots_adjust(right=0.88)` to reserve legend space.

## What `pp.subplots()` should do

### API shape (draft)

```python
fig, axes = pp.subplots(
    nrows=2, ncols=3,
    axes_size=(50, 30),            # mm per axes (data rectangle)
    hspace_mm=8, wspace_mm=8,       # gap between axes rows/cols
    legend_column_mm=30,            # reserve right margin for legend_group
    # optional — per-axes decoration hints
    title_mm=6,                     # reserved above each axes for titles
    xlabel_mm=8,                    # reserved below for x-label
    ylabel_mm=10,                   # reserved left for y-label
    outer_pad_mm=5,                 # figure outer margin
)
# figsize is computed automatically.
# Each axes sits at a deterministic (x, y, w, h) in figure inches.
```

Return value: standard `(fig, axes)` where `axes` is a numpy array like `plt.subplots`, but `fig` has its layout engine explicitly disabled (`layout=None`) and axes positions set manually via `ax.set_position(...)`.

### Size computation

```
axes_w_in = axes_size[0] / 25.4
axes_h_in = axes_size[1] / 25.4
fig_w_in = (
    outer_pad_mm + ylabel_mm
    + ncols * axes_w_in
    + (ncols - 1) * wspace_mm
    + legend_column_mm
    + outer_pad_mm
) / 25.4
fig_h_in = (
    outer_pad_mm + title_mm
    + nrows * axes_h_in
    + (nrows - 1) * hspace_mm
    + xlabel_mm
    + outer_pad_mm
) / 25.4
```

### Integration with LegendBuilder / `pp.legend_group`

When `legend_column_mm > 0`, the helper reserves that column by making figsize wider; it does NOT shrink any axes. `pp.legend_group(anchor=axes[-1])` automatically fits in the reserved column because the builder's mm-based positioning already places the legend `x_offset_mm` to the right of the anchor axes.

### The hard part — "taking decorations into account"

You flagged this: computing `hspace_mm` / `wspace_mm` / `title_mm` reservations up front is ~impossible because text size depends on font, content, and rotation. Options:

**A. Conservative static reservations.** `title_mm=8` covers a ~12pt title even with minor multi-line. Users tweak manually if they hit overflow. Simple, predictable, 80% solution.

**B. Two-pass rendering.** First pass: place axes with conservative reservations, render, measure actual title/xlabel/ylabel sizes, repeat with refined reservations until stable. What `tight_layout` does, but for a fixed-axes contract. Complex but robust.

**C. Hybrid: "measure first" dry run.** On first `pp.subplots(...)` call, render a throwaway sample, measure typical label sizes for the current rcParams, cache them. Fast for subsequent calls. Accurate enough for most plots.

My recommendation: **start with A**. Ship `pp.subplots()` with conservative static reservations that match the publiplots styles. Add clear error messages when the user's title/label text exceeds the reservation (e.g. detect `title_bbox.height > title_mm` after first draw, warn). Upgrade to B or C only if A proves insufficient.

### Composer coordination (future, not this PR)

```python
composer = pp.Composer(page='letter', margin_mm=20)
composer.add(fig_a, row=0, col=0, panel='A')
composer.add(fig_b, row=0, col=1, panel='B')
composer.add(fig_c, row=1, col=0, span_cols=2, panel='C')
composer.save('figure_1.pdf')
```

Each added figure is positioned such that its declared **axes rectangle** aligns to a shared grid. Panel labels ("A", "B", "C") placed at the top-left of each axes rectangle. Composer assumes every `fig` was made with `pp.subplots()` so it knows the axes offsets within the figure. For legacy figures, the composer could optionally take explicit panel positions.

Scope notes for the composer (when it comes): SVG/PDF output, read axes rectangles from the figure's saved metadata, respect the letter-size coordinate system, support multi-panel layouts with span support.

## Concrete plan for PR #79

**Scope** — keep it focused:
1. Create `src/publiplots/layout/subplots.py` with `pp.subplots()` and `pp.figure()` functions (option A — static reservations).
2. Export from `publiplots.__init__`.
3. Update one gallery example (`plot_14`) to use `pp.subplots()` instead of `plt.subplots(...) + fig.subplots_adjust(...)`.
4. Write `plot_15_legends.py` as the legend tutorial (single legend, overflow, `legend_group`, compound handles with marker+size+alpha). Use `pp.subplots()` to demonstrate the fixed-axes pattern.
5. Add collision warning to `LegendBuilder`: at register time, if `ax.x1 + x_offset_mm + estimated_legend_width_mm` extends past the next sibling axes's `x0`, emit a one-shot `UserWarning` suggesting `pp.legend_group` + `pp.subplots(legend_column_mm=...)`.

**Out of scope for PR #79**:
- The composer (PR #80+).
- Two-pass rendering (option B). Static reservations first.
- Changing any existing `pp.legend` / `LegendBuilder` public API.
- Auto-detecting `legend_group` when multiple legends are attached (we explicitly decided against this — explicit API wins).

**Files touched (rough)**:
- Create: `src/publiplots/layout/__init__.py`, `src/publiplots/layout/subplots.py`
- Create: `tests/test_subplots.py`
- Create: `examples/plots/plot_15_legends.py`
- Modify: `src/publiplots/__init__.py` (export)
- Modify: `src/publiplots/utils/legend.py` (collision warning in `LayoutReactor.register` path)
- Modify: `examples/plots/plot_14_edgecolor_control.py` (use `pp.subplots`)

**Tests to add** (in test_subplots.py):
- `test_subplots_figsize_matches_declared_axes_size`
- `test_subplots_preserves_axes_dimensions_regardless_of_legend_column`
- `test_subplots_axes_positions_are_deterministic` (same call twice → identical positions)
- `test_subplots_disables_layout_engine`
- `test_pp_figure_respects_axes_size_in_mm`

## Context for the next session

**Worktree to start in**: create a new worktree off `main` (after PR #78 merges):
```bash
git worktree add .worktrees/pp-subplots -b feat/pp-subplots origin/main
cd .worktrees/pp-subplots
uv venv && uv pip install -e ".[dev,docs]"
uv run pytest tests/ -q  # baseline should be 67 passing
```

**Key files to read first**:
- `src/publiplots/utils/legend_layout.py` — example of a focused dataclass with mm-based geometry.
- `src/publiplots/utils/layout_reactor.py` — the draw-event hook `pp.subplots()` should leave alone.
- `examples/plots/plot_14_edgecolor_control.py` — current manual pattern that `pp.subplots()` will replace.
- `src/publiplots/themes/styles.py` — publiplots style dict (no `constrained_layout.use` anymore).

**Decisions already made** (do not re-litigate):
- No auto-detect of `legend_group`. Explicit API.
- Static-reservation approach (option A) for first ship; measure-based later.
- `LayoutReactor` stays; publiplots styles don't force constrained_layout.
- mm is the canonical unit for declared dimensions.
- `pp.subplots()` disables the matplotlib layout engine (`layout=None`) and manages positions directly.

**Open design questions to raise early with the user**:
- Default values for `title_mm`, `xlabel_mm`, `ylabel_mm` under notebook vs. publication styles (should they differ? probably yes — publication is smaller fonts).
- Whether `pp.subplots(legend_column_mm=N)` should also create a default `pp.legend_group(anchor=axes[-1])` or leave that to the user. My lean: leave it to the user — `legend_column_mm` just reserves space; the user opts in by calling `pp.legend_group`.
- Whether `axes_size` should accept a single scalar for square axes, or a tuple always.

**What NOT to do**:
- Don't touch `LegendBuilder` beyond adding the collision warning.
- Don't add auto-composition features. That's PR #80+.
- Don't try to be smart about text measurement. Ship static-reservation first.
