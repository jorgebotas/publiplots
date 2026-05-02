# PR D — Axes-Only Returns Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Change every publiplots plot function's return type from `(fig, ax)` to `ax` (or `axes_dict` for composites). `pp.subplots` is unchanged — it creates the figure so the user needs the handle.

**Architecture:** Single-PR breaking change touching 13 public plot functions + 2 internal helpers + 12 test files + 14 gallery files + 3 docs files. No deprecation path — Python's natural `TypeError: cannot unpack non-iterable Axes object` is the user's migration signal.

**Tech Stack:** Python 3.12, matplotlib, pytest.

**Worktree:** `/home/sagemaker-user/publiplots/.worktrees/axes-only-returns` on branch `feat/axes-only-returns`, based on `main` at `106287e`. Baseline: 235 tests passing.

**Environment note:** `.venv` is inside the worktree. Some shells have a stale `VIRTUAL_ENV` var — always prefix pytest with `unset VIRTUAL_ENV && .venv/bin/pytest ...`.

---

## File Structure

Files modified (src):
- `src/publiplots/plot/bar.py` — `barplot` return (line 327) + return type annotation
- `src/publiplots/plot/box.py` — `boxplot` return (line 279)
- `src/publiplots/plot/violin.py` — `violinplot` return (line 301)
- `src/publiplots/plot/raincloud.py` — `raincloudplot` return (line 283)
- `src/publiplots/plot/scatter.py` — `scatterplot` return (line 340)
- `src/publiplots/plot/strip.py` — `stripplot` return (line 220)
- `src/publiplots/plot/swarm.py` — `swarmplot` return (line 216)
- `src/publiplots/plot/point.py` — `pointplot` return (line 328)
- `src/publiplots/plot/heatmap.py` — public `heatmap` (line 286), `_draw_heatmap` (line 371), `_draw_dot_heatmap` (line 542), `ComplexHeatmapBuilder.build` (line 1287), public `dendrogram` (line 1466), plus internal dispatchers at lines 231, 256
- `src/publiplots/plot/venn/diagram.py` — `venn` return (line 326)
- `src/publiplots/plot/upset/diagram.py` — `upsetplot` return (line 333, `(fig, 3-tuple)` → `dict`)

Files modified (tests):
- `tests/test_box_legend_stash.py`
- `tests/test_violin_legend_stash.py`
- `tests/test_scatter_legend_stash.py`
- `tests/test_bar_legend_stash.py`
- `tests/test_heatmap_legend_stash.py`
- `tests/test_raincloud_legend_group.py`
- `tests/test_edgecolor.py`
- `tests/test_edgecolor_rcparam.py`
- `tests/test_drop_figsize.py`
- `tests/test_plot_legend.py`
- `tests/test_subplots.py` (only plot-calling sections)
- `tests/test_bar_hatch_eq_categorical.py`

Files modified (examples):
- All 14 `examples/plots/plot_*.py`

Files modified (docs):
- `src/publiplots/__init__.py` (docstring example, line ~11)
- `docs/source/index.rst` (line ~70)
- `docs/source/quickstart.rst` (lines ~176–179)

Files created:
- `tests/test_axes_only_returns.py` — parametrized contract tests

---

## Task 1: Contract test file (red baseline)

**Files:**
- Create: `tests/test_axes_only_returns.py`

Ships first so the migration is TDD-driven. Initial state: most assertions fail.

- [ ] **Step 1: Write the test file**

Create `tests/test_axes_only_returns.py`:

```python
"""Contract tests for PR D — axes-only returns from plot functions.

Simple plots return a single Axes.
Composites (upsetplot, complex_heatmap.build) return a dict[str, Axes].
pp.subplots is unchanged (returns (fig, ax)) and not tested here.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import pandas as pd
import numpy as np
import pytest

import publiplots as pp


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def scatter_df():
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "x": rng.normal(size=50),
        "y": rng.normal(size=50),
        "g": rng.choice(["A", "B"], size=50),
    })


@pytest.fixture(scope="module")
def cat_df():
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "cat": rng.choice(["A", "B", "C"], size=60),
        "val": rng.normal(size=60),
        "g": rng.choice(["ctrl", "trt"], size=60),
    })


@pytest.fixture(scope="module")
def matrix_df():
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.normal(size=(4, 5)))


@pytest.fixture(scope="module")
def dot_df():
    rng = np.random.default_rng(0)
    rows, cols = ["r0", "r1", "r2"], ["c0", "c1", "c2", "c3"]
    data = []
    for r in rows:
        for c in cols:
            data.append({"row": r, "col": c, "val": rng.normal(), "sz": rng.uniform(1, 10)})
    return pd.DataFrame(data)


SIMPLE_PLOTS = [
    ("barplot",       "cat_df",     {"x": "cat", "y": "val"}),
    ("boxplot",       "cat_df",     {"x": "cat", "y": "val"}),
    ("violinplot",    "cat_df",     {"x": "cat", "y": "val"}),
    ("raincloudplot", "cat_df",     {"x": "cat", "y": "val"}),
    ("scatterplot",   "scatter_df", {"x": "x",   "y": "y"}),
    ("stripplot",     "cat_df",     {"x": "cat", "y": "val"}),
    ("swarmplot",     "cat_df",     {"x": "cat", "y": "val"}),
    ("pointplot",     "cat_df",     {"x": "cat", "y": "val"}),
    ("heatmap",       "matrix_df",  {}),
    ("heatmap",       "dot_df",     {"x": "col", "y": "row", "value": "val", "size": "sz"}),
]

_SIMPLE_IDS = [
    "barplot", "boxplot", "violinplot", "raincloudplot", "scatterplot",
    "stripplot", "swarmplot", "pointplot", "heatmap-categorical", "heatmap-dot",
]


@pytest.mark.parametrize("fn_name,df_fixture,kwargs", SIMPLE_PLOTS, ids=_SIMPLE_IDS)
def test_simple_plot_returns_axes(fn_name, df_fixture, kwargs, request):
    df = request.getfixturevalue(df_fixture)
    fn = getattr(pp, fn_name)
    result = fn(data=df, **kwargs)
    assert isinstance(result, Axes), \
        f"{fn_name} returned {type(result).__name__}; expected matplotlib.axes.Axes"


@pytest.mark.parametrize("fn_name,df_fixture,kwargs", SIMPLE_PLOTS, ids=_SIMPLE_IDS)
def test_simple_plot_figure_accessible(fn_name, df_fixture, kwargs, request):
    df = request.getfixturevalue(df_fixture)
    fn = getattr(pp, fn_name)
    ax = fn(data=df, **kwargs)
    assert ax.get_figure() is not None


@pytest.mark.parametrize("fn_name,df_fixture,kwargs", SIMPLE_PLOTS, ids=_SIMPLE_IDS)
def test_simple_plot_ax_kwarg_returns_same_ax(fn_name, df_fixture, kwargs, request):
    df = request.getfixturevalue(df_fixture)
    fn = getattr(pp, fn_name)
    fig, ax = pp.subplots(axes_size=(50, 30))
    result = fn(data=df, **kwargs, ax=ax)
    assert result is ax


@pytest.mark.parametrize("fn_name,df_fixture,kwargs", SIMPLE_PLOTS, ids=_SIMPLE_IDS)
def test_simple_plot_tuple_unpack_raises(fn_name, df_fixture, kwargs, request):
    """Regression guard: plot return is NOT a tuple."""
    df = request.getfixturevalue(df_fixture)
    fn = getattr(pp, fn_name)
    with pytest.raises(TypeError):
        fig, ax = fn(data=df, **kwargs)  # noqa: F841


# ---- Composite plots ----


def test_upsetplot_returns_dict():
    sets = {"A": {1, 2}, "B": {2, 3}, "C": {3, 4}}
    result = pp.upsetplot(sets)
    assert isinstance(result, dict)
    assert set(result.keys()) == {"intersections", "matrix", "sets"}
    for k, v in result.items():
        assert isinstance(v, Axes), f"upsetplot['{k}'] is {type(v).__name__}, expected Axes"


def test_upsetplot_figure_accessible():
    sets = {"A": {1, 2}, "B": {2, 3}, "C": {3, 4}}
    axes = pp.upsetplot(sets)
    assert axes["intersections"].get_figure() is not None


def test_upsetplot_tuple_unpack_raises():
    sets = {"A": {1, 2}, "B": {2, 3}, "C": {3, 4}}
    with pytest.raises(TypeError):
        fig, axes = pp.upsetplot(sets)  # noqa: F841


def test_complex_heatmap_build_returns_dict(matrix_df):
    axes = pp.complex_heatmap(matrix_df).build()
    assert isinstance(axes, dict)
    assert "main" in axes
    assert isinstance(axes["main"], Axes)


def test_complex_heatmap_build_figure_accessible(matrix_df):
    axes = pp.complex_heatmap(matrix_df).build()
    assert axes["main"].get_figure() is not None


def test_complex_heatmap_build_tuple_unpack_raises(matrix_df):
    with pytest.raises(TypeError):
        fig, axes = pp.complex_heatmap(matrix_df).build()  # noqa: F841


# ---- Dendrogram (separate public API) ----


def test_dendrogram_returns_axes():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(8, 5)))
    result = pp.dendrogram(data=df)
    assert isinstance(result, Axes)


def test_dendrogram_figure_accessible():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(8, 5)))
    ax = pp.dendrogram(data=df)
    assert ax.get_figure() is not None


def test_dendrogram_tuple_unpack_raises():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(8, 5)))
    with pytest.raises(TypeError):
        fig, ax = pp.dendrogram(data=df)  # noqa: F841
```

- [ ] **Step 2: Confirm red baseline**

Run: `unset VIRTUAL_ENV && .venv/bin/pytest tests/test_axes_only_returns.py -v`
Expected: 40 failures (everything except the `_figure_accessible` tests which currently pass because `ax.get_figure()` works on the returned ax even if it came via tuple unpack inside matplotlib code). Roughly:
- `test_simple_plot_returns_axes[*]`: 10 FAIL (return is tuple, not Axes)
- `test_simple_plot_ax_kwarg_returns_same_ax[*]`: 10 FAIL (returns `(fig, ax)`, not just `ax`)
- `test_simple_plot_tuple_unpack_raises[*]`: 10 FAIL (tuple unpack DOES succeed today)
- `test_upsetplot_*`: 3 FAIL (returns tuple, not dict; positional ax list)
- `test_complex_heatmap_*`: 3 FAIL
- `test_dendrogram_*`: 3 FAIL
Some `_figure_accessible` tests may also fail if the current tuple unpack changes the ax variable shape. Don't fix any of these — they turn green as tasks 2–9 land.

- [ ] **Step 3: Commit**

```bash
git add tests/test_axes_only_returns.py
git commit -m "test(axes-only): add red-baseline contract tests for PR D"
```

---

## Task 2: Migrate `barplot`, `boxplot`, `violinplot`, `raincloudplot`

**Files:**
- Modify: `src/publiplots/plot/bar.py`
- Modify: `src/publiplots/plot/box.py`
- Modify: `src/publiplots/plot/violin.py`
- Modify: `src/publiplots/plot/raincloud.py`

Mechanical edits: change each `return fig, ax` → `return ax` and update the `-> Tuple[plt.Figure, Axes]:` return annotation to `-> Axes:`.

- [ ] **Step 1: Edit `src/publiplots/plot/bar.py`**

1. Find the function signature `def barplot(...)` and change the closing annotation:

```python
) -> Tuple[plt.Figure, Axes]:
```

to:

```python
) -> Axes:
```

2. Find the last `return fig, ax` (around line 327) and change to:

```python
    return ax
```

3. After the edits, grep `grep -n "Tuple\[plt.Figure\|return fig" src/publiplots/plot/bar.py` to confirm both occurrences are gone.

4. If `Tuple` is no longer used elsewhere in the file, don't remove the import — many plots keep `Tuple` for other annotations.

- [ ] **Step 2: Apply the same pattern to `src/publiplots/plot/box.py`**

1. Signature `-> Tuple[plt.Figure, Axes]:` → `-> Axes:`
2. `return fig, ax` (line ~279) → `return ax`

- [ ] **Step 3: Apply to `src/publiplots/plot/violin.py`**

1. Signature annotation fix (line ~63).
2. `return fig, ax` (line ~301) → `return ax`.

- [ ] **Step 4: Apply to `src/publiplots/plot/raincloud.py`**

1. Signature annotation fix (line ~63).
2. `return fig, ax` (line ~283) → `return ax`.

- [ ] **Step 5: Run the 4-plot contract tests**

Run: `unset VIRTUAL_ENV && .venv/bin/pytest tests/test_axes_only_returns.py -v -k "barplot or boxplot or violinplot or raincloudplot"`
Expected: all 16 assertions (4 plots × 4 tests) PASS.

- [ ] **Step 6: Update the 4 plots' legend-stash tests to accept single `ax` return**

The tests in `tests/test_box_legend_stash.py`, `tests/test_violin_legend_stash.py`, `tests/test_bar_legend_stash.py`, `tests/test_raincloud_legend_group.py`, and `tests/test_bar_hatch_eq_categorical.py` currently do:

```python
fig, ax = pp.<plot>(data=df, ...)
```

For each occurrence (use grep to find them), replace with:

```python
ax = pp.<plot>(data=df, ...)
```

If any test uses `fig` afterward, replace with `ax.get_figure()`.

Files to grep:

```bash
grep -n "fig, ax = pp\.\(barplot\|boxplot\|violinplot\|raincloudplot\)" \
  tests/test_box_legend_stash.py tests/test_violin_legend_stash.py \
  tests/test_bar_legend_stash.py tests/test_raincloud_legend_group.py \
  tests/test_bar_hatch_eq_categorical.py
```

Edit each match. Then run:

```bash
unset VIRTUAL_ENV && .venv/bin/pytest \
  tests/test_box_legend_stash.py \
  tests/test_violin_legend_stash.py \
  tests/test_bar_legend_stash.py \
  tests/test_raincloud_legend_group.py \
  tests/test_bar_hatch_eq_categorical.py -v
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/publiplots/plot/bar.py src/publiplots/plot/box.py \
        src/publiplots/plot/violin.py src/publiplots/plot/raincloud.py \
        tests/test_box_legend_stash.py tests/test_violin_legend_stash.py \
        tests/test_bar_legend_stash.py tests/test_raincloud_legend_group.py \
        tests/test_bar_hatch_eq_categorical.py
git commit -m "feat(breaking)!: barplot/boxplot/violinplot/raincloudplot return ax only"
```

---

## Task 3: Migrate `scatterplot`, `stripplot`, `swarmplot`, `pointplot`

**Files:**
- Modify: `src/publiplots/plot/scatter.py` (line ~340)
- Modify: `src/publiplots/plot/strip.py` (line ~220)
- Modify: `src/publiplots/plot/swarm.py` (line ~216)
- Modify: `src/publiplots/plot/point.py` (line ~328)

Same pattern as Task 2.

- [ ] **Step 1: For each of the 4 files**

1. Find the public plot function's return annotation, change `-> Tuple[plt.Figure, Axes]:` → `-> Axes:`.
2. Find the `return fig, ax` near the bottom of the function, change to `return ax`.

- [ ] **Step 2: Run the contract tests**

Run: `unset VIRTUAL_ENV && .venv/bin/pytest tests/test_axes_only_returns.py -v -k "scatterplot or stripplot or swarmplot or pointplot"`
Expected: 16 PASS.

- [ ] **Step 3: Update tests that call these plots**

Grep for `fig, ax = pp.(scatter|strip|swarm|point)plot` across tests and replace with `ax = pp.<plot>(...)`:

```bash
grep -n "fig, ax = pp\.\(scatter\|strip\|swarm\|point\)plot" \
  tests/test_scatter_legend_stash.py tests/test_plot_legend.py \
  tests/test_edgecolor.py tests/test_edgecolor_rcparam.py
```

Replace each. Run the updated tests:

```bash
unset VIRTUAL_ENV && .venv/bin/pytest \
  tests/test_scatter_legend_stash.py tests/test_plot_legend.py \
  tests/test_edgecolor.py tests/test_edgecolor_rcparam.py -v
```

Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add src/publiplots/plot/scatter.py src/publiplots/plot/strip.py \
        src/publiplots/plot/swarm.py src/publiplots/plot/point.py \
        tests/test_scatter_legend_stash.py tests/test_plot_legend.py \
        tests/test_edgecolor.py tests/test_edgecolor_rcparam.py
git commit -m "feat(breaking)!: scatter/strip/swarm/point return ax only"
```

---

## Task 4: Migrate `heatmap` (public + internal helpers)

**Files:**
- Modify: `src/publiplots/plot/heatmap.py`

Four relevant functions:
- Public `heatmap(...)` — line 31 signature, line 286 return.
- Internal `_draw_heatmap(...)` — line 289 signature, line 371 return.
- Internal `_draw_dot_heatmap(...)` — line 374 signature, line 542 return.
- Internal dispatchers inside `heatmap()` at lines 231 and 256.

- [ ] **Step 1: Update the public `heatmap` signature annotation**

Change the return annotation from `-> Tuple[plt.Figure, Axes]:` to `-> Axes:` on line ~71.

- [ ] **Step 2: Update internal `_draw_heatmap` signature and return**

Change line ~309 annotation from `-> Tuple[plt.Figure, Axes]:` to `-> Axes:`.
Change line ~371 from `return fig, ax` to `return ax`.

- [ ] **Step 3: Update internal `_draw_dot_heatmap` signature and return**

Change line ~396 annotation from `-> Tuple[plt.Figure, Axes]:` to `-> Axes:`.
Change line ~542 from `return fig, ax` to `return ax`.

- [ ] **Step 4: Update internal dispatchers inside public `heatmap`**

Find the two `fig, ax = _draw_*` lines at ~231 and ~256:

```python
        fig, ax = _draw_dot_heatmap(...)
```

```python
        fig, ax = _draw_heatmap(...)
```

Change both to:

```python
        ax = _draw_dot_heatmap(...)
```

```python
        ax = _draw_heatmap(...)
```

- [ ] **Step 5: Update public `heatmap` final return**

Line ~286: `return fig, ax` → `return ax`.

- [ ] **Step 6: Confirm**

```bash
grep -n "return fig\|Tuple\[plt.Figure" src/publiplots/plot/heatmap.py
```

Expected: only `ComplexHeatmapBuilder.build` (line ~1287) and `dendrogram` (line ~1466 + its signature) should still match. Those are Tasks 5 and 6.

- [ ] **Step 7: Run heatmap contract + stash tests**

```bash
unset VIRTUAL_ENV && .venv/bin/pytest tests/test_axes_only_returns.py -v -k "heatmap-categorical or heatmap-dot"
```

Expected: 8 PASS.

Then update and run `tests/test_heatmap_legend_stash.py`. Grep for `fig, ax = pp.heatmap` and replace with `ax = pp.heatmap(...)`. Also check `tests/test_drop_figsize.py` for heatmap calls. Run:

```bash
unset VIRTUAL_ENV && .venv/bin/pytest tests/test_heatmap_legend_stash.py tests/test_drop_figsize.py -v
```

Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add src/publiplots/plot/heatmap.py tests/test_heatmap_legend_stash.py tests/test_drop_figsize.py
git commit -m "feat(breaking)!: heatmap returns ax only (public + _draw_* internals)"
```

---

## Task 5: Migrate `ComplexHeatmapBuilder.build`

**Files:**
- Modify: `src/publiplots/plot/heatmap.py` (line ~1287 return, line ~1143 signature)

- [ ] **Step 1: Update `build` signature annotation**

Find `def build(self) -> Tuple[plt.Figure, Dict[str, Union[Axes, List[Axes]]]]:` (line ~1143). Change return annotation to:

```python
    def build(self) -> Dict[str, Union[Axes, List[Axes]]]:
```

- [ ] **Step 2: Update the return statement**

Line ~1287: `return fig, axes` → `return axes`.

- [ ] **Step 3: Run contract tests**

```bash
unset VIRTUAL_ENV && .venv/bin/pytest tests/test_axes_only_returns.py -v -k "complex_heatmap"
```

Expected: 3 PASS.

- [ ] **Step 4: Commit**

```bash
git add src/publiplots/plot/heatmap.py
git commit -m "feat(breaking)!: ComplexHeatmapBuilder.build returns axes_dict only"
```

---

## Task 6: Migrate `dendrogram`

**Files:**
- Modify: `src/publiplots/plot/heatmap.py` (line ~1382 signature, line ~1466 return)

- [ ] **Step 1: Signature annotation**

Line ~1382: `-> Tuple[plt.Figure, Axes]:` → `-> Axes:`.

- [ ] **Step 2: Return**

Line ~1466: `return fig, ax` → `return ax`.

- [ ] **Step 3: Run contract tests**

```bash
unset VIRTUAL_ENV && .venv/bin/pytest tests/test_axes_only_returns.py -v -k "dendrogram"
```

Expected: 3 PASS.

- [ ] **Step 4: Commit**

```bash
git add src/publiplots/plot/heatmap.py
git commit -m "feat(breaking)!: dendrogram returns ax only"
```

---

## Task 7: Migrate `venn`

**Files:**
- Modify: `src/publiplots/plot/venn/diagram.py` (line ~207 signature, line ~326 return)

- [ ] **Step 1: Signature annotation**

Change `-> Tuple[plt.Figure, Axes]:` on line ~207 to `-> Axes:`.

- [ ] **Step 2: Return**

Line ~326: `return fig, ax` → `return ax`.

- [ ] **Step 3: Run contract tests**

```bash
unset VIRTUAL_ENV && .venv/bin/pytest tests/test_drop_figsize.py -v -k "venn"
```

(No venn-specific contract test in `test_axes_only_returns.py` — venn is exercised via `test_drop_figsize.py::test_venn_default_installs_auto_layout` etc. If `fig, ax = pp.venn(...)` appears there, update to `ax = pp.venn(...)`.)

Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add src/publiplots/plot/venn/diagram.py tests/test_drop_figsize.py
git commit -m "feat(breaking)!: venn returns ax only"
```

---

## Task 8: Migrate `upsetplot` — tuple → dict

**Files:**
- Modify: `src/publiplots/plot/upset/diagram.py` (line ~333 return, function signature)

- [ ] **Step 1: Find current return statement**

Line ~333: `return fig, (ax_intersections, ax_matrix, ax_sets)`.

- [ ] **Step 2: Update the return + annotation**

Change the function's return annotation. Find the `def upsetplot(...)` block (line ~31) and change its closing `-> ...` to:

```python
) -> Dict[str, Axes]:
```

(Add `from typing import Dict` if missing; `Axes` should already be imported.)

Change the return on line ~333 from:

```python
    return fig, (ax_intersections, ax_matrix, ax_sets)
```

to:

```python
    return {
        "intersections": ax_intersections,
        "matrix": ax_matrix,
        "sets": ax_sets,
    }
```

- [ ] **Step 3: Run contract tests**

```bash
unset VIRTUAL_ENV && .venv/bin/pytest tests/test_axes_only_returns.py -v -k "upsetplot"
```

Expected: 3 PASS.

- [ ] **Step 4: Commit**

```bash
git add src/publiplots/plot/upset/diagram.py
git commit -m "feat(breaking)!: upsetplot returns dict {intersections, matrix, sets}"
```

---

## Task 9: Update `tests/test_subplots.py`

**Files:**
- Modify: `tests/test_subplots.py`

`test_subplots.py` exists from PR #79 and tests both `pp.subplots` (which is unchanged — keeps tuple return) AND plot functions called through `pp.subplots`. Need to audit and fix only the plot-calling sections.

- [ ] **Step 1: Grep for plot-calling lines that need updates**

```bash
grep -n "fig, ax = pp\.\(barplot\|boxplot\|violinplot\|raincloudplot\|scatterplot\|stripplot\|swarmplot\|pointplot\|heatmap\)" tests/test_subplots.py
```

For each match, replace `fig, ax = pp.<plot>(...)` with `ax = pp.<plot>(...)`. Do NOT touch `fig, ax = pp.subplots(...)` lines — those stay.

- [ ] **Step 2: Run the subplots test file**

```bash
unset VIRTUAL_ENV && .venv/bin/pytest tests/test_subplots.py -v
```

Expected: all pass.

- [ ] **Step 3: Full suite**

```bash
unset VIRTUAL_ENV && .venv/bin/pytest -q
```

Expected: 235 (baseline) + 40 (new contract tests) = 275 passed.

- [ ] **Step 4: Commit**

```bash
git add tests/test_subplots.py
git commit -m "test(subplots): drop fig unpack from plot calls in tests"
```

---

## Task 10: Update the 14 gallery examples

**Files:**
- Modify: `examples/plots/plot_01_bar_plots.py` through `examples/plots/plot_14_edgecolor_control.py`

- [ ] **Step 1: Rewrite plot-calling lines**

For each file in `examples/plots/plot_*.py`:

1. Grep for `fig, ax = pp\.<plot>(...)` (everything except `pp.subplots`). Replace with `ax = pp.<plot>(...)`.
2. Grep for `fig, ax = pp.subplots(...)` — KEEP AS-IS.
3. For `plot_10_upset_plots.py`, grep for `fig, axes = pp.upsetplot(...)`. The new form is `axes = pp.upsetplot(...)`. If the code then does `axes[0]`, `axes[1]`, `axes[2]` (positional), change to `axes['intersections']`, `axes['matrix']`, `axes['sets']`. If it does `axes.foo(...)` with an array, that won't exist anymore.
4. For `plot_11_heatmap.py`, grep for `fig, axes = pp.complex_heatmap(...).build()`. The new form is `axes = pp.complex_heatmap(...).build()`. Dict access (e.g. `axes['main']`) is unchanged.

The simplest audit loop per file:

```bash
for f in examples/plots/plot_*.py; do
    echo "=== $f ==="
    grep -n "fig, ax\|fig, axes" "$f" | grep -v "pp.subplots"
done
```

Each line shown is a candidate for rewrite. Lines that reference `fig` later (e.g. `fig.savefig(...)`) should become `ax.get_figure().savefig(...)` (or use `plt.savefig(...)` directly).

- [ ] **Step 2: Run the full gallery**

```bash
unset VIRTUAL_ENV
for f in examples/plots/plot_*.py; do
    .venv/bin/python "$f" >/tmp/g.out 2>&1 && echo "$(basename $f): OK" || (echo "$(basename $f): FAIL"; tail -20 /tmp/g.out)
done
```

Expected: all 14 OK.

- [ ] **Step 3: Commit**

```bash
git add examples/plots/
git commit -m "docs(examples): drop fig unpack from plot calls in gallery"
```

---

## Task 11: Update docstrings in `src/publiplots/__init__.py` and sphinx docs

**Files:**
- Modify: `src/publiplots/__init__.py` (docstring around line 10–11)
- Modify: `docs/source/index.rst` (around line 70)
- Modify: `docs/source/quickstart.rst` (lines ~176–179)

Existing stale examples show `fig, ax = pp.barplot(...)` + `pp.savefig(fig, 'output.png')` (which was already wrong — pp.savefig takes a filepath, not a fig).

- [ ] **Step 1: Edit `src/publiplots/__init__.py`**

Read the file and find the module docstring. Update the "Example" block:

```python
    >>> import publiplots as pp
    >>> fig, ax = pp.barplot(data=df, x='category', y='value')
    >>> pp.savefig(fig, 'output.png')
```

to:

```python
    >>> import publiplots as pp
    >>> ax = pp.barplot(data=df, x='category', y='value')
    >>> pp.savefig('output.png')
```

- [ ] **Step 2: Edit `docs/source/index.rst`**

Find the quickstart-style example around line 70. Apply the same pattern: drop `fig, ` from `fig, ax = pp.<plot>(...)` and change `pp.savefig(fig, ...)` to `pp.savefig(...)`.

- [ ] **Step 3: Edit `docs/source/quickstart.rst`**

Around lines 176–179. Same pattern.

- [ ] **Step 4: Verify no other stale `pp.savefig(fig, ...)` references**

```bash
grep -rn "pp.savefig(fig\|publiplots.savefig(fig" . --include="*.py" --include="*.rst" --include="*.md"
```

Expected: zero hits.

- [ ] **Step 5: Commit**

```bash
git add src/publiplots/__init__.py docs/source/index.rst docs/source/quickstart.rst
git commit -m "docs: update stale fig-unpack + pp.savefig(fig,…) examples"
```

---

## Task 12: Final full-suite + gallery verification

**Files:** none — verification only.

- [ ] **Step 1: Full test suite**

```bash
unset VIRTUAL_ENV && .venv/bin/pytest -q
```

Expected: 275 passed (235 baseline + 40 new contract tests − 0 deleted).

- [ ] **Step 2: Full gallery**

```bash
unset VIRTUAL_ENV
for f in examples/plots/plot_*.py; do
    .venv/bin/python "$f" >/dev/null 2>&1 && echo "$(basename $f): OK" || echo "$(basename $f): FAIL"
done
```

Expected: all 14 OK.

- [ ] **Step 3: Audit — no stray `fig, ax = pp.<plot>` left**

```bash
grep -rn "fig, ax = pp\.\|fig, axes = pp\." . \
    --include="*.py" --include="*.rst" --include="*.md" \
    | grep -v "pp.subplots" | grep -v ".worktrees"
```

Expected: zero hits.

- [ ] **Step 4: Audit — no remaining `return fig, ` in public plot functions**

```bash
grep -rn "return fig," src/publiplots/plot/
```

Expected: zero hits. (Internal helpers in ComplexHeatmap that construct the figure may still legitimately return things — but after Task 5 they should all return dicts or axes.)

- [ ] **Step 5: No commit — this is just verification.**

---

## Task 13: Push branch and open PR

**Files:** none (git only).

- [ ] **Step 1: Push**

```bash
git push -u origin feat/axes-only-returns
```

- [ ] **Step 2: Open PR**

Run:

```bash
gh pr create --title "feat(breaking)!: plot functions return ax only (no fig)" --body "$(cat <<'EOF'
## BREAKING CHANGE

Every publiplots plot function now returns only the axes it drew into. ``pp.subplots`` is unchanged.

| Function | Before | After |
|---|---|---|
| ``barplot``, ``boxplot``, ``violinplot``, ``raincloudplot`` | ``(fig, ax)`` | ``ax`` |
| ``scatterplot``, ``stripplot``, ``swarmplot``, ``pointplot`` | ``(fig, ax)`` | ``ax`` |
| ``heatmap`` (categorical + dot) | ``(fig, ax)`` | ``ax`` |
| ``venn`` | ``(fig, ax)`` | ``ax`` |
| ``dendrogram`` | ``(fig, ax)`` | ``ax`` |
| ``upsetplot`` | ``(fig, (ax_i, ax_m, ax_s))`` | ``{'intersections', 'matrix', 'sets'}`` |
| ``complex_heatmap().build()`` | ``(fig, axes_dict)`` | ``axes_dict`` |

## Migration

\`\`\`python
# before
fig, ax = pp.barplot(data=df, x="x", y="y")
fig.savefig("out.png")

# after
ax = pp.barplot(data=df, x="x", y="y")
ax.get_figure().savefig("out.png")    # or plt.savefig("out.png") / pp.savefig("out.png")
\`\`\`

For ``upsetplot``, switch from positional to named axes:

\`\`\`python
axes = pp.upsetplot(sets)
axes['intersections'].set_title(...)
\`\`\`

## Why

Every gallery example did ``fig, ax = pp.<plot>(...)`` and then ignored ``fig``. ``pp.savefig`` already doesn't take a figure argument (operates on the current figure). This aligns with seaborn's convention and removes a dead variable.

## Test plan
- [x] New ``tests/test_axes_only_returns.py`` — 40 parametrized contract tests
- [x] All existing tests updated to drop ``fig, `` prefix
- [x] Full suite: 275 passed
- [x] Gallery smoke: all 14 ``examples/plots/*.py`` render clean
- [x] No stray ``fig, ax = pp.<plot>`` remaining (audit grep returns zero)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-review findings

**Spec coverage check (against `docs/superpowers/specs/2026-05-02-axes-only-returns-design.md`):**
- 12 plot functions → Tasks 2 (4 plots), 3 (4 plots), 4 (heatmap), 5 (complex_heatmap), 6 (dendrogram, extra find), 7 (venn), 8 (upsetplot) ✓
- Contract tests → Task 1 ✓
- Gallery updates → Task 10 ✓
- Docstring updates → Task 11 ✓
- Final verification → Task 12 ✓

Note: `dendrogram` was NOT listed in the spec's 12 functions. Added to the plan (Task 6) because grep found it as a public plot function with the same `(fig, ax)` return. The spec needs a minor update; I'll flag that to the user but the plan covers it.

**Placeholder scan:** none.

**Type consistency:**
- Every `-> Axes:` annotation consistent.
- Every `-> Dict[str, Axes]:` (upsetplot) / `-> Dict[str, Union[Axes, List[Axes]]]:` (complex_heatmap) consistent with spec contract.
- `upsetplot` dict keys `{intersections, matrix, sets}` match between contract test, plan task, and spec.
- `complex_heatmap` dict keys `{main, top, bottom, left, right}` unchanged from current (existing structure preserved, just dropping the `fig,` wrapper).

No gaps.
