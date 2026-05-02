# PR C — drop `figsize=` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove `figsize=` from 10 "simple" publiplots plot functions. All figure creation routes through `pp.subplots()` so `SubplotsAutoLayout` is always installed. Remove the `figure.figsize` rcParam; patch the two composite consumers (UpSet, ComplexHeatmap) to use a hardcoded fallback.

**Architecture:** Mechanical 10-file migration + 2-line rcParam fix + gallery cleanup. One contract test file covers all 10 plots parametrically.

**Tech Stack:** Python 3.12, matplotlib, pytest. Builds on existing `pp.subplots` + `SubplotsAutoLayout` infrastructure.

**Worktree:** `/home/sagemaker-user/publiplots/.worktrees/drop-figsize` on branch `feat/drop-figsize`, based on `main` at `e05df32`. Baseline: 200 tests passing.

**Environment note:** some shells have a stale `VIRTUAL_ENV` env var — always prefix pytest with `unset VIRTUAL_ENV && .venv/bin/pytest ...`.

---

## File Structure

Files modified:
- `src/publiplots/plot/bar.py` — drop `figsize` signature + docstring + figure-creation block
- `src/publiplots/plot/box.py` — same
- `src/publiplots/plot/violin.py` — same
- `src/publiplots/plot/raincloud.py` — drop own `figsize` + the `figsize=figsize` forward to violin at line 210
- `src/publiplots/plot/scatter.py` — same as bar/box
- `src/publiplots/plot/strip.py` — same
- `src/publiplots/plot/swarm.py` — same
- `src/publiplots/plot/point.py` — same
- `src/publiplots/plot/heatmap.py` — two signatures (public + dot), plus line 779 forward; plus the internal `resolve_param("figure.figsize")` fallback in `ComplexHeatmap` (line 885 in 0.5.0)
- `src/publiplots/plot/upset/diagram.py` — patch the `resolve_param("figure.figsize")` fallback (line 271)
- `src/publiplots/themes/rcparams.py` — drop `"figure.figsize"` key at line 25
- `examples/plots/plot_08_raincloud_plots.py:85` — drop `figsize=(4, 7)` from horizontal raincloud
- `examples/plots/plot_13_configuration.py:30, 53, 79` — swap `figure.figsize` references for `subplots.axes_size`

Files created:
- `tests/test_drop_figsize.py` — parametrized contract test for the 10 migrated plots

---

## Task 1: Write the parametrized contract test (red baseline)

**Files:**
- Create: `tests/test_drop_figsize.py`

This test ships first so the migration work is driven by it. Initially most assertions will fail because the plots still accept `figsize=`.

- [ ] **Step 1: Write the test file**

Create `tests/test_drop_figsize.py` with this content:

```python
"""Contract tests for PR C — figsize removal across simple plot functions.

Each migrated plot function must:
  1. Raise TypeError if called with figsize=.
  2. Install SubplotsAutoLayout on its figure by default.
  3. Honor ax= passed from pp.subplots without creating a second figure.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pytest

import publiplots as pp


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# --- fixtures --------------------------------------------------------------

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


# Each call-spec returns a zero-arg callable that performs the plot with the
# supplied overrides. This lets us test every plot's signature identically.
def _spec(fn_name, *, df_fixture, kwargs):
    return (fn_name, df_fixture, kwargs)


# All 10 migrated plots with minimal valid kwargs.
MIGRATED = [
    _spec("barplot",       df_fixture="cat_df",    kwargs={"x": "cat", "y": "val"}),
    _spec("boxplot",       df_fixture="cat_df",    kwargs={"x": "cat", "y": "val"}),
    _spec("violinplot",    df_fixture="cat_df",    kwargs={"x": "cat", "y": "val"}),
    _spec("raincloudplot", df_fixture="cat_df",    kwargs={"x": "cat", "y": "val"}),
    _spec("scatterplot",   df_fixture="scatter_df", kwargs={"x": "x",   "y": "y"}),
    _spec("stripplot",     df_fixture="cat_df",    kwargs={"x": "cat", "y": "val"}),
    _spec("swarmplot",     df_fixture="cat_df",    kwargs={"x": "cat", "y": "val"}),
    _spec("pointplot",     df_fixture="cat_df",    kwargs={"x": "cat", "y": "val"}),
    _spec("heatmap",       df_fixture="matrix_df", kwargs={}),
    # Dot-heatmap path: needs x/y/value/size.
    _spec("heatmap",       df_fixture="cat_df",    kwargs={"x": "cat", "y": "g", "value": "val", "size": "val"}),
]


@pytest.mark.parametrize("fn_name,df_fixture,kwargs", MIGRATED, ids=[
    "barplot", "boxplot", "violinplot", "raincloudplot", "scatterplot",
    "stripplot", "swarmplot", "pointplot", "heatmap-categorical", "heatmap-dot",
])
def test_figsize_kwarg_is_rejected(fn_name, df_fixture, kwargs, request):
    df = request.getfixturevalue(df_fixture)
    fn = getattr(pp, fn_name)
    with pytest.raises(TypeError, match="figsize"):
        fn(data=df, **kwargs, figsize=(4, 3))


@pytest.mark.parametrize("fn_name,df_fixture,kwargs", MIGRATED, ids=[
    "barplot", "boxplot", "violinplot", "raincloudplot", "scatterplot",
    "stripplot", "swarmplot", "pointplot", "heatmap-categorical", "heatmap-dot",
])
def test_default_call_installs_auto_layout(fn_name, df_fixture, kwargs, request):
    df = request.getfixturevalue(df_fixture)
    fn = getattr(pp, fn_name)
    fig, _ = fn(data=df, **kwargs)
    assert hasattr(fig, "_publiplots_auto_layout"), \
        f"{fn_name}: figure has no SubplotsAutoLayout — did it take a non-pp.subplots path?"


@pytest.mark.parametrize("fn_name,df_fixture,kwargs", MIGRATED, ids=[
    "barplot", "boxplot", "violinplot", "raincloudplot", "scatterplot",
    "stripplot", "swarmplot", "pointplot", "heatmap-categorical", "heatmap-dot",
])
def test_ax_kwarg_reuses_existing_figure(fn_name, df_fixture, kwargs, request):
    df = request.getfixturevalue(df_fixture)
    fn = getattr(pp, fn_name)
    fig0, ax0 = pp.subplots(axes_size=(50, 30))
    fig1, ax1 = fn(data=df, **kwargs, ax=ax0)
    assert fig1 is fig0
    assert ax1 is ax0


def test_rcparam_figure_figsize_removed():
    """After PR C, 'figure.figsize' must not appear in pp.rcParams."""
    assert "figure.figsize" not in pp.rcParams
```

- [ ] **Step 2: Run to confirm most assertions fail**

Run: `unset VIRTUAL_ENV && .venv/bin/pytest tests/test_drop_figsize.py -v`

Expected outcome (baseline):
- `test_figsize_kwarg_is_rejected`: all 10 parametrizations FAIL (`figsize` is currently accepted).
- `test_default_call_installs_auto_layout`: probably passes (default path already uses `pp.subplots`).
- `test_ax_kwarg_reuses_existing_figure`: probably passes.
- `test_rcparam_figure_figsize_removed`: FAIL (rcParam still exists).

Do not fix anything yet — this is the red baseline. The subsequent tasks will turn each failure green.

- [ ] **Step 3: Commit the red baseline**

```bash
git add tests/test_drop_figsize.py
git commit -m "test(drop-figsize): add red-baseline contract tests"
```

---

## Task 2: Migrate bar, box, violin

**Files:**
- Modify: `src/publiplots/plot/bar.py`
- Modify: `src/publiplots/plot/box.py`
- Modify: `src/publiplots/plot/violin.py`

Three structurally identical edits. Each file currently has:

```python
# signature
    figsize: Optional[Tuple[float, float]] = None,
```

```python
# docstring
    figsize : tuple, optional
        ...description...
```

```python
# figure creation
    if ax is None:
        if figsize is not None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            from publiplots.layout.subplots import subplots as _pp_subplots
            fig, ax = _pp_subplots()
    else:
        fig = ax.get_figure()
```

- [ ] **Step 1: Edit `src/publiplots/plot/bar.py`**

1. Delete the line `    figsize: Optional[Tuple[float, float]] = None,` from the `barplot(...)` signature (line 44).
2. Delete the two-line docstring entry for `figsize` (currently lines 93–94, "figsize : tuple, default=(4, 4)" and its description).
3. Replace the figure-creation block with:

```python
    if ax is None:
        from publiplots.layout.subplots import subplots as _pp_subplots
        fig, ax = _pp_subplots()
    else:
        fig = ax.get_figure()
```

4. If `from typing import ... Tuple ...` was only used for the `figsize` annotation and has no other reference in the file, leave the import alone (harmless; other types may still use it). Do a quick grep: `grep -n "Tuple" src/publiplots/plot/bar.py`. If `Tuple` appears only in the import line now, remove it from the import; otherwise leave it.
5. If `import matplotlib.pyplot as plt` is now unused in the file (grep for `plt\.`), remove the import. Otherwise leave it.

- [ ] **Step 2: Edit `src/publiplots/plot/box.py`**

Same three-step edit as bar. Line numbers: signature at 44, docstring at 98–99, figure-creation block at ~153–161.

- [ ] **Step 3: Edit `src/publiplots/plot/violin.py`**

Same three-step edit. Line numbers: signature at 55, docstring at 130–131, figure-creation block at ~200–208.

- [ ] **Step 4: Run the contract tests for bar/box/violin**

Run: `unset VIRTUAL_ENV && .venv/bin/pytest tests/test_drop_figsize.py -v -k "barplot or boxplot or violinplot"`

Expected: 9 passed (3 tests × 3 plots) + any heatmap failures still present (they're not yet migrated). The `test_rcparam_figure_figsize_removed` test will still fail — that's Task 6.

- [ ] **Step 5: Run the PR A regression tests**

Run: `unset VIRTUAL_ENV && .venv/bin/pytest tests/test_box_legend_stash.py tests/test_violin_legend_stash.py -v`

Expected: all pass (no regressions in the legend stash behavior).

- [ ] **Step 6: Commit**

```bash
git add src/publiplots/plot/bar.py src/publiplots/plot/box.py src/publiplots/plot/violin.py
git commit -m "feat(bar,box,violin): drop figsize=, always route through pp.subplots"
```

---

## Task 3: Migrate raincloud (drops own figsize + forwarded figsize to violin)

**Files:**
- Modify: `src/publiplots/plot/raincloud.py`

Raincloud is special because it also forwards `figsize=figsize` to its inner `violinplot(...)` call at line 210. Both the signature + the forwarded kwarg must go.

- [ ] **Step 1: Edit `src/publiplots/plot/raincloud.py`**

1. Delete `    figsize: Optional[Tuple[float, float]] = None,` from the `raincloudplot(...)` signature (line 57).
2. Delete the docstring entry for `figsize` (line 127–128, "figsize : tuple, optional" + its description).
3. In the violin-forwarding block around line 210, delete the line `        figsize=figsize,`. Leave all other kwargs forwarded.

- [ ] **Step 2: Run raincloud contract tests + existing raincloud tests**

Run: `unset VIRTUAL_ENV && .venv/bin/pytest tests/test_drop_figsize.py tests/test_raincloud_legend_group.py -v -k "raincloudplot"`

Expected: 3 new contract tests pass; existing raincloud_legend_group tests still pass.

- [ ] **Step 3: Commit**

```bash
git add src/publiplots/plot/raincloud.py
git commit -m "feat(raincloud): drop figsize= and figsize forward to violinplot"
```

---

## Task 4: Migrate scatter, strip, swarm, point

**Files:**
- Modify: `src/publiplots/plot/scatter.py`
- Modify: `src/publiplots/plot/strip.py`
- Modify: `src/publiplots/plot/swarm.py`
- Modify: `src/publiplots/plot/point.py`

Four structurally identical edits, same as Task 2.

- [ ] **Step 1: Edit each file**

For each of scatter, strip, swarm, point:

1. Delete the `figsize: Optional[Tuple[float, float]] = None,` line from the signature.
2. Delete the `figsize : tuple, ...` docstring entry (two lines).
3. Collapse the figure-creation block to:

```python
    if ax is None:
        from publiplots.layout.subplots import subplots as _pp_subplots
        fig, ax = _pp_subplots()
    else:
        fig = ax.get_figure()
```

Reference line numbers (from current main):
- `scatter.py`: signature 50, docstring 112–113, figure-creation ~197–208
- `strip.py`: signature 49, docstring 98–99, figure-creation ~142–150
- `swarm.py`: signature 48, docstring 95–96, figure-creation ~139–147
- `point.py`: signature 61, docstring 149–150, figure-creation ~206–214

- [ ] **Step 2: Run contract tests for these four**

Run: `unset VIRTUAL_ENV && .venv/bin/pytest tests/test_drop_figsize.py -v -k "scatterplot or stripplot or swarmplot or pointplot"`

Expected: 12 passed (3 tests × 4 plots).

- [ ] **Step 3: Run legend-stash regression tests**

Run: `unset VIRTUAL_ENV && .venv/bin/pytest tests/test_scatter_legend_stash.py -v`

Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add src/publiplots/plot/scatter.py src/publiplots/plot/strip.py src/publiplots/plot/swarm.py src/publiplots/plot/point.py
git commit -m "feat(scatter,strip,swarm,point): drop figsize= kwarg"
```

---

## Task 5: Migrate heatmap (categorical + dot public paths)

**Files:**
- Modify: `src/publiplots/plot/heatmap.py`

heatmap.py has two public entry points — the categorical heatmap `heatmap(...)` (line 59) and the dot heatmap `dot_heatmap(...)` / similar public function (line 631). Both drop `figsize`. The dot entrypoint also currently does `figsize=figsize` forward at line 779 — drop that too.

**Do NOT touch** `ComplexHeatmap` (line 831 class, line 885 uses `resolve_param("figure.figsize")`) — that's Task 6.

- [ ] **Step 1: Edit `heatmap.py`**

1. Delete `figsize: Optional[Tuple[float, float]] = None,` from the public `heatmap(...)` signature (line 59).
2. Delete the corresponding `figsize : tuple, optional` docstring entry (line 126–127).
3. Collapse the figure-creation block (around line 187–195) to:

```python
    if ax is None:
        from publiplots.layout.subplots import subplots as _pp_subplots
        fig, ax = _pp_subplots()
    else:
        fig = ax.get_figure()
```

4. Delete `figsize: Optional[Tuple[float, float]] = None,` from the dot-heatmap public signature at line 631.
5. Delete its docstring entry at line 694–695.
6. At line 779, delete the line `        figsize=figsize,` (the forward inside the dot-heatmap wrapper).
7. Do NOT touch line 885 (`ComplexHeatmap` reads `resolve_param("figure.figsize")`) — Task 6 handles that.

- [ ] **Step 2: Run contract tests**

Run: `unset VIRTUAL_ENV && .venv/bin/pytest tests/test_drop_figsize.py -v -k "heatmap"`

Expected: 6 passed (3 tests × 2 heatmap parametrizations).

- [ ] **Step 3: Run heatmap stash regression**

Run: `unset VIRTUAL_ENV && .venv/bin/pytest tests/test_heatmap_legend_stash.py -v`

Expected: all pass.

- [ ] **Step 4: Sanity-check ComplexHeatmap still works**

Run `unset VIRTUAL_ENV && .venv/bin/python examples/plots/plot_11_heatmap.py >/dev/null 2>&1 && echo OK` — should print `OK`. If it fails, the edit accidentally affected `ComplexHeatmap`.

- [ ] **Step 5: Commit**

```bash
git add src/publiplots/plot/heatmap.py
git commit -m "feat(heatmap): drop figsize= from categorical + dot public entries"
```

---

## Task 6: Remove the `figure.figsize` rcParam + patch composite fallbacks

**Files:**
- Modify: `src/publiplots/themes/rcparams.py`
- Modify: `src/publiplots/plot/upset/diagram.py`
- Modify: `src/publiplots/plot/heatmap.py` (ComplexHeatmap only — line 885 in 0.5.0)

Dropping the rcParam strands two consumers: UpSet and ComplexHeatmap. Replace both with a hardcoded `(4.0, 3.0)` default (matplotlib's historical default, preserves geometry).

- [ ] **Step 1: Edit `src/publiplots/themes/rcparams.py`**

Find the dict entry `"figure.figsize": [3, 1.8],` around line 25. Delete that entire line.

Also check for any docstring examples in the same file that reference `"figure.figsize"` — around line 242 and 246 there are example snippets. If they're in a `>>>` docstring example, replace them with `"subplots.axes_size"`. If they're in prose, change the wording similarly.

- [ ] **Step 2: Edit `src/publiplots/plot/upset/diagram.py`**

At line 271, change:

```python
    default_width, default_height = resolve_param("figure.figsize")
```

to:

```python
    # Historical matplotlib default; kept verbatim to preserve existing
    # UpSet geometry. Migrating UpSet to pp.subplots is deferred.
    default_width, default_height = (4.0, 3.0)
```

Verify `resolve_param` is still used elsewhere in the file before removing the import. Grep: `grep -n "resolve_param" src/publiplots/plot/upset/diagram.py`. If this was the only usage, remove the import.

- [ ] **Step 3: Edit `src/publiplots/plot/heatmap.py` (ComplexHeatmap only)**

At line 885 (in 0.5.0 source; confirm with a grep `grep -n "resolve_param.*figure.figsize" src/publiplots/plot/heatmap.py` first), change:

```python
        self._figsize = figsize or resolve_param("figure.figsize")
```

to:

```python
        # Historical matplotlib default. ComplexHeatmap still accepts
        # figsize=; migrating it to pp.subplots is deferred.
        self._figsize = figsize or (4.0, 3.0)
```

- [ ] **Step 4: Run the rcParam test**

Run: `unset VIRTUAL_ENV && .venv/bin/pytest tests/test_drop_figsize.py::test_rcparam_figure_figsize_removed -v`

Expected: PASS.

- [ ] **Step 5: Run the full suite**

Run: `unset VIRTUAL_ENV && .venv/bin/pytest -q`

Expected: 200 + N new passing (N = contract tests from Task 1). The exact final count matches the baseline plus the new contract tests.

- [ ] **Step 6: Run full gallery smoke**

Run:

```bash
unset VIRTUAL_ENV
for f in examples/plots/plot_*.py; do
  .venv/bin/python "$f" >/dev/null 2>&1 && echo "$(basename $f): OK" || echo "$(basename $f): FAIL"
done
```

Expected: all 14 OK. `plot_08` (horizontal raincloud `figsize=`) is expected to FAIL here — Task 7 fixes it. `plot_11` and `plot_09` should still pass because they use out-of-scope composites that retain `figsize=`.

- [ ] **Step 7: Commit**

```bash
git add src/publiplots/themes/rcparams.py src/publiplots/plot/upset/diagram.py src/publiplots/plot/heatmap.py
git commit -m "chore(rcparams): remove figure.figsize; hardcode (4,3) in composites"
```

---

## Task 7: Gallery updates

**Files:**
- Modify: `examples/plots/plot_08_raincloud_plots.py`
- Modify: `examples/plots/plot_13_configuration.py`

- [ ] **Step 1: Edit `examples/plots/plot_08_raincloud_plots.py`**

Find the horizontal raincloud block at lines 74–95 (search for `cloud_side="left"`). Remove the `    figsize=(4, 7),` line. The auto-layout plus the block's existing `box_offset=0.1, rain_offset=0.2` keeps the plot readable. If you want to preserve the wide look, change the block to compose:

```python
fig, ax = pp.subplots(axes_size=(70, 70))
pp.raincloudplot(
    data=raincloud_data,
    x='measurement',
    y='time',
    hue='group',
    cloud_side="left",
    title='Horizontal Raincloud Plot',
    xlabel='Measurement',
    ylabel='Time',
    cloud_alpha=0.6,
    palette="RdGyBu_r",
    box_offset=0.1,
    rain_offset=0.2,
    rain_kws=dict(
        linewidth=1,
        alpha=0.5,
        jitter=False,
        marker="x",
    ),
    ax=ax,
)
plt.show()
```

(Pick the simpler "just drop figsize=" route first; if auto-layout looks cramped, switch to the compose form.)

- [ ] **Step 2: Edit `examples/plots/plot_13_configuration.py`**

Three references need updating:

- Line 30: `print(f"  Figure size: {pp.rcParams['figure.figsize']}")` → `print(f"  Axes size (mm): {pp.rcParams['subplots.axes_size']}")`
- Line 53: same change as line 30.
- Line 79: `pp.rcParams['figure.figsize'] = (8, 5)  # Wider figures` → `pp.rcParams['subplots.axes_size'] = (80, 50)  # Wider axes (mm)`

If there's an accompanying comment above that explains the knob, update it to reflect mm-based sizing.

- [ ] **Step 3: Re-run full gallery**

Run:

```bash
unset VIRTUAL_ENV
for f in examples/plots/plot_*.py; do
  .venv/bin/python "$f" >/dev/null 2>&1 && echo "$(basename $f): OK" || echo "$(basename $f): FAIL"
done
```

Expected: all 14 OK.

- [ ] **Step 4: Run the full suite one more time**

Run: `unset VIRTUAL_ENV && .venv/bin/pytest -q`

Expected: 200 + N passed.

- [ ] **Step 5: Commit**

```bash
git add examples/plots/plot_08_raincloud_plots.py examples/plots/plot_13_configuration.py
git commit -m "docs(examples): drop figsize= from plot_08; use subplots.axes_size in plot_13"
```

---

## Task 8: Push and open PR

**Files:** none (git operations only).

- [ ] **Step 1: Push**

Run: `git push -u origin feat/drop-figsize`

- [ ] **Step 2: Open PR**

Run:

```bash
gh pr create --title "feat(breaking): remove figsize= from simple plot functions" --body "$(cat <<'EOF'
## BREAKING CHANGE

``figsize=`` is removed from: ``barplot``, ``boxplot``, ``violinplot``, ``raincloudplot``, ``scatterplot``, ``stripplot``, ``swarmplot``, ``pointplot``, ``heatmap`` (categorical + dot entries). The ``figure.figsize`` rcParam is also removed.

To customize axes dimensions, compose with ``pp.subplots``:

\`\`\`python
# before
pp.barplot(data=df, x="x", y="y", figsize=(4, 3))

# after
fig, ax = pp.subplots(axes_size=(80, 50))  # mm, not inches
pp.barplot(data=df, x="x", y="y", ax=ax)
\`\`\`

Or omit for auto-layout.

## Why

When a plot function received ``figsize=``, it took a ``plt.subplots(figsize=...)`` branch that **did not install** ``SubplotsAutoLayout``, causing legends, colorbar titles, and xlabel overflow to be cropped on ``savefig``. See the 0.5.0 known-issue entry re: the horizontal raincloud example in ``plot_08``.

``pp.subplots`` is now the single source of truth for figure+axes dimensions.

## Still accepts ``figsize=`` (deferred)

- ``ComplexHeatmap`` — composite layout with dynamic sizing; port deferred.
- ``venn`` — ``figsize`` has aspect-ratio semantics; keep.
- ``upset`` — composite; port deferred.

These three retain their ``figsize=`` kwarg but no longer read the dropped rcParam — they fall back to a hardcoded ``(4, 3)`` default.

## Test plan
- [x] New ``tests/test_drop_figsize.py`` — parametrized contract tests × 10 plots × 3 assertions each + rcParam test
- [x] Full suite: <N> passed
- [x] Gallery smoke: all 14 render clean

Closes the PR C milestone queued since 0.5.0.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-review

**Spec coverage check (against `docs/superpowers/specs/2026-05-01-drop-figsize-design.md`):**
- 10 migrated plots → Tasks 2, 3, 4, 5 ✓
- rcParam drop → Task 6 ✓
- Composite rcParam fallbacks (UpSet + ComplexHeatmap) → Task 6 ✓
- Gallery fixes (plot_08, plot_13) → Task 7 ✓
- Contract test file → Task 1 ✓
- Out-of-scope (Venn, ComplexHeatmap's `figsize=` kwarg, UpSet's `figsize=` kwarg) → preserved by not touching those signatures ✓

**Placeholder scan:** none.

**Type consistency:** all 10 migrated functions get identical figure-creation collapse. Contract test uses the same three-assertion shape per plot. `TypeError: unexpected keyword argument` is Python's natural error for a removed kwarg — no custom error handling needed.
