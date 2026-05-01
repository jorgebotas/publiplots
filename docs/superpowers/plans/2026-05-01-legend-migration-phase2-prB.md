# Legend Migration Phase 2 — PR B (bar + heatmap) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the legend migration by bringing `barplot` and `heatmap` onto the `LegendEntry` stash pattern, add a new `"hatch"` kind to support bar's secondary split dimension, create a shared `_stash_legend` utility, and retrofit `box` + `violin` to use it.

**Architecture:** Introduce `src/publiplots/utils/plot_legend.py` with two primitives: `stash_hue_legend` (single-kind hue helper used by box, violin, and heatmap's categorical path) and `render_entries` (per-axis render of stashed entries, used by any plot that does multi-kind stashing). Bar uses `render_entries` after emitting its four-case hue/hatch stash pattern. Heatmap's dot path stashes one hue entry (ScalarMappable → continuous hue detection) and one size entry.

**Tech Stack:** Python 3.12, matplotlib, pandas, seaborn, pytest. Reuses `src/publiplots/utils/legend_entries.py` (extended with `"hatch"` kind) and `src/publiplots/utils/legend.py`.

**Worktree:** `/home/sagemaker-user/publiplots/.worktrees/legend-migration-b` on branch `feat/legend-migration-b`, rebased onto `main` at `f621118` (PR #83 merged).

**Baseline:** 176 tests passing.

**Environment note:** the project's `.venv` is inside the worktree. Some shells have a stale `VIRTUAL_ENV` env var — always prefix pytest with `unset VIRTUAL_ENV && .venv/bin/pytest ...`.

---

## File Structure

Files created:
- `src/publiplots/utils/plot_legend.py` — `stash_hue_legend` + `render_entries` + `create_continuous_hue_mappable` utility
- `tests/test_plot_legend.py` — unit tests for the shared helpers
- `tests/test_bar_legend_stash.py` — bar migration tests (all 4 cases + per-kind flag)
- `tests/test_heatmap_legend_stash.py` — heatmap migration tests (categorical + dot)

Files modified:
- `src/publiplots/utils/legend_entries.py` — add `"hatch"` to `_LEGEND_KINDS`
- `src/publiplots/plot/box.py` — delegate `_stash_legend` to `stash_hue_legend`
- `src/publiplots/plot/violin.py` — delegate `_stash_legend` to `stash_hue_legend`
- `src/publiplots/plot/bar.py` — rewrite `_legend` helper (lines 455–549) to stash instead of render directly
- `src/publiplots/plot/heatmap.py` — replace both legend blocks (lines 340–361 and 497–523) with stash calls
- `tests/test_legend_entries.py` — extend `_LEGEND_KINDS` test(s) to include `"hatch"`
- `examples/plots/plot_01_bar_plots.py` — new panel demonstrating bar + `pp.legend_group`

---

## Task 1: Add `"hatch"` kind to `_LEGEND_KINDS`

**Files:**
- Modify: `src/publiplots/utils/legend_entries.py:13`
- Test: `tests/test_legend_entries.py` (extend an existing test)

- [ ] **Step 1: Write a failing test**

Append to `tests/test_legend_entries.py`:

```python
def test_legend_kinds_includes_hatch():
    """hatch is a supported legend kind for bar's secondary split dimension."""
    from publiplots.utils.legend_entries import _LEGEND_KINDS, resolve_legend_flags
    assert "hatch" in _LEGEND_KINDS
    flags = resolve_legend_flags({"hatch": False})
    assert flags["hatch"] is False
    assert flags["hue"] is True  # other kinds default to True
```

- [ ] **Step 2: Confirm it fails**

Run: `unset VIRTUAL_ENV && .venv/bin/pytest tests/test_legend_entries.py::test_legend_kinds_includes_hatch -v`
Expected: FAIL on the `"hatch" in _LEGEND_KINDS` assertion.

- [ ] **Step 3: Add `"hatch"` to the tuple**

In `src/publiplots/utils/legend_entries.py:13`, replace:

```python
_LEGEND_KINDS = ("hue", "size", "style", "marker")
```

with:

```python
_LEGEND_KINDS = ("hue", "size", "style", "marker", "hatch")
```

- [ ] **Step 4: Confirm the test passes**

Run: `unset VIRTUAL_ENV && .venv/bin/pytest tests/test_legend_entries.py -v`
Expected: all tests pass including the new one.

- [ ] **Step 5: Commit**

```bash
git add src/publiplots/utils/legend_entries.py tests/test_legend_entries.py
git commit -m "feat(legend): add hatch kind to LegendEntry kinds"
```

---

## Task 2: Create shared `plot_legend.py` utility

**Files:**
- Create: `src/publiplots/utils/plot_legend.py`
- Test: `tests/test_plot_legend.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_plot_legend.py`:

```python
"""Tests for the shared plot_legend helpers."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pytest

import publiplots as pp
from publiplots.utils.legend_entries import get_entries, LegendEntry, stash_entry
from publiplots.utils.plot_legend import stash_hue_legend, render_entries


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_stash_hue_legend_stashes_one_entry():
    """With hue + dict palette, stash_hue_legend produces one LegendEntry."""
    fig, ax = plt.subplots()
    palette = {"A": "#ff0000", "B": "#00ff00"}
    stash_hue_legend(
        ax,
        hue="group",
        palette=palette,
        edgecolor=None,
        alpha=0.5,
        linewidth=1.0,
        legend=True,
        legend_kws=None,
    )
    entries = get_entries(ax)
    assert len(entries) == 1
    assert (entries[0].name, entries[0].kind) == ("group", "hue")
    assert entries[0].labels == ("A", "B")


def test_stash_hue_legend_legend_false_short_circuits():
    fig, ax = plt.subplots()
    stash_hue_legend(
        ax,
        hue="group",
        palette={"A": "#ff0000"},
        edgecolor=None,
        alpha=0.5,
        linewidth=1.0,
        legend=False,
        legend_kws=None,
    )
    assert get_entries(ax) == []


def test_stash_hue_legend_no_hue_short_circuits():
    fig, ax = plt.subplots()
    stash_hue_legend(
        ax,
        hue=None,
        palette={"A": "#ff0000"},
        edgecolor=None,
        alpha=0.5,
        linewidth=1.0,
        legend=True,
        legend_kws=None,
    )
    assert get_entries(ax) == []


def test_stash_hue_legend_non_dict_palette_short_circuits():
    """Non-dict palette (e.g., a seaborn string name) short-circuits; caller
    should resolve palette to a dict before invoking."""
    fig, ax = plt.subplots()
    stash_hue_legend(
        ax,
        hue="group",
        palette="pastel",
        edgecolor=None,
        alpha=0.5,
        linewidth=1.0,
        legend=True,
        legend_kws=None,
    )
    assert get_entries(ax) == []


def test_stash_hue_legend_dict_flag_suppresses():
    fig, ax = plt.subplots()
    stash_hue_legend(
        ax,
        hue="group",
        palette={"A": "#ff0000"},
        edgecolor=None,
        alpha=0.5,
        linewidth=1.0,
        legend={"hue": False},
        legend_kws=None,
    )
    assert get_entries(ax) == []


def test_stash_hue_legend_respects_hue_label_kw():
    """legend_kws={'hue_label': 'Custom'} overrides the stashed name."""
    fig, ax = plt.subplots()
    stash_hue_legend(
        ax,
        hue="group",
        palette={"A": "#ff0000"},
        edgecolor=None,
        alpha=0.5,
        linewidth=1.0,
        legend=True,
        legend_kws={"hue_label": "Custom"},
    )
    entries = get_entries(ax)
    assert entries[0].name == "Custom"


def test_render_entries_renders_stashed_entries_per_axis():
    """After render_entries, a Legend artist is attached to ax."""
    from matplotlib.legend import Legend
    from publiplots.utils import create_legend_handles
    fig, ax = plt.subplots()
    handles = create_legend_handles(
        labels=["A", "B"], colors=["#ff0000", "#00ff00"],
        alpha=0.5, linewidth=1.0,
    )
    stash_entry(
        ax,
        LegendEntry.build(name="group", kind="hue", handles=handles, labels=("A", "B")),
    )
    render_entries(ax, flags={"hue": True, "size": True, "style": True, "marker": True, "hatch": True})
    per_axis_legends = [c for c in ax.get_children() if isinstance(c, Legend)]
    assert len(per_axis_legends) == 1


def test_render_entries_skips_entries_claimed_by_group():
    """When a legend_group claims an entry, render_entries does not attach a per-axis Legend."""
    from matplotlib.legend import Legend
    df = pd.DataFrame({
        "x": np.arange(10),
        "y": np.arange(10),
        "g": ["A"] * 5 + ["B"] * 5,
    })
    fig, axes = pp.subplots(1, 2, axes_size=(50, 40))
    pp.legend_group(anchor=axes[-1])
    from publiplots.utils import create_legend_handles
    handles = create_legend_handles(
        labels=["A", "B"], colors=["#ff0000", "#00ff00"],
        alpha=0.5, linewidth=1.0,
    )
    stash_entry(
        axes[0],
        LegendEntry.build(name="g", kind="hue", handles=handles, labels=("A", "B")),
    )
    render_entries(axes[0], flags={"hue": True, "size": True, "style": True, "marker": True, "hatch": True})
    per_axis_legends = [c for c in axes[0].get_children() if isinstance(c, Legend)]
    assert per_axis_legends == []


def test_render_entries_renders_continuous_hue_as_colorbar():
    """A ScalarMappable handle with empty labels routes through add_colorbar."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    fig, ax = plt.subplots()
    mappable = ScalarMappable(norm=Normalize(0, 10), cmap="viridis")
    stash_entry(
        ax,
        LegendEntry.build(name="score", kind="hue", handles=[mappable], labels=[]),
    )
    render_entries(ax, flags={"hue": True, "size": True, "style": True, "marker": True, "hatch": True})
    # We don't assert on the resulting artist type (publiplots colorbar vs Legend),
    # just that it doesn't crash and that no Legend was added (colorbars are separate axes).
    from matplotlib.legend import Legend
    per_axis_legends = [c for c in ax.get_children() if isinstance(c, Legend)]
    assert per_axis_legends == []
```

- [ ] **Step 2: Confirm they fail**

Run: `unset VIRTUAL_ENV && .venv/bin/pytest tests/test_plot_legend.py -v`
Expected: all FAIL with ImportError (module doesn't exist yet).

- [ ] **Step 3: Create the module**

Create `src/publiplots/utils/plot_legend.py` with exactly this content:

```python
"""Shared helpers for plot functions that stash + render LegendEntry objects.

Two primitives:
- ``stash_hue_legend``: one-shot hue stash used by plots with a single
  categorical hue dimension (box, violin, heatmap categorical path).
- ``render_entries``: per-axis render of any already-stashed entries that
  aren't claimed by a figure-level ``pp.legend_group``. Used by plots with
  multi-kind stashing (scatter-style: hue + size + style, or bar: hue + hatch).

Plots that do their own kind-specific stashing should call ``render_entries``
directly after stashing.
"""

from typing import Dict, List, Optional, Union

from matplotlib.axes import Axes

from publiplots.utils import create_legend_handles
from publiplots.utils.legend import legend as legend_fn
from publiplots.utils.legend_entries import (
    LegendEntry,
    stash_entry,
    get_entries,
    resolve_legend_flags,
    entry_is_in_group,
    is_continuous_hue,
)


def stash_hue_legend(
    ax: Axes,
    *,
    hue: Optional[str],
    palette: Optional[Union[str, Dict, List]],
    edgecolor: Optional[str],
    alpha: Optional[float],
    linewidth: Optional[float],
    legend: Union[bool, Dict],
    legend_kws: Optional[Dict],
) -> None:
    """Stash one hue LegendEntry and render it per-axis if not in a group.

    Short-circuits (no stash, no render) when any of these hold:
      - ``legend is False``
      - ``hue is None``
      - ``palette`` is not a dict (caller should resolve it first)
    """
    if legend is False or hue is None or not isinstance(palette, dict):
        return

    flags = resolve_legend_flags(legend)
    legend_kws = dict(legend_kws or {})
    hue_label = legend_kws.pop("hue_label", hue)

    if flags["hue"]:
        labels = list(palette.keys())
        handles = create_legend_handles(
            labels=labels,
            colors=list(palette.values()),
            edgecolors=[edgecolor] * len(palette) if edgecolor else None,
            alpha=alpha,
            linewidth=linewidth,
        )
        stash_entry(
            ax,
            LegendEntry.build(
                name=hue_label,
                kind="hue",
                handles=handles,
                labels=labels,
            ),
        )

    render_entries(ax, flags=flags)


def render_entries(ax: Axes, *, flags: Dict[str, bool]) -> None:
    """Render all stashed entries on ``ax`` not claimed by a figure-level group.

    For each stashed entry whose kind flag is True and that isn't claimed by
    the figure's legend_group, the handle is routed through the publiplots
    legend builder — continuous-hue (ScalarMappable) handles become a
    colorbar, everything else becomes a standard legend entry.

    If nothing remains to render, no builder is instantiated.
    """
    fig = ax.get_figure()
    to_render = [
        e for e in get_entries(ax)
        if flags[e.kind] and not entry_is_in_group(fig, e)
    ]
    if not to_render:
        return
    builder = legend_fn(ax=ax, auto=False)
    for entry in to_render:
        if entry.kind == "hue" and is_continuous_hue(entry.handles):
            builder.add_colorbar(
                mappable=entry.handles[0],
                label=entry.name,
            )
        else:
            builder.add_legend(
                handles=list(entry.handles),
                label=entry.name,
            )
```

- [ ] **Step 4: Confirm the new tests pass**

Run: `unset VIRTUAL_ENV && .venv/bin/pytest tests/test_plot_legend.py -v`
Expected: 9 passed.

- [ ] **Step 5: Full suite**

Run: `unset VIRTUAL_ENV && .venv/bin/pytest -q`
Expected: 176 existing + 9 new + 1 from Task 1 = 186 passed.

- [ ] **Step 6: Commit**

```bash
git add src/publiplots/utils/plot_legend.py tests/test_plot_legend.py
git commit -m "feat(legend): add shared stash_hue_legend + render_entries helpers"
```

---

## Task 3: Retrofit box + violin to use the shared helper

**Files:**
- Modify: `src/publiplots/plot/box.py` (the `_stash_legend` helper at the bottom of the file)
- Modify: `src/publiplots/plot/violin.py` (same helper location)

This is a pure refactor — behavior must not change. Existing tests from PR A (`test_box_legend_stash.py`, `test_violin_legend_stash.py`, `test_raincloud_legend_group.py`) must continue to pass without modification.

- [ ] **Step 1: Baseline run**

Run: `unset VIRTUAL_ENV && .venv/bin/pytest tests/test_box_legend_stash.py tests/test_violin_legend_stash.py tests/test_raincloud_legend_group.py -v`
Expected: 12 passed.

- [ ] **Step 2: Update box.py**

In `src/publiplots/plot/box.py`:

1. Find the imports section (around lines 22–29). Currently includes `from publiplots.utils.legend import legend as legend_fn` and the five-item `from publiplots.utils.legend_entries import (...)`. Replace these two imports with a single line:

```python
from publiplots.utils.plot_legend import stash_hue_legend
```

Remove the `from publiplots.utils.legend import legend as legend_fn` line and the `from publiplots.utils.legend_entries import (LegendEntry, stash_entry, get_entries, resolve_legend_flags, entry_is_in_group)` block. Keep `from publiplots.utils import is_categorical, create_legend_handles` unless `create_legend_handles` is now unused — if unused, drop it too. (To check: grep box.py for `create_legend_handles` — if the only remaining usage was inside `_stash_legend`, you can drop it.)

2. Replace the entire `_stash_legend(...)` helper at the bottom of the file with:

```python
def _stash_legend(
        ax: Axes,
        hue: Optional[str],
        palette: Optional[Union[str, Dict, List]],
        edgecolor: Optional[str],
        alpha: Optional[float],
        linewidth: Optional[float],
        legend: Union[bool, Dict],
        legend_kws: Optional[Dict],
    ) -> None:
    """Delegate to the shared hue-legend helper."""
    stash_hue_legend(
        ax,
        hue=hue,
        palette=palette,
        edgecolor=edgecolor,
        alpha=alpha,
        linewidth=linewidth,
        legend=legend,
        legend_kws=legend_kws,
    )
```

(Leave the existing call site inside `boxplot(...)` unchanged — it still calls `_stash_legend(...)` with the same kwargs.)

- [ ] **Step 3: Update violin.py**

Apply the same two edits to `src/publiplots/plot/violin.py`:
- Replace the `legend_fn` + `legend_entries` imports with `from publiplots.utils.plot_legend import stash_hue_legend`.
- Replace the `_stash_legend(...)` body with the same delegate call shown in Step 2.

- [ ] **Step 4: Confirm all legend tests still pass**

Run: `unset VIRTUAL_ENV && .venv/bin/pytest tests/test_box_legend_stash.py tests/test_violin_legend_stash.py tests/test_raincloud_legend_group.py tests/test_plot_legend.py -v`
Expected: 21 passed.

- [ ] **Step 5: Full suite**

Run: `unset VIRTUAL_ENV && .venv/bin/pytest -q`
Expected: 186 passed (same count as Task 2 — no new tests, refactor only).

- [ ] **Step 6: Commit**

```bash
git add src/publiplots/plot/box.py src/publiplots/plot/violin.py
git commit -m "refactor(box,violin): delegate _stash_legend to shared helper"
```

---

## Task 4: Migrate barplot to stash pattern

**Files:**
- Modify: `src/publiplots/plot/bar.py` (imports + `_legend` function at lines 455–549 + signature)
- Test: `tests/test_bar_legend_stash.py`

Bar has four legend-dispatch cases. The migration preserves the case structure exactly and only swaps `builder.add_legend(...)` calls for `stash_entry(...)` calls, with a single `render_entries(ax, flags=flags)` call at the end.

**Case mapping (entry.kind assignment):**

| Case | Condition | Stashed entries |
|---|---|---|
| combined | `hue == hatch` | 1 × `kind="hue"`, `name=hue_label` (handles include hatch pattern) |
| hatch-only | `hue == categorical_axis` | 1 × `kind="hatch"`, `name=hatch_label` |
| hue-only | `hatch == categorical_axis` | 1 × `kind="hue"`, `name=hue_label` |
| double-split | else | 1 × `kind="hue"` + 1 × `kind="hatch"` |

- [ ] **Step 1: Write failing tests**

Create `tests/test_bar_legend_stash.py`:

```python
"""Tests for barplot legend stashing via LegendEntry."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pytest

import publiplots as pp
from publiplots.utils.legend_entries import get_entries


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _bar_df(seed=0):
    rng = np.random.default_rng(seed)
    n = 60
    return pd.DataFrame({
        "cond": rng.choice(["A", "B", "C"], size=n),
        "treat": rng.choice(["ctrl", "trt"], size=n),
        "value": rng.normal(size=n),
    })


def test_bar_hue_only_stashes_one_hue_entry():
    """hatch == categorical_axis → only a hue entry."""
    df = _bar_df()
    fig, ax = pp.barplot(
        data=df, x="cond", y="value",
        hue="treat", hatch="cond",
        palette={"ctrl": "#ff0000", "trt": "#00ff00"},
    )
    names_kinds = [(e.name, e.kind) for e in get_entries(ax)]
    assert ("treat", "hue") in names_kinds
    assert not any(k == "hatch" for _, k in names_kinds)


def test_bar_hatch_only_stashes_one_hatch_entry():
    """hue == categorical_axis → only a hatch entry."""
    df = _bar_df()
    fig, ax = pp.barplot(
        data=df, x="cond", y="value",
        hue="cond", hatch="treat",
        palette={"A": "#ff0000", "B": "#00ff00", "C": "#0000ff"},
    )
    names_kinds = [(e.name, e.kind) for e in get_entries(ax)]
    assert ("treat", "hatch") in names_kinds
    assert not any(k == "hue" for _, k in names_kinds)


def test_bar_combined_stashes_one_hue_entry():
    """hue == hatch → one combined entry under kind=hue."""
    df = _bar_df()
    fig, ax = pp.barplot(
        data=df, x="cond", y="value",
        hue="cond", hatch="cond",
        palette={"A": "#ff0000", "B": "#00ff00", "C": "#0000ff"},
    )
    entries = get_entries(ax)
    names_kinds = [(e.name, e.kind) for e in entries]
    assert ("cond", "hue") in names_kinds
    assert len(entries) == 1


def test_bar_double_split_stashes_hue_and_hatch():
    """hue != hatch and neither == categorical → two entries."""
    df = _bar_df()
    fig, ax = pp.barplot(
        data=df, x="cond", y="value",
        hue="treat", hatch="treat",  # this is combined; we need a real double-split
    )
    # Build a real double-split setup
    fig, ax = pp.subplots(axes_size=(60, 40))
    # Add a third splitting variable
    df["role"] = np.tile(["lead", "sub"], len(df) // 2 + 1)[: len(df)]
    pp.barplot(
        data=df, x="cond", y="value",
        hue="treat", hatch="role",
        palette={"ctrl": "#ff0000", "trt": "#00ff00"},
        ax=ax,
    )
    names_kinds = [(e.name, e.kind) for e in get_entries(ax)]
    assert ("treat", "hue") in names_kinds
    assert ("role", "hatch") in names_kinds


def test_bar_legend_false_stashes_nothing():
    df = _bar_df()
    fig, ax = pp.barplot(
        data=df, x="cond", y="value",
        hue="treat", hatch="cond",
        palette={"ctrl": "#ff0000", "trt": "#00ff00"},
        legend=False,
    )
    assert get_entries(ax) == []


def test_bar_legend_dict_suppresses_hatch():
    """legend={'hatch': False} in a double-split scenario → only hue stashed."""
    df = _bar_df()
    df["role"] = np.tile(["lead", "sub"], len(df) // 2 + 1)[: len(df)]
    fig, ax = pp.subplots(axes_size=(60, 40))
    pp.barplot(
        data=df, x="cond", y="value",
        hue="treat", hatch="role",
        palette={"ctrl": "#ff0000", "trt": "#00ff00"},
        legend={"hatch": False},
        ax=ax,
    )
    names_kinds = [(e.name, e.kind) for e in get_entries(ax)]
    assert ("treat", "hue") in names_kinds
    assert not any(k == "hatch" for _, k in names_kinds)


def test_bar_in_group_suppresses_per_axis_render():
    from matplotlib.legend import Legend
    df = _bar_df()
    fig, axes = pp.subplots(1, 2, axes_size=(50, 40))
    pp.legend_group(anchor=axes[-1])
    pp.barplot(
        data=df, x="cond", y="value",
        hue="treat", hatch="cond",
        palette={"ctrl": "#ff0000", "trt": "#00ff00"},
        ax=axes[0],
    )
    fig.canvas.draw()
    per_axis_legends = [c for c in axes[0].get_children() if isinstance(c, Legend)]
    assert per_axis_legends == []
```

- [ ] **Step 2: Confirm most tests fail**

Run: `unset VIRTUAL_ENV && .venv/bin/pytest tests/test_bar_legend_stash.py -v`
Expected: the stash-shape tests FAIL (bar doesn't stash yet); `test_bar_legend_false_stashes_nothing` may already pass.

- [ ] **Step 3: Update bar imports and signature**

In `src/publiplots/plot/bar.py`:

1. Extend the imports. Find the existing `from publiplots.utils import is_categorical, as_categorical, create_legend_handles, legend` line (around line 18). Add immediately after it:

```python
from publiplots.utils.legend_entries import (
    LegendEntry,
    stash_entry,
    resolve_legend_flags,
)
from publiplots.utils.plot_legend import render_entries
```

2. In the `barplot(...)` signature at line 41, change `legend: bool = True,` to `legend: Union[bool, Dict] = True,`. Verify `Union` and `Dict` are already imported (they are — `bar.py:14` should have `from typing import Optional, List, Dict, Tuple, Union`).

- [ ] **Step 4: Rewrite `_legend` to stash instead of render**

Replace the `_legend` function body (lines 455–549) — keeping its signature and docstring intact — with this implementation. The four-case dispatch is preserved verbatim; only the `builder.add_legend(...)` calls become `stash_entry(...)` calls, and the final `render_entries(ax, flags=flags)` call replaces the builder's implicit render.

```python
def _legend(
        ax: Axes,
        hue: Optional[str],
        hatch: str,
        categorical_axis: str,
        alpha: float,
        linewidth: float,
        color: Optional[str],
        edgecolor: Optional[str],
        palette: Optional[Union[str, Dict, List]],
        hatch_map: Optional[Dict[str, str]],
        kwargs: Optional[Dict] = None,
        legend: Union[bool, Dict] = True,
    ) -> None:
    """
    Stash LegendEntry objects for bar plot, then render per-axis legends for
    entries not claimed by a figure-level group.

    Four cases (preserving the original dispatch):
      - hue == hatch           → 1 combined entry under kind="hue"
      - hue == categorical_axis → 1 hatch-only entry under kind="hatch"
      - hatch == categorical_axis → 1 hue-only entry under kind="hue"
      - double split           → 1 hue entry + 1 hatch entry
    """
    if legend is False:
        return

    flags = resolve_legend_flags(legend)
    kwargs = dict(kwargs or {})
    hue_label = kwargs.pop("hue_label", hue)
    hatch_label = kwargs.pop("hatch_label", hatch)
    handle_kwargs = dict(alpha=alpha, linewidth=linewidth, color=color, style="rectangle")

    if hue == hatch:
        # combined legend for hue and hatch
        if flags["hue"]:
            values = list(palette.keys())
            handles = create_legend_handles(
                labels=values,
                colors=[palette[v] for v in values],
                edgecolors=[edgecolor] * len(values) if edgecolor else None,
                hatches=[hatch_map[v] for v in values],
                **handle_kwargs,
            )
            stash_entry(
                ax,
                LegendEntry.build(
                    name=hue_label,
                    kind="hue",
                    handles=handles,
                    labels=values,
                ),
            )
    elif hue == categorical_axis:
        # legend for hatch only
        if flags["hatch"]:
            labels = list(hatch_map.keys())
            handles = create_legend_handles(
                labels=labels,
                colors=[resolve_param("color", color)] * len(hatch_map),
                edgecolors=[edgecolor] * len(hatch_map) if edgecolor else None,
                hatches=list(hatch_map.values()),
                **handle_kwargs,
            )
            stash_entry(
                ax,
                LegendEntry.build(
                    name=hatch_label,
                    kind="hatch",
                    handles=handles,
                    labels=labels,
                ),
            )
    elif hatch == categorical_axis:
        # legend for hue only
        if flags["hue"]:
            labels = list(palette.keys())
            handles = create_legend_handles(
                labels=labels,
                colors=[palette[v] for v in palette.keys()],
                edgecolors=[edgecolor] * len(palette) if edgecolor else None,
                hatches=None,
                **handle_kwargs,
            )
            stash_entry(
                ax,
                LegendEntry.build(
                    name=hue_label,
                    kind="hue",
                    handles=handles,
                    labels=labels,
                ),
            )
    else:
        # double split: hue first, then hatch
        if flags["hue"] and palette is not None and len(palette) > 0:
            labels = list(palette.keys())
            handles = create_legend_handles(
                labels=labels,
                colors=[palette[v] for v in palette.keys()],
                edgecolors=[edgecolor] * len(palette) if edgecolor else None,
                hatches=None,
                **handle_kwargs,
            )
            stash_entry(
                ax,
                LegendEntry.build(
                    name=hue_label,
                    kind="hue",
                    handles=handles,
                    labels=labels,
                ),
            )
        if flags["hatch"] and hatch_map is not None and len(hatch_map) > 0:
            # Use gray for hatch legend if hue exists, otherwise resolved color.
            hatch_color = "gray" if hue is not None else resolve_param("color", color)
            labels = list(hatch_map.keys())
            hatch_handle_kwargs = dict(handle_kwargs)
            hatch_handle_kwargs["color"] = hatch_color
            handles = create_legend_handles(
                labels=labels,
                colors=[hatch_color] * len(hatch_map),
                edgecolors=[edgecolor] * len(hatch_map) if edgecolor else None,
                hatches=list(hatch_map.values()),
                **{k: v for k, v in hatch_handle_kwargs.items() if k != "color"},
                # color already passed positionally-named above
            )
            stash_entry(
                ax,
                LegendEntry.build(
                    name=hatch_label,
                    kind="hatch",
                    handles=handles,
                    labels=labels,
                ),
            )

    render_entries(ax, flags=flags)
```

**Important note on the hatch-handle-kwargs spread:** `create_legend_handles` accepts `color` as a single kwarg OR `colors` as a list. The double-split hatch branch needs the `hatch_color` to paint every handle the same color, which is what `colors=[hatch_color] * len(hatch_map)` does. The `handle_kwargs` dict also has `color=color` in it from the outer scope, so we spread only the non-`color` items to avoid passing both. Read the final block carefully.

5. Update the call site for `_legend`. Locate it at bar.py:301 (inside `barplot(...)`):

```python
    # Add legend if hue or hatch is used
    if legend:
        _legend(
            ...
            kwargs=legend_kws,
        )
```

Change to:

```python
    # Stash entries and render per-axis unless claimed by a figure-level group.
    _legend(
        ax=ax,
        hue=hue,
        hatch=hatch,
        categorical_axis=categorical_axis,
        alpha=alpha,
        linewidth=linewidth,
        color=color,
        edgecolor=resolved_edgecolor,
        palette=palette,
        hatch_map=hatch_map,
        kwargs=legend_kws,
        legend=legend,
    )
```

The outer `if legend:` is gone — `_legend` now handles `legend is False` internally.

- [ ] **Step 5: Run new bar tests**

Run: `unset VIRTUAL_ENV && .venv/bin/pytest tests/test_bar_legend_stash.py -v`
Expected: 7 passed.

- [ ] **Step 6: Full suite**

Run: `unset VIRTUAL_ENV && .venv/bin/pytest -q`
Expected: 186 + 7 = 193 passed.

- [ ] **Step 7: Commit**

```bash
git add src/publiplots/plot/bar.py tests/test_bar_legend_stash.py
git commit -m "feat(bar): migrate legend to LegendEntry stash pattern with hatch kind"
```

---

## Task 5: Migrate heatmap (categorical path) to stash pattern

**Files:**
- Modify: `src/publiplots/plot/heatmap.py` (the categorical heatmap legend block at lines 340–361 + imports + signature)
- Test: `tests/test_heatmap_legend_stash.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_heatmap_legend_stash.py`:

```python
"""Tests for heatmap legend stashing via LegendEntry."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pytest

import publiplots as pp
from publiplots.utils.legend_entries import get_entries


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _matrix_df(rows=4, cols=5, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(size=(rows, cols)),
                        index=[f"r{i}" for i in range(rows)],
                        columns=[f"c{i}" for i in range(cols)])


def test_categorical_heatmap_stashes_continuous_hue_entry():
    """Categorical heatmap stashes one continuous-hue (colorbar) entry."""
    from publiplots.utils.legend_entries import is_continuous_hue
    df = _matrix_df()
    fig, ax = pp.heatmap(data=df)
    entries = get_entries(ax)
    assert len(entries) == 1
    assert entries[0].kind == "hue"
    assert is_continuous_hue(entries[0].handles)


def test_categorical_heatmap_legend_false_stashes_nothing():
    df = _matrix_df()
    fig, ax = pp.heatmap(data=df, legend=False)
    assert get_entries(ax) == []
```

- [ ] **Step 2: Confirm failures**

Run: `unset VIRTUAL_ENV && .venv/bin/pytest tests/test_heatmap_legend_stash.py -v`
Expected: 2 FAIL (heatmap doesn't stash yet).

- [ ] **Step 3: Update heatmap imports and signature**

In `src/publiplots/plot/heatmap.py`:

1. Add to the imports after the existing `from publiplots.utils import ...` line:

```python
from publiplots.utils.legend_entries import (
    LegendEntry,
    stash_entry,
    resolve_legend_flags,
)
from publiplots.utils.plot_legend import render_entries
```

2. In the public `heatmap(...)` signature around line 48, change `legend: bool = True,` to `legend: Union[bool, Dict] = True,`. Verify `Union` and `Dict` are already imported (they should be).

Also check the `_categorical_heatmap` helper signature around line 300 — change its `legend: bool` annotation to `legend: Union[bool, Dict]`.

- [ ] **Step 4: Replace the categorical legend block**

Find the block at lines 339–361 inside the categorical heatmap helper:

```python
    # Add colorbar via legend system
    if legend:
        # Determine normalization
        v_min = vmin if vmin is not None else matrix.min().min()
        v_max = vmax if vmax is not None else matrix.max().max()

        if center is not None:
            from matplotlib.colors import TwoSlopeNorm
            norm = TwoSlopeNorm(vmin=v_min, vcenter=center, vmax=v_max)
        else:
            norm = Normalize(vmin=v_min, vmax=v_max)

        mappable = ScalarMappable(norm=norm, cmap=cmap)

        builder = legend_builder(ax=ax, auto=False)
        builder.add_colorbar(
            mappable=mappable,
            label=legend_kws.get("value_label", value_col or ""),
            height=legend_kws.get("height", 20),
            width=legend_kws.get("width", 5),
        )
```

Replace with:

```python
    # Stash colorbar entry and render per-axis unless claimed by a group.
    if legend is not False:
        v_min = vmin if vmin is not None else matrix.min().min()
        v_max = vmax if vmax is not None else matrix.max().max()

        if center is not None:
            from matplotlib.colors import TwoSlopeNorm
            norm = TwoSlopeNorm(vmin=v_min, vcenter=center, vmax=v_max)
        else:
            norm = Normalize(vmin=v_min, vmax=v_max)

        mappable = ScalarMappable(norm=norm, cmap=cmap)

        flags = resolve_legend_flags(legend)
        if flags["hue"]:
            stash_entry(
                ax,
                LegendEntry.build(
                    name=legend_kws.get("value_label", value_col or ""),
                    kind="hue",
                    handles=[mappable],
                    labels=[],
                ),
            )
        render_entries(ax, flags=flags)
```

- [ ] **Step 5: Run heatmap tests**

Run: `unset VIRTUAL_ENV && .venv/bin/pytest tests/test_heatmap_legend_stash.py -v`
Expected: 2 passed.

- [ ] **Step 6: Full suite**

Run: `unset VIRTUAL_ENV && .venv/bin/pytest -q`
Expected: 193 + 2 = 195 passed.

- [ ] **Step 7: Commit**

```bash
git add src/publiplots/plot/heatmap.py tests/test_heatmap_legend_stash.py
git commit -m "feat(heatmap): migrate categorical-path legend to stash pattern"
```

---

## Task 6: Migrate heatmap (dot path) to stash pattern

**Files:**
- Modify: `src/publiplots/plot/heatmap.py` (the dot-heatmap legend block at lines 497–523)
- Test: extend `tests/test_heatmap_legend_stash.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_heatmap_legend_stash.py`:

```python
def _dot_df(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i, row in enumerate(["r0", "r1", "r2"]):
        for j, col in enumerate(["c0", "c1", "c2", "c3"]):
            rows.append({
                "row": row, "col": col,
                "value": rng.normal(),
                "size_var": rng.uniform(1, 10),
            })
    return pd.DataFrame(rows)


def test_dot_heatmap_stashes_hue_and_size_entries():
    """Dot heatmap (value_col + size_col) stashes one continuous-hue + one size entry."""
    from publiplots.utils.legend_entries import is_continuous_hue
    df = _dot_df()
    fig, ax = pp.heatmap(
        data=df, x="col", y="row", value="value", size="size_var",
    )
    entries = get_entries(ax)
    kinds = [e.kind for e in entries]
    assert "hue" in kinds
    assert "size" in kinds
    hue_entry = next(e for e in entries if e.kind == "hue")
    assert is_continuous_hue(hue_entry.handles)


def test_dot_heatmap_legend_dict_suppresses_size():
    df = _dot_df()
    fig, ax = pp.heatmap(
        data=df, x="col", y="row", value="value", size="size_var",
        legend={"size": False},
    )
    entries = get_entries(ax)
    kinds = [e.kind for e in entries]
    assert "hue" in kinds
    assert "size" not in kinds


def test_dot_heatmap_legend_false_stashes_nothing():
    df = _dot_df()
    fig, ax = pp.heatmap(
        data=df, x="col", y="row", value="value", size="size_var",
        legend=False,
    )
    assert get_entries(ax) == []
```

- [ ] **Step 2: Confirm new tests fail**

Run: `unset VIRTUAL_ENV && .venv/bin/pytest tests/test_heatmap_legend_stash.py -v`
Expected: the 3 new tests FAIL.

- [ ] **Step 3: Update dot-path signature annotation**

Find the `_dot_heatmap` helper signature around line 382 — change its `legend: bool` annotation to `legend: Union[bool, Dict]`.

- [ ] **Step 4: Replace the dot-path legend block**

Find the block at lines 497–523:

```python
    # Add legends
    if legend:
        builder = legend_builder(ax=ax, auto=False)

        # Color legend (colorbar)
        mappable = ScalarMappable(norm=color_norm, cmap=cmap)
        builder.add_colorbar(
            mappable=mappable,
            label=legend_kws.get("value_label", value_col or ""),
            height=legend_kws.get("colorbar_height", 20),
            width=legend_kws.get("colorbar_width", 5),
        )

        # Size legend
        size_handles, size_labels = _create_size_legend(
            marker_sizes=marker_sizes,
            sizes=sizes,
            size_norm=size_norm,
            alpha=alpha,
            linewidth=linewidth,
        )
        builder.add_legend(
            handles=size_handles,
            label=legend_kws.get("size_label", size_col or ""),
        )
```

Replace with:

```python
    # Stash colorbar + size entries and render per-axis unless claimed by a group.
    if legend is not False:
        flags = resolve_legend_flags(legend)

        if flags["hue"]:
            mappable = ScalarMappable(norm=color_norm, cmap=cmap)
            stash_entry(
                ax,
                LegendEntry.build(
                    name=legend_kws.get("value_label", value_col or ""),
                    kind="hue",
                    handles=[mappable],
                    labels=[],
                ),
            )

        if flags["size"]:
            size_handles, size_labels = _create_size_legend(
                marker_sizes=marker_sizes,
                sizes=sizes,
                size_norm=size_norm,
                alpha=alpha,
                linewidth=linewidth,
            )
            stash_entry(
                ax,
                LegendEntry.build(
                    name=legend_kws.get("size_label", size_col or ""),
                    kind="size",
                    handles=size_handles,
                    labels=size_labels,
                ),
            )

        render_entries(ax, flags=flags)
```

- [ ] **Step 5: Run all heatmap tests**

Run: `unset VIRTUAL_ENV && .venv/bin/pytest tests/test_heatmap_legend_stash.py -v`
Expected: 5 passed (2 from Task 5 + 3 new).

- [ ] **Step 6: Full suite**

Run: `unset VIRTUAL_ENV && .venv/bin/pytest -q`
Expected: 195 + 3 = 198 passed.

- [ ] **Step 7: Commit**

```bash
git add src/publiplots/plot/heatmap.py tests/test_heatmap_legend_stash.py
git commit -m "feat(heatmap): migrate dot-path legend to stash pattern"
```

---

## Task 7: Gallery smoke-run + new bar panel

**Files:**
- Modify: `examples/plots/plot_01_bar_plots.py` — append a new section demonstrating bar + `pp.legend_group`

- [ ] **Step 1: Run the gallery first, capture any new failures**

Run:

```bash
unset VIRTUAL_ENV
for f in examples/plots/plot_*.py; do
  .venv/bin/python "$f" >/dev/null 2>&1 && echo "$(basename $f): OK" || echo "$(basename $f): FAIL"
done
```

Expected: all 14 lines end with `OK`.

If any fails, run that file directly without stderr redirect to diagnose, then fix the migration or the example as appropriate.

- [ ] **Step 2: Add a shared-legend bar panel**

Append to the end of `examples/plots/plot_01_bar_plots.py`:

```python
# %%
# Shared Legend Across Subplots
# ------------------------------
# When several bar subplots share the same ``hue`` variable, attach
# ``pp.legend_group(anchor=...)`` before drawing. Each ``barplot`` stashes
# its hue entry on the corresponding axes; the group collects them and
# renders one legend on the right of the rightmost subplot.

np.random.seed(99)
shared_df = pd.DataFrame({
    "cat": np.tile(["A", "B", "C"], 60),
    "val": np.random.randn(180) + np.tile([0, 1, 2], 60),
    "g": np.repeat(["low", "mid", "high"], 60),
})

fig, axes = pp.subplots(1, 3, axes_size=(40, 35))
pp.legend_group(anchor=axes[-1])
for ax, title in zip(axes, ["Sample A", "Sample B", "Sample C"]):
    pp.barplot(
        data=shared_df, x="cat", y="val", hue="g",
        palette="pastel", title=title, ax=ax,
        errorbar="se",
    )
plt.show()
```

- [ ] **Step 3: Rerun the gallery**

Run the same loop from Step 1 — all 14 should still report `OK`.

- [ ] **Step 4: Commit**

```bash
git add examples/plots/plot_01_bar_plots.py
git commit -m "docs(examples): add shared-legend bar panel demonstrating auto-collection"
```

---

## Task 8: Push branch and open PR

- [ ] **Step 1: Push**

Run: `git push -u origin feat/legend-migration-b`
Expected: branch appears on GitHub.

- [ ] **Step 2: Open the PR**

Run:

```bash
gh pr create --title "feat(legend): migrate bar + heatmap to stash pattern; add shared helper" --body "$(cat <<'EOF'
## Summary
- Introduces ``src/publiplots/utils/plot_legend.py`` with two shared primitives: ``stash_hue_legend`` (single-kind hue stash + render) and ``render_entries`` (per-axis render of already-stashed entries not claimed by a group).
- Adds ``"hatch"`` as a new ``_LEGEND_KINDS`` entry to support bar's secondary split dimension.
- Migrates ``barplot`` (four-case hue/hatch dispatch preserved) and ``heatmap`` (both categorical and dot paths) to the stash pattern.
- Retrofits ``boxplot`` and ``violinplot`` to delegate to the shared helper, eliminating the byte-duplicate ``_stash_legend`` introduced in PR #83.

## Scope
- All publiplots plot functions now participate in ``pp.legend_group`` auto-collection.
- All plot functions now accept ``legend: bool | dict[kind, bool]`` for per-kind control.

## Test plan
- [x] ``tests/test_plot_legend.py`` — 9 tests for the new shared helpers
- [x] ``tests/test_bar_legend_stash.py`` — 7 tests across all four bar dispatch cases + per-kind flag + group integration
- [x] ``tests/test_heatmap_legend_stash.py`` — 5 tests for categorical + dot paths
- [x] Retrofit validation: PR #83's 12 tests still green without modification
- [x] Full suite: 198 passed (176 baseline + 22 new)
- [x] Gallery smoke-run: all 14 ``examples/plots/*.py`` render without errors, including new bar + ``legend_group`` panel

Closes part 2/2 of the follow-up work queued after PR #81.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Expected: PR URL printed.

---

## Self-review findings

**Spec coverage check (against `docs/superpowers/specs/2026-05-01-legend-migration-phase2-design.md`, "Per-plot migration plan — PR B" section):**
- `"hatch"` kind added → Task 1 ✓
- Shared helper → Task 2 ✓ (added above the spec's explicit requirement for cleanliness; PR A already shipped duplicated code that needed dedup)
- Box + violin retrofit → Task 3 ✓ (also above spec — consequence of the shared helper)
- Bar migration with four-case dispatch → Task 4 ✓
- Heatmap categorical path → Task 5 ✓
- Heatmap dot path (hue + size) → Task 6 ✓
- Testing strategy (per-case + group integration + per-kind flag) → Tasks 4–6 Step 1 ✓
- Gallery smoke + new bar + ``legend_group`` example → Task 7 ✓

**Placeholder scan:** None.

**Type consistency:**
- `stash_hue_legend` signature is consistent between `plot_legend.py` and the box/violin delegate calls.
- `render_entries(ax, *, flags)` kwargs match between definition and all callers (bar, heatmap, via `stash_hue_legend`).
- `kind="hatch"` used consistently in bar (only).
- `Union[bool, Dict]` used consistently across bar, heatmap, box, violin signatures.

No gaps found.
