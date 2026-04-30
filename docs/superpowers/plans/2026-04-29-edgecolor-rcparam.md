# Global `edgecolor` via rcParams — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `"edgecolor"` to `pp.rcParams` (default `None`) and wire every plot function to read it via `resolve_param`, preserving existing "auto" behavior.

**Architecture:** One entry in `PUBLIPLOTS_RCPARAMS`. One `resolve_param("edgecolor", edgecolor)` line at the top of each plot function. Zero changes to downstream artist logic — `None` propagates through the existing `edgecolor if edgecolor else <face>` ternaries.

**Tech Stack:** Python 3.9+, pandas, matplotlib, seaborn, pytest. Build/test via `uv`.

**Spec:** `docs/superpowers/specs/2026-04-29-edgecolor-rcparam-design.md`

**Working directory:** `/home/sagemaker-user/publiplots/.worktrees/edgecolor-rcparam` (branch `feat/edgecolor-rcparam`).

---

## Task 1: Register `edgecolor` in `PUBLIPLOTS_RCPARAMS`

**Files:**
- Modify: `src/publiplots/themes/rcparams.py` (around line 107–124, the `PUBLIPLOTS_RCPARAMS` dict)
- Test: `tests/test_edgecolor_rcparam.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_edgecolor_rcparam.py`:

```python
"""Tests for the global edgecolor rcParam."""
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

import pandas as pd
import numpy as np

import publiplots as pp


@pytest.fixture(autouse=True)
def _restore_edgecolor_rcparam():
    """Snapshot and restore pp.rcParams['edgecolor'] so tests don't leak state."""
    original = pp.rcParams["edgecolor"]
    yield
    pp.rcParams["edgecolor"] = original


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_edgecolor_rcparam_default_is_none():
    """The default edgecolor rcParam is None — 'auto' mode, preserves current behavior."""
    assert pp.rcParams["edgecolor"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_edgecolor_rcparam.py::test_edgecolor_rcparam_default_is_none -v`

Expected: FAIL with `KeyError: "Parameter 'edgecolor' not found in publiplots or matplotlib rcParams"`

- [ ] **Step 3: Add the rcParam entry**

Modify `src/publiplots/themes/rcparams.py`. Locate the `PUBLIPLOTS_RCPARAMS` dict (around line 107) and add `"edgecolor": None` after the `"alpha": 0.1` line:

```python
PUBLIPLOTS_RCPARAMS: Dict[str, Any] = {
    # Color and transparency
    "color": "#5d83c3",  # Default blue
    "alpha": 0.1,  # Default transparency for bars
    "edgecolor": None,  # Global edge color for patches and marker outlines; None = "auto" (match face)

    # Error bars
    "capsize": 0.0,  # Error bar cap size
    # ...rest unchanged...
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_edgecolor_rcparam.py::test_edgecolor_rcparam_default_is_none -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/publiplots/themes/rcparams.py tests/test_edgecolor_rcparam.py
git commit -m "feat(rcparams): add edgecolor with None default

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Wire `barplot` to rcParam (full behavior + precedence tests)

**Files:**
- Modify: `src/publiplots/plot/bar.py` (around line 141–146, after the other `resolve_param` calls)
- Test: `tests/test_edgecolor_rcparam.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_edgecolor_rcparam.py`:

```python
def _simple_bar_data():
    return pd.DataFrame({
        "category": pd.Categorical(["A", "B", "C"]),
        "value": [1.0, 2.0, 3.0],
    })


def test_barplot_rcparam_applies_when_arg_omitted():
    """Setting rcParams['edgecolor'] colors bar edges when no arg is passed."""
    pp.rcParams["edgecolor"] = "red"
    fig, ax = plt.subplots()
    pp.barplot(data=_simple_bar_data(), x="category", y="value", ax=ax)
    red = to_rgba("red")
    edges = [to_rgba(patch.get_edgecolor()) for patch in ax.patches]
    assert edges, "expected at least one bar patch"
    for edge in edges:
        assert edge == red


def test_barplot_explicit_arg_wins_over_rcparam():
    """When both rcParam and arg are set, the arg wins."""
    pp.rcParams["edgecolor"] = "red"
    fig, ax = plt.subplots()
    pp.barplot(data=_simple_bar_data(), x="category", y="value",
               edgecolor="blue", ax=ax)
    blue = to_rgba("blue")
    for patch in ax.patches:
        assert to_rgba(patch.get_edgecolor()) == blue


def test_barplot_passthrough_preserves_auto_edge():
    """With rcParam at its default None and no arg, bar edge matches face color."""
    assert pp.rcParams["edgecolor"] is None
    fig, ax = plt.subplots()
    pp.barplot(data=_simple_bar_data(), x="category", y="value", ax=ax)
    for patch in ax.patches:
        face = to_rgba(patch.get_facecolor())
        edge = to_rgba(patch.get_edgecolor())
        # Compare only RGB — bar face is alpha-dimmed but edge is full opacity
        assert face[:3] == edge[:3]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_edgecolor_rcparam.py -v`

Expected: The two new tests that depend on rcParam application (`test_barplot_rcparam_applies_when_arg_omitted`) FAIL — `barplot` ignores the rcParam. The explicit-wins and passthrough tests may pass already; that's fine. Task is not done until the rcParam test fails.

- [ ] **Step 3: Add `resolve_param` call at top of `barplot`**

Modify `src/publiplots/plot/bar.py`. In `barplot` (around line 141), add `edgecolor = resolve_param("edgecolor", edgecolor)` after the existing `resolve_param` block:

```python
# Read defaults from rcParams if not provided
figsize = resolve_param("figure.figsize", figsize)
linewidth = resolve_param("lines.linewidth", linewidth)
alpha = resolve_param("alpha", alpha)
capsize = resolve_param("capsize", capsize)
color = resolve_param("color", color)
edgecolor = resolve_param("edgecolor", edgecolor)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_edgecolor_rcparam.py -v`

Expected: all 4 tests PASS.

- [ ] **Step 5: Run full test suite to confirm no regression**

Run: `uv run pytest tests/ -q 2>&1 | tail -5`

Expected: all 33+ tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/publiplots/plot/bar.py tests/test_edgecolor_rcparam.py
git commit -m "feat(bar): resolve edgecolor from rcParams

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Wire `boxplot` to rcParam

**Files:**
- Modify: `src/publiplots/plot/box.py` (around line 135–138, after existing `resolve_param` calls)
- Test: `tests/test_edgecolor_rcparam.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_edgecolor_rcparam.py`:

```python
def test_boxplot_rcparam_applies():
    """boxplot respects rcParams['edgecolor']."""
    pp.rcParams["edgecolor"] = "red"
    data = pd.DataFrame({
        "group": pd.Categorical(np.repeat(["A", "B"], 20)),
        "value": np.concatenate([np.random.RandomState(0).randn(20),
                                  np.random.RandomState(1).randn(20)]),
    })
    fig, ax = plt.subplots()
    pp.boxplot(data=data, x="group", y="value", ax=ax)
    red = to_rgba("red")
    # Box edges live on ax.patches in seaborn's boxplot
    for patch in ax.patches:
        assert to_rgba(patch.get_edgecolor()) == red
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_edgecolor_rcparam.py::test_boxplot_rcparam_applies -v`

Expected: FAIL — box edges will match palette (auto), not red.

- [ ] **Step 3: Add `resolve_param` call at top of `boxplot`**

Modify `src/publiplots/plot/box.py`. In `boxplot` (around line 135), add `edgecolor = resolve_param("edgecolor", edgecolor)` after the existing `resolve_param` block:

```python
figsize = resolve_param("figure.figsize", figsize)
linewidth = resolve_param("lines.linewidth", linewidth)
alpha = resolve_param("alpha", alpha)
color = resolve_param("color", color)
edgecolor = resolve_param("edgecolor", edgecolor)
```

(Leave the existing `linecolor` backcompat logic further down untouched — it resolves `edgecolor` vs `linecolor` into `resolved_edgecolor` and that path continues to work.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_edgecolor_rcparam.py::test_boxplot_rcparam_applies -v`

Expected: PASS.

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest tests/ -q 2>&1 | tail -5`

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/publiplots/plot/box.py tests/test_edgecolor_rcparam.py
git commit -m "feat(box): resolve edgecolor from rcParams

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Wire `scatterplot` to rcParam

**Files:**
- Modify: `src/publiplots/plot/scatter.py` (around line 168–171, top of public `scatterplot`)
- Test: `tests/test_edgecolor_rcparam.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_edgecolor_rcparam.py`:

```python
def test_scatterplot_rcparam_applies():
    """scatterplot respects rcParams['edgecolor']."""
    pp.rcParams["edgecolor"] = "red"
    data = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    fig, ax = plt.subplots()
    pp.scatterplot(data=data, x="x", y="y", ax=ax)
    red = to_rgba("red")
    # scatter marker edges live on the PathCollection in ax.collections
    assert ax.collections, "expected a PathCollection from scatter"
    for collection in ax.collections:
        edges = collection.get_edgecolors()
        assert len(edges) > 0
        for edge in edges:
            assert tuple(edge) == red
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_edgecolor_rcparam.py::test_scatterplot_rcparam_applies -v`

Expected: FAIL.

- [ ] **Step 3: Add `resolve_param` call at top of `scatterplot`**

Modify `src/publiplots/plot/scatter.py`. In public `scatterplot` (around line 168), add after the existing `resolve_param` block:

```python
figsize = resolve_param("figure.figsize", figsize)
linewidth = resolve_param("lines.linewidth", linewidth)
alpha = resolve_param("alpha", alpha)
color = resolve_param("color", color)
edgecolor = resolve_param("edgecolor", edgecolor)
```

Do NOT modify the internal helper around line 474 (`_scatterplot_legend` or similar internal helper that also takes `edgecolor=None` as a kwarg — it receives the already-resolved value from the public function).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_edgecolor_rcparam.py::test_scatterplot_rcparam_applies -v`

Expected: PASS.

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest tests/ -q 2>&1 | tail -5`

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/publiplots/plot/scatter.py tests/test_edgecolor_rcparam.py
git commit -m "feat(scatter): resolve edgecolor from rcParams

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Wire `violinplot` to rcParam

**Files:**
- Modify: `src/publiplots/plot/violin.py` (around line 173–176, after existing `resolve_param` block)

- [ ] **Step 1: Add `resolve_param` call at top of `violinplot`**

Modify `src/publiplots/plot/violin.py`. In `violinplot` (around line 173), add after the existing `resolve_param` block:

```python
figsize = resolve_param("figure.figsize", figsize)
linewidth = resolve_param("lines.linewidth", linewidth)
alpha = resolve_param("alpha", alpha)
color = resolve_param("color", color)
edgecolor = resolve_param("edgecolor", edgecolor)
```

- [ ] **Step 2: Run full test suite to confirm no regression**

Run: `uv run pytest tests/ -q 2>&1 | tail -5`

Expected: all tests PASS (no new assertions for violin, but existing suite must stay green).

- [ ] **Step 3: Smoke-test manually via a tiny Python snippet**

Run:

```bash
uv run python - <<'PY'
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd, numpy as np
import publiplots as pp
from matplotlib.colors import to_rgba

pp.rcParams["edgecolor"] = "red"
data = pd.DataFrame({
    "group": pd.Categorical(np.repeat(["A", "B"], 30)),
    "value": np.random.RandomState(0).randn(60),
})
fig, ax = plt.subplots()
pp.violinplot(data=data, x="group", y="value", ax=ax)
red = to_rgba("red")
body_edges = {to_rgba(p.get_edgecolor()) for p in ax.collections}
print("violin body edges:", body_edges)
assert red in body_edges or any(red == e for e in body_edges), f"expected red edges, got {body_edges}"
print("OK")
PY
```

Expected output: `OK`.

- [ ] **Step 4: Commit**

```bash
git add src/publiplots/plot/violin.py
git commit -m "feat(violin): resolve edgecolor from rcParams

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Wire `stripplot` to rcParam

**Files:**
- Modify: `src/publiplots/plot/strip.py` (around line 127–130, after existing `resolve_param` block)

- [ ] **Step 1: Add `resolve_param` call at top of `stripplot`**

Modify `src/publiplots/plot/strip.py`. In `stripplot` (around line 127), add after the existing `resolve_param` block:

```python
figsize = resolve_param("figure.figsize", figsize)
linewidth = resolve_param("lines.linewidth", linewidth)
alpha = resolve_param("alpha", alpha)
color = resolve_param("color", color)
edgecolor = resolve_param("edgecolor", edgecolor)
```

- [ ] **Step 2: Run full test suite**

Run: `uv run pytest tests/ -q 2>&1 | tail -5`

Expected: all tests PASS.

- [ ] **Step 3: Commit**

```bash
git add src/publiplots/plot/strip.py
git commit -m "feat(strip): resolve edgecolor from rcParams

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Wire `swarmplot` to rcParam

**Files:**
- Modify: `src/publiplots/plot/swarm.py` (around line 124–127, after existing `resolve_param` block)

- [ ] **Step 1: Add `resolve_param` call at top of `swarmplot`**

Modify `src/publiplots/plot/swarm.py`. In `swarmplot` (around line 124), add after the existing `resolve_param` block:

```python
figsize = resolve_param("figure.figsize", figsize)
linewidth = resolve_param("lines.linewidth", linewidth)
alpha = resolve_param("alpha", alpha)
color = resolve_param("color", color)
edgecolor = resolve_param("edgecolor", edgecolor)
```

- [ ] **Step 2: Run full test suite**

Run: `uv run pytest tests/ -q 2>&1 | tail -5`

Expected: all tests PASS.

- [ ] **Step 3: Commit**

```bash
git add src/publiplots/plot/swarm.py
git commit -m "feat(swarm): resolve edgecolor from rcParams

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Wire `pointplot` to rcParam

**Files:**
- Modify: `src/publiplots/plot/point.py` (around line 183–189, after existing `resolve_param` block)

- [ ] **Step 1: Add `resolve_param` call at top of `pointplot`**

Modify `src/publiplots/plot/point.py`. In `pointplot` (around line 183), add after the existing `resolve_param` block (after `linestyle`):

```python
figsize = resolve_param("figure.figsize", figsize)
linewidth = resolve_param("lines.linewidth", linewidth)
markersize = resolve_param("lines.markersize", markersize)
markeredgewidth = resolve_param("lines.markeredgewidth", markeredgewidth)
alpha = resolve_param("alpha", alpha)
color = resolve_param("color", color)
linestyle = resolve_param("lines.linestyle", linestyle)
edgecolor = resolve_param("edgecolor", edgecolor)
```

- [ ] **Step 2: Run full test suite**

Run: `uv run pytest tests/ -q 2>&1 | tail -5`

Expected: all tests PASS.

- [ ] **Step 3: Smoke-test (marker-edge only; connecting line stays palette color)**

Run:

```bash
uv run python - <<'PY'
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd, numpy as np
import publiplots as pp
from matplotlib.colors import to_rgba

pp.rcParams["edgecolor"] = "red"
data = pd.DataFrame({
    "x": pd.Categorical(np.tile(["A", "B", "C"], 10)),
    "y": np.random.RandomState(0).randn(30),
})
fig, ax = plt.subplots()
pp.pointplot(data=data, x="x", y="y", ax=ax)
red = to_rgba("red")

# Marker edges should be red; Line2D strokes should NOT be red (they follow the series color)
marker_edges = []
line_strokes = []
for line in ax.get_lines():
    if line.get_marker() not in (None, "", "None"):
        marker_edges.append(to_rgba(line.get_markeredgecolor()))
    if line.get_linestyle() not in (None, "", "None"):
        line_strokes.append(to_rgba(line.get_color()))

print("marker edges:", marker_edges)
print("line strokes:", line_strokes)
assert red in marker_edges, f"expected red marker edges, got {marker_edges}"
assert red not in line_strokes, f"line strokes should NOT be red, got {line_strokes}"
print("OK")
PY
```

Expected output: `OK`.

- [ ] **Step 4: Commit**

```bash
git add src/publiplots/plot/point.py
git commit -m "feat(point): resolve edgecolor from rcParams (marker edges only)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: Wire `raincloudplot` to rcParam

**Files:**
- Modify: `src/publiplots/plot/raincloud.py` (around line 163–166, after existing `resolve_param` block)

- [ ] **Step 1: Add `resolve_param` call at top of `raincloudplot`**

Modify `src/publiplots/plot/raincloud.py`. In `raincloudplot` (around line 163), add after the existing `resolve_param` block:

```python
figsize = resolve_param("figure.figsize", figsize)
linewidth = resolve_param("lines.linewidth", linewidth)
cloud_alpha = resolve_param("alpha", cloud_alpha)
color = resolve_param("color", color)
edgecolor = resolve_param("edgecolor", edgecolor)
```

- [ ] **Step 2: Run full test suite**

Run: `uv run pytest tests/ -q 2>&1 | tail -5`

Expected: all tests PASS.

- [ ] **Step 3: Commit**

```bash
git add src/publiplots/plot/raincloud.py
git commit -m "feat(raincloud): resolve edgecolor from rcParams

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: Wire `heatmap` to rcParam

**Files:**
- Modify: `src/publiplots/plot/heatmap.py` (around line 175–177, after existing `resolve_param` block in `heatmap()`)

- [ ] **Step 1: Add `resolve_param` call at top of `heatmap`**

Modify `src/publiplots/plot/heatmap.py`. In `heatmap` (around line 175), add after the existing `resolve_param` block:

```python
figsize = resolve_param("figure.figsize", figsize)
linewidth = resolve_param("lines.linewidth", linewidth)
alpha = resolve_param("alpha", alpha)
edgecolor = resolve_param("edgecolor", edgecolor)
```

Do NOT modify `complex_heatmap()` or the `ComplexHeatmapBuilder` — they forward `edgecolor` through to `heatmap()`, so resolution at the single entry point is sufficient. When the builder's `.build()` method calls `heatmap(...)`, the rcParam value will be picked up there if the user didn't pass one explicitly up the chain.

- [ ] **Step 2: Run full test suite**

Run: `uv run pytest tests/ -q 2>&1 | tail -5`

Expected: all tests PASS.

- [ ] **Step 3: Smoke-test**

Run:

```bash
uv run python - <<'PY'
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd, numpy as np
import publiplots as pp
from matplotlib.colors import to_rgba

pp.rcParams["edgecolor"] = "red"
data = pd.DataFrame(np.random.RandomState(0).randn(4, 4),
                    index=["r1","r2","r3","r4"], columns=["c1","c2","c3","c4"])
fig, ax = plt.subplots()
pp.heatmap(data=data, ax=ax)
red = to_rgba("red")
# Cell edges live on the QuadMesh collection
qm = [c for c in ax.collections if hasattr(c, "get_edgecolors")][0]
edges = qm.get_edgecolors()
print("heatmap edges (first 3):", edges[:3] if len(edges) else "none")
# When edgecolor='face' (auto), matplotlib returns empty or same-as-face; when red is set, we expect red
assert any(tuple(e) == red for e in edges), f"expected red in edges, got {edges[:3]}"
print("OK")
PY
```

Expected output: `OK`.

- [ ] **Step 4: Commit**

```bash
git add src/publiplots/plot/heatmap.py
git commit -m "feat(heatmap): resolve edgecolor from rcParams

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 11: Documentation — README or docs note

**Files:**
- Modify: `docs/source/configuration.rst` (if it exists) OR append to `CHANGELOG.md` via unreleased entry

- [ ] **Step 1: Check whether `docs/source/configuration.rst` documents rcParams**

Run: `ls docs/source/configuration.rst 2>/dev/null && grep -n "rcParams\|alpha" docs/source/configuration.rst 2>/dev/null | head -10`

If the file exists and mentions rcParams, proceed with Step 2a. Otherwise, skip to Step 2b.

- [ ] **Step 2a: If `docs/source/configuration.rst` documents rcParams**

Add an entry to the rcParams list describing `edgecolor`:

```rst
.. note::

   ``pp.rcParams["edgecolor"]`` defaults to ``None``, which means each plot
   uses an "auto" edge color (typically the face/palette color). Set it to a
   matplotlib-recognized color string (e.g. ``"black"``) to apply uniform edges
   across all plots. Per-call ``edgecolor=`` arguments override the rcParam.
```

- [ ] **Step 2b: If no docs file, skip**

No changes needed. The rcParam is self-documenting via its inline comment in `rcparams.py`.

- [ ] **Step 3: Commit (only if docs were modified)**

```bash
git add docs/
git commit -m "docs: document edgecolor rcParam

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 12: Final verification and docs build

- [ ] **Step 1: Run the full test suite once more**

Run: `uv run pytest tests/ -q 2>&1 | tail -5`

Expected: all tests PASS (32 baseline + 5 new edgecolor tests = 37+).

- [ ] **Step 2: Build the docs gallery to catch regressions**

Run:

```bash
cd docs && uv run make html 2>&1 | tail -10
```

Expected: "build succeeded" and "Sphinx-Gallery successfully executed 13 out of 13 files".

- [ ] **Step 3: Push branch and open PR**

```bash
git push -u origin feat/edgecolor-rcparam
gh pr create --title "feat: global edgecolor via rcParams" --body "$(cat <<'EOF'
## Summary
- Add `pp.rcParams["edgecolor"]` (default `None` = "auto / match face").
- Each plot function resolves it via `resolve_param` — one line per module.
- Zero changes to downstream artist logic; existing `edgecolor=None` ternaries preserve current behavior.

## Test plan
- [x] 5 new unit tests covering default, rcParam application, explicit-wins, auto-passthrough, and coverage across barplot / boxplot / scatterplot.
- [x] Smoke tests for violin, point, heatmap confirming marker/patch/quadmesh edges react to the rcParam.
- [x] Full pytest suite green.
- [x] Docs gallery (`make html`) builds 13/13.

Spec: `docs/superpowers/specs/2026-04-29-edgecolor-rcparam-design.md`

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 4: Watch CI**

```bash
gh run watch $(gh run list --branch feat/edgecolor-rcparam --limit 1 --json databaseId --jq '.[0].databaseId') --exit-status
```

Expected: docs workflow PASS.
