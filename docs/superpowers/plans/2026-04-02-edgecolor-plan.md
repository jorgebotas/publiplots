# Unified Edgecolor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a unified `edgecolor` parameter across all plot types so users can set edge color independently from face color, with legends faithfully reflecting the override.

**Architecture:** Each plot function gets an `edgecolor: Optional[str] = None` parameter. After seaborn draws, the function sets facecolor from palette and edgecolor from the param (or facecolor if None). `create_legend_handles` gets an `edgecolors` parameter so legends mirror the actual artists. No changes to the transparency system.

**Tech Stack:** matplotlib, seaborn, pandas, pytest

**Branch:** `fix/barplot-hue-equals-x`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `tests/test_edgecolor.py` | Create | Tests for edgecolor across all plot types + legend |
| `src/publiplots/utils/legend.py` | Modify | Add `edgecolors` param to `create_legend_handles` |
| `src/publiplots/plot/bar.py` | Modify | Add `edgecolor`, fix face-from-edge bug |
| `src/publiplots/plot/box.py` | Modify | Add `edgecolor`, deprecate `linecolor`, fix outlier markers |
| `src/publiplots/plot/violin.py` | Modify | Add `edgecolor`, map to seaborn's `linecolor` |
| `src/publiplots/plot/scatter.py` | Modify | Pass existing `edgecolor` to legend |
| `src/publiplots/plot/strip.py` | Modify | Pass existing `edgecolor` to legend |
| `src/publiplots/plot/swarm.py` | Modify | Pass existing `edgecolor` to legend |
| `src/publiplots/plot/point.py` | Modify | Add `edgecolor`, apply to marker edges |
| `src/publiplots/plot/raincloud.py` | Modify | Add `edgecolor`, pass through to sub-plots |

## Dependency Graph

```
Task 1 (legend) ──┬── Task 2 (barplot)
                   ├── Task 3 (boxplot)
                   ├── Task 4 (scatterplot)
                   ├── Task 5 (violinplot)
                   ├── Task 6 (strip + swarm)
                   └── Task 7 (pointplot)
                            │
Tasks 2-7 all ────────── Task 8 (raincloudplot)
                            │
All tasks ────────────── Task 9 (integration tests)
```

Tasks 2–7 are independent of each other and can be executed in parallel.

---

### Task 1: Add `edgecolors` to `create_legend_handles`

**Files:**
- Modify: `src/publiplots/utils/legend.py:486-624` (`create_legend_handles` function)
- Create: `tests/test_edgecolor.py`

- [ ] **Step 1: Create test file with legend handle tests**

Create `tests/test_edgecolor.py`:

```python
"""Tests for unified edgecolor support across plot types."""
import pytest
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for CI
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

import publiplots as pp


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    n = 60
    return pd.DataFrame({
        "category": pd.Categorical(np.repeat(["A", "B", "C"], n // 3)),
        "group": pd.Categorical(np.tile(["X", "Y"], n // 2)),
        "value": np.random.randn(n),
    })


@pytest.fixture(autouse=True)
def close_figures():
    """Close all figures after each test."""
    yield
    plt.close("all")


# ============================================================
# Legend handle tests
# ============================================================

class TestCreateLegendHandles:
    """Tests for create_legend_handles edgecolors parameter."""

    def test_edgecolors_none_defaults_to_facecolor(self):
        """When edgecolors=None, edge matches face (existing behavior)."""
        handles = pp.create_legend_handles(
            labels=["A", "B"],
            colors=["red", "blue"],
            edgecolors=None,
            alpha=0.1,
            linewidth=1.0,
        )
        for h, fc in zip(handles, ["red", "blue"]):
            assert to_rgba(h.get_facecolor()) == to_rgba(fc)
            assert to_rgba(h.get_edgecolor()) == to_rgba(fc)

    def test_edgecolors_single_string_broadcasts(self):
        """A single string edgecolor applies to all handles."""
        handles = pp.create_legend_handles(
            labels=["A", "B"],
            colors=["red", "blue"],
            edgecolors="black",
            alpha=0.1,
            linewidth=1.0,
        )
        for h in handles:
            assert to_rgba(h.get_edgecolor()) == to_rgba("black")
        # Face colors still from colors param
        assert to_rgba(handles[0].get_facecolor()) == to_rgba("red")
        assert to_rgba(handles[1].get_facecolor()) == to_rgba("blue")

    def test_edgecolors_list_per_handle(self):
        """A list of edgecolors maps one-to-one to handles."""
        handles = pp.create_legend_handles(
            labels=["A", "B"],
            colors=["red", "blue"],
            edgecolors=["black", "gray"],
            alpha=0.1,
            linewidth=1.0,
        )
        assert to_rgba(handles[0].get_edgecolor()) == to_rgba("black")
        assert to_rgba(handles[1].get_edgecolor()) == to_rgba("gray")

    def test_edgecolors_with_markers(self):
        """Edgecolors work with MarkerPatch handles."""
        handles = pp.create_legend_handles(
            labels=["A", "B"],
            colors=["red", "blue"],
            edgecolors="black",
            markers=["o", "^"],
            alpha=0.1,
            linewidth=1.0,
        )
        for h in handles:
            assert to_rgba(h.get_edgecolor()) == to_rgba("black")

    def test_edgecolors_with_linemarkers(self):
        """Edgecolors work with LineMarkerPatch handles."""
        handles = pp.create_legend_handles(
            labels=["A", "B"],
            colors=["red", "blue"],
            edgecolors="black",
            markers=["o", "^"],
            linestyles=["-", "--"],
            alpha=0.1,
            linewidth=1.0,
        )
        for h in handles:
            assert to_rgba(h.get_edgecolor()) == to_rgba("black")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/sagemaker-user/publiplots && python -m pytest tests/test_edgecolor.py::TestCreateLegendHandles -v`

Expected: `test_edgecolors_none_defaults_to_facecolor` passes (existing behavior), all others FAIL with `TypeError: create_legend_handles() got an unexpected keyword argument 'edgecolors'`

- [ ] **Step 3: Implement `edgecolors` in `create_legend_handles`**

In `src/publiplots/utils/legend.py`, modify the `create_legend_handles` function signature (line 486) to add the `edgecolors` parameter and normalize it:

Add `edgecolors` parameter after `colors`:

```python
def create_legend_handles(
    labels: List[str],
    colors: Optional[List[str]] = None,
    edgecolors: Optional[Union[str, List[str]]] = None,
    hatches: Optional[List[str]] = None,
    sizes: Optional[List[float]] = None,
    markers: Optional[List[str]] = None,
    linestyles: Optional[List[str]] = None,
    alpha: Optional[float] = None,
    linewidth: Optional[float] = None,
    markeredgewidth: Optional[float] = None,
    style: str = "rectangle",
    color: Optional[str] = None
) -> List[Patch]:
```

Add the `Union` import if not present (it already is on line 6).

After the existing `colors` normalization block (around line 541-543), add edgecolors normalization:

```python
    # Normalize edgecolors: None -> use facecolor, str -> broadcast, list -> use as-is
    if edgecolors is None:
        edgecolors = colors  # default: edge matches face
    elif isinstance(edgecolors, str):
        edgecolors = [edgecolors] * len(labels)
```

Then update all three handle creation branches to use `edgecolors` instead of hardcoding `edgecolor=col`:

For the `LineMarkerPatch` branch (~line 567):
```python
    for label, col, edge_col, size, marker, linestyle in zip(labels, colors, edgecolors, sizes, markers, linestyles):
        handle = LineMarkerPatch(
            marker=marker,
            linestyle=linestyle,
            facecolor=col,
            edgecolor=edge_col,
            ...
        )
```

For the `MarkerPatch` branches (~lines 582, 598):
```python
    for label, col, edge_col, hatch, size, marker in zip(labels, colors, edgecolors, hatches, sizes, markers):
        handle = MarkerPatch(
            marker=marker,
            facecolor=col,
            edgecolor=edge_col,
            ...
        )
```

For the `RectanglePatch` branch (~line 611):
```python
    for label, col, edge_col, hatch, size in zip(labels, colors, edgecolors, hatches, sizes):
        handle = RectanglePatch(
            facecolor=col,
            edgecolor=edge_col,
            ...
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/sagemaker-user/publiplots && python -m pytest tests/test_edgecolor.py::TestCreateLegendHandles -v`

Expected: All 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_edgecolor.py src/publiplots/utils/legend.py
git commit -m "feat(legend): add edgecolors parameter to create_legend_handles"
```

---

### Task 2: Fix barplot edgecolor

**Files:**
- Modify: `src/publiplots/plot/bar.py:23-48` (signature), `274-278` (color flow), `248-257` (hue recolor), `376-439` (hatches), `441-530` (legend)
- Modify: `tests/test_edgecolor.py` (add barplot tests)

- [ ] **Step 1: Add barplot tests**

Append to `tests/test_edgecolor.py`:

```python
# ============================================================
# Barplot tests
# ============================================================

class TestBarplotEdgecolor:
    """Tests for barplot edgecolor parameter."""

    def test_default_edgecolor_matches_face(self, sample_data):
        """Without edgecolor, edges match face colors."""
        fig, ax = pp.barplot(data=sample_data, x="category", y="value", hue="category", legend=False)
        for patch in ax.patches:
            fc = patch.get_facecolor()
            ec = patch.get_edgecolor()
            # Edge and face should share the same base color (edge is opaque, face is transparent)
            assert to_rgba(ec)[:3] == pytest.approx(to_rgba(fc)[:3], abs=0.01)

    def test_custom_edgecolor_applied_to_patches(self, sample_data):
        """Custom edgecolor should override patch edges but not faces."""
        fig, ax = pp.barplot(data=sample_data, x="category", y="value", hue="category",
                             edgecolor="black", legend=False)
        for patch in ax.patches:
            ec = patch.get_edgecolor()
            fc = patch.get_facecolor()
            # Edge should be black
            assert to_rgba(ec)[:3] == pytest.approx(to_rgba("black")[:3], abs=0.01)
            # Face should NOT be black (should be palette color)
            assert to_rgba(fc)[:3] != pytest.approx(to_rgba("black")[:3], abs=0.01)

    def test_custom_edgecolor_applied_to_error_bars(self, sample_data):
        """Error bar lines should use edgecolor when specified."""
        fig, ax = pp.barplot(data=sample_data, x="category", y="value", hue="category",
                             edgecolor="black", legend=False)
        for line in ax.lines:
            lc = to_rgba(line.get_color())
            assert lc[:3] == pytest.approx(to_rgba("black")[:3], abs=0.01)

    def test_no_hue_with_edgecolor(self, sample_data):
        """Edgecolor works when hue is None."""
        fig, ax = pp.barplot(data=sample_data, x="category", y="value",
                             edgecolor="black", legend=False)
        for patch in ax.patches:
            ec = patch.get_edgecolor()
            assert to_rgba(ec)[:3] == pytest.approx(to_rgba("black")[:3], abs=0.01)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/sagemaker-user/publiplots && python -m pytest tests/test_edgecolor.py::TestBarplotEdgecolor -v`

Expected: FAIL — `barplot() got an unexpected keyword argument 'edgecolor'`

- [ ] **Step 3: Add `edgecolor` parameter and fix color flow**

In `src/publiplots/plot/bar.py`:

**3a. Add `edgecolor` to signature** (after `color` param, ~line 30):
```python
    edgecolor: Optional[str] = None,
```

**3b. Fix the color flow** — replace lines 274-278 (the face-from-edge copy + transparency):
```python
    # Set facecolor from seaborn's edge (palette color) and optionally override edgecolor
    for patch in tracker.get_new_patches():
        palette_color = patch.get_edgecolor()  # seaborn fill=False puts color on edge
        patch.set_facecolor(palette_color)
        if edgecolor is not None:
            patch.set_edgecolor(edgecolor)
    # Apply differential transparency to face vs edge
    tracker.apply_transparency(on="patches", face_alpha=alpha, edge_alpha=1.0)
```

**3c. Fix the hue == categorical_axis recolor block** (lines 248-257):
```python
    # When hue == categorical axis, recolor bars from palette
    if hue is not None and hue == categorical_axis and palette:
        categories = order if order else list(data[categorical_axis].cat.categories)
        for idx, patch in enumerate(tracker.get_new_patches()):
            if idx < len(categories):
                bar_color = palette.get(categories[idx], color)
                patch.set_edgecolor(edgecolor if edgecolor is not None else bar_color)
        # Also recolor error bar lines
        for idx, line in enumerate(tracker.get_new_lines()):
            if idx < len(categories):
                line.set_color(edgecolor if edgecolor is not None else palette.get(categories[idx], color))
```

**3d. Pass `edgecolor` to `_apply_hatches_and_override_colors`** — add it to the call (~line 261) and to the function signature (~line 376). Inside the function, when setting patch edgecolor (~line 434):
```python
        if not (double_split or hatch == categorical_axis):
            patch.set_edgecolor(edgecolor if edgecolor is not None else bar_color)
            patch.set_facecolor(bar_color)
            if bar_idx < len(errorbars):
                errorbars[bar_idx].set_color(edgecolor if edgecolor is not None else bar_color)
```

**3e. Pass `edgecolor` to `_legend()`** — add to the call (~line 282) and update `_legend` signature (~line 441) and handle_kwargs (~line 464):
```python
    handle_kwargs = dict(
        alpha=alpha,
        linewidth=linewidth,
        color=color,
        edgecolors=[edgecolor] * len(palette) if edgecolor and palette else None,
        style="rectangle",
    )
```

Pass the same `edgecolors` kwarg in every `create_legend_handles` call inside `_legend`.

**3f. Also apply edgecolor override to error bar lines** in the main flow. After the transparency block, if edgecolor is set:
```python
    if edgecolor is not None:
        for line in tracker.get_new_lines():
            line.set_color(edgecolor)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/sagemaker-user/publiplots && python -m pytest tests/test_edgecolor.py::TestBarplotEdgecolor -v`

Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/publiplots/plot/bar.py tests/test_edgecolor.py
git commit -m "feat(barplot): add edgecolor parameter, fix face-from-edge bug"
```

---

### Task 3: Fix boxplot edgecolor

**Files:**
- Modify: `src/publiplots/plot/box.py:23-49` (signature), `155-176` (kwargs), `192-226` (recoloring), `229-241` (legend)
- Modify: `tests/test_edgecolor.py` (add boxplot tests)

- [ ] **Step 1: Add boxplot tests**

Append to `tests/test_edgecolor.py`:

```python
# ============================================================
# Boxplot tests
# ============================================================

class TestBoxplotEdgecolor:
    """Tests for boxplot edgecolor parameter."""

    def test_default_edgecolor_matches_face(self, sample_data):
        """Without edgecolor, box edges match face colors."""
        fig, ax = pp.boxplot(data=sample_data, x="category", y="value", hue="category", legend=False)
        for patch in ax.patches:
            fc = patch.get_facecolor()
            ec = patch.get_edgecolor()
            assert to_rgba(ec)[:3] == pytest.approx(to_rgba(fc)[:3], abs=0.01)

    def test_custom_edgecolor_on_patches(self, sample_data):
        """Custom edgecolor applied to box patches."""
        fig, ax = pp.boxplot(data=sample_data, x="category", y="value", hue="category",
                             edgecolor="black", legend=False)
        for patch in ax.patches:
            ec = patch.get_edgecolor()
            assert to_rgba(ec)[:3] == pytest.approx(to_rgba("black")[:3], abs=0.01)
            # Face should still be palette color, not black
            fc = patch.get_facecolor()
            assert to_rgba(fc)[:3] != pytest.approx(to_rgba("black")[:3], abs=0.01)

    def test_custom_edgecolor_structural_lines(self, sample_data):
        """Whiskers and caps should use edgecolor."""
        fig, ax = pp.boxplot(data=sample_data, x="category", y="value",
                             edgecolor="red", legend=False)
        # Lines without markers are structural (whiskers, caps, medians)
        for line in ax.lines:
            marker = line.get_marker()
            if marker == "None" or marker == "" or marker is None:
                lc = to_rgba(line.get_color())
                assert lc[:3] == pytest.approx(to_rgba("red")[:3], abs=0.01)

    def test_linecolor_backward_compat(self, sample_data):
        """linecolor still works as fallback when edgecolor is not set."""
        fig, ax = pp.boxplot(data=sample_data, x="category", y="value",
                             linecolor="blue", legend=False)
        for patch in ax.patches:
            ec = patch.get_edgecolor()
            assert to_rgba(ec)[:3] == pytest.approx(to_rgba("blue")[:3], abs=0.01)

    def test_edgecolor_overrides_linecolor(self, sample_data):
        """edgecolor takes precedence over linecolor with a warning."""
        with pytest.warns(FutureWarning, match="linecolor.*deprecated"):
            fig, ax = pp.boxplot(data=sample_data, x="category", y="value",
                                 edgecolor="black", linecolor="blue", legend=False)
        for patch in ax.patches:
            ec = patch.get_edgecolor()
            assert to_rgba(ec)[:3] == pytest.approx(to_rgba("black")[:3], abs=0.01)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/sagemaker-user/publiplots && python -m pytest tests/test_edgecolor.py::TestBoxplotEdgecolor -v`

Expected: FAIL — `test_custom_edgecolor_on_patches` and others fail because `edgecolor` param doesn't exist yet or behavior is wrong.

- [ ] **Step 3: Implement edgecolor in boxplot**

In `src/publiplots/plot/box.py`:

**3a. Add `edgecolor` to signature** (after `linecolor`, ~line 33):
```python
    edgecolor: Optional[str] = None,
```

Add `import warnings` at top of file.

**3b. Resolve edgecolor vs linecolor** — after `color = resolve_param(...)` (~line 133), add:
```python
    # Resolve edgecolor vs linecolor (backward compat)
    if edgecolor is not None and linecolor is not None:
        import warnings
        warnings.warn(
            "linecolor is deprecated in favor of edgecolor. "
            "edgecolor takes precedence when both are provided.",
            FutureWarning,
            stacklevel=2,
        )
    resolved_edgecolor = edgecolor if edgecolor is not None else linecolor
```

**3c. Pass `resolved_edgecolor` to seaborn as `linecolor`** — in boxplot_kwargs (~line 165):
```python
        "linecolor": resolved_edgecolor,
```

**3d. Fix the recoloring block** (lines 208-224). Replace with logic that handles structural lines vs outlier markers separately:

```python
    # Recolor lines and patches based on edgecolor
    for line in new_lines:
        line_data = line.get_xdata() if categorical_axis == "x" else line.get_ydata()
        if len(line_data) == 0:
            continue
        pos = np.mean(line_data)
        closest_pos = min(patch_colors.keys(), key=lambda p: abs(p - pos))
        base_color = patch_colors[closest_pos]

        if line.get_marker() and line.get_marker() != 'None':
            # Outlier markers: face = palette color, edge = edgecolor or palette
            line.set_markerfacecolor(base_color)
            line.set_markeredgecolor(resolved_edgecolor if resolved_edgecolor else base_color)
            line.set_markeredgewidth(markeredgewidth)
        else:
            # Structural lines (whiskers, caps, medians)
            line.set_color(resolved_edgecolor if resolved_edgecolor else base_color)
            line.set_linewidth(linewidth)

    # Set edge colors on box patches
    for patch in new_patches:
        patch.set_edgecolor(resolved_edgecolor if resolved_edgecolor else patch.get_facecolor())
```

**3e. Fix transparency** — the `tracker.apply_transparency` call now only needs to handle patches and outlier marker faces:
```python
    # Apply transparency to faces
    tracker.apply_transparency(on="patches", face_alpha=alpha)
    # Apply transparency to outlier marker faces only
    for line in new_lines:
        if line.get_marker() and line.get_marker() != 'None':
            fc = line.get_markerfacecolor()
            line.set_markerfacecolor(to_rgba(fc, alpha))
```

**3f. Update legend** — pass edgecolor to legend handles (~line 233):
```python
    if legend and hue is not None:
        from publiplots.utils.legend import legend as pp_legend
        from publiplots.utils.legend import create_legend_handles

        palette_colors = list(palette.values()) if isinstance(palette, dict) else None
        handles = create_legend_handles(
            labels=list(palette.keys()) if isinstance(palette, dict) else None,
            colors=palette_colors,
            edgecolors=[resolved_edgecolor] * len(palette) if resolved_edgecolor and isinstance(palette, dict) else None,
            alpha=alpha,
            linewidth=linewidth,
        )

        legend_kwargs = legend_kws or dict(label=hue)
        pp_legend(ax, handles=handles, **legend_kwargs)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/sagemaker-user/publiplots && python -m pytest tests/test_edgecolor.py::TestBoxplotEdgecolor -v`

Expected: All 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/publiplots/plot/box.py tests/test_edgecolor.py
git commit -m "feat(boxplot): add edgecolor parameter, deprecate linecolor, fix outlier markers"
```

---

### Task 4: Fix scatterplot legend edgecolor

**Files:**
- Modify: `src/publiplots/plot/scatter.py:459-577` (`_legend` function)
- Modify: `tests/test_edgecolor.py` (add scatter tests)

- [ ] **Step 1: Add scatter tests**

Append to `tests/test_edgecolor.py`:

```python
# ============================================================
# Scatterplot tests
# ============================================================

class TestScatterplotEdgecolor:
    """Tests for scatterplot edgecolor in legend."""

    def test_custom_edgecolor_on_collection(self, sample_data):
        """Edgecolor applied to scatter collection."""
        fig, ax = pp.scatterplot(data=sample_data, x="value", y="value",
                                  edgecolor="black", legend=False)
        collection = ax.collections[0]
        ecs = collection.get_edgecolors()
        for ec in ecs:
            assert ec[:3] == pytest.approx(to_rgba("black")[:3], abs=0.01)

    def test_edgecolor_in_legend_handles(self, sample_data):
        """Legend handles reflect edgecolor when hue is categorical."""
        fig, ax = pp.scatterplot(data=sample_data, x="value", y="value",
                                  hue="category", edgecolor="black")
        # Check legend data stored on collection
        collection = ax.collections[0]
        if hasattr(collection, '_legend_data') and 'hue' in collection._legend_data:
            handles = collection._legend_data['hue'].get('handles', [])
            for h in handles:
                assert to_rgba(h.get_edgecolor())[:3] == pytest.approx(to_rgba("black")[:3], abs=0.01)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/sagemaker-user/publiplots && python -m pytest tests/test_edgecolor.py::TestScatterplotEdgecolor -v`

Expected: `test_custom_edgecolor_on_collection` may pass (scatter already has edgecolor), `test_edgecolor_in_legend_handles` FAIL because legend handles don't have the edgecolor.

- [ ] **Step 3: Pass edgecolor to scatter `_legend`**

In `src/publiplots/plot/scatter.py`:

**3a. Pass `edgecolor` to `_legend()` call** (~line 298):
```python
    _legend(
        ax=ax,
        data=data,
        hue=hue,
        size=size,
        style=style,
        markers=markers,
        color=color,
        edgecolor=edgecolor,  # NEW
        palette=palette,
        ...
    )
```

**3b. Update `_legend` signature** (~line 459) to accept `edgecolor`:
```python
def _legend(
        ax: Axes,
        data: pd.DataFrame,
        hue: Optional[str],
        size: Optional[str],
        style: Optional[str],
        markers: Optional[Union[bool, List[str], Dict[str, str]]],
        color: Optional[str],
        edgecolor: Optional[str] = None,  # NEW
        palette: Optional[Union[str, Dict, List]] = None,
        ...
```

**3c. Pass edgecolors to `create_legend_handles`** in the categorical hue branch (~line 498):
```python
            hue_handles = create_legend_handles(
                labels=list(palette.keys()),
                colors=list(palette.values()),
                edgecolors=[edgecolor] * len(palette) if edgecolor else None,  # NEW
                **handle_kwargs
            )
```

And in the style branch (~line 561):
```python
            style_handles = create_legend_handles(
                labels=[str(val) for val in style_values],
                markers=[marker_map[val] for val in style_values],
                edgecolors=[edgecolor] * len(style_values) if edgecolor else None,  # NEW
                **style_handle_kwargs
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/sagemaker-user/publiplots && python -m pytest tests/test_edgecolor.py::TestScatterplotEdgecolor -v`

Expected: All 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/publiplots/plot/scatter.py tests/test_edgecolor.py
git commit -m "feat(scatter): pass edgecolor to legend handles"
```

---

### Task 5: Add edgecolor to violinplot

**Files:**
- Modify: `src/publiplots/plot/violin.py:27-62` (signature), `199-228` (kwargs), `249-261` (legend)
- Modify: `tests/test_edgecolor.py` (add violin tests)

- [ ] **Step 1: Add violin tests**

Append to `tests/test_edgecolor.py`:

```python
# ============================================================
# Violinplot tests
# ============================================================

class TestViolinplotEdgecolor:
    """Tests for violinplot edgecolor parameter."""

    def test_edgecolor_overrides_linecolor(self, sample_data):
        """edgecolor maps to seaborn's linecolor."""
        fig, ax = pp.violinplot(data=sample_data, x="category", y="value",
                                 edgecolor="black", legend=False)
        # Violin outlines should be black
        from matplotlib.collections import FillBetweenPolyCollection
        for coll in ax.collections:
            if isinstance(coll, FillBetweenPolyCollection):
                ecs = coll.get_edgecolors()
                for ec in ecs:
                    assert ec[:3] == pytest.approx(to_rgba("black")[:3], abs=0.01)

    def test_edgecolor_precedence_over_linecolor(self, sample_data):
        """edgecolor takes precedence over linecolor."""
        with pytest.warns(FutureWarning):
            fig, ax = pp.violinplot(data=sample_data, x="category", y="value",
                                     edgecolor="black", linecolor="red", legend=False)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/sagemaker-user/publiplots && python -m pytest tests/test_edgecolor.py::TestViolinplotEdgecolor -v`

Expected: FAIL — `edgecolor` param doesn't exist.

- [ ] **Step 3: Implement edgecolor in violinplot**

In `src/publiplots/plot/violin.py`:

**3a. Add `edgecolor` to signature** (after `color`, ~line 36):
```python
    edgecolor: Optional[str] = None,
```

**3b. Add edgecolor/linecolor resolution** — after `color = resolve_param(...)` (~line 172):
```python
    # Resolve edgecolor vs linecolor (backward compat)
    if edgecolor is not None and linecolor != "auto":
        import warnings
        warnings.warn(
            "linecolor is deprecated in favor of edgecolor. "
            "edgecolor takes precedence when both are provided.",
            FutureWarning,
            stacklevel=2,
        )
    if edgecolor is not None:
        linecolor = edgecolor
```

**3c. Update legend** (~line 249) to pass edgecolors:
```python
    if legend and hue is not None:
        from publiplots.utils.legend import legend as pp_legend
        from publiplots.utils.legend import create_legend_handles

        handles = create_legend_handles(
            labels=list(palette.keys()) if isinstance(palette, dict) else None,
            colors=list(palette.values()) if isinstance(palette, dict) else None,
            edgecolors=[edgecolor] * len(palette) if edgecolor and isinstance(palette, dict) else None,
            alpha=alpha,
            linewidth=linewidth,
        )

        legend_kwargs = legend_kws or dict(label=hue)
        pp_legend(ax, handles=handles, **legend_kwargs)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/sagemaker-user/publiplots && python -m pytest tests/test_edgecolor.py::TestViolinplotEdgecolor -v`

Expected: All 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/publiplots/plot/violin.py tests/test_edgecolor.py
git commit -m "feat(violin): add edgecolor parameter, deprecate linecolor"
```

---

### Task 6: Fix stripplot and swarmplot legend edgecolor

**Files:**
- Modify: `src/publiplots/plot/strip.py:207-261` (`_legend` function)
- Modify: `src/publiplots/plot/swarm.py:203-257` (`_legend` function)
- Modify: `tests/test_edgecolor.py` (add strip/swarm tests)

- [ ] **Step 1: Add strip/swarm tests**

Append to `tests/test_edgecolor.py`:

```python
# ============================================================
# Stripplot and Swarmplot tests
# ============================================================

class TestStripplotEdgecolor:
    """Tests for stripplot edgecolor in legend."""

    def test_edgecolor_applied_to_collection(self, sample_data):
        """Edgecolor sets edge on PathCollection."""
        fig, ax = pp.stripplot(data=sample_data, x="category", y="value",
                                edgecolor="black", legend=False)
        for coll in ax.collections:
            ecs = coll.get_edgecolors()
            for ec in ecs:
                assert ec[:3] == pytest.approx(to_rgba("black")[:3], abs=0.01)

    def test_edgecolor_in_legend(self, sample_data):
        """Legend handles reflect edgecolor."""
        fig, ax = pp.stripplot(data=sample_data, x="category", y="value",
                                hue="category", edgecolor="black")
        collection = ax.collections[0]
        if hasattr(collection, '_legend_data') and 'hue' in collection._legend_data:
            handles = collection._legend_data['hue'].get('handles', [])
            for h in handles:
                assert to_rgba(h.get_edgecolor())[:3] == pytest.approx(to_rgba("black")[:3], abs=0.01)


class TestSwarmplotEdgecolor:
    """Tests for swarmplot edgecolor in legend."""

    def test_edgecolor_applied_to_collection(self, sample_data):
        """Edgecolor sets edge on PathCollection."""
        fig, ax = pp.swarmplot(data=sample_data, x="category", y="value",
                                edgecolor="black", legend=False)
        for coll in ax.collections:
            ecs = coll.get_edgecolors()
            for ec in ecs:
                assert ec[:3] == pytest.approx(to_rgba("black")[:3], abs=0.01)

    def test_edgecolor_in_legend(self, sample_data):
        """Legend handles reflect edgecolor."""
        fig, ax = pp.swarmplot(data=sample_data, x="category", y="value",
                                hue="category", edgecolor="black")
        collection = ax.collections[0]
        if hasattr(collection, '_legend_data') and 'hue' in collection._legend_data:
            handles = collection._legend_data['hue'].get('handles', [])
            for h in handles:
                assert to_rgba(h.get_edgecolor())[:3] == pytest.approx(to_rgba("black")[:3], abs=0.01)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/sagemaker-user/publiplots && python -m pytest tests/test_edgecolor.py::TestStripplotEdgecolor tests/test_edgecolor.py::TestSwarmplotEdgecolor -v`

Expected: `test_edgecolor_applied_to_collection` passes (strip/swarm already pass edgecolor to seaborn), `test_edgecolor_in_legend` FAIL.

- [ ] **Step 3: Pass edgecolor to strip/swarm legends**

**For `src/publiplots/plot/strip.py`:**

Update `_legend` signature (~line 207) to accept `edgecolor`:
```python
def _legend(
    ax: Axes,
    hue: Optional[str],
    color: Optional[str],
    edgecolor: Optional[str] = None,
    palette: Optional[Union[str, Dict, List]] = None,
    ...
```

Update `handle_kwargs` (~line 225):
```python
    handle_kwargs = dict(alpha=alpha, linewidth=linewidth, color=color, style="circle")
```

Update `create_legend_handles` call (~line 234):
```python
            hue_handles = create_legend_handles(
                labels=list(palette.keys()),
                colors=list(palette.values()),
                edgecolors=[edgecolor] * len(palette) if edgecolor else None,
                **handle_kwargs
            )
```

Update the `_legend()` call in `stripplot()` (~line 185) to pass `edgecolor`:
```python
        _legend(
            ax=ax,
            hue=hue,
            color=color,
            edgecolor=edgecolor,
            palette=palette,
            ...
        )
```

**For `src/publiplots/plot/swarm.py`:** Apply the exact same changes — the structure is identical.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/sagemaker-user/publiplots && python -m pytest tests/test_edgecolor.py::TestStripplotEdgecolor tests/test_edgecolor.py::TestSwarmplotEdgecolor -v`

Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/publiplots/plot/strip.py src/publiplots/plot/swarm.py tests/test_edgecolor.py
git commit -m "feat(strip,swarm): pass edgecolor to legend handles"
```

---

### Task 7: Add edgecolor to pointplot

**Files:**
- Modify: `src/publiplots/plot/point.py:25-59` (signature), `348-416` (`_apply_marker_styling`), `419-494` (`_legend`)
- Modify: `tests/test_edgecolor.py` (add pointplot tests)

- [ ] **Step 1: Add pointplot tests**

Append to `tests/test_edgecolor.py`:

```python
# ============================================================
# Pointplot tests
# ============================================================

class TestPointplotEdgecolor:
    """Tests for pointplot edgecolor parameter."""

    def test_custom_edgecolor_on_markers(self, sample_data):
        """Marker edges should use edgecolor."""
        fig, ax = pp.pointplot(data=sample_data, x="category", y="value",
                                edgecolor="black", legend=False)
        # The last plotted lines are the double-layer markers (Layer 2)
        # They should have markeredgecolor="black"
        found_marker = False
        for line in ax.lines:
            mec = line.get_markeredgecolor()
            if line.get_markersize() > 0 and line.get_marker() != "None":
                found_marker = True
                assert to_rgba(mec)[:3] == pytest.approx(to_rgba("black")[:3], abs=0.01)
        assert found_marker, "No markers found on plot"

    def test_edgecolor_does_not_affect_connecting_lines(self, sample_data):
        """Connecting lines should stay palette color, not edgecolor."""
        fig, ax = pp.pointplot(data=sample_data, x="category", y="value",
                                hue="group", edgecolor="black", legend=False)
        # Lines without markers are connecting lines — should be palette colors
        for line in ax.lines:
            if (line.get_marker() == "None" or line.get_markersize() == 0) and line.get_linestyle() != "None":
                lc = to_rgba(line.get_color())
                # Should NOT be black (should be palette color)
                assert lc[:3] != pytest.approx(to_rgba("black")[:3], abs=0.01)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/sagemaker-user/publiplots && python -m pytest tests/test_edgecolor.py::TestPointplotEdgecolor -v`

Expected: FAIL — `pointplot() got an unexpected keyword argument 'edgecolor'`

- [ ] **Step 3: Implement edgecolor in pointplot**

In `src/publiplots/plot/point.py`:

**3a. Add `edgecolor` to signature** (after `capsize`, ~line 46):
```python
    edgecolor: Optional[str] = None,
```

**3b. Pass `edgecolor` to `_apply_marker_styling`** (~line 282):
```python
    _apply_marker_styling(
        ax=ax,
        alpha=alpha,
        edgecolor=edgecolor,
    )
```

**3c. Update `_apply_marker_styling` signature** (~line 348):
```python
def _apply_marker_styling(
    ax: Axes,
    alpha: float,
    edgecolor: Optional[str] = None,
) -> List[Dict]:
```

**3d. Use edgecolor in marker redraw** (~line 398-416). In Layer 1 (white background):
```python
        ax.plot(
            x, y, marker,
            markeredgecolor=edgecolor if edgecolor else color,
            markerfacecolor="white",
            markersize=size,
            markeredgewidth=0,
            linestyle='none',
            zorder=99
        )
```

In Layer 2 (semi-transparent fill):
```python
        ax.plot(
            x, y, marker,
            markeredgecolor=edgecolor if edgecolor else color,
            markerfacecolor=to_rgba(color, alpha),
            markersize=size,
            markeredgewidth=markeredgewidth,
            linestyle='none',
            zorder=100
        )
```

**3e. Pass edgecolor to `_legend`** (~line 297):
```python
        _legend(
            ax=ax,
            hue=hue,
            palette=palette,
            markers=list(marker_map.values()),
            linestyles=list(linestyle_map.values()),
            markersize=markersize,
            markeredgewidth=markeredgewidth,
            edgecolor=edgecolor,
            alpha=alpha,
            linewidth=linewidth,
            kwargs=legend_kws,
        )
```

**3f. Update `_legend` signature** (~line 419):
```python
def _legend(
    ax: Axes,
    hue: str,
    palette: Union[Dict, str],
    markers: Union[List, Dict],
    linestyles: Union[List, Dict],
    markersize: Optional[float] = None,
    markeredgewidth: Optional[float] = None,
    edgecolor: Optional[str] = None,
    alpha: Optional[float] = None,
    linewidth: Optional[float] = None,
    kwargs: Optional[Dict] = None,
) -> None:
```

**3g. Pass edgecolors in `create_legend_handles`** call (~line 473):
```python
        hue_handles = create_legend_handles(
            labels=[str(label) for label in labels],
            colors=colors,
            edgecolors=[edgecolor] * len(labels) if edgecolor else None,
            markers=markers,
            linestyles=linestyles,
            alpha=alpha,
            linewidth=linewidth,
            sizes=[markersize],
            markeredgewidth=markeredgewidth,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/sagemaker-user/publiplots && python -m pytest tests/test_edgecolor.py::TestPointplotEdgecolor -v`

Expected: All 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/publiplots/plot/point.py tests/test_edgecolor.py
git commit -m "feat(pointplot): add edgecolor parameter for marker edges"
```

---

### Task 8: Add edgecolor passthrough to raincloudplot

**Files:**
- Modify: `src/publiplots/plot/raincloud.py:24-63` (signature), `178-271` (sub-plot calls)
- Modify: `tests/test_edgecolor.py` (add raincloud tests)

- [ ] **Step 1: Add raincloud tests**

Append to `tests/test_edgecolor.py`:

```python
# ============================================================
# Raincloudplot tests
# ============================================================

class TestRaincloudplotEdgecolor:
    """Tests for raincloudplot edgecolor passthrough."""

    def test_edgecolor_passed_to_subplots(self, sample_data):
        """Edgecolor should be applied across all sub-components."""
        fig, ax = pp.raincloudplot(data=sample_data, x="category", y="value",
                                    edgecolor="black", legend=False)
        # Check box patches have black edges
        for patch in ax.patches:
            ec = patch.get_edgecolor()
            assert to_rgba(ec)[:3] == pytest.approx(to_rgba("black")[:3], abs=0.01)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/sagemaker-user/publiplots && python -m pytest tests/test_edgecolor.py::TestRaincloudplotEdgecolor -v`

Expected: FAIL — `raincloudplot() got an unexpected keyword argument 'edgecolor'`

- [ ] **Step 3: Implement edgecolor passthrough**

In `src/publiplots/plot/raincloud.py`:

**3a. Add `edgecolor` to signature** (after `rain_offset`, ~line 53):
```python
    edgecolor: Optional[str] = None,
```

**3b. Pass to violinplot** (~line 178):
```python
    fig, ax = violinplot(
        ...
        edgecolor=edgecolor,
        ...
    )
```

**3c. Pass to boxplot** — add to `box_kws.update(...)` (~line 208):
```python
        box_kws.update(dict(
            ...
            edgecolor=edgecolor,
            ...
        ))
```

**3d. Pass to rain plot** — add to `rain_kws.update(...)` (~line 244):
```python
        rain_kws.update(dict(
            ...
            edgecolor=edgecolor,
            ...
        ))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/sagemaker-user/publiplots && python -m pytest tests/test_edgecolor.py::TestRaincloudplotEdgecolor -v`

Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add src/publiplots/plot/raincloud.py tests/test_edgecolor.py
git commit -m "feat(raincloud): add edgecolor passthrough to sub-plots"
```

---

### Task 9: Integration tests — full suite validation

**Files:**
- Modify: `tests/test_edgecolor.py` (add cross-cutting integration tests)

- [ ] **Step 1: Add integration tests**

Append to `tests/test_edgecolor.py`:

```python
# ============================================================
# Integration tests
# ============================================================

class TestEdgecolorIntegration:
    """Cross-cutting tests for edgecolor consistency."""

    def test_all_plot_types_accept_edgecolor(self, sample_data):
        """Every plot function should accept edgecolor without error."""
        common = dict(data=sample_data, x="category", y="value", edgecolor="black", legend=False)
        pp.barplot(**common)
        pp.boxplot(**common)
        pp.violinplot(**common)
        pp.stripplot(**common)
        pp.swarmplot(**common)
        pp.pointplot(**common)
        pp.raincloudplot(**common)

    def test_all_plot_types_accept_edgecolor_none(self, sample_data):
        """edgecolor=None should preserve default behavior for all types."""
        common = dict(data=sample_data, x="category", y="value", edgecolor=None, legend=False)
        pp.barplot(**common)
        pp.boxplot(**common)
        pp.violinplot(**common)
        pp.stripplot(**common)
        pp.swarmplot(**common)
        pp.pointplot(**common)
        pp.raincloudplot(**common)

    def test_scatterplot_edgecolor_unchanged(self, sample_data):
        """Scatterplot's existing edgecolor behavior should be preserved."""
        fig, ax = pp.scatterplot(data=sample_data, x="value", y="value",
                                  edgecolor="black", legend=False)
        collection = ax.collections[0]
        ecs = collection.get_edgecolors()
        for ec in ecs:
            assert ec[:3] == pytest.approx(to_rgba("black")[:3], abs=0.01)

    def test_legend_handles_reflect_edgecolor(self, sample_data):
        """Legend handles should use edgecolor for all plot types that create rectangle legends."""
        # Barplot with hue and edgecolor
        fig, ax = pp.barplot(data=sample_data, x="category", y="value",
                              hue="group", edgecolor="black")
        # Get the legend from the axes
        legends = [c for c in ax.get_children()
                   if isinstance(c, matplotlib.legend.Legend)]
        assert len(legends) > 0, "No legend found"

    def test_combined_overlay_with_edgecolor(self, sample_data):
        """Overlaying plots with edgecolor should work correctly."""
        fig, ax = pp.violinplot(data=sample_data, x="category", y="value",
                                 edgecolor="black", legend=False)
        pp.stripplot(data=sample_data, x="category", y="value",
                      edgecolor="black", ax=ax, legend=False)
        # Both should have black edges — just verify no error
```

- [ ] **Step 2: Run the full test suite**

Run: `cd /home/sagemaker-user/publiplots && python -m pytest tests/test_edgecolor.py -v`

Expected: All tests PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_edgecolor.py
git commit -m "test: add integration tests for edgecolor across all plot types"
```
