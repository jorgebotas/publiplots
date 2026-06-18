# Vertical 2-way Venn Diagram Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an `orientation="horizontal"|"vertical"` parameter to `pp.venn` so the 2-way Venn diagram can render with one circle stacked on top of the other.

**Architecture:** Vertical is computed as a 90°-clockwise rotation `(x, y) → (y, -x)` of the existing horizontal 2-way geometry (circle centers + intersection-label positions), with set-name labels authored directly for an "outer ends" placement (Set A above the top circle, Set B below the bottom circle). Scope is 2-way only; `orientation="vertical"` with 3+ sets raises `ValueError`. The horizontal default is byte-for-byte unchanged.

**Tech Stack:** Python, NumPy, Matplotlib, pytest. Spec: `docs/superpowers/specs/2026-06-18-venn-vertical-orientation-design.md`.

---

## File Structure

- **Modify** `src/publiplots/plot/venn/geometry.py` — add `orientation` to `compute_2way_geometry` (the rotation + vertical set-label logic) and thread it through `get_geometry`.
- **Modify** `src/publiplots/plot/venn/diagram.py` — add `orientation` param to `venn`, validate it, thread into `get_geometry`, update docstring.
- **Create** `tests/test_venn_orientation.py` — unit tests for vertical geometry, rotation invariants, validation errors, backward-compat, and the integration path through `pp.venn`.
- **Modify** `examples/plots/plot_17_venn_diagrams.py` — add a vertical-orientation example cell.

Note: existing venn behavior tests live in `tests/test_drop_figsize.py` (figsize/ax/layout). New orientation tests go in a dedicated file to keep responsibilities separate.

---

### Task 1: Vertical geometry in `compute_2way_geometry`

**Files:**
- Modify: `src/publiplots/plot/venn/geometry.py` (function `compute_2way_geometry`, lines 74-116)
- Test: `tests/test_venn_orientation.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_venn_orientation.py`:

```python
"""Tests for the orientation parameter on the 2-way Venn diagram."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import publiplots as pp
from publiplots.plot.venn.geometry import compute_2way_geometry, get_geometry


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def test_vertical_stacks_first_set_on_top():
    circles, labels, set_labels = compute_2way_geometry(orientation="vertical")
    # Both circle centers lie on the vertical axis (x == 0)
    assert circles[0].x_offset == pytest.approx(0.0)
    assert circles[1].x_offset == pytest.approx(0.0)
    # Set A (index 0) sits above Set B (index 1)
    assert circles[0].y_offset > circles[1].y_offset


def test_vertical_intersection_labels():
    _, labels, _ = compute_2way_geometry(orientation="vertical")
    # "10" (only A) above origin, "01" (only B) below, "11" (both) at origin
    assert labels["10"][0] == pytest.approx(0.0)
    assert labels["01"][0] == pytest.approx(0.0)
    assert labels["10"][1] > 0
    assert labels["01"][1] < 0
    assert labels["11"] == pytest.approx((0.0, 0.0))


def test_vertical_set_labels_at_outer_ends():
    circles, _, set_labels = compute_2way_geometry(orientation="vertical")
    # Set A label above the top circle, Set B label below the bottom circle
    assert set_labels[0][0] == pytest.approx(0.0)
    assert set_labels[1][0] == pytest.approx(0.0)
    assert set_labels[0][1] > circles[0].y_offset
    assert set_labels[1][1] < circles[1].y_offset


def test_vertical_is_rotation_of_horizontal():
    h_circles, h_labels, _ = compute_2way_geometry(orientation="horizontal")
    v_circles, v_labels, _ = compute_2way_geometry(orientation="vertical")
    # 90-deg clockwise rotation: (x, y) -> (y, -x)
    for hc, vc in zip(h_circles, v_circles):
        assert vc.x_offset == pytest.approx(hc.y_offset)
        assert vc.y_offset == pytest.approx(-hc.x_offset)
        assert vc.radius_a == pytest.approx(hc.radius_a)
        assert vc.radius_b == pytest.approx(hc.radius_b)
    for key in h_labels:
        hx, hy = h_labels[key]
        vx, vy = v_labels[key]
        assert (vx, vy) == pytest.approx((hy, -hx))


def test_horizontal_default_unchanged():
    # Default arg and explicit "horizontal" must be identical
    default = compute_2way_geometry()
    explicit = compute_2way_geometry(orientation="horizontal")
    assert [c.x_offset for c in default[0]] == [c.x_offset for c in explicit[0]]
    assert default[1] == explicit[1]
    assert default[2] == explicit[2]
    # And matches the known horizontal centers
    circles, _, _ = default
    assert circles[0].x_offset < 0  # Set A on the left
    assert circles[1].x_offset > 0  # Set B on the right
    assert circles[0].y_offset == pytest.approx(0.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/sagemaker-user/publiplots && uv run pytest tests/test_venn_orientation.py -v`
Expected: FAIL — `compute_2way_geometry()` raises `TypeError: unexpected keyword argument 'orientation'`.

- [ ] **Step 3: Implement the vertical geometry**

In `src/publiplots/plot/venn/geometry.py`, replace the `compute_2way_geometry` signature and body. The current function (lines 74-116) ends with `return circles, label_positions, set_label_positions`. Change the signature and insert the vertical branch immediately before that `return`:

```python
def compute_2way_geometry(
    overlap_size: float = 0.5,
    orientation: str = "horizontal",
) -> Tuple[List[Circle], Dict[str, Tuple[float, float]], List[Tuple[float, float]]]:
```

Keep the existing horizontal body that builds `a_radius`, `b_radius`, `x_dist`,
`circles`, `label_positions`, and `set_label_positions` exactly as-is. Then,
just before the final `return`, add:

```python
    if orientation == "vertical":
        # Vertical = 90-deg clockwise rotation of the horizontal layout:
        # (x, y) -> (y, -x). Set A (index 0) lands on top, Set B on the bottom.
        circles = [
            Circle(
                x_offset=c.y_offset,
                y_offset=-c.x_offset,
                radius_a=c.radius_a,
                radius_b=c.radius_b,
                theta_offset=c.theta_offset,
            )
            for c in circles
        ]
        label_positions = {
            key: (y, -x) for key, (x, y) in label_positions.items()
        }
        # Set-name labels are NOT rotated: place them at the outer ends so they
        # don't collide with the overlap. Set A above the top circle, Set B
        # below the bottom circle.
        set_label_positions = [
            (0.0, x_dist + a_radius + 0.3),    # Set A (top)
            (0.0, -x_dist - b_radius - 0.3),   # Set B (bottom)
        ]
    elif orientation != "horizontal":
        raise ValueError(
            f"orientation must be 'horizontal' or 'vertical', got {orientation!r}"
        )

    return circles, label_positions, set_label_positions
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/sagemaker-user/publiplots && uv run pytest tests/test_venn_orientation.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
cd /home/sagemaker-user/publiplots
git add src/publiplots/plot/venn/geometry.py tests/test_venn_orientation.py
git commit -m "feat(venn): vertical orientation in compute_2way_geometry

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Thread `orientation` through `get_geometry`

**Files:**
- Modify: `src/publiplots/plot/venn/geometry.py` (function `get_geometry`, lines 393-425)
- Test: `tests/test_venn_orientation.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_venn_orientation.py`:

```python
def test_get_geometry_threads_orientation_for_2way():
    direct = compute_2way_geometry(orientation="vertical")
    via = get_geometry(2, orientation="vertical")
    # Same circle centers
    assert [c.y_offset for c in via[0]] == [c.y_offset for c in direct[0]]
    assert via[1] == direct[1]


def test_get_geometry_3way_unaffected_by_default_orientation():
    # New signature must not change 3/4/5-way output for the default
    circles, labels, set_labels = get_geometry(3)
    assert len(circles) == 3
    assert "111" in labels
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/sagemaker-user/publiplots && uv run pytest tests/test_venn_orientation.py::test_get_geometry_threads_orientation_for_2way -v`
Expected: FAIL — `get_geometry()` raises `TypeError: unexpected keyword argument 'orientation'`.

- [ ] **Step 3: Implement the threading**

In `src/publiplots/plot/venn/geometry.py`, change the `get_geometry` signature
and the 2-way branch. Current signature (line 393):

```python
def get_geometry(n_sets: int) -> Tuple[List[Circle], Dict[str, Tuple[float, float]], List[Tuple[float, float]]]:
```

becomes:

```python
def get_geometry(
    n_sets: int,
    orientation: str = "horizontal",
) -> Tuple[List[Circle], Dict[str, Tuple[float, float]], List[Tuple[float, float]]]:
```

And change the 2-way branch (currently `return compute_2way_geometry()`):

```python
    if n_sets == 2:
        return compute_2way_geometry(orientation=orientation)
```

Leave the 3/4/5-way branches untouched — they ignore `orientation` (and are
never reached with a non-default orientation because `venn` validates first;
see Task 3).

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/sagemaker-user/publiplots && uv run pytest tests/test_venn_orientation.py -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
cd /home/sagemaker-user/publiplots
git add src/publiplots/plot/venn/geometry.py tests/test_venn_orientation.py
git commit -m "feat(venn): thread orientation through get_geometry

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Add `orientation` param + validation to `pp.venn`

**Files:**
- Modify: `src/publiplots/plot/venn/diagram.py` (function `venn`, lines 199-332)
- Test: `tests/test_venn_orientation.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_venn_orientation.py`:

```python
def test_venn_vertical_returns_axes_with_stacked_circles():
    from matplotlib.patches import Ellipse
    ax = pp.venn(sets=[{1, 2, 3}, {3, 4, 5}], orientation="vertical")
    ellipses = [p for p in ax.patches if isinstance(p, Ellipse)]
    assert len(ellipses) == 2
    cy = [e.center[1] for e in ellipses]
    cx = [e.center[0] for e in ellipses]
    # Both on the vertical axis, first set above second
    assert cx[0] == pytest.approx(0.0)
    assert cx[1] == pytest.approx(0.0)
    assert cy[0] > cy[1]


def test_venn_vertical_with_three_sets_raises():
    with pytest.raises(ValueError, match="2-way"):
        pp.venn(sets=[{1}, {2}, {3}], orientation="vertical")


def test_venn_invalid_orientation_raises():
    with pytest.raises(ValueError, match="horizontal.*vertical|orientation"):
        pp.venn(sets=[{1, 2}, {2, 3}], orientation="diagonal")


def test_venn_horizontal_default_unchanged():
    from matplotlib.patches import Ellipse
    ax = pp.venn(sets=[{1, 2, 3}, {3, 4, 5}])
    ellipses = [p for p in ax.patches if isinstance(p, Ellipse)]
    cy = [e.center[1] for e in ellipses]
    # Horizontal: both centers on the y == 0 axis
    assert cy[0] == pytest.approx(0.0)
    assert cy[1] == pytest.approx(0.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/sagemaker-user/publiplots && uv run pytest tests/test_venn_orientation.py::test_venn_vertical_returns_axes_with_stacked_circles -v`
Expected: FAIL — `venn()` raises `TypeError: unexpected keyword argument 'orientation'`.

- [ ] **Step 3: Implement the param + validation + threading**

In `src/publiplots/plot/venn/diagram.py`:

(a) Add the parameter to the `venn` signature. Current signature (lines 199-207)
ends with `color_labels: bool = True,`. Add after it:

```python
    color_labels: bool = True,
    orientation: str = "horizontal",
```

(b) After the existing `n_sets` validation block (the one ending at line 300
with the "Venn diagram supports 2 to 5 sets" `ValueError`), add orientation
validation:

```python
    # Validate orientation
    if orientation not in ("horizontal", "vertical"):
        raise ValueError(
            f"orientation must be 'horizontal' or 'vertical', got {orientation!r}"
        )
    if orientation == "vertical" and n_sets != 2:
        raise ValueError(
            "orientation='vertical' is only supported for 2-way Venn diagrams."
        )
```

(c) Thread `orientation` into the internal `_venn` call. The current call
(lines 323-330) is:

```python
    ax = _venn(
        petal_labels=petal_labels,
        dataset_labels=labels,
        colors=colors,
        alpha=alpha,
        ax=ax,
        color_labels=color_labels,
    )
```

Add `orientation=orientation,` before the closing paren.

(d) Update `_venn` to accept and forward `orientation`. Current `_venn`
signature (lines 63-71) ends with `color_labels: bool = True,`. Add:

```python
    color_labels: bool = True,
    orientation: str = "horizontal",
```

Then change the geometry call inside `_venn` (line 85) from:

```python
    circles, label_positions, set_label_positions = get_geometry(n_sets)
```

to:

```python
    circles, label_positions, set_label_positions = get_geometry(n_sets, orientation=orientation)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/sagemaker-user/publiplots && uv run pytest tests/test_venn_orientation.py -v`
Expected: PASS (11 tests).

- [ ] **Step 5: Commit**

```bash
cd /home/sagemaker-user/publiplots
git add src/publiplots/plot/venn/diagram.py tests/test_venn_orientation.py
git commit -m "feat(venn): orientation param on pp.venn with validation

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Document the `orientation` parameter

**Files:**
- Modify: `src/publiplots/plot/venn/diagram.py` (docstring of `venn`, lines 208-283)

- [ ] **Step 1: Add the parameter to the docstring**

In the Parameters section of the `venn` docstring, after the `color_labels`
entry (lines 240-241), add:

```python
    orientation : str, default='horizontal'
        Layout for a 2-way Venn diagram. ``'horizontal'`` places the two
        circles side-by-side; ``'vertical'`` stacks them with the first set
        on top and the second below. Only valid for 2 sets — passing
        ``'vertical'`` with 3 or more sets raises ``ValueError``.
```

- [ ] **Step 2: Add a vertical example to the docstring**

In the Examples section, after the simple 2-way example (lines 258-262), add:

```python
    Vertical 2-way Venn (circles stacked):

    >>> ax = pp.venn([set1, set2], labels=['A', 'B'], orientation='vertical')
```

- [ ] **Step 3: Verify the module still imports**

Run: `cd /home/sagemaker-user/publiplots && uv run python -c "import publiplots as pp; help(pp.venn)" | head -5`
Expected: prints the `venn` help header with no import error.

- [ ] **Step 4: Commit**

```bash
cd /home/sagemaker-user/publiplots
git add src/publiplots/plot/venn/diagram.py
git commit -m "docs(venn): document orientation parameter

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Add a vertical example to the gallery

**Files:**
- Modify: `examples/plots/plot_17_venn_diagrams.py` (after the 2-way percentage cell, around line 52)

- [ ] **Step 1: Add the example cell**

In `examples/plots/plot_17_venn_diagrams.py`, after the "2-Way Venn with
Percentage Format" cell (which ends with `pp.show()` near line 52), insert:

```python
# %%
# Vertical 2-Way Venn Diagram
# ---------------------------
# Stack the two circles vertically (first set on top) instead of
# side-by-side. Vertical orientation is supported for 2-way diagrams only.

set1 = set(range(1, 51))    # 1-50
set2 = set(range(30, 81))   # 30-80

# A tall, narrow axes suits the vertical stack. Compose pp.subplots and
# pass ax=.
fig, ax = pp.subplots(axes_size=(60, 100))
pp.venn(
    sets=[set1, set2],
    labels=['Set A', 'Set B'],
    colors=pp.color_palette('pastel', n_colors=2),
    orientation='vertical',
    ax=ax,
)
pp.show()
```

- [ ] **Step 2: Verify the example runs**

Run: `cd /home/sagemaker-user/publiplots && uv run python examples/plots/plot_17_venn_diagrams.py`
Expected: runs to completion with no exception (it builds all cells, including the new vertical one).

- [ ] **Step 3: Commit**

```bash
cd /home/sagemaker-user/publiplots
git add examples/plots/plot_17_venn_diagrams.py
git commit -m "docs(venn): add vertical orientation gallery example

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Visual verification (eyeball the rendered output)

**Files:**
- Temporary: `/tmp/venn_vertical_check.py`, `/tmp/venn_vertical.pdf`, `/tmp/venn_vertical.png`

Per the standing rule that visual features must be eyeballed before claiming
done — passing tests and review are necessary but not sufficient.

- [ ] **Step 1: Write a render script**

Create `/tmp/venn_vertical_check.py`:

```python
import matplotlib
matplotlib.use("Agg")
import publiplots as pp

set1 = set(range(1, 51))
set2 = set(range(30, 81))

fig, ax = pp.subplots(axes_size=(60, 100))
pp.venn(
    sets=[set1, set2],
    labels=['Set A', 'Set B'],
    colors=pp.color_palette('pastel', n_colors=2),
    orientation='vertical',
    ax=ax,
)
fig.savefig("/tmp/venn_vertical.pdf")
print("saved /tmp/venn_vertical.pdf")
```

- [ ] **Step 2: Render and rasterize**

Run:
```bash
cd /home/sagemaker-user/publiplots && uv run python /tmp/venn_vertical_check.py
pdftocairo -png -r 150 /tmp/venn_vertical.pdf /tmp/venn_vertical
```
Expected: produces `/tmp/venn_vertical-1.png`.

- [ ] **Step 3: Eyeball the PNG**

Use the Read tool on `/tmp/venn_vertical-1.png` and confirm:
- Two circles are stacked vertically (one above the other), not side-by-side.
- "Set A" label is above the top circle; "Set B" is below the bottom circle.
- The three region counts (only-A, both, only-B) sit in their correct regions:
  only-A near the top, both in the overlap middle, only-B near the bottom.
- Nothing overlaps or is clipped.

If anything is visually wrong, STOP and fix the geometry (Task 1) before
proceeding — do not claim completion on green tests alone.

- [ ] **Step 4: Clean up (no commit — temp files only)**

```bash
rm -f /tmp/venn_vertical_check.py /tmp/venn_vertical.pdf /tmp/venn_vertical-1.png
```

---

### Task 7: Full regression + skill-doc refresh

**Files:**
- Possibly modify: `skills/publiplots-guide/SKILL.md` (only if it carries venn's signature / Plots inventory)

- [ ] **Step 1: Run the full venn-related test suite**

Run: `cd /home/sagemaker-user/publiplots && uv run pytest tests/test_venn_orientation.py tests/test_drop_figsize.py -v`
Expected: all PASS (new orientation tests + existing venn figsize/ax/layout tests).

- [ ] **Step 2: Run the whole suite to check for regressions**

Run: `cd /home/sagemaker-user/publiplots && uv run pytest -q`
Expected: no new failures introduced by this change.

- [ ] **Step 3: Check whether the guide skill references venn's signature**

Run: `cd /home/sagemaker-user/publiplots && grep -n "venn" skills/publiplots-guide/SKILL.md`
- If the venn signature / parameter list is documented there, add `orientation`
  to it (per the release-skills-refresh convention).
- If venn is only mentioned by name with no signature, no edit is needed.

- [ ] **Step 4: Commit (only if SKILL.md changed)**

```bash
cd /home/sagemaker-user/publiplots
git add skills/publiplots-guide/SKILL.md
git commit -m "docs(venn): note orientation param in publiplots-guide skill

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Notes for the implementer

- Use `uv run` for all Python commands (not conda).
- The horizontal layout must stay byte-for-byte identical; `test_horizontal_default_unchanged` (Task 1) and `test_venn_horizontal_default_unchanged` (Task 3) guard this.
- `_get_set_label_alignments` in `diagram.py` needs no change — it derives text alignment from label-vs-center geometry, and the vertical set-label positions (directly above/below their centers) already resolve to `ha='center'` with the correct `va`.
- `get_coordinate_ranges` and `init_axes` need no change — `set_aspect("equal")` keeps circles round and frames the tall vertical bounding box automatically.
