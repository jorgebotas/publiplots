# Composer PR 3 Implementation Plan — Multi-row + PanelGrid + PanelText + canvas.align

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lift the single-row restriction; add `PanelGrid` (sub-grid of axes inside one panel) + `PanelText` (text-only panel); add `canvas.align(panels, edge, mode)` for explicit alignment overrides. After PR 3, the spec's worked-example figure (3-row, schematic + scatter + 1×3 grid + barplot+text) is producible end-to-end (modulo the `PanelImage` schematic, which is PR 5).

**Architecture:** PR 1's `add_row` synchronously created a `Figure` with one row. PR 3 introduces **lazy finalization**: each `add_row` stages a row record; the matplotlib `Figure` is created on first access (`canvas.figure`, `canvas[label]`, `canvas.savefig`). Each row gets its own internal `FigureLayout(1, N_row)`-equivalent and lays out independently; the canvas combines them vertically. `PanelGrid` sub-grids are realised as `pp.subplots`-style sub-axes within the panel's mm rect. `canvas.align` operates after finalization by shifting axes positions within their slots (slot mm-rects stay inviolate, per spec).

**Tech Stack:** Python ≥ 3.10, matplotlib, pytest. No new dependencies. Editable install at `/home/sagemaker-user/publiplots/`.

**Spec reference:** `docs/superpowers/specs/2026-05-14-composer-design.md` §6.d (PR 3 contract, line 630–633) + §"alignment" (line 422–456) + §"PanelGrid"/"PanelText" docs (lines 141–172).

---

## What's IN scope for PR 3

- **Multi-row layouts** — `add_row` callable multiple times. Rows stack top-to-bottom with `vpad` (configurable per-row). Each row may have a different number of panels (heterogeneous columns).
- **Lazy figure finalization** — the matplotlib `Figure` is created on first access (`.figure`, `__getitem__`, `.savefig`), not at `add_row` time. Rationale: a multi-row canvas can't size its figure until all rows are declared.
- **`PanelGrid(label, *, shape=(r, c), axes_size=(w, h), sharex, sharey, hspace, wspace, label_style)`** — sub-grid of axes inside a single panel. `canvas[label].axes` returns a 2D numpy array of `Axes`; `canvas[label].ax` raises `AttributeError` (use `.axes` for grids). PanelGrid's panel mm-rect is `axes_size * shape + (shape-1)*spacing`.
- **`PanelText(label, *, text, size=(w, h), text_kw, label_style)`** — text-only panel. Internally a hidden axes (axis off, no patch) with one `ax.text(0.5, 0.5, text, ha='center', va='center')`. Supports basic mathtext via the underlying matplotlib renderer.
- **`canvas.align(panels, *, edge, mode='axes', anchor=None)`** — explicit alignment. Edges: `'left'`, `'right'`, `'top'`, `'bottom'`, `'center_x'`, `'center_y'`, `'baseline'`. Modes: `'axes'` (data-rect edges) or `'tight'` (tightbbox edges). Multiple `align()` calls accumulate, applied at finalization time. **Slot mm-rects are inviolate; align shifts axes WITHIN their slots.** If a shift would push axes outside the slot, raises `ComposerError` with a helpful message.
- **`add_column(side, *panels)`** — vertical strip on the left or right of the canvas (for legend strips, etc.). LOWER priority than the rest — implemented as a stretch goal in this PR; if it adds complexity, defer to PR 3.5.
- **`canvas.finalize()`** — explicit no-op-if-already-finalized helper users can call to materialize the figure without accessing `.figure`. Useful for `canvas.figure.canvas.draw()` workflows.
- **CHANGELOG entry** under `[Unreleased]`.
- **2 gallery examples** showing multi-row + PanelGrid + canvas.align.

## What's OUT of scope for PR 3

- `pp.legend` `rows=`/`cols=`/`span=`/`ax=` upgrade → **PR 4**
- mm-precision regression test infra → **PR 4.5**
- Vector PDF/SVG compositing, `PanelImage`, `embed_figure` → **PR 5/6** (PR 3's `add_row` will accept `PanelImage` instances and raise `NotImplementedError` with a PR 5 pointer — same pattern as PR 1's PDF/SVG savefig)
- `canvas.inspect()` + composer-guide skill → **PR 7**
- Promoting `ComposerError`/`ComposerOverflowError` to top-level `pp.*` (deferred from PR 1 review)
- The reactor-settling docs note (deferred from PR 2 review) — lands with the composer-guide skill in PR 7
- `valign` per-row policies (`'top'`/`'bottom'`/`'center'`/`'baseline'`) — implemented but only `'top'` exercised in PR 3 tests. Other policies are stubbed with `NotImplementedError` for v1.
- Flex height / vfill across rows — PR 3 does NOT introduce vertical flex. Multi-row stacks at the natural sum of declared row heights + vpads.

This list is the contract for the spec-compliance reviewer. Anything in this list that the implementer ships counts as scope creep and should be flagged.

---

## File Structure

```
src/publiplots/composer/
├── __init__.py                       # MODIFY — re-export PanelGrid, PanelText
├── canvas.py                         # MAJOR refactor — split add_row into _stage_row + _finalize
├── panels.py                         # MODIFY — add PanelGrid + PanelText dataclasses
├── alignment.py                      # NEW — align() resolver, edge/mode/anchor logic
├── _layout.py                        # NEW — multi-row geometry builder (extends FigureLayout pattern)
├── presets.py                        # unchanged
├── exceptions.py                     # MODIFY — add ComposerAlignmentError (subclass of ComposerError)
├── _save.py                          # unchanged
├── _flex.py                          # unchanged
└── abc_labels.py                     # MODIFY — sequencer extends across multiple rows

tests/composer/
├── test_canvas_construction.py       # unchanged
├── test_panels.py                    # MODIFY — extend for PanelGrid/PanelText validation
├── test_add_row.py                   # MODIFY — multi-call add_row tests; UPDATE the
│                                     #   "called twice raises NotImplementedError" test
├── test_indexing.py                  # MODIFY — int indexing across rows
├── test_decoration_reservations.py   # unchanged
├── test_savefig.py                   # MODIFY — multi-row save sanity check
├── test_layout_integration.py        # unchanged
├── test_overflow_advisor.py          # unchanged
├── test_journal_presets.py           # unchanged
├── test_label_style.py               # MODIFY — abc sequencing across rows
├── test_abc_labels.py                # MODIFY — multi-row sequencer tests
├── test_multi_row.py                 # NEW — geometry, vpad, heterogeneous columns
├── test_panel_grid.py                # NEW — PanelGrid sub-grid axes, sharex/sharey
├── test_panel_text.py                # NEW — PanelText rendering, mathtext smoke
├── test_alignment.py                 # NEW — canvas.align edges, modes, anchor, slot-bounds
└── test_finalize.py                  # NEW — lazy finalization triggers, idempotency

examples/composer/                    # extend
├── cell_2col_multirow.py             # NEW — 2-row Cell figure with abc letters across rows
└── nature_2col_panel_grid.py         # NEW — 1-row Nature figure with PanelGrid + PanelText
```

13 modified or new files (5 modified + 8 new). Estimated ~900 LOC code + ~1100 LOC tests + 2 examples.

**File-size budget (warn if exceeded):**
- `canvas.py`: ≤ 450 LOC (post-refactor; PR 2 left it at 385, +/- the `add_row` split)
- `panels.py`: ≤ 280 LOC (was 168; +PanelGrid +PanelText ~ +100)
- `alignment.py`: ≤ 250 LOC (new)
- `_layout.py`: ≤ 180 LOC (new)
- `abc_labels.py`: ≤ 280 LOC (was 230; multi-row sequencer ~ +30)
- `exceptions.py`: ≤ 80 LOC (was 49; +ComposerAlignmentError ~ +20)
- Each test file: ≤ 300 LOC

Going over the budget is allowed if necessary, but the implementer MUST flag it as `DONE_WITH_CONCERNS`.

---

## Branch + worktree setup (Task 0)

**Files:** none (git only)

- [ ] **Step 1: Wait for PR 2 (#164) to merge to main**

```bash
cd /home/sagemaker-user/publiplots
git fetch origin --quiet
gh pr view 164 --json state,mergedAt
```

Expected: `{"state":"MERGED","mergedAt":"<timestamp>"}`. If still OPEN, STOP — PR 3 cannot start until PR 2 lands.

- [ ] **Step 2: Sync main**

```bash
cd /home/sagemaker-user/publiplots
git checkout main
git pull origin main --ff-only
```

Expected: HEAD is the squash-merge of PR 2 (e.g. `feat(composer): journal presets + flex sizing + abc labels (#164)`).

- [ ] **Step 3: Create the feature branch**

```bash
cd /home/sagemaker-user/publiplots
git checkout -b feat/composer-multirow-grid-align-pr3
```

- [ ] **Step 4: Run the existing test suite as a baseline**

```bash
cd /home/sagemaker-user/publiplots
uv run pytest tests/ -q --no-cov --tb=no 2>&1 | tail -3
```

Capture the count. Expected: ~1158 passed, 1 failed (pre-existing residplot — unrelated). The 140 composer tests from PR 1+2 are now part of the baseline.

---

## Task 1: Add ComposerAlignmentError exception type

**Files:**
- Modify: `src/publiplots/composer/exceptions.py`
- Modify: `src/publiplots/composer/__init__.py` — re-export
- Test: extend `tests/composer/test_canvas_construction.py` (the exceptions section)

**Why first:** smallest, most contained change. Lets the implementer warm up and gives Tasks 5+ a typed error to raise.

- [ ] **Step 1: Write the failing test**

Append to `/home/sagemaker-user/publiplots/tests/composer/test_canvas_construction.py` this exact content:

```python


# ---------------------------------------------------------------------------
# PR 3: ComposerAlignmentError — subclass of ComposerError
# ---------------------------------------------------------------------------

def test_composer_alignment_error_is_composer_error_subclass():
    """ComposerAlignmentError inherits from ComposerError so callers can
    catch all composer errors with one except clause."""
    from publiplots.composer.exceptions import (
        ComposerError,
        ComposerAlignmentError,
    )
    assert issubclass(ComposerAlignmentError, ComposerError)


def test_composer_alignment_error_carries_panels_and_edge():
    """ComposerAlignmentError exposes which panels and edge failed
    so callers can format helpful messages without re-parsing the str."""
    from publiplots.composer.exceptions import ComposerAlignmentError
    err = ComposerAlignmentError(
        "alignment shift would push panel A's left edge outside its slot",
        panels=("A", "B"),
        edge="left",
    )
    assert err.panels == ("A", "B")
    assert err.edge == "left"
    assert "A" in str(err)
```

- [ ] **Step 2: Run, verify it fails**

```bash
cd /home/sagemaker-user/publiplots
uv run pytest tests/composer/test_canvas_construction.py::test_composer_alignment_error_is_composer_error_subclass tests/composer/test_canvas_construction.py::test_composer_alignment_error_carries_panels_and_edge -v --no-cov
```

Expected: ImportError on `ComposerAlignmentError`.

- [ ] **Step 3: Implement ComposerAlignmentError**

Edit `/home/sagemaker-user/publiplots/src/publiplots/composer/exceptions.py`. Append after the existing `ComposerOverflowError` class:

```python


class ComposerAlignmentError(ComposerError):
    """Raised when a `canvas.align()` request can't be satisfied.

    The most common cause is that the requested shift would push a
    panel's content outside its (inviolate) slot mm-rect. The panels
    + edge attributes let callers format helpful messages without
    re-parsing the message text.
    """

    def __init__(
        self,
        message: str,
        *,
        panels: tuple,
        edge: str,
    ) -> None:
        super().__init__(message)
        self.panels = tuple(panels)
        self.edge = str(edge)
```

- [ ] **Step 4: Re-export from composer package**

Edit `/home/sagemaker-user/publiplots/src/publiplots/composer/__init__.py`. Add `ComposerAlignmentError` to the imports + `__all__`. The current shape is:

```python
from publiplots.composer.exceptions import (
    ComposerError,
    ComposerOverflowError,
)
```

Update to:

```python
from publiplots.composer.exceptions import (
    ComposerAlignmentError,
    ComposerError,
    ComposerOverflowError,
)
```

And in `__all__`, add `"ComposerAlignmentError"` (alphabetical position).

- [ ] **Step 5: Run tests**

```bash
cd /home/sagemaker-user/publiplots
uv run pytest tests/composer/test_canvas_construction.py -v --no-cov 2>&1 | tail -10
```

Expected: 2 new tests PASS, all PR 1+2 tests still pass.

- [ ] **Step 6: Commit**

```bash
cd /home/sagemaker-user/publiplots
git add src/publiplots/composer/exceptions.py \
        src/publiplots/composer/__init__.py \
        tests/composer/test_canvas_construction.py
git commit -m "feat(composer): ComposerAlignmentError exception for canvas.align"
```

---

## Task 2: PanelText dataclass

**Files:**
- Modify: `src/publiplots/composer/panels.py` — add `PanelText`
- Modify: `src/publiplots/composer/__init__.py` — re-export
- Create: `tests/composer/test_panel_text.py`

**Why second:** PanelText is the simplest of the two new panel kinds. Lands its data shape before PanelGrid (Task 3) and before any canvas wiring (Tasks 4+).

- [ ] **Step 1: Write the failing tests**

Create `/home/sagemaker-user/publiplots/tests/composer/test_panel_text.py` with this EXACT content:

```python
"""Tests for PanelText dataclass — PR 3 introduction."""

import pytest

import publiplots as pp
from publiplots.composer.panels import PanelText


# ---------------------------------------------------------------------------
# Construction + validation
# ---------------------------------------------------------------------------

def test_panel_text_basic_construction():
    p = PanelText(label="A", text="hello", size=(40.0, 20.0))
    assert p.label == "A"
    assert p.text == "hello"
    assert p.size == (40.0, 20.0)
    assert p.text_kw == {}
    assert p.label_style is None


def test_panel_text_text_required():
    with pytest.raises(TypeError):
        PanelText(label="A", size=(40.0, 20.0))


def test_panel_text_text_must_be_str():
    with pytest.raises(TypeError, match="text must be a str"):
        PanelText(label="A", text=123, size=(40.0, 20.0))


def test_panel_text_label_modes_inherit_panelaxes_contract():
    """label=None / False / str — same contract as PanelAxes."""
    p_none = PanelText(label=None, text="x", size=(40.0, 20.0))
    p_false = PanelText(label=False, text="x", size=(40.0, 20.0))
    p_str = PanelText(label="A", text="x", size=(40.0, 20.0))
    assert p_none.label is None
    assert p_false.label is False
    assert p_str.label == "A"


def test_panel_text_label_true_rejected():
    """Same bool-True trap as PanelAxes."""
    with pytest.raises(TypeError, match="label must be"):
        PanelText(label=True, text="x", size=(40.0, 20.0))


def test_panel_text_size_must_be_2_tuple_of_positive_numerics():
    with pytest.raises(ValueError, match="size must be a 2-tuple"):
        PanelText(label="A", text="x", size=20.0)
    with pytest.raises(ValueError, match="positive"):
        PanelText(label="A", text="x", size=(0.0, 20.0))


def test_panel_text_size_flex_width_accepted():
    """PanelText supports flex sizing the same way PanelAxes does."""
    p = PanelText(label="A", text="x", size=("flex", 20.0))
    assert p.size == ("flex", 20.0)


def test_panel_text_text_kw_must_be_mapping_or_none():
    p = PanelText(label="A", text="x", size=(40.0, 20.0), text_kw={"fontsize": 12})
    assert p.text_kw == {"fontsize": 12}
    with pytest.raises(TypeError, match="text_kw"):
        PanelText(label="A", text="x", size=(40.0, 20.0), text_kw="not-a-dict")


def test_panel_text_supports_label_style():
    """Same per-panel label_style override as PanelAxes."""
    p = PanelText(label="A", text="x", size=(40.0, 20.0),
                  label_style={"loc": "ur"})
    assert p.label_style == {"loc": "ur"}


# ---------------------------------------------------------------------------
# Top-level export
# ---------------------------------------------------------------------------

def test_panel_text_exported_at_top_level():
    assert hasattr(pp, "PanelText")
    p = pp.PanelText(label="A", text="x", size=(40.0, 20.0))
    assert isinstance(p, PanelText)
```

- [ ] **Step 2: Run, verify it fails**

```bash
cd /home/sagemaker-user/publiplots
uv run pytest tests/composer/test_panel_text.py -v --no-cov 2>&1 | tail -10
```

Expected: ImportError on `PanelText`.

- [ ] **Step 3: Implement PanelText in panels.py**

Edit `/home/sagemaker-user/publiplots/src/publiplots/composer/panels.py`. Append AFTER the existing `Panel` class:

```python


@dataclass(frozen=True)
class PanelText:
    """Input record for a text-only panel.

    Pass to :meth:`Canvas.add_row` to declare a panel containing a
    single block of text (centered by default). Internally rendered as
    a hidden axes (axis off, no patch) with one ``ax.text(...)`` call.
    Supports basic matplotlib mathtext via the underlying renderer.

    Parameters
    ----------
    label : str, None, or False
        Caption-addressable identifier. Same contract as :class:`PanelAxes`.
    text : str
        The text content. Supports mathtext (``r"$\\alpha$"``) and
        newlines.
    size : tuple of (width, height_mm)
        Panel mm rect. ``('flex', h_mm)`` accepted for flex width.
    text_kw : Mapping or None, default None (= empty dict)
        Forwarded to :func:`matplotlib.axes.Axes.text` (e.g.
        ``{'fontsize': 12, 'color': 'gray'}``). The axes text is placed
        at ``(0.5, 0.5)`` with ``ha='center', va='center'`` by default;
        these can be overridden via ``text_kw``.
    label_style : Mapping or None, default None
        Per-panel label_style override, same contract as
        :class:`PanelAxes`.

    Notes
    -----
    Frozen dataclass — instances are immutable and hashable.
    """

    label: Any
    text: str
    size: Tuple[Any, Any]
    text_kw: Optional[Mapping[str, Any]] = None
    label_style: Optional[Mapping[str, Any]] = None

    def __post_init__(self) -> None:
        # Validate label using the same logic as PanelAxes — extract to
        # a shared helper for DRY.
        _validate_panel_label(self.label)

        # Validate text
        if not isinstance(self.text, str):
            raise TypeError(
                f"text must be a str, got {type(self.text).__name__}"
            )

        # Validate size using the same logic as PanelAxes.
        normalized_size = _validate_panel_size(self.size, allow_flex_width=True)
        object.__setattr__(self, "size", normalized_size)

        # Validate text_kw: Mapping or None
        if self.text_kw is None:
            object.__setattr__(self, "text_kw", {})
        elif not hasattr(self.text_kw, "keys"):
            raise TypeError(
                f"text_kw must be a Mapping or None, "
                f"got {type(self.text_kw).__name__}"
            )

        # Validate label_style: Mapping or None
        if self.label_style is not None:
            if not hasattr(self.label_style, "keys"):
                raise TypeError(
                    f"label_style must be a Mapping or None, "
                    f"got {type(self.label_style).__name__}"
                )
```

- [ ] **Step 4: Extract shared label/size validators**

PanelText, PanelAxes, and (Task 3) PanelGrid all need the same label + size validation. Extract them into module-private helpers in `panels.py`. Add these BEFORE the `PanelAxes` class:

```python
def _validate_panel_label(label: Any) -> None:
    """Validate that label is str, None, or False. Raises TypeError otherwise."""
    if label is True:
        raise TypeError(
            "label must be a str, None (auto-letter), or False "
            "(no label); got True (did you mean abc=True on the Canvas?)"
        )
    if label is not None and label is not False:
        if not isinstance(label, str):
            raise TypeError(
                f"label must be a str, None (auto-letter), or False "
                f"(no label); got {type(label).__name__}"
            )


def _validate_panel_size(
    size: Any,
    *,
    allow_flex_width: bool = True,
) -> Tuple[Any, float]:
    """Validate a panel size 2-tuple. Returns the normalized tuple
    (with int→float coercion on numeric width)."""
    try:
        n = len(size)
    except TypeError:
        raise ValueError(
            f"size must be a 2-tuple of (width_mm, height_mm), got {size!r}"
        )
    if n != 2:
        raise ValueError(
            f"size must be a 2-tuple of (width_mm, height_mm), got length {n}"
        )

    w, h = size

    # Width: 'flex' string OR positive numeric
    if isinstance(w, str):
        if w != "flex" or not allow_flex_width:
            raise ValueError(
                f"size width must be numeric or 'flex', got {w!r}"
            )
        # 'flex' is OK — leave the string sentinel.
    else:
        try:
            fw = float(w)
        except (TypeError, ValueError):
            raise ValueError(f"size width must be numeric or 'flex', got {w!r}")
        if fw <= 0:
            raise ValueError(f"size width must be positive, got {fw}")
        w = fw

    # Height: positive numeric (no 'flex')
    if isinstance(h, str):
        raise ValueError(
            f"size height must be numeric, got {h!r} "
            "('flex' is width-only)"
        )
    try:
        fh = float(h)
    except (TypeError, ValueError):
        raise ValueError(f"size height must be numeric, got {h!r}")
    if fh <= 0:
        raise ValueError(f"size height must be positive, got {fh}")

    return (w, fh)
```

Then refactor `PanelAxes.__post_init__` to use these helpers:

```python
    def __post_init__(self) -> None:
        _validate_panel_label(self.label)
        normalized = _validate_panel_size(self.size, allow_flex_width=True)
        object.__setattr__(self, "size", normalized)
        if self.label_style is not None:
            if not hasattr(self.label_style, "keys"):
                raise TypeError(
                    f"label_style must be a Mapping or None, "
                    f"got {type(self.label_style).__name__}"
                )
```

- [ ] **Step 5: Re-export PanelText from composer package**

Edit `/home/sagemaker-user/publiplots/src/publiplots/composer/__init__.py`. Add `PanelText` to the imports + `__all__`:

```python
from publiplots.composer.panels import Panel, PanelAxes, PanelText
```

In `__all__`, add `"PanelText"` (alphabetical position).

- [ ] **Step 6: Re-export from top-level package**

Edit `/home/sagemaker-user/publiplots/src/publiplots/__init__.py`. Find the existing composer import:

```python
from publiplots.composer import Canvas, PanelAxes, Panel
```

Update to:

```python
from publiplots.composer import Canvas, Panel, PanelAxes, PanelText
```

And add `"PanelText"` to `__all__` near the other composer entries.

- [ ] **Step 7: Run all tests**

```bash
cd /home/sagemaker-user/publiplots
uv run pytest tests/composer/ -q --no-cov 2>&1 | tail -3
```

Expected: 142+ composer tests pass (was 140 + 9 new PanelText tests = 149, but the 2 alignment-error tests from Task 1 also count). Adjust if exact count differs.

- [ ] **Step 8: Commit**

```bash
cd /home/sagemaker-user/publiplots
git add src/publiplots/composer/panels.py \
        src/publiplots/composer/__init__.py \
        src/publiplots/__init__.py \
        tests/composer/test_panel_text.py
git commit -m "feat(composer): PanelText dataclass + shared label/size validators"
```

---

## Task 3: PanelGrid dataclass

**Files:**
- Modify: `src/publiplots/composer/panels.py` — add `PanelGrid`
- Modify: `src/publiplots/composer/__init__.py` — re-export
- Modify: `src/publiplots/__init__.py` — re-export
- Create: `tests/composer/test_panel_grid.py`

**Note:** Task 3 lands the dataclass + input validation only. Actual axes-grid CONSTRUCTION (creating the inner `Axes` array within the panel's mm rect) lands in Task 6 along with the canvas refactor.

- [ ] **Step 1: Write the failing tests**

Create `/home/sagemaker-user/publiplots/tests/composer/test_panel_grid.py` with this EXACT content:

```python
"""Tests for PanelGrid dataclass — PR 3 introduction.

Tests the input record only. Actual axes-grid construction is exercised
by tests/composer/test_multi_row.py + test_add_row.py after Task 6 lands
the canvas refactor.
"""

import pytest

import publiplots as pp
from publiplots.composer.panels import PanelGrid


# ---------------------------------------------------------------------------
# Construction + validation
# ---------------------------------------------------------------------------

def test_panel_grid_basic_construction():
    p = PanelGrid(label="C", shape=(2, 3), axes_size=(40.0, 30.0))
    assert p.label == "C"
    assert p.shape == (2, 3)
    assert p.axes_size == (40.0, 30.0)
    assert p.sharex is False
    assert p.sharey is False
    assert p.hspace == 2.0
    assert p.wspace == 2.0


def test_panel_grid_shape_must_be_2_tuple_of_positive_ints():
    with pytest.raises(ValueError, match="shape must be"):
        PanelGrid(label="C", shape=(2,), axes_size=(40.0, 30.0))
    with pytest.raises(ValueError, match="shape must be"):
        PanelGrid(label="C", shape=(0, 3), axes_size=(40.0, 30.0))
    with pytest.raises(ValueError, match="shape must be"):
        PanelGrid(label="C", shape=(2, -1), axes_size=(40.0, 30.0))


def test_panel_grid_axes_size_must_be_2_tuple_of_positive_floats():
    """axes_size in PanelGrid is per-CELL; flex is NOT supported here
    (would require resolving inside the panel's slot)."""
    with pytest.raises(ValueError, match="axes_size"):
        PanelGrid(label="C", shape=(2, 3), axes_size=("flex", 30.0))
    with pytest.raises(ValueError, match="positive"):
        PanelGrid(label="C", shape=(2, 3), axes_size=(0.0, 30.0))


def test_panel_grid_sharex_sharey_validated():
    """sharex/sharey accept bool or 'all'/'row'/'col'/'none' — same as
    pp.subplots."""
    PanelGrid(label="C", shape=(2, 3), axes_size=(40.0, 30.0), sharex=True)
    PanelGrid(label="C", shape=(2, 3), axes_size=(40.0, 30.0), sharex="row")
    PanelGrid(label="C", shape=(2, 3), axes_size=(40.0, 30.0), sharey="col")
    with pytest.raises(ValueError, match="share"):
        PanelGrid(label="C", shape=(2, 3), axes_size=(40.0, 30.0), sharex="invalid")


def test_panel_grid_hspace_wspace_must_be_non_negative():
    with pytest.raises(ValueError, match="hspace"):
        PanelGrid(label="C", shape=(2, 3), axes_size=(40.0, 30.0), hspace=-1.0)
    with pytest.raises(ValueError, match="wspace"):
        PanelGrid(label="C", shape=(2, 3), axes_size=(40.0, 30.0), wspace=-1.0)


def test_panel_grid_label_modes_match_panelaxes_contract():
    PanelGrid(label=None, shape=(2, 3), axes_size=(40.0, 30.0))
    PanelGrid(label=False, shape=(2, 3), axes_size=(40.0, 30.0))
    with pytest.raises(TypeError, match="label must be"):
        PanelGrid(label=True, shape=(2, 3), axes_size=(40.0, 30.0))


def test_panel_grid_size_mm_property_computed_from_shape_and_axes_size():
    """The panel's outer mm rect = (cols*w + (cols-1)*wspace,
                                   rows*h + (rows-1)*hspace)."""
    p = PanelGrid(label="C", shape=(2, 3), axes_size=(40.0, 30.0),
                  hspace=2.0, wspace=2.0)
    # 3 cols of 40mm + 2 wspaces of 2mm = 124mm
    # 2 rows of 30mm + 1 hspace of 2mm = 62mm
    assert p.size_mm == (124.0, 62.0)


def test_panel_grid_size_mm_with_default_spacing():
    p = PanelGrid(label="C", shape=(1, 3), axes_size=(50.0, 30.0))
    # default wspace=2.0, hspace=2.0; 3 cells of 50 + 2 wspaces = 154mm
    assert p.size_mm == (154.0, 30.0)


def test_panel_grid_supports_label_style():
    p = PanelGrid(label="C", shape=(2, 3), axes_size=(40.0, 30.0),
                  label_style={"loc": "ur"})
    assert p.label_style == {"loc": "ur"}


# ---------------------------------------------------------------------------
# Top-level export
# ---------------------------------------------------------------------------

def test_panel_grid_exported_at_top_level():
    assert hasattr(pp, "PanelGrid")
    p = pp.PanelGrid(label="C", shape=(2, 3), axes_size=(40.0, 30.0))
    assert isinstance(p, PanelGrid)
```

- [ ] **Step 2: Run to verify it fails**

```bash
cd /home/sagemaker-user/publiplots
uv run pytest tests/composer/test_panel_grid.py -v --no-cov 2>&1 | tail -10
```

Expected: ImportError on `PanelGrid`.

- [ ] **Step 3: Implement PanelGrid in panels.py**

Edit `/home/sagemaker-user/publiplots/src/publiplots/composer/panels.py`. Append AFTER `PanelText`:

```python


_VALID_SHARE = frozenset({True, False, "all", "row", "col", "none"})


@dataclass(frozen=True)
class PanelGrid:
    """Input record for an axes-grid panel.

    A sub-grid of axes laid out inside the panel's mm rect. The panel's
    outer dimensions are computed from ``shape`` and ``axes_size`` plus
    inter-cell spacing — there is no ``size`` kwarg.

    Parameters
    ----------
    label : str, None, or False
        Caption-addressable identifier. Same contract as :class:`PanelAxes`.
    shape : tuple of (nrows, ncols)
        Inner grid shape. Both must be positive integers.
    axes_size : tuple of (width_mm, height_mm)
        Per-cell axes mm dimensions. Both must be positive floats.
        ``'flex'`` is NOT supported here (would require resolving inside
        the panel's slot, which complicates the geometry; users can
        compute the per-cell width from the available slot mm by hand
        if they need it).
    sharex, sharey : bool or {'all', 'row', 'col', 'none'}, default False
        Axis-sharing semantics, matching :func:`matplotlib.pyplot.subplots`.
    hspace, wspace : float, default 2.0 mm
        Inter-cell spacing in millimeters.
    label_style : Mapping or None, default None
        Per-panel label_style override.

    Notes
    -----
    Frozen dataclass — instances are immutable and hashable.

    The panel's outer mm rect is exposed via :attr:`size_mm`:
    ``(ncols*w + (ncols-1)*wspace, nrows*h + (nrows-1)*hspace)``.
    """

    label: Any
    shape: Tuple[int, int]
    axes_size: Tuple[float, float]
    sharex: Any = False
    sharey: Any = False
    hspace: float = 2.0
    wspace: float = 2.0
    label_style: Optional[Mapping[str, Any]] = None

    def __post_init__(self) -> None:
        _validate_panel_label(self.label)

        # shape: 2-tuple of positive ints
        try:
            ns = len(self.shape)
        except TypeError:
            raise ValueError(
                f"shape must be a 2-tuple of (nrows, ncols), got {self.shape!r}"
            )
        if ns != 2:
            raise ValueError(
                f"shape must be a 2-tuple of (nrows, ncols), got length {ns}"
            )
        nr, nc = self.shape
        for name, v in (("nrows", nr), ("ncols", nc)):
            if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
                raise ValueError(
                    f"shape must be (positive int, positive int); "
                    f"{name}={v!r} is not a positive int"
                )

        # axes_size: 2-tuple of positive numerics, NO flex
        normalized_axes = _validate_panel_size(
            self.axes_size, allow_flex_width=False
        )
        # _validate_panel_size returns (w, h_float). For axes_size,
        # width must also be coerced to float (no 'flex').
        nw, nh = normalized_axes
        if isinstance(nw, str):  # paranoia: shouldn't be reachable
            raise ValueError(
                f"axes_size width must be numeric (no 'flex' for grid), got {nw!r}"
            )
        object.__setattr__(self, "axes_size", (float(nw), nh))

        # sharex/sharey
        for name, v in (("sharex", self.sharex), ("sharey", self.sharey)):
            if v not in _VALID_SHARE:
                raise ValueError(
                    f"{name} must be bool or one of 'all'/'row'/'col'/'none', "
                    f"got {v!r}"
                )

        # hspace/wspace: non-negative numerics
        for name, v in (("hspace", self.hspace), ("wspace", self.wspace)):
            if not isinstance(v, (int, float)) or isinstance(v, bool) or v < 0:
                raise ValueError(
                    f"{name} must be a non-negative number, got {v!r}"
                )

        # label_style
        if self.label_style is not None:
            if not hasattr(self.label_style, "keys"):
                raise TypeError(
                    f"label_style must be a Mapping or None, "
                    f"got {type(self.label_style).__name__}"
                )

    @property
    def size_mm(self) -> Tuple[float, float]:
        """Outer mm rect computed from shape + axes_size + spacing."""
        nr, nc = self.shape
        w, h = self.axes_size
        outer_w = nc * w + max(nc - 1, 0) * self.wspace
        outer_h = nr * h + max(nr - 1, 0) * self.hspace
        return (outer_w, outer_h)
```

- [ ] **Step 4: Re-export PanelGrid**

Edit `/home/sagemaker-user/publiplots/src/publiplots/composer/__init__.py`:

```python
from publiplots.composer.panels import Panel, PanelAxes, PanelGrid, PanelText
```

Add `"PanelGrid"` to `__all__`.

Edit `/home/sagemaker-user/publiplots/src/publiplots/__init__.py`:

```python
from publiplots.composer import Canvas, Panel, PanelAxes, PanelGrid, PanelText
```

Add `"PanelGrid"` to top-level `__all__`.

- [ ] **Step 5: Run tests**

```bash
cd /home/sagemaker-user/publiplots
uv run pytest tests/composer/test_panel_grid.py tests/composer/test_panel_text.py tests/composer/test_panels.py -v --no-cov 2>&1 | tail -10
```

Expected: PR 1+2 panels tests pass; PR 3 PanelText + PanelGrid tests pass.

- [ ] **Step 6: Commit**

```bash
cd /home/sagemaker-user/publiplots
git add src/publiplots/composer/panels.py \
        src/publiplots/composer/__init__.py \
        src/publiplots/__init__.py \
        tests/composer/test_panel_grid.py
git commit -m "feat(composer): PanelGrid dataclass — sub-grid of axes within a panel"
```

---

## Task 4: Multi-row geometry — `_layout.py` pure-Python helper

**Files:**
- Create: `src/publiplots/composer/_layout.py`
- Test: dedicated tests for the geometry math (no canvas yet)

**Why this task:** the multi-row geometry is the most subtle piece of PR 3. Lock it in as a pure-Python primitive with thorough tests BEFORE wiring it into the canvas.

The geometry:

- Each row has `n_panels`, `panel_widths` (resolved post-flex), `row_height` (max of panel heights), `vpad` (mm above this row; 0 for first row).
- Canvas height = `outer_pad×2 + sum(title_space) + sum(row_height) + sum(xlabel_space) + sum(vpad) + (R-1)×hspace_inter_row`. Note: `vpad` and `hspace_inter_row` are conceptually similar; we'll merge them — `vpad` per-row replaces the global `hspace`.
- Each panel's `bbox_mm` is computed within its row's allocated band.

- [ ] **Step 1: Write the failing tests for `_layout.py`**

Create `/home/sagemaker-user/publiplots/tests/composer/test_multi_row.py` (this file will grow to also cover the canvas integration in Task 6; for now it has the geometry-helper tests). Initial content:

```python
"""Tests for multi-row canvas geometry — PR 3 introduction."""

import pytest

import publiplots as pp
from publiplots.composer._layout import compute_canvas_geometry


# ---------------------------------------------------------------------------
# compute_canvas_geometry — pure math, no matplotlib
# ---------------------------------------------------------------------------

def test_single_row_geometry_matches_pr1():
    """Single-row case must match what add_row produced in PR 1+2."""
    g = compute_canvas_geometry(
        rows=[{
            "panel_widths_mm": (70.0, 70.0),
            "row_height_mm": 40.0,
            "vpad_mm": 0.0,
        }],
        canvas_width_mm=174.0,
        outer_pad=2.0,
        ylabel_space=10.0,
        right=2.0,
        wspace=3.0,
        title_space=5.0,
        xlabel_space=8.0,
    )
    # Width: 2+10+70+2+3+10+70+2+2 = 171
    # Height: 2+5+40+8+2 = 57
    assert abs(g.canvas_width_mm - 171.0) < 0.01
    assert abs(g.canvas_height_mm - 57.0) < 0.01
    # Panel positions
    assert len(g.row_axes_rects_mm) == 1
    assert len(g.row_axes_rects_mm[0]) == 2  # 2 panels in row 0


def test_two_row_geometry_stacks_correctly():
    """Two rows: heights sum + vpad on the second row."""
    g = compute_canvas_geometry(
        rows=[
            {"panel_widths_mm": (70.0, 70.0), "row_height_mm": 40.0, "vpad_mm": 0.0},
            {"panel_widths_mm": (70.0, 70.0), "row_height_mm": 30.0, "vpad_mm": 4.0},
        ],
        canvas_width_mm=174.0,
        outer_pad=2.0,
        ylabel_space=10.0,
        right=2.0,
        wspace=3.0,
        title_space=5.0,
        xlabel_space=8.0,
    )
    # Width same as single-row: 171mm
    assert abs(g.canvas_width_mm - 171.0) < 0.01
    # Height: 2 (outer top)
    #         + 5 (title row 0) + 40 (row 0) + 8 (xlabel row 0)
    #         + 4 (vpad row 1)
    #         + 5 (title row 1) + 30 (row 1) + 8 (xlabel row 1)
    #         + 2 (outer bottom)
    #       = 104
    assert abs(g.canvas_height_mm - 104.0) < 0.01


def test_heterogeneous_columns_per_row():
    """Row 0 has 2 panels of width 70; row 1 has 1 panel of width 150."""
    g = compute_canvas_geometry(
        rows=[
            {"panel_widths_mm": (70.0, 70.0), "row_height_mm": 40.0, "vpad_mm": 0.0},
            {"panel_widths_mm": (150.0,), "row_height_mm": 30.0, "vpad_mm": 4.0},
        ],
        canvas_width_mm=174.0,
        outer_pad=2.0,
        ylabel_space=10.0,
        right=2.0,
        wspace=3.0,
        title_space=5.0,
        xlabel_space=8.0,
    )
    # Width is the MAX of any single row's width:
    #   row 0: 2+10+70+2+3+10+70+2+2 = 171
    #   row 1: 2+10+150+2+2 = 166
    # canvas width = max(171, 166) = 171
    assert abs(g.canvas_width_mm - 171.0) < 0.01

    # Row 1's single panel is centered in the canvas (or left-justified —
    # we'll lock the convention here): per spec, justify='start' is default,
    # so the panel sits at the LEFT (after outer_pad + ylabel_space).
    row1_rects = g.row_axes_rects_mm[1]
    assert len(row1_rects) == 1
    x_mm, y_mm, w_mm, h_mm = row1_rects[0]
    # Left edge should be 2 (outer_pad) + 10 (ylabel) = 12 mm
    assert abs(x_mm - 12.0) < 0.01
    assert abs(w_mm - 150.0) < 0.01


def test_panel_positions_have_correct_y_coords_top_down():
    """y_mm uses bottom-left origin (matplotlib convention). Row 0 (top)
    has the LARGEST y; row 1 (bottom) has the SMALLEST y."""
    g = compute_canvas_geometry(
        rows=[
            {"panel_widths_mm": (70.0, 70.0), "row_height_mm": 40.0, "vpad_mm": 0.0},
            {"panel_widths_mm": (70.0, 70.0), "row_height_mm": 30.0, "vpad_mm": 4.0},
        ],
        canvas_width_mm=174.0,
        outer_pad=2.0,
        ylabel_space=10.0,
        right=2.0,
        wspace=3.0,
        title_space=5.0,
        xlabel_space=8.0,
    )
    row0_y = g.row_axes_rects_mm[0][0][1]
    row1_y = g.row_axes_rects_mm[1][0][1]
    # Row 0 is on top → its y is larger.
    assert row0_y > row1_y


def test_zero_rows_raises():
    with pytest.raises(ValueError, match="at least one row"):
        compute_canvas_geometry(
            rows=[],
            canvas_width_mm=174.0,
            outer_pad=2.0, ylabel_space=10.0, right=2.0, wspace=3.0,
            title_space=5.0, xlabel_space=8.0,
        )
```

- [ ] **Step 2: Run, verify they fail**

Expected: ImportError on `_layout`.

- [ ] **Step 3: Implement `_layout.py`**

Create `/home/sagemaker-user/publiplots/src/publiplots/composer/_layout.py` with this EXACT content:

```python
"""Pure-geometry multi-row canvas layout.

Computes canvas dimensions and per-panel mm-rects for a multi-row
canvas. No matplotlib imports — pure math.

The canvas is laid out top-to-bottom: row 0 at the top, row R-1 at the
bottom. y-coordinates use matplotlib's bottom-left origin convention,
so row 0's y_mm > row 1's y_mm > ... > row (R-1)'s y_mm.

A row carries its own panel widths (post-flex resolution), a single
row_height (max of its panel heights), and a vpad (mm above this row;
0 for the first row).

Decoration reservations (title_space, xlabel_space, ylabel_space, right)
are scalar in PR 3 — they apply uniformly to every row/column. This
matches what `pp.subplots`' `SubplotsAutoLayout` will measure at draw
time. Per-row variation is left for a future PR if user feedback
demands it.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple


@dataclass(frozen=True)
class CanvasGeometry:
    """Result of :func:`compute_canvas_geometry`.

    Attributes
    ----------
    canvas_width_mm, canvas_height_mm : float
        Overall canvas dimensions.
    row_axes_rects_mm : list of list of (x, y, w, h)
        ``row_axes_rects_mm[r][c]`` is the mm rect for panel c in row r.
        Coordinates use bottom-left origin. ``(x, y)`` is the bottom-
        left corner of the panel's axes (NOT including ylabel/xlabel
        decoration space).
    """

    canvas_width_mm: float
    canvas_height_mm: float
    row_axes_rects_mm: List[List[Tuple[float, float, float, float]]]


def compute_canvas_geometry(
    *,
    rows: Sequence[Dict],
    canvas_width_mm: float,
    outer_pad: float,
    ylabel_space: float,
    right: float,
    wspace: float,
    title_space: float,
    xlabel_space: float,
) -> CanvasGeometry:
    """Compute canvas dimensions + per-panel mm rects for multi-row layouts.

    Parameters
    ----------
    rows : sequence of dict
        Each row dict has keys:
        - ``panel_widths_mm`` (tuple of float, after flex resolution)
        - ``row_height_mm`` (float — max of panel heights in the row)
        - ``vpad_mm`` (float — mm above this row; 0 for first row)
    canvas_width_mm : float
        The canvas's declared width (NOT necessarily what the figure
        ends up — that depends on whether any row had flex panels).
    outer_pad, ylabel_space, right, wspace, title_space, xlabel_space : float
        rcParams-derived reservations in mm.

    Returns
    -------
    :class:`CanvasGeometry`
    """
    if not rows:
        raise ValueError("compute_canvas_geometry requires at least one row")

    # Canvas width = max of any row's required width. Each row's required
    # width = outer_pad + sum(ylabel + panel_w + right) + (n-1)*wspace + outer_pad.
    row_required_widths = []
    for row in rows:
        widths = row["panel_widths_mm"]
        n = len(widths)
        if n == 0:
            raise ValueError("each row must have at least one panel")
        rw = (
            2 * outer_pad
            + n * ylabel_space
            + sum(widths)
            + n * right
            + max(n - 1, 0) * wspace
        )
        row_required_widths.append(rw)
    canvas_width = max(row_required_widths)

    # The user-supplied canvas_width_mm is treated as a maximum hint; we
    # use the computed max-row width as the actual figure width. This
    # matches PR 1's behavior (figure width = panels + decorations) and
    # leaves the canvas budget as a CAP rather than a TARGET.
    canvas_width_mm = max(canvas_width, canvas_width_mm)

    # Canvas height = outer_pad + sum_rows(title + row_height + xlabel + vpad) + outer_pad.
    canvas_height = 2 * outer_pad
    for row in rows:
        canvas_height += row["vpad_mm"]
        canvas_height += title_space
        canvas_height += row["row_height_mm"]
        canvas_height += xlabel_space

    # Compute per-panel rects. Iterate top-down (row 0 at top), tracking
    # the running y-cursor from the top of the canvas downward.
    row_axes_rects: List[List[Tuple[float, float, float, float]]] = []
    y_cursor_from_top = outer_pad  # mm from the top of the canvas
    for row in rows:
        widths = row["panel_widths_mm"]
        row_h = row["row_height_mm"]

        y_cursor_from_top += row["vpad_mm"]
        y_cursor_from_top += title_space

        # The panel's axes-rect y_mm (bottom-left origin) = canvas_height - y_cursor - row_h
        y_mm = canvas_height - y_cursor_from_top - row_h

        # Per-panel x positions: left-justify (justify='start' default)
        x_cursor = outer_pad
        rects = []
        for w in widths:
            x_cursor += ylabel_space
            rects.append((x_cursor, y_mm, w, row_h))
            x_cursor += w
            x_cursor += right
            x_cursor += wspace
        row_axes_rects.append(rects)

        y_cursor_from_top += row_h
        y_cursor_from_top += xlabel_space

    return CanvasGeometry(
        canvas_width_mm=canvas_width_mm,
        canvas_height_mm=canvas_height,
        row_axes_rects_mm=row_axes_rects,
    )
```

- [ ] **Step 4: Run the geometry tests**

```bash
cd /home/sagemaker-user/publiplots
uv run pytest tests/composer/test_multi_row.py -v --no-cov
```

Expected: 5 tests PASS (the 5 we wrote in Step 1).

- [ ] **Step 5: Commit**

```bash
cd /home/sagemaker-user/publiplots
git add src/publiplots/composer/_layout.py \
        tests/composer/test_multi_row.py
git commit -m "feat(composer): _layout.py — multi-row canvas geometry helper"
```

---

## Task 5: Refactor canvas.py — extract `_compute_geometry`/`_build_axes` helpers

**Files:**
- Modify: `src/publiplots/composer/canvas.py`

**Why this task:** PR 2's review flagged `canvas.py` (385 LOC) for splitting `add_row` into smaller helpers. PR 3 will rewrite `add_row` substantially anyway (lazy finalization, multi-row); the cleanest path is to split FIRST (no behavioral change), then add multi-row support in Task 6 with a clean structure.

This task is BEHAVIORALLY a no-op — all PR 1+2 tests must continue passing.

- [ ] **Step 1: Run the full composer test suite as a baseline**

```bash
cd /home/sagemaker-user/publiplots
uv run pytest tests/composer/ -q --no-cov 2>&1 | tail -3
```

Capture the pass count. The refactor must NOT change this number.

- [ ] **Step 2: Refactor `add_row` into helpers**

Edit `/home/sagemaker-user/publiplots/src/publiplots/composer/canvas.py`. Extract these private methods on `Canvas`, called by `add_row`:

- `_validate_row_inputs(self, panels)` — multi-call rejection, type check, duplicate-label check, zero-panel check
- `_resolve_decorations(self)` — read all the rcParams `subplots.*` values into a dict
- `_resolve_row_widths(self, panels, decorations)` — call `_flex.resolve_flex_widths`, return `(col_widths, n_flex)`. Wraps the resolver's ValueError into ComposerOverflowError.
- `_check_pinned_overflow(self, col_widths, decorations, n_flex)` — only fires when `n_flex == 0`; raises ComposerOverflowError with the advisor.
- `_build_layout(self, col_widths, row_height, decorations)` — construct the `FigureLayout`.
- `_create_figure_and_axes(self, layout)` — `plt.figure(figsize=...)` + per-panel `add_axes`. Returns the (fig, axes_list).
- `_register_panels(self, panels, col_widths, axes_list, layout)` — resolve labels, merge styles, render labels, populate `self._panels` + `self._panels_ordered`.
- `_attach_reactor(self, fig, layout)` — wire up `SubplotsAutoLayout`.

The `add_row` method then becomes a short orchestrator:

```python
    def add_row(self, *panels: PanelAxes) -> None:
        self._validate_row_inputs(panels)
        decorations = self._resolve_decorations()
        col_widths, n_flex = self._resolve_row_widths(panels, decorations)
        self._check_pinned_overflow(col_widths, decorations, n_flex)
        row_height = max(p.size[1] for p in panels)
        layout = self._build_layout(col_widths, row_height, decorations)
        fig, axes_list = self._create_figure_and_axes(layout)
        self._figure = fig
        self._register_panels(panels, col_widths, axes_list, layout)
        self._attach_reactor(fig, layout)
        self._row_added = True
```

(Detailed implementation: read the current `add_row`, copy each contiguous block of code into the appropriate helper, preserving all logic verbatim. Imports stay inline where they currently are.)

- [ ] **Step 3: Run tests — they MUST all still pass**

```bash
cd /home/sagemaker-user/publiplots
uv run pytest tests/composer/ -q --no-cov 2>&1 | tail -3
```

Expected: SAME pass count as the baseline from Step 1. NO regressions.

If anything fails, the refactor introduced a behavioral change — STOP, diff against the original `add_row` block-by-block, and find the divergence.

- [ ] **Step 4: Verify file size**

```bash
cd /home/sagemaker-user/publiplots
wc -l src/publiplots/composer/canvas.py
```

Expected: roughly the same line count as before (the helpers add some method-def overhead but the body is the same lines). Acceptable target: ≤ 420 LOC.

- [ ] **Step 5: Commit**

```bash
cd /home/sagemaker-user/publiplots
git add src/publiplots/composer/canvas.py
git commit -m "refactor(composer): split add_row into _validate/_resolve/_build/_register helpers"
```

---

## Task 6: Multi-row support in Canvas

**Files:**
- Modify: `src/publiplots/composer/canvas.py` — add staging, lazy finalization, multi-row geometry
- Modify: `tests/composer/test_add_row.py` — UPDATE the "called twice raises NotImplementedError" test
- Test: extend `tests/composer/test_multi_row.py` with canvas-level tests
- Test: extend `tests/composer/test_indexing.py` with multi-row int indexing
- Create: `tests/composer/test_finalize.py`

This is the biggest behavioral task in PR 3. The shape:

1. `add_row` no longer creates the figure. It STAGES a `_RowStaging` record into `self._rows: List[_RowStaging]`.
2. The figure is created lazily when any of these is touched: `canvas.figure`, `canvas.figure_size_mm`, `canvas[label]`/`canvas[i]`, `canvas.savefig`. Or explicitly via `canvas.finalize()`.
3. Finalization runs through the new `_layout.compute_canvas_geometry`.
4. abc sequencing now spans across rows — `resolve_labels` is called once with the FLAT panel list (rows concatenated in order).

- [ ] **Step 1: Update the "called twice raises" test**

Open `/home/sagemaker-user/publiplots/tests/composer/test_add_row.py`. Find the test:

```python
def test_add_row_called_twice_raises():
    """PR 3 adds multi-row support; PR 1 rejects it with a clear hint."""
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(70.0, 40.0)))
    with pytest.raises(NotImplementedError, match="multi-row"):
        canvas.add_row(pp.PanelAxes(label="B", size=(70.0, 40.0)))
```

DELETE this test entirely (PR 3 lifts the restriction).

- [ ] **Step 2: Append the multi-row canvas tests**

Append to `/home/sagemaker-user/publiplots/tests/composer/test_multi_row.py`:

```python


# ---------------------------------------------------------------------------
# PR 3: multi-row Canvas integration
# ---------------------------------------------------------------------------

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


MM_TOL = 0.01


def test_add_row_called_twice_succeeds():
    """PR 3 lifts PR 1's NotImplementedError on multi-call add_row."""
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(70.0, 40.0)))
    canvas.add_row(pp.PanelAxes(label="B", size=(70.0, 40.0)))
    # No exception — both rows added.
    assert "A" in canvas._panels
    assert "B" in canvas._panels


def test_two_row_canvas_figure_height_grows():
    """Two rows of 40mm panels — figure height should be roughly
    2 * (panel + xlabel + title) + outer_pad×2 + vpad."""
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(70.0, 40.0)))
    canvas.add_row(pp.PanelAxes(label="B", size=(70.0, 40.0)))
    w_mm, h_mm = canvas.figure_size_mm
    # Per the geometry helper:
    # height = 2 (outer top)
    #        + 5 (title) + 40 (row 0) + 8 (xlabel)
    #        + vpad (default — see Canvas.add_row defaults) + 5 + 40 + 8
    #        + 2 (outer bottom)
    # For default vpad=4 (configurable): total = 114
    assert h_mm > 100  # rough lower bound; exact value depends on vpad default


def test_canvas_indexing_int_spans_rows():
    """canvas[0] is row 0 panel 0; canvas[1] is row 0 panel 1 (or row 1
    panel 0 if row 0 only had 1 panel) — flat insertion order."""
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(
        pp.PanelAxes(label="A", size=(70.0, 40.0)),
        pp.PanelAxes(label="B", size=(70.0, 40.0)),
    )
    canvas.add_row(pp.PanelAxes(label="C", size=(140.0, 30.0)))
    assert canvas[0].label == "A"
    assert canvas[1].label == "B"
    assert canvas[2].label == "C"


def test_canvas_abc_sequencing_continues_across_rows():
    """abc='upper' produces 'A','B','C',... — across all rows in
    insertion order (row 0 first, then row 1, etc.)."""
    canvas = pp.Canvas("cell-2col", abc="upper")
    canvas.add_row(
        pp.PanelAxes(label=None, size=(60.0, 40.0)),
        pp.PanelAxes(label=None, size=(60.0, 40.0)),
    )
    canvas.add_row(
        pp.PanelAxes(label=None, size=(60.0, 40.0)),
    )
    assert canvas[0].label == "A"
    assert canvas[1].label == "B"
    assert canvas[2].label == "C"


def test_canvas_two_rows_panels_have_correct_y_ordering():
    """Row 0 panel sits HIGHER on the canvas than row 1 panel.
    bbox_mm uses bottom-left origin, so row 0's y > row 1's y."""
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(70.0, 40.0)))
    canvas.add_row(pp.PanelAxes(label="B", size=(70.0, 40.0)))
    a_y = canvas["A"].bbox_mm[1]
    b_y = canvas["B"].bbox_mm[1]
    assert a_y > b_y


def test_canvas_heterogeneous_row_widths():
    """Row 0: 2 panels of 60mm; Row 1: 1 panel of 130mm.
    No errors, geometry sane."""
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(
        pp.PanelAxes(label="A", size=(60.0, 40.0)),
        pp.PanelAxes(label="B", size=(60.0, 40.0)),
    )
    canvas.add_row(pp.PanelAxes(label="C", size=(130.0, 30.0)))
    assert canvas["A"].size_mm == (60.0, 40.0)
    assert canvas["C"].size_mm == (130.0, 30.0)
```

- [ ] **Step 3: Append integer-indexing multi-row tests**

Append to `/home/sagemaker-user/publiplots/tests/composer/test_indexing.py`:

```python


def test_int_indexing_spans_multiple_rows():
    """canvas[i] iterates panels in flat insertion order across rows."""
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(
        pp.PanelAxes(label="A", size=(60.0, 40.0)),
        pp.PanelAxes(label="B", size=(60.0, 40.0)),
    )
    canvas.add_row(
        pp.PanelAxes(label="C", size=(60.0, 40.0)),
        pp.PanelAxes(label="D", size=(60.0, 40.0)),
    )
    assert canvas[0].label == "A"
    assert canvas[1].label == "B"
    assert canvas[2].label == "C"
    assert canvas[3].label == "D"
    with pytest.raises(KeyError):
        canvas[4]
```

- [ ] **Step 4: Write `test_finalize.py`**

Create `/home/sagemaker-user/publiplots/tests/composer/test_finalize.py`:

```python
"""Tests for lazy figure finalization in PR 3 multi-row Canvas."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

import publiplots as pp


def test_canvas_figure_is_none_until_finalize_or_access():
    """After add_row but before any figure access, .figure is still None."""
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(70.0, 40.0)))
    # NOTE: PR 1+2 created the figure synchronously in add_row.
    # PR 3 makes this LAZY — the figure isn't built until something
    # forces materialization. canvas.figure DOES force it (since it's
    # the public accessor). canvas._figure is the underlying state.
    assert canvas._figure is None


def test_canvas_figure_property_triggers_finalization():
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(70.0, 40.0)))
    fig = canvas.figure  # accessor triggers
    assert fig is not None
    assert canvas._figure is fig


def test_canvas_indexing_triggers_finalization():
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(70.0, 40.0)))
    p = canvas["A"]  # indexing triggers
    assert canvas._figure is not None
    assert p.ax is not None


def test_canvas_savefig_triggers_finalization(tmp_path):
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(70.0, 40.0)))
    canvas.savefig(tmp_path / "fig.png")  # savefig triggers
    assert canvas._figure is not None
    assert (tmp_path / "fig.png").exists()


def test_canvas_finalize_explicit_call():
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(70.0, 40.0)))
    canvas.finalize()
    assert canvas._figure is not None


def test_canvas_finalize_idempotent():
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(70.0, 40.0)))
    canvas.finalize()
    fig1 = canvas._figure
    canvas.finalize()  # second call must be a no-op
    assert canvas._figure is fig1


def test_canvas_add_row_after_finalize_raises():
    """Once finalized, add_row is no longer accepted."""
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(70.0, 40.0)))
    canvas.finalize()
    with pytest.raises(RuntimeError, match="finalize"):
        canvas.add_row(pp.PanelAxes(label="B", size=(70.0, 40.0)))


def test_canvas_savefig_before_add_row_still_raises():
    """An empty canvas (no rows) can't be saved."""
    canvas = pp.Canvas("custom", width=174.0)
    with pytest.raises(RuntimeError, match="add_row"):
        canvas.savefig("/tmp/nope.png")


def test_canvas_figure_size_mm_is_none_before_finalize():
    canvas = pp.Canvas("custom", width=174.0)
    assert canvas.figure_size_mm is None
    canvas.add_row(pp.PanelAxes(label="A", size=(70.0, 40.0)))
    # Still None — figure isn't built yet.
    assert canvas.figure_size_mm is None
    canvas.finalize()
    # Now it has a value.
    assert canvas.figure_size_mm is not None
```

- [ ] **Step 5: Run all tests to see baseline failures**

```bash
cd /home/sagemaker-user/publiplots
uv run pytest tests/composer/ -v --no-cov 2>&1 | tail -30
```

Expected: many failures in test_multi_row, test_finalize, test_indexing — they exercise behavior PR 3 hasn't built yet. Capture the count.

- [ ] **Step 6: Implement multi-row staging + lazy finalization in canvas.py**

This is the largest single edit in PR 3. Conceptually:

1. Add a `_RowStaging` dataclass holding `panels: tuple, vpad_mm: float`.
2. Replace `_panels: Dict, _panels_ordered: List, _figure: None, _row_added: bool` state with: `_rows: List[_RowStaging] = []`, `_panels: Dict[str, Panel] = {}` (populated at finalize), `_panels_ordered: List[Panel] = []` (same), `_figure = None`, `_finalized: bool = False`.
3. Rewrite `add_row` to:
   - Validate inputs (no multi-call check anymore — that restriction is lifted)
   - Stage a `_RowStaging(panels, vpad_mm=...)` into `self._rows`
   - Pull `vpad` from a kwarg (default 4.0; 0.0 for the first row regardless of kwarg)
4. Add `_finalize_if_needed(self)` method that:
   - Returns immediately if `self._finalized`
   - Resolves flex per row, aborts with `ComposerOverflowError` if any row overflows
   - Calls `compute_canvas_geometry` from `_layout.py`
   - Creates `plt.figure(figsize=...)` at the canvas dims
   - Per row, per panel: `fig.add_axes(rect)`, then for `PanelGrid` create the inner sub-grid using `pp.subplots`-style spacing
   - For `PanelText`: `ax.set_axis_off()`, `ax.text(0.5, 0.5, ...)`
   - For `PanelImage` (PR 5): raise `NotImplementedError` with a "PR 5" pointer
   - Resolve labels via `resolve_labels` over the FLAT cross-row panel list
   - Render labels via `render_label`
   - Populate `_panels` + `_panels_ordered`
   - Attach `SubplotsAutoLayout` (constructed from a top-level `FigureLayout` that mirrors the multi-row geometry — or, if `SubplotsAutoLayout` doesn't support multi-row natively, skip the reactor for multi-row cases initially and document the trade-off; PR 1's reactor was 1×N and may need extension. **DECISION: for v1, the reactor is attached only for the SINGLE-row case. Multi-row canvases skip the reactor — the figure is sized at the static rcParams initial values. PR 4.5 will refine this when the snapshot test infra lands.** Document this clearly in the code.)
   - Set `self._finalized = True`
5. `finalize()` is the public entry point — calls `_finalize_if_needed`.
6. `figure`, `figure_size_mm`, `__getitem__`, `savefig` all call `_finalize_if_needed` first.
7. After finalization, `add_row` raises `RuntimeError("Canvas already finalized; add_row not accepted")`.

The implementation is verbose. The implementer should:
- Read the post-Task-5 `canvas.py` to see the helper structure.
- Sketch the new `_finalize_if_needed` flow on paper before coding.
- Land it incrementally: get the basic single-row case working through the new lazy path FIRST (all PR 1+2 tests pass), THEN add multi-row support.

Detailed step-by-step:

**6a.** Add `_RowStaging` dataclass to `canvas.py` (or `panels.py` if preferred — module-private):

```python
from dataclasses import dataclass, field

@dataclass
class _RowStaging:
    panels: tuple   # tuple[PanelAxes | PanelGrid | PanelText, ...]
    vpad_mm: float
    justify: str = "start"
    valign: str = "top"
```

**6b.** Rewrite `Canvas.__init__` to add `_rows`, `_finalized`. Remove `_row_added` (replaced by `_finalized`).

**6c.** Rewrite `add_row(self, *panels, vpad=4.0, justify='start', valign='top')`:
- Reject if `_finalized`
- Validate panel types (now allow PanelAxes, PanelGrid, PanelText; raise on PanelImage with PR 5 hint)
- Stage `_RowStaging` (for the first row, override `vpad_mm=0.0`)

**6d.** Add `_finalize_if_needed(self)` that runs the full geometry + figure-creation pipeline using `compute_canvas_geometry`.

**6e.** Add `finalize(self)` public method.

**6f.** Update `figure` / `figure_size_mm` / `__getitem__` / `savefig` to call `_finalize_if_needed` first.

**6g.** Reactor strategy: attach `SubplotsAutoLayout` ONLY when there's exactly one row. Multi-row canvases get the static layout. Document with an inline comment + a note in the Canvas docstring under "Notes".

(The implementer will need to read the existing canvas.py carefully and rewrite roughly half of it. Estimated diff: ~150-200 lines changed.)

- [ ] **Step 7: Run all PR 1+2 tests to confirm no regressions**

```bash
cd /home/sagemaker-user/publiplots
uv run pytest tests/composer/ -q --no-cov --ignore=tests/composer/test_multi_row.py --ignore=tests/composer/test_finalize.py 2>&1 | tail -3
```

Expected: same count as PR 2 baseline (~140 + 11 from Tasks 1-4 of PR 3 = ~151).

- [ ] **Step 8: Run the new multi-row + finalize tests**

```bash
cd /home/sagemaker-user/publiplots
uv run pytest tests/composer/test_multi_row.py tests/composer/test_finalize.py tests/composer/test_indexing.py -v --no-cov 2>&1 | tail -20
```

Expected: ALL pass.

- [ ] **Step 9: Full composer regression**

```bash
cd /home/sagemaker-user/publiplots
uv run pytest tests/composer/ -q --no-cov 2>&1 | tail -3
```

Expected: ~170-175 passed total (~150 from PR 1+2 + Tasks 1-4 of PR 3 + ~20 new from Task 6).

- [ ] **Step 10: Commit**

```bash
cd /home/sagemaker-user/publiplots
git add src/publiplots/composer/canvas.py \
        tests/composer/test_add_row.py \
        tests/composer/test_multi_row.py \
        tests/composer/test_indexing.py \
        tests/composer/test_finalize.py
git commit -m "feat(composer): multi-row layouts + lazy finalization"
```

---

## Task 7: PanelGrid axes construction

**Files:**
- Modify: `src/publiplots/composer/canvas.py` — handle `PanelGrid` in `_finalize_if_needed`
- Test: extend `tests/composer/test_panel_grid.py` with canvas-integration tests

**Why now:** Task 6's `_finalize_if_needed` likely punted on PanelGrid (handling only `PanelAxes` for the initial multi-row plumbing). Task 7 fills that in.

The PanelGrid's outer mm rect is computed by `panel.size_mm` (the `@property` from Task 3). Inside that rect, we lay out a `shape=(r, c)` sub-grid using `pp.subplots`-style logic but at the panel-local coordinate level.

- [ ] **Step 1: Append failing canvas-integration tests**

Append to `/home/sagemaker-user/publiplots/tests/composer/test_panel_grid.py`:

```python


# ---------------------------------------------------------------------------
# Canvas integration — the panel actually creates a sub-grid of axes
# ---------------------------------------------------------------------------

import matplotlib
matplotlib.use("Agg")


def test_panel_grid_canvas_axes_attribute_returns_2d_array():
    """canvas[label].axes returns a 2D numpy array of Axes for a grid panel."""
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelGrid(label="C", shape=(2, 3), axes_size=(40.0, 30.0)))
    panel = canvas["C"]
    import numpy as np
    from matplotlib.axes import Axes
    assert isinstance(panel.axes, np.ndarray)
    assert panel.axes.shape == (2, 3)
    for row in panel.axes:
        for ax in row:
            assert isinstance(ax, Axes)


def test_panel_grid_kind_is_axesgrid():
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelGrid(label="C", shape=(2, 3), axes_size=(40.0, 30.0)))
    assert canvas["C"].kind == "axesgrid"


def test_panel_grid_ax_accessor_raises():
    """For PanelGrid, .ax raises (use .axes instead)."""
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelGrid(label="C", shape=(2, 3), axes_size=(40.0, 30.0)))
    with pytest.raises(AttributeError, match="axes"):
        _ = canvas["C"].ax


def test_panel_grid_outer_size_matches_size_mm_property():
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelGrid(label="C", shape=(1, 3), axes_size=(50.0, 30.0)))
    panel = canvas["C"]
    # PR 3: panel.size_mm reflects the OUTER mm rect of the grid panel
    # (not the per-cell size).
    assert panel.size_mm == (154.0, 30.0)  # 3*50 + 2*2 wspace = 154


def test_panel_grid_inner_axes_have_correct_per_cell_size():
    """Each inner axes occupies axes_size mm; verify by reading the
    per-axes bbox in figure-fraction and converting back to mm."""
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelGrid(label="C", shape=(1, 3), axes_size=(50.0, 30.0)))
    canvas.figure.canvas.draw()  # settle layout
    fig_w_in, fig_h_in = canvas.figure.get_size_inches()
    for ax in canvas["C"].axes.flat:
        bbox = ax.get_position()
        ax_w_mm = bbox.width * fig_w_in * 25.4
        ax_h_mm = bbox.height * fig_h_in * 25.4
        assert abs(ax_w_mm - 50.0) < 0.5  # mm precision (publiplots default)
        assert abs(ax_h_mm - 30.0) < 0.5


def test_panel_grid_sharex_propagates_to_inner_axes():
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelGrid(label="C", shape=(2, 3), axes_size=(40.0, 30.0),
                                 sharex=True))
    axes = canvas["C"].axes
    # sharex=True: all 6 axes share the same xaxis with axes[0,0].
    primary_x = axes[0, 0].get_shared_x_axes()
    for r in range(2):
        for c in range(3):
            if r == 0 and c == 0:
                continue
            assert primary_x.joined(axes[0, 0], axes[r, c])


def test_panel_grid_in_multi_row():
    """Mix PanelAxes and PanelGrid across rows."""
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(140.0, 40.0)))
    canvas.add_row(pp.PanelGrid(label="C", shape=(1, 3), axes_size=(40.0, 30.0)))
    assert canvas["A"].kind == "axes"
    assert canvas["C"].kind == "axesgrid"
```

- [ ] **Step 2: Implement PanelGrid construction in `_finalize_if_needed`**

Edit `/home/sagemaker-user/publiplots/src/publiplots/composer/canvas.py`. In the per-panel finalize loop, dispatch by panel type:

```python
        if isinstance(panel_input, PanelGrid):
            inner_axes = self._build_panel_grid_axes(
                panel_input, fig, outer_rect_mm, canvas_w_mm, canvas_h_mm,
            )
            panel = Panel(
                label=resolved_label,
                kind="axesgrid",
                ax=None,
                axes=inner_axes,         # 2D ndarray
                size_mm=panel_input.size_mm,
                bbox_mm=outer_bbox_mm,
                resolved_label_style=resolved_style,
            )
        elif isinstance(panel_input, PanelText):
            ax = self._build_panel_text_axes(panel_input, fig, outer_rect_mm, ...)
            panel = Panel(label=..., kind="text", ax=ax, axes=None, ...)
        else:  # PanelAxes
            ax = fig.add_axes(...)
            panel = Panel(label=..., kind="axes", ax=ax, axes=None, ...)
```

The `_build_panel_grid_axes` helper:

```python
    def _build_panel_grid_axes(self, panel_grid, fig, outer_rect_mm,
                                canvas_w_mm, canvas_h_mm):
        """Lay out the inner axes grid inside the panel's mm rect."""
        nr, nc = panel_grid.shape
        cell_w, cell_h = panel_grid.axes_size
        hspace, wspace = panel_grid.hspace, panel_grid.wspace

        x0_mm, y0_mm, w_mm, h_mm = outer_rect_mm

        import numpy as np
        from matplotlib.axes import Axes

        axes = np.empty((nr, nc), dtype=object)
        for r in range(nr):
            for c in range(nc):
                cell_x_mm = x0_mm + c * (cell_w + wspace)
                # y_mm is bottom-left; row 0 is at the TOP of the panel,
                # so its y is highest.
                cell_y_mm = y0_mm + (nr - 1 - r) * (cell_h + hspace)
                rect_frac = (
                    cell_x_mm / canvas_w_mm,
                    cell_y_mm / canvas_h_mm,
                    cell_w / canvas_w_mm,
                    cell_h / canvas_h_mm,
                )
                share_x = self._resolve_panel_grid_share(
                    panel_grid.sharex, axes, r, c, axis="x"
                )
                share_y = self._resolve_panel_grid_share(
                    panel_grid.sharey, axes, r, c, axis="y"
                )
                kwargs = {}
                if share_x is not None:
                    kwargs["sharex"] = share_x
                if share_y is not None:
                    kwargs["sharey"] = share_y
                axes[r, c] = fig.add_axes(rect_frac, **kwargs)
        return axes
```

Plus `_resolve_panel_grid_share` mirroring `pp.subplots`'s `_resolve_shared` (read `src/publiplots/layout/subplots.py:_resolve_shared` for the canonical implementation; copy verbatim or import + reuse).

The `Panel` dataclass needs a new field `axes: Optional[Any] = None` (numpy array of Axes). Add it to `panels.py`:

```python
@dataclass(frozen=True)
class Panel:
    label: Any
    kind: PanelKind
    ax: Optional[Any]
    size_mm: Tuple[float, float]
    bbox_mm: Tuple[float, float, float, float]
    resolved_label_style: Optional[Mapping[str, Any]] = None
    axes: Optional[Any] = None  # numpy ndarray of Axes for kind='axesgrid'
```

And the `Panel.ax` accessor for `kind='axesgrid'` should raise `AttributeError`. Since `Panel` is a frozen dataclass (no descriptor), the cleanest approach is to have `kind=axesgrid` set `ax=None` AND add a `__post_init__` check that fields are consistent (`ax is None xor axes is None`). The `AttributeError` from `panel.ax` access is achieved naturally because `ax=None` and we add a property for graceful access — actually no, `ax=None` means `panel.ax is None`, not raise.

Cleanest fix: change the test to `assert canvas["C"].ax is None` (rather than expecting AttributeError). OR: add a `Panel.ax` property that raises when `kind == "axesgrid"`. That requires making Panel non-frozen or overriding `__post_init__` to install a descriptor. Pragmatic decision: **Panel.ax stays as a frozen-dataclass field. It is None for kind='axesgrid'. Update the test from "AttributeError" to "is None".**

Update `test_panel_grid_ax_accessor_raises` to:

```python
def test_panel_grid_ax_is_none_use_axes_instead():
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelGrid(label="C", shape=(2, 3), axes_size=(40.0, 30.0)))
    panel = canvas["C"]
    assert panel.ax is None
    assert panel.axes is not None
```

- [ ] **Step 3: Run the panel-grid tests**

```bash
cd /home/sagemaker-user/publiplots
uv run pytest tests/composer/test_panel_grid.py -v --no-cov 2>&1 | tail -15
```

Expected: ALL pass.

- [ ] **Step 4: Full regression**

```bash
cd /home/sagemaker-user/publiplots
uv run pytest tests/composer/ -q --no-cov 2>&1 | tail -3
```

Expected: previous count + ~7 new panel-grid tests.

- [ ] **Step 5: Commit**

```bash
cd /home/sagemaker-user/publiplots
git add src/publiplots/composer/canvas.py \
        src/publiplots/composer/panels.py \
        tests/composer/test_panel_grid.py
git commit -m "feat(composer): PanelGrid sub-grid axes construction in finalize"
```

---

## Task 8: PanelText axes construction

**Files:**
- Modify: `src/publiplots/composer/canvas.py` — handle `PanelText` in `_finalize_if_needed`
- Test: extend `tests/composer/test_panel_text.py`

Same pattern as Task 7. `PanelText` creates a hidden axes inside the panel's mm rect, then places one `ax.text(0.5, 0.5, text, ha='center', va='center')` (with overrides from `text_kw`).

- [ ] **Step 1: Append failing canvas-integration tests**

Append to `/home/sagemaker-user/publiplots/tests/composer/test_panel_text.py`:

```python


# ---------------------------------------------------------------------------
# Canvas integration — text panel renders a hidden axes with text
# ---------------------------------------------------------------------------

import matplotlib
matplotlib.use("Agg")


def test_panel_text_canvas_kind_is_text():
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelText(label="E", text="hello", size=(60.0, 20.0)))
    assert canvas["E"].kind == "text"


def test_panel_text_canvas_renders_text_artist():
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelText(label="E", text="n = 1,234", size=(60.0, 20.0)))
    ax = canvas["E"].ax
    text_artists = [t for t in ax.texts if t.get_text() == "n = 1,234"]
    assert len(text_artists) == 1


def test_panel_text_axes_has_no_visible_spines():
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelText(label="E", text="x", size=(60.0, 20.0)))
    ax = canvas["E"].ax
    # PanelText hides spines + ticks via set_axis_off
    for spine in ax.spines.values():
        assert not spine.get_visible()


def test_panel_text_supports_mathtext():
    """Mathtext (e.g., r"$\\alpha$") is rendered by matplotlib's text engine.
    Smoke test: ensure rendering doesn't raise."""
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelText(label="E", text=r"$\alpha = 0.05$", size=(60.0, 20.0)))
    canvas.figure.canvas.draw()  # forces text rendering
    # If mathtext failed, draw would have raised.
    text_artists = [t for t in canvas["E"].ax.texts if "alpha" in t.get_text()]
    assert len(text_artists) == 1


def test_panel_text_text_kw_overrides_apply():
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelText(label="E", text="x", size=(60.0, 20.0),
                                 text_kw={"fontsize": 14, "color": "red"}))
    text_artists = [t for t in canvas["E"].ax.texts if t.get_text() == "x"]
    assert len(text_artists) == 1
    artist = text_artists[0]
    assert artist.get_fontsize() == 14
    assert artist.get_color() == "red"


def test_panel_text_in_multi_row():
    """Mix PanelAxes and PanelText across rows."""
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(140.0, 40.0)))
    canvas.add_row(pp.PanelText(label="E", text="caption", size=(140.0, 15.0)))
    assert canvas["A"].kind == "axes"
    assert canvas["E"].kind == "text"
```

- [ ] **Step 2: Implement PanelText construction**

Edit `/home/sagemaker-user/publiplots/src/publiplots/composer/canvas.py`. In `_finalize_if_needed`, add the `PanelText` branch:

```python
    def _build_panel_text_axes(self, panel_text, fig, outer_rect_mm,
                                canvas_w_mm, canvas_h_mm):
        """Create a hidden axes for a text panel and place the text."""
        x0_mm, y0_mm, w_mm, h_mm = outer_rect_mm
        rect_frac = (
            x0_mm / canvas_w_mm,
            y0_mm / canvas_h_mm,
            w_mm / canvas_w_mm,
            h_mm / canvas_h_mm,
        )
        ax = fig.add_axes(rect_frac)
        ax.set_axis_off()
        ax.patch.set_visible(False)

        text_kw = {"ha": "center", "va": "center"}
        text_kw.update(panel_text.text_kw or {})
        ax.text(0.5, 0.5, panel_text.text, transform=ax.transAxes, **text_kw)
        return ax
```

And dispatch:

```python
        if isinstance(panel_input, PanelText):
            ax = self._build_panel_text_axes(...)
            panel = Panel(label=..., kind="text", ax=ax, ...)
```

- [ ] **Step 3: Run tests**

```bash
cd /home/sagemaker-user/publiplots
uv run pytest tests/composer/test_panel_text.py -v --no-cov 2>&1 | tail -10
```

Expected: ALL pass.

- [ ] **Step 4: Commit**

```bash
cd /home/sagemaker-user/publiplots
git add src/publiplots/composer/canvas.py \
        tests/composer/test_panel_text.py
git commit -m "feat(composer): PanelText hidden-axes construction in finalize"
```

---

## Task 9: `canvas.align()` — alignment resolver + Canvas method

**Files:**
- Create: `src/publiplots/composer/alignment.py`
- Modify: `src/publiplots/composer/canvas.py` — add `align` method, run alignment in `_finalize_if_needed`
- Create: `tests/composer/test_alignment.py`

This is the alignment escape hatch. The implementation:

1. `canvas.align(panels, *, edge, mode='axes', anchor=None)` records the request into `self._alignments: List[_AlignmentRequest]`.
2. At finalize time, AFTER all panels have their axes positions computed, iterate `_alignments` and apply each shift.
3. Each shift uses the panel's CURRENT mm rect, computes the desired edge mm value (per `anchor` or default-leftmost-wins), and shifts the axes positions WITHIN their slots. If a shift would push axes outside the slot bounds, raise `ComposerAlignmentError`.

- [ ] **Step 1: Write the failing tests**

Create `/home/sagemaker-user/publiplots/tests/composer/test_alignment.py` with this EXACT content:

```python
"""Tests for canvas.align() — explicit alignment overrides."""

import matplotlib
matplotlib.use("Agg")
import pytest

import publiplots as pp
from publiplots.composer.exceptions import ComposerAlignmentError


MM_TOL = 0.01


# ---------------------------------------------------------------------------
# Construction + recording
# ---------------------------------------------------------------------------

def test_canvas_align_method_exists():
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(70.0, 40.0)))
    canvas.add_row(pp.PanelAxes(label="B", size=(70.0, 40.0)))
    canvas.align(["A", "B"], edge="left")
    assert len(canvas._alignments) == 1


def test_canvas_align_unknown_edge_raises():
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(70.0, 40.0)))
    with pytest.raises(ValueError, match="edge"):
        canvas.align(["A"], edge="invalid")


def test_canvas_align_unknown_mode_raises():
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(70.0, 40.0)))
    with pytest.raises(ValueError, match="mode"):
        canvas.align(["A"], edge="left", mode="weird")


def test_canvas_align_unknown_panel_raises():
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(70.0, 40.0)))
    with pytest.raises(KeyError, match="panel"):
        canvas.align(["A", "Z"], edge="left")


def test_canvas_align_after_finalize_raises():
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(70.0, 40.0)))
    canvas.finalize()
    with pytest.raises(RuntimeError, match="finalize"):
        canvas.align(["A"], edge="left")


# ---------------------------------------------------------------------------
# Alignment math — left edge
# ---------------------------------------------------------------------------

def test_align_left_makes_panels_share_left_edge():
    """Two panels in different rows with naturally different left edges
    (because their slots have different ylabel reservations) should
    end up with the same axes-bbox left edge after alignment."""
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(70.0, 40.0)))
    canvas.add_row(pp.PanelAxes(label="B", size=(70.0, 40.0)))
    # Without explicit align: row 1's panel B starts at x = outer_pad +
    # ylabel = 12mm. Same for A. They're already aligned by topology.
    # But test the explicit case anyway:
    canvas.align(["A", "B"], edge="left")
    canvas.finalize()
    a_x = canvas["A"].bbox_mm[0]
    b_x = canvas["B"].bbox_mm[0]
    assert abs(a_x - b_x) < MM_TOL


# ---------------------------------------------------------------------------
# Anchor — non-default
# ---------------------------------------------------------------------------

def test_align_anchor_specifies_which_edge_wins():
    """anchor='B' makes B's edge the reference; A shifts to match."""
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(70.0, 40.0)))
    canvas.add_row(pp.PanelAxes(label="B", size=(70.0, 40.0)))
    canvas.align(["A", "B"], edge="left", anchor="B")
    canvas.finalize()
    # Both end up at B's original left edge (which equals A's anyway in
    # this trivial case; just confirm no exception).
    a_x = canvas["A"].bbox_mm[0]
    b_x = canvas["B"].bbox_mm[0]
    assert abs(a_x - b_x) < MM_TOL


# ---------------------------------------------------------------------------
# Slot-boundary check — alignment can't push panels outside their slots
# ---------------------------------------------------------------------------

def test_align_shift_outside_slot_raises():
    """If aligning would require shifting panel B's content beyond what
    its slot can absorb, raise ComposerAlignmentError."""
    # Construct two panels where the slot widths leave no room for shift.
    # Tight test — depends on exact slot mm rects. Use a configuration
    # where panel A's slot allows only a tiny shift.
    canvas = pp.Canvas("custom", width=174.0)
    # Row 0: pinned panel at exact width, no slack
    canvas.add_row(pp.PanelAxes(label="A", size=(140.0, 40.0)))
    # Row 1: panel at a different position
    canvas.add_row(pp.PanelAxes(label="B", size=(70.0, 40.0)))
    # Try to align right — would require shifting B 70mm right, but
    # B's slot doesn't have 70mm of slack.
    with pytest.raises(ComposerAlignmentError) as exc_info:
        canvas.align(["A", "B"], edge="right")
        canvas.finalize()
    assert "B" in str(exc_info.value) or exc_info.value.panels


# ---------------------------------------------------------------------------
# Mode — axes vs tight (PR 3 ships axes mode primarily)
# ---------------------------------------------------------------------------

def test_align_mode_axes_default():
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(70.0, 40.0)))
    canvas.add_row(pp.PanelAxes(label="B", size=(70.0, 40.0)))
    canvas.align(["A", "B"], edge="left")  # mode='axes' is default
    canvas.finalize()
    # No exception; both share axes-bbox left edges.
    assert abs(canvas["A"].bbox_mm[0] - canvas["B"].bbox_mm[0]) < MM_TOL


def test_align_mode_tight_supported_or_documented():
    """mode='tight' aligns OUTER tightbboxes (incl. ticklabels). PR 3
    accepts the kwarg; full implementation may be PR 3.5 if tight-bbox
    measurement turns out non-trivial."""
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(70.0, 40.0)))
    # Just verify the kwarg is accepted; behavior is "best-effort" for v1.
    canvas.align(["A"], edge="left", mode="tight")
```

- [ ] **Step 2: Run, verify they fail**

Expected: `Canvas.align` doesn't exist; tests fail with AttributeError.

- [ ] **Step 3: Implement `alignment.py`**

Create `/home/sagemaker-user/publiplots/src/publiplots/composer/alignment.py` with this EXACT content:

```python
"""canvas.align() resolver — pure-Python edge/mode/anchor logic.

Each align call records an :class:`_AlignmentRequest` on the canvas;
at finalization time, the resolver iterates the requests and computes
the per-panel mm shifts. Slot mm-rects are inviolate — if a shift would
push axes outside their slot, raise ComposerAlignmentError.

PR 3 ships ``mode='axes'`` (axes-bbox edges) primarily. ``mode='tight'``
is accepted but may use the same path as 'axes' until PR 3.5/PR 4 add
proper tightbbox measurement.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


VALID_EDGES = frozenset({
    "left", "right", "top", "bottom",
    "center_x", "center_y", "baseline",
})
VALID_MODES = frozenset({"axes", "tight"})


@dataclass
class _AlignmentRequest:
    panels: Tuple[str, ...]
    edge: str
    mode: str
    anchor: Optional[str]


def _compute_target_mm(
    *,
    edge: str,
    panel_rects_mm: Dict[str, Tuple[float, float, float, float]],
    panels: Sequence[str],
    anchor: Optional[str],
) -> float:
    """Determine the target mm coord for the shared edge."""
    if anchor is not None:
        rect = panel_rects_mm[anchor]
    else:
        # Default convention: leftmost panel's edge wins for left/center_x;
        # topmost for top/center_y/baseline; etc.
        if edge in ("left", "center_x", "right"):
            # Pick the rect with the SMALLEST left for left/center_x;
            # SMALLEST right for right.
            if edge == "right":
                rect = max((panel_rects_mm[p] for p in panels),
                           key=lambda r: r[0] + r[2])
            else:
                rect = min((panel_rects_mm[p] for p in panels),
                           key=lambda r: r[0])
        else:  # top/bottom/center_y/baseline
            if edge in ("top", "center_y"):
                rect = max((panel_rects_mm[p] for p in panels),
                           key=lambda r: r[1] + r[3])
            else:
                rect = min((panel_rects_mm[p] for p in panels),
                           key=lambda r: r[1])
    x, y, w, h = rect
    if edge == "left":      return x
    if edge == "right":     return x + w
    if edge == "top":       return y + h
    if edge == "bottom":    return y
    if edge == "baseline":  return y      # PR 3 alias for bottom; refines later
    if edge == "center_x":  return x + w / 2
    if edge == "center_y":  return y + h / 2
    raise ValueError(f"unhandled edge {edge!r}")


def apply_alignments(
    *,
    requests: Sequence[_AlignmentRequest],
    panel_rects_mm: Dict[str, Tuple[float, float, float, float]],
    slot_rects_mm: Dict[str, Tuple[float, float, float, float]],
) -> Dict[str, Tuple[float, float, float, float]]:
    """Apply each alignment request, returning the updated panel rects.

    `slot_rects_mm` are the inviolate mm bounds within which each panel
    can be shifted. If a request would push a panel outside its slot,
    raises :class:`publiplots.composer.exceptions.ComposerAlignmentError`.
    """
    from publiplots.composer.exceptions import ComposerAlignmentError

    rects = dict(panel_rects_mm)
    for req in requests:
        target = _compute_target_mm(
            edge=req.edge,
            panel_rects_mm=rects,
            panels=req.panels,
            anchor=req.anchor,
        )
        for p in req.panels:
            x, y, w, h = rects[p]
            sx, sy, sw, sh = slot_rects_mm[p]
            # Compute the shift along the relevant axis.
            if req.edge in ("left", "right", "center_x"):
                if req.edge == "left":
                    new_x = target
                elif req.edge == "right":
                    new_x = target - w
                else:  # center_x
                    new_x = target - w / 2
                new_y = y
                # Check that new_x is inside the slot.
                if new_x < sx - 1e-6 or new_x + w > sx + sw + 1e-6:
                    raise ComposerAlignmentError(
                        f"alignment shift would push panel {p!r} outside its slot "
                        f"(slot x=[{sx:.2f},{sx+sw:.2f}], requested x=[{new_x:.2f},{new_x+w:.2f}])",
                        panels=tuple(req.panels),
                        edge=req.edge,
                    )
            else:  # top/bottom/center_y/baseline
                new_x = x
                if req.edge == "top":
                    new_y = target - h
                elif req.edge == "bottom" or req.edge == "baseline":
                    new_y = target
                else:  # center_y
                    new_y = target - h / 2
                if new_y < sy - 1e-6 or new_y + h > sy + sh + 1e-6:
                    raise ComposerAlignmentError(
                        f"alignment shift would push panel {p!r} outside its slot "
                        f"(slot y=[{sy:.2f},{sy+sh:.2f}], requested y=[{new_y:.2f},{new_y+h:.2f}])",
                        panels=tuple(req.panels),
                        edge=req.edge,
                    )
            rects[p] = (new_x, new_y, w, h)

    return rects
```

- [ ] **Step 4: Wire `align` into Canvas**

Edit `/home/sagemaker-user/publiplots/src/publiplots/composer/canvas.py`:

(a) Add `_alignments: List[_AlignmentRequest] = []` to `__init__`.

(b) Add a public `align` method (insert after `label_style`):

```python
    def align(
        self,
        panels: Sequence[str],
        *,
        edge: str,
        mode: str = "axes",
        anchor: Optional[str] = None,
    ) -> None:
        """Record an alignment request to apply at finalize time.

        Parameters
        ----------
        panels : sequence of str
            Panel labels to align.
        edge : str
            One of 'left', 'right', 'top', 'bottom', 'center_x',
            'center_y', 'baseline'.
        mode : str, default 'axes'
            'axes' (axes-bbox edges, default) or 'tight' (tightbbox).
        anchor : str, optional
            If given, this panel's edge is the reference. Otherwise the
            leftmost/topmost panel's edge wins per edge type.
        """
        from publiplots.composer.alignment import (
            _AlignmentRequest, VALID_EDGES, VALID_MODES,
        )
        if self._finalized:
            raise RuntimeError(
                "Canvas already finalized; align() must be called before "
                "any figure access (canvas.figure, canvas[...], canvas.savefig)"
            )
        if edge not in VALID_EDGES:
            raise ValueError(f"edge must be one of {sorted(VALID_EDGES)}, got {edge!r}")
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {sorted(VALID_MODES)}, got {mode!r}")
        # Verify the panels are known. We check against staged rows since
        # finalize hasn't run yet; collect labels from all staged rows.
        known_labels = set()
        for row in self._rows:
            for p in row.panels:
                if isinstance(p.label, str):
                    known_labels.add(p.label)
        # Plus any None-input panels that abc will resolve to letters —
        # for now, only validate against verbatim-str labels. None labels
        # can be referenced via canvas[i] integer indexing.
        for p in panels:
            if p not in known_labels:
                raise KeyError(
                    f"panel {p!r} not found among staged rows; known str-labeled "
                    f"panels: {sorted(known_labels)}"
                )
        if anchor is not None and anchor not in panels:
            raise ValueError(
                f"anchor must be one of the panels in the request; "
                f"got anchor={anchor!r}, panels={list(panels)}"
            )
        self._alignments.append(_AlignmentRequest(
            panels=tuple(panels),
            edge=edge,
            mode=mode,
            anchor=anchor,
        ))
```

(c) In `_finalize_if_needed`, AFTER computing `panel_rects_mm` for all panels but BEFORE creating the matplotlib axes, run the alignment resolver and update the rects:

```python
        if self._alignments:
            from publiplots.composer.alignment import apply_alignments
            # Build slot_rects_mm: the natural rect of each panel
            # (BEFORE alignment), serving as the inviolate slot.
            slot_rects_mm = dict(panel_rects_mm)
            panel_rects_mm = apply_alignments(
                requests=self._alignments,
                panel_rects_mm=panel_rects_mm,
                slot_rects_mm=slot_rects_mm,
            )
        # Then proceed to fig.add_axes(...) using the updated panel_rects_mm.
```

- [ ] **Step 5: Run alignment tests**

```bash
cd /home/sagemaker-user/publiplots
uv run pytest tests/composer/test_alignment.py -v --no-cov 2>&1 | tail -15
```

Expected: ALL pass.

- [ ] **Step 6: Full regression**

```bash
cd /home/sagemaker-user/publiplots
uv run pytest tests/composer/ -q --no-cov 2>&1 | tail -3
```

Expected: previous count + ~9 new alignment tests.

- [ ] **Step 7: Commit**

```bash
cd /home/sagemaker-user/publiplots
git add src/publiplots/composer/alignment.py \
        src/publiplots/composer/canvas.py \
        tests/composer/test_alignment.py
git commit -m "feat(composer): canvas.align() — explicit alignment within slots"
```

---

## Task 10: Gallery examples + CHANGELOG + open PR

**Files:**
- Create: `examples/composer/cell_2col_multirow.py`
- Create: `examples/composer/nature_2col_panel_grid.py`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Write `cell_2col_multirow.py`**

Create `/home/sagemaker-user/publiplots/examples/composer/cell_2col_multirow.py`:

```python
"""Cell 2-row figure — multi-row support, abc letters spanning rows.

Demonstrates:
- Cell-2col preset (174mm wide, abc='upper')
- Multi-row layout: row 0 has panels A+B, row 1 has panel C+D
- abc sequencing 'A','B','C','D' across rows
- canvas.align(['A', 'C'], edge='left') keeping the column visually crisp
- Saves to docs/images/composer/cell-2col-multirow.png
"""

from pathlib import Path

import numpy as np
import publiplots as pp


def main() -> None:
    rng = np.random.default_rng(0)
    canvas = pp.Canvas("cell-2col")

    canvas.add_row(
        pp.PanelAxes(label=None, size=(70, 40)),  # A
        pp.PanelAxes(label=None, size=("flex", 40)),  # B
    )
    canvas.add_row(
        pp.PanelAxes(label=None, size=(70, 40)),  # C
        pp.PanelText(label=None, text="n = 1,234\nP < 0.001",
                      size=("flex", 40)),  # D
    )

    canvas["A"].ax.scatter(rng.normal(0, 1, 200), rng.normal(0, 1, 200), s=4, alpha=0.6)
    canvas["A"].ax.set_xlabel("x"); canvas["A"].ax.set_ylabel("y")

    t = np.linspace(0, 10, 100)
    canvas["B"].ax.plot(t, np.sin(t) + rng.normal(0, 0.1, 100))
    canvas["B"].ax.set_xlabel("time"); canvas["B"].ax.set_ylabel("signal")

    cats = ["a", "b", "c", "d"]
    canvas["C"].ax.bar(cats, rng.uniform(1, 5, 4))
    canvas["C"].ax.set_xlabel("group"); canvas["C"].ax.set_ylabel("count")

    canvas.align(["A", "C"], edge="left")

    out_dir = Path(__file__).resolve().parents[2] / "docs" / "images" / "composer"
    out_dir.mkdir(parents=True, exist_ok=True)
    canvas.savefig(out_dir / "cell-2col-multirow.png")
    print(f"wrote {out_dir / 'cell-2col-multirow.png'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write `nature_2col_panel_grid.py`**

Create `/home/sagemaker-user/publiplots/examples/composer/nature_2col_panel_grid.py`:

```python
"""Nature 2-col figure — PanelGrid + PanelText, mathtext caption.

Demonstrates:
- nature-2col preset (183mm wide, abc='lower')
- PanelGrid: a 1×3 sub-grid of small lineplots inside one panel
- PanelText with mathtext caption
- Saves to docs/images/composer/nature-2col-panel-grid.png
"""

from pathlib import Path

import numpy as np
import publiplots as pp


def main() -> None:
    rng = np.random.default_rng(2)
    canvas = pp.Canvas("nature-2col")

    canvas.add_row(
        pp.PanelAxes(label=None, size=(60, 40)),  # a
        pp.PanelGrid(label=None, shape=(1, 3), axes_size=(35, 40)),  # b
    )
    canvas.add_row(
        pp.PanelText(label=None,
                      text=r"Linear regression on 100 samples ($\alpha = 0.05$)",
                      size=("flex", 8)),
    )

    # Plot in panel 'a'
    x = rng.normal(0, 1, 100)
    y = 0.5 * x + rng.normal(0, 0.5, 100)
    canvas["a"].ax.scatter(x, y, s=4, alpha=0.6)
    canvas["a"].ax.set_xlabel("x"); canvas["a"].ax.set_ylabel("y")

    # Plot in panel 'b' grid (3 small lineplots)
    for i, ax in enumerate(canvas["b"].axes.flat):
        t = np.linspace(0, 10, 50)
        ax.plot(t, np.sin(t * (i + 1)) + rng.normal(0, 0.1, 50), lw=0.8)
        ax.set_xlabel("t"); ax.set_ylabel(f"y{i+1}")

    out_dir = Path(__file__).resolve().parents[2] / "docs" / "images" / "composer"
    out_dir.mkdir(parents=True, exist_ok=True)
    canvas.savefig(out_dir / "nature-2col-panel-grid.png")
    print(f"wrote {out_dir / 'nature-2col-panel-grid.png'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run both examples**

```bash
cd /home/sagemaker-user/publiplots
uv run python examples/composer/cell_2col_multirow.py
uv run python examples/composer/nature_2col_panel_grid.py
```

Expected: each prints a "wrote ..." line; PNG outputs exist.

- [ ] **Step 4: Update CHANGELOG**

Edit `/home/sagemaker-user/publiplots/CHANGELOG.md`. Append to the existing `## [Unreleased] / ### Added` block (keep the PR 1 + PR 2 bullets):

```markdown
- `pp.Canvas` multi-row layouts — `add_row` may now be called multiple
  times. Rows stack top-to-bottom with configurable `vpad`. abc
  sequencing continues across rows. `canvas[i]` (integer index) walks
  panels in flat insertion order across rows.
- `pp.PanelGrid(label, *, shape, axes_size, sharex, sharey, hspace, wspace, label_style)` —
  a sub-grid of axes inside a single panel. ``canvas['<label>'].axes``
  returns a 2D numpy array of `Axes`.
- `pp.PanelText(label, *, text, size, text_kw, label_style)` — text-only
  panel (mathtext-supporting). Renders to a hidden axes with one
  centered `text`.
- `canvas.align(panels, *, edge, mode='axes', anchor=None)` — explicit
  alignment override. Edges: `'left'`, `'right'`, `'top'`, `'bottom'`,
  `'center_x'`, `'center_y'`, `'baseline'`. Modes: `'axes'` /
  `'tight'`. Slot mm-rects are inviolate; alignment shifts content
  WITHIN slots. Raises `ComposerAlignmentError` if a shift would push
  axes outside their slot.
- `canvas.finalize()` — explicit no-op-if-already-finalized helper to
  materialize the figure without accessing `.figure`. The figure is
  also finalized lazily on first access of `canvas.figure`,
  `canvas.figure_size_mm`, `canvas[label]`, `canvas[i]`, or
  `canvas.savefig`.
- `ComposerAlignmentError` — new exception subclass of `ComposerError`,
  carrying `panels` and `edge` attributes.
- Two new gallery examples under ``examples/composer/`` showing
  multi-row + PanelGrid + PanelText + canvas.align.
```

- [ ] **Step 5: Run full test suite**

```bash
cd /home/sagemaker-user/publiplots
uv run pytest tests/ -q --no-cov --tb=no 2>&1 | tail -3
```

Expected: ~1175+ passed, 1 pre-existing residplot failure (unchanged).

- [ ] **Step 6: Commit examples + CHANGELOG**

```bash
cd /home/sagemaker-user/publiplots
git add examples/composer/cell_2col_multirow.py \
        examples/composer/nature_2col_panel_grid.py \
        docs/images/composer/cell-2col-multirow.png \
        docs/images/composer/nature-2col-panel-grid.png \
        CHANGELOG.md
git commit -m "docs(composer): multi-row + panel-grid gallery + CHANGELOG"
```

- [ ] **Step 7: Push + open PR**

```bash
cd /home/sagemaker-user/publiplots
git push -u origin feat/composer-multirow-grid-align-pr3
gh pr create \
  --title "feat(composer): multi-row + PanelGrid + PanelText + canvas.align" \
  --body "$(cat <<'EOF'
## Summary

PR 3 of the Composer rollout. Lifts the single-row restriction; adds `PanelGrid` (sub-grid of axes inside a panel) + `PanelText` (text-only panel); adds `canvas.align(panels, edge, mode)` for explicit alignment overrides; adds `canvas.finalize()` + lazy figure materialization.

After this PR lands, the spec's worked-example multi-row figure (schematic + scatter + 1×3 grid + barplot+text caption) is producible end-to-end except for the schematic image (`PanelImage` lands in PR 5).

## What's in this PR

- **Multi-row layouts** — `add_row` callable repeatedly. abc sequencing continues across rows. Heterogeneous columns per row supported.
- **`PanelGrid`** — `(shape=(r,c), axes_size=(w,h), sharex, sharey, hspace, wspace)`. `canvas[label].axes` returns a 2D ndarray of `Axes`.
- **`PanelText`** — text-only panel rendering to a hidden axes with one `ax.text(...)`. Mathtext supported.
- **`canvas.align(panels, edge, mode='axes', anchor=None)`** — explicit alignment within inviolate slots. Edges: left/right/top/bottom/center_x/center_y/baseline. `ComposerAlignmentError` raised if a shift would exit the slot.
- **Lazy finalization** — figure created on first access (`.figure`, `[label]`, `.savefig`) or via explicit `canvas.finalize()`.
- **`ComposerAlignmentError`** — new exception subclass of `ComposerError`.
- **2 gallery examples**.

## What's NOT in this PR

- `pp.legend` rows/cols/span/ax (PR 4)
- mm-precision regression test infra (PR 4.5)
- vector PDF/SVG (PR 5/6) — `canvas.savefig('fig.pdf')` still raises NotImplementedError; PanelImage ingestion errors with a PR 5 hint
- canvas.inspect() + composer-guide skill (PR 7)
- Tight-bbox alignment mode (`mode='tight'`) accepts the kwarg but uses the axes-bbox path; full tightbbox measurement deferred to PR 4.5
- `valign` modes other than `'top'`

## Spike findings status

- ✅ Finding 4 (PR 1) — decoration clipping
- 🚧 Findings 1, 2, 3, 5, 6, 7 — queued for PRs 5/6

## Test plan

- [x] All ~190 composer tests pass (was 140 in PR 2; +~50 new)
- [x] Full suite green (~1175 passed, 1 pre-existing residplot failure unchanged)
- [x] Multi-row geometry verified mm-precise via `_layout.py` unit tests
- [x] `canvas.align()` slot-bounds enforcement raises `ComposerAlignmentError` cleanly
- [x] PanelGrid sub-grid axes have correct per-cell mm dims and sharex/sharey propagation
- [x] PanelText renders mathtext without raising
- [x] Two gallery examples render to PNG without errors

## Implementation notes

- `_layout.py` is pure-Python (no matplotlib) — separates multi-row geometry math from figure creation.
- `_RowStaging` records each `add_row` call into `Canvas._rows`; finalization runs once over the full list.
- `_finalize_if_needed` is idempotent: subsequent calls are no-ops.
- Reactor strategy: `SubplotsAutoLayout` is attached only for SINGLE-row canvases. Multi-row canvases use static rcParams reservations (PR 4.5 will refine when snapshot infra lands).
- `canvas.py` was refactored in Task 5 (no behavioral change) before multi-row support landed in Task 6.
- The duplicate-label check in `add_row` correctly skips `None`/`False` labels.

## Follow-ups

PR 4 starts from this branch's tip. `pp.legend` rows/cols/span/ax kwargs.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
gh pr view --json number,url,state,title
```

Capture the PR URL for the human.

---

## Acceptance criteria for PR 3

The PR is ready for human merge when ALL of:

1. All 11 tasks complete.
2. Full test suite green (~1175 passed; 1 pre-existing residplot failure unchanged).
3. `add_row` callable multiple times.
4. `PanelGrid` produces a 2D numpy array of axes accessible via `canvas[label].axes`.
5. `PanelText` renders mathtext without errors.
6. `canvas.align(panels, edge)` enforces slot bounds (raises `ComposerAlignmentError` when a shift would exit the slot).
7. `canvas.finalize()` is idempotent + triggered lazily on figure access.
8. `canvas.savefig('fig.pdf')` and `canvas.savefig('fig.svg')` STILL raise `NotImplementedError` (PR 5/PR 6 deferral preserved).
9. `PanelImage` panels (PR 5 territory) raise `NotImplementedError` with a PR 5 pointer.
10. CHANGELOG entry added under `[Unreleased]` extending PR 1+2's `### Added`.
11. Two gallery examples render to PNG without errors.
12. Out-of-scope items (legend kwargs, vector save, inspect, skill, mode='tight' beyond axes-bbox path) are NOT in the diff.

If any of #1–#12 fail, the PR is not ready.
