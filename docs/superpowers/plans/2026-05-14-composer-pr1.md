# Composer PR 1 Implementation Plan — Canvas + PanelAxes (single-row, raster save)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the Composer's foundation: `pp.Canvas('custom', width=N)` + `pp.PanelAxes(label, size=(w_mm, h_mm))` + `canvas.add_row(*panels)` + `canvas[label].ax` indexing + `canvas.savefig(path)` for raster outputs (PNG/JPG/TIFF). Single-row layouts only. End-to-end usable for axes-only multi-panel figures with raster output.

**Architecture:** New `publiplots.composer` subpackage built on top of the existing `FigureLayout` + `SubplotsAutoLayout` machinery — Composer is a thin orchestrator that constructs a `FigureLayout(nrows=1, ncols=N)`, places one Axes per panel, and reuses `SubplotsAutoLayout` to reserve decoration space (xlabel/ylabel/title) so panels don't clip at the canvas edge (spike Finding 4). `canvas.savefig` delegates to the existing `pp.savefig` for rasters; vector PDF/SVG dispatch is deferred to PRs 5/6 and currently raises `NotImplementedError` with a clear hint.

**Tech Stack:** Python ≥ 3.10, matplotlib, pytest. No new dependencies. Editable install at `/home/sagemaker-user/publiplots/`.

**Spec reference:** `docs/superpowers/specs/2026-05-14-composer-design.md`.
**Spike reference:** `spikes/composer/composer-spike.md` (verdict YELLOW, with 7 findings — Finding 4 lands in this PR; Findings 1-3 + 5-7 land in PRs 5-6).

---

## What's IN scope for PR 1

- `pp.Canvas('custom', width=<mm>)` — only the `'custom'` preset (journal presets land in PR 2)
- `pp.PanelAxes(label, size=(w_mm, h_mm))` — only `PanelAxes` (PanelGrid/PanelImage/PanelText land in PR 3+5)
- `canvas.add_row(*panels)` — single-row only; calling `add_row` more than once raises `NotImplementedError` with a "multi-row support lands in PR 3" message
- `canvas[label]` returns a `Panel` object exposing `.ax`, `.label`, `.size_mm`, `.bbox_mm`
- `canvas.figure` exposes the underlying matplotlib Figure
- `canvas.savefig(path)` raster pipeline (PNG/JPG/TIFF) via `pp.savefig`; PDF/SVG → `NotImplementedError` for now
- `canvas.figure_size_mm` — read-only `(w_mm, h_mm)` tuple
- Auto-grow canvas height from panels' declared `h_mm` plus `xlabel_space`/`title_space` reservations (the spike Finding 4 fix)
- Width fixed by `Canvas(width=N)`; row width validated to fit
- Per-canvas `outer_pad`, `hpad` (between panels) configurable; sensible defaults from rcParams

## What's OUT of scope for PR 1

- Journal presets (`'cell-2col'`, `'nature-2col'`, etc.) → **PR 2**
- abc panel labels → **PR 2**
- Flex sizing (`size=('flex', h_mm)`) → **PR 2**
- Overflow advisory with suggested scaling factor → **PR 2**
- `PanelGrid`, `PanelText` → **PR 3**
- `canvas.align(panels, edge, mode)` → **PR 3**
- `pp.legend` `rows=`/`cols=`/`span=`/`ax=` upgrade → **PR 4**
- `mm-precision` regression test infra → **PR 4.5**
- `PanelImage` (vector PDF/SVG schematics) → **PR 5**
- `canvas.embed_figure(panel, fig)` → **PR 6**
- `canvas.inspect()` introspection helper + composer-guide skill → **PR 7**
- Multi-row layouts (`add_row` called more than once) → **PR 3**

This list is the contract for the spec-compliance reviewer. Anything in this list that the implementer ships counts as scope creep and should be flagged.

---

## File Structure

```
src/publiplots/composer/                       ← NEW subpackage
├── __init__.py                                # public re-exports: Canvas, PanelAxes
├── canvas.py                                  # Canvas class
├── panels.py                                  # PanelAxes dataclass + Panel result type
├── presets.py                                 # PRESETS dict — only 'custom' in PR 1
├── exceptions.py                              # ComposerError, ComposerOverflowError
└── _save.py                                   # save dispatch (raster works, PDF/SVG NotImplementedError)

src/publiplots/__init__.py                     # MODIFY — add Canvas + PanelAxes to public API

tests/composer/                                ← NEW test directory
├── __init__.py                                # empty (so pytest finds the package)
├── conftest.py                                # close_figures fixture + Agg backend
├── test_canvas_construction.py                # Canvas() validation, .figure, .figure_size_mm
├── test_panels.py                             # PanelAxes dataclass, label/size validation
├── test_add_row.py                            # add_row geometry, multi-call rejection
├── test_indexing.py                           # canvas[label].ax, KeyError on bad label
├── test_savefig.py                            # raster works, PDF/SVG NotImplementedError
├── test_layout_integration.py                 # Composer-built figure obeys mm precision
└── test_decoration_reservations.py            # spike Finding 4 — xlabel doesn't clip
```

8 new files + 1 modified file. Estimated ~600 LOC code + ~700 LOC tests.

**Layering invariants this plan upholds (from spec §"Architecture"):**
- `composer/canvas.py` and `composer/panels.py` are matplotlib-aware (they create Figures and Axes)
- `composer/presets.py` and `composer/exceptions.py` are pure-Python (no matplotlib imports)
- `composer/_save.py` is matplotlib-aware (it dispatches to `pp.savefig`)
- The pure-geometry layer (`FigureLayout` and `SubplotsAutoLayout`) is REUSED, not reimplemented

**File-size budget (warn if exceeded; flag for review):**
- `canvas.py`: ≤ 250 LOC
- `panels.py`: ≤ 80 LOC
- `presets.py`: ≤ 40 LOC
- `_save.py`: ≤ 60 LOC
- `exceptions.py`: ≤ 30 LOC
- Each test file: ≤ 200 LOC

Going over the budget is allowed if necessary, but the implementer MUST flag it as `DONE_WITH_CONCERNS`.

---

## Branch + worktree setup (Task 0)

**Files:** none (git only)

- [ ] **Step 1: Sync main**

```bash
cd /home/sagemaker-user/publiplots
git checkout main
git pull origin main --ff-only
```

Expected: HEAD is at `769e1a0 spike(composer): vector pipeline derisking (#162)` (or later if other PRs merged in the meantime).

- [ ] **Step 2: Create the feature branch**

```bash
cd /home/sagemaker-user/publiplots
git checkout -b feat/composer-canvas-pr1
```

Expected: `git branch --show-current` prints `feat/composer-canvas-pr1`.

- [ ] **Step 3: Run the existing test suite once as a baseline**

```bash
cd /home/sagemaker-user/publiplots
uv run pytest tests/ -q --tb=no
```

Expected: all tests pass before we touch anything (so any failure later is attributable to PR 1's changes, not to a pre-existing bug).

Capture the test count.

- [ ] **Step 4: Commit-free checkpoint** — no commit yet, this task is just workspace prep.

---

## Task 1: Exception types + presets table (pure-Python foundation)

**Files:**
- Create: `src/publiplots/composer/__init__.py` (initially empty stub re-exports added later)
- Create: `src/publiplots/composer/exceptions.py`
- Create: `src/publiplots/composer/presets.py`
- Test: `tests/composer/__init__.py`, `tests/composer/conftest.py`, `tests/composer/test_canvas_construction.py` (the construction-side cases that need exceptions)

**Why first:** these have zero matplotlib dependency, and Tasks 2-5 all import them. Land them once with their own tests so later tasks can assume they work.

- [ ] **Step 1: Write the failing test for ComposerError + ComposerOverflowError**

Create `tests/composer/__init__.py` (empty file).

Create `tests/composer/conftest.py` with this exact content:

```python
"""Shared pytest fixtures for tests/composer/."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest


@pytest.fixture(autouse=True)
def _close_figures():
    """Close all matplotlib figures after each test to keep memory clean."""
    yield
    plt.close("all")
```

Create `tests/composer/test_canvas_construction.py` with this exact content (only the exceptions-related tests for now — Task 2 fills in the rest):

```python
"""Construction tests for pp.Canvas — exceptions, presets, validation."""

import pytest

import publiplots as pp
from publiplots.composer.exceptions import (
    ComposerError,
    ComposerOverflowError,
)


def test_composer_error_is_value_error_subclass():
    """ComposerError is the base class for all composer-specific errors;
    inheriting from ValueError lets users catch with the standard idiom."""
    assert issubclass(ComposerError, ValueError)


def test_composer_overflow_error_is_composer_error():
    assert issubclass(ComposerOverflowError, ComposerError)


def test_composer_overflow_error_carries_offending_dim():
    """ComposerOverflowError exposes the requested + available dims so
    callers (and Task 2's row-width validator) can format helpful
    messages without re-parsing the str."""
    err = ComposerOverflowError(
        "row 0 width 200mm exceeds canvas budget 174mm",
        requested_mm=200.0,
        available_mm=174.0,
    )
    assert err.requested_mm == 200.0
    assert err.available_mm == 174.0
    assert "200" in str(err)
    assert "174" in str(err)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /home/sagemaker-user/publiplots
uv run pytest tests/composer/test_canvas_construction.py -v
```

Expected: `ImportError: No module named 'publiplots.composer'` or similar — fails because the module doesn't exist yet.

- [ ] **Step 3: Implement `exceptions.py`**

Create `src/publiplots/composer/__init__.py` with this exact content (we'll grow it later):

```python
"""publiplots Composer — multi-panel paper-figure builder.

Public API for PR 1 (single-row, axes-only):
- :class:`Canvas` — programmatic mm-precise canvas
- :class:`PanelAxes` — axes panel constructor
- :class:`Panel` — result type returned by ``canvas[label]``

See ``docs/superpowers/specs/2026-05-14-composer-design.md`` for the
full design.
"""

from publiplots.composer.exceptions import (
    ComposerError,
    ComposerOverflowError,
)

__all__ = [
    "ComposerError",
    "ComposerOverflowError",
]
```

Create `src/publiplots/composer/exceptions.py` with this exact content:

```python
"""Composer-specific exceptions.

All composer errors inherit from ComposerError (which itself is a
ValueError) so callers can catch with either the composer-specific or
standard-library type.
"""

from typing import Optional


class ComposerError(ValueError):
    """Base class for all publiplots Composer errors."""


class ComposerOverflowError(ComposerError):
    """Raised when a row's panels overflow the canvas width budget.

    Carries ``requested_mm`` and ``available_mm`` attributes so callers
    (and PR 2's overflow advisor) can compute the suggested per-row
    scaling factor without re-parsing the message text.
    """

    def __init__(
        self,
        message: str,
        *,
        requested_mm: float,
        available_mm: float,
    ) -> None:
        super().__init__(message)
        self.requested_mm = float(requested_mm)
        self.available_mm = float(available_mm)
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd /home/sagemaker-user/publiplots
uv run pytest tests/composer/test_canvas_construction.py -v
```

Expected: 3 tests PASS.

- [ ] **Step 5: Write the failing test for `presets.py` (only `'custom'` in PR 1)**

Append to `tests/composer/test_canvas_construction.py`:

```python
# ---------------------------------------------------------------------------
# presets.py — only 'custom' in PR 1; journal presets land in PR 2
# ---------------------------------------------------------------------------
from publiplots.composer.presets import PRESETS, resolve_preset


def test_only_custom_preset_in_pr1():
    """PR 1 ships ONLY the 'custom' preset. Journal presets (cell, nature,
    nature-methods, science) land in PR 2. This test will be UPDATED by
    PR 2; in PR 1 it pins the preset list to exactly {'custom'} so we
    don't accidentally ship a half-baked Cell preset."""
    assert set(PRESETS.keys()) == {"custom"}


def test_resolve_preset_custom_requires_width():
    """'custom' has no default width — caller must supply ``width=``."""
    with pytest.raises(ValueError, match="width"):
        resolve_preset("custom", width=None)


def test_resolve_preset_custom_returns_width():
    p = resolve_preset("custom", width=174.0)
    assert p["width_mm"] == 174.0


def test_resolve_preset_unknown_raises():
    with pytest.raises(KeyError, match="unknown preset"):
        resolve_preset("not-a-preset", width=100.0)


def test_resolve_preset_rejects_non_positive_width():
    with pytest.raises(ValueError, match="positive"):
        resolve_preset("custom", width=0.0)
    with pytest.raises(ValueError, match="positive"):
        resolve_preset("custom", width=-10.0)
```

- [ ] **Step 6: Run to verify the new tests fail**

```bash
cd /home/sagemaker-user/publiplots
uv run pytest tests/composer/test_canvas_construction.py -v
```

Expected: the 3 original tests still PASS; 5 new tests FAIL with `ModuleNotFoundError: No module named 'publiplots.composer.presets'`.

- [ ] **Step 7: Implement `presets.py`**

Create `src/publiplots/composer/presets.py` with this exact content:

```python
"""Canvas presets for the Composer.

PR 1 ships only the ``'custom'`` preset. Journal presets (Cell, Nature,
Nature Methods, Science) land in PR 2 with verified mm dimensions. To
add a journal preset later, add an entry to :data:`PRESETS` here and
extend :func:`resolve_preset` to handle its specific defaults.

Pure-Python module — no matplotlib imports. Pure data + a tiny resolver.
"""

from typing import Any, Dict, Optional


# PR 1: only 'custom'. PR 2 adds: 'cell-1col', 'cell-1.5col', 'cell-2col',
# 'nature-1col', 'nature-1.5col', 'nature-2col', 'nature-methods-*',
# 'science-1col', 'science-2col'.
PRESETS: Dict[str, Dict[str, Any]] = {
    "custom": {
        # 'custom' has no preset width; the caller MUST supply width=.
        "default_width_mm": None,
        "max_height_mm": None,  # no enforcement for 'custom' in PR 1
    },
}


def resolve_preset(
    name: str,
    *,
    width: Optional[float],
) -> Dict[str, Any]:
    """Resolve a preset name + user width into a config dict.

    Parameters
    ----------
    name : str
        Preset key (e.g., ``'custom'``). Must be in :data:`PRESETS`.
    width : float or None
        User-supplied canvas width in mm. Required for ``'custom'``;
        optional for journal presets (which provide a default width).

    Returns
    -------
    dict
        Keys: ``width_mm`` (float), ``max_height_mm`` (float or None).

    Raises
    ------
    KeyError
        If ``name`` is not a known preset.
    ValueError
        If ``width`` is None for a preset that requires it, or if
        ``width`` is non-positive.
    """
    if name not in PRESETS:
        raise KeyError(
            f"unknown preset {name!r}; known presets: {sorted(PRESETS)}"
        )
    spec = PRESETS[name]
    default_width = spec["default_width_mm"]
    if width is None:
        if default_width is None:
            raise ValueError(
                f"preset {name!r} has no default width; pass width=<mm>"
            )
        width = default_width
    if width <= 0:
        raise ValueError(f"width must be positive, got {width}")
    return {
        "width_mm": float(width),
        "max_height_mm": spec["max_height_mm"],
    }
```

- [ ] **Step 8: Run to verify all tests pass**

```bash
cd /home/sagemaker-user/publiplots
uv run pytest tests/composer/test_canvas_construction.py -v
```

Expected: 8 tests PASS.

- [ ] **Step 9: Commit**

```bash
cd /home/sagemaker-user/publiplots
git add src/publiplots/composer/__init__.py \
        src/publiplots/composer/exceptions.py \
        src/publiplots/composer/presets.py \
        tests/composer/__init__.py \
        tests/composer/conftest.py \
        tests/composer/test_canvas_construction.py
git commit -m "feat(composer): foundation — exceptions + custom preset"
```

---

## Task 2: PanelAxes dataclass + Panel result type

**Files:**
- Create: `src/publiplots/composer/panels.py`
- Test: `tests/composer/test_panels.py`

**Why second:** `Canvas.add_row` (Task 3) takes `PanelAxes` instances; we need the dataclass + validation in place first.

- [ ] **Step 1: Write the failing test**

Create `tests/composer/test_panels.py` with this exact content:

```python
"""Tests for PanelAxes dataclass + the Panel result type."""

import pytest

from publiplots.composer.panels import Panel, PanelAxes


# ---------------------------------------------------------------------------
# PanelAxes — input dataclass passed to canvas.add_row(*panels)
# ---------------------------------------------------------------------------

def test_panel_axes_basic_construction():
    p = PanelAxes(label="A", size=(70.0, 40.0))
    assert p.label == "A"
    assert p.size == (80.0, 40.0)


def test_panel_axes_label_required_in_pr1():
    """PR 2 will introduce auto-letter sequencing (label=None → 'A','B',...).
    PR 1 requires an explicit label string. This test will be UPDATED in
    PR 2 to allow label=None; pin to required-string for now."""
    with pytest.raises(TypeError):
        PanelAxes(size=(70.0, 40.0))  # no label


def test_panel_axes_label_must_be_str():
    with pytest.raises(TypeError, match="label must be a string"):
        PanelAxes(label=123, size=(70.0, 40.0))


def test_panel_axes_size_required():
    with pytest.raises(TypeError):
        PanelAxes(label="A")  # no size


def test_panel_axes_size_must_be_2_tuple():
    with pytest.raises(ValueError, match="size must be a 2-tuple"):
        PanelAxes(label="A", size=80.0)
    with pytest.raises(ValueError, match="size must be a 2-tuple"):
        PanelAxes(label="A", size=(80.0,))
    with pytest.raises(ValueError, match="size must be a 2-tuple"):
        PanelAxes(label="A", size=(80.0, 40.0, 10.0))


def test_panel_axes_size_must_be_positive():
    with pytest.raises(ValueError, match="positive"):
        PanelAxes(label="A", size=(0.0, 40.0))
    with pytest.raises(ValueError, match="positive"):
        PanelAxes(label="A", size=(80.0, -1.0))


def test_panel_axes_size_must_be_numeric():
    with pytest.raises(ValueError, match="numeric"):
        PanelAxes(label="A", size=("flex", 40.0))


def test_panel_axes_size_coerces_to_floats():
    """Integers are accepted and stored as floats (mm precision matters)."""
    p = PanelAxes(label="A", size=(70, 40))
    assert p.size == (80.0, 40.0)
    assert isinstance(p.size[0], float)
    assert isinstance(p.size[1], float)


# ---------------------------------------------------------------------------
# Panel — result type returned by canvas[label]
# ---------------------------------------------------------------------------

def test_panel_exposes_label_and_kind():
    """Panel is a frozen dataclass-like view; label/kind are read-only."""
    # Use a sentinel ax (None) for unit-test purposes; integration tests
    # in test_indexing.py use a real Axes via canvas[label].
    p = Panel(
        label="A",
        kind="axes",
        ax=None,
        size_mm=(80.0, 40.0),
        bbox_mm=(5.0, 5.0, 80.0, 40.0),
    )
    assert p.label == "A"
    assert p.kind == "axes"
    assert p.size_mm == (80.0, 40.0)
    assert p.bbox_mm == (5.0, 5.0, 80.0, 40.0)


def test_panel_axes_attribute_accessible_for_axes_kind():
    """For kind='axes', .ax returns whatever the canvas stored (a real
    Axes in production; None in this unit test)."""
    p = Panel(label="A", kind="axes", ax=None,
              size_mm=(80.0, 40.0), bbox_mm=(5.0, 5.0, 80.0, 40.0))
    # .ax is just an attribute — no special accessor logic in PR 1.
    assert p.ax is None  # The integration tests check the real-Axes case.


def test_panel_is_immutable():
    """Panel is a frozen dataclass — users shouldn't mutate it."""
    p = Panel(label="A", kind="axes", ax=None,
              size_mm=(80.0, 40.0), bbox_mm=(5.0, 5.0, 80.0, 40.0))
    with pytest.raises((AttributeError, Exception)):  # FrozenInstanceError or AttributeError
        p.label = "B"
```

- [ ] **Step 2: Run to verify it fails**

```bash
cd /home/sagemaker-user/publiplots
uv run pytest tests/composer/test_panels.py -v
```

Expected: all FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `panels.py`**

Create `src/publiplots/composer/panels.py` with this exact content:

```python
"""Panel constructors and result types for the Composer.

PR 1 ships only :class:`PanelAxes`. PR 3 adds :class:`PanelGrid` /
:class:`PanelText`; PR 5 adds :class:`PanelImage`.

Pure-Python module — no matplotlib imports. The result type
:class:`Panel` carries an opaque ``ax`` reference that the canvas
populates; that's the only object that crosses the Python/matplotlib
boundary, and it's typed as ``Any`` here for layering purity.
"""

from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Tuple


PanelKind = Literal["axes", "axesgrid", "image", "text"]  # only "axes" used in PR 1


@dataclass(frozen=True)
class PanelAxes:
    """Input record for an axes panel.

    Pass this to :meth:`Canvas.add_row` to declare a panel containing a
    single matplotlib Axes. The canvas allocates the Axes at the panel's
    mm rect and stores the result as a :class:`Panel`.

    Parameters
    ----------
    label : str
        Caption-addressable identifier (``'A'``, ``'B'``, ``'a.i'``, ...).
        PR 1 requires an explicit string; PR 2 adds auto-letter
        sequencing for ``label=None``.
    size : tuple of (width_mm, height_mm)
        Panel mm rect. Both dimensions must be positive numerics. PR 2
        adds ``'flex'`` for the width to absorb leftover row space.

    Notes
    -----
    Frozen dataclass — instances are immutable and hashable.
    """

    label: str
    size: Tuple[float, float]

    def __post_init__(self) -> None:
        # Validate label
        if not isinstance(self.label, str):
            raise TypeError(
                f"label must be a string, got {type(self.label).__name__}"
            )

        # Validate size: must be a 2-tuple of positive numerics
        try:
            n = len(self.size)
        except TypeError:
            raise ValueError(
                f"size must be a 2-tuple of (width_mm, height_mm), got {self.size!r}"
            )
        if n != 2:
            raise ValueError(
                f"size must be a 2-tuple of (width_mm, height_mm), got length {n}"
            )
        w, h = self.size
        for name, v in (("width", w), ("height", h)):
            if isinstance(v, str):
                raise ValueError(
                    f"size {name} must be numeric (PR 2 will add 'flex'), got {v!r}"
                )
            try:
                fv = float(v)
            except (TypeError, ValueError):
                raise ValueError(
                    f"size {name} must be numeric, got {v!r}"
                )
            if fv <= 0:
                raise ValueError(f"size {name} must be positive, got {fv}")

        # Coerce to floats. dataclass(frozen=True) requires object.__setattr__.
        object.__setattr__(self, "size", (float(w), float(h)))


@dataclass(frozen=True)
class Panel:
    """Result type returned by ``canvas[label]``.

    Carries the resolved axes reference and the panel's mm geometry.
    Frozen so users don't accidentally mutate cached metadata.

    For ``kind='axes'``, ``ax`` is a single :class:`matplotlib.axes.Axes`.
    PR 3 adds ``kind='axesgrid'`` (where ``axes`` is a numpy array) and
    other panel kinds.

    Attributes
    ----------
    label : str
        The caption-addressable identifier.
    kind : str
        One of ``'axes'``, ``'axesgrid'``, ``'image'``, ``'text'``.
        PR 1 only emits ``'axes'``.
    ax : matplotlib.axes.Axes or None
        The underlying axes for ``kind='axes'``. ``None`` for non-axes
        panels (PR 5+).
    size_mm : tuple of (width_mm, height_mm)
        Panel mm rect dimensions.
    bbox_mm : tuple of (x_mm, y_mm, w_mm, h_mm)
        Panel mm rect position relative to the canvas. ``(x, y)`` is the
        bottom-left corner; ``(w, h)`` matches ``size_mm``.
    """

    label: str
    kind: PanelKind
    ax: Optional[Any]  # matplotlib.axes.Axes | None — typed Any for layering
    size_mm: Tuple[float, float]
    bbox_mm: Tuple[float, float, float, float]
```

- [ ] **Step 4: Run to verify all tests pass**

```bash
cd /home/sagemaker-user/publiplots
uv run pytest tests/composer/test_panels.py -v
```

Expected: 11 tests PASS.

- [ ] **Step 5: Re-export from the composer package**

Edit `src/publiplots/composer/__init__.py` to add the new names:

```python
"""publiplots Composer — multi-panel paper-figure builder.

Public API for PR 1 (single-row, axes-only):
- :class:`Canvas` — programmatic mm-precise canvas
- :class:`PanelAxes` — axes panel constructor
- :class:`Panel` — result type returned by ``canvas[label]``

See ``docs/superpowers/specs/2026-05-14-composer-design.md`` for the
full design.
"""

from publiplots.composer.exceptions import (
    ComposerError,
    ComposerOverflowError,
)
from publiplots.composer.panels import Panel, PanelAxes

__all__ = [
    "ComposerError",
    "ComposerOverflowError",
    "Panel",
    "PanelAxes",
]
```

- [ ] **Step 6: Verify the package-level import works**

```bash
cd /home/sagemaker-user/publiplots
uv run python -c "from publiplots.composer import PanelAxes, Panel; p = PanelAxes(label='A', size=(70, 40)); print(p)"
```

Expected: prints `PanelAxes(label='A', size=(70.0, 40.0))`.

- [ ] **Step 7: Commit**

```bash
cd /home/sagemaker-user/publiplots
git add src/publiplots/composer/panels.py \
        src/publiplots/composer/__init__.py \
        tests/composer/test_panels.py
git commit -m "feat(composer): PanelAxes input + Panel result dataclasses"
```

---

## Task 3: Canvas class — construction, custom preset, .figure_size_mm

**Files:**
- Create: `src/publiplots/composer/canvas.py`
- Modify: `src/publiplots/composer/__init__.py` (re-export `Canvas`)
- Test: extend `tests/composer/test_canvas_construction.py`

This is the structural backbone. The implementation reuses `FigureLayout` directly: no new geometry math. The canvas sets `nrows=1, ncols=N` where N is the number of panels in the row, with each column's width = panel's `size[0]`.

- [ ] **Step 1: Write the failing tests for Canvas construction**

Append to `tests/composer/test_canvas_construction.py`:

```python
# ---------------------------------------------------------------------------
# Canvas construction — PR 1 only supports preset='custom' with width=
# ---------------------------------------------------------------------------
import publiplots as pp


def test_canvas_construction_custom_preset_with_width():
    canvas = pp.Canvas("custom", width=174.0)
    assert canvas.width_mm == 174.0


def test_canvas_construction_custom_requires_width():
    with pytest.raises(ValueError, match="width"):
        pp.Canvas("custom")


def test_canvas_construction_unknown_preset_raises():
    with pytest.raises(KeyError, match="unknown preset"):
        pp.Canvas("not-a-preset", width=100.0)


def test_canvas_construction_journal_preset_not_in_pr1():
    """Journal presets land in PR 2; PR 1 only accepts 'custom'.
    This test will be REMOVED in PR 2."""
    with pytest.raises(KeyError):
        pp.Canvas("cell-2col")


def test_canvas_figure_attribute_is_none_until_finalize():
    """A canvas before any add_row exposes .figure as None — the
    matplotlib Figure is created lazily when add_row is called.
    Rationale: a canvas with zero panels has no defined height."""
    canvas = pp.Canvas("custom", width=174.0)
    assert canvas.figure is None


def test_canvas_figure_size_mm_is_none_until_finalize():
    canvas = pp.Canvas("custom", width=174.0)
    assert canvas.figure_size_mm is None
```

- [ ] **Step 2: Run to verify it fails**

```bash
cd /home/sagemaker-user/publiplots
uv run pytest tests/composer/test_canvas_construction.py -v
```

Expected: the 6 new tests fail with `AttributeError: module 'publiplots' has no attribute 'Canvas'`.

- [ ] **Step 3: Implement Canvas (construction-only — no add_row yet)**

Create `src/publiplots/composer/canvas.py` with this exact content:

```python
"""Canvas — the Composer's top-level orchestrator.

PR 1 supports single-row layouts via :meth:`Canvas.add_row` and raster
:meth:`Canvas.savefig`. The figure is created lazily when ``add_row``
runs; ``Canvas(...)`` itself just records configuration.

Internally, the canvas builds a 1-row × N-column :class:`FigureLayout`
where each column's mm width matches one panel's declared width. The
existing :class:`SubplotsAutoLayout` reactor handles xlabel / ylabel /
title decoration reservation (per spike Finding 4 — without it, axis
decorations clip at the canvas mediabox edge).
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from publiplots.composer.exceptions import (
    ComposerError,
    ComposerOverflowError,
)
from publiplots.composer.panels import Panel, PanelAxes
from publiplots.composer.presets import resolve_preset
from publiplots.themes.rcparams import resolve_param


class Canvas:
    """Programmatic mm-precise canvas for multi-panel paper figures.

    Parameters
    ----------
    preset : str
        Preset name. PR 1 only supports ``'custom'``; journal presets
        (Cell, Nature, Nature Methods, Science) land in PR 2.
    width : float, optional
        Canvas width in millimeters. Required for ``preset='custom'``.

    Examples
    --------
    Single-row, two axes panels:

    >>> import publiplots as pp
    >>> canvas = pp.Canvas("custom", width=174.0)
    >>> canvas.add_row(
    ...     pp.PanelAxes(label="A", size=(70, 40)),
    ...     pp.PanelAxes(label="B", size=(70, 40)),
    ... )
    >>> pp.scatterplot(data=df, x="x", y="y", ax=canvas["A"].ax)
    >>> canvas.savefig("fig.png")

    Notes
    -----
    Multi-row layouts and ``add_column`` land in PR 3. Vector PDF/SVG
    save dispatches land in PR 5/PR 6. ``canvas.savefig('fig.pdf')``
    raises :class:`NotImplementedError` in PR 1.
    """

    def __init__(self, preset: str, *, width: Optional[float] = None) -> None:
        self._preset_name = preset
        spec = resolve_preset(preset, width=width)
        self._width_mm: float = spec["width_mm"]
        self._max_height_mm: Optional[float] = spec["max_height_mm"]

        # Lazy-initialized on first add_row():
        self._figure = None  # matplotlib.figure.Figure | None
        self._panels: Dict[str, Panel] = {}
        self._row_added: bool = False

    # ------------------------------------------------------------------
    # Read-only attributes
    # ------------------------------------------------------------------
    @property
    def width_mm(self) -> float:
        """Canvas width in millimeters."""
        return self._width_mm

    @property
    def figure(self):
        """The underlying matplotlib :class:`Figure`, or ``None`` until
        :meth:`add_row` runs."""
        return self._figure

    @property
    def figure_size_mm(self) -> Optional[Tuple[float, float]]:
        """``(width_mm, height_mm)`` after :meth:`add_row`, else ``None``."""
        if self._figure is None:
            return None
        # Convert mpl figure size (inches) back to mm.
        w_in, h_in = self._figure.get_size_inches()
        return (w_in * 25.4, h_in * 25.4)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------
    def __getitem__(self, label: str) -> Panel:
        if label not in self._panels:
            raise KeyError(
                f"no panel with label {label!r}; known labels: {sorted(self._panels)}"
            )
        return self._panels[label]
```

(The class is INTENTIONALLY incomplete — `add_row` and `savefig` arrive in Tasks 4 and 6. We commit construction-only behavior first to keep each task small.)

- [ ] **Step 4: Re-export `Canvas` from the package**

Edit `src/publiplots/composer/__init__.py` to add Canvas to the imports + `__all__`:

```python
"""publiplots Composer — multi-panel paper-figure builder.

Public API for PR 1 (single-row, axes-only):
- :class:`Canvas` — programmatic mm-precise canvas
- :class:`PanelAxes` — axes panel constructor
- :class:`Panel` — result type returned by ``canvas[label]``

See ``docs/superpowers/specs/2026-05-14-composer-design.md`` for the
full design.
"""

from publiplots.composer.canvas import Canvas
from publiplots.composer.exceptions import (
    ComposerError,
    ComposerOverflowError,
)
from publiplots.composer.panels import Panel, PanelAxes

__all__ = [
    "Canvas",
    "ComposerError",
    "ComposerOverflowError",
    "Panel",
    "PanelAxes",
]
```

- [ ] **Step 5: Re-export from the top-level package**

Edit `src/publiplots/__init__.py`. Find the existing block of imports near the top (after the `from publiplots.layout.jointgrid import ...` line) and add:

```python
# Composer (PR 1: single-row + raster save)
from publiplots.composer import Canvas, PanelAxes, Panel
```

Find the `__all__` list and add `"Canvas"`, `"PanelAxes"`, `"Panel"` entries (one line each, after `"subplots"`).

(The exact line numbers will depend on the current state of `__init__.py`. Read it first with the `Read` tool to find the right insertion points.)

- [ ] **Step 6: Run the construction tests to verify they pass**

```bash
cd /home/sagemaker-user/publiplots
uv run pytest tests/composer/test_canvas_construction.py -v
```

Expected: ALL tests in this file PASS (the 8 from Task 1 + the 6 new ones = 14 passing).

- [ ] **Step 7: Verify the top-level import works**

```bash
cd /home/sagemaker-user/publiplots
uv run python -c "import publiplots as pp; c = pp.Canvas('custom', width=174); print(c.width_mm, c.figure)"
```

Expected: prints `174.0 None`.

- [ ] **Step 8: Commit**

```bash
cd /home/sagemaker-user/publiplots
git add src/publiplots/composer/canvas.py \
        src/publiplots/composer/__init__.py \
        src/publiplots/__init__.py \
        tests/composer/test_canvas_construction.py
git commit -m "feat(composer): Canvas construction with custom preset"
```

---

## Task 4: `canvas.add_row` — geometry + Figure creation

**Files:**
- Modify: `src/publiplots/composer/canvas.py` (extend with `add_row`)
- Test: `tests/composer/test_add_row.py`
- Test: `tests/composer/test_indexing.py`

This is the core geometry task. The implementation:
1. Validates that `add_row` was not called before (single-row in PR 1).
2. Validates that the sum of panel widths + (n-1) × hpad ≤ canvas width.
3. Maps panels to columns of a `FigureLayout(nrows=1, ncols=N)` with `col_widths` = panel widths and `row_heights` = panel heights.
4. Creates the `Figure` and adds one `Axes` per panel.
5. Wires up `SubplotsAutoLayout` (so xlabel / ylabel / title don't clip — spike Finding 4).
6. Stores each panel in `self._panels[label]`.

- [ ] **Step 1: Write the failing test for add_row geometry**

Create `tests/composer/test_add_row.py` with this exact content:

```python
"""Tests for canvas.add_row — geometry, figure creation, layout reactor."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

import publiplots as pp
from publiplots.composer.exceptions import ComposerOverflowError


# ---------------------------------------------------------------------------
# add_row creates the figure + populates panels
# ---------------------------------------------------------------------------

def test_add_row_creates_figure():
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(
        pp.PanelAxes(label="A", size=(70.0, 40.0)),
        pp.PanelAxes(label="B", size=(70.0, 40.0)),
    )
    assert canvas.figure is not None


def test_add_row_populates_two_panels():
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(
        pp.PanelAxes(label="A", size=(70.0, 40.0)),
        pp.PanelAxes(label="B", size=(70.0, 40.0)),
    )
    a = canvas["A"]
    b = canvas["B"]
    assert a.label == "A" and a.kind == "axes"
    assert b.label == "B" and b.kind == "axes"


def test_add_row_panels_each_get_a_real_axes():
    """canvas[label].ax is a matplotlib.axes.Axes — not None."""
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(
        pp.PanelAxes(label="A", size=(70.0, 40.0)),
        pp.PanelAxes(label="B", size=(70.0, 40.0)),
    )
    from matplotlib.axes import Axes
    assert isinstance(canvas["A"].ax, Axes)
    assert isinstance(canvas["B"].ax, Axes)
    assert canvas["A"].ax is not canvas["B"].ax


def test_add_row_single_panel():
    """A single panel still works — width = panel width + outer_pad×2 +
    ylabel + right reservations from rcParams."""
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(140.0, 50.0)))
    assert canvas["A"].size_mm == (140.0, 50.0)


def test_add_row_three_panels():
    canvas = pp.Canvas("custom", width=174.0)
    # 3 panels of width 50 + 2 hpads of 4mm + outer/ylabel/right reservations
    canvas.add_row(
        pp.PanelAxes(label="A", size=(40.0, 30.0)),
        pp.PanelAxes(label="B", size=(40.0, 30.0)),
        pp.PanelAxes(label="C", size=(40.0, 30.0)),
    )
    assert set(canvas._panels.keys()) == {"A", "B", "C"}


# ---------------------------------------------------------------------------
# Multi-call add_row is rejected in PR 1
# ---------------------------------------------------------------------------

def test_add_row_called_twice_raises():
    """PR 3 adds multi-row support; PR 1 rejects it with a clear hint."""
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(70.0, 40.0)))
    with pytest.raises(NotImplementedError, match="multi-row"):
        canvas.add_row(pp.PanelAxes(label="B", size=(70.0, 40.0)))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_add_row_zero_panels_raises():
    canvas = pp.Canvas("custom", width=174.0)
    with pytest.raises(ValueError, match="at least one panel"):
        canvas.add_row()


def test_add_row_duplicate_labels_raises():
    canvas = pp.Canvas("custom", width=174.0)
    with pytest.raises(ValueError, match="duplicate.*label"):
        canvas.add_row(
            pp.PanelAxes(label="A", size=(40.0, 30.0)),
            pp.PanelAxes(label="A", size=(40.0, 30.0)),
        )


def test_add_row_rejects_non_panelaxes():
    """PR 3 adds PanelGrid/PanelText; PR 1 only accepts PanelAxes."""
    canvas = pp.Canvas("custom", width=174.0)
    with pytest.raises(TypeError, match="PanelAxes"):
        canvas.add_row("not a panel")


def test_add_row_width_overflow_raises_with_dims():
    """Two 100mm panels in a 174mm canvas overflow; the error carries
    requested + available dims."""
    canvas = pp.Canvas("custom", width=174.0)
    with pytest.raises(ComposerOverflowError) as exc_info:
        canvas.add_row(
            pp.PanelAxes(label="A", size=(100.0, 40.0)),
            pp.PanelAxes(label="B", size=(100.0, 40.0)),
        )
    err = exc_info.value
    assert err.requested_mm > 174.0
    assert err.available_mm <= 174.0


# ---------------------------------------------------------------------------
# Geometry — mm precision
# ---------------------------------------------------------------------------

MM_TOL = 0.01  # 0.01 mm tolerance for figure-size assertions


def test_add_row_figure_width_equals_panels_plus_decorations():
    """In PR 1 the figure width = sum(panel widths) + decoration
    reservations, NOT the canvas width budget. (PR 2 adds 'flex'
    sizing to absorb the slack so the figure equals canvas width
    exactly.) With 2×70mm panels + rcParams defaults
    (outer_pad=2, ylabel=10, right=2, wspace=3):
    figure_width = 2+10+70+2+3+10+70+2+2 = 171 mm."""
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(
        pp.PanelAxes(label="A", size=(70.0, 40.0)),
        pp.PanelAxes(label="B", size=(70.0, 40.0)),
    )
    w_mm, _ = canvas.figure_size_mm
    assert abs(w_mm - 171.0) < MM_TOL


def test_add_row_figure_width_never_exceeds_canvas_width():
    """The figure width must NEVER exceed the canvas budget — that's
    what overflow validation guarantees. With panels sized to barely
    fit (2×71mm = 142 + 31 decorations = 173 ≤ 174), the figure is
    173 mm wide, just inside the budget."""
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(
        pp.PanelAxes(label="A", size=(71.0, 40.0)),
        pp.PanelAxes(label="B", size=(71.0, 40.0)),
    )
    w_mm, _ = canvas.figure_size_mm
    assert w_mm <= 174.0 + MM_TOL
    assert abs(w_mm - 173.0) < MM_TOL


def test_add_row_figure_height_grows_to_fit_panels_plus_decorations():
    """Height = panel max height + xlabel_space + title_space + outer_pad×2.
    With rcParams defaults: 40 + 8 (xlabel) + 5 (title) + 4 (2×outer_pad)
    = 57 mm, give or take. Decoration auto-measurement may inflate it
    further on first draw (we don't draw here, so this asserts the
    INITIAL figure size from rcParams defaults)."""
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(70.0, 40.0)))
    _, h_mm = canvas.figure_size_mm
    # Panel height + initial decoration reservations
    expected_initial = 40.0 + 8.0 + 5.0 + 2.0 + 2.0  # = 57 mm
    assert abs(h_mm - expected_initial) < MM_TOL


def test_add_row_panel_bbox_has_correct_width():
    """Each panel's bbox_mm width equals its declared size width."""
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(
        pp.PanelAxes(label="A", size=(70.0, 40.0)),
        pp.PanelAxes(label="B", size=(50.0, 40.0)),
    )
    a_bbox = canvas["A"].bbox_mm
    b_bbox = canvas["B"].bbox_mm
    assert abs(a_bbox[2] - 80.0) < MM_TOL  # bbox is (x, y, w, h)
    assert abs(b_bbox[2] - 60.0) < MM_TOL


def test_add_row_panel_a_is_left_of_panel_b():
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(
        pp.PanelAxes(label="A", size=(70.0, 40.0)),
        pp.PanelAxes(label="B", size=(70.0, 40.0)),
    )
    a_x = canvas["A"].bbox_mm[0]
    b_x = canvas["B"].bbox_mm[0]
    assert a_x < b_x


def test_add_row_panels_preserve_ordering_in_dict():
    """canvas._panels iterates in add_row order — important for PR 2's
    auto-letter sequencing."""
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(
        pp.PanelAxes(label="C", size=(40.0, 30.0)),
        pp.PanelAxes(label="A", size=(40.0, 30.0)),
        pp.PanelAxes(label="B", size=(40.0, 30.0)),
    )
    assert list(canvas._panels.keys()) == ["C", "A", "B"]
```

- [ ] **Step 2: Write the failing test for indexing**

Create `tests/composer/test_indexing.py` with this exact content:

```python
"""Tests for canvas[label] indexing."""

import pytest

import publiplots as pp


def test_canvas_indexing_returns_panel_for_known_label():
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(70.0, 40.0)))
    p = canvas["A"]
    assert p.label == "A"


def test_canvas_indexing_raises_keyerror_for_unknown_label():
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(70.0, 40.0)))
    with pytest.raises(KeyError, match="no panel with label"):
        canvas["B"]


def test_canvas_indexing_raises_keyerror_before_add_row():
    """Indexing on a not-yet-finalized canvas — no panels yet."""
    canvas = pp.Canvas("custom", width=174.0)
    with pytest.raises(KeyError):
        canvas["A"]


def test_canvas_indexing_keyerror_lists_known_labels():
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(
        pp.PanelAxes(label="A", size=(40.0, 30.0)),
        pp.PanelAxes(label="C", size=(40.0, 30.0)),
    )
    with pytest.raises(KeyError, match=r"\['A', 'C'\]"):
        canvas["B"]
```

- [ ] **Step 3: Run to verify they fail**

```bash
cd /home/sagemaker-user/publiplots
uv run pytest tests/composer/test_add_row.py tests/composer/test_indexing.py -v
```

Expected: tests FAIL with `AttributeError: 'Canvas' object has no attribute 'add_row'`.

- [ ] **Step 4: Implement `add_row` in canvas.py**

Append to `src/publiplots/composer/canvas.py` (inside the `Canvas` class):

```python
    # ------------------------------------------------------------------
    # add_row — single-row layout for PR 1
    # ------------------------------------------------------------------
    def add_row(self, *panels: PanelAxes) -> None:
        """Add a row of axes panels to the canvas.

        PR 1 supports calling ``add_row`` exactly once per canvas
        (single-row layouts only). PR 3 will lift this restriction.

        Parameters
        ----------
        *panels : PanelAxes
            One or more :class:`PanelAxes` instances. Their declared
            widths plus inter-panel ``hpad`` (3 mm by default from
            rcParams) plus outer + ylabel + right reservations must fit
            in the canvas width.

        Raises
        ------
        NotImplementedError
            If ``add_row`` was already called on this canvas.
        ValueError
            If no panels are passed, or if duplicate labels appear, or
            if a non-:class:`PanelAxes` argument is passed.
        ComposerOverflowError
            If the row's total width exceeds the canvas budget.
        """
        # --- guard against multi-row in PR 1 ------------------------
        if self._row_added:
            raise NotImplementedError(
                "Canvas.add_row called twice; multi-row support lands in PR 3"
            )

        # --- validate inputs ----------------------------------------
        if len(panels) == 0:
            raise ValueError("Canvas.add_row requires at least one panel")
        for p in panels:
            if not isinstance(p, PanelAxes):
                raise TypeError(
                    f"Canvas.add_row only accepts PanelAxes in PR 1, "
                    f"got {type(p).__name__} (PanelGrid/PanelText land in PR 3, "
                    f"PanelImage in PR 5)"
                )
        labels = [p.label for p in panels]
        seen: set = set()
        for lbl in labels:
            if lbl in seen:
                raise ValueError(f"duplicate panel label: {lbl!r}")
            seen.add(lbl)

        # --- compute geometry ---------------------------------------
        # Use the same rcParams defaults that pp.subplots uses so the
        # initial figure size is consistent with single-grid figures.
        outer_pad = float(resolve_param("subplots.outer_pad", None))
        hpad = float(resolve_param("subplots.wspace", None))  # inter-panel gap
        title_space = float(resolve_param("subplots.title_space", None))
        xlabel_space = float(resolve_param("subplots.xlabel_space", None))
        ylabel_space = float(resolve_param("subplots.ylabel_space", None))
        right = float(resolve_param("subplots.right", None))

        col_widths = tuple(p.size[0] for p in panels)
        # All panels in one row share the row height; PR 1 requires equal
        # heights (PR 2 adds 'flex'/'match' grammar). Use the max for now;
        # if heights differ, that's a future-PR issue, not a PR 1 error.
        row_height = max(p.size[1] for p in panels)

        ncols = len(panels)
        # Canvas width budget: panels + (n-1) hpads + ylabel reservations
        # for each column + right reservation for each column + outer pads.
        # ylabel_space and right are PER-COLUMN per FigureLayout's contract.
        decorations_width = (
            2 * outer_pad
            + ncols * ylabel_space
            + ncols * right
            + max(ncols - 1, 0) * hpad
        )
        panels_width = sum(col_widths)
        requested_width = panels_width + decorations_width
        if requested_width > self._width_mm + 1e-6:  # 1µm tolerance for float noise
            raise ComposerOverflowError(
                f"row width {requested_width:.2f}mm exceeds canvas width "
                f"{self._width_mm:.2f}mm; reduce panel widths or use a wider canvas",
                requested_mm=requested_width,
                available_mm=self._width_mm,
            )

        # NOTE: PR 1 does NOT auto-absorb width slack. If the user passes
        # panels that sum to less than the canvas width, the produced
        # figure is narrower than `self._width_mm`. Callers can size
        # panels to fit exactly (use the overflow-error math in reverse:
        # max-panel-width-per-row = (canvas_width - decorations_width)).
        # PR 2 adds 'flex' panel sizing which absorbs slack automatically.

        # --- build FigureLayout (1 row × N cols) --------------------
        from publiplots.layout.figure_layout import FigureLayout
        from publiplots.layout.auto_layout import SubplotsAutoLayout
        import matplotlib.pyplot as plt

        layout = FigureLayout(
            nrows=1,
            ncols=ncols,
            axes_size=(col_widths[0], row_height),  # fallback; col_widths overrides
            col_widths=col_widths,
            row_heights=(row_height,),
            title_space=(title_space,),
            xlabel_space=(xlabel_space,),
            ylabel_space=(ylabel_space,) * ncols,
            right=(right,) * ncols,
            hspace=0.0,           # only 1 row, irrelevant
            wspace=hpad,
            outer_pad=outer_pad,
            legend_column=0.0,
            suptitle_space=0.0,
        )
        W_mm, H_mm = layout.figure_size()

        # --- create the matplotlib figure at the computed mm size ---
        _MM2INCH = 1.0 / 25.4
        fig = plt.figure(
            figsize=(W_mm * _MM2INCH, H_mm * _MM2INCH),
            layout=None,
        )
        self._figure = fig

        # --- create axes per panel ----------------------------------
        for col_idx, panel_input in enumerate(panels):
            x0_frac, y0_frac, w_frac, h_frac = layout.axes_position(0, col_idx)
            ax = fig.add_axes((x0_frac, y0_frac, w_frac, h_frac))
            # bbox_mm is (x_mm, y_mm, w_mm, h_mm) bottom-left origin.
            bbox_mm = (
                x0_frac * W_mm,
                y0_frac * H_mm,
                w_frac * W_mm,
                h_frac * H_mm,
            )
            self._panels[panel_input.label] = Panel(
                label=panel_input.label,
                kind="axes",
                ax=ax,
                size_mm=(panel_input.size[0], panel_input.size[1]),
                bbox_mm=bbox_mm,
            )

        # --- attach SubplotsAutoLayout reactor (spike Finding 4) ----
        # All four auto-measurable sides start at rcParams defaults and
        # auto-grow on first draw to fit decorations. This prevents
        # axis labels from clipping at the canvas mediabox edge — the
        # exact bug the spike's bare-mpl fixture exhibited.
        fig._publiplots_auto_layout = SubplotsAutoLayout(
            fig, layout,
            locked=set(),               # let all four sides auto-measure
            locked_positions={},
        )

        self._row_added = True
```

- [ ] **Step 5: Run the tests**

```bash
cd /home/sagemaker-user/publiplots
uv run pytest tests/composer/test_add_row.py tests/composer/test_indexing.py -v
```

Expected: ALL tests PASS.

- [ ] **Step 6: Commit**

```bash
cd /home/sagemaker-user/publiplots
git add src/publiplots/composer/canvas.py \
        tests/composer/test_add_row.py \
        tests/composer/test_indexing.py
git commit -m "feat(composer): add_row geometry + figure creation"
```

---

## Task 5: Decoration-reservation regression test (spike Finding 4)

**Files:**
- Test: `tests/composer/test_decoration_reservations.py`

This is the load-bearing test for spike Finding 4. The bug it prevents: a Composer-built figure where xlabel / ylabel / title clip at the canvas mediabox edge because `SubplotsAutoLayout` wasn't attached. Task 4's implementation does attach it; this test confirms the wiring works end-to-end.

The test renders a small canvas, sets an xlabel + title on the panel, runs `fig.canvas.draw()` to settle the layout, then asserts the figure's height grew beyond the initial allocation (proving auto-measurement fired). It also asserts the xlabel's tightbbox lies inside the figure's pixel bounds (proving no clipping).

- [ ] **Step 1: Write the failing test**

Create `tests/composer/test_decoration_reservations.py` with this exact content:

```python
"""Spike Finding 4 regression test.

The bug: a Composer-built figure where the bottom xlabel / left ylabel /
top title clips at the canvas mediabox edge, because the figure was
constructed at the panel-only mm size without reserving decoration
space. The spike's bare-mpl fixture exhibited this exact bug; the
Composer's add_row attaches SubplotsAutoLayout to fix it.

This test confirms the fix:
  1. Canvas auto-grows to fit decorations after first draw.
  2. The panel's xlabel tightbbox lies inside the figure's mediabox
     (no clipping).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

import publiplots as pp


def test_canvas_height_grows_to_fit_xlabel_after_draw():
    """The canvas's initial height is panel + initial xlabel/title
    reservations; after drawing, auto-measurement may grow xlabel_space
    further if the rendered text demands more space. Either way, the
    figure mediabox MUST be at least the initial reservation."""
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(70.0, 40.0)))
    canvas["A"].ax.set_xlabel("a long-ish x-axis label that takes vertical space")
    canvas["A"].ax.set_ylabel("y")
    canvas["A"].ax.set_title("Panel title")

    initial_w_mm, initial_h_mm = canvas.figure_size_mm

    # Drive the layout reactor to convergence.
    canvas.figure.canvas.draw()
    canvas.figure._publiplots_auto_layout.settle()

    settled_w_mm, settled_h_mm = canvas.figure_size_mm

    # Width should be unchanged (canvas width is fixed by Canvas(width=174)).
    assert abs(settled_w_mm - initial_w_mm) < 0.01

    # Height should be >= initial; never shrinks.
    assert settled_h_mm >= initial_h_mm - 0.01


def test_canvas_xlabel_does_not_clip_at_canvas_bottom():
    """Spike Finding 4 regression — the rendered xlabel's tightbbox must
    lie INSIDE the figure mediabox (no clipping at canvas bottom)."""
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(70.0, 40.0)))
    canvas["A"].ax.set_xlabel("x")  # use a short xlabel like the spike fixture

    fig = canvas.figure
    fig.canvas.draw()
    fig._publiplots_auto_layout.settle()

    # Get the renderer and the xlabel tightbbox in display coordinates.
    renderer = fig.canvas.get_renderer()
    xlabel_artist = canvas["A"].ax.xaxis.get_label()
    xlabel_bbox_px = xlabel_artist.get_window_extent(renderer=renderer)

    # Figure bbox in display coordinates.
    fig_bbox_px = fig.bbox

    # The xlabel's bottom edge must be at or above the figure's bottom edge.
    # Allow 0.5 px of float-noise tolerance (matplotlib pt rounding).
    assert xlabel_bbox_px.y0 >= fig_bbox_px.y0 - 0.5, (
        f"xlabel bottom (px y0={xlabel_bbox_px.y0:.2f}) clips below "
        f"figure bottom (px y0={fig_bbox_px.y0:.2f}) — "
        f"SubplotsAutoLayout did not reserve enough xlabel_space"
    )


def test_canvas_title_does_not_clip_at_canvas_top():
    """Same regression check, top edge."""
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(70.0, 40.0)))
    canvas["A"].ax.set_title("My panel title")

    fig = canvas.figure
    fig.canvas.draw()
    fig._publiplots_auto_layout.settle()

    renderer = fig.canvas.get_renderer()
    title_artist = canvas["A"].ax.title
    title_bbox_px = title_artist.get_window_extent(renderer=renderer)
    fig_bbox_px = fig.bbox

    assert title_bbox_px.y1 <= fig_bbox_px.y1 + 0.5, (
        f"title top (px y1={title_bbox_px.y1:.2f}) clips above "
        f"figure top (px y1={fig_bbox_px.y1:.2f}) — "
        f"SubplotsAutoLayout did not reserve enough title_space"
    )


def test_canvas_ylabel_does_not_clip_at_canvas_left():
    """Same regression check, left edge."""
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(70.0, 40.0)))
    canvas["A"].ax.set_ylabel("Density (counts / mm²)")  # longer-than-default

    fig = canvas.figure
    fig.canvas.draw()
    fig._publiplots_auto_layout.settle()

    renderer = fig.canvas.get_renderer()
    ylabel_artist = canvas["A"].ax.yaxis.get_label()
    ylabel_bbox_px = ylabel_artist.get_window_extent(renderer=renderer)
    fig_bbox_px = fig.bbox

    assert ylabel_bbox_px.x0 >= fig_bbox_px.x0 - 0.5, (
        f"ylabel left (px x0={ylabel_bbox_px.x0:.2f}) clips left of "
        f"figure left (px x0={fig_bbox_px.x0:.2f}) — "
        f"SubplotsAutoLayout did not reserve enough ylabel_space"
    )
```

- [ ] **Step 2: Run to verify the tests fail or pass**

```bash
cd /home/sagemaker-user/publiplots
uv run pytest tests/composer/test_decoration_reservations.py -v
```

These tests SHOULD pass already because Task 4 wired up `SubplotsAutoLayout`. If they fail, it means Task 4's implementation didn't connect the reactor correctly.

If they pass: great — the Finding 4 regression test is the spec-compliance gate.

If they fail: STOP and read Task 4's implementation. The most likely failure is that `SubplotsAutoLayout` was instantiated but the figure's `draw_event` wasn't connected. Verify by reading `src/publiplots/layout/auto_layout.py` lines 80-83 — the `mpl_connect("draw_event", self._on_draw)` line is the load-bearing wire.

- [ ] **Step 3: Commit**

```bash
cd /home/sagemaker-user/publiplots
git add tests/composer/test_decoration_reservations.py
git commit -m "test(composer): spike Finding 4 — decoration reservations don't clip"
```

---

## Task 6: `canvas.savefig` raster pipeline + PDF/SVG NotImplementedError

**Files:**
- Create: `src/publiplots/composer/_save.py`
- Modify: `src/publiplots/composer/canvas.py` — add `savefig` method
- Test: `tests/composer/test_savefig.py`

PR 1 ships only the raster path. PDF/SVG dispatch raises `NotImplementedError` so users get a clear message rather than a silent rasterized PDF.

- [ ] **Step 1: Write the failing test**

Create `tests/composer/test_savefig.py` with this exact content:

```python
"""Tests for canvas.savefig — raster pipeline only in PR 1."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import pytest

import publiplots as pp


def test_savefig_png_works(tmp_path):
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(70.0, 40.0)))
    canvas["A"].ax.scatter([1, 2, 3], [1, 4, 2])

    out = tmp_path / "fig.png"
    canvas.savefig(out)

    assert out.exists()
    assert out.stat().st_size > 0


def test_savefig_jpg_works(tmp_path):
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(70.0, 40.0)))
    canvas["A"].ax.scatter([1, 2], [1, 2])

    out = tmp_path / "fig.jpg"
    canvas.savefig(out)
    assert out.exists()


def test_savefig_tiff_works(tmp_path):
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(70.0, 40.0)))
    canvas["A"].ax.scatter([1, 2], [1, 2])

    out = tmp_path / "fig.tiff"
    canvas.savefig(out)
    assert out.exists()


def test_savefig_pdf_raises_not_implemented(tmp_path):
    """PR 1 doesn't ship the PDF compositing pipeline (lands in PR 5).
    A clear NotImplementedError is better than a silent rasterized PDF."""
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(70.0, 40.0)))

    out = tmp_path / "fig.pdf"
    with pytest.raises(NotImplementedError, match="PR 5"):
        canvas.savefig(out)


def test_savefig_svg_raises_not_implemented(tmp_path):
    """SVG vector pipeline lands in PR 6."""
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(70.0, 40.0)))

    out = tmp_path / "fig.svg"
    with pytest.raises(NotImplementedError, match="PR 6"):
        canvas.savefig(out)


def test_savefig_unknown_extension_raises(tmp_path):
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(70.0, 40.0)))

    out = tmp_path / "fig.xyz"
    with pytest.raises(ValueError, match="unknown.*extension"):
        canvas.savefig(out)


def test_savefig_before_add_row_raises():
    """Saving an empty canvas — no figure to write."""
    canvas = pp.Canvas("custom", width=174.0)
    with pytest.raises(RuntimeError, match="add_row"):
        canvas.savefig("/tmp/never-written.png")


def test_savefig_accepts_str_and_path(tmp_path):
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(70.0, 40.0)))

    out_str = str(tmp_path / "via_str.png")
    out_path = tmp_path / "via_path.png"

    canvas.savefig(out_str)
    canvas.savefig(out_path)

    assert Path(out_str).exists()
    assert out_path.exists()
```

- [ ] **Step 2: Run to verify it fails**

```bash
cd /home/sagemaker-user/publiplots
uv run pytest tests/composer/test_savefig.py -v
```

Expected: tests FAIL with `AttributeError: 'Canvas' object has no attribute 'savefig'`.

- [ ] **Step 3: Implement `_save.py`**

Create `src/publiplots/composer/_save.py` with this exact content:

```python
"""Save dispatch for the Composer.

PR 1 ships only the raster pipeline; PDF and SVG raise
:class:`NotImplementedError` with pointers to the PRs that will add
those backends. The dispatch is by file extension — same shape that
the production design uses, just with most branches deferred.
"""

from pathlib import Path
from typing import Any, Union

from matplotlib.figure import Figure

from publiplots.utils.io import savefig as _pp_savefig


_RASTER_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
_VECTOR_PDF_EXTS = {".pdf"}
_VECTOR_SVG_EXTS = {".svg"}


def dispatch_savefig(
    figure: Figure,
    path: Union[str, Path],
    **kwargs: Any,
) -> None:
    """Dispatch ``canvas.savefig`` by file extension.

    PR 1: raster paths (PNG/JPG/TIFF) delegate to :func:`pp.savefig`,
    which inherits publiplots' rcParams defaults (transparent
    background, 600 dpi, ``bbox_inches=None``).

    PR 5/6: PDF and SVG dispatch to vector compositing pipelines. In
    PR 1, those raise :class:`NotImplementedError` with a pointer.
    """
    p = Path(path)
    ext = p.suffix.lower()

    if ext in _RASTER_EXTS:
        _pp_savefig(str(p), **kwargs)
        return

    if ext in _VECTOR_PDF_EXTS:
        raise NotImplementedError(
            "Canvas.savefig('*.pdf') requires the vector compositing pipeline, "
            "which lands in PR 5. For now, save to PNG/JPG/TIFF, or call "
            "fig.savefig(...) directly on canvas.figure to bypass compositing."
        )
    if ext in _VECTOR_SVG_EXTS:
        raise NotImplementedError(
            "Canvas.savefig('*.svg') requires the in-tree SVG composer, "
            "which lands in PR 6. For now, save to PNG/JPG/TIFF, or call "
            "fig.savefig(...) directly on canvas.figure to bypass compositing."
        )

    raise ValueError(
        f"unknown savefig extension {ext!r}; "
        f"supported in PR 1: {sorted(_RASTER_EXTS)}; "
        f"PDF lands in PR 5, SVG in PR 6"
    )
```

- [ ] **Step 4: Add `Canvas.savefig` method**

Append to the `Canvas` class in `src/publiplots/composer/canvas.py`:

```python
    # ------------------------------------------------------------------
    # savefig — raster only in PR 1
    # ------------------------------------------------------------------
    def savefig(self, path, **kwargs) -> None:
        """Save the canvas to a file.

        PR 1 supports raster formats (PNG, JPG, TIFF). PDF and SVG raise
        :class:`NotImplementedError` until PR 5 / PR 6 land the vector
        compositing pipelines.

        Parameters
        ----------
        path : str or Path
            Output file path. Extension determines the format.
        **kwargs
            Forwarded to :func:`publiplots.savefig`.

        Raises
        ------
        RuntimeError
            If :meth:`add_row` has not been called yet.
        NotImplementedError
            If ``path`` ends in ``.pdf`` (PR 5) or ``.svg`` (PR 6).
        ValueError
            If the path's extension is not a known raster or vector type.
        """
        if self._figure is None:
            raise RuntimeError(
                "Canvas has no figure yet; call add_row() before savefig()"
            )
        from publiplots.composer._save import dispatch_savefig
        dispatch_savefig(self._figure, path, **kwargs)
```

- [ ] **Step 5: Run the tests**

```bash
cd /home/sagemaker-user/publiplots
uv run pytest tests/composer/test_savefig.py -v
```

Expected: ALL tests PASS.

- [ ] **Step 6: Commit**

```bash
cd /home/sagemaker-user/publiplots
git add src/publiplots/composer/_save.py \
        src/publiplots/composer/canvas.py \
        tests/composer/test_savefig.py
git commit -m "feat(composer): savefig dispatch — raster works, PDF/SVG NotImplementedError"
```

---

## Task 7: Layout integration — Composer figures obey publiplots rcParams

**Files:**
- Test: `tests/composer/test_layout_integration.py`

This is a behavioral test: a Composer-built figure should inherit publiplots' rcParams (Arial fonts, 0.75pt strokes, etc.), the same way `pp.subplots`-built figures do. No new code — just a test pinning the contract.

- [ ] **Step 1: Write the failing test**

Create `tests/composer/test_layout_integration.py` with this exact content:

```python
"""Integration tests — Composer figures inherit publiplots rcParams + style."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

import publiplots as pp


def test_composer_figure_uses_publiplots_savefig_dpi():
    """publiplots' savefig.dpi rcParam is 600. The Composer's raster
    savefig path inherits it via pp.savefig."""
    assert pp.rcParams["savefig.dpi"] == 600


def test_composer_figure_uses_arial_font():
    """publiplots' init_rcparams sets Arial as the default sans-serif.
    A Composer-built figure inherits this — confirmed by the panel's
    text artists having Arial in their font.family."""
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(70.0, 40.0)))
    canvas["A"].ax.set_xlabel("x label")
    fig = canvas.figure
    fig.canvas.draw()

    xlabel_text = canvas["A"].ax.xaxis.get_label()
    family = xlabel_text.get_fontfamily()
    # `family` is a list-like; matplotlib's default Arial fallback is
    # in there.
    assert any("Arial" in f or f == "sans-serif" for f in family)


def test_composer_panel_axes_is_a_real_matplotlib_axes():
    """canvas[label].ax is a real Axes — usable with pp.scatterplot etc."""
    from matplotlib.axes import Axes

    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(70.0, 40.0)))
    assert isinstance(canvas["A"].ax, Axes)


def test_composer_can_plot_with_pp_scatterplot():
    """Smoke test — pp.scatterplot accepts canvas[label].ax as its ax=
    kwarg (the canonical Composer plotting pattern)."""
    import pandas as pd

    df = pd.DataFrame({"x": [1, 2, 3], "y": [1, 4, 2], "g": ["a", "b", "a"]})
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(70.0, 40.0)))
    pp.scatterplot(data=df, x="x", y="y", hue="g", ax=canvas["A"].ax)

    # The axes now has at least one collection (the scatter).
    assert len(canvas["A"].ax.collections) >= 1


def test_composer_figure_size_matches_pp_subplots_for_equivalent_layout():
    """A Canvas with one 70×40mm panel + default rcParams should produce
    the same figure size as pp.subplots(1, 1, axes_size=(70, 40)) ±
    decoration auto-measurement noise. This pins the equivalence between
    the two layout entry points (the Canvas reuses pp.subplots' layout
    machinery)."""
    fig_subplots, _ = pp.subplots(1, 1, axes_size=(70.0, 40.0))
    fig_subplots.canvas.draw()
    sub_w_in, sub_h_in = fig_subplots.get_size_inches()

    # Set canvas width to a comfortable upper bound; the Composer's
    # figure width is determined by panel + decorations, NOT the canvas
    # budget (in PR 1 — PR 2 adds 'flex' sizing).
    canvas = pp.Canvas("custom", width=200.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(70.0, 40.0)))
    canvas.figure.canvas.draw()
    canv_w_in, canv_h_in = canvas.figure.get_size_inches()

    # Both layouts route through FigureLayout + SubplotsAutoLayout with
    # the same rcParams reservations, so width AND height match within
    # decoration auto-measurement noise (~1 mm).
    assert abs(canv_w_in - sub_w_in) < 0.05
    assert abs(canv_h_in - sub_h_in) < 0.05
```

- [ ] **Step 2: Run to verify**

```bash
cd /home/sagemaker-user/publiplots
uv run pytest tests/composer/test_layout_integration.py -v
```

Expected: ALL tests PASS (no implementation changes needed; these are integration assertions on existing behavior).

If any fail, STOP and inspect — it likely means the canvas isn't inheriting rcParams correctly. Most likely cause: a missed `import publiplots` somewhere downstream that didn't trigger `init_rcparams`.

- [ ] **Step 3: Commit**

```bash
cd /home/sagemaker-user/publiplots
git add tests/composer/test_layout_integration.py
git commit -m "test(composer): layout integration — rcParams inheritance + pp.scatterplot interop"
```

---

## Task 8: Run the full test suite + commit fixes if anything regressed

**Files:** none directly; remediation only if needed.

- [ ] **Step 1: Run the full publiplots test suite**

```bash
cd /home/sagemaker-user/publiplots
uv run pytest tests/ -v --tb=short
```

Expected: all tests pass — both the existing tests AND the ~30 new composer tests.

- [ ] **Step 2: If anything regressed**

If any pre-existing tests now fail, the most likely cause is that the new `from publiplots.composer import Canvas, PanelAxes, Panel` import in `src/publiplots/__init__.py` introduced a circular import or a side-effect at import time. Diagnose by:

```bash
cd /home/sagemaker-user/publiplots
uv run python -c "import publiplots; print('ok')"
```

If that itself errors, the import chain is broken — read the traceback and fix.

If the import works but a specific test fails, capture the failure and STOP. Report back BLOCKED with the failure details so the controller can decide whether to fix it inline or revert and rethink.

- [ ] **Step 3: Add a CHANGELOG entry**

Read `/home/sagemaker-user/publiplots/CHANGELOG.md` to find the `## [Unreleased]` section. Add a new `### Added` block under it (or extend the existing one) with this exact content:

```markdown
- `pp.Canvas('custom', width=N)` + `pp.PanelAxes(label, size=(w_mm, h_mm))`
  — first PR of the Composer rollout. Single-row layouts of axes panels
  with mm-precise width and auto-grow height that reserves
  xlabel/ylabel/title decoration space (per spike Finding 4). Raster
  save (PNG/JPG/TIFF) works; PDF and SVG raise `NotImplementedError`
  pending the vector compositing pipelines in PR 5/6. Indexing via
  `canvas['<label>'].ax` returns the panel's `matplotlib.axes.Axes`.
  Plotting with `pp.scatterplot(..., ax=canvas['A'].ax)` etc. is the
  canonical idiom. See `docs/superpowers/specs/2026-05-14-composer-design.md`.
```

- [ ] **Step 4: Commit the CHANGELOG**

```bash
cd /home/sagemaker-user/publiplots
git add CHANGELOG.md
git commit -m "chore(changelog): composer PR 1 — Canvas + PanelAxes (single-row, raster)"
```

---

## Task 9: Open the PR

**Files:** none (gh CLI only)

- [ ] **Step 1: Push the branch**

```bash
cd /home/sagemaker-user/publiplots
git push -u origin feat/composer-canvas-pr1
```

- [ ] **Step 2: Open the PR**

```bash
cd /home/sagemaker-user/publiplots
gh pr create \
  --title "feat(composer): Canvas + PanelAxes (single-row, raster)" \
  --body "$(cat <<'EOF'
## Summary

First PR of the Composer rollout (`docs/superpowers/specs/2026-05-14-composer-design.md`). Lands `pp.Canvas('custom', width=N)`, `pp.PanelAxes(label, size=(w_mm, h_mm))`, `canvas.add_row(*panels)`, `canvas[label].ax` indexing, and `canvas.savefig(path)` raster pipeline. Single-row layouts only.

## What's in this PR

- `src/publiplots/composer/` — new subpackage with `Canvas`, `PanelAxes`, `Panel`, exception types, presets table, save dispatch.
- `tests/composer/` — ~30 tests covering construction, geometry (mm precision), indexing, multi-call rejection, savefig dispatch, decoration reservations (spike Finding 4 regression), and layout-integration smoke tests.
- `CHANGELOG.md` — Unreleased entry for the composer foundation.
- Top-level `pp.Canvas`, `pp.PanelAxes`, `pp.Panel` re-exports.

## What's NOT in this PR

Per the design spec's PR roadmap, the following are explicitly deferred:

- Journal presets (Cell/Nature/Science) — PR 2
- abc panel labels + flex sizing — PR 2
- Multi-row layouts, `canvas.align`, PanelGrid/PanelText — PR 3
- `pp.legend` `rows=`/`cols=`/`span=` upgrade — PR 4
- mm-precision regression test infra — PR 4.5
- Vector PDF compositing + PanelImage — PR 5 (saves raise `NotImplementedError`)
- Vector SVG composer + `embed_figure` — PR 6 (saves raise `NotImplementedError`)
- `canvas.inspect()` + composer-guide skill — PR 7

## Spike findings addressed in this PR

- **Finding 4** (bare-mpl figure clips xlabel/ylabel/title): the Composer wires up `SubplotsAutoLayout` so decoration space is auto-measured on first draw. `tests/composer/test_decoration_reservations.py` is the regression gate.

Findings 1, 2, 3, 5, 6, 7 are tracked for PRs 5/6 per `spikes/composer/composer-spike.md`.

## Test plan

- [x] All ~30 new composer tests pass (`uv run pytest tests/composer/`).
- [x] Full test suite green (`uv run pytest tests/`).
- [x] mm-precision regression: figure width matches `Canvas(width=N)` exactly; height grows to fit decorations.
- [x] Spike Finding 4: xlabel/ylabel/title don't clip at canvas mediabox edge.
- [x] PDF/SVG saves raise `NotImplementedError` with PR pointers.
- [x] Multi-call `add_row` rejected with PR 3 hint.
- [x] `pp.scatterplot(..., ax=canvas['A'].ax)` works (the canonical plotting idiom).

## Follow-ups

PR 2 (`feat(composer): journal presets + flex sizing + abc labels`) starts from this branch's tip. The Canvas class is already structured to absorb new presets (just add to `presets.py::PRESETS`); flex sizing requires extending `add_row`'s width validator; abc labels require a new sequencer module + per-panel rendering.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Capture the PR URL.

- [ ] **Step 3: Verify the PR is open**

```bash
cd /home/sagemaker-user/publiplots
gh pr view --json number,url,state,title
```

Expected: state OPEN, title matches.

---

## Acceptance criteria for PR 1

The PR is ready for human merge when ALL of:

1. All 9 tasks complete.
2. Full test suite green (existing + ~30 new composer tests).
3. `pp.Canvas('custom', width=174)`, `pp.PanelAxes('A', size=(70, 40))`, `canvas.add_row(...)`, `canvas['A'].ax`, `canvas.savefig('fig.png')` all work end-to-end.
4. `canvas.savefig('fig.pdf')` and `canvas.savefig('fig.svg')` raise `NotImplementedError` with explicit PR 5/PR 6 pointers.
5. Spike Finding 4 regression test passes — Composer-built figures don't clip decorations.
6. CHANGELOG entry added under `[Unreleased]`.
7. PR open against `main` with the exact title `feat(composer): Canvas + PanelAxes (single-row, raster)`.
8. No new dependencies in `pyproject.toml`.
9. Out-of-scope items (journal presets, abc labels, alignment, multi-row, PDF/SVG, embed_figure, inspect, skill) are NOT in the diff.

If any of #1-#9 fail, the PR is not ready.
