# Composer PR 2 Implementation Plan — Journal presets + flex sizing + abc labels

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Promote the Composer from "developer playground" to "actual paper-figure tool". Lands the four authoritative journal presets (Cell / Nature / Nature Methods / Science), `'flex'` panel sizing that absorbs leftover canvas width so the figure equals the canvas budget exactly, an overflow advisor that suggests a per-row scaling factor when panels don't fit, and ultraplot-style abc panel labels (auto-letter sequencing + per-canvas/per-panel `label_style`).

**Architecture:** Pure additions on top of PR 1's foundation. Presets land as new keys in `presets.py::PRESETS` with the verified mm dimensions. Flex sizing extends `PanelAxes.size` validation to accept `('flex', h_mm)`, and `Canvas.add_row` resolves flex widths after computing the slack. Overflow advisor extends `ComposerOverflowError`'s message format. abc labels live in a new `composer/abc_labels.py` module that exposes a sequencer + per-panel renderer wired into `add_row`.

**Tech Stack:** Python ≥ 3.10, matplotlib, pytest. No new dependencies. Editable install at `/home/sagemaker-user/publiplots/`.

**Spec reference:** `docs/superpowers/specs/2026-05-14-composer-design.md` §6.d (PR 2 contract, line 625).

**Journal-dimensions reference (verified 2026-05-15 — see "Verified figure dimensions" section below):**
- Cell: 85 / 114 / 174 mm widths, max-height ~225 mm (derived). Source: cell.com/figureguidelines (Cloudflare-blocked; verified via web.archive.org snapshot 2023-12-30).
- Nature: 89 / 120-136 / 183 mm widths, max-height 247 mm. Source: nature.com/nature/for-authors/final-submission (live).
- Nature Methods: 89 / 120-136 / 180 mm widths (note 180 vs Nature's 183), max-height 247 mm. Source: nature.com/nmeth/submission-guidelines/aip-and-formatting (live).
- Science: 57 / 121 / 184 mm widths, max-height ~240 mm (derived). Source: science.org/content/page/instructions-preparing-initial-manuscript (Cloudflare-blocked; verified via web.archive.org snapshot 2024-04-21).

Each derived value gets a `# DERIVED:` comment in `presets.py` so a future review knows it's not a direct journal quote.

---

## What's IN scope for PR 2

- **Journal presets**: `'cell-1col'`, `'cell-1.5col'`, `'cell-2col'`, `'nature-1col'`, `'nature-1.5col'`, `'nature-2col'`, `'nature-methods-1col'`, `'nature-methods-1.5col'`, `'nature-methods-2col'`, `'science-1col'`, `'science-1.5col'`, `'science-2col'`.
- **Flex sizing**: `pp.PanelAxes(label='B', size=('flex', 40))` — the width grows to fill leftover row space. Multiple flex panels split the leftover equally. Canvas-width invariant: when ≥1 flex panel exists, `figure_size_mm[0] == canvas.width_mm` exactly.
- **Overflow advisor**: when overflow is detected, error message includes a suggested scaling factor (`f"... multiply panel widths by {scale:.3f} to fit"`).
- **abc panel labels**:
  - `pp.Canvas('cell-2col', abc='upper'|'lower'|'A.'|False|['i','ii',...])` — auto-letter mode
  - `pp.PanelAxes(label=None)` (auto-letter slot) and `pp.PanelAxes(label='B.i')` (verbatim override) and `pp.PanelAxes(label=False)` (skip label) — three label modes
  - `canvas.label_style(weight, size, family, loc, pad_mm, border, bbox)` — canvas-wide override
  - per-panel `label_style={...}` kwarg
  - `loc` ∈ {`'ul'`, `'ur'`, `'ll'`, `'lr'`, `'uc'`, `'lc'`, `'cl'`, `'cr'`} (ultraplot vocab)
  - per-preset defaults: Cell upper-bold 9pt, Nature lower-bold 8pt, Science upper-bold 10pt — sourced from preset entries
- **CHANGELOG entry** under `[Unreleased]`.

## What's OUT of scope for PR 2

- Multi-row layouts, `canvas.align`, `PanelGrid`, `PanelText` → **PR 3**
- `pp.legend` `rows=`/`cols=`/`span=`/`ax=` upgrade → **PR 4**
- mm-precision regression test infra (snapshot framework) → **PR 4.5**
- Vector PDF/SVG compositing, `PanelImage`, `embed_figure` → **PR 5/6**
- `canvas.inspect()` + composer-guide skill → **PR 7**
- Promoting `ComposerError`/`ComposerOverflowError` to top-level `pp.*` (a deferred polish from PR 1's review note 3 — fold into PR 7 with the user-facing docs)

This list is the contract for the spec-compliance reviewer. Anything in this list that the implementer ships counts as scope creep and should be flagged.

---

## File Structure

```
src/publiplots/composer/
├── __init__.py                         # MODIFY — no new exports (Canvas/PanelAxes already exposed)
├── canvas.py                           # MODIFY — flex resolution + abc rendering wired into add_row
├── panels.py                           # MODIFY — PanelAxes accepts ('flex', h) + label=None/False
├── presets.py                          # MODIFY — 12 journal presets (was just 'custom')
├── exceptions.py                       # MODIFY — overflow advisor adds suggested scaling factor
├── abc_labels.py                       # NEW — auto-letter sequencer + label_style merge + per-panel render
├── _save.py                            # unchanged
└── _flex.py                            # NEW — pure-geometry flex resolver (no matplotlib)

src/publiplots/__init__.py              # unchanged — no new top-level exports

tests/composer/                         # extend
├── test_canvas_construction.py         # MODIFY — new tests for journal presets; UPDATE the
│                                       #   "only 'custom' in PR 1" + "journal preset not in PR 1"
│                                       #   tests (they were marked for revision in PR 2)
├── test_panels.py                      # MODIFY — extend size-validation for flex; add label=None/False
├── test_add_row.py                     # MODIFY — flex resolution geometry, slack-absorption invariant
├── test_overflow_advisor.py            # NEW — error message contains suggested scale factor
├── test_abc_labels.py                  # NEW — auto-letter sequencing, label modes, label_style
├── test_journal_presets.py             # NEW — verified dimensions per preset, font defaults, max-height
└── test_label_style.py                 # NEW — per-canvas label_style; per-panel override; ultraplot loc vocab

examples/composer/                      # NEW directory
├── cell_2col_simple.py                 # NEW — 2-panel Cell figure with flex
└── nature_2col_with_abc.py             # NEW — 3-panel Nature figure with auto abc labels
```

11 modified or new files (5 modified + 6 new). Estimated ~500 LOC code + ~700 LOC tests + 2 examples.

**File-size budget (warn if exceeded):**
- `canvas.py`: ≤ 350 LOC (PR 1 left it at 285; +flex resolution + abc rendering hooks ~ +50)
- `panels.py`: ≤ 130 LOC (PR 1 at 114; +flex/label-mode validation ~ +15)
- `presets.py`: ≤ 180 LOC (PR 1 at 71; +12 preset entries ~ +90)
- `abc_labels.py`: ≤ 220 LOC (new)
- `_flex.py`: ≤ 80 LOC (new, pure-geometry)
- `exceptions.py`: ≤ 60 LOC (PR 1 at 32; +scale-factor formatting ~ +15)
- Each test file: ≤ 250 LOC

---

## Branch + worktree setup (Task 0)

**Files:** none (git only)

- [ ] **Step 1: Wait for PR 1 (#163) to merge to main**

The plan assumes PR 1 has merged. Confirm:

```bash
cd /home/sagemaker-user/publiplots
git fetch origin --quiet
gh pr view 163 --json state,mergedAt
```

Expected: `{"state":"MERGED","mergedAt":"<timestamp>"}`. If still OPEN, STOP — PR 2 cannot start until PR 1 lands.

- [ ] **Step 2: Sync main**

```bash
cd /home/sagemaker-user/publiplots
git checkout main
git pull origin main --ff-only
```

Expected: HEAD is the squash-merge of PR 1 (e.g. `feat(composer): Canvas + PanelAxes (single-row, raster) (#163)`).

- [ ] **Step 3: Create the feature branch**

```bash
cd /home/sagemaker-user/publiplots
git checkout -b feat/composer-presets-flex-abc-pr2
```

- [ ] **Step 4: Run the existing test suite as a baseline**

```bash
cd /home/sagemaker-user/publiplots
uv run pytest tests/ -q --no-cov --tb=no 2>&1 | tail -3
```

Capture the count. Expected: ~1082 passed, 1 failed (pre-existing residplot — unrelated). The 63 composer tests from PR 1 are now part of the baseline.

---

## Task 1: Overflow advisor — add suggested scaling factor

**Files:**
- Modify: `src/publiplots/composer/exceptions.py`
- Modify: `src/publiplots/composer/canvas.py` — pass scale factor to the error
- Create: `tests/composer/test_overflow_advisor.py`

**Why first:** smallest, most contained change; lets the implementer warm up on the codebase before tackling the bigger surface (presets, flex, abc).

- [ ] **Step 1: Write the failing test**

Create `/home/sagemaker-user/publiplots/tests/composer/test_overflow_advisor.py` with this exact content:

```python
"""Tests for ComposerOverflowError's suggested-scale-factor advisor."""

import pytest

import publiplots as pp
from publiplots.composer.exceptions import ComposerOverflowError


def test_overflow_error_carries_suggested_scale_factor():
    """The overflow advisor proposes a scale factor s such that
    requested_mm * s ≤ available_mm, telling the user how much to
    shrink the panels."""
    err = ComposerOverflowError(
        "row width 200mm exceeds canvas budget 174mm",
        requested_mm=200.0,
        available_mm=174.0,
    )
    # PR 2 adds .scale_to_fit; the formula is available / requested.
    expected = 174.0 / 200.0
    assert abs(err.scale_to_fit - expected) < 1e-9


def test_overflow_error_message_mentions_scale_factor():
    """The default __str__ includes the scale factor so users see it
    without inspecting attributes."""
    err = ComposerOverflowError(
        "row width 200mm exceeds canvas budget 174mm",
        requested_mm=200.0,
        available_mm=174.0,
    )
    msg = str(err)
    # 174/200 = 0.870 — must appear in the message at 3-digit precision.
    assert "0.87" in msg


def test_canvas_add_row_overflow_message_includes_scale_factor():
    """canvas.add_row's overflow path wraps the error with a hint;
    the user-visible message must include both the dim numbers AND
    the suggested scale factor."""
    canvas = pp.Canvas("custom", width=174.0)
    with pytest.raises(ComposerOverflowError) as exc_info:
        canvas.add_row(
            pp.PanelAxes(label="A", size=(100.0, 40.0)),
            pp.PanelAxes(label="B", size=(100.0, 40.0)),
        )
    msg = str(exc_info.value)
    assert "multiply" in msg.lower() or "scale" in msg.lower()
    # Suggested scale = 174 / (200 + decorations) ≈ 174/231 ≈ 0.753
    assert "0.7" in msg


def test_overflow_error_scale_factor_is_one_when_fit_exact():
    """Edge case: requested == available → scale = 1.0 (no shrinkage
    needed). Should not raise — but if it did, the factor would be 1.0."""
    err = ComposerOverflowError(
        "edge case",
        requested_mm=174.0,
        available_mm=174.0,
    )
    assert err.scale_to_fit == 1.0


def test_overflow_error_scale_factor_handles_zero_requested():
    """Defensive: if requested is 0 (degenerate), scale is +inf or 1.0
    — pick a stable, non-NaN convention. Use 1.0 (nothing to shrink)."""
    err = ComposerOverflowError(
        "degenerate",
        requested_mm=0.0,
        available_mm=174.0,
    )
    # Per implementation: if requested <= 0, return 1.0 (no shrink).
    assert err.scale_to_fit == 1.0
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /home/sagemaker-user/publiplots
uv run pytest tests/composer/test_overflow_advisor.py -v --no-cov
```

Expected: tests FAIL with `AttributeError: 'ComposerOverflowError' has no attribute 'scale_to_fit'`.

- [ ] **Step 3: Implement the scale-factor attribute**

Edit `/home/sagemaker-user/publiplots/src/publiplots/composer/exceptions.py`. Replace the `ComposerOverflowError` class with this exact content (preserving the file's existing imports / docstring / `ComposerError` class):

```python
class ComposerOverflowError(ComposerError):
    """Raised when a row's panels overflow the canvas width budget.

    Carries ``requested_mm`` and ``available_mm`` attributes so callers
    can format helpful messages without re-parsing the string. Also
    exposes ``scale_to_fit`` — a multiplicative factor that, when
    applied to all panel widths in the offending row, makes the row
    fit the canvas width budget.

    The default ``__str__`` appends the suggested scale factor to the
    user's message so it surfaces in tracebacks without further work.
    """

    def __init__(
        self,
        message: str,
        *,
        requested_mm: float,
        available_mm: float,
    ) -> None:
        self.requested_mm = float(requested_mm)
        self.available_mm = float(available_mm)
        # Scale factor: clamp degenerate cases (requested <= 0 → 1.0;
        # requested == available → 1.0). Otherwise available/requested.
        if self.requested_mm <= 0.0:
            self.scale_to_fit = 1.0
        else:
            self.scale_to_fit = self.available_mm / self.requested_mm
        # Build the augmented message. If scale < 1, suggest shrinkage;
        # if scale >= 1, just keep the user's message (no advisor needed).
        if self.scale_to_fit < 1.0:
            advisor = (
                f" — multiply panel widths by {self.scale_to_fit:.3f} to fit"
            )
            super().__init__(message + advisor)
        else:
            super().__init__(message)
```

- [ ] **Step 4: Update `add_row`'s overflow message to drop the now-redundant inline message text**

Edit `/home/sagemaker-user/publiplots/src/publiplots/composer/canvas.py`. Find the `raise ComposerOverflowError(...)` block in `add_row`. The existing block looks like:

```python
        if requested_width > self._width_mm + 1e-6:  # 1µm tolerance for float noise
            raise ComposerOverflowError(
                f"row width {requested_width:.2f}mm exceeds canvas width "
                f"{self._width_mm:.2f}mm; reduce panel widths or use a wider canvas",
                requested_mm=requested_width,
                available_mm=self._width_mm,
            )
```

Replace with this exact content (the message text is unchanged; the advisor string is appended automatically by `__init__`):

```python
        if requested_width > self._width_mm + 1e-6:  # 1µm tolerance for float noise
            raise ComposerOverflowError(
                f"row width {requested_width:.2f}mm exceeds canvas width "
                f"{self._width_mm:.2f}mm; reduce panel widths or use a wider canvas",
                requested_mm=requested_width,
                available_mm=self._width_mm,
            )
```

(Same code — the advisor is appended by `ComposerOverflowError.__init__`, so the call site doesn't change.)

- [ ] **Step 5: Run the new tests**

```bash
cd /home/sagemaker-user/publiplots
uv run pytest tests/composer/test_overflow_advisor.py -v --no-cov
```

Expected: 5 passed.

- [ ] **Step 6: Run regression to confirm PR 1 tests still pass**

```bash
cd /home/sagemaker-user/publiplots
uv run pytest tests/composer/ -q --no-cov 2>&1 | tail -3
```

Expected: 68 passed (63 PR1 + 5 new). NOTE: the PR 1 test `test_composer_overflow_error_carries_offending_dim` asserts that "200" and "174" appear in `str(err)`. With the advisor appended, both still appear (the original message is preserved). If that test now fails, the advisor formatting broke something; STOP and report.

- [ ] **Step 7: Commit**

```bash
cd /home/sagemaker-user/publiplots
git add src/publiplots/composer/exceptions.py \
        tests/composer/test_overflow_advisor.py
git commit -m "feat(composer): overflow advisor — suggest per-row scale factor"
```

(Note: `canvas.py` is NOT in the commit — Step 4 confirmed the call site doesn't change.)

---

## Task 2: PanelAxes flex sizing + label-mode validation

**Files:**
- Modify: `src/publiplots/composer/panels.py`
- Create: `src/publiplots/composer/_flex.py`
- Modify: `tests/composer/test_panels.py` — extend size-validation tests + new label-mode tests

**Why second:** flex sizing changes `PanelAxes.size`'s validation contract; abc labels change `label`'s. Land both validation extensions before the geometry/rendering code that consumes them (Tasks 3-5).

- [ ] **Step 1: Write the failing tests**

Append to `/home/sagemaker-user/publiplots/tests/composer/test_panels.py` this exact content:

```python


# ---------------------------------------------------------------------------
# PR 2: flex sizing — size=('flex', h_mm) is now valid
# ---------------------------------------------------------------------------

def test_panel_axes_size_accepts_flex_width():
    p = PanelAxes(label="A", size=("flex", 40.0))
    assert p.size == ("flex", 40.0)


def test_panel_axes_flex_width_height_must_still_be_positive_numeric():
    with pytest.raises(ValueError, match="positive"):
        PanelAxes(label="A", size=("flex", 0.0))
    with pytest.raises(ValueError, match="positive"):
        PanelAxes(label="A", size=("flex", -1.0))


def test_panel_axes_flex_height_must_be_numeric():
    with pytest.raises(ValueError, match="numeric"):
        PanelAxes(label="A", size=("flex", "flex"))


def test_panel_axes_flex_width_only_keyword_accepted():
    """The ONLY string accepted in the width slot is 'flex'. Anything else
    is still rejected as non-numeric."""
    with pytest.raises(ValueError, match="numeric|flex"):
        PanelAxes(label="A", size=("auto", 40.0))


def test_panel_axes_size_height_cannot_be_flex():
    """PR 2 does NOT add flex height (PR 3 might — for vfill policies).
    Flex is width-only in PR 2."""
    with pytest.raises(ValueError, match="numeric|flex"):
        PanelAxes(label="A", size=(70.0, "flex"))


# ---------------------------------------------------------------------------
# PR 2: label modes — None (auto), False (no label), str (verbatim)
# ---------------------------------------------------------------------------

def test_panel_axes_label_none_is_auto_slot():
    """label=None reserves an auto-letter slot; resolution happens in
    canvas.add_row() based on the canvas's abc=. PR 1 required str;
    PR 2 makes None valid."""
    p = PanelAxes(label=None, size=(70.0, 40.0))
    assert p.label is None


def test_panel_axes_label_false_is_no_label():
    """label=False explicitly suppresses any label rendering and is
    skipped from the auto-letter sequence."""
    p = PanelAxes(label=False, size=(70.0, 40.0))
    assert p.label is False


def test_panel_axes_label_str_unchanged():
    """The PR 1 contract for str labels is preserved verbatim."""
    p = PanelAxes(label="B.i", size=(70.0, 40.0))
    assert p.label == "B.i"


def test_panel_axes_label_other_types_rejected():
    """label must be None, False, or a str. int/float/list/etc. raise."""
    with pytest.raises(TypeError, match="label must be"):
        PanelAxes(label=123, size=(70.0, 40.0))
    with pytest.raises(TypeError, match="label must be"):
        PanelAxes(label=["A", "B"], size=(70.0, 40.0))


def test_panel_axes_label_true_rejected():
    """label=True isn't a meaningful state and we don't want users
    confusing it with abc=True (which lives on the Canvas, not the panel)."""
    with pytest.raises(TypeError, match="label must be"):
        PanelAxes(label=True, size=(70.0, 40.0))


# ---------------------------------------------------------------------------
# PR 2: label_style override at panel construction
# ---------------------------------------------------------------------------

def test_panel_axes_accepts_label_style_dict():
    p = PanelAxes(label="A", size=(70.0, 40.0), label_style={"loc": "ur", "size": 11})
    assert p.label_style == {"loc": "ur", "size": 11}


def test_panel_axes_label_style_defaults_to_none():
    p = PanelAxes(label="A", size=(70.0, 40.0))
    assert p.label_style is None


def test_panel_axes_label_style_must_be_mapping_or_none():
    with pytest.raises(TypeError, match="label_style"):
        PanelAxes(label="A", size=(70.0, 40.0), label_style="not-a-dict")
```

- [ ] **Step 2: Find and remove the now-obsolete PR 1 tests that contradict the PR 2 contract**

Two PR 1 tests in `test_panels.py` must be UPDATED:

- `test_panel_axes_label_required_in_pr1` (asserts `label=None` raises) — DELETE this test
- `test_panel_axes_size_must_be_numeric` (asserts `('flex', 40.0)` raises) — DELETE this test

Use the `Edit` tool to delete each function block, including its decorators and docstring. Verify with:

```bash
cd /home/sagemaker-user/publiplots
grep -n "test_panel_axes_label_required_in_pr1\|test_panel_axes_size_must_be_numeric" tests/composer/test_panels.py
```

Expected: no output (both functions removed).

- [ ] **Step 3: Run the test file to confirm new tests fail**

```bash
cd /home/sagemaker-user/publiplots
uv run pytest tests/composer/test_panels.py -v --no-cov
```

Expected: PR 1 tests still pass (minus the 2 deleted); 13 new tests fail with various errors.

- [ ] **Step 4: Implement label-mode validation in `panels.py`**

Edit `/home/sagemaker-user/publiplots/src/publiplots/composer/panels.py`. Replace the `PanelAxes` class with this exact content (preserves the module docstring, imports, `PanelKind` Literal, and `Panel` class):

```python
@dataclass(frozen=True)
class PanelAxes:
    """Input record for an axes panel.

    Pass this to :meth:`Canvas.add_row` to declare a panel containing a
    single matplotlib Axes. The canvas allocates the Axes at the panel's
    mm rect and stores the result as a :class:`Panel`.

    Parameters
    ----------
    label : str, None, or False
        Caption-addressable identifier:

        - ``str`` — verbatim label (e.g., ``'A'``, ``'B.i'``).
        - ``None`` — auto-letter slot. The canvas's ``abc=`` template
          resolves it to the next sequence value (``'A'`` if
          ``abc='upper'``, ``'a'`` if ``'lower'``, etc.).
        - ``False`` — explicitly suppress label rendering and skip from
          the auto-letter sequence.
    size : tuple of (width, height_mm)
        Panel mm rect:

        - ``(w_mm, h_mm)`` — both pinned (PR 1 contract).
        - ``('flex', h_mm)`` — width grows to absorb leftover row width.
          Multiple flex panels in a row split the leftover equally.

        The height must always be a positive numeric. PR 2 does NOT
        introduce flex heights.
    label_style : Mapping or None, default None
        Per-panel override of the canvas-wide ``label_style``. Accepts
        any subset of the canvas-wide keys (``loc``, ``size``,
        ``weight``, ``family``, ``pad_mm``, ``border``, ``bbox``).
        Missing keys fall through to the canvas default.

    Notes
    -----
    Frozen dataclass — instances are immutable and hashable.
    """

    label: Any            # str | None | False (type-narrowing in __post_init__)
    size: Tuple[Any, Any]  # (float | 'flex', float)
    label_style: Optional[Mapping[str, Any]] = None

    def __post_init__(self) -> None:
        # Validate label: must be str, None, or False (NOT True).
        if self.label is not None and self.label is not False:
            if not isinstance(self.label, str):
                raise TypeError(
                    f"label must be a str, None (auto-letter), or False "
                    f"(no label); got {type(self.label).__name__}"
                )
        # Reject the ambiguous bool-True case (False is fine because
        # `is not False` short-circuits above).
        if self.label is True:
            raise TypeError(
                "label must be a str, None (auto-letter), or False "
                "(no label); got True (did you mean abc=True on the Canvas?)"
            )

        # Validate size: must be a 2-tuple
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

        # Validate width: 'flex' string OR positive numeric
        if isinstance(w, str):
            if w != "flex":
                raise ValueError(
                    f"size width must be numeric or 'flex', got {w!r}"
                )
            # 'flex' width is OK — leave as the string sentinel.
        else:
            try:
                fw = float(w)
            except (TypeError, ValueError):
                raise ValueError(f"size width must be numeric or 'flex', got {w!r}")
            if fw <= 0:
                raise ValueError(f"size width must be positive, got {fw}")
            w = fw  # coerce int → float

        # Validate height: must be positive numeric (no 'flex' in PR 2)
        if isinstance(h, str):
            raise ValueError(
                f"size height must be numeric, got {h!r} "
                "('flex' is width-only in PR 2)"
            )
        try:
            fh = float(h)
        except (TypeError, ValueError):
            raise ValueError(f"size height must be numeric, got {h!r}")
        if fh <= 0:
            raise ValueError(f"size height must be positive, got {fh}")
        h = fh

        # Validate label_style: must be Mapping or None
        if self.label_style is not None:
            if not hasattr(self.label_style, "keys"):
                raise TypeError(
                    f"label_style must be a Mapping or None, "
                    f"got {type(self.label_style).__name__}"
                )

        # Coerce normalized values onto the frozen instance.
        object.__setattr__(self, "size", (w, h))
```

You will also need to add `Mapping` to the imports at the top of `panels.py`:

```python
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Optional, Tuple
```

- [ ] **Step 5: Implement the pure-geometry flex resolver**

Create `/home/sagemaker-user/publiplots/src/publiplots/composer/_flex.py` with this exact content:

```python
"""Pure-geometry flex resolver for PR 2 sizing.

Given a row of panel widths (some pinned mm, some 'flex' sentinels) and
a canvas budget minus decorations, resolve every flex panel to a
concrete mm width such that:

  sum(resolved_widths) + decorations + (n-1) * hpad == canvas_width

Multiple flex panels split the leftover width equally. If no flex panels
exist, the function is a no-op (returns the input pinned widths) and
the caller is responsible for raising overflow if pinned + decorations
> canvas_width.

No matplotlib imports. Pure math.
"""

from typing import Any, Sequence, Tuple


def resolve_flex_widths(
    raw_widths: Sequence[Any],   # mix of float and the literal 'flex'
    *,
    canvas_width_mm: float,
    decorations_width_mm: float,
) -> Tuple[Tuple[float, ...], int]:
    """Resolve 'flex' entries to concrete mm widths.

    Parameters
    ----------
    raw_widths : Sequence
        Mix of floats (pinned mm widths) and the string ``'flex'``.
    canvas_width_mm : float
        The canvas's declared width budget.
    decorations_width_mm : float
        Sum of outer_pad×2 + ncols×ylabel_space + ncols×right + (n-1)×hpad.
        Pre-computed by the caller from rcParams.

    Returns
    -------
    resolved_widths : tuple of float
        All entries are concrete floats; flex entries replaced with
        their share of the leftover width.
    n_flex : int
        Count of flex entries (0 when no flex). When ``n_flex == 0``
        the caller treats the widths as pinned-only and may raise
        :class:`ComposerOverflowError`. When ``n_flex >= 1`` the
        resolver fills the canvas budget exactly (modulo float noise).

    Raises
    ------
    ValueError
        If a flex entry would resolve to a non-positive width (i.e.
        the pinned widths plus decorations already exceed the canvas
        budget; flex panels would need to be 0 mm or negative).
    """
    n_flex = sum(1 for w in raw_widths if w == "flex")
    if n_flex == 0:
        # No flex; just coerce + return.
        return tuple(float(w) for w in raw_widths), 0

    pinned_total = sum(float(w) for w in raw_widths if w != "flex")
    leftover = canvas_width_mm - decorations_width_mm - pinned_total
    per_flex = leftover / n_flex

    if per_flex <= 0.0:
        raise ValueError(
            f"flex panels would resolve to non-positive width "
            f"({per_flex:.2f} mm each); pinned panels {pinned_total:.2f}mm "
            f"+ decorations {decorations_width_mm:.2f}mm already exceed "
            f"canvas budget {canvas_width_mm:.2f}mm"
        )

    resolved = tuple(per_flex if w == "flex" else float(w) for w in raw_widths)
    return resolved, n_flex
```

- [ ] **Step 6: Run the panels tests**

```bash
cd /home/sagemaker-user/publiplots
uv run pytest tests/composer/test_panels.py -v --no-cov
```

Expected: all tests PASS. The 2 deleted PR 1 tests are gone; the 13 new PR 2 tests pass.

- [ ] **Step 7: Run regression**

```bash
cd /home/sagemaker-user/publiplots
uv run pytest tests/composer/ -q --no-cov 2>&1 | tail -3
```

Expected: 79 passed (68 from Task 1 + 13 new − 2 deleted = 79). Note the PR 1 `test_canvas_indexing_returns_panel_for_known_label` and friends still pass because they use `label="A"` (str), not None or False.

- [ ] **Step 8: Commit**

```bash
cd /home/sagemaker-user/publiplots
git add src/publiplots/composer/panels.py \
        src/publiplots/composer/_flex.py \
        tests/composer/test_panels.py
git commit -m "feat(composer): PanelAxes — flex width + label modes (None/False/str)"
```

---

## Task 3: Wire flex resolution into `add_row`

**Files:**
- Modify: `src/publiplots/composer/canvas.py` — call `_flex.resolve_flex_widths` in `add_row`
- Modify: `tests/composer/test_add_row.py` — add flex geometry tests

The current `add_row` computes `col_widths = tuple(p.size[0] for p in panels)` and then validates `requested_width <= canvas_width`. With flex, the flow becomes:

1. Compute decorations_width.
2. Call `resolve_flex_widths(raw_widths, canvas_width_mm, decorations_width_mm)`.
3. If `n_flex == 0`: keep PR 1's overflow check (now using resolved pinned-only widths).
4. If `n_flex >= 1`: skip overflow check (resolver raised already if leftover ≤ 0).
5. Use resolved widths for FigureLayout.

- [ ] **Step 1: Write the failing tests**

Append to `/home/sagemaker-user/publiplots/tests/composer/test_add_row.py` this exact content:

```python


# ---------------------------------------------------------------------------
# PR 2: flex sizing geometry
# ---------------------------------------------------------------------------

def test_add_row_with_flex_panel_fills_canvas_width_exactly():
    """When ≥1 flex panel exists, figure_width equals canvas.width_mm
    exactly (modulo float noise) — the slack is absorbed."""
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(
        pp.PanelAxes(label="A", size=(60.0, 40.0)),
        pp.PanelAxes(label="B", size=("flex", 40.0)),
    )
    w_mm, _ = canvas.figure_size_mm
    assert abs(w_mm - 174.0) < MM_TOL


def test_add_row_with_flex_panel_resolves_to_leftover_width():
    """One pinned 60mm panel + one flex panel in a 174mm canvas.
    Decorations: 2 + 10 + 60 + 2 + 3 + 10 + flex + 2 + 2 = 91 + flex
    Setting equal to 174: flex = 83mm."""
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(
        pp.PanelAxes(label="A", size=(60.0, 40.0)),
        pp.PanelAxes(label="B", size=("flex", 40.0)),
    )
    a_w = canvas["A"].size_mm[0]
    b_w = canvas["B"].size_mm[0]
    assert abs(a_w - 60.0) < MM_TOL
    assert abs(b_w - 83.0) < MM_TOL


def test_add_row_two_flex_panels_split_leftover_equally():
    """Two flex panels in a 174mm canvas with no pinned panels.
    Decorations: 2 + 10 + flex + 2 + 3 + 10 + flex + 2 + 2 = 31 + 2*flex
    Setting equal to 174: each flex = 71.5mm."""
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(
        pp.PanelAxes(label="A", size=("flex", 40.0)),
        pp.PanelAxes(label="B", size=("flex", 40.0)),
    )
    a_w = canvas["A"].size_mm[0]
    b_w = canvas["B"].size_mm[0]
    assert abs(a_w - 71.5) < MM_TOL
    assert abs(b_w - 71.5) < MM_TOL


def test_add_row_flex_with_pinned_overflow_raises():
    """If pinned panels alone overflow the canvas, the flex resolver
    refuses to make flex panels go to ≤0 mm."""
    canvas = pp.Canvas("custom", width=174.0)
    with pytest.raises((ComposerOverflowError, ValueError)) as exc_info:
        canvas.add_row(
            pp.PanelAxes(label="A", size=(150.0, 40.0)),
            pp.PanelAxes(label="B", size=("flex", 40.0)),
        )
    # Either error type is acceptable; the message should mention flex
    # or non-positive.
    msg = str(exc_info.value).lower()
    assert "flex" in msg or "non-positive" in msg or "exceed" in msg


def test_add_row_pinned_only_still_uses_pr1_overflow_path():
    """If no flex panels, the overflow check + advisor from Task 1 fires."""
    canvas = pp.Canvas("custom", width=174.0)
    with pytest.raises(ComposerOverflowError) as exc_info:
        canvas.add_row(
            pp.PanelAxes(label="A", size=(100.0, 40.0)),
            pp.PanelAxes(label="B", size=(100.0, 40.0)),
        )
    msg = str(exc_info.value)
    # Task 1 advisor mentioned scaling — verify it's still in the message.
    assert "multiply" in msg.lower() or "0.7" in msg
```

(Note: `MM_TOL`, `pp`, `pytest`, `ComposerOverflowError` are already imported at the top of the file from PR 1.)

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd /home/sagemaker-user/publiplots
uv run pytest tests/composer/test_add_row.py -v --no-cov 2>&1 | tail -30
```

Expected: 5 new tests fail (e.g., `test_add_row_with_flex_panel_fills_canvas_width_exactly` will fail on the figure-width assertion because PR 1's geometry doesn't absorb slack).

- [ ] **Step 3: Modify `add_row` to call the flex resolver**

Edit `/home/sagemaker-user/publiplots/src/publiplots/composer/canvas.py`. Find the geometry section in `add_row`. The current shape (after PR 1) is roughly:

```python
        # --- compute geometry ---------------------------------------
        outer_pad = float(resolve_param("subplots.outer_pad", None))
        ...
        col_widths = tuple(p.size[0] for p in panels)
        ...
        requested_width = panels_width + decorations_width
        if requested_width > self._width_mm + 1e-6:
            raise ComposerOverflowError(...)
```

Replace from `col_widths = ...` through the overflow `raise` with this exact content:

```python
        # Raw widths — may contain 'flex' sentinels which we'll resolve next.
        raw_widths = tuple(p.size[0] for p in panels)
        # All panels in one row share the row height; PR 1 requires equal
        # heights (PR 2 still does — flex is width-only). Use the max for now.
        row_height = max(p.size[1] for p in panels)

        ncols = len(panels)
        # Canvas width budget: panels + (n-1) hpads + ylabel reservations
        # for each column + right reservation for each column + outer pads.
        decorations_width = (
            2 * outer_pad
            + ncols * ylabel_space
            + ncols * right
            + max(ncols - 1, 0) * hpad
        )

        # --- resolve flex sizing ------------------------------------
        from publiplots.composer._flex import resolve_flex_widths
        try:
            col_widths, n_flex = resolve_flex_widths(
                raw_widths,
                canvas_width_mm=self._width_mm,
                decorations_width_mm=decorations_width,
            )
        except ValueError as e:
            # The resolver raises ValueError when pinned widths alone
            # overflow. Convert to ComposerOverflowError so the user
            # gets the suggested-scale-factor advisor.
            pinned_total = sum(float(w) for w in raw_widths if w != "flex")
            requested = pinned_total + decorations_width
            raise ComposerOverflowError(
                str(e),
                requested_mm=requested,
                available_mm=self._width_mm,
            )

        # --- pinned-only overflow check (PR 1 path) -----------------
        if n_flex == 0:
            panels_width = sum(col_widths)
            requested_width = panels_width + decorations_width
            if requested_width > self._width_mm + 1e-6:  # 1µm tolerance
                raise ComposerOverflowError(
                    f"row width {requested_width:.2f}mm exceeds canvas width "
                    f"{self._width_mm:.2f}mm; reduce panel widths or use a wider canvas",
                    requested_mm=requested_width,
                    available_mm=self._width_mm,
                )
        # When n_flex >= 1, the resolver guarantees figure width == canvas
        # width exactly (modulo float noise). No additional check needed.
```

- [ ] **Step 4: Update the panel-record creation to use resolved widths**

Find the `for col_idx, panel_input in enumerate(panels):` loop later in `add_row`. The line

```python
                size_mm=(panel_input.size[0], panel_input.size[1]),
```

must use the resolved width when the input was flex. Replace that line with:

```python
                size_mm=(col_widths[col_idx], panel_input.size[1]),
```

- [ ] **Step 5: Run the new tests + regression**

```bash
cd /home/sagemaker-user/publiplots
uv run pytest tests/composer/test_add_row.py tests/composer/test_overflow_advisor.py tests/composer/test_panels.py -v --no-cov 2>&1 | tail -10
```

Expected: ALL pass. The new flex tests resolve correctly; the pinned-only PR 1 tests still raise overflow.

- [ ] **Step 6: Full composer regression**

```bash
cd /home/sagemaker-user/publiplots
uv run pytest tests/composer/ -q --no-cov 2>&1 | tail -3
```

Expected: 84 passed (79 + 5 new flex tests).

- [ ] **Step 7: Commit**

```bash
cd /home/sagemaker-user/publiplots
git add src/publiplots/composer/canvas.py \
        tests/composer/test_add_row.py
git commit -m "feat(composer): add_row flex sizing — figure width matches canvas budget"
```

---

## Task 4: Journal presets (Cell, Nature, Nat Methods, Science)

**Files:**
- Modify: `src/publiplots/composer/presets.py`
- Create: `tests/composer/test_journal_presets.py`
- Modify: `tests/composer/test_canvas_construction.py` — UPDATE the "only 'custom' in PR 1" test + remove the "journal preset not in PR 1" test

This is the largest single addition by LOC. The 12 presets each carry: `default_width_mm`, `max_height_mm`, `abc` default ('upper'/'lower'/'A.'), `label_size_pt` default. The Canvas constructor accepts the new names; if the user omits `width=`, it falls back to the preset's `default_width_mm`. abc + label_size defaults flow through to `Canvas.label_style` (Task 5 wires that).

- [ ] **Step 1: Write the failing tests**

Create `/home/sagemaker-user/publiplots/tests/composer/test_journal_presets.py` with this exact content:

```python
"""Verified-dimension tests for the 12 journal presets.

Source verification (2026-05-15):
- Cell: cell.com/figureguidelines (verified via web.archive.org snapshot 2023-12-30)
- Nature: nature.com/nature/for-authors/final-submission (live)
- Nature Methods: nature.com/nmeth/submission-guidelines/aip-and-formatting (live)
- Science: science.org/...instructions-preparing-initial-manuscript (verified via
  web.archive.org snapshot 2024-04-21)

Three values are DERIVED rather than direct quotes (flagged in presets.py):
- Cell max-height 225mm (page-fit derivation)
- Science max-height 240mm (trim derivation)
- Nature Methods 1-col / 1.5-col widths (inherited from Nature family)
"""

import pytest

import publiplots as pp
from publiplots.composer.presets import PRESETS, resolve_preset


# ---------------------------------------------------------------------------
# Preset registry — count + naming
# ---------------------------------------------------------------------------

EXPECTED_PRESETS = {
    "custom",
    "cell-1col", "cell-1.5col", "cell-2col",
    "nature-1col", "nature-1.5col", "nature-2col",
    "nature-methods-1col", "nature-methods-1.5col", "nature-methods-2col",
    "science-1col", "science-1.5col", "science-2col",
}


def test_all_journal_presets_registered():
    assert set(PRESETS.keys()) == EXPECTED_PRESETS


# ---------------------------------------------------------------------------
# Cell — 85 / 114 / 174 mm (verified)
# ---------------------------------------------------------------------------

def test_cell_1col_width():
    p = resolve_preset("cell-1col", width=None)
    assert p["width_mm"] == 85.0


def test_cell_1_5col_width():
    p = resolve_preset("cell-1.5col", width=None)
    assert p["width_mm"] == 114.0


def test_cell_2col_width():
    p = resolve_preset("cell-2col", width=None)
    assert p["width_mm"] == 174.0


def test_cell_max_height_is_225_derived():
    """Cell only specifies "fit on a single page". 225mm is the
    page-fit derivation; flagged DERIVED in presets.py."""
    p = resolve_preset("cell-2col", width=None)
    assert p["max_height_mm"] == 225.0


# ---------------------------------------------------------------------------
# Nature — 89 / 120-136 / 183 mm + 247 mm max (verified live)
# ---------------------------------------------------------------------------

def test_nature_1col_width():
    p = resolve_preset("nature-1col", width=None)
    assert p["width_mm"] == 89.0


def test_nature_1_5col_width():
    """Nature publishes 120-136 mm range; we commit the lower bound 120
    to leave room for the user; they can override via width=."""
    p = resolve_preset("nature-1.5col", width=None)
    assert p["width_mm"] == 120.0


def test_nature_2col_width():
    p = resolve_preset("nature-2col", width=None)
    assert p["width_mm"] == 183.0


def test_nature_max_height_is_247():
    p = resolve_preset("nature-2col", width=None)
    assert p["max_height_mm"] == 247.0


# ---------------------------------------------------------------------------
# Nature Methods — inherits Nature columns BUT caps double-col at 180 mm
# ---------------------------------------------------------------------------

def test_nature_methods_1col_inherits_from_nature():
    p = resolve_preset("nature-methods-1col", width=None)
    assert p["width_mm"] == 89.0


def test_nature_methods_1_5col_inherits_from_nature():
    p = resolve_preset("nature-methods-1.5col", width=None)
    assert p["width_mm"] == 120.0


def test_nature_methods_2col_caps_at_180_not_183():
    """Nature Methods AIP page quotes 180mm max; differs from Nature's 183."""
    p = resolve_preset("nature-methods-2col", width=None)
    assert p["width_mm"] == 180.0


def test_nature_methods_max_height_inherits():
    p = resolve_preset("nature-methods-2col", width=None)
    assert p["max_height_mm"] == 247.0


# ---------------------------------------------------------------------------
# Science — 57 / 121 / 184 mm + 240 mm max-height (derived)
# ---------------------------------------------------------------------------

def test_science_1col_width():
    """Verified 57mm (corrects ultraplot's rounded 55mm)."""
    p = resolve_preset("science-1col", width=None)
    assert p["width_mm"] == 57.0


def test_science_1_5col_width():
    p = resolve_preset("science-1.5col", width=None)
    assert p["width_mm"] == 121.0


def test_science_2col_width():
    p = resolve_preset("science-2col", width=None)
    assert p["width_mm"] == 184.0


def test_science_max_height_is_240_derived():
    """Science doesn't quote a max figure height; ~240 mm is the trim
    derivation. Flagged DERIVED in presets.py."""
    p = resolve_preset("science-2col", width=None)
    assert p["max_height_mm"] == 240.0


# ---------------------------------------------------------------------------
# Width override — user can pass width= to override the preset default
# ---------------------------------------------------------------------------

def test_user_can_override_preset_width():
    """User wants Nature-2col but with custom 175mm width — they pass width=."""
    p = resolve_preset("nature-2col", width=175.0)
    assert p["width_mm"] == 175.0


def test_custom_preset_unchanged_from_pr1():
    """The 'custom' escape hatch is unchanged: requires explicit width."""
    with pytest.raises(ValueError, match="width"):
        resolve_preset("custom", width=None)


# ---------------------------------------------------------------------------
# abc default per preset (for Task 5)
# ---------------------------------------------------------------------------

def test_cell_default_abc_is_upper():
    p = resolve_preset("cell-2col", width=None)
    assert p["abc_default"] == "upper"


def test_nature_default_abc_is_lower():
    p = resolve_preset("nature-2col", width=None)
    assert p["abc_default"] == "lower"


def test_science_default_abc_is_upper():
    p = resolve_preset("science-2col", width=None)
    assert p["abc_default"] == "upper"


def test_custom_default_abc_is_upper():
    p = resolve_preset("custom", width=174.0)
    assert p["abc_default"] == "upper"


# ---------------------------------------------------------------------------
# Default label font size per preset (for Task 5)
# ---------------------------------------------------------------------------

def test_cell_default_label_size_is_9pt():
    p = resolve_preset("cell-2col", width=None)
    assert p["label_size_pt"] == 9


def test_nature_default_label_size_is_8pt():
    p = resolve_preset("nature-2col", width=None)
    assert p["label_size_pt"] == 8


def test_science_default_label_size_is_10pt():
    p = resolve_preset("science-2col", width=None)
    assert p["label_size_pt"] == 10


# ---------------------------------------------------------------------------
# Canvas construction with journal presets (top-level smoke)
# ---------------------------------------------------------------------------

def test_canvas_cell_2col_no_width_arg_works():
    """Cell-2col supplies its own default width; user need not pass width=."""
    canvas = pp.Canvas("cell-2col")
    assert canvas.width_mm == 174.0


def test_canvas_nature_2col_no_width_arg_works():
    canvas = pp.Canvas("nature-2col")
    assert canvas.width_mm == 183.0


def test_canvas_science_2col_no_width_arg_works():
    canvas = pp.Canvas("science-2col")
    assert canvas.width_mm == 184.0
```

- [ ] **Step 2: Update the obsolete PR 1 tests in `test_canvas_construction.py`**

Two tests must change:

(a) `test_only_custom_preset_in_pr1` — UPDATE to assert the full PR 2 set:

Find the function and replace its body with:

```python
def test_full_journal_preset_set_in_pr2():
    """PR 2 ships Cell / Nature / Nat Methods / Science presets plus
    the 'custom' escape hatch. PR 1 only had 'custom'."""
    expected = {
        "custom",
        "cell-1col", "cell-1.5col", "cell-2col",
        "nature-1col", "nature-1.5col", "nature-2col",
        "nature-methods-1col", "nature-methods-1.5col", "nature-methods-2col",
        "science-1col", "science-1.5col", "science-2col",
    }
    assert set(PRESETS.keys()) == expected
```

(Rename the function from `test_only_custom_preset_in_pr1` to `test_full_journal_preset_set_in_pr2`.)

(b) `test_canvas_construction_journal_preset_not_in_pr1` — DELETE this test entirely. Its body asserted that `pp.Canvas("cell-2col")` raises `KeyError` — in PR 2 it succeeds.

- [ ] **Step 3: Run the tests to verify they fail**

```bash
cd /home/sagemaker-user/publiplots
uv run pytest tests/composer/test_journal_presets.py tests/composer/test_canvas_construction.py -v --no-cov
```

Expected: most tests fail with `KeyError: 'unknown preset'` for the new presets.

- [ ] **Step 4: Implement the 12 journal presets**

Edit `/home/sagemaker-user/publiplots/src/publiplots/composer/presets.py`. Replace the `PRESETS` dict and `resolve_preset` function with this exact content (preserving the module docstring + imports):

```python
# PRESETS table — verified 2026-05-15.
#
# Sources:
#   Cell:           cell.com/figureguidelines (web.archive.org/2023-12-30 snapshot;
#                   live page is Cloudflare-blocked)
#   Nature:         nature.com/nature/for-authors/final-submission (live)
#   Nature Methods: nature.com/nmeth/submission-guidelines/aip-and-formatting (live)
#   Science:        science.org/...instructions-preparing-initial-manuscript
#                   (web.archive.org/2024-04-21 snapshot; live is Cloudflare-blocked)
#
# DERIVED values are flagged inline. Three exist:
#   - Cell max_height_mm=225 (Cell only specifies "fit on one 8.5×11 page";
#     225 is the conservative trim minus running heads/captions)
#   - Science max_height_mm=240 (Science quotes no max figure height; 240 is
#     the trim derivation from Science's ~9.75″ usable depth)
#   - Nature Methods 1-col & 1.5-col widths inherit from Nature family;
#     only the 180mm 2-col cap is journal-specific
#
# The `abc_default` and `label_size_pt` keys feed Canvas.label_style
# defaults (Task 5).

PRESETS: Dict[str, Dict[str, Any]] = {
    # 'custom' — caller supplies width, no journal constraints.
    "custom": {
        "default_width_mm": None,
        "max_height_mm": None,
        "abc_default": "upper",
        "label_size_pt": 9,
    },

    # --- Cell Press family ---
    # Source: cell.com/figureguidelines. Widths verified.
    "cell-1col": {
        "default_width_mm": 85.0,
        "max_height_mm": 225.0,  # DERIVED: Cell page-fit (no quoted max)
        "abc_default": "upper",
        "label_size_pt": 9,
    },
    "cell-1.5col": {
        "default_width_mm": 114.0,
        "max_height_mm": 225.0,  # DERIVED
        "abc_default": "upper",
        "label_size_pt": 9,
    },
    "cell-2col": {
        "default_width_mm": 174.0,
        "max_height_mm": 225.0,  # DERIVED
        "abc_default": "upper",
        "label_size_pt": 9,
    },

    # --- Nature ---
    # Source: nature.com author guide. 89/120-136/183mm, 247mm max.
    # We commit 120mm (lower bound of 120-136 range).
    "nature-1col": {
        "default_width_mm": 89.0,
        "max_height_mm": 247.0,
        "abc_default": "lower",
        "label_size_pt": 8,
    },
    "nature-1.5col": {
        "default_width_mm": 120.0,
        "max_height_mm": 247.0,
        "abc_default": "lower",
        "label_size_pt": 8,
    },
    "nature-2col": {
        "default_width_mm": 183.0,
        "max_height_mm": 247.0,
        "abc_default": "lower",
        "label_size_pt": 8,
    },

    # --- Nature Methods ---
    # Source: nature.com/nmeth AIP page. Only 180mm max-width is
    # journal-specific; 1-col + 1.5-col DERIVED from Nature inheritance.
    "nature-methods-1col": {
        "default_width_mm": 89.0,    # DERIVED from Nature family
        "max_height_mm": 247.0,      # DERIVED from Nature family
        "abc_default": "lower",
        "label_size_pt": 8,
    },
    "nature-methods-1.5col": {
        "default_width_mm": 120.0,   # DERIVED from Nature family
        "max_height_mm": 247.0,      # DERIVED from Nature family
        "abc_default": "lower",
        "label_size_pt": 8,
    },
    "nature-methods-2col": {
        "default_width_mm": 180.0,   # journal-specific (Nature is 183mm)
        "max_height_mm": 247.0,      # DERIVED from Nature family
        "abc_default": "lower",
        "label_size_pt": 8,
    },

    # --- Science (AAAS) ---
    # Source: science.org instructions. 57/121/184mm verified
    # (corrects ultraplot's rounded 55/120). 240mm DERIVED.
    "science-1col": {
        "default_width_mm": 57.0,
        "max_height_mm": 240.0,  # DERIVED: Science quotes no max
        "abc_default": "upper",
        "label_size_pt": 10,
    },
    "science-1.5col": {
        "default_width_mm": 121.0,
        "max_height_mm": 240.0,  # DERIVED
        "abc_default": "upper",
        "label_size_pt": 10,
    },
    "science-2col": {
        "default_width_mm": 184.0,
        "max_height_mm": 240.0,  # DERIVED
        "abc_default": "upper",
        "label_size_pt": 10,
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
        Preset key. Must be in :data:`PRESETS`.
    width : float or None
        User-supplied canvas width in mm. Required for ``'custom'``;
        optional for journal presets (which provide a default).

    Returns
    -------
    dict
        Keys: ``width_mm`` (float), ``max_height_mm`` (float or None),
        ``abc_default`` (str — 'upper', 'lower', or 'A.'),
        ``label_size_pt`` (int — preset-suggested label font size).

    Raises
    ------
    KeyError
        If ``name`` is not a known preset.
    ValueError
        If ``width`` is None for a preset that requires it
        (``'custom'`` only), or if ``width`` is non-positive.
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
        "abc_default": spec["abc_default"],
        "label_size_pt": spec["label_size_pt"],
    }
```

- [ ] **Step 5: Run the preset tests**

```bash
cd /home/sagemaker-user/publiplots
uv run pytest tests/composer/test_journal_presets.py tests/composer/test_canvas_construction.py -v --no-cov
```

Expected: ALL pass.

- [ ] **Step 6: Full composer regression**

```bash
cd /home/sagemaker-user/publiplots
uv run pytest tests/composer/ -q --no-cov 2>&1 | tail -3
```

Expected: 84 (Tasks 1-3) + 26 (test_journal_presets) - 1 (deleted "not in PR 1" test) = 109 passed. Adjust if the actual count differs by ±2 due to test renames.

- [ ] **Step 7: Commit**

```bash
cd /home/sagemaker-user/publiplots
git add src/publiplots/composer/presets.py \
        tests/composer/test_journal_presets.py \
        tests/composer/test_canvas_construction.py
git commit -m "feat(composer): journal presets — Cell, Nature, Nat Methods, Science"
```

---

## Task 5: abc panel labels — auto-letter sequencer + label_style

**Files:**
- Create: `src/publiplots/composer/abc_labels.py`
- Modify: `src/publiplots/composer/canvas.py` — accept `abc=` kwarg, store `label_style`, render labels in `add_row`
- Create: `tests/composer/test_abc_labels.py`
- Create: `tests/composer/test_label_style.py`

This is the single largest behavioral addition. The implementation has three pieces:

1. **Sequencer**: given `abc` mode + ordered panels, produce label strings.
2. **Style merger**: canvas-wide style ⊕ per-panel style → resolved style per panel.
3. **Renderer**: place each label as an `ax.text` with the resolved style at the resolved `loc`.

- [ ] **Step 1: Write the failing tests for the sequencer**

Create `/home/sagemaker-user/publiplots/tests/composer/test_abc_labels.py` with this exact content:

```python
"""Tests for abc auto-letter sequencing + render-time label resolution."""

import pytest

import publiplots as pp
from publiplots.composer.abc_labels import resolve_labels


# ---------------------------------------------------------------------------
# resolve_labels — pure-Python sequencer
# ---------------------------------------------------------------------------

def test_resolve_labels_upper_simple():
    out = resolve_labels(
        panel_labels=[None, None, None],
        abc="upper",
    )
    assert out == ["A", "B", "C"]


def test_resolve_labels_lower_simple():
    out = resolve_labels(
        panel_labels=[None, None, None],
        abc="lower",
    )
    assert out == ["a", "b", "c"]


def test_resolve_labels_template_dot():
    out = resolve_labels(
        panel_labels=[None, None, None],
        abc="A.",
    )
    assert out == ["A.", "B.", "C."]


def test_resolve_labels_template_lower_dot():
    out = resolve_labels(
        panel_labels=[None, None, None],
        abc="a.",
    )
    assert out == ["a.", "b.", "c."]


def test_resolve_labels_explicit_list():
    out = resolve_labels(
        panel_labels=[None, None, None, None],
        abc=["i", "ii", "iii", "iv"],
    )
    assert out == ["i", "ii", "iii", "iv"]


def test_resolve_labels_false_disables_all():
    """abc=False suppresses every auto-letter; explicit str labels stay."""
    out = resolve_labels(
        panel_labels=[None, "X", None],
        abc=False,
    )
    assert out == [None, "X", None]


def test_resolve_labels_explicit_str_passes_through_and_consumes_slot():
    """A panel with label='X' stays as 'X' AND consumes a slot in the
    sequence: subsequent None panels still increment correctly."""
    out = resolve_labels(
        panel_labels=[None, "Z", None],
        abc="upper",
    )
    # Slot 0 → 'A'; slot 1 → 'Z' (verbatim, but consumes the 'B' slot);
    # slot 2 → 'C' (next after the consumed 'B').
    assert out == ["A", "Z", "C"]


def test_resolve_labels_false_panel_skips_sequence_slot():
    """label=False panels are SKIPPED from the sequence — the next auto
    panel uses the next letter."""
    out = resolve_labels(
        panel_labels=[None, False, None],
        abc="upper",
    )
    # Slot 0 → 'A'; slot 1 → False (no label, skip from sequence);
    # slot 2 → 'B' (NOT 'C' — the False didn't consume a slot).
    assert out == ["A", False, "B"]


def test_resolve_labels_explicit_list_too_short_raises():
    with pytest.raises(ValueError, match="abc list"):
        resolve_labels(
            panel_labels=[None, None, None, None],
            abc=["i", "ii"],
        )


def test_resolve_labels_unknown_mode_raises():
    with pytest.raises(ValueError, match="abc"):
        resolve_labels(panel_labels=[None], abc="upper-roman")


def test_resolve_labels_after_z_uses_aa_continuation():
    """Past 26 panels, abc='upper' continues 'AA','AB','AC',...
    Important: we don't expect anyone to actually have 27 panels, but
    it's a low-risk fallback."""
    panels = [None] * 28
    out = resolve_labels(panel_labels=panels, abc="upper")
    assert out[0] == "A"
    assert out[25] == "Z"
    assert out[26] == "AA"
    assert out[27] == "AB"


# ---------------------------------------------------------------------------
# Canvas integration — abc kwarg + per-panel label resolution
# ---------------------------------------------------------------------------

def test_canvas_abc_default_upper_for_cell():
    canvas = pp.Canvas("cell-2col")
    canvas.add_row(
        pp.PanelAxes(label=None, size=(60.0, 40.0)),
        pp.PanelAxes(label=None, size=(60.0, 40.0)),
    )
    # PR 2: canvas exposes resolved labels on each Panel.
    assert canvas[0].label == "A"
    assert canvas[1].label == "B"


def test_canvas_abc_default_lower_for_nature():
    canvas = pp.Canvas("nature-2col")
    canvas.add_row(
        pp.PanelAxes(label=None, size=(60.0, 40.0)),
        pp.PanelAxes(label=None, size=(60.0, 40.0)),
    )
    assert canvas[0].label == "a"
    assert canvas[1].label == "b"


def test_canvas_abc_explicit_kwarg_overrides_preset_default():
    canvas = pp.Canvas("nature-2col", abc="upper")  # override Nature's lower default
    canvas.add_row(
        pp.PanelAxes(label=None, size=(60.0, 40.0)),
        pp.PanelAxes(label=None, size=(60.0, 40.0)),
    )
    assert canvas[0].label == "A"
    assert canvas[1].label == "B"


def test_canvas_abc_false_disables_labels():
    canvas = pp.Canvas("cell-2col", abc=False)
    canvas.add_row(
        pp.PanelAxes(label=None, size=(60.0, 40.0)),
        pp.PanelAxes(label=None, size=(60.0, 40.0)),
    )
    # When abc=False AND label=None, the resolved label is None → no
    # render. canvas[0].label stays None to signal "no label".
    assert canvas[0].label is None
    assert canvas[1].label is None


def test_canvas_indexing_by_resolved_label():
    """canvas['A'] should resolve to the panel even if the input had
    label=None — the resolved letter is the lookup key."""
    canvas = pp.Canvas("cell-2col")
    canvas.add_row(
        pp.PanelAxes(label=None, size=(60.0, 40.0)),
        pp.PanelAxes(label=None, size=(60.0, 40.0)),
    )
    a = canvas["A"]
    b = canvas["B"]
    assert a.bbox_mm[0] < b.bbox_mm[0]
```

NOTE: this test introduces a new convention — `canvas[i]` (integer index) returns the i-th panel by insertion order. Several tests rely on it. The `Canvas.__getitem__` from PR 1 only handles str keys and raises KeyError for unknown labels. Task 5 must extend `__getitem__` to handle integer indices.

- [ ] **Step 2: Write the failing tests for label_style**

Create `/home/sagemaker-user/publiplots/tests/composer/test_label_style.py` with this exact content:

```python
"""Tests for canvas-wide label_style + per-panel override."""

import pytest

import publiplots as pp


def test_canvas_label_style_setter_updates_canvas_state():
    canvas = pp.Canvas("cell-2col")
    canvas.label_style(loc="ur", size=11)
    assert canvas._label_style["loc"] == "ur"
    assert canvas._label_style["size"] == 11


def test_canvas_label_style_partial_update_preserves_other_keys():
    """Calling label_style(loc='ur') doesn't reset weight/size."""
    canvas = pp.Canvas("cell-2col")
    initial_weight = canvas._label_style["weight"]
    canvas.label_style(loc="ur")
    assert canvas._label_style["weight"] == initial_weight
    assert canvas._label_style["loc"] == "ur"


def test_canvas_label_style_default_for_cell():
    """Cell preset: loc='ul', size=9 (preset default), weight='bold'."""
    canvas = pp.Canvas("cell-2col")
    assert canvas._label_style["loc"] == "ul"
    assert canvas._label_style["size"] == 9
    assert canvas._label_style["weight"] == "bold"


def test_canvas_label_style_default_for_nature():
    canvas = pp.Canvas("nature-2col")
    assert canvas._label_style["size"] == 8


def test_canvas_label_style_default_for_science():
    canvas = pp.Canvas("science-2col")
    assert canvas._label_style["size"] == 10


def test_canvas_label_style_loc_ultraplot_vocab():
    """The 8 ultraplot loc values: ul, ur, ll, lr, uc, lc, cl, cr."""
    valid_locs = {"ul", "ur", "ll", "lr", "uc", "lc", "cl", "cr"}
    for loc in valid_locs:
        canvas = pp.Canvas("cell-2col")
        canvas.label_style(loc=loc)
        assert canvas._label_style["loc"] == loc


def test_canvas_label_style_invalid_loc_raises():
    canvas = pp.Canvas("cell-2col")
    with pytest.raises(ValueError, match="loc"):
        canvas.label_style(loc="middle")


def test_per_panel_label_style_overrides_canvas():
    """Per-panel label_style merges INTO the canvas-wide style:
    panel keys win, others fall through."""
    canvas = pp.Canvas("cell-2col")  # canvas: loc='ul', size=9
    canvas.add_row(
        pp.PanelAxes(label="A", size=(60.0, 40.0)),
        pp.PanelAxes(label="B", size=(60.0, 40.0),
                     label_style={"loc": "ur", "size": 11}),
    )
    # Resolved per-panel styles should be available on the Panel object.
    a_style = canvas["A"].resolved_label_style
    b_style = canvas["B"].resolved_label_style
    assert a_style["loc"] == "ul"   # canvas default
    assert a_style["size"] == 9
    assert b_style["loc"] == "ur"   # per-panel override
    assert b_style["size"] == 11    # per-panel override
    assert b_style["weight"] == "bold"  # canvas default falls through


def test_per_panel_label_style_partial_override():
    """Per-panel can override just one key; others fall through."""
    canvas = pp.Canvas("cell-2col")  # canvas: weight='bold', size=9
    canvas.add_row(
        pp.PanelAxes(label="A", size=(60.0, 40.0),
                     label_style={"size": 14}),  # only size override
    )
    style = canvas["A"].resolved_label_style
    assert style["size"] == 14
    assert style["weight"] == "bold"  # canvas default falls through


def test_label_renders_in_figure_after_add_row():
    """Smoke test: after add_row, the panel's axes has a Text artist
    placed at the resolved loc."""
    canvas = pp.Canvas("cell-2col")
    canvas.add_row(pp.PanelAxes(label="A", size=(60.0, 40.0)))

    # The label is added as an ax.text artist; find it by its text content.
    ax = canvas["A"].ax
    label_artists = [t for t in ax.texts if t.get_text() == "A"]
    assert len(label_artists) == 1, (
        f"expected exactly 1 'A' text artist, got {len(label_artists)}"
    )


def test_label_false_renders_no_text_artist():
    canvas = pp.Canvas("cell-2col")
    canvas.add_row(pp.PanelAxes(label=False, size=(60.0, 40.0)))
    ax = list(canvas._panels.values())[0].ax
    # No text artist for a False label.
    label_texts = [t.get_text() for t in ax.texts]
    assert all(t == "" or t is None for t in label_texts)


def test_abc_false_renders_no_text_for_none_panels():
    canvas = pp.Canvas("cell-2col", abc=False)
    canvas.add_row(pp.PanelAxes(label=None, size=(60.0, 40.0)))
    ax = list(canvas._panels.values())[0].ax
    label_texts = [t.get_text() for t in ax.texts]
    assert all(t == "" or t is None for t in label_texts)
```

- [ ] **Step 3: Run to verify they fail**

```bash
cd /home/sagemaker-user/publiplots
uv run pytest tests/composer/test_abc_labels.py tests/composer/test_label_style.py -v --no-cov 2>&1 | tail -10
```

Expected: many failures. The `resolve_labels` function doesn't exist; `Canvas.label_style` doesn't exist; `Canvas.__getitem__` only handles str keys.

- [ ] **Step 4: Implement `abc_labels.py`**

Create `/home/sagemaker-user/publiplots/src/publiplots/composer/abc_labels.py` with this exact content:

```python
"""abc panel-label sequencing + style resolution.

PR 2 introduces auto-letter labels (``abc='upper'``, ``'lower'``,
``'A.'``, ``'a.'``, or an explicit list), with per-canvas and per-panel
``label_style`` overrides. The ultraplot ``loc=`` vocabulary is
adopted verbatim: ``'ul'``, ``'ur'``, ``'ll'``, ``'lr'``, ``'uc'``,
``'lc'``, ``'cl'``, ``'cr'``.

Pure-Python except for the matplotlib ``ax.text`` call in
:func:`render_label` — which is gated to "render only" and used by
``canvas.add_row`` after the figure is built.
"""

from string import ascii_lowercase, ascii_uppercase
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union


VALID_LOCS = frozenset({"ul", "ur", "ll", "lr", "uc", "lc", "cl", "cr"})

# Default label_style — keys present on every Canvas; preset overrides
# `size` (and may override `loc` in future).
DEFAULT_LABEL_STYLE: Dict[str, Any] = {
    "weight": "bold",
    "size": 9,                # overridden by preset's label_size_pt
    "family": None,           # falls back to rcParams['font.family']
    "loc": "ul",
    "pad_mm": (0.0, 0.0),
    "border": False,
    "bbox": False,
}


def _seq_letter(idx: int, base: str) -> str:
    """0→A, 1→B, …, 25→Z, 26→AA, 27→AB, … (using `base` alphabet)."""
    if idx < 0:
        raise ValueError(f"sequence index must be non-negative, got {idx}")
    out = ""
    n = idx
    while True:
        out = base[n % 26] + out
        n = n // 26 - 1
        if n < 0:
            return out


def resolve_labels(
    *,
    panel_labels: Sequence[Union[str, None, bool]],
    abc: Union[str, bool, Sequence[str]],
) -> List[Union[str, None, bool]]:
    """Resolve a list of panel-input labels into a list of render-time labels.

    Each input slot is one of:
      - ``None`` — auto-letter slot, resolved per ``abc``.
      - ``False`` — explicitly suppressed; output is ``False``.
      - ``str`` — verbatim; output is the same str. Consumes a slot in
        the auto-letter sequence (so the next ``None`` skips ahead).

    ``abc`` is one of:
      - ``'upper'`` → 'A','B','C',...
      - ``'lower'`` → 'a','b','c',...
      - ``'A.'`` → 'A.','B.','C.',...
      - ``'a.'`` → 'a.','b.','c.',...
      - ``False`` → all auto slots resolve to ``None`` (no label)
      - ``Sequence[str]`` → explicit list, must match the count of
        non-``False`` panels.
    """
    if abc is False:
        # All auto slots become None; explicit strs pass through; False stays.
        return [v if v is not None else None for v in panel_labels]

    if isinstance(abc, str):
        # Detect template form (suffix after letter).
        if abc in ("upper", "lower"):
            base = ascii_uppercase if abc == "upper" else ascii_lowercase
            suffix = ""
        elif len(abc) >= 2 and abc[0].isalpha() and not abc[1:].isalpha():
            base = ascii_uppercase if abc[0].isupper() else ascii_lowercase
            suffix = abc[1:]
        else:
            raise ValueError(
                f"abc must be 'upper', 'lower', a template like 'A.' or 'a.', "
                f"False, or an explicit list; got {abc!r}"
            )
        out: List[Any] = []
        slot = 0
        for v in panel_labels:
            if v is False:
                out.append(False)
                # False does NOT consume a slot.
            elif v is None:
                out.append(_seq_letter(slot, base) + suffix)
                slot += 1
            else:
                # str — pass through, consumes a slot.
                out.append(v)
                slot += 1
        return out

    # abc is Sequence[str]
    try:
        abc_list = list(abc)
    except TypeError:
        raise ValueError(
            f"abc must be 'upper', 'lower', a template, False, or a list; "
            f"got {type(abc).__name__}"
        )
    n_consuming = sum(1 for v in panel_labels if v is not False)
    if len(abc_list) < n_consuming:
        raise ValueError(
            f"abc list has {len(abc_list)} entries but {n_consuming} panels "
            f"need labels (panels with label=False are skipped)"
        )
    out2: List[Any] = []
    slot = 0
    for v in panel_labels:
        if v is False:
            out2.append(False)
        elif v is None:
            out2.append(abc_list[slot])
            slot += 1
        else:
            out2.append(v)
            slot += 1
    return out2


def merge_label_style(
    canvas_style: Mapping[str, Any],
    panel_override: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Merge canvas-wide style with per-panel override; panel wins."""
    merged = dict(canvas_style)
    if panel_override:
        merged.update(panel_override)
    return merged


def render_label(
    ax,
    label_text: str,
    *,
    style: Mapping[str, Any],
) -> None:
    """Place a text artist on ``ax`` at the resolved ``loc`` with style.

    Coordinates are axes-fraction; the loc-codes map to the 8 anchors
    defined by ultraplot: ul, ur, ll, lr (corners) + uc, lc (top/bottom
    center) + cl, cr (left/right middle).

    The pad_mm offset shifts the label INWARD from the chosen anchor,
    converted from mm to axes-fraction via the figure's dpi at draw
    time; for PR 2 we use a fixed 1mm = ~0.01 axes-fraction approximation
    (refine in PR 3 if needed).
    """
    loc = style["loc"]
    if loc not in VALID_LOCS:
        raise ValueError(f"loc must be one of {sorted(VALID_LOCS)}, got {loc!r}")

    # Map loc codes → (x_frac, y_frac, ha, va)
    loc_table = {
        "ul": (0.0, 1.0, "left",   "top"),
        "ur": (1.0, 1.0, "right",  "top"),
        "ll": (0.0, 0.0, "left",   "bottom"),
        "lr": (1.0, 0.0, "right",  "bottom"),
        "uc": (0.5, 1.0, "center", "top"),
        "lc": (0.5, 0.0, "center", "bottom"),
        "cl": (0.0, 0.5, "left",   "center"),
        "cr": (1.0, 0.5, "right",  "center"),
    }
    x, y, ha, va = loc_table[loc]

    # Apply pad_mm as inward shift in axes fraction (mm → frac via dpi
    # would be exact; this approximation is fine for PR 2's 0.0 default).
    pad_x_mm, pad_y_mm = style.get("pad_mm", (0.0, 0.0))
    # Convert mm to axes-fraction via the panel's pixel size at draw time.
    # For now: if the loc is on the left, pad_x_mm shifts right (positive);
    # on the right, shifts left (negative); etc.
    bbox = ax.get_window_extent()
    fig = ax.figure
    dpi = fig.dpi if fig.dpi > 0 else 100.0
    if bbox.width > 0:
        x_frac_per_mm = (dpi / 25.4) / bbox.width
    else:
        x_frac_per_mm = 0.0
    if bbox.height > 0:
        y_frac_per_mm = (dpi / 25.4) / bbox.height
    else:
        y_frac_per_mm = 0.0
    if ha == "left":
        x += pad_x_mm * x_frac_per_mm
    elif ha == "right":
        x -= pad_x_mm * x_frac_per_mm
    if va == "top":
        y -= pad_y_mm * y_frac_per_mm
    elif va == "bottom":
        y += pad_y_mm * y_frac_per_mm

    text_kwargs = dict(
        x=x, y=y, s=label_text,
        ha=ha, va=va,
        transform=ax.transAxes,
        fontweight=style["weight"],
        fontsize=style["size"],
    )
    if style["family"]:
        text_kwargs["fontfamily"] = style["family"]

    bbox_kwargs = None
    if style["bbox"]:
        bbox_kwargs = {"boxstyle": "square,pad=0.2", "fc": "white", "ec": "none"}
    if style["border"]:
        # Border = white outline around the text glyphs.
        import matplotlib.patheffects as path_effects
        text = ax.text(**text_kwargs)
        text.set_path_effects([
            path_effects.Stroke(linewidth=2, foreground="white"),
            path_effects.Normal(),
        ])
        if bbox_kwargs:
            text.set_bbox(bbox_kwargs)
        return

    text = ax.text(**text_kwargs)
    if bbox_kwargs:
        text.set_bbox(bbox_kwargs)
```

- [ ] **Step 5: Wire abc + label_style into `Canvas`**

Edit `/home/sagemaker-user/publiplots/src/publiplots/composer/canvas.py`. Make these changes:

(a) Update `__init__` to accept `abc=` kwarg + initialize `_label_style`:

Find the existing `__init__`:

```python
    def __init__(self, preset: str, *, width: Optional[float] = None) -> None:
        self._preset_name = preset
        spec = resolve_preset(preset, width=width)
        self._width_mm: float = spec["width_mm"]
        self._max_height_mm: Optional[float] = spec["max_height_mm"]

        # Lazy-initialized on first add_row():
        self._figure = None
        self._panels: Dict[str, Panel] = {}
        self._row_added: bool = False
```

Replace with:

```python
    def __init__(
        self,
        preset: str,
        *,
        width: Optional[float] = None,
        abc: Union[str, bool, Sequence[str], None] = None,
    ) -> None:
        from publiplots.composer.abc_labels import DEFAULT_LABEL_STYLE

        self._preset_name = preset
        spec = resolve_preset(preset, width=width)
        self._width_mm: float = spec["width_mm"]
        self._max_height_mm: Optional[float] = spec["max_height_mm"]

        # abc resolution: if user passed None, fall back to preset default.
        self._abc = abc if abc is not None else spec["abc_default"]

        # Initialize label_style from the default + preset's label_size_pt.
        self._label_style: Dict[str, Any] = dict(DEFAULT_LABEL_STYLE)
        self._label_style["size"] = spec["label_size_pt"]

        # Lazy-initialized on first add_row():
        self._figure = None
        self._panels: Dict[str, Panel] = {}
        self._panels_ordered: List[Panel] = []  # insertion-order for int indexing
        self._row_added: bool = False
```

Add `Union, Sequence, Any, List` to the typing imports at the top of `canvas.py`:

```python
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union
```

(b) Add a `label_style` method. Insert after the `__getitem__` method, before `add_row`:

```python
    def label_style(self, **kwargs: Any) -> None:
        """Update the canvas-wide label style.

        Accepts any subset of: ``weight``, ``size``, ``family``, ``loc``,
        ``pad_mm``, ``border``, ``bbox``. Missing keys are unchanged.
        ``loc`` must be one of the 8 ultraplot locs (``'ul'``, ``'ur'``,
        ``'ll'``, ``'lr'``, ``'uc'``, ``'lc'``, ``'cl'``, ``'cr'``).
        """
        from publiplots.composer.abc_labels import VALID_LOCS

        if "loc" in kwargs and kwargs["loc"] not in VALID_LOCS:
            raise ValueError(
                f"loc must be one of {sorted(VALID_LOCS)}, got {kwargs['loc']!r}"
            )
        self._label_style.update(kwargs)
```

(c) Extend `__getitem__` to accept integer indices:

Replace the existing `__getitem__`:

```python
    def __getitem__(self, key) -> Panel:
        if isinstance(key, int):
            if not self._panels_ordered:
                raise KeyError(
                    f"no panels yet; call add_row() first"
                )
            try:
                return self._panels_ordered[key]
            except IndexError:
                raise KeyError(
                    f"panel index {key} out of range; "
                    f"only {len(self._panels_ordered)} panel(s) exist"
                )
        if key not in self._panels:
            raise KeyError(
                f"no panel with label {key!r}; "
                f"known labels: {sorted(self._panels)}"
            )
        return self._panels[key]
```

(d) In `add_row`, after panels are constructed, resolve labels and render them. Find the loop:

```python
        for col_idx, panel_input in enumerate(panels):
            x0_frac, y0_frac, w_frac, h_frac = layout.axes_position(0, col_idx)
            ax = fig.add_axes((x0_frac, y0_frac, w_frac, h_frac))
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
                size_mm=(col_widths[col_idx], panel_input.size[1]),
                bbox_mm=bbox_mm,
            )
```

Replace with:

```python
        # Resolve labels per the canvas's abc mode.
        from publiplots.composer.abc_labels import (
            resolve_labels, merge_label_style, render_label,
        )
        raw_panel_labels = [p.label for p in panels]
        resolved_labels = resolve_labels(
            panel_labels=raw_panel_labels,
            abc=self._abc,
        )

        for col_idx, panel_input in enumerate(panels):
            x0_frac, y0_frac, w_frac, h_frac = layout.axes_position(0, col_idx)
            ax = fig.add_axes((x0_frac, y0_frac, w_frac, h_frac))
            bbox_mm = (
                x0_frac * W_mm,
                y0_frac * H_mm,
                w_frac * W_mm,
                h_frac * H_mm,
            )

            resolved_label = resolved_labels[col_idx]
            resolved_style = merge_label_style(
                self._label_style, panel_input.label_style
            )

            panel = Panel(
                label=resolved_label,
                kind="axes",
                ax=ax,
                size_mm=(col_widths[col_idx], panel_input.size[1]),
                bbox_mm=bbox_mm,
                resolved_label_style=resolved_style,
            )

            # Render the label if it's a non-empty string.
            if isinstance(resolved_label, str) and resolved_label:
                render_label(ax, resolved_label, style=resolved_style)

            # Register by resolved label (so canvas['A'] works for an
            # auto-letter panel that started as label=None) AND by
            # insertion order. False/None labels skip the dict registry.
            if isinstance(resolved_label, str) and resolved_label:
                self._panels[resolved_label] = panel
            self._panels_ordered.append(panel)
```

(e) The `Panel` dataclass needs a new field. Edit `/home/sagemaker-user/publiplots/src/publiplots/composer/panels.py`. Add `resolved_label_style` field to the `Panel` dataclass:

```python
@dataclass(frozen=True)
class Panel:
    """Result type returned by ``canvas[label]``.
    ...  (unchanged docstring) ...
    """

    label: Any                    # str | None | False
    kind: PanelKind
    ax: Optional[Any]
    size_mm: Tuple[float, float]
    bbox_mm: Tuple[float, float, float, float]
    resolved_label_style: Optional[Mapping[str, Any]] = None
```

- [ ] **Step 6: Run the abc tests**

```bash
cd /home/sagemaker-user/publiplots
uv run pytest tests/composer/test_abc_labels.py tests/composer/test_label_style.py -v --no-cov 2>&1 | tail -15
```

Expected: ALL pass.

If `test_label_renders_in_figure_after_add_row` fails because the text artist isn't on the axes, debug `render_label` — most likely cause: `bbox.width == 0` before draw (axes haven't been laid out yet). The fallback `x_frac_per_mm = 0.0` should handle it; if not, use `transform=ax.transAxes` and skip the pad math when bbox is degenerate.

- [ ] **Step 7: Run full composer regression**

```bash
cd /home/sagemaker-user/publiplots
uv run pytest tests/composer/ -q --no-cov 2>&1 | tail -3
```

Expected: 109 (Tasks 1-4) + ~30 (test_abc_labels + test_label_style) = ~139 passed.

Two PR 1 tests may need amendment if they hardcoded `canvas[label].label == 'A'` against `label=None` — re-run and capture; the failures should be small + obvious if any.

- [ ] **Step 8: Commit**

```bash
cd /home/sagemaker-user/publiplots
git add src/publiplots/composer/abc_labels.py \
        src/publiplots/composer/canvas.py \
        src/publiplots/composer/panels.py \
        tests/composer/test_abc_labels.py \
        tests/composer/test_label_style.py
git commit -m "feat(composer): abc panel labels — sequencer + label_style + render"
```

---

## Task 6: Examples

**Files:**
- Create: `examples/composer/cell_2col_simple.py`
- Create: `examples/composer/nature_2col_with_abc.py`

Two runnable demonstrations of PR 2 features. Each saves a PNG to `docs/images/composer/`.

- [ ] **Step 1: Write `cell_2col_simple.py`**

Create `/home/sagemaker-user/publiplots/examples/composer/cell_2col_simple.py` with this exact content:

```python
"""Cell 2-column figure with two axes panels — flex sizing, auto abc.

Demonstrates:
- Cell-2col preset (174mm wide, abc='upper', 9pt bold labels)
- Flex sizing: panel B fills the leftover width so the figure is
  exactly 174mm wide
- Auto abc labels (panels labeled 'A' and 'B' from the canvas's abc='upper')
- Saves to docs/images/composer/cell-2col-simple.png

Run from the repo root:

    uv run python examples/composer/cell_2col_simple.py
"""

from pathlib import Path

import numpy as np
import publiplots as pp


def main() -> None:
    rng = np.random.default_rng(0)
    df_a = {"x": rng.normal(0, 1, 200), "y": rng.normal(0, 1, 200)}
    df_b = {"x": np.arange(50), "y": np.cumsum(rng.normal(0, 1, 50))}

    canvas = pp.Canvas("cell-2col")
    canvas.add_row(
        pp.PanelAxes(label=None, size=(70, 50)),       # A — pinned 70mm
        pp.PanelAxes(label=None, size=("flex", 50)),   # B — fills leftover
    )

    canvas["A"].ax.scatter(df_a["x"], df_a["y"], s=4, alpha=0.6)
    canvas["A"].ax.set_xlabel("measurement A")
    canvas["A"].ax.set_ylabel("measurement B")

    canvas["B"].ax.plot(df_b["x"], df_b["y"], lw=1.0)
    canvas["B"].ax.set_xlabel("time (frames)")
    canvas["B"].ax.set_ylabel("cumulative drift")

    out_dir = Path(__file__).resolve().parents[2] / "docs" / "images" / "composer"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "cell-2col-simple.png"
    canvas.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write `nature_2col_with_abc.py`**

Create `/home/sagemaker-user/publiplots/examples/composer/nature_2col_with_abc.py` with this exact content:

```python
"""Nature 2-column figure with three panels — abc lower, mixed labels.

Demonstrates:
- nature-2col preset (183mm wide, abc='lower', 8pt bold labels)
- Three flex panels splitting leftover width equally
- Mix of auto-letter (None → 'a','c') and verbatim ('b.i') labels
- Per-panel label_style override on one panel
- Saves to docs/images/composer/nature-2col-abc.png

Run from the repo root:

    uv run python examples/composer/nature_2col_with_abc.py
"""

from pathlib import Path

import numpy as np
import publiplots as pp


def main() -> None:
    rng = np.random.default_rng(1)
    canvas = pp.Canvas("nature-2col")  # abc='lower' by default

    canvas.add_row(
        pp.PanelAxes(label=None,    size=("flex", 40)),                # → 'a'
        pp.PanelAxes(label="b.i",   size=("flex", 40),
                     label_style={"size": 10}),                        # custom label + size
        pp.PanelAxes(label=None,    size=("flex", 40)),                # → 'c' (b consumed)
    )

    for i, key in enumerate(["a", "b.i", "c"]):
        ax = canvas[key].ax
        x = rng.normal(0, 1, 100)
        y = (i + 1) * 0.3 * x + rng.normal(0, 0.5, 100)
        ax.scatter(x, y, s=4, alpha=0.6)
        ax.set_xlabel("x")
        ax.set_ylabel(f"y{i+1}")

    out_dir = Path(__file__).resolve().parents[2] / "docs" / "images" / "composer"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "nature-2col-abc.png"
    canvas.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run both examples to confirm they execute**

```bash
cd /home/sagemaker-user/publiplots
uv run python examples/composer/cell_2col_simple.py
uv run python examples/composer/nature_2col_with_abc.py
```

Expected: each prints `wrote <path>` and the PNGs exist in `docs/images/composer/`.

If either raises, capture the error verbatim and STOP — most likely a small API mismatch with what the prior tasks landed.

- [ ] **Step 4: Verify the produced PNGs are non-trivial**

```bash
cd /home/sagemaker-user/publiplots
ls -la docs/images/composer/
```

Expected: two PNG files, each non-empty (> 5 KB).

- [ ] **Step 5: Commit**

```bash
cd /home/sagemaker-user/publiplots
git add examples/composer/cell_2col_simple.py \
        examples/composer/nature_2col_with_abc.py \
        docs/images/composer/cell-2col-simple.png \
        docs/images/composer/nature-2col-abc.png
git commit -m "docs(composer): cell-2col + nature-2col-abc gallery examples"
```

---

## Task 7: Full suite + CHANGELOG

**Files:** none directly; `CHANGELOG.md` modify.

- [ ] **Step 1: Run the full publiplots test suite**

```bash
cd /home/sagemaker-user/publiplots
uv run pytest tests/ -q --no-cov --tb=short 2>&1 | tail -10
```

Expected: ~1082 + ~76 new composer tests − pre-existing residplot fail = ~1157 passed, 1 failed (residplot).

If any other test fails, STOP and diagnose. The most likely cause is that PR 2's changes to `__getitem__` (now accepts ints) or to `Panel` (added `resolved_label_style`) inadvertently broke a PR 1 test that hardcoded the old shape.

- [ ] **Step 2: Add CHANGELOG entry**

Edit `/home/sagemaker-user/publiplots/CHANGELOG.md`. Find the existing `## [Unreleased] / ### Added` block (PR 1's bullet should already be there). Append the PR 2 bullet to that same block:

```markdown
- `pp.Canvas('cell-2col' | 'nature-2col' | 'science-2col' | …)` —
  Composer journal presets (Cell, Nature, Nature Methods, Science) with
  verified mm widths, max-height bounds, and per-preset label-size
  defaults. Twelve presets total; ``Canvas('custom', width=N)`` remains
  the escape hatch.
- `pp.PanelAxes(label, size=('flex', h_mm))` — flex panel sizing. The
  width grows to fill the row's leftover canvas budget; multiple flex
  panels split equally. When ≥1 flex panel exists, the figure width
  exactly matches `Canvas(width=N)`.
- `ComposerOverflowError` now carries a `scale_to_fit` attribute and
  appends a "multiply panel widths by 0.870 to fit" advisor to its
  message string when the row overflows.
- `pp.PanelAxes(label=None | False | str)` — three label modes:
  ``None`` reserves an auto-letter slot, ``False`` suppresses the
  label and is skipped from the sequence, ``str`` is verbatim. Plus
  per-panel ``label_style={...}`` override.
- `pp.Canvas(..., abc='upper'|'lower'|'A.'|False|['i','ii',...])` —
  auto-letter panel labels with per-canvas `canvas.label_style(...)`
  setter using the ultraplot ``loc`` vocabulary
  (``'ul'``, ``'ur'``, ``'ll'``, ``'lr'``, ``'uc'``, ``'lc'``, ``'cl'``,
  ``'cr'``) and optional ``border`` outline + ``bbox`` background.
- `canvas[i]` (integer index) returns the i-th panel by insertion order;
  `canvas['<label>']` (str) is unchanged.
- Two new gallery examples under ``examples/composer/``.
```

- [ ] **Step 3: Commit the CHANGELOG**

```bash
cd /home/sagemaker-user/publiplots
git add CHANGELOG.md
git commit -m "chore(changelog): composer PR 2 — presets, flex, abc"
```

---

## Task 8: Open the PR

**Files:** none (gh CLI only)

- [ ] **Step 1: Push the branch**

```bash
cd /home/sagemaker-user/publiplots
git push -u origin feat/composer-presets-flex-abc-pr2
```

- [ ] **Step 2: Open the PR**

```bash
cd /home/sagemaker-user/publiplots
gh pr create \
  --title "feat(composer): journal presets + flex sizing + abc labels" \
  --body "$(cat <<'EOF'
## Summary

PR 2 of the Composer rollout (`docs/superpowers/specs/2026-05-14-composer-design.md`). Promotes the Composer from "developer playground" to "actual paper-figure tool" by adding the four journal preset families, flex panel sizing, an overflow advisor with a suggested per-row scaling factor, and ultraplot-style abc panel labels.

## What's in this PR

- **12 journal presets** in `presets.py` — Cell, Nature, Nature Methods, Science (×3 widths each) plus `'custom'`. Dimensions verified against authoritative sources on 2026-05-15; three derived values are flagged inline with comments. Each preset carries its own `default_width_mm`, `max_height_mm`, `abc_default`, and `label_size_pt`.
- **Flex sizing** — `pp.PanelAxes(label, size=('flex', h_mm))`. Multiple flex panels split the leftover width equally. When ≥1 flex panel exists, `figure_size_mm[0] == canvas.width_mm` exactly.
- **Overflow advisor** — `ComposerOverflowError` now carries `scale_to_fit` and appends a "multiply panel widths by X to fit" hint to its message.
- **abc panel labels** — `pp.Canvas(abc='upper'|'lower'|'A.'|False|list)` + per-canvas `canvas.label_style(...)` + per-panel `label_style={...}`. Three label modes: `None` (auto-letter), `False` (no label, skips sequence), `str` (verbatim, consumes a slot).
- **`canvas[i]`** (integer index) returns the i-th panel by insertion order. `canvas['<label>']` (str) is unchanged.
- **2 gallery examples** under `examples/composer/`.

## Spike findings status

PR 1 addressed **Finding 4** (decoration clipping). PR 2 doesn't touch any other findings; Findings 1, 2, 3, 5, 6, 7 remain queued for PRs 5/6 per `spikes/composer/composer-spike.md`.

## Out of scope

Multi-row layouts, `canvas.align`, PanelGrid/PanelText (PR 3); `pp.legend` rows/cols/span (PR 4); mm-precision regression infra (PR 4.5); vector PDF/SVG (PR 5/6); inspect + composer-guide skill (PR 7).

## Test plan

- [x] All ~76 new composer tests pass (`uv run pytest tests/composer/`).
- [x] Full suite green (1 pre-existing residplot failure unchanged).
- [x] Bit-identical `pp.subplots` ↔ `pp.Canvas` equivalence preserved.
- [x] Spike Finding 4 regression tests still pass (no clipping at canvas edge).
- [x] Verified journal dimensions (Cell, Nature, Nature Methods, Science) committed with source URLs in `presets.py` comments.
- [x] Two gallery examples render to PNG without errors.

## Verification notes

Three values in `presets.py` are DERIVED rather than direct journal quotes:

- Cell `max_height_mm = 225` — page-fit derivation (Cell quotes "fit on one page", not an explicit max)
- Science `max_height_mm = 240` — trim derivation (Science quotes no figure-height max)
- Nature Methods 1-col / 1.5-col widths — inherited from Nature family (the AIP page only quotes the 180 mm 2-col cap)

Each is marked `# DERIVED:` inline with a comment.

## Follow-ups

PR 3 starts from this branch's tip. Multi-row + `canvas.align` + PanelGrid/PanelText.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 3: Verify the PR is open**

```bash
gh pr view --json number,url,state,title
```

Expected: state=OPEN, title matches.

---

## Acceptance criteria for PR 2

The PR is ready for human merge when ALL of:

1. All 8 tasks complete.
2. Full test suite green (existing + ~76 new composer tests).
3. `pp.Canvas('cell-2col')`, `pp.Canvas('nature-2col')`, `pp.Canvas('science-2col')`, `pp.Canvas('nature-methods-2col')` all work without explicit `width=`.
4. Flex sizing produces `figure_size_mm[0] == canvas.width_mm` exactly when ≥1 flex panel exists.
5. `pp.PanelAxes(label=None)` + `Canvas(abc='upper')` produces resolved labels `'A','B','C',...`.
6. `ComposerOverflowError` includes a suggested scale factor in its message.
7. The 8 ultraplot loc codes (`ul/ur/ll/lr/uc/lc/cl/cr`) all work in `canvas.label_style(loc=...)`.
8. CHANGELOG entry added under `[Unreleased]` extending PR 1's `### Added`.
9. Two gallery examples render to PNG without errors.
10. Out-of-scope items (multi-row, alignment, PanelGrid/Text, vector save, inspect, skill) are NOT in the diff.

If any of #1–#10 fail, the PR is not ready.
