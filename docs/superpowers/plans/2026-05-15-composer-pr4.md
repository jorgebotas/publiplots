# Composer PR 4 Implementation Plan — `pp.legend` rows / cols / span / ax kwargs

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Strictly-additive upgrade to `pp.legend(...)`: add `rows=`, `cols=`, `span=`, `ax=` kwargs for grid-scope figure legends. Independent of the Composer — works on any figure carrying `_publiplots_axes` (i.e., `pp.subplots` AND `pp.Canvas`). After PR 4, callers can write `pp.legend(rows=0, side='top')` for a row-0 band, `pp.legend(rows=(1,3), cols=2)` for a sub-rect band, `pp.legend(ax=[ax1, ax2, ax3])` for an explicit list with handle dedupe, and `pp.legend(span='fig')` for a full-figure band.

**Architecture:** `pp.legend()` already accepts a list-of-axes scope via the positional `axes=` arg, which `MultiAxesLegendGroup` consumes through `_ScopeAnchor`. PR 4 adds **a single resolver step** in front of the existing factory: `_resolve_grid_scope(fig, *, rows, cols, span, ax) → Optional[List[Axes]]`. The resolver translates the new kwargs into a list of axes; the factory then dispatches into the existing list-of-axes path (no changes to `MultiAxesLegendGroup`/`_ScopeAnchor` themselves). This keeps the diff small and the back-compat surface untouched.

**Internal-attribute caveat:** the existing `MultiAxesLegendGroup` exposes its scope as `self._scope_axes` (NOT `self._axes`). Tests that need to assert on the resolved scope must use `_scope_axes`. We accept this as deliberate brittleness — the alternative (a public `scope_axes` property) is out of scope for PR 4 and risks bikeshedding. If/when the legend module gets a public scope accessor, the tests should migrate.

**Canvas integration deferral:** `pp.Canvas` does NOT currently attach `fig._publiplots_axes` (it builds axes via `fig.add_axes(...)` in `composer/canvas.py:_finalize_if_needed` without populating the matrix). Wiring `_publiplots_axes` for Canvas would require deciding the matrix shape for ragged multi-row layouts (PanelGrid sub-grids are a single panel slot but produce r×c sub-axes; multi-row Canvas with heterogeneous columns is non-rectangular). That decision is out of scope for PR 4. **PR 4's resolver therefore raises a clear "use pp.subplots" error on Canvas figures**; full Canvas integration moves to PR 7 alongside `canvas.inspect()` and the composer-guide skill. Acceptance criterion #11 covers this explicitly.

**Tech Stack:** Python ≥ 3.10, matplotlib, pytest. No new dependencies. Editable install at `/home/sagemaker-user/publiplots/`.

**Spec reference:** `docs/superpowers/specs/2026-05-14-composer-design.md` §"`pp.legend` upgrade" (lines 248–271) + §"PR 4" contract (line 635–638). Spec is the contract; this plan is the implementation breakdown.

---

## What's IN scope for PR 4

- **New kwargs on `pp.legend(...)`** (all keyword-only, all default `None`):
  - `rows: int | tuple[int, int] | None` — single row index or inclusive `(start, end)` range. Resolves over `fig._publiplots_axes`.
  - `cols: int | tuple[int, int] | None` — single col index or inclusive `(start, end)` range. Resolves over `fig._publiplots_axes`.
  - `span: Literal["row", "col", "fig"] | None` — sugar:
    - `'fig'` → full figure (equivalent to `axes=None, anchor=None`; useful as a self-documenting alternative)
    - `'row'` → full row containing the positional anchor (requires `axes=ax` positional, single Axes)
    - `'col'` → full column containing the positional anchor (same rule)
  - `ax: Sequence[Axes] | None` — explicit list of axes; handles deduped by label across the list.
- **Resolver helper `_resolve_grid_scope`** — pure-Python translator from `(fig, rows, cols, span, ax)` → `Optional[List[Axes]]` returning the resolved scope (or `None` to mean "no grid scoping; fall through to figure-level").
- **Validation**:
  - Raises `ValueError` if `rows`/`cols` are given but `fig._publiplots_axes` is missing or empty (the figure was not built by `pp.subplots`). On `pp.Canvas` figures, the resolver raises with the same message — Canvas integration is deferred to PR 7 (see "What's OUT of scope" below).
  - Raises `ValueError` if `rows`/`cols` indices are out of range for the matrix, with a message naming the actual matrix shape.
  - Raises `ValueError` if a `(start, end)` range tuple has `start > end`, with a message suggesting the swapped form.
  - Raises `ValueError` if a negative index is passed, with a message naming the equivalent positive index. (Python wrap-around is intentionally NOT supported for clarity.)
  - Raises `ValueError` if `span` is invalid, or if `span='row'`/`'col'` is given without a positional `axes=ax` anchor.
  - Raises `ValueError` if `ax=[]` is empty.
  - Raises `ValueError` if MORE THAN ONE of (`rows`, `cols`, `span`, `ax`) is incompatible with the others — specifically, `ax=` is mutually exclusive with `rows`/`cols`/`span` (different addressing modes); `span=` is mutually exclusive with `rows=`/`cols=`. Error messages NAME the colliding kwargs.
  - Raises `ValueError` if `figure=` was passed but the resolved axes belong to a different figure (drift guard).
- **`ax=[...]` dedupe-by-label** — `pp.legend(ax=[a1, a2, a3])` collects from those axes; if the same label is stashed on multiple axes with **identical** handle props, the duplicate is dropped silently. If the same label appears with **mismatched** handle props, the existing `MultiAxesLegendGroup._merge_entries` path emits its `UserWarning` (PR 4 inherits this; no new behavior).
- **CHANGELOG entry** under `[Unreleased]`.
- **`legend-placement` skill update** — add a new "Grid scoping" section documenting the four new kwargs with copyable snippets.
- **One small gallery snippet in `examples/`** — a 2×3 `pp.subplots` figure with `pp.legend(rows=0, side='top')` showing a row-scoped band, saved to PNG. Lives in `examples/legend/grid_scope_demo.py`.

## What's OUT of scope for PR 4

- **`pp.Canvas` grid-scope integration** — Canvas does NOT currently attach `fig._publiplots_axes`, and deciding the matrix shape for ragged multi-row layouts (heterogeneous columns per row, PanelGrid sub-grids) requires its own design pass. PR 4's resolver raises `ValueError` on Canvas figures with the same "use `pp.subplots`" message as raw `plt.subplots`. Full Canvas integration moves to PR 7 alongside `canvas.inspect()` and the composer-guide skill — at that point we'll decide whether to attach a numpy `dtype=object` array (with `None` padding for ragged rows) or a list-of-rows + add a Canvas-specific resolver branch.
- Multi-row group **band** layout (e.g., a top-row legend AND a bottom-row legend on the same figure): already supported via two separate `pp.legend()` calls per existing scope plumbing — no new behavior, but exercised in tests.
- `embed_figure` legend forwarding — that's PR 6.
- mm-precision regression infra — PR 4.5.
- Vector compositing — PR 5/6.
- `canvas.inspect()` + composer-guide skill — PR 7.
- Promoting `ComposerError`/`ComposerOverflowError`/`ComposerAlignmentError` to top-level `pp.*` — deferred to PR 7 (lands with the skill).
- A public `MultiAxesLegendGroup.scope_axes` accessor — tests use `_scope_axes` directly; refactor for next legend-module pass.

This list is the contract for the spec-compliance reviewer. Anything in this list that the implementer ships counts as scope creep and should be flagged.

---

## File Structure

```
src/publiplots/utils/
├── legend_group.py                   # MODIFY — add _resolve_grid_scope + 4 new kwargs on legend()

tests/
├── test_legend_grid_scope.py         # NEW — resolver unit tests + factory integration tests

skills/
└── legend-placement/SKILL.md         # MODIFY — add "Grid scoping" section

examples/legend/
└── grid_scope_demo.py                # NEW — small demo, renders to PNG

CHANGELOG.md                          # MODIFY — `[Unreleased]` section
```

4 modified or new files (2 modified + 2 new). Estimated ~150 LOC code + ~350 LOC tests + 1 example. Comfortably under the spec's "~200 LOC + tests" budget.

**File-size budget (warn if exceeded):**
- `legend_group.py`: ≤ 1400 LOC (was 1171; +~150 for resolver + factory wiring)
- `test_legend_grid_scope.py`: ≤ 400 LOC

Going over is allowed if necessary, but the implementer MUST flag it as `DONE_WITH_CONCERNS`.

---

## Branch + worktree setup (Task 0)

**Files:** none (git only)

- [ ] **Step 1: Verify main is synced**

```bash
cd /home/sagemaker-user/publiplots
git fetch origin
git checkout main
git status
git log --oneline -3
```

Expected: HEAD at `67d3eb2` (= PR 165's squash-merge commit). Working tree clean.

- [ ] **Step 2: Create branch from main tip**

```bash
cd /home/sagemaker-user/publiplots
git checkout -b feat/composer-legend-scopes-pr4
```

- [ ] **Step 3: Sanity check baseline**

```bash
cd /home/sagemaker-user/publiplots
uv run pytest tests/composer tests/test_legend_group_axes_scope.py tests/test_legend_unified.py -x --no-cov 2>&1 | tail -5
```

Expected: 211 composer tests pass + the relevant legend tests pass. The 1 residplot pre-existing failure is in a different file; ignore.

---

## Task 1: `_resolve_grid_scope` resolver helper

**Files:**
- Modify: `src/publiplots/utils/legend_group.py` — add `_resolve_grid_scope` near the top (after `_handle_repr`, before `_ScopeAnchor`)
- Create: `tests/test_legend_grid_scope.py` — unit tests for the pure resolver

**Why first:** the resolver is a pure function over `fig._publiplots_axes` + the four kwargs. Lands the validation logic + happy path in isolation, before any wiring touches the `legend()` factory. TDD gives the implementer a specific contract to satisfy.

- [ ] **Step 1: Write the failing tests**

Create `/home/sagemaker-user/publiplots/tests/test_legend_grid_scope.py` with this EXACT content:

```python
"""Tests for `_resolve_grid_scope` and the new pp.legend grid-scope kwargs.

PR 4 of the Composer rollout: strictly-additive upgrade to `pp.legend`
adding `rows=`, `cols=`, `span=`, `ax=` for grid-scope figure legends.
"""
from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import publiplots as pp
from publiplots.utils.legend_group import _resolve_grid_scope, MultiAxesLegendGroup


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# `_resolve_grid_scope` — pure-resolver unit tests
# ---------------------------------------------------------------------------


def test_resolve_grid_scope_all_none_returns_none():
    """All-None kwargs → resolver returns None (no grid scoping; caller falls through)."""
    fig, _ = pp.subplots(nrows=2, ncols=3)
    result = _resolve_grid_scope(fig, rows=None, cols=None, span=None, ax=None)
    assert result is None


def test_resolve_grid_scope_rows_int_single_row():
    """rows=1 → all axes in row 1 of the publiplots matrix."""
    fig, axes = pp.subplots(nrows=2, ncols=3)
    result = _resolve_grid_scope(fig, rows=1, cols=None, span=None, ax=None)
    assert result is not None
    assert len(result) == 3
    expected = list(axes[1])
    assert [id(a) for a in result] == [id(a) for a in expected]


def test_resolve_grid_scope_rows_tuple_inclusive():
    """rows=(0, 1) → rows 0 AND 1 (inclusive)."""
    fig, axes = pp.subplots(nrows=3, ncols=2)
    result = _resolve_grid_scope(fig, rows=(0, 1), cols=None, span=None, ax=None)
    assert len(result) == 4  # 2 rows × 2 cols


def test_resolve_grid_scope_cols_int_single_col():
    """cols=2 → all axes in col 2."""
    fig, axes = pp.subplots(nrows=2, ncols=3)
    result = _resolve_grid_scope(fig, rows=None, cols=2, span=None, ax=None)
    assert len(result) == 2
    assert [id(a) for a in result] == [id(axes[0, 2]), id(axes[1, 2])]


def test_resolve_grid_scope_rows_and_cols_intersection():
    """rows=1, cols=2 → exactly one cell (axes[1, 2])."""
    fig, axes = pp.subplots(nrows=2, ncols=3)
    result = _resolve_grid_scope(fig, rows=1, cols=2, span=None, ax=None)
    assert len(result) == 1
    assert result[0] is axes[1, 2]


def test_resolve_grid_scope_rows_out_of_range_raises():
    """rows index outside matrix shape → ValueError naming the matrix shape."""
    fig, _ = pp.subplots(nrows=2, ncols=3)
    with pytest.raises(ValueError, match=r"rows.*out of range.*\(2, 3\)"):
        _resolve_grid_scope(fig, rows=5, cols=None, span=None, ax=None)


def test_resolve_grid_scope_cols_negative_raises_with_hint():
    """Negative col index → ValueError naming the equivalent positive index."""
    fig, _ = pp.subplots(nrows=2, ncols=3)
    with pytest.raises(ValueError, match=r"negative indices.*Use `cols=2`"):
        _resolve_grid_scope(fig, rows=None, cols=-1, span=None, ax=None)


def test_resolve_grid_scope_rows_inverted_range_raises_with_hint():
    """rows=(2, 0) → ValueError suggesting the swapped form."""
    fig, _ = pp.subplots(nrows=3, ncols=2)
    with pytest.raises(ValueError, match=r"start > end.*Use `rows=\(0, 2\)`"):
        _resolve_grid_scope(fig, rows=(2, 0), cols=None, span=None, ax=None)


def test_resolve_grid_scope_no_publiplots_axes_raises():
    """rows/cols on a non-publiplots figure → ValueError mentioning pp.subplots."""
    fig, ax = plt.subplots(2, 3)
    # plt.subplots does NOT set _publiplots_axes
    with pytest.raises(ValueError, match=r"pp\.subplots"):
        _resolve_grid_scope(fig, rows=0, cols=None, span=None, ax=None)


def test_resolve_grid_scope_canvas_figure_raises_with_pr7_hint():
    """rows/cols on a Canvas figure → ValueError pointing at PR 7 / pp.subplots.

    Canvas integration is deferred to PR 7. Until then the resolver raises
    the same way it does for a raw plt.subplots() figure (no
    `_publiplots_axes` matrix attached).
    """
    canvas = pp.Canvas("cell-2col")
    canvas.add_row(pp.PanelAxes(label="A", size=(70, 40)),
                   pp.PanelAxes(label="B", size=(70, 40)))
    fig = canvas.figure  # triggers lazy finalization
    with pytest.raises(ValueError, match=r"pp\.subplots"):
        _resolve_grid_scope(fig, rows=0, cols=None, span=None, ax=None)


def test_resolve_grid_scope_ax_list_returns_list():
    """ax=[ax1, ax2] → exactly that list (no fig._publiplots_axes lookup)."""
    fig, axes = pp.subplots(nrows=2, ncols=3)
    sel = [axes[0, 0], axes[1, 2]]
    result = _resolve_grid_scope(fig, rows=None, cols=None, span=None, ax=sel)
    assert result == sel


def test_resolve_grid_scope_ax_list_works_without_publiplots_axes():
    """ax= path doesn't require fig._publiplots_axes (works on raw plt.subplots too)."""
    fig, axes = plt.subplots(2, 2)
    sel = [axes[0, 0], axes[1, 1]]
    result = _resolve_grid_scope(fig, rows=None, cols=None, span=None, ax=sel)
    assert result == sel


def test_resolve_grid_scope_ax_empty_raises():
    """ax=[] is invalid (caller probably meant axes=None)."""
    fig, _ = pp.subplots(nrows=2, ncols=2)
    with pytest.raises(ValueError, match=r"ax.*empty"):
        _resolve_grid_scope(fig, rows=None, cols=None, span=None, ax=[])


def test_resolve_grid_scope_ax_with_rows_raises():
    """ax= is mutually exclusive with rows/cols/span."""
    fig, axes = pp.subplots(nrows=2, ncols=2)
    with pytest.raises(ValueError, match=r"mutually exclusive"):
        _resolve_grid_scope(fig, rows=0, cols=None, span=None, ax=[axes[0, 0]])


def test_resolve_grid_scope_span_fig_returns_none():
    """span='fig' → resolver returns None (full figure; caller falls through)."""
    fig, _ = pp.subplots(nrows=2, ncols=3)
    result = _resolve_grid_scope(fig, rows=None, cols=None, span="fig", ax=None)
    assert result is None


def test_resolve_grid_scope_span_invalid_raises():
    fig, _ = pp.subplots(nrows=2, ncols=2)
    with pytest.raises(ValueError, match=r"span.*'row'.*'col'.*'fig'"):
        _resolve_grid_scope(fig, rows=None, cols=None, span="invalid", ax=None)


def test_resolve_grid_scope_span_with_rows_raises():
    """span='fig' is mutually exclusive with explicit rows/cols (different modes)."""
    fig, _ = pp.subplots(nrows=2, ncols=2)
    with pytest.raises(ValueError, match=r"mutually exclusive"):
        _resolve_grid_scope(fig, rows=0, cols=None, span="fig", ax=None)
```

- [ ] **Step 2: Run, verify it fails**

```bash
cd /home/sagemaker-user/publiplots
uv run pytest tests/test_legend_grid_scope.py -v --no-cov 2>&1 | tail -25
```

Expected: 16 tests collected, all FAIL on `ImportError` for `_resolve_grid_scope`.

- [ ] **Step 3: Implement `_resolve_grid_scope`**

Edit `/home/sagemaker-user/publiplots/src/publiplots/utils/legend_group.py`. Insert this function AFTER `def _handle_repr(...)` and BEFORE `class _ScopeAnchor`:

```python


def _resolve_grid_scope(
    fig: Figure,
    *,
    rows: Optional[object] = None,
    cols: Optional[object] = None,
    span: Optional[str] = None,
    ax: Optional[Sequence[Axes]] = None,
) -> Optional[List[Axes]]:
    """Translate grid-scope kwargs into a concrete axes list.

    Returns
    -------
    list of Axes
        The resolved scope when ``rows``/``cols``/``ax`` produce a
        concrete subset.
    None
        When ``rows``/``cols``/``span``/``ax`` are all None, OR when
        ``span='fig'`` (both mean "no grid scoping; fall through to
        figure-level"). Caller's responsibility to dispatch.

    Raises
    ------
    ValueError
        On out-of-range indices, missing ``_publiplots_axes`` matrix
        when one is required, conflicting kwargs, empty ``ax=`` list,
        or invalid ``span`` value.
    """
    # Normalize: count how many of the four are actually set.
    rows_set = rows is not None
    cols_set = cols is not None
    span_set = span is not None
    ax_set = ax is not None

    # All-None → fall through.
    if not (rows_set or cols_set or span_set or ax_set):
        return None

    # `ax=` is mutually exclusive with all three others.
    if ax_set and (rows_set or cols_set or span_set):
        collided = [n for n, s in (("rows", rows_set), ("cols", cols_set),
                                   ("span", span_set)) if s]
        raise ValueError(
            f"pp.legend: `ax=` and `{'`, `'.join(collided)}=` are mutually "
            "exclusive — they're alternative addressing modes. Pick one."
        )

    # `span=` is mutually exclusive with explicit rows/cols.
    if span_set and (rows_set or cols_set):
        collided = [n for n, s in (("rows", rows_set), ("cols", cols_set)) if s]
        raise ValueError(
            f"pp.legend: `span=` and `{'`, `'.join(collided)}=` are mutually "
            "exclusive. Use `span` for sugar OR `rows`/`cols` for explicit "
            "scoping."
        )

    # Explicit ax= path — no _publiplots_axes lookup needed.
    if ax_set:
        ax_list = list(ax)
        if not ax_list:
            raise ValueError(
                "pp.legend: `ax=` was an empty sequence. Pass at least one "
                "Axes, or omit `ax=` for figure-level scoping."
            )
        return ax_list

    # `span=` path.
    if span_set:
        if span not in ("row", "col", "fig"):
            raise ValueError(
                f"pp.legend: span={span!r} invalid. Expected 'row', 'col', "
                "or 'fig'."
            )
        if span == "fig":
            return None  # full figure → caller falls through
        # 'row'/'col' need a positional anchor; the caller (`legend()`)
        # is responsible for passing that context. The resolver itself
        # raises here because we have no anchor.
        raise ValueError(
            f"pp.legend: span={span!r} requires a positional Axes anchor "
            "(`pp.legend(ax_anchor, span='row')`). Without an anchor, use "
            f"`rows=`/`cols=` for explicit indices or `span='fig'` for the "
            "full figure."
        )

    # `rows=`/`cols=` path — needs the _publiplots_axes matrix.
    # Note: pp.Canvas does NOT currently attach _publiplots_axes (PR 7
    # territory). The error message therefore points only at pp.subplots.
    matrix = getattr(fig, "_publiplots_axes", None)
    if not matrix:
        raise ValueError(
            "pp.legend: `rows=`/`cols=` requires a figure built by "
            "`pp.subplots` (no `_publiplots_axes` matrix on this figure). "
            "For raw matplotlib figures or pp.Canvas figures, use "
            "`ax=[ax1, ax2, ...]` instead."
        )

    n_rows = len(matrix)
    n_cols = len(matrix[0]) if matrix else 0

    def _normalize_range(name: str, value: object, length: int) -> tuple[int, int]:
        if isinstance(value, tuple):
            if len(value) != 2:
                raise ValueError(
                    f"pp.legend: `{name}=` tuple must be (start, end) — got "
                    f"{value!r}."
                )
            start, end = value
        else:
            start = end = int(value)  # type: ignore[arg-type]
        # Disallow negative indices: explicit, no Python wrap-around.
        if start < 0 or end < 0:
            last = length - 1
            raise ValueError(
                f"pp.legend: `{name}={value!r}` — negative indices are not "
                f"supported. Use `{name}={last}` for the last "
                f"{'row' if name == 'rows' else 'column'}."
            )
        # Disallow inverted ranges with a clearer message than "out of range".
        if start > end:
            raise ValueError(
                f"pp.legend: `{name}={value!r}` has start > end. Use "
                f"`{name}=({end}, {start})` to specify an inclusive range."
            )
        if not (start < length and end < length):
            raise ValueError(
                f"pp.legend: `{name}={value!r}` out of range for "
                f"_publiplots_axes shape ({n_rows}, {n_cols})."
            )
        return start, end

    if rows_set:
        r_start, r_end = _normalize_range("rows", rows, n_rows)
    else:
        r_start, r_end = 0, n_rows - 1

    if cols_set:
        c_start, c_end = _normalize_range("cols", cols, n_cols)
    else:
        c_start, c_end = 0, n_cols - 1

    out: List[Axes] = []
    for r in range(r_start, r_end + 1):
        for c in range(c_start, c_end + 1):
            out.append(matrix[r][c])
    return out
```

- [ ] **Step 4: Run tests, verify all 14 pass**

```bash
cd /home/sagemaker-user/publiplots
uv run pytest tests/test_legend_grid_scope.py -v --no-cov 2>&1 | tail -25
```

Expected: 16 PASSED. Any failure → debug before moving on.

- [ ] **Step 5: Commit**

```bash
cd /home/sagemaker-user/publiplots
git add src/publiplots/utils/legend_group.py tests/test_legend_grid_scope.py
git commit -m "feat(legend): _resolve_grid_scope helper for rows/cols/span/ax"
```

---

## Task 2: Wire new kwargs into `pp.legend()` factory

**Files:**
- Modify: `src/publiplots/utils/legend_group.py` — extend the `legend(...)` signature + dispatch
- Modify: `tests/test_legend_grid_scope.py` — add factory-level integration tests

**Why second:** with the resolver done in isolation (Task 1), this task is purely the public-API wiring. Keeps the diff readable.

- [ ] **Step 1: Append the failing factory-integration tests**

Append to `/home/sagemaker-user/publiplots/tests/test_legend_grid_scope.py`:

```python


# ---------------------------------------------------------------------------
# `pp.legend(rows=, cols=, span=, ax=)` factory integration
# ---------------------------------------------------------------------------


def test_legend_factory_rows_int_returns_group_with_correct_scope():
    """pp.legend(rows=0) returns a MultiAxesLegendGroup whose scope is row 0."""
    fig, axes = pp.subplots(nrows=2, ncols=3)
    group = pp.legend(rows=0, side="top")
    assert isinstance(group, MultiAxesLegendGroup)
    scope_ids = {id(a) for a in group._scope_axes}  # internal: list of scoped axes
    assert scope_ids == {id(axes[0, 0]), id(axes[0, 1]), id(axes[0, 2])}


def test_legend_factory_rows_tuple_and_cols_int():
    """pp.legend(rows=(0,1), cols=2) → axes[0,2] and axes[1,2]."""
    fig, axes = pp.subplots(nrows=3, ncols=3)
    group = pp.legend(rows=(0, 1), cols=2, side="right")
    scope_ids = {id(a) for a in group._scope_axes}
    assert scope_ids == {id(axes[0, 2]), id(axes[1, 2])}


def test_legend_factory_ax_list_dedupes_handles_by_label():
    """pp.legend(ax=[ax1, ax2]) collects from those axes; same-label/handle dedupes."""
    from publiplots.utils.legend_entries import LegendEntry, stash_entry
    from matplotlib.patches import Rectangle

    fig, axes = pp.subplots(nrows=1, ncols=2)
    h_red = Rectangle((0, 0), 1, 1, facecolor="red", label="A")
    h_red2 = Rectangle((0, 0), 1, 1, facecolor="red", label="A")
    stash_entry(axes[0, 0], LegendEntry.build(name="hue", kind="hue",
                                              handles=[h_red], labels=["A"]))
    stash_entry(axes[0, 1], LegendEntry.build(name="hue", kind="hue",
                                              handles=[h_red2], labels=["A"]))
    group = pp.legend(ax=[axes[0, 0], axes[0, 1]], side="top")
    # Two stashes with same label + same color → group dedupes to one entry.
    # Implementation detail: _merge_entries handles this; we just assert
    # no warnings fire (mismatched-handle path would warn).
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        group._materialize()  # triggers entry collection + merge


def test_legend_factory_span_fig_equals_axes_none():
    """pp.legend(span='fig') is sugar for the figure-level default."""
    fig, _ = pp.subplots(nrows=2, ncols=2)
    group_span = pp.legend(span="fig", side="bottom")
    plt.close(fig)
    fig, _ = pp.subplots(nrows=2, ncols=2)
    group_default = pp.legend(side="bottom")
    # Both should produce a group with scope=None (full grid).
    assert group_span._scope_axes is None
    assert group_default._scope_axes is None


def test_legend_factory_rows_with_legacy_axes_positional_raises():
    """Mixing the new `rows=` with the old positional `axes=` is a TypeError-equivalent."""
    fig, axes = pp.subplots(nrows=2, ncols=2)
    with pytest.raises(ValueError, match=r"`axes=`.*`rows=`.*mutually exclusive"):
        pp.legend(axes[0, 0], rows=0, side="top")


def test_legend_factory_grid_scope_renders_without_error():
    """Smoke: a row-scoped band on a 2x3 grid actually renders to PNG."""
    import io
    from publiplots.utils.legend_entries import LegendEntry, stash_entry
    from matplotlib.patches import Rectangle

    fig, axes = pp.subplots(nrows=2, ncols=3)
    for ax in axes.flat:
        stash_entry(ax, LegendEntry.build(name="hue", kind="hue",
                                          handles=[Rectangle((0,0),1,1,
                                                             facecolor="C0",
                                                             label="A")],
                                          labels=["A"]))
    pp.legend(rows=0, side="top")
    buf = io.BytesIO()
    fig.savefig(buf, format="png")  # no exceptions
    assert buf.getbuffer().nbytes > 0
```

- [ ] **Step 2: Run, verify the new tests fail**

```bash
cd /home/sagemaker-user/publiplots
uv run pytest tests/test_legend_grid_scope.py -v --no-cov 2>&1 | tail -25
```

Expected: 16 prior PASS + 6 new FAIL on `TypeError: legend() got an unexpected keyword argument 'rows'`.

- [ ] **Step 3: Extend the `legend()` factory signature**

Edit `/home/sagemaker-user/publiplots/src/publiplots/utils/legend_group.py`. Modify the `def legend(...)` signature (around line 977) to add four new keyword-only kwargs:

Find:
```python
def legend(
    axes=None,
    collect: Optional[Sequence[str]] = None,
    *,
    side: str = "right",
    anchor: Optional[Axes] = None,
    figure: Optional[Figure] = None,
    orientation: str = "auto",
    align: str = "auto",
    x_offset: Optional[float] = None,
    y_offset: Optional[float] = None,
    gap: float = 2,
    column_spacing: float = 5,
    vpad: Optional[float] = None,
    max_width: Optional[float] = None,
) -> MultiAxesLegendGroup:
```

Replace with:
```python
def legend(
    axes=None,
    collect: Optional[Sequence[str]] = None,
    *,
    side: str = "right",
    anchor: Optional[Axes] = None,
    figure: Optional[Figure] = None,
    orientation: str = "auto",
    align: str = "auto",
    x_offset: Optional[float] = None,
    y_offset: Optional[float] = None,
    gap: float = 2,
    column_spacing: float = 5,
    vpad: Optional[float] = None,
    max_width: Optional[float] = None,
    # PR 4: grid-scope kwargs (additive; all default None → existing behavior)
    rows: Optional[object] = None,
    cols: Optional[object] = None,
    span: Optional[str] = None,
    ax: Optional[Sequence[Axes]] = None,
) -> MultiAxesLegendGroup:
```

- [ ] **Step 4: Add resolver dispatch at the top of the function body**

Find the docstring's closing `"""` and the line that starts the resolution logic:
```python
    # Resolve anchor + axes per the rules:
```

Insert a new block immediately AFTER the closing `"""` and BEFORE the `# Resolve anchor + axes per the rules:` comment:

```python
    # PR 4: if any of the new grid-scope kwargs are set, resolve them up-front
    # to a list-of-axes scope (or None for figure-level), then fall through to
    # the existing resolution logic by setting `axes`.
    if rows is not None or cols is not None or span is not None or ax is not None:
        # Mixing new kwargs with the legacy `axes=` positional is ambiguous —
        # they're alternative addressing modes. Raise rather than guess.
        if axes is not None:
            raise ValueError(
                "pp.legend: legacy positional `axes=` is mutually exclusive "
                "with the new `rows=`/`cols=`/`span=`/`ax=` kwargs. Use one "
                "addressing mode at a time."
            )
        # Resolve the figure for the resolver (mirrors the figure-resolution
        # used later in this function for full-figure scope).
        resolver_fig = figure if figure is not None else plt.gcf()
        resolved_scope = _resolve_grid_scope(
            resolver_fig, rows=rows, cols=cols, span=span, ax=ax,
        )
        if resolved_scope is not None:
            # Drift guard: if `figure=` was explicit AND the resolved axes
            # belong to a different figure, bail rather than letting the
            # downstream group silently switch figures off the resolved
            # axes' get_figure(). The `ax=` path is the most common way
            # to hit this — caller passes ax= from one figure but figure=
            # naming another.
            if figure is not None:
                for a in resolved_scope:
                    if a.get_figure() is not figure:
                        raise ValueError(
                            "pp.legend: resolved axes belong to a different "
                            "Figure than the one passed via `figure=`. "
                            "Either drop `figure=` or pass axes from that "
                            "figure."
                        )
            # Concrete subset → feed into the existing list-of-axes branch.
            axes = resolved_scope
        # else: span='fig' or all-None → leave axes=None (figure-level default).

```

- [ ] **Step 5: Run all relevant tests**

```bash
cd /home/sagemaker-user/publiplots
uv run pytest tests/test_legend_grid_scope.py tests/test_legend_group.py tests/test_legend_unified.py tests/test_legend_group_axes_scope.py tests/test_legend_group_sides.py -v --no-cov 2>&1 | tail -25
```

Expected: ALL pass. The 22 grid-scope tests so far (16 from Task 1 + 6 from Task 2) + every existing legend test (no regression). The factory's existing branches are reachable via the `axes` arg the same as before.

- [ ] **Step 6: Commit**

```bash
cd /home/sagemaker-user/publiplots
git add src/publiplots/utils/legend_group.py tests/test_legend_grid_scope.py
git commit -m "feat(legend): pp.legend rows/cols/span/ax grid-scope kwargs"
```

---

## Task 3: `span='row'` / `span='col'` with positional anchor

**Files:**
- Modify: `src/publiplots/utils/legend_group.py` — extend the resolver-dispatch block in `legend()` to handle `span='row'/'col'` with `axes=ax_anchor`
- Modify: `tests/test_legend_grid_scope.py` — add 4 tests

**Why third:** `span='row'`/`'col'` requires a positional anchor; that wiring lives in the factory (the resolver doesn't know about positional args). Keeping it in its own task isolates the factory-side logic from Task 2's resolver-side logic.

- [ ] **Step 1: Append failing tests**

Append to `/home/sagemaker-user/publiplots/tests/test_legend_grid_scope.py`:

```python


# ---------------------------------------------------------------------------
# `span='row'` / `span='col'` with positional anchor
# ---------------------------------------------------------------------------


def test_legend_span_row_with_anchor_expands_to_row():
    """pp.legend(axes[1,0], span='row') scopes to all of row 1."""
    fig, axes = pp.subplots(nrows=2, ncols=3)
    group = pp.legend(axes[1, 0], span="row", side="top")
    scope_ids = {id(a) for a in group._scope_axes}
    assert scope_ids == {id(axes[1, 0]), id(axes[1, 1]), id(axes[1, 2])}


def test_legend_span_col_with_anchor_expands_to_col():
    """pp.legend(axes[0,1], span='col') scopes to all of col 1."""
    fig, axes = pp.subplots(nrows=3, ncols=2)
    group = pp.legend(axes[0, 1], span="col", side="right")
    scope_ids = {id(a) for a in group._scope_axes}
    assert scope_ids == {id(axes[0, 1]), id(axes[1, 1]), id(axes[2, 1])}


def test_legend_span_row_without_anchor_raises():
    """span='row' without a positional Axes → ValueError."""
    fig, _ = pp.subplots(nrows=2, ncols=2)
    with pytest.raises(ValueError, match=r"span='row'.*positional Axes"):
        pp.legend(span="row", side="top")


def test_legend_span_row_with_non_publiplots_anchor_raises():
    """span='row' on a raw plt figure → ValueError."""
    fig, axes = plt.subplots(2, 2)
    with pytest.raises(ValueError, match=r"_publiplots_axes"):
        pp.legend(axes[0, 0], span="row", side="top")


def test_legend_positional_with_rows_and_span_collides():
    """pp.legend(axes[0,0], rows=0, span='row') — positional + rows + span all
    set; the is_positional_anchor_span guard requires rows/cols/ax all None,
    so this falls through to the mutually-exclusive raise."""
    fig, axes = pp.subplots(nrows=2, ncols=2)
    with pytest.raises(ValueError, match=r"mutually exclusive"):
        pp.legend(axes[0, 0], rows=0, span="row", side="top")
```

- [ ] **Step 2: Run, verify the 4 new tests fail**

```bash
cd /home/sagemaker-user/publiplots
uv run pytest tests/test_legend_grid_scope.py -v --no-cov 2>&1 | tail -10
```

Expected: 5 new FAIL (4 new + the positional-with-rows-and-span collision test). The first two fail because `span='row'` currently raises in the resolver (no anchor); the others fail with the wrong error message or pass-through behavior.

- [ ] **Step 3: Add a small `_expand_span_with_anchor` helper**

Edit `/home/sagemaker-user/publiplots/src/publiplots/utils/legend_group.py`. Add this helper IMMEDIATELY after `_resolve_grid_scope` (i.e., still before `class _ScopeAnchor`):

```python


def _expand_span_with_anchor(
    fig: Figure,
    anchor: Axes,
    span: str,
) -> List[Axes]:
    """Expand `span='row'/'col'` against a positional Axes anchor.

    Locates ``anchor`` in ``fig._publiplots_axes`` and returns the full
    row or column containing it.

    Raises
    ------
    ValueError
        If the figure has no ``_publiplots_axes`` matrix, or if
        ``anchor`` is not in that matrix.
    """
    matrix = getattr(fig, "_publiplots_axes", None)
    if not matrix:
        raise ValueError(
            "pp.legend: `span='row'`/`'col'` requires a figure built by "
            "`pp.subplots` or `pp.Canvas` (no `_publiplots_axes` matrix on "
            "this figure)."
        )
    target_id = id(anchor)
    for r, row in enumerate(matrix):
        for c, ax_in_row in enumerate(row):
            if id(ax_in_row) == target_id:
                if span == "row":
                    return list(row)
                if span == "col":
                    return [matrix[rr][c] for rr in range(len(matrix))]
                raise ValueError(
                    f"pp.legend: span={span!r} not 'row' or 'col'."
                )
    raise ValueError(
        "pp.legend: positional anchor was not found in this figure's "
        "`_publiplots_axes` matrix."
    )
```

- [ ] **Step 4: Update the factory's resolver-dispatch block to handle span='row'/'col'**

Edit the dispatch block from Task 2 inside `def legend(...)`. Replace the existing block:

```python
    if rows is not None or cols is not None or span is not None or ax is not None:
        # Mixing new kwargs with the legacy `axes=` positional is ambiguous —
        # they're alternative addressing modes. Raise rather than guess.
        if axes is not None:
            raise ValueError(
                "pp.legend: legacy positional `axes=` is mutually exclusive "
                "with the new `rows=`/`cols=`/`span=`/`ax=` kwargs. Use one "
                "addressing mode at a time."
            )
        ...
```

With this expanded block:

```python
    if rows is not None or cols is not None or span is not None or ax is not None:
        # span='row'/'col' is the SOLE exception to "new kwargs are mutually
        # exclusive with the positional `axes=` arg" — those two span values
        # REQUIRE a positional anchor, so they consume `axes=ax` instead of
        # raising on it.
        is_positional_anchor_span = (
            span in ("row", "col")
            and isinstance(axes, Axes)
            and rows is None and cols is None and ax is None
        )

        if axes is not None and not is_positional_anchor_span:
            raise ValueError(
                "pp.legend: legacy positional `axes=` is mutually exclusive "
                "with the new `rows=`/`cols=`/`span=`/`ax=` kwargs. Use one "
                "addressing mode at a time."
            )

        resolver_fig = figure if figure is not None else plt.gcf()

        if is_positional_anchor_span:
            # Expand against the anchor; consume `axes` and continue down the
            # list-of-axes branch.
            expanded = _expand_span_with_anchor(resolver_fig, axes, span)
            axes = expanded
        else:
            if span in ("row", "col") and not isinstance(axes, Axes):
                raise ValueError(
                    f"pp.legend: span={span!r} requires a positional Axes "
                    "anchor (e.g. `pp.legend(axes[0,1], span='row')`)."
                )
            resolved_scope = _resolve_grid_scope(
                resolver_fig, rows=rows, cols=cols, span=span, ax=ax,
            )
            if resolved_scope is not None:
                axes = resolved_scope
            # else: span='fig' or all-None → leave axes=None.

```

- [ ] **Step 5: Run tests**

```bash
cd /home/sagemaker-user/publiplots
uv run pytest tests/test_legend_grid_scope.py tests/test_legend_group.py tests/test_legend_unified.py tests/test_legend_group_axes_scope.py tests/test_legend_group_sides.py -v --no-cov 2>&1 | tail -10
```

Expected: ALL pass (27 grid-scope + every prior legend test).

- [ ] **Step 6: Commit**

```bash
cd /home/sagemaker-user/publiplots
git add src/publiplots/utils/legend_group.py tests/test_legend_grid_scope.py
git commit -m "feat(legend): span='row'/'col' with positional Axes anchor"
```

---

## Task 4: Skill update + CHANGELOG + gallery snippet

**Files:**
- Modify: `skills/legend-placement/SKILL.md` — new "Grid scoping" section
- Create: `examples/legend/grid_scope_demo.py` — small demo
- Modify: `CHANGELOG.md` — `[Unreleased]` section

**Why fourth:** behavior is in place (Tasks 1-3); now the user-facing surface (skill + CHANGELOG) tracks it.

- [ ] **Step 1: Add a "Grid scoping" section to the skill**

Read `/home/sagemaker-user/publiplots/skills/legend-placement/SKILL.md` first to identify the right insertion point (look for an existing section after the "axes scope" content, or near the end before "Common pitfalls"). Then add a new section with:

```markdown
## Grid scoping (PR 4 / v0.12)

For figures built by `pp.subplots` or `pp.Canvas`, `pp.legend` accepts
four kwargs that resolve to a sub-rect of the grid:

```python
# Row-scoped band over row 0 (e.g., a top-row group legend)
pp.legend(rows=0, side='top')

# Inclusive row range × specific column
pp.legend(rows=(1, 3), cols=2, side='right')

# Sugar: full-figure band (alternative to default kwargs)
pp.legend(span='fig', side='bottom')

# Sugar: full-row band keyed off a positional Axes anchor
pp.legend(axes[0, 1], span='row', side='top')

# Explicit list with handle dedupe
pp.legend(ax=[ax_a, ax_b, ax_c], side='top')
```

**Mutual exclusivity:** `rows=`/`cols=`/`span=`/`ax=` are alternative
addressing modes. Mixing them (other than `span='row'`/`'col'` with a
positional anchor) raises `ValueError`. The legacy positional `axes=`
arg is also exclusive with the new kwargs except for the
positional-anchor `span` form.

**Out-of-range indices** raise with the actual `_publiplots_axes`
shape so the message points at the cause: e.g.,
`pp.legend(rows=5)` on a 2×3 grid → `rows=5 out of range for shape (2, 3)`.

**Raw matplotlib figures:** `rows=`/`cols=` need the publiplots-built
matrix; on raw `plt.subplots()` figures use `ax=[ax1, ax2, ...]`
instead.
```

- [ ] **Step 2: Create the gallery demo**

Create `/home/sagemaker-user/publiplots/examples/legend/grid_scope_demo.py`:

```python
"""Demo: pp.legend(rows=, cols=) for grid-scoped figure legends.

Renders a 2×3 grid where row 0 shares one legend band on top and
row 1 shares another on bottom, demonstrating how the new grid-scope
kwargs let two independent legend groups coexist on a single figure.
"""
import numpy as np

import publiplots as pp


def main(out_path: str = "examples/legend/grid_scope_demo.png") -> None:
    rng = np.random.default_rng(0)
    fig, axes = pp.subplots(nrows=2, ncols=3, figsize=(180, 80))

    # Row 0: scatter, hue = 'cond_a'
    for ax in axes[0]:
        x = rng.normal(size=80)
        y = rng.normal(size=80)
        cond = rng.choice(["lo", "hi"], size=80)
        pp.scatterplot(x=x, y=y, hue=cond, ax=ax)

    # Row 1: bar, hue = 'cond_b'
    cats = ["A", "B", "C"]
    for ax in axes[1]:
        for cat_idx, cat in enumerate(cats):
            for cond_idx, cond in enumerate(["x", "y"]):
                ax.bar(
                    cat_idx + cond_idx * 0.4,
                    rng.uniform(0.5, 1.5),
                    width=0.4,
                    label=cond if cat_idx == 0 else None,
                )

    # One band per row.
    pp.legend(rows=0, side="top")
    pp.legend(rows=1, side="bottom")

    fig.savefig(out_path)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Render it**

```bash
cd /home/sagemaker-user/publiplots
mkdir -p examples/legend
uv run python examples/legend/grid_scope_demo.py 2>&1 | tail -5
```

Expected: completes without error; a PNG appears at `examples/legend/grid_scope_demo.png`. Spot-check it visually if running locally.

- [ ] **Step 4: CHANGELOG**

Read `/home/sagemaker-user/publiplots/CHANGELOG.md` and find the `## [Unreleased]` block. Add the following bullet to the `### Added` subsection (creating it if needed; PR 1+2+3 added entries already exist, so the section should be present):

```markdown
- `pp.legend(rows=, cols=, span=, ax=)` — grid-scope kwargs for
  figure legends. `rows`/`cols` resolve over the
  `pp.subplots`/`pp.Canvas` axes matrix to a single index or inclusive
  `(start, end)` range. `span='fig'` is sugar for the full-figure
  default; `span='row'`/`'col'` with a positional Axes anchor expands
  to that anchor's row or column. `ax=[a1, a2, ...]` collects from an
  explicit list with handle dedupe by label. All four are
  mutually-exclusive addressing modes; existing call signatures
  unchanged.
```

- [ ] **Step 5: Run the full test suite to check no regression elsewhere**

```bash
cd /home/sagemaker-user/publiplots
uv run pytest --no-cov 2>&1 | tail -10
```

Expected: 1230 + 27 = 1257 passed (1019 baseline + 211 composer + 27 new grid-scope tests), 1 pre-existing residplot failure unchanged. Any other failure → debug.

- [ ] **Step 6: Commit**

```bash
cd /home/sagemaker-user/publiplots
git add skills/legend-placement/SKILL.md \
        examples/legend/grid_scope_demo.py \
        examples/legend/grid_scope_demo.png \
        CHANGELOG.md
git commit -m "docs(legend): grid-scope skill section + gallery demo + CHANGELOG"
```

---

## Task 5: Open the PR

**Files:** none (git only)

- [ ] **Step 1: Push the branch**

```bash
cd /home/sagemaker-user/publiplots
git push -u origin feat/composer-legend-scopes-pr4
```

- [ ] **Step 2: Open PR**

```bash
cd /home/sagemaker-user/publiplots
gh pr create \
  --title "feat(legend): pp.legend rows/cols/span/ax grid-scope kwargs" \
  --body "$(cat <<'EOF'
## Summary

PR 4 of the Composer rollout. Strictly-additive upgrade to `pp.legend`:
adds `rows=`, `cols=`, `span=`, `ax=` kwargs for grid-scope figure
legends. Independent of the Composer — works on any figure built by
`pp.subplots` or `pp.Canvas`. All existing call signatures unchanged.

## What's in this PR

- **`pp.legend(rows=N | (a,b))`** — figure-level band over a single
  row or inclusive range, resolved against `fig._publiplots_axes`.
- **`pp.legend(cols=N | (a,b))`** — same for columns.
- **`pp.legend(rows=, cols=)`** — intersection (a sub-rect of the
  grid).
- **`pp.legend(span='fig')`** — sugar for the full-figure default.
- **`pp.legend(axes[0,1], span='row' | 'col')`** — sugar for "expand
  to the row/column containing this anchor".
- **`pp.legend(ax=[ax1, ax2, ...])`** — explicit list with handle
  dedupe by label.
- **Validation:** out-of-range indices raise with the actual matrix
  shape; addressing modes are mutually exclusive; `rows=`/`cols=`
  on a raw matplotlib figure raises pointing at `pp.subplots`/`Canvas`.
- **`legend-placement` skill** — new "Grid scoping" section.
- **Gallery demo** — `examples/legend/grid_scope_demo.py`.

## What's NOT in this PR

- Composer-specific behavior — `pp.Canvas` already attaches
  `_publiplots_axes` post-finalization, so `pp.legend(rows=0)` on a
  Canvas figure works through the same code path; no Canvas-side code.
- mm-precision regression infra (PR 4.5)
- Vector PDF/SVG (PR 5/6)
- `canvas.inspect()` + composer-guide skill (PR 7)
- Promoting `ComposerError`/`Composer*Error` to top-level `pp.*`
  (deferred to PR 7 with the skill)

## Test plan

- [x] 27 new tests in `tests/test_legend_grid_scope.py` (resolver
      unit tests + factory integration tests + `span` with anchor +
      Canvas-deferral test)
- [x] All existing legend tests still pass (no regression)
- [x] Full suite green: 1230 + 27 = 1257 passed; 1 pre-existing
      residplot failure unchanged
- [x] Gallery demo renders to PNG without error

## Implementation notes

- `_resolve_grid_scope` is a pure-Python helper — no matplotlib
  drawing. All geometry math lives in front of `MultiAxesLegendGroup`,
  which is unchanged.
- `span='row'`/`'col'` consumes the positional `axes=` arg as the
  anchor; this is the SOLE exception to "new kwargs are mutually
  exclusive with positional `axes=`".
- Negative indices are NOT supported (no Python wrap-around); we
  reject them with the same out-of-range message so the contract is
  unambiguous.
- `ax=[...]` empty-list raises (caller probably meant `axes=None` for
  figure-level).

## Follow-ups

PR 4.5 (mm-precision + golden-output test infra) is the next prereq
for PR 5 (vector PDF). PR 4's grid-scope kwargs are also exercised
implicitly by PR 5's golden compositions.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
gh pr view --json number,url,state,title
```

Capture the PR URL for the human.

---

## Acceptance criteria for PR 4

The PR is ready for human merge when ALL of:

1. All 5 tasks complete.
2. Full test suite green: 1230 prior + 27 new grid-scope tests = 1257 passed; 1 pre-existing residplot failure unchanged.
3. `pp.legend(rows=N)` / `pp.legend(rows=(a, b))` produces a band scoped to those rows.
4. `pp.legend(cols=N)` / `pp.legend(cols=(a, b))` produces a band scoped to those cols.
5. `pp.legend(rows=, cols=)` produces a band scoped to the intersection sub-rect.
6. `pp.legend(span='fig')` produces a full-figure band identical to `pp.legend(side=...)`.
7. `pp.legend(axes[r,c], span='row')` and `pp.legend(axes[r,c], span='col')` expand correctly.
8. `pp.legend(ax=[a1, a2, ...])` collects from those axes with handle dedupe.
9. Mixing `rows=`/`cols=`/`span=`/`ax=` with the legacy positional `axes=` raises `ValueError` (except the `span='row'/'col'` anchor case).
10. Out-of-range indices raise `ValueError` naming the matrix shape; negative indices raise with a positive-index hint; `start > end` ranges raise with the swapped form suggested.
11. `_resolve_grid_scope` on a raw `plt.subplots()` figure AND on a `pp.Canvas` figure (which doesn't attach `_publiplots_axes`) both raise `ValueError` mentioning `pp.subplots`. Canvas integration is deferred to PR 7.
12. CHANGELOG entry added under `[Unreleased] / Added`.
13. `legend-placement` skill has a new "Grid scoping" section.
14. `examples/legend/grid_scope_demo.py` renders to PNG without warnings.
15. Out-of-scope items (Canvas-side `_publiplots_axes` wiring, mm-precision infra, vector save, inspect, skill, public `scope_axes` accessor) are NOT in the diff.

If any of #1–#15 fail, the PR is not ready.

---

## Per-PR agent team

Per the rollout convention (see `2026-05-15` rollout memory + spec §"Per-PR agent team"):

1. **`code-architect`** (opus) — reviews the resolver contract + factory dispatch interface BEFORE any code lands. Confirms:
   - `_resolve_grid_scope` returns `Optional[List[Axes]]` semantics
   - the four kwargs' mutual-exclusivity matrix
   - error-message wording for the canonical failure cases
2. **`test-designer`** (general-purpose, opus) — already embedded in this plan as the failing-test scaffolds in each task. The implementer's job is to fill in the implementation against those scaffolds; net-new tests beyond the plan should be flagged for review.
3. **Implementer** (general-purpose / ml-implementer, opus) — fills in the implementation per Tasks 1-4 in TDD order.
4. **Spec-compliance reviewer** (general-purpose, opus) — adversarial review against the "What's IN" / "What's OUT" lists at the top of this plan, the spec §"`pp.legend` upgrade" prose (lines 248-271), and the Acceptance criteria.
5. **Code-quality reviewer** (general-purpose, opus) — independent code-review focused on validation completeness, error-message quality, and back-compat preservation.
6. **`debugger`** — invoked on any test failure or unexpected behavior during implementation.

All sub-agents on opus per the global rule.

---

## Post-merge steps (for the human, not the implementer)

After PR 4 merges:
1. Update the rollout memory's status table: PR 4 → ✅ MERGED.
2. The next PR is PR 4.5 (mm-precision + golden-output test infra). Plan to be written when PR 4 lands.
3. PR 5 (vector-PDF + PanelImage) is blocked until PR 4.5 ships.
