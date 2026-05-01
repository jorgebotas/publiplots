# Legend Auto-Collection + Width-Aware Sizing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `pp.legend_group(anchor=..., collect=None|[names])` that auto-collects per-axes legend entries across a grid, plus `SubplotsAutoLayout` measuring the group width so `legend_column` is always automatic.

**Architecture:** Introduce a shared `LegendEntry` dataclass + per-axes `_publiplots_legend_entries` store. Plot functions (in this PR: scatter / strip / swarm / point — the 4 that already stash) migrate to the new store. `MultiAxesLegendGroup` gains a `collect` kwarg and a `_materialize` pass that walks the grid on first draw, dedups by `(name, kind)`, and renders via the existing `LegendBuilder`. `SubplotsAutoLayout._measure` gains a fifth measurement (`legend_column`) that reads the rendered group width and feeds it into `FigureLayout.with_updated_reservations`. `pp.subplots()` drops the `legend_column` kwarg.

**Tech Stack:** Python 3.11+, matplotlib, numpy, pytest (Agg backend for integration tests).

**Worktree:** `.worktrees/legend-positioning` on `feat/legend-positioning`, branched from `main` after PR #80 merged. Baseline: 121 tests passing.

**Spec:** `docs/superpowers/specs/2026-05-01-legend-auto-collection-design.md` (authoritative — re-read before starting).

---

## Scope reminder

**In-scope plot migrations (this PR):**
- `src/publiplots/plot/scatter.py`
- `src/publiplots/plot/strip.py`
- `src/publiplots/plot/swarm.py`
- `src/publiplots/plot/point.py`

**Deferred to follow-up PRs (keep current behavior):**
- `src/publiplots/plot/bar.py`
- `src/publiplots/plot/box.py`
- `src/publiplots/plot/violin.py`
- `src/publiplots/plot/raincloud.py`
- `src/publiplots/plot/heatmap.py`

These 5 continue to render legends inline. `pp.legend_group` will not auto-collect their entries; users fall back to manual `group.add_legend(handles=...)` exactly as today for those plots.

---

## Rules of the road

- TDD strictly: failing test first, verify fail, then implementation.
- After EACH task, run the full suite `unset VIRTUAL_ENV; .venv/bin/python -m pytest tests/ -q`. Targets update as tasks land.
- Commit after every task with a conventional-commits message. Never amend.
- Do NOT modify public API surfaces beyond what each task specifies.
- `unset VIRTUAL_ENV` before every python/pytest command — the shell alias to the lemur venv is a known trap on this machine.
- Treat the 5 deferred plots as untouched — do not modify their files in this plan.

---

## File structure

**Create:**
- `src/publiplots/utils/legend_entries.py` — `LegendEntry`, `stash_entry`, `get_entries`, `resolve_legend_flags`, `entry_is_in_group`, `is_continuous_hue`.
- `tests/test_legend_entries.py`
- `tests/test_legend_group_auto_collect.py`
- `tests/test_scatter_legend_stash.py`

**Modify:**
- `src/publiplots/utils/legend_group.py` — `collect` kwarg, `_materialize`, `claims`, `_render_entry`, figure registration.
- `src/publiplots/layout/auto_layout.py` — `_measure_legend_column`, extend `_ALL_SIDES`, extend `_needs_update`.
- `src/publiplots/layout/subplots.py` — drop `legend_column` kwarg; raise `TypeError` if passed.
- `src/publiplots/utils/legend.py` — rewrite auto-mode (`legend()` public function + `_get_legend_data`) to read the new `LegendEntry` store with fallback to the legacy dict store.
- `src/publiplots/plot/scatter.py` — migrate to stash `LegendEntry`.
- `src/publiplots/plot/strip.py` — same.
- `src/publiplots/plot/swarm.py` — same.
- `src/publiplots/plot/point.py` — same.
- `examples/plots/plot_14_edgecolor_control.py` — drop `legend_column=30` from the one `pp.subplots` call that still references a migrated plot (the 1×3 bar/box/scatter section; the 1×2 raincloud stays with manual `group.add_legend` because raincloud is NOT migrated).
- `tests/test_subplots.py` — drop `test_subplots_legend_column_reserves_extra_width`; add width-awareness tests.

---

## Task 1: `LegendEntry` + entry store helpers

Creates the foundation module that every other task imports from.

**Files:**
- Create: `src/publiplots/utils/legend_entries.py`
- Create: `tests/test_legend_entries.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_legend_entries.py` with this exact content:

```python
"""Tests for legend_entries.py — entries, stashing, flag resolution."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from publiplots.utils.legend_entries import (
    LegendEntry,
    stash_entry,
    get_entries,
    resolve_legend_flags,
    entry_is_in_group,
    is_continuous_hue,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


class _FakeHandle:
    def __init__(self, fc="red"):
        self._fc = fc
    def get_facecolor(self):
        return self._fc


def test_legend_entry_build_is_deterministic():
    e1 = LegendEntry.build("g", "hue", [_FakeHandle("red")], ["A"])
    e2 = LegendEntry.build("g", "hue", [_FakeHandle("red")], ["A"])
    assert e1.signature == e2.signature


def test_legend_entry_different_handles_differ():
    e1 = LegendEntry.build("g", "hue", [_FakeHandle("red")], ["A"])
    e2 = LegendEntry.build("g", "hue", [_FakeHandle("blue")], ["A"])
    assert e1.signature != e2.signature


def test_legend_entry_name_and_kind_preserved():
    e = LegendEntry.build("treatment", "size", [], [])
    assert e.name == "treatment"
    assert e.kind == "size"
    assert e.handles == ()
    assert e.labels == ()


def test_stash_entry_creates_list_first_time():
    fig, ax = plt.subplots()
    assert get_entries(ax) == []
    stash_entry(ax, LegendEntry.build("g", "hue", [], []))
    assert len(get_entries(ax)) == 1


def test_stash_entry_appends_in_order():
    fig, ax = plt.subplots()
    stash_entry(ax, LegendEntry.build("g1", "hue", [], []))
    stash_entry(ax, LegendEntry.build("g2", "size", [], []))
    names = [e.name for e in get_entries(ax)]
    assert names == ["g1", "g2"]


def test_resolve_legend_flags_true():
    flags = resolve_legend_flags(True)
    assert flags == {"hue": True, "size": True, "style": True, "marker": True}


def test_resolve_legend_flags_false():
    flags = resolve_legend_flags(False)
    assert flags == {"hue": False, "size": False, "style": False, "marker": False}


def test_resolve_legend_flags_dict_partial_missing_defaults_to_true():
    flags = resolve_legend_flags({"hue": False})
    assert flags == {"hue": False, "size": True, "style": True, "marker": True}


def test_resolve_legend_flags_dict_full():
    flags = resolve_legend_flags({"hue": True, "size": False, "style": False, "marker": False})
    assert flags == {"hue": True, "size": False, "style": False, "marker": False}


def test_resolve_legend_flags_rejects_bad_type():
    with pytest.raises(TypeError, match="legend must be bool or dict"):
        resolve_legend_flags(1)


def test_entry_is_in_group_no_group():
    fig, ax = plt.subplots()
    entry = LegendEntry.build("g", "hue", [], [])
    assert entry_is_in_group(fig, entry) is False


def test_entry_is_in_group_with_claim():
    """entry_is_in_group delegates to group.claims(name) — mock one."""
    fig, ax = plt.subplots()
    entry = LegendEntry.build("treatment", "hue", [], [])

    class _FakeGroup:
        def claims(self, name):
            return name == "treatment"

    fig._publiplots_legend_group = _FakeGroup()
    assert entry_is_in_group(fig, entry) is True

    other = LegendEntry.build("other", "hue", [], [])
    assert entry_is_in_group(fig, other) is False


def test_is_continuous_hue_true_for_scalar_mappable():
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    sm = ScalarMappable(cmap="viridis", norm=Normalize(0, 1))
    assert is_continuous_hue([sm]) is True


def test_is_continuous_hue_false_for_empty():
    assert is_continuous_hue([]) is False


def test_is_continuous_hue_false_for_regular_handles():
    assert is_continuous_hue([_FakeHandle()]) is False
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `unset VIRTUAL_ENV; .venv/bin/python -m pytest tests/test_legend_entries.py -v`
Expected: all FAIL with `ModuleNotFoundError: No module named 'publiplots.utils.legend_entries'`.

- [ ] **Step 3: Create `src/publiplots/utils/legend_entries.py`**

```python
"""
Shared legend-entry infrastructure.

Plot functions stash LegendEntry objects on ax._publiplots_legend_entries.
pp.legend(ax) reads from this store to render per-axis legends.
pp.legend_group(anchor=ax) aggregates entries across a grid of axes.
"""

from dataclasses import dataclass
from typing import Tuple
import hashlib


_LEGEND_KINDS = ("hue", "size", "style", "marker")


@dataclass(frozen=True)
class LegendEntry:
    """A single stashed legend entry on an axes.

    Attributes
    ----------
    name : str
        The variable name the user passed to the plot function
        (e.g. ``hue='treatment'`` -> ``name='treatment'``).
    kind : str
        One of ``"hue"``, ``"size"``, ``"style"``, ``"marker"``.
    handles : tuple
        Matplotlib-compatible handles. For continuous hue, the first
        handle is a ``ScalarMappable`` (see :func:`is_continuous_hue`).
    labels : tuple of str
        Display labels. Empty for continuous hue (colorbar path).
    signature : str
        Short hash of (kind, labels, handle-type + key visual props).
        Used by pp.legend_group for dedup and mismatch detection.
    """
    name: str
    kind: str
    handles: tuple
    labels: tuple
    signature: str

    @classmethod
    def build(cls, name, kind, handles, labels) -> "LegendEntry":
        """Construct an entry with a computed signature."""
        return cls(
            name=name,
            kind=kind,
            handles=tuple(handles),
            labels=tuple(labels),
            signature=_hash_handles(handles, labels),
        )


def _hash_handles(handles, labels) -> str:
    parts = []
    for h, lab in zip(handles, labels):
        parts.append(type(h).__name__)
        parts.append(str(lab))
        for attr in ("get_facecolor", "get_marker", "get_markersize",
                     "get_linewidth"):
            fn = getattr(h, attr, None)
            if fn is not None:
                try:
                    parts.append(repr(fn()))
                except Exception:
                    pass
    # If no labels but there ARE handles (continuous hue / colorbar),
    # include the handle types at least.
    if not labels and handles:
        for h in handles:
            parts.append(type(h).__name__)
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:12]


def stash_entry(ax, entry: LegendEntry) -> None:
    """Append an entry to ``ax._publiplots_legend_entries``.

    Creates the list attribute on first call. Order is preserved;
    later calls append.
    """
    existing = getattr(ax, "_publiplots_legend_entries", None)
    if existing is None:
        existing = []
        ax._publiplots_legend_entries = existing
    existing.append(entry)


def get_entries(ax) -> list:
    """Return the ordered list of entries stashed on ``ax``."""
    return list(getattr(ax, "_publiplots_legend_entries", []))


def resolve_legend_flags(legend) -> dict:
    """Convert ``legend=`` (bool | dict) to a per-kind include map.

    - ``True``  -> all kinds True
    - ``False`` -> all kinds False
    - ``dict``  -> as given; missing keys default to True
    """
    if legend is True:
        return {k: True for k in _LEGEND_KINDS}
    if legend is False:
        return {k: False for k in _LEGEND_KINDS}
    if isinstance(legend, dict):
        return {k: bool(legend.get(k, True)) for k in _LEGEND_KINDS}
    raise TypeError(
        f"legend must be bool or dict[str, bool], got {type(legend).__name__}"
    )


def entry_is_in_group(fig, entry: LegendEntry) -> bool:
    """True if the figure's legend_group (if any) claims this entry."""
    group = getattr(fig, "_publiplots_legend_group", None)
    if group is None:
        return False
    return group.claims(entry.name)


def is_continuous_hue(handles) -> bool:
    """True if the handles list represents a continuous colormap.

    Detection is by the presence of a ``ScalarMappable`` as the first
    handle — categorical hue handles are publiplots' RectanglePatch /
    MarkerPatch / etc., never ScalarMappable.
    """
    if not handles:
        return False
    try:
        from matplotlib.cm import ScalarMappable
    except ImportError:
        return False
    return isinstance(handles[0], ScalarMappable)
```

- [ ] **Step 4: Run the tests and verify they pass**

Run: `unset VIRTUAL_ENV; .venv/bin/python -m pytest tests/test_legend_entries.py -v`
Expected: all PASS (14 tests).

- [ ] **Step 5: Run the full suite for regressions**

Run: `unset VIRTUAL_ENV; .venv/bin/python -m pytest tests/ -q`
Expected: 121 baseline + 14 new = 135 passed.

- [ ] **Step 6: Commit**

```bash
git add src/publiplots/utils/legend_entries.py tests/test_legend_entries.py
git commit -m "feat(legend): LegendEntry + per-axes stash store"
```

---

## Task 2: `MultiAxesLegendGroup.claims` + figure registration

Adds the group-claim check and registers the group on the figure — no auto-collection yet. This lets future tasks gate per-axis rendering on group claims.

**Files:**
- Modify: `src/publiplots/utils/legend_group.py`
- Create: `tests/test_legend_group_auto_collect.py` (stub; fleshes out in later tasks)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_legend_group_auto_collect.py`:

```python
"""Tests for MultiAxesLegendGroup auto-collect (PR #81)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

import publiplots as pp


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# Claims + figure registration
# ---------------------------------------------------------------------------


def test_legend_group_registers_on_figure():
    fig, axes = pp.subplots(1, 2, axes_size=(40, 30))
    group = pp.legend_group(anchor=axes[-1])
    assert fig._publiplots_legend_group is group


def test_legend_group_claims_collect_none_returns_true():
    fig, axes = pp.subplots(1, 2, axes_size=(40, 30))
    group = pp.legend_group(anchor=axes[-1])
    assert group.claims("anything") is True
    assert group.claims("other") is True


def test_legend_group_claims_with_collect_list():
    fig, axes = pp.subplots(1, 2, axes_size=(40, 30))
    group = pp.legend_group(anchor=axes[-1], collect=["treatment"])
    assert group.claims("treatment") is True
    assert group.claims("dose") is False


def test_legend_group_rejects_bare_string_collect():
    fig, axes = pp.subplots(1, 2, axes_size=(40, 30))
    with pytest.raises(TypeError, match="list/tuple"):
        pp.legend_group(anchor=axes[-1], collect="treatment")
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `unset VIRTUAL_ENV; .venv/bin/python -m pytest tests/test_legend_group_auto_collect.py -v`
Expected: all 4 FAIL — either `TypeError: unexpected kwarg 'collect'` or `AttributeError: claims`.

- [ ] **Step 3: Modify `src/publiplots/utils/legend_group.py`**

Replace the existing `MultiAxesLegendGroup.__init__` and the module-level `legend_group` factory with this version (keep the rest of the file as-is):

```python
"""
Shared legend column across multiple subplots.

MultiAxesLegendGroup composes one unified legend column anchored to a
chosen axes even when individual legends/colorbars are attached to other
axes in the same figure. This is the primary tool for complex subplot
layouts.
"""

from typing import List, Optional, Sequence

from matplotlib.axes import Axes
from matplotlib.legend import Legend
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import Colorbar

from publiplots.utils.legend import LegendBuilder


class MultiAxesLegendGroup:
    """
    Unified legend column across multiple axes.

    All elements share a single mm-based layout anchored to ``anchor``. Each
    element can be attached to a different axes (via ``ax=`` on ``add_*``
    calls) for hit-testing and picking; its POSITION is always computed
    against the anchor's right edge regardless of which axes owns the artist.

    Parameters
    ----------
    anchor : Axes
        The axes whose right edge defines x=0 for the shared column.
    collect : list or tuple of str, optional
        Names of entries to auto-collect from across the grid's stashed
        ``LegendEntry`` objects. ``None`` (default) collects everything.
        A list filters and orders — e.g. ``collect=['treatment', 'dose']``
        renders only those two names, in that order.
    x_offset, y_offset, gap, column_spacing, vpad, max_width
        Same meaning as :class:`LegendBuilder` — all in millimeters.
    """

    def __init__(
        self,
        anchor: Axes,
        collect: Optional[Sequence[str]] = None,
        x_offset: float = 2,
        y_offset: Optional[float] = None,
        gap: float = 2,
        column_spacing: float = 5,
        vpad: float = 5,
        max_width: Optional[float] = None,
    ):
        self.anchor = anchor
        if collect is not None:
            if isinstance(collect, str) or not hasattr(collect, "__iter__"):
                raise TypeError(
                    "collect must be None or a list/tuple of names; "
                    "got a bare string. Wrap in a list: collect=['name']"
                )
            collect = list(collect)
        self._collect = collect
        # Track whether _materialize has already run (set by Task 3).
        self._materialized = False
        self._warned_mismatch = False
        # anchor_ax=anchor pins position math/reactor to the anchor regardless
        # of self.ax swaps during add_* calls.
        self._builder = LegendBuilder(
            ax=anchor,
            anchor_ax=anchor,
            x_offset=x_offset,
            y_offset=y_offset,
            gap=gap,
            column_spacing=column_spacing,
            vpad=vpad,
            max_width=max_width,
            external_to_axis=True,
        )
        # Register on the figure so plot functions can check claims.
        anchor.get_figure()._publiplots_legend_group = self

    def claims(self, name: str) -> bool:
        """True if the group will render an entry with this name."""
        if self._collect is None:
            return True
        return name in self._collect

    def add_legend(
        self,
        handles: List,
        label: str = "",
        *,
        ax: Optional[Axes] = None,
        **kwargs,
    ) -> Legend:
        """Add a legend to the shared column.

        The artist is attached to ax (defaults to anchor); position is
        always computed against the anchor's right edge.
        """
        target_ax = ax if ax is not None else self.anchor
        original_ax = self._builder.ax
        try:
            self._builder.ax = target_ax
            legend = self._builder.add_legend(handles=handles, label=label, **kwargs)
        finally:
            self._builder.ax = original_ax
        return legend

    def add_colorbar(
        self,
        mappable: Optional[ScalarMappable] = None,
        *,
        ax: Optional[Axes] = None,
        **kwargs,
    ) -> Colorbar:
        """Add a colorbar to the shared column. See add_legend for ax semantics."""
        target_ax = ax if ax is not None else self.anchor
        original_ax = self._builder.ax
        try:
            self._builder.ax = target_ax
            cbar = self._builder.add_colorbar(mappable=mappable, **kwargs)
        finally:
            self._builder.ax = original_ax
        return cbar


def legend_group(
    anchor: Axes,
    *,
    collect: Optional[Sequence[str]] = None,
    x_offset: float = 2,
    y_offset: Optional[float] = None,
    gap: float = 2,
    column_spacing: float = 5,
    vpad: float = 5,
    max_width: Optional[float] = None,
) -> MultiAxesLegendGroup:
    """Create a shared legend column anchored to ``anchor``.

    See :class:`MultiAxesLegendGroup` for parameter docs.
    """
    return MultiAxesLegendGroup(
        anchor=anchor,
        collect=collect,
        x_offset=x_offset,
        y_offset=y_offset,
        gap=gap,
        column_spacing=column_spacing,
        vpad=vpad,
        max_width=max_width,
    )
```

- [ ] **Step 4: Run the new tests and verify they pass**

Run: `unset VIRTUAL_ENV; .venv/bin/python -m pytest tests/test_legend_group_auto_collect.py -v`
Expected: 4 PASSED.

- [ ] **Step 5: Run the full suite**

Run: `unset VIRTUAL_ENV; .venv/bin/python -m pytest tests/ -q`
Expected: 135 + 4 = 139 passed.

- [ ] **Step 6: Commit**

```bash
git add src/publiplots/utils/legend_group.py tests/test_legend_group_auto_collect.py
git commit -m "feat(legend_group): collect kwarg + figure registration"
```

---

## Task 3: `_materialize()` — auto-collection pass

Walks the grid, dedups entries, filters by `collect=`, renders via the existing `add_legend` / `add_colorbar` dispatch.

**Files:**
- Modify: `src/publiplots/utils/legend_group.py`
- Modify: `tests/test_legend_group_auto_collect.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_legend_group_auto_collect.py`:

```python
# ---------------------------------------------------------------------------
# _materialize — auto-collection
# ---------------------------------------------------------------------------

from publiplots.utils.legend_entries import LegendEntry, stash_entry


def _stub_handle(color="red"):
    """Minimal object with a get_facecolor method for signature hashing."""
    class H:
        def __init__(self, c):
            self._c = c
        def get_facecolor(self):
            return self._c
    return H(color)


def _stash_on_axes(ax, name, kind, color="red", labels=("A",)):
    entry = LegendEntry.build(
        name=name, kind=kind,
        handles=[_stub_handle(color) for _ in labels],
        labels=labels,
    )
    stash_entry(ax, entry)


def test_materialize_dedups_identical_entries_across_axes():
    fig, axes = pp.subplots(1, 3, axes_size=(40, 30))
    _stash_on_axes(axes[0], "treatment", "hue")
    _stash_on_axes(axes[1], "treatment", "hue")
    _stash_on_axes(axes[2], "treatment", "hue")
    group = pp.legend_group(anchor=axes[-1])
    group._materialize()
    # Only one legend element added to the builder (not 3).
    legend_elements = [e for e in group._builder.elements if e[0] == "legend"]
    assert len(legend_elements) == 1


def test_materialize_collects_distinct_names_across_axes():
    fig, axes = pp.subplots(1, 2, axes_size=(40, 30))
    _stash_on_axes(axes[0], "treatment", "hue")
    _stash_on_axes(axes[1], "dose", "hue")
    group = pp.legend_group(anchor=axes[-1])
    group._materialize()
    legend_elements = [e for e in group._builder.elements if e[0] == "legend"]
    assert len(legend_elements) == 2


def test_materialize_with_collect_filter_drops_unlisted_names():
    fig, axes = pp.subplots(1, 2, axes_size=(40, 30))
    _stash_on_axes(axes[0], "treatment", "hue")
    _stash_on_axes(axes[1], "dose", "hue")
    group = pp.legend_group(anchor=axes[-1], collect=["treatment"])
    group._materialize()
    legend_elements = [e for e in group._builder.elements if e[0] == "legend"]
    assert len(legend_elements) == 1


def test_materialize_collect_list_preserves_order():
    fig, axes = pp.subplots(1, 2, axes_size=(40, 30))
    _stash_on_axes(axes[0], "dose", "hue")
    _stash_on_axes(axes[1], "treatment", "hue")
    # First-in-axes-order is dose; but collect says treatment first.
    group = pp.legend_group(anchor=axes[-1], collect=["treatment", "dose"])
    group._materialize()
    legend_elements = [e for e in group._builder.elements if e[0] == "legend"]
    assert len(legend_elements) == 2
    # Legend.get_title().get_text() gives the "label" of each legend.
    titles = [e[1].get_title().get_text() for e in legend_elements]
    assert titles == ["treatment", "dose"]


def test_materialize_is_idempotent():
    fig, axes = pp.subplots(1, 2, axes_size=(40, 30))
    _stash_on_axes(axes[0], "treatment", "hue")
    group = pp.legend_group(anchor=axes[-1])
    group._materialize()
    count_after_first = len(group._builder.elements)
    group._materialize()
    count_after_second = len(group._builder.elements)
    assert count_after_first == count_after_second


def test_materialize_mismatched_signature_warns_once():
    fig, axes = pp.subplots(1, 3, axes_size=(40, 30))
    _stash_on_axes(axes[0], "treatment", "hue", color="red")
    _stash_on_axes(axes[1], "treatment", "hue", color="blue")   # different palette
    _stash_on_axes(axes[2], "treatment", "hue", color="green")  # yet another
    group = pp.legend_group(anchor=axes[-1])
    with pytest.warns(UserWarning, match="differs between axes"):
        group._materialize()
    # Warn-once: calling _materialize again should not re-warn.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)  # fail if any warning fires
        # Reset the _materialized flag so it re-runs the collection
        group._materialized = False
        group._materialize()  # must not warn again


def test_materialize_no_publiplots_axes_is_noop():
    """Group on a plt.subplots figure (no _publiplots_axes) must not crash."""
    fig, ax = plt.subplots()
    group = pp.legend_group(anchor=ax)
    group._materialize()
    assert group._builder.elements == []


def test_materialize_manual_and_auto_coexist():
    from publiplots.utils.legend import create_legend_handles
    fig, axes = pp.subplots(1, 2, axes_size=(40, 30))
    _stash_on_axes(axes[0], "treatment", "hue")
    group = pp.legend_group(anchor=axes[-1])
    # Manual add BEFORE materialize
    group.add_legend(
        handles=create_legend_handles(labels=["x"], colors=["#000"], alpha=0.5, linewidth=1.0),
        label="manual",
    )
    # Now materialize
    group._materialize()
    legend_elements = [e for e in group._builder.elements if e[0] == "legend"]
    # One manual + one auto-collected treatment entry.
    assert len(legend_elements) == 2
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `unset VIRTUAL_ENV; .venv/bin/python -m pytest tests/test_legend_group_auto_collect.py -v -k "materialize"`
Expected: 8 FAIL with `AttributeError: _materialize`.

- [ ] **Step 3: Add `_materialize`, `_render_entry`, and warning helpers to `src/publiplots/utils/legend_group.py`**

Add these imports near the top of the file (alongside existing imports):

```python
import warnings

from publiplots.utils.legend_entries import (
    LegendEntry,
    get_entries,
    is_continuous_hue,
)
```

Then add these methods to `MultiAxesLegendGroup` (after `claims` and before `add_legend`):

```python
    def _materialize(self) -> None:
        """Collect stashed entries from every grid axes and render them.

        Called by SubplotsAutoLayout during the settle pass (Task 6).
        Idempotent — subsequent calls after the first return immediately.
        """
        if self._materialized:
            return
        self._materialized = True

        fig = self.anchor.get_figure()
        axes_matrix = getattr(fig, "_publiplots_axes", None)
        if axes_matrix is None:
            return

        seen = {}  # (name, kind) -> LegendEntry
        order = []  # list of (name, kind) in collection order
        for row in axes_matrix:
            for ax in row:
                for entry in get_entries(ax):
                    if self._collect is not None and entry.name not in self._collect:
                        continue
                    key = (entry.name, entry.kind)
                    if key in seen:
                        if seen[key].signature != entry.signature and not self._warned_mismatch:
                            warnings.warn(
                                f"legend entry {entry.name!r} ({entry.kind}) "
                                "differs between axes; group uses first occurrence",
                                UserWarning,
                                stacklevel=2,
                            )
                            self._warned_mismatch = True
                        continue
                    seen[key] = entry
                    order.append(key)

        if self._collect is not None:
            # Stable sort by the user's collect order; ties (same name with
            # different kinds) stay in the order they were encountered.
            order.sort(key=lambda k: (self._collect.index(k[0]), 0))

        for key in order:
            self._render_entry(seen[key])

    def _render_entry(self, entry: LegendEntry) -> None:
        """Route to add_legend (categorical) or add_colorbar (continuous)."""
        if entry.kind == "hue" and is_continuous_hue(entry.handles):
            mappable = entry.handles[0]
            self.add_colorbar(mappable=mappable, label=entry.name)
        else:
            self.add_legend(
                handles=list(entry.handles),
                label=entry.name,
            )
```

Note: `add_legend` does not currently take a `labels=` kwarg — it derives labels from the handles. Leave as-is for the auto-collect path; dedup ensures the correct handles are passed through.

- [ ] **Step 4: Run the tests and verify they pass**

Run: `unset VIRTUAL_ENV; .venv/bin/python -m pytest tests/test_legend_group_auto_collect.py -v -k "materialize"`
Expected: 8 PASSED.

- [ ] **Step 5: Run the full suite**

Run: `unset VIRTUAL_ENV; .venv/bin/python -m pytest tests/ -q`
Expected: 139 + 8 = 147 passed.

- [ ] **Step 6: Commit**

```bash
git add src/publiplots/utils/legend_group.py tests/test_legend_group_auto_collect.py
git commit -m "feat(legend_group): _materialize() auto-collection pass"
```

---

## Task 4: `pp.subplots()` — drop `legend_column` kwarg

Makes `legend_column` auto-only. `FigureLayout.legend_column` stays as an internal data field, populated by `SubplotsAutoLayout` (next task). `pp.subplots()` raises if the user tries to pass the kwarg.

**Files:**
- Modify: `src/publiplots/layout/subplots.py`
- Modify: `tests/test_subplots.py`

- [ ] **Step 1: Write the failing test and delete the obsolete one**

Delete `test_subplots_legend_column_reserves_extra_width` from `tests/test_subplots.py` (grep for that name; the whole function goes). Replace with (or append below the existing `pp.subplots` API tests):

```python
def test_subplots_rejects_legend_column_kwarg():
    with pytest.raises(TypeError, match="legend_column"):
        pp.subplots(axes_size=(50, 30), legend_column=30)
```

- [ ] **Step 2: Run the tests and verify the new one fails**

Run: `unset VIRTUAL_ENV; .venv/bin/python -m pytest tests/test_subplots.py::test_subplots_rejects_legend_column_kwarg -v`
Expected: FAIL. (The signature still accepts `legend_column`.)

- [ ] **Step 3: Modify `src/publiplots/layout/subplots.py`**

Find the `subplots()` signature around lines 28-44 and remove the `legend_column: float = 0.0,` parameter. The new signature must be:

```python
def subplots(
    nrows: int = 1,
    ncols: int = 1,
    *,
    axes_size: Union[Tuple[float, float], float, None] = None,
    sharex: Union[bool, str] = False,
    sharey: Union[bool, str] = False,
    title_space: Optional[float] = None,
    xlabel_space: Optional[float] = None,
    ylabel_space: Optional[float] = None,
    right: Optional[float] = None,
    hspace: Optional[float] = None,
    wspace: Optional[float] = None,
    outer_pad: Optional[float] = None,
    **fig_kw,
):
```

Add the rejection check — find the block where `figsize` is rejected (around `if "figsize" in fig_kw:`) and append:

```python
    if "legend_column" in fig_kw:
        raise TypeError(
            "pp.subplots() no longer accepts legend_column. Attach a "
            "pp.legend_group(anchor=...) to the figure; the column is "
            "auto-sized based on the rendered group width."
        )
```

Find the `FigureLayout(...)` construction call and change `legend_column=float(legend_column),` to the literal `legend_column=0.0,`. (The auto-layout hook will update it on first draw — Task 5 wires that.)

Remove the `legend_column` entries from the user_values dict / resolved dict handling (they should no longer be referenced in this function).

Remove the `if legend_column < 0:` validation block — no more user input to validate.

- [ ] **Step 4: Run the test and verify it passes**

Run: `unset VIRTUAL_ENV; .venv/bin/python -m pytest tests/test_subplots.py::test_subplots_rejects_legend_column_kwarg -v`
Expected: PASS.

- [ ] **Step 5: Run the full suite**

Run: `unset VIRTUAL_ENV; .venv/bin/python -m pytest tests/ -q`
Expected: 147 + 1 (new) - 1 (deleted) = 147 passed. Scan the output carefully — any test that passed `legend_column=` to `pp.subplots` will now fail. Those tests must be updated (remove the kwarg; `pp.subplots` still creates a figure sized identically because `legend_column` was previously user-supplied and is now always 0 until Task 5 runs).

If there are other existing tests that pass `legend_column=` — update them inline as part of this task. Expected affected test: `test_subplots_legend_column_reserves_extra_width` (delete per step 1) and possibly any that use `legend_column` on a real figure — check via grep:

Run: `grep -rn "legend_column" tests/`

Update or remove any matches. Re-run the full suite until green.

- [ ] **Step 6: Commit**

```bash
git add src/publiplots/layout/subplots.py tests/test_subplots.py
git commit -m "feat(layout): drop legend_column kwarg from pp.subplots"
```

---

## Task 5: `SubplotsAutoLayout._measure_legend_column`

Adds width-awareness. Calls `group._materialize()` (idempotent) from within the measurement so the first settle pass creates artists to measure.

**Files:**
- Modify: `src/publiplots/layout/auto_layout.py`
- Modify: `tests/test_subplots.py` (append width-awareness tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_subplots.py` (after the existing SubplotsAutoLayout tests):

```python
# ---------------------------------------------------------------------------
# Width-awareness: legend_column auto-sizing
# ---------------------------------------------------------------------------

from publiplots.utils.legend_entries import LegendEntry, stash_entry


def _stub_handle(color="red"):
    class H:
        def __init__(self, c):
            self._c = c
        def get_facecolor(self):
            return self._c
    return H(color)


def test_legend_column_is_zero_without_group():
    fig, ax = pp.subplots(axes_size=(50, 30))
    # No pp.legend_group call.
    fig.canvas.draw()
    assert fig._publiplots_layout.legend_column == 0.0


def test_legend_column_auto_grows_with_group():
    from publiplots.utils.legend import create_legend_handles
    fig, axes = pp.subplots(1, 2, axes_size=(40, 30))
    # Stash so auto-collect picks something up
    stash_entry(
        axes[0],
        LegendEntry.build(
            "g", "hue",
            handles=create_legend_handles(labels=["A", "B"], colors=["#000", "#fff"],
                                          alpha=0.5, linewidth=1.0),
            labels=("A", "B"),
        ),
    )
    pp.legend_group(anchor=axes[-1])
    fig.savefig("/tmp/test_legend_column_auto.png")
    layout = fig._publiplots_layout
    assert layout.legend_column > 5.0, (
        f"legend_column should have grown; got {layout.legend_column}"
    )


def test_legend_column_stays_small_with_empty_group():
    """No stashed entries and no manual adds -> group has no artists ->
    legend_column stays near zero."""
    fig, axes = pp.subplots(1, 2, axes_size=(40, 30))
    pp.legend_group(anchor=axes[-1])
    # No stashing, no manual add_legend calls.
    fig.savefig("/tmp/test_legend_column_empty.png")
    layout = fig._publiplots_layout
    assert layout.legend_column < 2.0, (
        f"legend_column should stay near 0; got {layout.legend_column}"
    )
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `unset VIRTUAL_ENV; .venv/bin/python -m pytest tests/test_subplots.py -v -k "legend_column"`
Expected: the two "with group" tests FAIL — `auto_grows_with_group` because `legend_column` is never updated, `stays_small_with_empty_group` probably already passes but keep it for the contract. `is_zero_without_group` passes.

- [ ] **Step 3: Modify `src/publiplots/layout/auto_layout.py`**

Add `legend_column` to the auto-measurable set:

```python
_ALL_SIDES = {"title_space", "xlabel_space", "ylabel_space", "right", "legend_column"}
```

In `_measure()`, after the existing per-side tuple measurement loop (and before the `return measured` line), add:

```python
        if "legend_column" not in self._locked:
            measured["legend_column"] = self._measure_legend_column()
```

In `_needs_update()`, replace the existing tuple-only comparison body with a branching version:

```python
    def _needs_update(self, measured: Dict[str, ...]) -> bool:
        for side, new_val in measured.items():
            current = getattr(self._layout, side)
            if side == "legend_column":
                if abs(new_val - current) >= _UPDATE_THRESHOLD_MM:
                    return True
            else:
                # tuple comparison (per-row / per-col reservations)
                if len(new_val) != len(current):
                    return True
                for nv, cv in zip(new_val, current):
                    if abs(nv - cv) >= _UPDATE_THRESHOLD_MM:
                        return True
        return False
```

Add these two new methods to the class (after `_needs_update` and before `_apply` or wherever fits cleanly):

```python
    def _measure_legend_column(self) -> float:
        """mm width past the anchor axes' right edge, plus 1 mm padding."""
        group = getattr(self._fig, "_publiplots_legend_group", None)
        if group is None:
            return 0.0

        # Force collection so artists exist to measure.
        group._materialize()
        if not group._builder.elements:
            return 0.0

        dpi = self._fig.dpi
        if dpi <= 0:
            return 0.0

        anchor_bb = group.anchor.get_window_extent()
        max_x1 = anchor_bb.x1
        for _, obj in group._builder.elements:
            extent = self._artist_window_extent(obj)
            if extent is None:
                continue
            max_x1 = max(max_x1, extent.x1)
        overhang_px = max_x1 - anchor_bb.x1
        if overhang_px <= 0:
            return 0.0
        return overhang_px / dpi * 25.4 + 1.0

    def _artist_window_extent(self, obj):
        """Duck-typed window-extent accessor (Legend/Colorbar/Text)."""
        if hasattr(obj, "get_window_extent"):
            try:
                return obj.get_window_extent()
            except Exception:
                pass
        if hasattr(obj, "ax"):  # Colorbar stores geometry on .ax
            return obj.ax.get_window_extent()
        return None
```

- [ ] **Step 4: Run the tests and verify they pass**

Run: `unset VIRTUAL_ENV; .venv/bin/python -m pytest tests/test_subplots.py -v -k "legend_column"`
Expected: 3 PASSED.

- [ ] **Step 5: Run the full suite**

Run: `unset VIRTUAL_ENV; .venv/bin/python -m pytest tests/ -q`
Expected: 147 + 3 = 150 passed.

- [ ] **Step 6: Commit**

```bash
git add src/publiplots/layout/auto_layout.py tests/test_subplots.py
git commit -m "feat(layout): SubplotsAutoLayout measures legend_group width"
```

---

## Task 6: Rewrite `pp.legend(ax)` auto-mode to use the new store

Replaces the legacy `_get_legend_data` reader with one that reads `ax._publiplots_legend_entries`. For backward compat with axes that have the legacy `_legend_data` attribute (e.g. third-party extensions), fall back.

**Files:**
- Modify: `src/publiplots/utils/legend.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_legend_entries.py`:

```python
def test_pp_legend_auto_reads_new_store():
    """pp.legend(ax) in auto mode should render from stashed LegendEntry."""
    import matplotlib.pyplot as plt
    import publiplots as pp
    from publiplots.utils.legend import create_legend_handles

    fig, ax = pp.subplots(axes_size=(60, 40))
    ax.scatter([0, 1, 2], [0, 1, 0])
    stash_entry(
        ax,
        LegendEntry.build(
            "group", "hue",
            handles=create_legend_handles(
                labels=["A", "B"], colors=["#111", "#222"],
                alpha=0.5, linewidth=1.0,
            ),
            labels=("A", "B"),
        ),
    )
    builder = pp.legend(ax)  # auto mode
    fig.canvas.draw()
    # At least one legend artist should exist on the axes now.
    from matplotlib.legend import Legend
    legends = [c for c in ax.get_children() if isinstance(c, Legend)]
    assert len(legends) >= 1
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `unset VIRTUAL_ENV; .venv/bin/python -m pytest tests/test_legend_entries.py -v -k "pp_legend_auto_reads_new_store"`
Expected: FAIL — `pp.legend` reads the old store, which is absent here.

- [ ] **Step 3: Update `src/publiplots/utils/legend.py`**

Add to the top of the file (near other `publiplots.utils` imports):

```python
from publiplots.utils.legend_entries import (
    LegendEntry,
    get_entries,
    is_continuous_hue,
)
```

Find the existing auto-mode block in the `legend()` function (around lines 1451-1472, starting with `if auto:`). Replace with:

```python
    # Auto mode
    if auto:
        # Prefer the new LegendEntry store.
        entries = get_entries(ax)
        if entries:
            seen = set()  # (name, kind) dedup within one axes
            for entry in entries:
                key = (entry.name, entry.kind)
                if key in seen:
                    continue
                seen.add(key)
                if entry.kind == "hue" and is_continuous_hue(entry.handles):
                    builder.add_colorbar(
                        mappable=entry.handles[0],
                        label=entry.name,
                    )
                else:
                    builder.add_legend(
                        handles=list(entry.handles),
                        label=entry.name,
                        **kwargs,
                    )
        else:
            # Back-compat: read legacy _legend_data dict store (plot
            # functions not migrated in PR #81 still use this path).
            legend_data = _get_legend_data(ax)
            if legend_data:
                if 'hue' in legend_data:
                    hue_data = legend_data['hue'].copy()
                    if hue_data.get('type') == 'colorbar':
                        hue_data.pop('type', None)
                        builder.add_colorbar(**hue_data)
                    else:
                        builder.add_legend(**hue_data, **kwargs)
                if 'size' in legend_data:
                    size_data = legend_data['size'].copy()
                    builder.add_legend(**size_data, **kwargs)
                if 'style' in legend_data:
                    style_data = legend_data['style'].copy()
                    builder.add_legend(**style_data, **kwargs)

    return builder
```

- [ ] **Step 4: Run the test and verify it passes**

Run: `unset VIRTUAL_ENV; .venv/bin/python -m pytest tests/test_legend_entries.py -v -k "pp_legend_auto_reads_new_store"`
Expected: PASS.

- [ ] **Step 5: Run the full suite**

Run: `unset VIRTUAL_ENV; .venv/bin/python -m pytest tests/ -q`
Expected: 150 + 1 = 151 passed. The legacy-store fallback path keeps existing scatter/strip/swarm/point tests passing until Tasks 7-10 migrate them.

- [ ] **Step 6: Commit**

```bash
git add src/publiplots/utils/legend.py tests/test_legend_entries.py
git commit -m "feat(legend): auto-mode reads LegendEntry store with legacy fallback"
```

---

## Task 7: Migrate `scatter.py` to stash `LegendEntry`

Replaces the current `legend_data` dict assignment with `stash_entry` calls per kind. Also threads the new `legend=` kwarg semantics: accepts `bool | dict`, routes through `resolve_legend_flags`, suppresses per-axis rendering for entries claimed by a `legend_group`.

**Files:**
- Modify: `src/publiplots/plot/scatter.py`
- Create: `tests/test_scatter_legend_stash.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_scatter_legend_stash.py`:

```python
"""Tests for scatterplot legend stashing via LegendEntry."""
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


def _scatter_df(seed=0):
    rng = np.random.default_rng(seed)
    n = 30
    return pd.DataFrame({
        "x": rng.normal(size=n),
        "y": rng.normal(size=n),
        "g": rng.choice(["A", "B", "C"], size=n),
        "m": rng.uniform(1, 5, size=n),
    })


def test_scatterplot_stashes_hue_entry():
    df = _scatter_df()
    fig, ax = pp.scatterplot(data=df, x="x", y="y", hue="g", palette="pastel")
    entries = get_entries(ax)
    names_kinds = [(e.name, e.kind) for e in entries]
    assert ("g", "hue") in names_kinds


def test_scatterplot_stashes_size_entry():
    df = _scatter_df()
    fig, ax = pp.scatterplot(data=df, x="x", y="y", size="m")
    entries = get_entries(ax)
    names_kinds = [(e.name, e.kind) for e in entries]
    assert ("m", "size") in names_kinds


def test_scatterplot_legend_dict_suppresses_hue():
    df = _scatter_df()
    fig, ax = pp.scatterplot(
        data=df, x="x", y="y", hue="g", size="m",
        palette="pastel",
        legend={"hue": False},
    )
    entries = get_entries(ax)
    names = [e.name for e in entries]
    # hue is suppressed, size is still stashed
    assert "g" not in names
    assert "m" in names


def test_scatterplot_legend_false_stashes_nothing():
    df = _scatter_df()
    fig, ax = pp.scatterplot(
        data=df, x="x", y="y", hue="g", size="m",
        palette="pastel",
        legend=False,
    )
    assert get_entries(ax) == []


def test_scatterplot_in_group_suppresses_per_axis_render():
    """With a legend_group active, the scatter should NOT attach a per-axis
    Legend artist to ax (the group handles it)."""
    from matplotlib.legend import Legend
    df = _scatter_df()
    fig, axes = pp.subplots(1, 2, axes_size=(50, 40))
    pp.legend_group(anchor=axes[-1])
    pp.scatterplot(data=df, x="x", y="y", hue="g", palette="pastel", ax=axes[0])
    fig.canvas.draw()
    per_axis_legends = [c for c in axes[0].get_children() if isinstance(c, Legend)]
    assert per_axis_legends == []


def test_scatterplot_no_group_renders_per_axis_legend():
    from matplotlib.legend import Legend
    df = _scatter_df()
    fig, ax = pp.scatterplot(data=df, x="x", y="y", hue="g", palette="pastel")
    fig.canvas.draw()
    per_axis_legends = [c for c in ax.get_children() if isinstance(c, Legend)]
    assert len(per_axis_legends) >= 1
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `unset VIRTUAL_ENV; .venv/bin/python -m pytest tests/test_scatter_legend_stash.py -v`
Expected: most FAIL — current scatter uses the legacy `_legend_data` dict, not the new store; `legend=dict(...)` probably crashes.

- [ ] **Step 3: Update `src/publiplots/plot/scatter.py`**

Add to the imports at the top of the file:

```python
from publiplots.utils.legend_entries import (
    LegendEntry,
    stash_entry,
    get_entries,
    resolve_legend_flags,
    entry_is_in_group,
    is_continuous_hue,
)
```

Find the `_legend()` helper at the bottom of scatter.py (around line 466+ `def _legend(`). Locate the block where `legend_data = {}` is built and the per-kind "hue"/"size"/"style" sections populate it. Rewrite that section to stash `LegendEntry` objects instead:

```python
    # `legend` is bool | dict from the user; translate to per-kind flags.
    flags = resolve_legend_flags(legend)

    # HUE
    if hue is not None and flags["hue"]:
        hue_label = kwargs.pop("hue_label", hue)
        if isinstance(palette, dict):  # categorical
            hue_handles = create_legend_handles(
                labels=list(palette.keys()),
                colors=list(palette.values()),
                edgecolors=[edgecolor] * len(palette) if edgecolor else None,
                **handle_kwargs,
            )
            stash_entry(
                ax,
                LegendEntry.build(
                    name=hue_label,
                    kind="hue",
                    handles=hue_handles,
                    labels=list(palette.keys()),
                ),
            )
        else:  # continuous -> colorbar
            mappable = ScalarMappable(norm=hue_norm, cmap=palette)
            stash_entry(
                ax,
                LegendEntry.build(
                    name=hue_label,
                    kind="hue",
                    handles=[mappable],
                    labels=[],  # continuous
                ),
            )

    # SIZE
    if size is not None and flags["size"]:
        tick_color = color if hue is None else "gray"
        size_handle_kwargs = handle_kwargs.copy()
        size_handle_kwargs["color"] = tick_color
        tick_labels, tick_sizes = _get_size_ticks(
            values=data[size].dropna().values,
            sizes=sizes,
            size_norm=size_norm,
            nbins=kwargs.pop("size_nbins", 4),
            min_n_ticks=kwargs.pop("size_min_n_ticks", 3),
            include_min_max=kwargs.pop("size_include_min_max", False),
        )
        size_handles = create_legend_handles(
            labels=tick_labels,
            sizes=tick_sizes,
            **size_handle_kwargs,
        )
        size_label = kwargs.pop("size_label", size)
        stash_entry(
            ax,
            LegendEntry.build(
                name=size_label,
                kind="size",
                handles=size_handles,
                labels=tick_labels,
            ),
        )

    # STYLE
    if style is not None and flags["style"]:
        style_values = data[style].unique()
        style_label = kwargs.pop("style_label", style)
        marker_param = markers if isinstance(markers, (list, dict)) else None
        marker_map = resolve_marker_map(
            values=list(style_values),
            marker_map=marker_param,
        )
        style_color = color if hue is None else "gray"
        style_handle_kwargs = handle_kwargs.copy()
        style_handle_kwargs["color"] = style_color
        style_handle_kwargs.pop("style", None)
        style_labels = [str(val) for val in style_values]
        style_handles = create_legend_handles(
            labels=style_labels,
            markers=[marker_map[val] for val in style_values],
            edgecolors=[edgecolor] * len(style_values) if edgecolor else None,
            **style_handle_kwargs,
        )
        stash_entry(
            ax,
            LegendEntry.build(
                name=style_label,
                kind="style",
                handles=style_handles,
                labels=style_labels,
            ),
        )
```

Delete the lines that set `ax.collections[0]._legend_data = legend_data` — no longer needed.

Now the per-axis rendering. Replace the `if create_legend:` block at the bottom of `_legend()` with:

```python
    if create_legend:
        # Suppress per-axis rendering for entries owned by a legend_group
        fig = ax.get_figure()
        entries_to_render = [
            e for e in get_entries(ax)
            if flags[e.kind] and not entry_is_in_group(fig, e)
        ]
        if not entries_to_render:
            return

        builder = legend(ax=ax, auto=False)
        for entry in entries_to_render:
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

Important: the scatterplot function also calls `_legend` conditionally on `legend` being truthy. The outer caller must now pass `legend` through unchanged (including `legend=dict(...)`), since `_legend` does its own resolution. Find the call site that invokes `_legend(...)`:

```bash
grep -n "_legend(" src/publiplots/plot/scatter.py | grep -v "def "
```

Change the call site from something like `if legend: _legend(..., legend=True, ...)` to simply pass the user's `legend` value through. (In scatter's current code, `_legend` takes a `create_legend` kwarg; keep that as True when the function wants to render per-axis, otherwise False. The `legend` kwarg is what drives per-kind flags; `create_legend` gates rendering.)

The shape of the outer call should end up like:

```python
# Outer call (at the site where scatter decides to run _legend)
_legend(
    data=data, x=x, y=y, hue=hue, size=size, style=style, ax=ax,
    palette=palette, hue_norm=hue_norm, size_norm=size_norm, sizes=sizes,
    markers=markers, color=color, edgecolor=edgecolor, alpha=alpha,
    linewidth=linewidth, create_legend=True,
    legend=legend,   # pass the user's bool | dict through
    **legend_kws,
)
```

- [ ] **Step 4: Run the tests and verify they pass**

Run: `unset VIRTUAL_ENV; .venv/bin/python -m pytest tests/test_scatter_legend_stash.py -v`
Expected: 6 PASSED.

- [ ] **Step 5: Run the full suite**

Run: `unset VIRTUAL_ENV; .venv/bin/python -m pytest tests/ -q`
Expected: 151 + 6 = 157 passed. Watch for existing scatter tests that may break — the legacy `_legend_data` attribute is gone, so any test that reads `ax.collections[0]._legend_data` must be updated to use `get_entries(ax)`.

Check for such tests:
```bash
grep -rn "_legend_data" tests/
```

Update any hits to use `get_entries(ax)`.

- [ ] **Step 6: Commit**

```bash
git add src/publiplots/plot/scatter.py tests/test_scatter_legend_stash.py
git commit -m "feat(scatter): migrate to LegendEntry stash + group-aware render"
```

---

## Task 8: Migrate `strip.py`

Same pattern as scatter but simpler — strip only has `hue`.

**Files:**
- Modify: `src/publiplots/plot/strip.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_scatter_legend_stash.py`:

```python
# ---------------------------------------------------------------------------
# stripplot migration
# ---------------------------------------------------------------------------


def test_stripplot_stashes_hue_entry():
    df = _scatter_df()
    fig, ax = pp.stripplot(data=df, x="g", y="y", hue="g", palette="pastel")
    entries = get_entries(ax)
    names_kinds = [(e.name, e.kind) for e in entries]
    assert ("g", "hue") in names_kinds


def test_stripplot_legend_false_stashes_nothing():
    df = _scatter_df()
    fig, ax = pp.stripplot(
        data=df, x="g", y="y", hue="g", palette="pastel", legend=False,
    )
    assert get_entries(ax) == []
```

- [ ] **Step 2: Run and verify they fail**

Run: `unset VIRTUAL_ENV; .venv/bin/python -m pytest tests/test_scatter_legend_stash.py -v -k "strip"`
Expected: 2 FAIL.

- [ ] **Step 3: Update `src/publiplots/plot/strip.py`**

Find the block where `legend_data` is built (around line 242). Current shape:

```python
hue_handles = create_legend_handles(...)
legend_data["hue"] = {"handles": hue_handles, ...}
# ...
ax.collections[0]._legend_data = legend_data
```

Replace with:

```python
from publiplots.utils.legend_entries import (
    LegendEntry,
    stash_entry,
    get_entries,
    resolve_legend_flags,
    entry_is_in_group,
)

# ... inside the plot function, at the legend section ...
flags = resolve_legend_flags(legend)
if hue is not None and flags["hue"]:
    hue_label = kwargs.pop("hue_label", hue) if isinstance(kwargs, dict) else hue
    hue_handles = create_legend_handles(
        labels=list(palette.keys()) if isinstance(palette, dict) else list(hue_levels),
        colors=(list(palette.values()) if isinstance(palette, dict)
                else color_palette(palette, len(hue_levels))),
        # keep existing handle_kwargs as strip currently computes them
        **handle_kwargs,
    )
    stash_entry(
        ax,
        LegendEntry.build(
            name=hue_label,
            kind="hue",
            handles=hue_handles,
            labels=list(palette.keys()) if isinstance(palette, dict) else list(hue_levels),
        ),
    )

# Remove the old `ax.collections[0]._legend_data = legend_data` line.

# Per-axis render:
if legend:
    fig = ax.get_figure()
    entries_to_render = [
        e for e in get_entries(ax)
        if flags[e.kind] and not entry_is_in_group(fig, e)
    ]
    if entries_to_render:
        from publiplots.utils.legend import legend as pp_legend
        builder = pp_legend(ax=ax, auto=False)
        for entry in entries_to_render:
            builder.add_legend(
                handles=list(entry.handles),
                label=entry.name,
            )
```

Note: strip already has `hue_levels` and `palette` resolution — preserve them. The edit above is shape-preserving: replace the dict-write with `stash_entry`, and the `ax.collections[0]._legend_data = legend_data` line with the per-axis render gated on `entry_is_in_group`.

Read the current strip.py file carefully before editing. The existing variable names (`hue_levels`, `handle_kwargs`, etc.) should be preserved; only the stash mechanism changes.

- [ ] **Step 4: Run tests and verify they pass**

Run: `unset VIRTUAL_ENV; .venv/bin/python -m pytest tests/test_scatter_legend_stash.py -v -k "strip"`
Expected: 2 PASSED.

- [ ] **Step 5: Run the full suite**

Run: `unset VIRTUAL_ENV; .venv/bin/python -m pytest tests/ -q`
Expected: 157 + 2 = 159 passed.

- [ ] **Step 6: Commit**

```bash
git add src/publiplots/plot/strip.py tests/test_scatter_legend_stash.py
git commit -m "feat(strip): migrate to LegendEntry stash + group-aware render"
```

---

## Task 9: Migrate `swarm.py`

Same pattern as strip — swarm only has `hue`.

**Files:**
- Modify: `src/publiplots/plot/swarm.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_scatter_legend_stash.py`:

```python
# ---------------------------------------------------------------------------
# swarmplot migration
# ---------------------------------------------------------------------------


def test_swarmplot_stashes_hue_entry():
    df = _scatter_df()
    fig, ax = pp.swarmplot(data=df, x="g", y="y", hue="g", palette="pastel")
    entries = get_entries(ax)
    names_kinds = [(e.name, e.kind) for e in entries]
    assert ("g", "hue") in names_kinds


def test_swarmplot_legend_false_stashes_nothing():
    df = _scatter_df()
    fig, ax = pp.swarmplot(
        data=df, x="g", y="y", hue="g", palette="pastel", legend=False,
    )
    assert get_entries(ax) == []
```

- [ ] **Step 2: Run and verify they fail**

Run: `unset VIRTUAL_ENV; .venv/bin/python -m pytest tests/test_scatter_legend_stash.py -v -k "swarm"`
Expected: 2 FAIL.

- [ ] **Step 3: Update `src/publiplots/plot/swarm.py`**

Same migration pattern as strip (Task 8, Step 3): replace the dict-write + legacy stash with `stash_entry(ax, LegendEntry.build(...))`, and gate per-axis rendering on `entry_is_in_group`. Read `src/publiplots/plot/swarm.py` around line 238 for the current code.

- [ ] **Step 4: Run tests and verify they pass**

Run: `unset VIRTUAL_ENV; .venv/bin/python -m pytest tests/test_scatter_legend_stash.py -v -k "swarm"`
Expected: 2 PASSED.

- [ ] **Step 5: Run the full suite**

Run: `unset VIRTUAL_ENV; .venv/bin/python -m pytest tests/ -q`
Expected: 159 + 2 = 161 passed.

- [ ] **Step 6: Commit**

```bash
git add src/publiplots/plot/swarm.py tests/test_scatter_legend_stash.py
git commit -m "feat(swarm): migrate to LegendEntry stash + group-aware render"
```

---

## Task 10: Migrate `point.py`

pointplot stashes on `ax.lines[0]._legend_data`. Same `hue`-only kind.

**Files:**
- Modify: `src/publiplots/plot/point.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_scatter_legend_stash.py`:

```python
# ---------------------------------------------------------------------------
# pointplot migration
# ---------------------------------------------------------------------------


def test_pointplot_stashes_hue_entry():
    df = _scatter_df()
    fig, ax = pp.pointplot(data=df, x="g", y="y", hue="g", palette="pastel")
    entries = get_entries(ax)
    names_kinds = [(e.name, e.kind) for e in entries]
    assert ("g", "hue") in names_kinds


def test_pointplot_legend_false_stashes_nothing():
    df = _scatter_df()
    fig, ax = pp.pointplot(
        data=df, x="g", y="y", hue="g", palette="pastel", legend=False,
    )
    assert get_entries(ax) == []
```

- [ ] **Step 2: Run and verify they fail**

Run: `unset VIRTUAL_ENV; .venv/bin/python -m pytest tests/test_scatter_legend_stash.py -v -k "pointplot"`
Expected: 2 FAIL.

- [ ] **Step 3: Update `src/publiplots/plot/point.py`**

Same migration pattern as strip/swarm (Tasks 8-9, Step 3). Read the current file around line 484 for context. The stash attribute today is `ax.lines[0]._legend_data`; the migration target is `ax._publiplots_legend_entries` via `stash_entry`.

- [ ] **Step 4: Run tests and verify they pass**

Run: `unset VIRTUAL_ENV; .venv/bin/python -m pytest tests/test_scatter_legend_stash.py -v -k "pointplot"`
Expected: 2 PASSED.

- [ ] **Step 5: Run the full suite**

Run: `unset VIRTUAL_ENV; .venv/bin/python -m pytest tests/ -q`
Expected: 161 + 2 = 163 passed.

- [ ] **Step 6: Commit**

```bash
git add src/publiplots/plot/point.py tests/test_scatter_legend_stash.py
git commit -m "feat(point): migrate to LegendEntry stash + group-aware render"
```

---

## Task 11: Migrate gallery `plot_14`

Drops the `legend_column=30` kwarg from the 1×3 section (which uses migrated plot types) and drops the explicit `group.add_legend(...)` call — auto-collect handles it. The 1×2 raincloud section keeps its explicit `group.add_legend(...)` because raincloud is NOT migrated in this PR.

**Files:**
- Modify: `examples/plots/plot_14_edgecolor_control.py`

- [ ] **Step 1: Read the current file state**

Run:
```bash
grep -n "legend_column\|legend_group\|group.add_legend" examples/plots/plot_14_edgecolor_control.py
```

- [ ] **Step 2: Migrate the 1×3 Uniform Edges section**

In the "Uniform Edges Across Plot Types" section, two edits:

1. Change `fig, axes = pp.subplots(1, 3, axes_size=(45, 30), legend_column=30)` to `fig, axes = pp.subplots(1, 3, axes_size=(45, 30))`.

2. Replace the block:
```python
group = pp.legend_group(anchor=axes[-1])
group.add_legend(
    handles=pp.create_legend_handles(
        labels=['A', 'B', 'C'],
        colors=list(pp.color_palette('pastel', 3)),
        alpha=pp.rcParams['alpha'],
        linewidth=pp.rcParams['lines.linewidth'],
        edgecolors=pp.rcParams['edgecolor'],
    ),
    label='group',
)
```

**BUT**: in this section the plot functions used are `pp.barplot`, `pp.boxplot`, `pp.scatterplot`. Only scatterplot is migrated in this PR; bar and box are NOT. If we drop the explicit `group.add_legend` and replace with bare `pp.legend_group(anchor=axes[-1])`, the group will only pick up the scatterplot entry — the bar and box entries are missing.

So the correct migration for THIS section in THIS PR:

```python
# legend_column removed (auto-sized now)
fig, axes = pp.subplots(1, 3, axes_size=(45, 30))
# ... existing barplot/boxplot/scatterplot calls unchanged ...

# Keep the explicit add_legend for now — raincloud / bar / box don't
# stash yet (follow-up PR). Once they migrate, this can become a bare
# pp.legend_group(anchor=axes[-1]).
group = pp.legend_group(anchor=axes[-1])
group.add_legend(
    handles=pp.create_legend_handles(
        labels=['A', 'B', 'C'],
        colors=list(pp.color_palette('pastel', 3)),
        alpha=pp.rcParams['alpha'],
        linewidth=pp.rcParams['lines.linewidth'],
        edgecolors=pp.rcParams['edgecolor'],
    ),
    label='group',
)
```

The ONE change in the 1×3 section: drop `legend_column=30` from `pp.subplots`. Keep the explicit `add_legend` until the follow-up PRs migrate bar + box.

- [ ] **Step 3: Migrate the 1×2 raincloud section**

Same single change: drop `legend_column=30` from the `pp.subplots(1, 2, axes_size=(55, 50), sharey=True, legend_column=30)` → `pp.subplots(1, 2, axes_size=(55, 50), sharey=True)`.

Keep the explicit `group.add_legend(...)` call — raincloud is not migrated.

- [ ] **Step 4: Smoke-run the example**

Run: `unset VIRTUAL_ENV; .venv/bin/python examples/plots/plot_14_edgecolor_control.py 2>&1 | tail -5`
Expected: exits 0, no traceback.

- [ ] **Step 5: Run the full suite**

Run: `unset VIRTUAL_ENV; .venv/bin/python -m pytest tests/ -q`
Expected: 163 passed (no new tests in this task).

- [ ] **Step 6: Commit**

```bash
git add examples/plots/plot_14_edgecolor_control.py
git commit -m "docs(gallery): drop legend_column=30 from plot_14 (auto-sized now)"
```

---

## Task 12: Final verification + PR prep

**Files:** none modified.

- [ ] **Step 1: Run the full test suite one final time**

Run: `unset VIRTUAL_ENV; .venv/bin/python -m pytest tests/ -q`
Expected: 163 passed (121 baseline + 42 new).

- [ ] **Step 2: Smoke-run every gallery example**

```bash
for f in examples/plots/plot_*.py; do
  name=$(basename "$f")
  unset VIRTUAL_ENV; .venv/bin/python "$f" > /dev/null 2>&1 \
    && echo "$name OK" || echo "$name FAIL"
done
```

Expected: every script prints "OK". Any FAIL needs diagnosis before proceeding.

- [ ] **Step 3: Sanity-check the auto-collect user flow**

Run this one-liner to confirm the headline win:

```bash
unset VIRTUAL_ENV; .venv/bin/python - <<'EOF'
import matplotlib; matplotlib.use("Agg")
import pandas as pd, numpy as np
import publiplots as pp

np.random.seed(0)
df = pd.DataFrame({
    "x": np.random.randn(60), "y": np.random.randn(60),
    "g": np.random.choice(list("ABC"), 60),
})

fig, axes = pp.subplots(1, 3, axes_size=(45, 30))
pp.scatterplot(data=df, x="x", y="y", hue="g", palette="pastel", ax=axes[0])
pp.scatterplot(data=df, x="x", y="y", hue="g", palette="pastel", ax=axes[1])
pp.scatterplot(data=df, x="x", y="y", hue="g", palette="pastel", ax=axes[2])
pp.legend_group(anchor=axes[-1])  # zero kwargs — no legend_column, no handles

fig.savefig("/tmp/auto_collect_demo.png")

L = fig._publiplots_layout
print(f"legend_column auto-sized to: {L.legend_column:.1f} mm")
print(f"figure size: {L.figure_size()} mm")
EOF
```

Expected output: `legend_column auto-sized to: <non-zero value> mm`. If it's zero, width-awareness is broken.

- [ ] **Step 4: Push and open PR**

```bash
git push -u origin feat/legend-positioning
gh pr create --title "feat(legend): auto-collect legend_group + width-aware legend_column" \
  --body "$(cat <<'PRBODY'
## Summary

Two coordinated pieces:

1. **Auto-collect.** `pp.legend_group(anchor=...)` walks the grid, collects stashed `LegendEntry` objects across all axes, dedups by `(name, kind)`, and renders one unified legend. Users no longer construct `handles=[...]` manually for the common case.
2. **Width-awareness.** `SubplotsAutoLayout` measures the rendered legend_group width and grows `FigureLayout.legend_column` to fit. The `legend_column` kwarg is dropped from `pp.subplots()` entirely.

Legend width is now fully automatic — no more hand-tuned `legend_column=N` guess.

## Breaking changes

- `pp.subplots(legend_column=N)` raises `TypeError`. Attach a `pp.legend_group(anchor=...)` to the figure instead; the column is auto-sized.
- Plot functions accept `legend=` as `bool | dict[kind, bool]`. `True` / `False` unchanged; `dict` is new.

## Scope

- Infrastructure: `LegendEntry`, `stash_entry`, `resolve_legend_flags`, `entry_is_in_group`.
- `MultiAxesLegendGroup.collect` and `_materialize` auto-collection.
- `SubplotsAutoLayout._measure_legend_column`.
- Migrations: `scatterplot`, `stripplot`, `swarmplot`, `pointplot`.

## Deferred

- Migrations for `bar`, `box`, `violin`, `raincloud`, `heatmap` — these plots render legends inline today and require a bigger refactor per plot. Documented in `pp.legend_group`'s docstring. Until their follow-up PRs, users fall back to manual `group.add_legend(handles=...)` for those plot types.

## Test plan

- [x] Full suite: 163 passing (121 baseline + 42 new).
- [x] `auto_collect_demo.png` renders correctly — legend_column auto-grows.
- [x] All 14 gallery examples smoke-run cleanly.
PRBODY
)"
```

---

## Self-review (pre-execution)

**Spec coverage:**
- `LegendEntry` + store → Task 1.
- `resolve_legend_flags` → Task 1.
- `entry_is_in_group` → Task 1.
- `is_continuous_hue` → Task 1.
- `MultiAxesLegendGroup.collect`, `claims`, figure registration → Task 2.
- `_materialize`, dedup, warning, order filter → Task 3.
- `pp.subplots()` drops `legend_column` kwarg → Task 4.
- `SubplotsAutoLayout._measure_legend_column`, `_ALL_SIDES` extension, `_needs_update` branch → Task 5.
- `pp.legend(ax)` auto-mode reads new store with legacy fallback → Task 6.
- scatter migration → Task 7.
- strip migration → Task 8.
- swarm migration → Task 9.
- point migration → Task 10.
- gallery migration (drop `legend_column=30`) → Task 11.
- Final verification → Task 12.

All spec requirements mapped.

**Placeholders:** grepped for TBD / TODO / placeholder — none in this plan.

**Type consistency:** `LegendEntry` signature identical across Tasks 1, 3, 6, 7-10. `stash_entry`, `get_entries`, `resolve_legend_flags`, `entry_is_in_group`, `is_continuous_hue` names consistent. `_publiplots_legend_entries` attribute name consistent. `_publiplots_legend_group` attribute name consistent. `_measure_legend_column`, `_ALL_SIDES` refs consistent.

Ready for execution.
