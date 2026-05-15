"""Snapshot, golden-image, and mm-precision helpers for composer tests.

PR 4.5 ships three of them; PR 5 will extend with PDF helpers.

mm-snapshot format
------------------

JSON dict with shape::

    {
      "schema_version": 1,
      "preset": "cell-2col",
      "figure_size_mm": [W_mm, H_mm],
      "rows": [
        {
          "panels": [
            {"label": "A" | None, "kind": "axes" | "axesgrid" | "text" | "image",
             "size_mm": [w, h],
             "bbox_mm": [x_bottom_left, y_bottom_left, w, h]},
            ...
          ]
        },
        ...
      ]
    }

All numeric values are rounded to 0.01 mm (2 decimal places) so JSON
diffs don't flap on float noise. Snapshots are stable across matplotlib
patch versions as long as the layout math doesn't change.

Regen workflow
--------------

When the layout math intentionally changes (e.g., a new preset, a vpad
default tweak), regenerate goldens with::

    python tools/composer/regen_fixtures.py

Or set ``PUBLIPLOTS_REGEN_GOLDEN=1`` and run pytest — missing/diffing
fixtures will be written instead of asserted. Review the diff manually
before committing.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping

import publiplots as pp

GOLDEN_DIR = Path(__file__).parent
SNAPSHOT_DIR = GOLDEN_DIR / "mm_snapshots"
PNG_DIR = GOLDEN_DIR / "png"

REGEN_ENV = "PUBLIPLOTS_REGEN_GOLDEN"
SCHEMA_VERSION = 1
DEFAULT_TOL_MM = 0.01
DEFAULT_PNG_TOL = 10


def _round_geometry(value: float, tol_mm: float = DEFAULT_TOL_MM) -> float:
    """Round ``value`` to the nearest ``tol_mm`` step.

    For ``tol_mm=0.01`` this is equivalent to ``round(value, 2)`` —
    snapshot stability is the goal, not arbitrary granularity.
    """
    if tol_mm <= 0:
        raise ValueError(f"tol_mm must be > 0; got {tol_mm}")
    # round-half-to-even to match Python's default; for 0.01 step this
    # gives clean 2-dp output.
    steps = round(value / tol_mm)
    return round(steps * tol_mm, 10)  # 10 dp = float-noise floor


def snapshot(canvas: pp.Canvas) -> Dict[str, Any]:
    """Extract the mm-geometry snapshot dict from a finalized canvas.

    Touches ``canvas.figure`` to trigger lazy finalization. The returned
    dict is JSON-serialisable and rounded to ``DEFAULT_TOL_MM``.

    Reads two PR-1+ private attributes on Canvas: ``_preset_name`` (the
    preset string passed to the constructor) and ``_rows`` (the list of
    staged ``_RowStaging`` records, each with a ``.panels`` tuple of
    panel-input dataclasses). Canvas does NOT currently expose a
    ``preset`` property or ``__iter__`` — adding either is out of scope
    for PR 4.5 (Acceptance criterion #14 forbids any
    ``src/publiplots/composer/*`` change). The private-attr access is
    pinned to the PR-1+ Composer surface; if a future PR renames
    ``_rows``/``_preset_name``, this helper updates in lockstep.
    """
    # Force finalization (cheap if already done).
    _ = canvas.figure

    fig_size = canvas.figure_size_mm
    if fig_size is None:
        raise RuntimeError(
            "canvas.figure_size_mm is None after finalization — this "
            "usually means add_row was never called. snapshot() requires "
            "a non-empty canvas."
        )

    # Walk staged _RowStaging records (PR-1+ private surface) AND
    # consume the post-finalize panels in order, matching them up by
    # row position. We need both: _rows gives row structure;
    # _panels_list gives the post-finalize Panel records carrying
    # bbox_mm / size_mm / kind.
    panels_iter = iter(canvas._panels_list)  # noqa: SLF001
    rows_out: List[Dict[str, Any]] = []
    for staging in canvas._rows:  # noqa: SLF001
        panels_out: List[Dict[str, Any]] = []
        for _staged_panel in staging.panels:
            panel = next(panels_iter)
            label = panel.label
            # Normalise False / None labels to JSON-friendly None.
            if label is False or label is None:
                label_out: Any = None
            else:
                label_out = str(label)
            panels_out.append({
                "label": label_out,
                "kind": str(panel.kind),
                "size_mm": [_round_geometry(panel.size_mm[0]),
                            _round_geometry(panel.size_mm[1])],
                "bbox_mm": [_round_geometry(panel.bbox_mm[0]),
                            _round_geometry(panel.bbox_mm[1]),
                            _round_geometry(panel.bbox_mm[2]),
                            _round_geometry(panel.bbox_mm[3])],
            })
        rows_out.append({"panels": panels_out})

    return {
        "schema_version": SCHEMA_VERSION,
        "preset": canvas._preset_name,  # noqa: SLF001
        "figure_size_mm": [_round_geometry(fig_size[0]),
                           _round_geometry(fig_size[1])],
        "rows": rows_out,
    }


def _load_snapshot(name: str) -> Dict[str, Any]:
    path = SNAPSHOT_DIR / f"{name}.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_snapshot(name: str, snap: Dict[str, Any]) -> None:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    path = SNAPSHOT_DIR / f"{name}.json"
    text = json.dumps(snap, indent=2, sort_keys=True) + "\n"
    # Idempotent: skip if bytes identical.
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing == text:
            return
    path.write_text(text, encoding="utf-8")


def _format_snapshot_diff(
    expected: Mapping[str, Any], actual: Mapping[str, Any], tol_mm: float,
) -> str:
    """Build a human-readable diff naming which panel + dimension drifted."""
    lines: List[str] = []
    if expected.get("preset") != actual.get("preset"):
        lines.append(
            f"  preset: {expected.get('preset')!r} → {actual.get('preset')!r}"
        )
    exp_fs = expected.get("figure_size_mm", [None, None])
    act_fs = actual.get("figure_size_mm", [None, None])
    for axis, e, a in zip(("W", "H"), exp_fs, act_fs):
        if e != a and (e is None or a is None or abs(e - a) > tol_mm):
            lines.append(f"  figure_size_mm[{axis}]: {e} → {a}")
    exp_rows = expected.get("rows", [])
    act_rows = actual.get("rows", [])
    if len(exp_rows) != len(act_rows):
        lines.append(f"  row count: {len(exp_rows)} → {len(act_rows)}")
    for r, (er, ar) in enumerate(zip(exp_rows, act_rows)):
        ep = er.get("panels", [])
        ap = ar.get("panels", [])
        if len(ep) != len(ap):
            lines.append(f"  row {r}: panel count {len(ep)} → {len(ap)}")
            continue
        for p, (epn, apn) in enumerate(zip(ep, ap)):
            label = epn.get("label") or apn.get("label") or f"#{p}"
            for field in ("size_mm", "bbox_mm"):
                ev = epn.get(field, [])
                av = apn.get(field, [])
                for i, (e, a) in enumerate(zip(ev, av)):
                    if abs(e - a) > tol_mm:
                        lines.append(
                            f"  row {r} panel {label!r} {field}[{i}]: "
                            f"{e} → {a} (Δ={a - e:+.3f} mm)"
                        )
    return "\n".join(lines) if lines else "  (snapshots equal but compared unequal — recheck rounding)"


def assert_snapshot_matches(
    canvas: pp.Canvas, name: str, *, tol_mm: float = DEFAULT_TOL_MM,
) -> None:
    """Assert ``canvas`` matches the committed snapshot for ``name``.

    Raises ``AssertionError`` on drift > ``tol_mm`` for any numeric
    field, with a diff message identifying the drifted panel + dimension.

    Regen behaviour
    ---------------

    If the snapshot file is missing AND ``PUBLIPLOTS_REGEN_GOLDEN=1`` is
    set in the environment, the snapshot is written and the assertion
    passes. Otherwise the missing file fails the test with instructions.
    """
    actual = snapshot(canvas)
    expected = _load_snapshot(name)

    if not expected:
        if os.environ.get(REGEN_ENV) == "1":
            _write_snapshot(name, actual)
            return
        raise AssertionError(
            f"mm-snapshot golden missing for {name!r}. "
            f"Run `python tools/composer/regen_fixtures.py --only {name}` "
            f"to create it (or set PUBLIPLOTS_REGEN_GOLDEN=1 and re-run "
            f"pytest)."
        )

    if expected == actual:
        return

    if os.environ.get(REGEN_ENV) == "1":
        _write_snapshot(name, actual)
        return

    diff = _format_snapshot_diff(expected, actual, tol_mm)
    raise AssertionError(
        f"mm-snapshot drift > {tol_mm} mm for {name!r}:\n{diff}\n\n"
        f"If the change is intentional, run "
        f"`python tools/composer/regen_fixtures.py --only {name}` and "
        f"commit the updated JSON."
    )


# PR 5 territory — declared here to keep the import surface stable.
def assert_pdf_matches(*args, **kwargs):  # noqa: D401
    """Stub — PR 5 will replace this with the pypdf-based comparator."""
    raise NotImplementedError(
        "assert_pdf_matches lands in PR 5 (vector-PDF compositing)."
    )
