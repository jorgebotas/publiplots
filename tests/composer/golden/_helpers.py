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
PDF_DIR = GOLDEN_DIR / "pdf"

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
            e_label = epn.get("label")
            a_label = apn.get("label")
            label = e_label if e_label is not None else (
                a_label if a_label is not None else f"#{p}"
            )
            # Non-numeric drift (label change, panel kind change) — name
            # the field so the diff message points at the cause.
            if e_label != a_label:
                lines.append(
                    f"  row {r} panel #{p} label: {e_label!r} → {a_label!r}"
                )
            if epn.get("kind") != apn.get("kind"):
                lines.append(
                    f"  row {r} panel {label!r} kind: "
                    f"{epn.get('kind')!r} → {apn.get('kind')!r}"
                )
            for field in ("size_mm", "bbox_mm"):
                ev = epn.get(field, [])
                av = apn.get(field, [])
                for i, (e, a) in enumerate(zip(ev, av)):
                    if abs(e - a) > tol_mm:
                        lines.append(
                            f"  row {r} panel {label!r} {field}[{i}]: "
                            f"{e} → {a} (Δ={a - e:+.3f} mm)"
                        )
    if lines:
        return "\n".join(lines)
    # All structured comparisons agreed but the dicts compared unequal —
    # this should not be reachable, but emit a structural dump as a
    # safety net rather than an opaque "recheck rounding" message.
    import pprint
    return (
        "  (no field-level drift identified; structural dump:)\n"
        f"  expected: {pprint.pformat(expected, width=80, compact=True)}\n"
        f"  actual:   {pprint.pformat(actual, width=80, compact=True)}"
    )


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


def _png_path(name: str) -> Path:
    return PNG_DIR / f"{name}.png"


def assert_png_matches(
    canvas: pp.Canvas, name: str, *, tol: float = DEFAULT_PNG_TOL,
    dpi: int = 600,
) -> None:
    """Assert canvas-rendered PNG matches committed golden within ``tol``.

    Uses ``matplotlib.testing.compare_images`` for a forgiving byte-level
    image comparison (``tol=10`` catches gross regressions, not pixel
    diffs).

    Regen behaviour mirrors :func:`assert_snapshot_matches`: missing
    fixture + ``PUBLIPLOTS_REGEN_GOLDEN=1`` → write + pass; otherwise
    fails with the regen-CLI hint.
    """
    import tempfile
    from matplotlib.testing.compare import compare_images

    PNG_DIR.mkdir(parents=True, exist_ok=True)
    golden_path = _png_path(name)

    # Render the canvas to a temp file at the configured DPI. Don't call
    # canvas.savefig directly into the golden path even on regen — go
    # through a temp + write to keep the regen path symmetric.
    with tempfile.TemporaryDirectory() as tmpdir:
        actual_path = Path(tmpdir) / f"{name}.png"
        canvas.savefig(str(actual_path), dpi=dpi)

        if not golden_path.exists():
            if os.environ.get(REGEN_ENV) == "1":
                # Idempotent: only write if bytes differ (always differ
                # when missing, but the helper is reused by regen CLI
                # which calls in a loop).
                golden_path.write_bytes(actual_path.read_bytes())
                return
            raise AssertionError(
                f"PNG golden missing for {name!r}. "
                f"Run `python tools/composer/regen_fixtures.py --only {name}` "
                f"to create it (or set PUBLIPLOTS_REGEN_GOLDEN=1 and re-run)."
            )

        # compare_images returns None on match, str message on mismatch.
        result = compare_images(str(golden_path), str(actual_path), tol=tol)
        if result is None:
            return

        if os.environ.get(REGEN_ENV) == "1":
            golden_path.write_bytes(actual_path.read_bytes())
            return

        raise AssertionError(
            f"PNG regression for {name!r} (tol={tol}): {result}\n"
            f"If the change is intentional, run "
            f"`python tools/composer/regen_fixtures.py --only {name}` and "
            f"commit the updated PNG."
        )


def assert_pdf_matches(
    canvas: pp.Canvas, name: str, *, mode: str = "mediabox",
    tol_pt: float = 0.5, tol_image: float = 20,
) -> None:
    """Assert canvas-rendered PDF matches the committed golden.

    Modes
    -----
    'mediabox'
        Page mediabox (in pt) matches ``canvas.figure_size_mm * MM2PT``
        within ``tol_pt``.
    'structure'
        Page count == 1 AND the produced PDF contains at least one
        XObject (proves vector compositing happened, not a rasterized
        re-render).
    'render_compare'
        Rasterize both the produced PDF and the golden via Pillow at
        200 DPI; compare via ``compare_images`` with ``tol=tol_image``.

    Regen behaviour mirrors :func:`assert_snapshot_matches` and
    :func:`assert_png_matches`.
    """
    import io
    import tempfile

    from publiplots.composer.compositing._geometry import MM2PT
    import pypdf
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    golden = PDF_DIR / f"{name}.pdf"

    with tempfile.TemporaryDirectory() as tmpdir:
        actual = Path(tmpdir) / f"{name}.pdf"
        canvas.savefig(str(actual))

        if not golden.exists():
            if os.environ.get(REGEN_ENV) == "1":
                golden.write_bytes(actual.read_bytes())
                return
            raise AssertionError(
                f"PDF golden missing for {name!r}. "
                f"Run `python tools/composer/regen_fixtures.py --only {name}` "
                f"to create it (or set PUBLIPLOTS_REGEN_GOLDEN=1)."
            )

        if mode == "mediabox":
            reader = pypdf.PdfReader(actual)
            mb = reader.pages[0].mediabox
            fig_w_mm, fig_h_mm = canvas.figure_size_mm
            assert abs(float(mb.width) - fig_w_mm * MM2PT) < tol_pt, (
                f"PDF {name!r} mediabox width {float(mb.width)}pt vs "
                f"expected {fig_w_mm * MM2PT}pt (tol={tol_pt}pt)"
            )
            assert abs(float(mb.height) - fig_h_mm * MM2PT) < tol_pt
            return

        if mode == "structure":
            reader = pypdf.PdfReader(actual)
            assert len(reader.pages) == 1, (
                f"PDF {name!r}: expected 1 page, got {len(reader.pages)}"
            )
            # Vector compositing evidence: either ≥1 XObject in resources
            # (raster/PDF sources contribute these) OR a non-trivial
            # content stream length (cairosvg-output SVGs have no
            # XObjects of their own — pypdf inlines their content into
            # the canvas page's content stream rather than wrapping in a
            # Form XObject). Either signal proves the schematic was
            # stamped, not silently dropped.
            page = reader.pages[0]
            resources = page.get("/Resources", {})
            if hasattr(resources, "get_object"):
                resources = resources.get_object()
            xobjs = resources.get("/XObject", {}) if resources else {}
            if hasattr(xobjs, "get_object"):
                xobjs = xobjs.get_object()
            n_xobj = len(xobjs) if xobjs else 0
            # Approximate content-stream size as a fallback heuristic for
            # SVG sources. matplotlib's empty-canvas content is tiny;
            # adding even a small SVG bumps it well past 200 bytes.
            try:
                content = page.get_contents()
                if content is not None:
                    content_data = content.get_data() if hasattr(content, "get_data") else b""
                    content_len = len(content_data)
                else:
                    content_len = 0
            except Exception:
                content_len = 0
            assert n_xobj >= 1 or content_len >= 200, (
                f"PDF {name!r}: structure check failed — XObjects={n_xobj}, "
                f"content_stream_len={content_len} (was the schematic stamped?)"
            )
            return

        if mode == "render_compare":
            from matplotlib.testing.compare import compare_images
            actual_png = Path(tmpdir) / f"{name}-actual.png"
            golden_png = Path(tmpdir) / f"{name}-golden.png"
            _rasterize_pdf(actual, actual_png, dpi=200)
            _rasterize_pdf(golden, golden_png, dpi=200)
            result = compare_images(str(golden_png), str(actual_png),
                                    tol=tol_image)
            if result is None:
                return
            if os.environ.get(REGEN_ENV) == "1":
                golden.write_bytes(actual.read_bytes())
                return
            raise AssertionError(
                f"PDF render_compare regression for {name!r}: {result}\n"
                f"Run `python tools/composer/regen_fixtures.py --only {name}`."
            )

        raise ValueError(
            f"assert_pdf_matches: unknown mode={mode!r}. "
            f"Expected 'mediabox', 'structure', or 'render_compare'."
        )


def _rasterize_pdf(pdf_path: Path, out_png: Path, *, dpi: int = 200) -> None:
    """Rasterize a PDF page 0 to PNG at ``dpi`` for visual comparison.

    Tries in order:
      1. ``pdf2image.convert_from_path`` (Python wrapper around poppler)
      2. ``pdftocairo`` subprocess (poppler binary; usually present
         alongside matplotlib's PDF backend on conda envs)

    Raises ``RuntimeError`` if no viable rasterizer is available. The
    helper deliberately does NOT silently fall back to a blank PNG —
    that would let ``render_compare`` mode silently pass on regressions.
    Callers (e.g. test parametrization) should check
    :func:`_pdf_rasterizer_available` and ``pytest.skip()`` rather than
    hit this RuntimeError.
    """
    # Option 1: pdf2image (poppler) — the high-fidelity path.
    try:
        from pdf2image import convert_from_path
        images = convert_from_path(str(pdf_path), dpi=dpi,
                                    first_page=1, last_page=1)
        images[0].save(out_png)
        return
    except ImportError:
        pass

    # Option 2: pdftocairo subprocess — present alongside matplotlib's
    # poppler-backed PDF tooling on most conda/system installs.
    import shutil
    import subprocess
    if shutil.which("pdftocairo") is not None:
        # pdftocairo writes "<prefix>.png" given "-png" + "-singlefile".
        prefix = out_png.with_suffix("")
        subprocess.run(
            ["pdftocairo", "-png", "-singlefile", "-r", str(dpi),
             str(pdf_path), str(prefix)],
            check=True,
            capture_output=True,
        )
        return

    raise RuntimeError(
        "No PDF rasterizer available for `assert_pdf_matches(mode='render_compare')`. "
        "Install pdf2image (`pip install pdf2image`) or ensure pdftocairo is on PATH. "
        "Use mode='mediabox' or 'structure' for poppler-free CI."
    )


def _pdf_rasterizer_available() -> bool:
    """True iff `_rasterize_pdf` has a viable backend.

    Tests can use this to conditionally skip ``render_compare`` mode
    rather than raising RuntimeError.
    """
    try:
        import pdf2image  # noqa: F401
        return True
    except ImportError:
        pass
    import shutil
    return shutil.which("pdftocairo") is not None
