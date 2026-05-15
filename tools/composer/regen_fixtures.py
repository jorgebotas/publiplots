#!/usr/bin/env python3
"""Regenerate canonical composer golden fixtures.

Walks ``tests/composer/golden/_compositions.py``'s ``COMPOSITIONS``
registry and writes one JSON snapshot + one PNG per entry.

Usage
-----
::

    python tools/composer/regen_fixtures.py             # all compositions
    python tools/composer/regen_fixtures.py --only NAME # one composition
    python tools/composer/regen_fixtures.py --check     # dry-run; exits 1 on diff

Idempotent — files are only written when bytes differ. Safe to run
under CI as a sanity gate.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _load_module(name: str, file: Path):
    """Load a module from a source file path.

    Avoids relying on PEP 420 namespace-package import for the
    ``tests/composer/golden/`` tree, which lacks a top-level
    ``tests/__init__.py``. The CLI is run as a script (no
    pytest-managed sys.path) so the standard
    ``from tests.composer.golden import X`` form is fragile.
    """
    spec = importlib.util.spec_from_file_location(name, file)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {name} from {file}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_compositions = _load_module(
    "composer_golden_compositions",
    REPO / "tests" / "composer" / "golden" / "_compositions.py",
)
_helpers = _load_module(
    "composer_golden_helpers",
    REPO / "tests" / "composer" / "golden" / "_helpers.py",
)
COMPOSITIONS = _compositions.COMPOSITIONS
PNG_DIR = _helpers.PNG_DIR
SNAPSHOT_DIR = _helpers.SNAPSHOT_DIR
PDF_DIR = _helpers.PDF_DIR
SVG_DIR = _helpers.SVG_DIR
snapshot = _helpers.snapshot


def _pdf_structure_signature(pdf_bytes: bytes) -> tuple:
    """Cheap, deterministic signature for PDF diff detection in --check mode.

    Returns (page_count, mediabox_w_pt, mediabox_h_pt, xobject_count_page0).
    Mediabox values are rounded to 1 decimal place to absorb pypdf's
    sub-point float noise.
    """
    import io
    import pypdf
    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    n_pages = len(reader.pages)
    if n_pages == 0:
        return (0, 0.0, 0.0, 0)
    page0 = reader.pages[0]
    mb = page0.mediabox
    resources = page0.get("/Resources", {})
    if hasattr(resources, "get_object"):
        resources = resources.get_object()
    xobjs = resources.get("/XObject", {}) if resources else {}
    if hasattr(xobjs, "get_object"):
        xobjs = xobjs.get_object()
    return (
        n_pages,
        round(float(mb.width), 1),
        round(float(mb.height), 1),
        len(xobjs) if xobjs else 0,
    )


def _regen_one_pdf(name: str, build_fn, *, check: bool) -> bool:
    """Render the composition to PDF; return True iff a regen would change it.

    In --check mode, compares structure signatures (not bytes) — bytes
    differ across pypdf patch versions even with pinned /CreationDate.
    """
    canvas = build_fn()
    pdf_path = PDF_DIR / f"{name}.pdf"
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_out = Path(tmpdir) / f"{name}.pdf"
        canvas.savefig(str(pdf_out))
        new_bytes = pdf_out.read_bytes()
    if not pdf_path.exists():
        pdf_diff = True
    else:
        # In check mode, compare structure signatures.
        if check:
            existing_sig = _pdf_structure_signature(pdf_path.read_bytes())
            new_sig = _pdf_structure_signature(new_bytes)
            pdf_diff = existing_sig != new_sig
        else:
            # In write mode, always rewrite if bytes differ — the
            # idempotent skip-write check at byte level is a cheap win.
            pdf_diff = pdf_path.read_bytes() != new_bytes
    if not check and pdf_diff:
        pdf_path.write_bytes(new_bytes)
    return pdf_diff


def _composition_has_pdf_golden(name: str) -> bool:
    """Compositions that exercise the PDF golden gate (PR 5+).

    PR 5 ships PDF goldens for the two PanelImage compositions; PR 6b
    adds the embed_figure golden.
    """
    return name in {
        "cell-2col-with-svg-schematic",
        "cell-2col-with-png-schematic",
        "cell-2col-with-embed-figure",
    }


def _composition_has_svg_golden(name: str) -> bool:
    """Compositions that exercise the SVG golden gate (PR 6a+).

    Same set as the PDF goldens — both PanelImage compositions and the
    PR 6b embed_figure composition are saved to PDF + SVG.
    """
    return name in {
        "cell-2col-with-svg-schematic",
        "cell-2col-with-png-schematic",
        "cell-2col-with-embed-figure",
    }


def _svg_structure_signature(svg_bytes: bytes) -> tuple:
    """Cheap, deterministic signature for SVG diff detection in --check mode.

    Returns (root_tag, viewbox_tuple, n_panel_image_groups).
    The viewBox is rounded to 1 decimal place to absorb matplotlib
    sub-pt float noise; the panel-image group count is a structural
    invariant we never want to lose.
    """
    from lxml import etree as _lxml_etree
    root = _lxml_etree.fromstring(svg_bytes)
    SVG_NS = "http://www.w3.org/2000/svg"
    vb = root.get("viewBox") or ""
    parts = vb.replace(",", " ").split()
    if len(parts) == 4:
        vb_tuple = tuple(round(float(p), 1) for p in parts)
    else:
        vb_tuple = ()
    n_panel = len(root.xpath(
        "//svg:g[starts-with(@id, 'publiplots-panel-image-')]",
        namespaces={"svg": SVG_NS},
    ))
    # Strip Clark-style namespace prefix from root tag for legibility.
    tag = root.tag
    if tag.startswith("{"):
        tag = tag.split("}", 1)[1]
    return (tag, vb_tuple, n_panel)


def _regen_one_svg(name: str, build_fn, *, check: bool) -> bool:
    """Render the composition to SVG; return True iff regen would change it.

    In --check mode, compares structure signatures (not bytes) — bytes
    can shift across matplotlib patch versions even with pinned
    hashsalt + Date.
    """
    canvas = build_fn()
    svg_path = SVG_DIR / f"{name}.svg"
    SVG_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        svg_out = Path(tmpdir) / f"{name}.svg"
        canvas.savefig(str(svg_out))
        new_bytes = svg_out.read_bytes()
    if not svg_path.exists():
        diff = True
    else:
        if check:
            try:
                existing_sig = _svg_structure_signature(svg_path.read_bytes())
                new_sig = _svg_structure_signature(new_bytes)
                diff = existing_sig != new_sig
            except Exception:
                # Malformed existing → diff
                diff = True
        else:
            diff = svg_path.read_bytes() != new_bytes
    if not check and diff:
        svg_path.write_bytes(new_bytes)
    return diff


def _regen_one(name: str, build_fn, *, check: bool) -> bool:
    """Regen JSON + PNG (+ PDF for PR 5 goldens) for one composition.

    In check mode: returns True iff the live output DIFFERS from the
    committed golden. JSON uses byte equality (deterministic);
    PNG uses ``compare_images(tol=10)`` to match the test gate (font
    cache rebuilds + matplotlib timestamp metadata can shift PNG bytes
    without visible changes); PDF uses a structure-tuple signature
    (page count + mediabox + XObject count) to absorb pypdf patch noise.
    """
    from matplotlib.testing.compare import compare_images

    canvas = build_fn()
    snap = snapshot(canvas)
    snap_path = SNAPSHOT_DIR / f"{name}.json"
    snap_existing = snap_path.read_text(encoding="utf-8") if snap_path.exists() else ""
    import json
    snap_text = json.dumps(snap, indent=2, sort_keys=True) + "\n"
    snap_diff = snap_existing != snap_text

    png_path = PNG_DIR / f"{name}.png"
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / f"{name}.png"
        canvas.savefig(str(out), dpi=600)
        new_bytes = out.read_bytes()
        # Determine PNG diff:
        #  - missing golden → diff
        #  - byte-identical → no diff (skip-write idempotency)
        #  - byte-different → use compare_images(tol=10) to match the
        #    test-gate semantics; this absorbs metadata noise.
        if not png_path.exists():
            png_diff = True
        elif png_path.read_bytes() == new_bytes:
            png_diff = False
        else:
            cmp = compare_images(str(png_path), str(out), tol=10)
            png_diff = cmp is not None

    pdf_diff = False
    if _composition_has_pdf_golden(name):
        pdf_diff = _regen_one_pdf(name, build_fn, check=check)

    svg_diff = False
    if _composition_has_svg_golden(name):
        svg_diff = _regen_one_svg(name, build_fn, check=check)

    if check:
        return snap_diff or png_diff or pdf_diff or svg_diff

    if snap_diff:
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        snap_path.write_text(snap_text, encoding="utf-8")
    if png_diff:
        png_path.write_bytes(new_bytes)
    return snap_diff or png_diff or pdf_diff or svg_diff


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="regen_fixtures",
        description="Regenerate composer golden fixtures (JSON snapshots + PNGs).",
    )
    parser.add_argument(
        "--only",
        help="Regenerate only this composition (kebab-case name from COMPOSITIONS).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Dry-run; exit 1 if any committed fixture differs from live output.",
    )
    args = parser.parse_args(argv)

    registry = COMPOSITIONS
    if args.only is not None:
        names = {n for n, _ in registry}
        if args.only not in names:
            print(f"error: --only {args.only!r} not in registry; "
                  f"valid names: {sorted(names)}", file=sys.stderr)
            return 2
        registry = [(n, fn) for n, fn in registry if n == args.only]

    any_diff = False
    for name, build_fn in registry:
        diff = _regen_one(name, build_fn, check=args.check)
        verb = "DIFF" if diff else "OK  "
        if args.check:
            print(f"[{verb}] {name}")
        else:
            print(f"[{'WROTE' if diff else 'SKIP '}] {name}")
        any_diff = any_diff or diff

    if args.check and any_diff:
        print(
            "\nFixtures are out of date. Run "
            "`python tools/composer/regen_fixtures.py` and commit "
            "the updated golden files.",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
