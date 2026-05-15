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
snapshot = _helpers.snapshot


def _regen_one(name: str, build_fn, *, check: bool) -> bool:
    """Regen JSON + PNG for one composition.

    In check mode: returns True iff the live output DIFFERS from the
    committed golden. JSON uses byte equality (deterministic);
    PNG uses ``compare_images(tol=10)`` to match the test gate (font
    cache rebuilds + matplotlib timestamp metadata can shift PNG bytes
    without visible changes).
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

    if check:
        return snap_diff or png_diff

    if snap_diff:
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        snap_path.write_text(snap_text, encoding="utf-8")
    if png_diff:
        png_path.write_bytes(new_bytes)
    return snap_diff or png_diff


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
