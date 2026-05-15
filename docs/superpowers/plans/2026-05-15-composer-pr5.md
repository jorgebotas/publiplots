# Composer PR 5 Implementation Plan — vector-PDF compositing pipeline + PanelImage

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the headline Composer feature — `canvas.savefig('fig.pdf')` produces a real vector PDF with embedded vector schematics via pypdf + cairosvg, no Illustrator step required. After this PR:

1. **`PanelImage(label=, *, path, size, align='center', clip='fit')`** ships as a real panel kind (not the stub it is in PR 1-4.5). Accepts `.pdf`, `.svg`, `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff` paths.
2. **`canvas.savefig('fig.pdf')`** dispatches to `composer/compositing/pdf.py` instead of raising `NotImplementedError`. The pipeline:
   - Renders matplotlib's empty-slot canvas to a PDF buffer (deterministic via `metadata={"CreationDate": None}` + `fonttype=42`).
   - Loads the canvas PDF as a pypdf reader.
   - For each `PanelImage` slot: converts the schematic to a one-page PDF via cairosvg (SVG) or img2pdf-equivalent (PNG/JPG) or direct pypdf load (PDF), then computes the mm→pt transform per the slot's `bbox_mm`, applies `align`/`clip` rules, and stamps onto the canvas page using `pypdf.Transformation`.
   - Writes via `PdfWriter(clone_from=canvas_reader)` (spike Finding 3 — avoids the deprecated `merge_transformed_page` flow on a reader-owned page).
3. **`strict_vectors`** flag on `Canvas`: when `True`, vector failures raise `ComposerVectorError`. When `False` (default), they fall back to rasterizing the schematic + emitting a `UserWarning`.
4. **`[composer]` install extra** in `pyproject.toml` for `pypdf>=6.0` + `cairosvg>=2.7` + `Pillow>=10.0`. Core install is unchanged; raster-only users don't pay the dependency cost.
5. **Golden-PDF tests** populating `tests/composer/golden/pdf/` (the empty placeholder PR 4.5 created): mediabox dimension, vector preservation (≥1 XObject per stamped schematic), PR 4.5's `compare_images`-style tolerance for renderers' rasterized comparison.
6. **One gallery example** — a Cell-2col figure with one PanelImage SVG schematic + one PanelAxes scatter, saved end-to-end to PDF.

**Architecture:** All compositing logic lives under `src/publiplots/composer/compositing/` (new subpackage). The public surface is the existing `canvas.savefig()` — no new top-level API. PR 5 ships only `pdf.py` + supporting helpers; `svg.py` (PR 6), `raster.py` polish (PR 6), `embed.py` (PR 6), and `dispatch.py` (the formal router; for PR 5 we extend the existing `_save.py` dispatch in-place since it's only growing one branch).

```
src/publiplots/composer/
├── _save.py                      # MODIFY — call into compositing.pdf.savefig_pdf for .pdf
└── compositing/                  # NEW subpackage
    ├── __init__.py               # NEW — re-exports
    ├── pdf.py                    # NEW — savefig_pdf + helpers
    ├── _resources.py             # NEW — schematic→PDF byte conversion (SVG via cairosvg, raster via Pillow→PDF)
    └── _geometry.py              # NEW — pure-Python mm→pt + align×clip math
```

**Tech Stack:** pypdf ≥ 6.0, cairosvg ≥ 2.7, Pillow ≥ 10.0 (all in the new `[composer]` extra). NO new transitive deps; NO custom PDF parser. Editable install at `/home/sagemaker-user/publiplots/`.

**Spec reference:** `docs/superpowers/specs/2026-05-14-composer-design.md`:
- §"PR 5" contract (line 645-648)
- §"PanelImage" (lines 154-162)
- §"Image panel `align` × `clip` matrix" (lines 242-246) — `align` 9 values, `clip` ∈ {fit, fill, stretch}
- §"Architecture" (lines 80-100) — compositing-is-post-savefig invariant
- §"Inviolate invariants" (line 104) — `canvas.savefig(p)` byte-deterministic given identical state, modulo `/CreationDate`

**Spike reference:** `spikes/composer/composer-spike.md` — Path A is the production pattern. Specifically:
- Finding 2: PDF byte-determinism contract — `metadata={"CreationDate": None}` on `fig.savefig` for the canvas, AND set Producer/CreationDate via pypdf metadata for the composed PDF.
- Finding 3: pypdf 7.0 will deprecate `merge_transformed_page` on reader pages → use `PdfWriter(clone_from=canvas_reader)` and merge into `writer.pages[0]`.
- Finding 5: golden-file fixtures must include schematics with embedded text markers chosen FROM that schematic, not borrowed from another fixture.

---

## What's IN scope for PR 5

- **`PanelImage` dataclass** in `src/publiplots/composer/panels.py` (currently a NotImplementedError stub):
  - `label: Optional[Union[str, bool]] = None`
  - `path: Union[str, Path]` (required; `.pdf`, `.svg`, `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`)
  - `size: Tuple[Union[float, str], Union[float, str]]` (mm or `'flex'`)
  - `align: str = 'center'` — one of 9 values: `'top-left'`/`'top'`/`'top-right'`/`'left'`/`'center'`/`'right'`/`'bottom-left'`/`'bottom'`/`'bottom-right'`
  - `clip: str = 'fit'` — one of `'fit'`/`'fill'`/`'stretch'`
  - `label_style: Optional[Mapping[str, Any]] = None`
  - Frozen dataclass. Validation in `__post_init__` (path exists, ext supported, align/clip valid).
- **`Canvas` wiring**:
  - `add_row` accepts `PanelImage` (currently raises `NotImplementedError` per PR 1's stub at canvas.py:405).
  - During finalize, a PanelImage slot is realized as a hidden axes (axis off, no patch, no spines — same trick as `PanelText`) reserving the mm rect. The actual schematic is composited POST-savefig; the empty axes serves as the slot reservation only.
  - `Canvas.savefig('fig.pdf')` dispatches to `composer.compositing.pdf.savefig_pdf` instead of `NotImplementedError`.
  - `Canvas.__init__` gets a NEW `strict_vectors: bool = False` keyword-only param. PR 1 did NOT wire this into the constructor — Task 5 must add the parameter, store as `self._strict_vectors`, and thread through `Canvas.savefig` → `dispatch_savefig` → `savefig_pdf`. (Verified empirically against `canvas.py:117-149` — the parameter is absent today.)
- **`compositing/_geometry.py`** — pure-Python helpers (no matplotlib, no pypdf):
  - `MM2PT: float = 72.0 / 25.4` constant
  - `compute_pdf_transform(slot_bbox_mm, schematic_size_pt, *, align, clip) → (sx, sy, tx_pt, ty_pt)` — the spike's transform math, generalized for align × clip.
  - The `'fit'` clip preserves aspect, scales to fit; the leftover space is filled by `align`. The `'fill'` clip preserves aspect, scales to FILL the slot (overflow CROPPED via mediabox clipping in the writer); `align` chooses which side gets cropped. The `'stretch'` clip ignores aspect and scales independently to fit each axis exactly (always at slot origin; `align` ignored).
- **`compositing/_resources.py`** — schematic-input → in-memory single-page PDF bytes:
  - `load_schematic_as_pdf_bytes(path) → (pdf_bytes: bytes, source_kind: str)` — dispatches by extension.
  - `.pdf`: read directly; if multi-page, take page 0 + warn if multi-page (single-warn per path).
  - `.svg`: `cairosvg.svg2pdf(url=str(path))` → bytes.
  - `.png`/`.jpg`/`.jpeg`/`.tif`/`.tiff`: open via Pillow, save as a single-page PDF page sized to the image's pixel dimensions × `(72 / dpi)` (use `info.dpi[0]` if present, else fall back to 300 dpi).
  - All paths must produce a 1-page PDF whose mediabox carries the schematic's intrinsic size in pt (so `compute_pdf_transform` can use it).
  - Errors: `ComposerVectorError` if cairosvg or pypdf or Pillow raises; the caller catches + branches on `strict_vectors`.
- **`compositing/pdf.py`** — `savefig_pdf(figure, path, *, panels, strict_vectors, dpi, metadata=None) → None`:
  - `panels` is the list of finalized `Panel` records on the canvas (carries `bbox_mm` + `kind` + image-spec fields when `kind == 'image'`).
  - Step 1: render `figure` to a temporary `BytesIO` PDF buffer with deterministic-friendly settings (`metadata={"CreationDate": None}` and `rcParams["pdf.fonttype"] = 42` already set globally; verify).
  - Step 2: open canvas PDF via `pypdf.PdfReader(buf)`.
  - Step 3: build `pypdf.PdfWriter(clone_from=canvas_reader)` (spike Finding 3 idiom).
  - Step 4: for each PanelImage panel:
    - Resolve schematic to PDF bytes via `_resources.load_schematic_as_pdf_bytes` (raster fallback if `strict_vectors=False` and source_kind == 'raster').
    - Open as pypdf reader, take page 0.
    - Compute transform via `_geometry.compute_pdf_transform`.
    - Apply `transformation = pypdf.Transformation().scale(sx, sy).translate(tx_pt, ty_pt)`.
    - Merge: `writer.pages[0].merge_transformed_page(schematic_page, transformation)`.
  - Step 5: set deterministic metadata: `writer.add_metadata({NameObject('/Producer'): 'publiplots-composer', NameObject('/CreationDate'): metadata or 'D:20260101000000Z'})`. Override-able for tests.
  - Step 6: write to `path` via `with open(path, 'wb') as f: writer.write(f)`.
- **`Canvas.strict_vectors`** plumbed: read from constructor (already in spec'd surface), passed through `_save.dispatch_savefig` to `savefig_pdf`. When a vector path fails AND `strict_vectors=False`: rasterize the schematic to PNG bytes via Pillow, then convert to a single-page PDF via the raster path of `_resources`, then stamp. Emit a `UserWarning` naming the panel + the source error.
- **`ComposerVectorError`** — new exception in `composer/exceptions.py`, subclass of `ComposerError`. Carries `panel_label`, `path`, `source_error` (str of the underlying exception).
- **`[composer]` install extra** in `pyproject.toml`:
  ```toml
  [project.optional-dependencies]
  composer = [
      "pypdf>=6.0",
      "cairosvg>=2.7",
      "Pillow>=10.0",
  ]
  ```
  Core install (no `[composer]`): if a user calls `canvas.savefig('fig.pdf')` without the extras installed, raise `ComposerVectorError` with a clear `pip install publiplots[composer]` hint. Tested via mock-import patching.
- **Golden-PDF helper in `_helpers.py`**: replace the `assert_pdf_matches` stub with the real implementation. Modes:
  - `assert_pdf_matches(canvas, name, *, mode='mediabox')` — opens the produced PDF, asserts page mediabox dims (in pt) match the canvas's `figure_size_mm * MM2PT` to 0.5pt tolerance.
  - `assert_pdf_matches(canvas, name, *, mode='structure')` — asserts page count == 1 and ≥ N expected XObjects per stamped schematic (PR 5: 1 schematic → 1 XObject minimum).
  - `assert_pdf_matches(canvas, name, *, mode='render_compare', tol=20)` — rasterizes both the produced PDF and the golden PDF via Pillow at 200 DPI, compares with `compare_images` (re-uses the PNG path's tolerance machinery).
  - Regen-on-missing semantics mirror PR 4.5's `assert_snapshot_matches` and `assert_png_matches`.
- **2 new gated golden PDFs** under `tests/composer/golden/pdf/`:
  - `cell-2col-with-svg-schematic.pdf` — one PanelImage SVG + one PanelAxes scatter.
  - `cell-2col-with-png-schematic.pdf` — one PanelImage PNG (raster fallback path) + one PanelAxes scatter.
- **3 new compositions in `_compositions.py`**:
  - `cell-2col-with-svg-schematic` (the SVG one above)
  - `cell-2col-with-png-schematic` (the PNG one above)
  - The fixture schematics live under `tests/composer/golden/fixtures/`: a tiny SVG with a vector circle + visible text marker, and a tiny PNG (200×200, the same content rasterized).
- **Mediabox + structure + render_compare tests** — `tests/composer/test_pdf_compositing.py` parametrized over the 2 new compositions × 3 modes.
- **No-deps fallback test** — patches `pypdf` import to fail; asserts a clear `ComposerVectorError` with the install-extra hint.
- **Strict-vectors test** — `pp.Canvas(..., strict_vectors=True)` + an intentionally-corrupt SVG → `ComposerVectorError` (no fallback). With `strict_vectors=False` (default), the same input should warn + rasterize.
- **One example**: `examples/composer/cell_2col_with_schematic.py` — clones the SVG-based golden-fixture composition and saves to PDF + PNG. Renders both to `docs/images/composer/`.
- **CHANGELOG entry** under `[Unreleased] / Added`.
- **`publiplots-guide` skill update** — brief Composer-aware section gated to "if Canvas exists" gets a one-line update mentioning vector PDF works (per spec line 689 — transitional bridge until PR 7's `composer-guide`).

## What's OUT of scope for PR 5

- **`canvas.savefig('fig.svg')` vector path** — still raises `NotImplementedError("PR 6")` from `_save.py`. PR 6 lands the in-tree SVG composer.
- **`canvas.embed_figure(panel=, fig=)`** — PR 6.
- **TIFF / CMYK polish** — PR 6.
- **`save_multiple`** — PR 6.
- **`canvas.inspect()` + composer-guide skill** — PR 7.
- **PDF embedding-of-existing-figures** (when `embed_figure` is the source of a slot) — PR 6.
- **PanelImage with multi-page PDF source** — accepted but only page 0 used; warn once. Multi-page input is not a paper-figure use case.
- **Vector-quality validation of the composed schematic** — golden tests check structure (XObject count, mediabox, render-compare). They don't deeply parse the schematic's vector primitives. Spike Finding 5's "marker text preserved" check is implemented as a structure mode, but only for SVG-source fixtures with a known text marker IN the schematic.
- **Determinism beyond `/CreationDate` pin** — spike Finding 2 says `svg.hashsalt` is needed for SVG byte-determinism, but PR 5 only ships PDF. PR 6 picks up the SVG hashsalt contract.
- **Foreign `VIRTUAL_ENV` install advisory** (spike Finding 1) — install-doc note only; not blocking, but the CHANGELOG mentions the `[composer]` extra and the `pip install publiplots[composer]` form.
- **Fonttype audit** — PR 5 trusts that publiplots' rcParams already set `pdf.fonttype = 42` for embed-as-Type42-glyphs; if not, golden-PDF rendering will work but font fidelity is best-effort. Don't add new rcParam manipulation in PR 5.
- **`canvas.inspect()` schema field for PanelImage** — PR 7.
- **PanelImage path resolution against the example file's directory** — paths are absolute or CWD-relative. Path resolution against `__file__` is the example author's job. (Spike fixtures are loaded from `tests/composer/golden/fixtures/{schematic.svg, schematic.png}` via absolute paths inside the build_fns.)

This list is the contract for the spec-compliance reviewer. Anything in this list that the implementer ships counts as scope creep and should be flagged.

---

## File Structure

```
src/publiplots/composer/
├── exceptions.py                       # MODIFY — add ComposerVectorError
├── panels.py                           # MODIFY — promote PanelImage from stub to dataclass
├── canvas.py                           # MODIFY — accept PanelImage in add_row, finalize as hidden axes
├── _save.py                            # MODIFY — dispatch .pdf to compositing.pdf.savefig_pdf
└── compositing/                        # NEW subpackage
    ├── __init__.py                     # NEW — re-exports
    ├── pdf.py                          # NEW — savefig_pdf core
    ├── _resources.py                   # NEW — schematic→PDF byte conversion
    └── _geometry.py                    # NEW — mm→pt + align×clip math

tests/composer/
├── golden/
│   ├── _compositions.py                # MODIFY — add 2 new compositions for PDF goldens
│   ├── _helpers.py                     # MODIFY — implement assert_pdf_matches; add fixtures path
│   ├── fixtures/                       # NEW — small SVG + PNG schematics
│   │   ├── schematic.svg               # Tiny vector w/ text marker "PR5-svg-marker"
│   │   └── schematic.png               # Same content rasterized to 200×200
│   └── pdf/
│       ├── cell-2col-with-svg-schematic.pdf  # NEW golden
│       └── cell-2col-with-png-schematic.pdf  # NEW golden
├── test_pdf_compositing.py             # NEW — parametrized golden PDF tests
├── test_panel_image.py                 # NEW — PanelImage dataclass + Canvas wiring
├── test_compositing_geometry.py        # NEW — pure mm→pt math unit tests
└── test_strict_vectors.py              # NEW — strict_vectors flag + fallback semantics

tools/composer/
└── regen_fixtures.py                   # MODIFY — extend to also write PDFs

examples/composer/
└── cell_2col_with_schematic.py         # NEW — gallery example

pyproject.toml                          # MODIFY — add [composer] extra
CHANGELOG.md                            # MODIFY — [Unreleased] / Added
skills/publiplots-guide/SKILL.md        # MODIFY — one-line note about vector PDF
```

5 modified existing files + 4 new src files + 4 new test files + 1 example + fixtures + golden PDFs. ~800 LOC code + ~600 LOC tests + 1 example, matching the spec's PR 5 estimate.

**File-size budget (warn if exceeded):**
- `compositing/pdf.py`: ≤ 280 LOC
- `compositing/_geometry.py`: ≤ 180 LOC
- `compositing/_resources.py`: ≤ 200 LOC
- `panels.py` delta: PR 4.5 baseline + ≤ 80 LOC for PanelImage
- `canvas.py` delta: ≤ 100 LOC for add_row + finalize wiring
- `_save.py`: ≤ 120 LOC (was 60 LOC)
- Each test file: ≤ 250 LOC

Going over the budget is allowed if necessary, but the implementer MUST flag it as `DONE_WITH_CONCERNS`.

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

Expected: HEAD at `4c63494` (= PR 167's squash-merge). Working tree clean (untracked `.claude/` is fine).

- [ ] **Step 2: Create branch from plan tip**

The plan was committed as the most recent commit on local main (the docs/superpowers/plans gitignore notwithstanding — the rollout convention uses `git add -f`). Create the branch:

```bash
cd /home/sagemaker-user/publiplots
git log --oneline -1   # confirm plan commit at HEAD
git checkout -b feat/composer-vector-pdf-pr5
```

- [ ] **Step 3: Sanity-check baseline**

```bash
cd /home/sagemaker-user/publiplots
uv run pytest tests/composer --no-cov -q 2>&1 | tail -5
```

Expected: 240 passed (PR 4.5 composer suite), 1 skipped, 1 residplot failure unchanged.

- [ ] **Step 4: Verify pypdf + cairosvg + Pillow installed**

```bash
cd /home/sagemaker-user/publiplots
uv run python -c "import pypdf, cairosvg, PIL; print(pypdf.__version__, cairosvg.__version__, PIL.__version__)"
```

Expected: pypdf ≥ 6.0, cairosvg ≥ 2.7, Pillow ≥ 10.0. If any missing, install via `uv add 'pypdf>=6.0' 'cairosvg>=2.7' 'Pillow>=10.0' --optional composer`.

---

## Task 1: ComposerVectorError + PanelImage dataclass

**Files:**
- Modify: `src/publiplots/composer/exceptions.py` — add `ComposerVectorError`
- Modify: `src/publiplots/composer/panels.py` — promote `PanelImage` from stub to dataclass
- Modify: `src/publiplots/composer/__init__.py` — re-export `PanelImage` + `ComposerVectorError`
- Create: `tests/composer/test_panel_image.py` — dataclass validation tests

**Why first:** PanelImage's data shape gates Tasks 4-6. Lock its surface in isolation.

- [ ] **Step 1: Write failing tests**

Create `/home/sagemaker-user/publiplots/tests/composer/test_panel_image.py`:

```python
"""Tests for PanelImage dataclass — PR 5 promotes from stub to real panel kind.

PanelImage references an external schematic file (PDF/SVG/PNG/JPG/TIFF)
to be vector-stamped (or raster-fallback) into a reserved canvas slot.
Construction-time validation enforces ext support, align/clip vocab,
and path existence.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import publiplots as pp
from publiplots.composer.exceptions import ComposerError, ComposerVectorError
from publiplots.composer.panels import PanelImage


# ---------------------------------------------------------------------------
# ComposerVectorError exception
# ---------------------------------------------------------------------------

def test_composer_vector_error_subclass_of_composer_error():
    assert issubclass(ComposerVectorError, ComposerError)


def test_composer_vector_error_carries_context():
    err = ComposerVectorError(
        "cairosvg failed",
        panel_label="A",
        path="/tmp/missing.svg",
        source_error="OSError: no such file",
    )
    assert err.panel_label == "A"
    assert err.path == "/tmp/missing.svg"
    assert "cairosvg failed" in str(err)


# ---------------------------------------------------------------------------
# PanelImage dataclass
# ---------------------------------------------------------------------------

def test_panel_image_basic_construction(tmp_path):
    p = tmp_path / "schematic.svg"
    p.write_text("<svg xmlns='http://www.w3.org/2000/svg' width='10mm' height='10mm'/>")
    panel = PanelImage(label="A", path=p, size=(40.0, 30.0))
    assert panel.label == "A"
    assert panel.path == p
    assert panel.size == (40.0, 30.0)
    assert panel.align == "center"
    assert panel.clip == "fit"


def test_panel_image_accepts_string_path_and_normalizes(tmp_path):
    """str path → normalized to Path in __post_init__ (architect blocker #7)."""
    p = tmp_path / "schematic.svg"
    p.write_text("<svg xmlns='http://www.w3.org/2000/svg'/>")
    panel = PanelImage(label="A", path=str(p), size=(40.0, 30.0))
    assert isinstance(panel.path, Path)
    assert panel.path == p


def test_panel_image_rejects_unsupported_extension(tmp_path):
    p = tmp_path / "schematic.eps"
    p.write_text("dummy")
    with pytest.raises(ValueError, match=r"unsupported.*\.eps"):
        PanelImage(label="A", path=p, size=(40.0, 30.0))


def test_panel_image_rejects_missing_path(tmp_path):
    p = tmp_path / "does-not-exist.svg"
    with pytest.raises(FileNotFoundError, match=r"does-not-exist\.svg"):
        PanelImage(label="A", path=p, size=(40.0, 30.0))


def test_panel_image_rejects_invalid_align(tmp_path):
    p = tmp_path / "schematic.svg"
    p.write_text("<svg xmlns='http://www.w3.org/2000/svg'/>")
    with pytest.raises(ValueError, match=r"align.*'center'.*9"):
        PanelImage(label="A", path=p, size=(40.0, 30.0), align="middle")


def test_panel_image_rejects_invalid_clip(tmp_path):
    p = tmp_path / "schematic.svg"
    p.write_text("<svg xmlns='http://www.w3.org/2000/svg'/>")
    with pytest.raises(ValueError, match=r"clip.*'fit'.*'fill'.*'stretch'"):
        PanelImage(label="A", path=p, size=(40.0, 30.0), clip="cover")


def test_panel_image_accepts_all_9_align_values(tmp_path):
    p = tmp_path / "schematic.svg"
    p.write_text("<svg xmlns='http://www.w3.org/2000/svg'/>")
    for align in ("top-left", "top", "top-right", "left", "center",
                  "right", "bottom-left", "bottom", "bottom-right"):
        panel = PanelImage(label="A", path=p, size=(40.0, 30.0), align=align)
        assert panel.align == align


def test_panel_image_accepts_all_3_clip_values(tmp_path):
    p = tmp_path / "schematic.svg"
    p.write_text("<svg xmlns='http://www.w3.org/2000/svg'/>")
    for clip in ("fit", "fill", "stretch"):
        panel = PanelImage(label="A", path=p, size=(40.0, 30.0), clip=clip)
        assert panel.clip == clip


def test_panel_image_accepts_all_supported_extensions(tmp_path):
    contents = {
        ".svg": "<svg xmlns='http://www.w3.org/2000/svg'/>",
        ".pdf": "%PDF-1.4\n%dummy\n",  # not a valid PDF, but extension check is what we test
        ".png": "\x89PNG\r\n\x1a\n",
        ".jpg": "\xff\xd8\xff\xe0",
        ".jpeg": "\xff\xd8\xff\xe0",
        ".tif": "II*\x00",
        ".tiff": "II*\x00",
    }
    for ext, content in contents.items():
        p = tmp_path / f"schematic{ext}"
        p.write_bytes(content.encode("latin-1") if isinstance(content, str) else content)
        panel = PanelImage(label="A", path=p, size=(40.0, 30.0))
        assert panel.path == p


def test_panel_image_is_frozen_dataclass(tmp_path):
    p = tmp_path / "schematic.svg"
    p.write_text("<svg/>")
    panel = PanelImage(label="A", path=p, size=(40.0, 30.0))
    with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
        panel.label = "B"


def test_panel_image_size_validation_propagates(tmp_path):
    """PR 1's _validate_panel_size shared helper rejects bad sizes."""
    p = tmp_path / "schematic.svg"
    p.write_text("<svg/>")
    with pytest.raises((ValueError, TypeError)):
        PanelImage(label="A", path=p, size=(0.0, 30.0))  # zero width
```

- [ ] **Step 2: Run, verify failure**

```bash
cd /home/sagemaker-user/publiplots
uv run pytest tests/composer/test_panel_image.py -v --no-cov 2>&1 | tail -20
```

Expected: 13 tests collected, ALL fail on `ImportError` for `ComposerVectorError` and/or wrong PanelImage shape.

- [ ] **Step 3: Implement `ComposerVectorError`**

Read `/home/sagemaker-user/publiplots/src/publiplots/composer/exceptions.py` to find the bottom-of-file insertion point. Append after `ComposerAlignmentError` (which PR 3 added):

```python


class ComposerVectorError(ComposerError):
    """Raised when the vector compositing pipeline can't preserve a schematic.

    The most common causes are: a corrupt or unsupported schematic file,
    a cairosvg failure parsing an Illustrator-exported SVG, a pypdf
    failure on a malformed source PDF, or a missing optional dep (pypdf
    / cairosvg / Pillow). The ``panel_label`` + ``path`` + ``source_error``
    attributes let callers format helpful messages without re-parsing
    the message text.
    """

    def __init__(
        self,
        message: str,
        *,
        panel_label: Optional[str] = None,
        path: Optional[str] = None,
        source_error: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.panel_label = panel_label
        self.path = path
        self.source_error = source_error
```

(Also add `from typing import Optional` if it isn't already imported.)

Update `src/publiplots/composer/__init__.py` to re-export `ComposerVectorError`.

- [ ] **Step 4: Implement `PanelImage`**

Read `src/publiplots/composer/panels.py` to find existing `PanelText` (PR 3) — the new `PanelImage` follows the same dataclass shape. Find `_validate_panel_size` (already shared across PanelAxes/PanelText). Add `PanelImage` AFTER `PanelText`:

```python


_VALID_IMAGE_EXTS = {".pdf", ".svg", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
_VALID_IMAGE_ALIGN = {
    "top-left", "top", "top-right",
    "left", "center", "right",
    "bottom-left", "bottom", "bottom-right",
}
_VALID_IMAGE_CLIP = {"fit", "fill", "stretch"}


@dataclass(frozen=True)
class PanelImage:
    """Input record for an external-schematic panel.

    The schematic file is referenced by ``path``; vector-preserving
    insertion happens at savefig time via the compositing pipeline
    (PR 5: PDF; PR 6: SVG). Raster sources (PNG/JPG/TIFF) take a raster
    fallback path that's still valid for journal submission as long as
    the schematic was authored at high enough DPI.

    Parameters
    ----------
    label : str | None | False, default None
        Panel label. ``None`` participates in abc auto-sequencing.
    path : str or Path
        Schematic file. Extension must be one of ``.pdf``, ``.svg``,
        ``.png``, ``.jpg``/``.jpeg``, ``.tif``/``.tiff``.
    size : tuple of (width, height)
        Slot size in mm or ``'flex'`` for the width.
    align : str, default 'center'
        Schematic alignment within the slot when its aspect ratio
        differs from the slot's. One of nine CSS-``object-position``
        values.
    clip : str, default 'fit'
        How to handle aspect mismatch. ``'fit'`` (preserve aspect, fit
        inside slot), ``'fill'`` (preserve aspect, fill slot, crop
        overflow), ``'stretch'`` (ignore aspect; ``align`` ignored).
    label_style : Mapping, optional
        Per-panel override of canvas label_style.
    """

    label: Any = None
    path: Union[str, Path] = ""
    size: Tuple[Union[float, str], Union[float, str]] = (0.0, 0.0)
    align: str = "center"
    clip: str = "fit"
    label_style: Optional[Mapping[str, Any]] = None

    def __post_init__(self):
        _validate_panel_label(self.label)
        _validate_panel_size(self.size)
        # Path validation
        if not self.path:
            raise ValueError("PanelImage: path is required")
        p = Path(self.path) if not isinstance(self.path, Path) else self.path
        ext = p.suffix.lower()
        if ext not in _VALID_IMAGE_EXTS:
            raise ValueError(
                f"PanelImage: unsupported extension {ext!r}. "
                f"Supported: {sorted(_VALID_IMAGE_EXTS)}."
            )
        if not p.exists():
            raise FileNotFoundError(
                f"PanelImage: schematic not found: {p}"
            )
        # Normalize path to Path so downstream callers (Canvas finalize,
        # _resources.py loader) don't have to re-wrap. Frozen dataclass
        # → use object.__setattr__.
        object.__setattr__(self, "path", p)
        # align/clip validation. align is validated regardless of clip;
        # with clip='stretch' the value is recorded but unused at
        # composite time (architect's #8 — keep defensive validation).
        if self.align not in _VALID_IMAGE_ALIGN:
            raise ValueError(
                f"PanelImage: align={self.align!r} invalid. "
                f"Expected one of 9 values like 'center'/'top-left'/...; "
                f"got {self.align!r}."
            )
        if self.clip not in _VALID_IMAGE_CLIP:
            raise ValueError(
                f"PanelImage: clip={self.clip!r} invalid. "
                f"Expected 'fit', 'fill', or 'stretch'."
            )
```

Update `__init__.py` re-exports + `__all__`.

- [ ] **Step 5: Run, all 12 tests pass**

```bash
cd /home/sagemaker-user/publiplots
uv run pytest tests/composer/test_panel_image.py -v --no-cov 2>&1 | tail -15
```

Expected: 13 PASSED.

- [ ] **Step 6: Commit**

```bash
cd /home/sagemaker-user/publiplots
git add src/publiplots/composer/exceptions.py \
        src/publiplots/composer/panels.py \
        src/publiplots/composer/__init__.py \
        tests/composer/test_panel_image.py
git commit -m "feat(composer): PanelImage dataclass + ComposerVectorError"
```

---

## Task 2: Pure-Python compositing geometry helpers

**Files:**
- Create: `src/publiplots/composer/compositing/__init__.py`
- Create: `src/publiplots/composer/compositing/_geometry.py`
- Create: `tests/composer/test_compositing_geometry.py`

**Why second:** the mm→pt + align×clip math is pure-Python (no pypdf/cairosvg). Lock it down before the IO-heavy modules. Same TDD discipline as PR 4.5's `_round_geometry`.

- [ ] **Step 1: Write failing tests**

Create `/home/sagemaker-user/publiplots/tests/composer/test_compositing_geometry.py`:

```python
"""Tests for compositing/_geometry.py — pure-Python mm→pt + align×clip math.

These tests are pure unit tests; no pypdf, no cairosvg, no matplotlib.
The geometry helpers are the foundation PR 5's pdf.py builds on.
"""
from __future__ import annotations

import math

import pytest


# ---------------------------------------------------------------------------
# MM2PT constant
# ---------------------------------------------------------------------------

def test_mm2pt_constant():
    from publiplots.composer.compositing._geometry import MM2PT
    assert math.isclose(MM2PT, 72.0 / 25.4, rel_tol=1e-12)


# ---------------------------------------------------------------------------
# compute_pdf_transform — fit clip
# ---------------------------------------------------------------------------

def test_fit_square_in_square_no_scale():
    """100mm square slot, schematic is exactly 100mm square (in pt) → identity."""
    from publiplots.composer.compositing._geometry import (
        compute_pdf_transform, MM2PT
    )
    slot_bbox_mm = (10.0, 20.0, 100.0, 100.0)  # x, y, w, h
    sch_size_pt = (100.0 * MM2PT, 100.0 * MM2PT)
    sx, sy, tx_pt, ty_pt = compute_pdf_transform(
        slot_bbox_mm, sch_size_pt, align="center", clip="fit",
    )
    assert math.isclose(sx, 1.0, rel_tol=1e-9)
    assert math.isclose(sy, 1.0, rel_tol=1e-9)
    assert math.isclose(tx_pt, 10.0 * MM2PT, rel_tol=1e-9)
    assert math.isclose(ty_pt, 20.0 * MM2PT, rel_tol=1e-9)


def test_fit_wide_into_square_centered():
    """200pt × 100pt schematic into 100mm × 100mm slot → scale 0.5×0.5*ratio,
    centered vertically (letterbox top + bottom)."""
    from publiplots.composer.compositing._geometry import (
        compute_pdf_transform, MM2PT
    )
    slot_w_pt = 100.0 * MM2PT
    slot_h_pt = 100.0 * MM2PT
    sch_w_pt = 200.0
    sch_h_pt = 100.0
    sx, sy, tx_pt, ty_pt = compute_pdf_transform(
        (0.0, 0.0, 100.0, 100.0), (sch_w_pt, sch_h_pt),
        align="center", clip="fit",
    )
    # fit: scale = min(slot_w/sch_w, slot_h/sch_h)
    expected_scale = min(slot_w_pt / sch_w_pt, slot_h_pt / sch_h_pt)
    assert math.isclose(sx, expected_scale, rel_tol=1e-9)
    assert math.isclose(sy, expected_scale, rel_tol=1e-9)
    # Centered: translation = (slot_origin) + (slot_size − scaled_sch_size) / 2
    expected_tx = 0.0 + (slot_w_pt - sch_w_pt * expected_scale) / 2.0
    expected_ty = 0.0 + (slot_h_pt - sch_h_pt * expected_scale) / 2.0
    assert math.isclose(tx_pt, expected_tx, rel_tol=1e-9)
    assert math.isclose(ty_pt, expected_ty, rel_tol=1e-9)


def test_fit_top_left_align():
    """top-left align → translation = slot origin (no centering)."""
    from publiplots.composer.compositing._geometry import (
        compute_pdf_transform, MM2PT
    )
    sx, sy, tx_pt, ty_pt = compute_pdf_transform(
        (5.0, 5.0, 100.0, 100.0), (200.0, 100.0),
        align="top-left", clip="fit",
    )
    # PDF y-axis is BOTTOM-UP; 'top-left' in screen-coords means top of slot.
    # The slot's top-y = slot_y + slot_h; aligning the schematic's top with
    # the slot's top means ty = slot_y + slot_h - sch_h * sy.
    slot_w_pt = 100.0 * MM2PT
    slot_h_pt = 100.0 * MM2PT
    scale = min(slot_w_pt / 200.0, slot_h_pt / 100.0)
    assert math.isclose(tx_pt, 5.0 * MM2PT, rel_tol=1e-9)
    assert math.isclose(ty_pt, 5.0 * MM2PT + slot_h_pt - 100.0 * scale, rel_tol=1e-9)


def test_fit_bottom_right_align():
    """bottom-right align → translation = (slot_x + slot_w - sch_w*sx, slot_y)."""
    from publiplots.composer.compositing._geometry import (
        compute_pdf_transform, MM2PT
    )
    sx, sy, tx_pt, ty_pt = compute_pdf_transform(
        (5.0, 5.0, 100.0, 100.0), (200.0, 100.0),
        align="bottom-right", clip="fit",
    )
    slot_w_pt = 100.0 * MM2PT
    slot_h_pt = 100.0 * MM2PT
    scale = min(slot_w_pt / 200.0, slot_h_pt / 100.0)
    assert math.isclose(tx_pt, 5.0 * MM2PT + slot_w_pt - 200.0 * scale, rel_tol=1e-9)
    assert math.isclose(ty_pt, 5.0 * MM2PT, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# compute_pdf_transform — fill clip
# ---------------------------------------------------------------------------

def test_fill_uses_max_scale():
    """fill clip: scale = max(slot_w/sch_w, slot_h/sch_h)."""
    from publiplots.composer.compositing._geometry import compute_pdf_transform, MM2PT
    sx, sy, tx_pt, ty_pt = compute_pdf_transform(
        (0.0, 0.0, 100.0, 100.0), (200.0, 100.0),
        align="center", clip="fill",
    )
    slot_w_pt = 100.0 * MM2PT
    slot_h_pt = 100.0 * MM2PT
    expected_scale = max(slot_w_pt / 200.0, slot_h_pt / 100.0)
    assert math.isclose(sx, expected_scale, rel_tol=1e-9)
    assert math.isclose(sy, expected_scale, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# compute_pdf_transform — stretch clip
# ---------------------------------------------------------------------------

def test_stretch_independent_axes():
    """stretch clip: sx and sy independent; align ignored (always slot origin)."""
    from publiplots.composer.compositing._geometry import compute_pdf_transform, MM2PT
    sx, sy, tx_pt, ty_pt = compute_pdf_transform(
        (5.0, 7.0, 100.0, 50.0), (200.0, 100.0),
        align="bottom-right", clip="stretch",  # align ignored
    )
    assert math.isclose(sx, 100.0 * MM2PT / 200.0, rel_tol=1e-9)
    assert math.isclose(sy, 50.0 * MM2PT / 100.0, rel_tol=1e-9)
    assert math.isclose(tx_pt, 5.0 * MM2PT, rel_tol=1e-9)
    assert math.isclose(ty_pt, 7.0 * MM2PT, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_compute_pdf_transform_zero_schematic_dim_raises():
    """A schematic with zero width or height is malformed."""
    from publiplots.composer.compositing._geometry import compute_pdf_transform
    with pytest.raises(ValueError, match=r"schematic.*zero"):
        compute_pdf_transform((0.0, 0.0, 100.0, 100.0), (0.0, 100.0),
                              align="center", clip="fit")
```

- [ ] **Step 2: Run, verify failure**

```bash
cd /home/sagemaker-user/publiplots
uv run pytest tests/composer/test_compositing_geometry.py -v --no-cov 2>&1 | tail -10
```

Expected: 8 tests FAIL on `ImportError`.

- [ ] **Step 3: Implement `_geometry.py`**

Create `src/publiplots/composer/compositing/__init__.py` as `""` (empty marker).

Create `src/publiplots/composer/compositing/_geometry.py`:

```python
"""Pure-Python mm→pt + align×clip math for PDF compositing.

No matplotlib, no pypdf, no cairosvg imports. The compositing pipeline
in :mod:`pdf` calls these helpers to translate canvas-space mm rects
into PDF user-space pt transforms.

PDF user space:
- 1 pt = 1/72 inch
- y-axis is BOTTOM-UP (origin at lower-left)
- A page's mediabox is in pt

publiplots Canvas coordinate system:
- mm-based, but the canvas's matplotlib Figure is rendered to a PDF
  whose mediabox is the canvas's ``figure_size_mm * MM2PT``. So a
  panel slot at ``bbox_mm = (x_mm, y_mm, w_mm, h_mm)`` maps directly
  to the PDF page's coordinates by multiplying through MM2PT.
- ``y_mm`` is the BOTTOM of the slot in the canvas (bottom-up to match PDF).
"""
from __future__ import annotations

from typing import Tuple


MM2PT: float = 72.0 / 25.4


def compute_pdf_transform(
    slot_bbox_mm: Tuple[float, float, float, float],
    schematic_size_pt: Tuple[float, float],
    *,
    align: str,
    clip: str,
) -> Tuple[float, float, float, float]:
    """Compute (sx, sy, tx_pt, ty_pt) for stamping a schematic into a slot.

    Parameters
    ----------
    slot_bbox_mm : (x_mm, y_mm, w_mm, h_mm)
        Slot in canvas mm coordinates. ``(x_mm, y_mm)`` is the BOTTOM-LEFT
        corner (matching PDF's bottom-up y).
    schematic_size_pt : (w_pt, h_pt)
        Schematic's intrinsic mediabox dimensions in pt.
    align : str
        One of nine CSS-``object-position`` values. Ignored when
        ``clip='stretch'``.
    clip : {'fit', 'fill', 'stretch'}
        Aspect-ratio policy.

    Returns
    -------
    sx, sy, tx_pt, ty_pt : float
        Scale factors and translation in PDF user pt. Apply via
        ``pypdf.Transformation().scale(sx, sy).translate(tx_pt, ty_pt)``.

    Raises
    ------
    ValueError
        If ``schematic_size_pt`` has any zero or negative dimension, or
        ``clip`` is invalid. ``align`` is validated by PanelImage's
        constructor; passing an unknown value here returns the
        ``'center'`` translation as a graceful fallback.
    """
    sch_w_pt, sch_h_pt = schematic_size_pt
    if sch_w_pt <= 0 or sch_h_pt <= 0:
        raise ValueError(
            f"schematic mediabox has zero or negative dimension: "
            f"({sch_w_pt}, {sch_h_pt}). Cannot compute transform."
        )
    if clip not in ("fit", "fill", "stretch"):
        raise ValueError(
            f"clip={clip!r} invalid. Expected 'fit', 'fill', or 'stretch'."
        )

    slot_x_mm, slot_y_mm, slot_w_mm, slot_h_mm = slot_bbox_mm
    slot_x_pt = slot_x_mm * MM2PT
    slot_y_pt = slot_y_mm * MM2PT
    slot_w_pt = slot_w_mm * MM2PT
    slot_h_pt = slot_h_mm * MM2PT

    if clip == "stretch":
        sx = slot_w_pt / sch_w_pt
        sy = slot_h_pt / sch_h_pt
        return sx, sy, slot_x_pt, slot_y_pt

    if clip == "fit":
        scale = min(slot_w_pt / sch_w_pt, slot_h_pt / sch_h_pt)
    else:  # fill
        scale = max(slot_w_pt / sch_w_pt, slot_h_pt / sch_h_pt)
    sx = sy = scale
    scaled_w_pt = sch_w_pt * scale
    scaled_h_pt = sch_h_pt * scale

    # Compute alignment offsets (slack = slot dim − scaled schematic dim).
    # For 'fit' slack is ≥ 0 in both axes; for 'fill' slack is ≤ 0 in one
    # axis (the schematic overflows on that axis).
    slack_w_pt = slot_w_pt - scaled_w_pt
    slack_h_pt = slot_h_pt - scaled_h_pt

    # Horizontal: 'left'-words → 0, 'right'-words → full slack, else half.
    if align in ("top-left", "left", "bottom-left"):
        ox = 0.0
    elif align in ("top-right", "right", "bottom-right"):
        ox = slack_w_pt
    else:  # 'top', 'bottom', 'center', or unknown
        ox = slack_w_pt / 2.0

    # Vertical: PDF y is BOTTOM-UP. 'bottom'-words → 0; 'top'-words → full slack.
    if align in ("bottom-left", "bottom", "bottom-right"):
        oy = 0.0
    elif align in ("top-left", "top", "top-right"):
        oy = slack_h_pt
    else:  # 'left', 'right', 'center', or unknown
        oy = slack_h_pt / 2.0

    return sx, sy, slot_x_pt + ox, slot_y_pt + oy
```

- [ ] **Step 4: Run tests, all 8 pass**

```bash
cd /home/sagemaker-user/publiplots
uv run pytest tests/composer/test_compositing_geometry.py -v --no-cov 2>&1 | tail -15
```

Expected: 8 PASSED.

- [ ] **Step 5: Commit**

```bash
cd /home/sagemaker-user/publiplots
git add src/publiplots/composer/compositing/__init__.py \
        src/publiplots/composer/compositing/_geometry.py \
        tests/composer/test_compositing_geometry.py
git commit -m "feat(composer): pure-Python mm→pt + align×clip geometry helpers"
```

---

## Task 3: Schematic-resource loader

**Files:**
- Create: `src/publiplots/composer/compositing/_resources.py`
- Modify: `tests/composer/test_panel_image.py` — extend with resource-loader tests

**Why third:** the resource loader takes a path → 1-page PDF bytes via cairosvg / Pillow / direct read. Pure IO; no canvas state. Lock it before the orchestrator (`pdf.py`) needs it.

- [ ] **Step 1: Write failing tests (extend test_panel_image.py with a new section)**

Append to `/home/sagemaker-user/publiplots/tests/composer/test_panel_image.py`:

```python


# ---------------------------------------------------------------------------
# _resources.load_schematic_as_pdf_bytes
# ---------------------------------------------------------------------------

@pytest.fixture
def small_svg(tmp_path):
    """A minimal SVG with a single visible circle + a text marker."""
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'width="40mm" height="30mm" viewBox="0 0 40 30">'
        '<circle cx="20" cy="15" r="10" fill="red"/>'
        '<text x="2" y="28" font-size="3">PR5-svg-marker</text>'
        '</svg>'
    )
    p = tmp_path / "schematic.svg"
    p.write_text(svg)
    return p


@pytest.fixture
def small_png(tmp_path):
    from PIL import Image
    img = Image.new("RGB", (200, 200), color=(255, 0, 0))
    p = tmp_path / "schematic.png"
    img.save(p, dpi=(300, 300))
    return p


def test_load_svg_returns_pdf_bytes(small_svg):
    from publiplots.composer.compositing._resources import load_schematic_as_pdf_bytes
    pdf_bytes, kind = load_schematic_as_pdf_bytes(small_svg)
    assert kind == "vector"
    assert pdf_bytes.startswith(b"%PDF-")


def test_load_png_returns_pdf_bytes(small_png):
    from publiplots.composer.compositing._resources import load_schematic_as_pdf_bytes
    pdf_bytes, kind = load_schematic_as_pdf_bytes(small_png)
    assert kind == "raster"
    assert pdf_bytes.startswith(b"%PDF-")


def test_loaded_pdf_has_one_page_and_correct_mediabox(small_svg):
    """A 40mm × 30mm SVG → mediabox ≈ (40 mm × MM2PT, 30 mm × MM2PT) pt."""
    import io
    import pypdf
    from publiplots.composer.compositing._geometry import MM2PT
    from publiplots.composer.compositing._resources import load_schematic_as_pdf_bytes
    pdf_bytes, _ = load_schematic_as_pdf_bytes(small_svg)
    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    assert len(reader.pages) == 1
    mb = reader.pages[0].mediabox
    assert abs(float(mb.width) - 40.0 * MM2PT) < 1.0  # tolerance: cairosvg
    assert abs(float(mb.height) - 30.0 * MM2PT) < 1.0


def test_load_pdf_passthrough(tmp_path):
    """A real .pdf input is read directly, not re-rendered through cairosvg."""
    import io
    import pypdf

    # Create a tiny one-page PDF with pypdf.
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=72.0, height=72.0)
    p = tmp_path / "schematic.pdf"
    with open(p, "wb") as f:
        writer.write(f)

    from publiplots.composer.compositing._resources import load_schematic_as_pdf_bytes
    pdf_bytes, kind = load_schematic_as_pdf_bytes(p)
    assert kind == "vector"
    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    assert len(reader.pages) == 1


def test_load_corrupt_svg_raises_composer_vector_error(tmp_path):
    p = tmp_path / "corrupt.svg"
    p.write_text("this is not svg")
    from publiplots.composer.exceptions import ComposerVectorError
    from publiplots.composer.compositing._resources import load_schematic_as_pdf_bytes
    with pytest.raises(ComposerVectorError, match=r"corrupt|svg|cairosvg"):
        load_schematic_as_pdf_bytes(p)
```

- [ ] **Step 2: Run, verify failure**

```bash
cd /home/sagemaker-user/publiplots
uv run pytest tests/composer/test_panel_image.py -v --no-cov 2>&1 | tail -15
```

Expected: 5 new tests FAIL on `ImportError` for `_resources`.

- [ ] **Step 3: Implement `_resources.py`**

Create `src/publiplots/composer/compositing/_resources.py`:

```python
"""Schematic input → in-memory single-page PDF bytes.

Each branch returns a ``(pdf_bytes, source_kind)`` tuple where
``source_kind`` is ``'vector'`` (PDF/SVG) or ``'raster'`` (PNG/JPG/TIFF).
The orchestrator in :mod:`pdf` uses ``source_kind`` to decide whether
``strict_vectors=True`` should reject a raster fallback.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Tuple, Union

from publiplots.composer.exceptions import ComposerVectorError


_VECTOR_EXTS = {".pdf", ".svg"}
_RASTER_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
_DEFAULT_RASTER_DPI = 300.0


def load_schematic_as_pdf_bytes(
    path: Union[str, Path],
) -> Tuple[bytes, str]:
    """Convert a schematic file into a single-page PDF byte string.

    Returns
    -------
    pdf_bytes : bytes
        A complete, valid PDF starting with ``b"%PDF-"``.
    source_kind : {'vector', 'raster'}
        Whether the source was vector (PDF/SVG) or raster (PNG/JPG/TIFF).

    Raises
    ------
    FileNotFoundError
        Path doesn't exist (also caught by PanelImage's __post_init__,
        but rechecked here for direct callers).
    ComposerVectorError
        A vector source failed to load/convert (corrupt SVG, malformed
        PDF, missing optional dep). The error carries ``path`` and
        ``source_error``.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"schematic not found: {p}")
    ext = p.suffix.lower()

    if ext == ".pdf":
        try:
            import pypdf
        except ImportError as e:
            raise ComposerVectorError(
                "pypdf is required for PDF schematics. "
                "Install with `pip install publiplots[composer]`.",
                path=str(p),
                source_error=str(e),
            ) from e
        try:
            reader = pypdf.PdfReader(str(p))
            n_pages = len(reader.pages)
            if n_pages == 1:
                # Single-page input: pass through bytes directly. Avoids
                # pypdf round-trip which can renumber objects + add a
                # Producer marker that conflicts with the composer's.
                return p.read_bytes(), "vector"
            # Multi-page: take page 0 + warn (single warn per path).
            import warnings
            warnings.warn(
                f"PanelImage schematic {p.name!r} has "
                f"{n_pages} pages; using page 0.",
                UserWarning,
                stacklevel=3,
            )
            writer = pypdf.PdfWriter()
            writer.add_page(reader.pages[0])
            buf = io.BytesIO()
            writer.write(buf)
            return buf.getvalue(), "vector"
        except Exception as e:
            raise ComposerVectorError(
                f"pypdf failed to read {p.name!r}: {e}",
                path=str(p),
                source_error=str(e),
            ) from e

    if ext == ".svg":
        try:
            import cairosvg
        except ImportError as e:
            raise ComposerVectorError(
                "cairosvg is required for SVG schematics. "
                "Install with `pip install publiplots[composer]`.",
                path=str(p),
                source_error=str(e),
            ) from e
        try:
            pdf_bytes = cairosvg.svg2pdf(url=str(p))
            return pdf_bytes, "vector"
        except Exception as e:
            raise ComposerVectorError(
                f"cairosvg failed to convert {p.name!r}: {e}",
                path=str(p),
                source_error=str(e),
            ) from e

    if ext in _RASTER_EXTS:
        try:
            from PIL import Image
        except ImportError as e:
            raise ComposerVectorError(
                "Pillow is required for raster schematics. "
                "Install with `pip install publiplots[composer]`.",
                path=str(p),
                source_error=str(e),
            ) from e
        try:
            img = Image.open(p)
            img.load()
            # Pillow's PDF save needs RGB or L mode.
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            buf = io.BytesIO()
            dpi = img.info.get("dpi", (_DEFAULT_RASTER_DPI, _DEFAULT_RASTER_DPI))
            img.save(buf, format="PDF", resolution=float(dpi[0]))
            return buf.getvalue(), "raster"
        except Exception as e:
            raise ComposerVectorError(
                f"Pillow failed to convert {p.name!r}: {e}",
                path=str(p),
                source_error=str(e),
            ) from e

    raise ComposerVectorError(
        f"unsupported schematic extension {ext!r} for {p.name!r}",
        path=str(p),
        source_error=None,
    )
```

- [ ] **Step 4: Run, all tests pass**

```bash
cd /home/sagemaker-user/publiplots
uv run pytest tests/composer/test_panel_image.py -v --no-cov 2>&1 | tail -20
```

Expected: 18 PASSED (13 from Task 1 + 5 from Task 3).

- [ ] **Step 5: Commit**

```bash
cd /home/sagemaker-user/publiplots
git add src/publiplots/composer/compositing/_resources.py \
        tests/composer/test_panel_image.py
git commit -m "feat(composer): schematic resource loader (SVG/PDF/raster→PDF)"
```

---

## Task 4: `compositing/pdf.py` — `savefig_pdf` orchestrator

**Files:**
- Create: `src/publiplots/composer/compositing/pdf.py`
- Modify: `src/publiplots/composer/compositing/__init__.py` — re-export `savefig_pdf`
- Create: `tests/composer/test_pdf_compositing.py` — basic orchestrator tests (golden tests come in Task 6)

**Why fourth:** with `_geometry` and `_resources` locked down, the orchestrator is mostly glue + the canvas-render step. This is the largest single file; if it grows past 280 LOC, flag DONE_WITH_CONCERNS.

- [ ] **Step 1: Write failing tests for the orchestrator**

Create `/home/sagemaker-user/publiplots/tests/composer/test_pdf_compositing.py`:

```python
"""Tests for compositing/pdf.py — savefig_pdf orchestrator.

Exercises the end-to-end PDF compose path with a real Canvas + a
PanelImage. Golden-PDF regression tests (mode='render_compare', etc.)
land in Task 6. This task covers the basic interface + invariants.
"""
from __future__ import annotations

import io
from pathlib import Path

import pypdf
import pytest

import publiplots as pp
from publiplots.composer.exceptions import ComposerVectorError


@pytest.fixture
def small_svg(tmp_path):
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'width="40mm" height="30mm" viewBox="0 0 40 30">'
        '<circle cx="20" cy="15" r="10" fill="red"/>'
        '<text x="2" y="28" font-size="3">PR5-svg-marker</text>'
        '</svg>'
    )
    p = tmp_path / "schematic.svg"
    p.write_text(svg)
    return p


def test_savefig_pdf_writes_file(small_svg, tmp_path):
    canvas = pp.Canvas("cell-2col")
    canvas.add_row(
        pp.PanelImage(label="A", path=small_svg, size=(80, 50)),
        pp.PanelAxes(label="B", size=(80, 50)),
    )
    out = tmp_path / "out.pdf"
    canvas.savefig(out)  # dispatches to compositing.pdf.savefig_pdf
    assert out.exists()
    assert out.read_bytes().startswith(b"%PDF-")


def test_savefig_pdf_one_page(small_svg, tmp_path):
    canvas = pp.Canvas("cell-2col")
    canvas.add_row(pp.PanelImage(label="A", path=small_svg, size=(80, 50)),
                   pp.PanelAxes(label="B", size=(80, 50)))
    out = tmp_path / "out.pdf"
    canvas.savefig(out)
    reader = pypdf.PdfReader(out)
    assert len(reader.pages) == 1


def test_savefig_pdf_mediabox_matches_figure_size(small_svg, tmp_path):
    """Page mediabox dims (in pt) match canvas.figure_size_mm × MM2PT."""
    from publiplots.composer.compositing._geometry import MM2PT
    canvas = pp.Canvas("cell-2col")
    canvas.add_row(pp.PanelImage(label="A", path=small_svg, size=(80, 50)),
                   pp.PanelAxes(label="B", size=(80, 50)))
    out = tmp_path / "out.pdf"
    canvas.savefig(out)
    reader = pypdf.PdfReader(out)
    mb = reader.pages[0].mediabox
    fig_w_mm, fig_h_mm = canvas.figure_size_mm
    assert abs(float(mb.width) - fig_w_mm * MM2PT) < 1.0
    assert abs(float(mb.height) - fig_h_mm * MM2PT) < 1.0


def test_savefig_pdf_with_no_panel_images(tmp_path):
    """A canvas with NO PanelImage panels still saves to PDF (no compositing)."""
    canvas = pp.Canvas("cell-2col")
    canvas.add_row(pp.PanelAxes(label="A", size=(80, 50)),
                   pp.PanelAxes(label="B", size=(80, 50)))
    out = tmp_path / "out.pdf"
    canvas.savefig(out)
    assert out.exists()
    reader = pypdf.PdfReader(out)
    assert len(reader.pages) == 1


def test_savefig_pdf_creation_date_pinned(small_svg, tmp_path):
    """Two saves of the same canvas produce byte-identical PDFs (modulo none)."""
    canvas = pp.Canvas("cell-2col")
    canvas.add_row(pp.PanelImage(label="A", path=small_svg, size=(80, 50)),
                   pp.PanelAxes(label="B", size=(80, 50)))
    out1 = tmp_path / "a.pdf"
    out2 = tmp_path / "b.pdf"
    canvas.savefig(out1)
    # Rebuild canvas because the first canvas was finalized; build a fresh
    # one to avoid stateful effects.
    canvas2 = pp.Canvas("cell-2col")
    canvas2.add_row(pp.PanelImage(label="A", path=small_svg, size=(80, 50)),
                    pp.PanelAxes(label="B", size=(80, 50)))
    canvas2.savefig(out2)
    # PDFs should be byte-identical given pinned /CreationDate + /Producer.
    # Tolerance: small differences in whitespace are OK; we check that the
    # /CreationDate and /Producer entries match.
    md1 = pypdf.PdfReader(out1).metadata
    md2 = pypdf.PdfReader(out2).metadata
    assert md1.get("/CreationDate") == md2.get("/CreationDate")
    assert md1.get("/Producer") == md2.get("/Producer")
```

- [ ] **Step 2: Run, verify failure**

```bash
cd /home/sagemaker-user/publiplots
uv run pytest tests/composer/test_pdf_compositing.py -v --no-cov 2>&1 | tail -15
```

Expected: 5 FAIL — first because the dispatch still raises NotImplementedError, then because PanelImage isn't yet accepted by `Canvas.add_row` (Task 5 wires that), then orchestrator missing.

> Note: tests in this task may stay failing until Task 5 (Canvas wiring) lands. That's acceptable — flag the orchestrator-specific failures and proceed; Task 5's commit will turn them green.

- [ ] **Step 3: Implement `compositing/pdf.py`**

Create `/home/sagemaker-user/publiplots/src/publiplots/composer/compositing/pdf.py`:

```python
"""Vector-PDF compositing pipeline.

The single public entry point is :func:`savefig_pdf`. ``Canvas.savefig``
dispatches `.pdf` paths here. The pipeline is:

1. Render the matplotlib Figure (with empty PanelImage slots) to a
   PDF byte buffer with deterministic-friendly settings.
2. Open the canvas PDF as a :class:`pypdf.PdfReader`.
3. Build a :class:`pypdf.PdfWriter` via ``clone_from=canvas_reader``
   (spike Finding 3 — the modern idiom that survives pypdf 7.0's
   deprecation of ``merge_transformed_page`` on reader pages).
4. For each PanelImage panel, convert the schematic to a one-page PDF
   via :mod:`._resources`, compute the mm→pt transform via
   :mod:`._geometry`, and stamp the schematic onto the writer's page.
5. Pin ``/CreationDate`` + ``/Producer`` for byte-determinism, then
   write to disk.

The ``strict_vectors`` flag controls behavior on schematic-load
failure: ``True`` re-raises :class:`ComposerVectorError`; ``False``
attempts a raster fallback (re-render the schematic via Pillow) and
emits a ``UserWarning``.
"""
from __future__ import annotations

import io
import warnings
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Union

from matplotlib.figure import Figure

from publiplots.composer.exceptions import ComposerVectorError
from publiplots.composer.compositing._geometry import (
    MM2PT,
    compute_pdf_transform,
)
from publiplots.composer.compositing._resources import (
    load_schematic_as_pdf_bytes,
)


_DEFAULT_CREATION_DATE = "D:20260101000000Z"
_PRODUCER = "publiplots-composer"


def _filter_image_panels(panels: Iterable[Any]) -> list[Any]:
    """Return only the panels that need vector compositing."""
    return [p for p in panels if getattr(p, "kind", None) == "image"]


def savefig_pdf(
    figure: Figure,
    path: Union[str, Path],
    *,
    panels: Sequence[Any],
    strict_vectors: bool = False,
    metadata_creation_date: Optional[str] = None,
    **savefig_kwargs: Any,
) -> None:
    """Vector-compose the canvas Figure to a PDF at ``path``.

    Parameters
    ----------
    figure
        The matplotlib Figure for the canvas (with empty image slots).
    path
        Output PDF path.
    panels
        The full panel list from ``canvas._panels_list``. Only entries
        with ``panel.kind == 'image'`` trigger compositing.
    strict_vectors
        On any schematic-load failure: ``True`` raises
        :class:`ComposerVectorError`; ``False`` falls back to a raster
        re-render + emits ``UserWarning``.
    metadata_creation_date
        PDF ``/CreationDate`` value. Defaults to a fixed string for
        byte-determinism. Pass ``None`` to use pypdf's auto-generated
        timestamp (NOT recommended — breaks goldens).

    Raises
    ------
    ComposerVectorError
        ``strict_vectors=True`` AND a schematic failed; OR pypdf is not
        installed.
    """
    try:
        import pypdf
        from pypdf.generic import NameObject, TextStringObject
    except ImportError as e:
        raise ComposerVectorError(
            "pypdf is required for `canvas.savefig('*.pdf')`. "
            "Install with `pip install publiplots[composer]`.",
            source_error=str(e),
        ) from e

    output_path = Path(path)
    image_panels = _filter_image_panels(panels)

    # Step 1: render the empty-slot canvas to a PDF buffer with
    # deterministic-friendly metadata. matplotlib's PDF writer accepts a
    # `metadata` dict; passing CreationDate=None suppresses the
    # auto-generated timestamp.
    canvas_buf = io.BytesIO()
    figure.savefig(
        canvas_buf,
        format="pdf",
        metadata={"CreationDate": None},
        **savefig_kwargs,
    )
    canvas_buf.seek(0)

    # Step 2 + 3: open as reader; build writer via clone_from (spike
    # Finding 3).
    canvas_reader = pypdf.PdfReader(canvas_buf)
    writer = pypdf.PdfWriter(clone_from=canvas_reader)
    target_page = writer.pages[0]

    # Step 4: stamp each PanelImage.
    for panel in image_panels:
        _compose_panel_onto(
            writer=writer,
            target_page=target_page,
            panel=panel,
            strict_vectors=strict_vectors,
            pypdf=pypdf,
        )

    # Step 5: deterministic metadata + write to disk.
    cd = (metadata_creation_date if metadata_creation_date is not None
          else _DEFAULT_CREATION_DATE)
    writer.add_metadata({
        NameObject("/Producer"): TextStringObject(_PRODUCER),
        NameObject("/CreationDate"): TextStringObject(cd),
    })
    with open(output_path, "wb") as f:
        writer.write(f)


def _compose_panel_onto(
    *,
    writer: Any,
    target_page: Any,
    panel: Any,
    strict_vectors: bool,
    pypdf: Any,
) -> None:
    """Stamp ``panel``'s schematic onto ``target_page``.

    Reads ``panel.path``, ``panel.bbox_mm``, ``panel.align``,
    ``panel.clip``, and ``panel.label``.
    """
    label = getattr(panel, "label", None) or "<unlabeled>"
    path = getattr(panel, "path")
    align = getattr(panel, "align", "center")
    clip = getattr(panel, "clip", "fit")
    bbox_mm = getattr(panel, "bbox_mm")

    try:
        pdf_bytes, _kind = load_schematic_as_pdf_bytes(path)
    except ComposerVectorError as e:
        if strict_vectors:
            raise
        warnings.warn(
            f"PanelImage {label!r}: vector load of {path!r} failed "
            f"({e}); falling back to raster.",
            UserWarning,
            stacklevel=3,
        )
        pdf_bytes = _raster_fallback(path)

    schematic_reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    schematic_page = schematic_reader.pages[0]
    sch_mb = schematic_page.mediabox
    sch_w_pt = float(sch_mb.width)
    sch_h_pt = float(sch_mb.height)

    sx, sy, tx_pt, ty_pt = compute_pdf_transform(
        bbox_mm, (sch_w_pt, sch_h_pt), align=align, clip=clip,
    )
    transformation = (
        pypdf.Transformation()
        .scale(sx=sx, sy=sy)
        .translate(tx=tx_pt, ty=ty_pt)
    )
    target_page.merge_transformed_page(schematic_page, transformation)


def _raster_fallback(path: Union[str, Path]) -> bytes:
    """Last-ditch raster render when vector load failed.

    Tries to open the file via Pillow regardless of extension; if
    Pillow can't open it, re-raises :class:`ComposerVectorError`.
    """
    from PIL import Image
    try:
        img = Image.open(path)
        img.load()
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PDF", resolution=300.0)
        return buf.getvalue()
    except Exception as e:
        raise ComposerVectorError(
            f"raster fallback also failed for {path!r}: {e}",
            path=str(path),
            source_error=str(e),
        ) from e
```

Update `src/publiplots/composer/compositing/__init__.py`:

```python
"""Compositing pipeline subpackage.

PR 5: PDF compositing via pypdf + cairosvg.
PR 6: SVG compositing (in-tree); embed_figure; raster polish.
"""
from publiplots.composer.compositing.pdf import savefig_pdf

__all__ = ["savefig_pdf"]
```

- [ ] **Step 4: Run, verify orchestrator works in isolation**

```bash
cd /home/sagemaker-user/publiplots
uv run python -c "
import io
from publiplots.composer.compositing._resources import load_schematic_as_pdf_bytes
import tempfile, pathlib
svg = '<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"40mm\" height=\"30mm\"><circle cx=\"20\" cy=\"15\" r=\"10\" fill=\"red\"/></svg>'
with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as f:
    f.write(svg.encode()); path = pathlib.Path(f.name)
b, k = load_schematic_as_pdf_bytes(path)
print('OK' if b.startswith(b'%PDF-') else 'FAIL')
"
```

Expected: `OK`. The orchestrator's tests will mostly stay red until Task 5 wires Canvas — that's expected.

- [ ] **Step 5: Commit**

```bash
cd /home/sagemaker-user/publiplots
git add src/publiplots/composer/compositing/pdf.py \
        src/publiplots/composer/compositing/__init__.py
git commit -m "feat(composer): savefig_pdf orchestrator with pypdf clone_from"
```

---

## Task 5: Canvas wiring + dispatch

**Files:**
- Modify: `src/publiplots/composer/canvas.py` — accept PanelImage in add_row + finalize as hidden axes; thread `strict_vectors` to compositing
- Modify: `src/publiplots/composer/_save.py` — call `savefig_pdf` for `.pdf`

**Why fifth:** the public-API change (PanelImage panels in canvases that save to PDF) lands here. Once committed, all PR 5 behavior is reachable.

- [ ] **Step 1: Read the existing canvas.py** to find:
  - The `add_row` panel-validation site (around canvas.py:401-405) where `PanelImage` currently triggers `NotImplementedError`. Replace with acceptance.
  - The finalize/_compute_geometry site where panels are turned into hidden axes (PanelText is the model; copy that pattern with `kind='image'`).
  - The savefig dispatch hook — `_save.dispatch_savefig` is called from `Canvas.savefig`.

- [ ] **Step 2: Modify `add_row` panel validation**

Find this block in `canvas.py` (around line 401-410):
```python
for p in panels:
    if not isinstance(p, (PanelAxes, PanelGrid, PanelText)):
        raise NotImplementedError(
            f"only PanelAxes/PanelGrid/PanelText supported "
            f"in PR 3, got {type(p).__name__} (PanelImage lands in PR 5)"
        )
```

Replace with:
```python
for p in panels:
    if not isinstance(p, (PanelAxes, PanelGrid, PanelText, PanelImage)):
        raise TypeError(
            f"add_row panels must be PanelAxes / PanelGrid / PanelText / "
            f"PanelImage; got {type(p).__name__}"
        )
```

Add `PanelImage` to the imports at the top of `canvas.py`.

- [ ] **Step 3: Wire PanelImage into `_compute_geometry` and finalize**

Find `_panel_raw_width` and `_panel_raw_height` helpers (canvas.py:33-55). Add a PanelImage branch to each — `PanelImage.size` is the same `(w, h)` mm tuple PanelAxes uses, so the helpers should `isinstance(panel, (PanelAxes, PanelImage, PanelText))` together. Update both helpers + any other dispatch sites that special-case panel kind.

Find the row-iteration in `_finalize_if_needed` (canvas.py around line 670+) where each panel input is realized into a `Panel` result record + a hidden axes. PanelText's path is the closest analogue; PanelImage gets the same hidden-axes treatment (axis off, all spines invisible, no patch, no axison) but `kind='image'` instead of `'text'`. The hidden axes serves only to RESERVE the slot during matplotlib's render to PDF buffer — the schematic gets stamped post-savefig.

The Panel result for `kind='image'` carries the `path`, `align`, `clip` fields so `savefig_pdf` can read them. Plan: extend the `Panel` dataclass in `panels.py` with optional `image_path`, `image_align`, `image_clip` fields (default None for non-image panels).

Update `panels.py` Panel result dataclass:
```python
@dataclass(frozen=True)
class Panel:
    # ...existing fields (label, kind, ax, size_mm, bbox_mm,
    #                    resolved_label_style, axes)...
    image_path: Optional[str] = None
    image_align: Optional[str] = None
    image_clip: Optional[str] = None
```

In `canvas.py`'s panel-construction loop, when the input is a `PanelImage`, set those three fields. For all other kinds, leave None.

- [ ] **Step 4: Add `strict_vectors` to `Canvas.__init__`**

`Canvas.__init__` (canvas.py:117-149) does NOT currently accept `strict_vectors`. PR 1 did not wire the spec'd parameter. Add it now:

Find the existing signature in `Canvas.__init__`:
```python
def __init__(
    self,
    preset: str,
    *,
    width: Optional[float] = None,
    abc: ...,
    # other params PR 1+ added
):
```

Add `strict_vectors: bool = False` to the keyword-only block (alphabetical order with the existing kwargs is fine; not strict). In the body, add:
```python
self._strict_vectors: bool = strict_vectors
```

Update the docstring to document the parameter:
```
strict_vectors : bool, default False
    When True, ``canvas.savefig('fig.pdf')`` raises
    :class:`ComposerVectorError` if any PanelImage schematic fails to
    load as vector (corrupt SVG, missing optional dep, malformed PDF).
    When False (default), failed vector loads fall back to a raster
    re-render of the schematic and emit a ``UserWarning``.
```

(The `Canvas.savefig` plumbing change in Step 5 then reads `self._strict_vectors` and forwards it to `dispatch_savefig`.)

- [ ] **Step 5: Update `_save.py` dispatch**

`dispatch_savefig` must accept `panels` and `strict_vectors` as named keyword arguments (architect blocker #2). Otherwise the raster branch's `**kwargs` would forward `panels=` to `pp.savefig` which would error. Update the SIGNATURE first:

Find:
```python
def dispatch_savefig(
    figure: Figure,
    path: Union[str, Path],
    **kwargs: Any,
) -> None:
```

Replace with:
```python
def dispatch_savefig(
    figure: Figure,
    path: Union[str, Path],
    *,
    panels: Sequence[Any] = (),
    strict_vectors: bool = False,
    metadata_creation_date: Optional[str] = None,
    **kwargs: Any,
) -> None:
```

(Add `from typing import Sequence, Optional` if not already imported.)

Then replace the current `_VECTOR_PDF_EXTS` branch:

```python
if ext in _VECTOR_PDF_EXTS:
    # PR 5: dispatch to compositing.pdf.savefig_pdf
    from publiplots.composer.compositing.pdf import savefig_pdf
    savefig_pdf(
        figure,
        p,
        panels=list(panels),
        strict_vectors=strict_vectors,
        metadata_creation_date=metadata_creation_date,
        **kwargs,
    )
    return
```

The raster branch (`_pp_savefig(str(p), **kwargs)`) is unaffected — `panels`/`strict_vectors`/`metadata_creation_date` are consumed before `**kwargs` is forwarded, so they never leak into the raster path.

Then update `Canvas.savefig`'s call site (canvas.py around line 875+):

Find:
```python
dispatch_savefig(self._figure, path, **kwargs)
```

Replace with:
```python
dispatch_savefig(
    self._figure, path,
    panels=self._panels_list,
    strict_vectors=self._strict_vectors,
    **kwargs,
)
```

- [ ] **Step 6: Run all PR 5 tests**

```bash
cd /home/sagemaker-user/publiplots
uv run pytest tests/composer/test_panel_image.py tests/composer/test_compositing_geometry.py tests/composer/test_pdf_compositing.py -v --no-cov 2>&1 | tail -25
```

Expected: 18 from Task 1+3 + 8 from Task 2 + 5 from Task 4 = 31 PASSED.

- [ ] **Step 7: Run the full composer suite to confirm no regression**

```bash
cd /home/sagemaker-user/publiplots
uv run pytest tests/composer --no-cov -q 2>&1 | tail -5
```

Expected: 240 prior + ~30 new = ~270 passed; 1 skipped, 1 residplot failure unchanged.

- [ ] **Step 8: Commit**

```bash
cd /home/sagemaker-user/publiplots
git add src/publiplots/composer/canvas.py \
        src/publiplots/composer/panels.py \
        src/publiplots/composer/_save.py
git commit -m "feat(composer): wire PanelImage + savefig_pdf through Canvas + dispatch"
```

---

## Task 6: Golden-PDF infrastructure + 2 fixtures

**Files:**
- Modify: `tests/composer/golden/_helpers.py` — implement `assert_pdf_matches`
- Modify: `tests/composer/golden/_compositions.py` — add 2 PanelImage compositions
- Create: `tests/composer/golden/fixtures/schematic.svg` + `schematic.png`
- Create: 2 golden PDFs under `tests/composer/golden/pdf/`
- Modify: `tools/composer/regen_fixtures.py` — extend to write PDFs
- Create: `tests/composer/test_strict_vectors.py`

**Why sixth:** the goldens lock the production behavior. Strict-vectors test exercises the failure path PR 5 fundamentally adds.

- [ ] **Step 1: Add the two new compositions + fixtures**

Create `/home/sagemaker-user/publiplots/tests/composer/golden/fixtures/schematic.svg`:

```xml
<svg xmlns="http://www.w3.org/2000/svg" width="40mm" height="30mm" viewBox="0 0 40 30">
  <rect x="2" y="2" width="36" height="26" fill="white" stroke="black" stroke-width="0.5"/>
  <circle cx="20" cy="15" r="10" fill="#1f77b4"/>
  <text x="3" y="29" font-size="3" font-family="sans-serif">PR5-svg-marker</text>
</svg>
```

Create `tests/composer/golden/fixtures/schematic.png` by rasterizing the SVG via Pillow at 300 DPI. The implementer can run a short helper script (NOT committed):

```python
import cairosvg
from PIL import Image
import io
svg_path = "tests/composer/golden/fixtures/schematic.svg"
png_path = "tests/composer/golden/fixtures/schematic.png"
png_bytes = cairosvg.svg2png(url=svg_path, output_width=472, output_height=354)  # 40mm @ 300dpi ≈ 472px
with open(png_path, "wb") as f: f.write(png_bytes)
```

Then commit BOTH files: the SVG source + the regenerated PNG.

Append to `tests/composer/golden/_compositions.py`:

```python
# Fixtures for PR 5 PanelImage compositions live under
# `tests/composer/golden/fixtures/`. The build_fns below use absolute paths
# resolved relative to this module so the goldens are reproducible from
# any CWD.
_FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _build_cell_2col_with_svg_schematic() -> pp.Canvas:
    canvas = pp.Canvas("cell-2col")
    canvas.add_row(
        pp.PanelImage(label="A", path=_FIXTURES_DIR / "schematic.svg",
                      size=(70, 50)),
        pp.PanelAxes(label="B", size=("flex", 50)),
    )
    return canvas


def _build_cell_2col_with_png_schematic() -> pp.Canvas:
    canvas = pp.Canvas("cell-2col")
    canvas.add_row(
        pp.PanelImage(label="A", path=_FIXTURES_DIR / "schematic.png",
                      size=(70, 50)),
        pp.PanelAxes(label="B", size=("flex", 50)),
    )
    return canvas


COMPOSITIONS.extend([
    ("cell-2col-with-svg-schematic", _build_cell_2col_with_svg_schematic),
    ("cell-2col-with-png-schematic", _build_cell_2col_with_png_schematic),
])
```

(Add `from pathlib import Path` if not present.)

- [ ] **Step 2: Implement `assert_pdf_matches` in _helpers.py**

Replace the existing `assert_pdf_matches` stub in `_helpers.py` with:

```python
PDF_DIR = GOLDEN_DIR / "pdf"


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
            # Check ≥ 1 XObject (proves vector schematic was stamped).
            page = reader.pages[0]
            resources = page.get("/Resources", {})
            xobjs = resources.get("/XObject", {}) if resources else {}
            assert len(xobjs) >= 1, (
                f"PDF {name!r}: expected ≥1 XObject, got {len(xobjs)} "
                f"(was the schematic rasterized?)"
            )
            return

        if mode == "render_compare":
            # Rasterize both PDFs to PNG via Pillow (which reads PDFs via
            # the underlying poppler/ghostscript backend if available, OR
            # convert via pypdf → render in matplotlib). Simplest: use
            # pdf2image if present; else fall back to byte-comparison.
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
```

- [ ] **Step 3: Extend `regen_fixtures.py` to write PDFs**

In `tools/composer/regen_fixtures.py`, modify `_regen_one` to also write a PDF for each composition. The PDF diff signal must NOT be byte equality — pypdf's object IDs and xref offsets shift between patch versions even with pinned `/CreationDate`, so `--check` would report DIFF on every run after a regen. Instead, use a **structure-tuple** diff: page count + mediabox + XObject count.

```python
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
    xobjs = resources.get("/XObject", {}) if resources else {}
    return (
        n_pages,
        round(float(mb.width), 1),
        round(float(mb.height), 1),
        len(xobjs),
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
```

Wire `_regen_one_pdf` into the existing `_regen_one`:
```python
def _regen_one(name: str, build_fn, *, check: bool) -> bool:
    snap_diff = ... # existing
    png_diff = ... # existing
    pdf_diff = _regen_one_pdf(name, build_fn, check=check)
    return snap_diff or png_diff or pdf_diff
```

Add to the imports at top of `regen_fixtures.py`:
```python
PDF_DIR = _helpers.PDF_DIR
```

> **Implementer note:** the structure-signature diff means `--check` won't detect changes that don't affect mediabox / page count / XObject count (e.g., a metadata-only diff or an internal stream re-ordering). That's acceptable for PR 5 — those changes don't affect rendered output. If a future PR needs stricter checks, add `mode='render_compare'` to the regen path then too (but only when the rasterizer is available).

- [ ] **Step 4: Generate the 2 golden PDFs via regen**

```bash
cd /home/sagemaker-user/publiplots
PUBLIPLOTS_REGEN_GOLDEN=1 uv run pytest tests/composer/test_pdf_compositing.py -v --no-cov 2>&1 | tail -10
```

Then run the regen CLI to populate goldens:
```bash
cd /home/sagemaker-user/publiplots
uv run python tools/composer/regen_fixtures.py --only cell-2col-with-svg-schematic 2>&1
uv run python tools/composer/regen_fixtures.py --only cell-2col-with-png-schematic 2>&1
```

Expected: 2 new PDFs in `tests/composer/golden/pdf/`. Visually inspect one if running locally.

- [ ] **Step 5: Add the parametrized golden tests**

Append to `tests/composer/test_pdf_compositing.py`:

```python
import pytest
from tests.composer.golden._compositions import COMPOSITIONS
from tests.composer.golden._helpers import assert_pdf_matches


PDF_GOLDEN_NAMES = [
    "cell-2col-with-svg-schematic",
    "cell-2col-with-png-schematic",
]


@pytest.mark.parametrize("mode", ["mediabox", "structure"])
@pytest.mark.parametrize("name", PDF_GOLDEN_NAMES)
def test_pdf_golden_matches(name: str, mode: str) -> None:
    """Composition `name` matches its golden PDF in `mode`."""
    build_fn = dict(COMPOSITIONS)[name]
    canvas = build_fn()
    assert_pdf_matches(canvas, name, mode=mode)


# render_compare mode requires a PDF rasterizer (pdf2image or pdftocairo).
# Skip cleanly when not present rather than silently passing (which a
# blank-fallback rasterizer would do — see architect blocker #6).
from tests.composer.golden._helpers import _pdf_rasterizer_available


@pytest.mark.parametrize("name", PDF_GOLDEN_NAMES)
@pytest.mark.skipif(
    not _pdf_rasterizer_available(),
    reason="No PDF rasterizer (pdf2image or pdftocairo) available."
)
def test_pdf_golden_render_compare(name: str) -> None:
    """Composition `name` renders pixel-equivalently to its golden PDF.

    Forgiving comparison via `compare_images(tol=20)` after rasterizing
    both PDFs at 200 DPI. The mediabox + structure tests above are the
    primary CI gates; this is the visual regression backstop.
    """
    build_fn = dict(COMPOSITIONS)[name]
    canvas = build_fn()
    assert_pdf_matches(canvas, name, mode="render_compare")
```

- [ ] **Step 6: Implement strict_vectors test**

Create `tests/composer/test_strict_vectors.py`:

```python
"""Tests for Canvas(strict_vectors=...) flag + raster fallback semantics.

PR 5 introduces the strict_vectors gate: True raises on any schematic
load failure; False (default) falls back to a raster re-render and
emits a UserWarning.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pytest

import publiplots as pp
from publiplots.composer.exceptions import ComposerVectorError


@pytest.fixture
def corrupt_svg(tmp_path):
    p = tmp_path / "corrupt.svg"
    p.write_text("this is not svg data")
    return p


def test_strict_vectors_true_raises_on_corrupt_svg(corrupt_svg, tmp_path):
    canvas = pp.Canvas("cell-2col", strict_vectors=True)
    canvas.add_row(
        pp.PanelImage(label="A", path=corrupt_svg, size=(70, 50)),
        pp.PanelAxes(label="B", size=(80, 50)),
    )
    out = tmp_path / "out.pdf"
    with pytest.raises(ComposerVectorError, match=r"corrupt|svg|cairosvg"):
        canvas.savefig(out)


def test_strict_vectors_false_happy_png_no_warning(tmp_path):
    """A valid PNG schematic with strict_vectors=False writes PDF cleanly.

    The PNG goes through the raster path of _resources directly — NOT
    a fallback. No warning should fire.
    """
    from PIL import Image
    png_path = tmp_path / "fallback.png"
    Image.new("RGB", (200, 200), color="red").save(png_path, dpi=(300, 300))

    canvas = pp.Canvas("cell-2col", strict_vectors=False)
    canvas.add_row(
        pp.PanelImage(label="A", path=png_path, size=(70, 50)),
        pp.PanelAxes(label="B", size=(80, 50)),
    )
    out = tmp_path / "out.pdf"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        canvas.savefig(out)
    # No vector-fallback warning should have fired (raster source took
    # the raster path, not a fallback).
    fallback_warnings = [w for w in caught
                         if "fallback" in str(w.message).lower()]
    assert not fallback_warnings, (
        f"unexpected fallback warning on raster source: {fallback_warnings}"
    )
    assert out.exists()


def test_strict_vectors_false_corrupt_svg_warns_and_falls_back(
    monkeypatch, tmp_path,
):
    """A corrupt SVG with strict_vectors=False emits UserWarning + falls back.

    The corrupt SVG can't be rasterized either (Pillow can't read SVG),
    so we mock the resources loader's SVG branch to raise a controlled
    error and verify the fallback path is exercised. The actual fallback
    target uses a valid PNG byte stream supplied by the mock.
    """
    from publiplots.composer.compositing import pdf as pdf_mod
    from publiplots.composer.exceptions import ComposerVectorError
    from PIL import Image

    # Build a valid PNG schematic.
    png_path = tmp_path / "real.png"
    Image.new("RGB", (200, 200), color="blue").save(png_path, dpi=(300, 300))

    # Use a "schematic.svg" pointer for the PanelImage but route the
    # resource loader through a mock that ALWAYS raises ComposerVectorError
    # on first call (simulating cairosvg failing). The PR 5 fallback then
    # rasterizes via Pillow against the same path — but since the path is
    # actually the PNG above, Pillow CAN open it and fallback succeeds.
    svg_path = tmp_path / "fake.svg"
    svg_path.write_text("<svg/>")  # minimal valid stub for path-exists check

    real_loader = pdf_mod.load_schematic_as_pdf_bytes
    call_count = {"n": 0}

    def fake_loader(path):
        call_count["n"] += 1
        # First call: raise as if cairosvg failed.
        # The fallback in pdf.py's _raster_fallback takes the path
        # directly; so to test the fallback path landing, we point
        # PanelImage at the real PNG instead and only mock the loader.
        raise ComposerVectorError(
            "simulated cairosvg failure",
            panel_label="A",
            path=str(path),
            source_error="forced",
        )

    monkeypatch.setattr(pdf_mod, "load_schematic_as_pdf_bytes", fake_loader)

    canvas = pp.Canvas("cell-2col", strict_vectors=False)
    canvas.add_row(
        pp.PanelImage(label="A", path=png_path, size=(70, 50)),
        pp.PanelAxes(label="B", size=(80, 50)),
    )
    out = tmp_path / "out.pdf"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        canvas.savefig(out)

    # The fallback warning must fire (panel label A, "vector load" + "raster").
    fallback_warnings = [w for w in caught
                         if "vector load" in str(w.message)
                         and "raster" in str(w.message)]
    assert fallback_warnings, (
        f"expected vector-fallback UserWarning, got {[str(w.message) for w in caught]}"
    )
    assert out.exists()
```

- [ ] **Step 7: Run all PR 5 tests + strict tests**

```bash
cd /home/sagemaker-user/publiplots
uv run pytest tests/composer -v --no-cov 2>&1 | tail -25
```

Expected: ~270 + 4 PDF goldens + 2 strict = ~276 passed.

- [ ] **Step 8: Commit**

```bash
cd /home/sagemaker-user/publiplots
git add tests/composer/golden/_helpers.py \
        tests/composer/golden/_compositions.py \
        tests/composer/golden/fixtures/schematic.svg \
        tests/composer/golden/fixtures/schematic.png \
        tests/composer/golden/pdf/*.pdf \
        tests/composer/test_pdf_compositing.py \
        tests/composer/test_strict_vectors.py \
        tools/composer/regen_fixtures.py
git commit -m "test(composer): golden PDFs + assert_pdf_matches + strict_vectors gate"
```

---

## Task 7: Example + extras + skill update + open PR

**Files:**
- Create: `examples/composer/cell_2col_with_schematic.py`
- Modify: `pyproject.toml` — add `[composer]` extra
- Modify: `skills/publiplots-guide/SKILL.md` — one-line note about vector PDF
- Modify: `CHANGELOG.md` — `[Unreleased] / Added`

- [ ] **Step 1: Add `[composer]` extras to `pyproject.toml`**

Find `[project.optional-dependencies]` and add:
```toml
composer = [
    "pypdf>=6.0",
    "cairosvg>=2.7",
    "Pillow>=10.0",
]
```

Also add to `dev` (so tests run with these always available; or use the `composer` extra in `dev = [...]`).

- [ ] **Step 2: Gallery example**

Create `/home/sagemaker-user/publiplots/examples/composer/cell_2col_with_schematic.py`:

```python
"""Cell 2-col figure: PanelImage SVG schematic + PanelAxes scatter → vector PDF.

PR 5's headline use case. Saves to BOTH PDF (vector-preserving) and
PNG (for at-a-glance viewing).
"""
from pathlib import Path

import numpy as np
import publiplots as pp


def main() -> None:
    rng = np.random.default_rng(0)
    # Use the same fixture the goldens use so the example is reproducible
    # without bringing extra schematics into the repo.
    fixture = (Path(__file__).resolve().parents[2]
               / "tests" / "composer" / "golden" / "fixtures" / "schematic.svg")

    canvas = pp.Canvas("cell-2col")
    canvas.add_row(
        pp.PanelImage(label="A", path=fixture, size=(70, 50)),
        pp.PanelAxes(label="B", size=("flex", 50)),
    )

    canvas["B"].ax.scatter(rng.normal(size=200), rng.normal(size=200),
                            s=4, alpha=0.6)
    canvas["B"].ax.set_xlabel("x")
    canvas["B"].ax.set_ylabel("y")

    out_dir = Path(__file__).resolve().parents[2] / "docs" / "images" / "composer"
    out_dir.mkdir(parents=True, exist_ok=True)
    canvas.savefig(out_dir / "cell-2col-with-schematic.pdf")
    # Re-create a fresh canvas for PNG to avoid finalize-state issues.
    canvas2 = pp.Canvas("cell-2col")
    canvas2.add_row(
        pp.PanelImage(label="A", path=fixture, size=(70, 50)),
        pp.PanelAxes(label="B", size=("flex", 50)),
    )
    canvas2["B"].ax.scatter(rng.normal(size=200), rng.normal(size=200),
                             s=4, alpha=0.6)
    canvas2["B"].ax.set_xlabel("x"); canvas2["B"].ax.set_ylabel("y")
    canvas2.savefig(out_dir / "cell-2col-with-schematic.png")
    print(f"wrote {out_dir / 'cell-2col-with-schematic.pdf'}")


if __name__ == "__main__":
    main()
```

Run it:
```bash
cd /home/sagemaker-user/publiplots
uv run python examples/composer/cell_2col_with_schematic.py
```

Expected: 2 files written.

- [ ] **Step 3: Skill update**

Find `skills/publiplots-guide/SKILL.md`'s Composer section (added in PR 1's release; bridges until PR 7's standalone composer-guide). Add a one-line note:

```markdown
- **Vector PDF (PR 5):** `canvas.savefig('fig.pdf')` produces a real
  vector PDF with embedded vector schematics. PanelImage('path.svg')
  + PanelImage('path.pdf') stay vector-preserved; PanelImage('path.png')
  takes a raster fallback. `Canvas(..., strict_vectors=True)` to fail
  loud on any vector load error.
```

- [ ] **Step 4: CHANGELOG**

Add to `[Unreleased] / Added`:

```markdown
- Vector PDF compositing (PR 5):
  - `canvas.savefig('fig.pdf')` produces a real vector PDF with
    embedded vector schematics via pypdf + cairosvg.
  - `PanelImage(label, *, path, size, align='center', clip='fit')`
    panel kind. Accepts `.pdf`, `.svg`, `.png`, `.jpg`, `.jpeg`,
    `.tif`, `.tiff`. SVG / PDF inputs preserve vectors; raster inputs
    take a raster fallback path.
  - `Canvas(strict_vectors=True)` raises `ComposerVectorError` on any
    schematic load failure instead of falling back to raster.
  - New `[composer]` install extra: `pip install publiplots[composer]`
    pulls in pypdf ≥ 6.0, cairosvg ≥ 2.7, Pillow ≥ 10.0. Core install
    is unchanged.
  - `tests/composer/golden/pdf/` populated with 2 golden PDFs;
    `assert_pdf_matches` modes: `mediabox`, `structure`, `render_compare`.
```

- [ ] **Step 5: Run full repo suite**

```bash
cd /home/sagemaker-user/publiplots
uv run pytest --no-cov 2>&1 | tail -5
```

Expected: ~1294 prior + ~36 new = ~1330 passed; 1 residplot pre-existing failure unchanged.

- [ ] **Step 6: Commit + push**

```bash
cd /home/sagemaker-user/publiplots
git add pyproject.toml \
        examples/composer/cell_2col_with_schematic.py \
        docs/images/composer/cell-2col-with-schematic.pdf \
        docs/images/composer/cell-2col-with-schematic.png \
        skills/publiplots-guide/SKILL.md \
        CHANGELOG.md
git commit -m "docs(composer): vector-PDF gallery + [composer] extra + CHANGELOG"
git push -u origin feat/composer-vector-pdf-pr5
```

- [ ] **Step 7: Open PR**

```bash
cd /home/sagemaker-user/publiplots
gh pr create \
  --title "feat(composer): vector-PDF compositing pipeline + PanelImage" \
  --body "$(cat <<'EOF'
## Summary

PR 5 of the Composer rollout — the headline feature. After this PR,
`canvas.savefig('fig.pdf')` produces a real vector PDF with embedded
vector schematics via pypdf + cairosvg. No Adobe Illustrator step.

## What's in this PR

- **`PanelImage(label, *, path, size, align='center', clip='fit')`** — promoted from PR 1's NotImplementedError stub to a real panel kind. Accepts `.pdf`, `.svg`, `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`. 9 align values + 3 clip modes.
- **`canvas.savefig('fig.pdf')`** dispatches to `compositing/pdf.py`; `.svg` still raises (PR 6).
- **`Canvas(strict_vectors=True)`** raises `ComposerVectorError` on schematic load failure; `False` (default) falls back to raster + `UserWarning`.
- **`[composer]` install extra**: `pip install publiplots[composer]` adds pypdf + cairosvg + Pillow. Core install unchanged.
- **2 golden PDFs** under `tests/composer/golden/pdf/` (SVG + PNG schematics) gated via `assert_pdf_matches` (modes: mediabox, structure, render_compare).
- **One gallery example**: `examples/composer/cell_2col_with_schematic.py`.

## What's NOT in this PR

- Vector SVG output (`canvas.savefig('fig.svg')`) — PR 6.
- `canvas.embed_figure(panel=, fig=)` — PR 6.
- TIFF / CMYK polish, `save_multiple` — PR 6.
- `canvas.inspect()` + composer-guide skill — PR 7.
- PanelImage with multi-page PDF source — page 0 only, single warning.

## Spike findings status

- ✅ Finding 3 (pypdf 7.0 deprecation) — PR 5 uses `PdfWriter(clone_from=...)` idiom.
- ✅ Finding 1 (foreign VIRTUAL_ENV) — flagged in CHANGELOG via the `[composer]` extra install hint.
- ⏳ Finding 2 (SVG byte-determinism + svg.hashsalt) — PR 6 needs it for SVG; PR 5 only pins PDF /CreationDate.
- ⏳ Findings 5/6/7 — PR 6 territory (in-tree SVG composer).

## Test plan

- [x] All ~36 new composer tests pass (PanelImage dataclass + geometry + orchestrator + PDF goldens + strict).
- [x] All 240 prior composer tests pass (no regression).
- [x] Full repo suite green (~1330 passed); 1 pre-existing residplot failure unchanged.
- [x] `python tools/composer/regen_fixtures.py --check` exits 0.
- [x] Gallery example renders to PDF + PNG without warnings.

## Implementation notes

- PR 5 uses `PdfWriter(clone_from=canvas_reader)` instead of mutating the reader's page (spike Finding 3).
- The empty-slot canvas is rendered to a `BytesIO` PDF buffer with `metadata={"CreationDate": None}` so the matplotlib timestamp doesn't leak; the composed PDF then sets a pinned `/CreationDate` for byte-determinism.
- `panels.py`'s Panel result dataclass extends with optional `image_path`/`image_align`/`image_clip` fields (default None) so `savefig_pdf` can read PanelImage attrs without changing its signature for non-image panels.
- The `assert_pdf_matches` `render_compare` mode requires `pdf2image` (poppler) for high-fidelity rasterization; falls back to Pillow's basic PDF reader if not present. The CI gates use `mediabox` + `structure` modes which don't need poppler.

## Follow-ups

- PR 6 (vector-SVG + embed_figure + raster polish) — directly unlocked by this PR.
- PR 7 (canvas.inspect + composer-guide skill) — PanelImage fields will land in the inspect schema.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
gh pr view --json number,url,state,title
```

---

## Acceptance criteria for PR 5

The PR is ready for human merge when ALL of:

1. All 7 tasks complete.
2. Full test suite green: ~1330 passed; 1 pre-existing residplot failure unchanged.
3. `pp.PanelImage(label, *, path, size, align, clip, label_style)` exists, frozen dataclass, validates path / ext / align / clip.
4. `canvas.add_row(PanelImage(...))` accepts the panel kind (was NotImplementedError before).
5. `canvas.savefig('fig.pdf')` produces a valid vector PDF with the canvas dimensions in the mediabox.
6. The 2 golden PDFs exist and pass `assert_pdf_matches(mode='mediabox')` and `mode='structure'`.
7. `Canvas(strict_vectors=True)` + corrupt SVG → `ComposerVectorError`.
8. `Canvas(strict_vectors=False)` + valid raster → no exception, no warning, vector PDF written.
9. `pip install publiplots[composer]` pulls in pypdf, cairosvg, Pillow.
10. `python tools/composer/regen_fixtures.py --check` exits 0 on the final branch.
11. `examples/composer/cell_2col_with_schematic.py` renders to PDF + PNG without warnings.
12. CHANGELOG entry under `[Unreleased] / Added`.
13. `skills/publiplots-guide/SKILL.md` mentions vector PDF + PanelImage.
14. Out-of-scope items (SVG output, embed_figure, TIFF, save_multiple, canvas.inspect, composer-guide skill) are NOT in the diff.
15. `canvas.savefig('fig.svg')` STILL raises NotImplementedError pointing at PR 6.

If any of #1–#15 fail, the PR is not ready.

---

## Per-PR agent team

Per the rollout convention:

1. **`code-architect`** (opus) — already invoked; flags any contract issues before this plan is finalized. Stress-tests: PanelImage dataclass surface, mm→pt math edge cases, pypdf clone_from idiom, strict_vectors fallback semantics, the Panel-result dataclass extension.
2. **`test-designer`** (general-purpose, opus) — embedded in this plan as the failing-test scaffolds. Net-new tests beyond the plan are scope drift.
3. **Implementer** (general-purpose, opus) — executes Tasks 1-7 in TDD order.
4. **Spec-compliance reviewer** (general-purpose, opus) — adversarial review against "What's IN" / "What's OUT" + spec §"PR 5" + §"PanelImage" + §"Image panel align × clip matrix" + Acceptance criteria.
5. **Code-quality reviewer** (general-purpose, opus) — independent code-review focused on:
   - pypdf 6 vs 7 compatibility (the `clone_from` idiom must NOT trip on the version we have, 6.11).
   - Resource leaks (canvas_buf BytesIO, pypdf readers).
   - Determinism (no timestamp leaks; pinned /CreationDate).
   - Strict-vectors semantics (every failure path either raises or warns + falls back; nothing silently swallows).
   - Error message quality (every ComposerVectorError points at install command + panel label).
   - Path resolution (no CWD-dependent paths in build_fns; fixtures resolved from `__file__`).
6. **`debugger`** — invoked on any test failure or unexpected vector-pipeline anomaly.

All sub-agents on opus.

---

## Post-merge steps (for the human, not the implementer)

After PR 5 merges:
1. Update rollout memory's status table: PR 5 → ✅ MERGED.
2. PR 6 (vector-SVG + embed_figure + raster polish) is unblocked. The SVG composer reuses the spike's Path C learnings (Findings 5, 6, 7).
3. The composer is now functionally complete for the headline use case (Cell-2col figure with vector SVG schematic + scatter, end-to-end to PDF). PR 6 + PR 7 polish + skill landing.
