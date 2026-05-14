# Composer Vector-Pipeline Spike — Findings

- **Branch:** `feat/composer-spike`
- **Date:** 2026-05-14
- **Spec reference:** [`docs/superpowers/specs/2026-05-14-composer-design.md`](../../docs/superpowers/specs/2026-05-14-composer-design.md)

---

## 1. Summary verdict

**YELLOW.**

All three compositing paths are structurally feasible: Path A (pypdf) lands the schematic at the right mediabox dimensions and preserves vectors; Path B (cairosvg → PDF → pypdf stamp) reproduces Path A's output for SVG-source schematics with text searchable in PDF readers; Path C (in-tree SVG XML splice) parses, composes, and renders in Firefox. None of the paths are blocked, and the design spec stands as written.

The yellow flag (rather than green) tracks four real, PR-relevant issues uncovered during the spike: a pypdf 7.0 deprecation that will silently break Path A, an SVG byte-determinism contract that is not yet documented, an ElementTree namespace-rewrite hazard in Path C, and a unit-conversion bug in Path C's mm→canvas-user-units pipeline. Each is small in isolation but each must be resolved before PR 5 / PR 6, so the spike's job is to surface them now rather than during the production rollout.

---

## 2. Per-path results

### 2.1 Path A — pypdf

| Field | Value |
|---|---|
| Status | **PASS** |
| Vectors preserved | yes (schematic stamped as a single XObject, n_xobjects = 1) |
| Output | [`outputs/path_a_composed.pdf`](outputs/path_a_composed.pdf) (32 415 bytes); also [`outputs/path_a_composed_complex.pdf`](outputs/path_a_composed_complex.pdf) (26 591 bytes) for the claudecode-color schematic |
| Mediabox (mm) | actual `(173.99999984, 80.00000016)` vs expected `(174.0, 80.0)`; within tolerance |
| n_pages | 1 |
| has_xobjects | true |
| Programmatic evidence | [`path_a_pypdf/report.json`](path_a_pypdf/report.json) |

**Evidence — both schematics composited:** Path A was exercised on BOTH `schematic_simple.svg` (→ `path_a_composed.pdf`) AND `schematic_complex.svg` (→ `path_a_composed_complex.pdf`, the claudecode-color variant), per Step 6 of Task 3's plan. Both produced single-page PDFs with the schematic stamped into the reserved slot.

**Manual viewer evidence (user):** the composed PDF opens cleanly in Adobe Illustrator; the schematic is in the reserved slot and selectable as vector geometry. The user's spot inspection of `path_a_composed_complex.pdf` also passed (no visible regressions vs `path_a_composed.pdf`). The right-panel xlabel `x` is clipped at the canvas bottom edge — this is inherited from the canvas fixture (Finding 4), not introduced by Path A.

**Issues found:**
- pypdf 7.0 deprecation warning on `merge_transformed_page` (Finding 3) — the current call pattern (`canvas_page` belongs to a `PdfReader`, not the `PdfWriter`) is being phased out.
- xlabel clipping inherited from the bare-mpl canvas fixture (Finding 4) — geometry concern for PR 1, not a Path A defect.

**Recommendation for PR 5:** adopt as the production PDF compositor with the `PdfWriter(clone_from=canvas_reader)` migration applied. See `path_a_pypdf/compose.py:42-60` for the current call site.

---

### 2.2 Path B — cairosvg

| Field | Value |
|---|---|
| Status | **PARTIAL** |
| Vectors preserved | yes (text extractable from `path_b_simple.pdf`; cairosvg emits glyph operators, not rasters) |
| Outputs | [`outputs/path_b_simple.pdf`](outputs/path_b_simple.pdf) (32 424 bytes), [`outputs/path_b_complex.pdf`](outputs/path_b_complex.pdf) (26 600 bytes) |
| Programmatic evidence | [`path_b_cairosvg/report.json`](path_b_cairosvg/report.json) |

| File | n_pages | extracted text length | marker `"test schematic"` preserved |
|---|---:|---:|:---:|
| `path_b_simple.pdf` | 1 | 85 chars | TRUE |
| `path_b_complex.pdf` | 1 | 49 chars | FALSE *(see caveat)* |

The `marker_simple_text_preserved=FALSE` row for `path_b_complex.pdf` is a **measurement-design artifact**, not a cairosvg defect: validate.py searches for `"test schematic"` in both PDFs, but that string lives only in `schematic_simple.svg`. `schematic_complex.svg` (the claudecode-color schematic) contains different text. See Finding 5 for the consequence.

**Manual viewer evidence (user):** both PDFs open in standard readers; the user reported no visible issues with either output. Path B's complex-schematic text-preservation behavior is **programmatically unvalidated** — the only signal we have is the human visual check.

**Issues found:**
- Marker chosen for the simple schematic was reused as the complex-schematic marker, so the complex case has no programmatic text-preservation signal (Finding 5).

**Recommendation for PR 5:** adopt for the SVG-source schematic ingest path (cairosvg already runs inside Path A for the same purpose). PR 5's golden-file tests must include at least one schematic per text-handling regime (basic font, complex Illustrator-exported font) **with markers chosen from the schematic under test**, not reused across schematics.

---

### 2.3 Path C — in-tree SVG

| Field | Value |
|---|---|
| Status | **PARTIAL** |
| Vectors preserved | yes (root tag `{http://www.w3.org/2000/svg}svg`; 60 `<g>` elements; 2 `<text>` elements; 2 `<circle>` elements) |
| Output | [`outputs/path_c_composed.svg`](outputs/path_c_composed.svg) (48 394 bytes) |
| Programmatic evidence | [`path_c_in_tree_svg/report.json`](path_c_in_tree_svg/report.json) |
| Namespace audit | 237 `ns4:href` references, 0 `xlink:href` references after composition |

**Manual viewer evidence (user):** the composed SVG opens in Firefox and the right-panel scatter plot renders correctly (xlink hrefs resolve via the declared `xmlns:ns4`). However, side-by-side with `path_a_composed.pdf` the schematic in `path_c_composed.svg` is visibly **smaller and mis-positioned** — it sits at roughly point coordinates rather than mm coordinates, at ~35% of the intended size. Confirmed by direct visual diff between the two outputs.

**Issues found:**
- ElementTree namespace rewrite: only the SVG default namespace was pre-registered, so the serializer auto-generated `ns0…nsN` prefixes for xlink/cc/dc/rdf. Output is technically valid XML but trips lexical-match tooling (Finding 6).
- Unit conversion bug: `compose.py` applied mm-valued transforms to a matplotlib SVG canvas whose viewBox is in pt (`0 0 493.228 226.771`), so the schematic was scaled and translated as if the canvas were in mm (Finding 7). See `path_c_in_tree_svg/compose.py:60-68`.

**Recommendation for PR 6:** Path C is feasible but needs two non-trivial pieces of code before it can be the production SVG compositor: (a) a namespace pre-registration block (or a switch to lxml — recommended), and (b) a `_resolve_svg_units()` helper that detects canvas viewBox unit and schematic viewBox dimensions independently of `width`/`height` declarations.

---

## 3. Architectural revisions for PR 5 / PR 6

| Revision | PR | Source finding | One-line action |
|---|---|---|---|
| Use `PdfWriter(clone_from=canvas_reader)` for the compose call; mutate `writer.pages[0]`, not the reader's page | PR 5 | Finding 3 | Replace the in-place `canvas_page.merge_transformed_page` pattern in `path_a_pypdf/compose.py:42-60` |
| Document the cairosvg + pypdf install path; warn about foreign `VIRTUAL_ENV` mis-targeting | PR 5 | Finding 1 | Add a "Working from a sub-directory" note to the `[composer]` extra install docs |
| Golden-file tests must place text markers inside the schematic under test (one per font regime) | PR 5 | Finding 5 | Update PR 5's test plan to require at least basic-font + complex-font fixtures, each with a marker drawn from its own SVG |
| `CanvasLayout` (or the production equivalent) MUST reserve `xlabel_space`, `ylabel_space`, `title_space` per row/column — mirror `FigureLayout` | PR 1 | Finding 4 | Promote bare-mpl canvas to use the same decoration-space reservation pattern that `FigureLayout` already exposes |
| Determinism contract for byte-stable golden files: PR 5 (PDF) must pass `metadata={"CreationDate": None}` to `fig.savefig(..., format='pdf', ...)`; PR 6 (SVG) must pass `metadata={"Date": None}` to `fig.savefig(..., format='svg', ...)` AND set `plt.rcParams["svg.hashsalt"] = "<deterministic-string>"` before savefig (otherwise SVG `<defs>` IDs randomize). All three knobs are required. | PR 5 + PR 6 | Finding 2 | Make this part of the determinism contract; document in the composer's "Determinism" section |
| Pre-register xlink/cc/dc/rdf namespaces before parsing **or** switch to lxml (recommended) | PR 6 | Finding 6 | lxml preserves source prefixes natively — removes a class of latent bugs whenever upstream SVG introduces a new namespace |
| `_resolve_svg_units()` helper: detect canvas viewBox unit + schematic viewBox dims independently of width/height declarations; compute tx/ty/sx/sy in canvas user-units | PR 6 | Finding 7 | Add a dedicated unit-resolver with a permutation matrix of viewBox/width/height test cases |

These are *amendments tracked in this report*, not edits to the spec doc — the spec stands as written.

---

## 4. Dependency notes

| Component | Version / status |
|---|---|
| pypdf | **6.11.0** (installed, deprecation warning for 7.0 on current call pattern — see Finding 3) |
| cairosvg | **2.9.0** (installed, no deprecation warnings observed) |
| libcairo2 | **not installed via dpkg** — `dpkg -l libcairo2` returns "no packages found matching libcairo2" yet cairosvg runs successfully on this SageMaker image. cairocffi is a CFFI binding and does NOT ship libcairo itself; the underlying libcairo must therefore be reaching the process via a non-dpkg path (most likely a pip-installed cairo wheel pulled in transitively, or a non-dpkg-managed shared-library location). PR 5's install docs must verify this empirically before promising self-contained installation; consider testing the `[composer]` extra on a clean Ubuntu / macOS / Windows venv in CI to detect when this assumption breaks. |
| Python | 3.12 (project venv at `/home/sagemaker-user/publiplots/.venv`) |

**Install gotcha — foreign `VIRTUAL_ENV` (Finding 1):** during Task 1, `uv pip install pypdf cairosvg` from inside `/home/sagemaker-user/publiplots` resolved to `/home/sagemaker-user/lemur/.venv` because the parent shell's `VIRTUAL_ENV` was inherited. Workaround: either `unset VIRTUAL_ENV` before invoking `uv pip install`, set `VIRTUAL_ENV=/home/sagemaker-user/publiplots/.venv` explicitly, or use `uv run` (which always targets the project venv). Re-running the validate scripts surfaces the same warning today: `warning: VIRTUAL_ENV=/home/sagemaker-user/lemur/.venv does not match the project environment path .venv and will be ignored`.

For PR 5: install docs should warn about this; consider testing the `[composer]` extra install in CI from a clean venv to catch regressions.

---

## 5. Open questions for PR 5 / PR 6

These are deliberately deferred to PR implementation rather than pre-answered now.

1. **PR 5** — What is the exact pypdf API surface after the `PdfWriter(clone_from=...)` migration? Does `clone_from` deep-copy or share state, and how does that interact with the spec's "compose multiple schematics into one canvas" use case?
2. **PR 5** — Should `[composer]` pin pypdf strictly below 7.0 until the migration is in, or migrate first and pin `>=` to the migration release? (Driven by upstream's release timing.)
3. **PR 5** — When a user supplies a PDF schematic instead of an SVG, do we still go through cairosvg as a normalization step, or branch the ingest path? (Spec is silent; spike data shows pypdf-only is sufficient for PDF-source schematics.)
4. **PR 5** — How do golden-file tests handle the empirical libcairo2 question? Is there a fixture-rendering CI image where libcairo2 is explicitly absent / present, to detect regression of the cairocffi-bundled-binding assumption?
5. **PR 6** — Adopt lxml or stick with stdlib ElementTree + manual namespace registration? Decision belongs in PR 6 once the namespace-handling code is being written; both are tractable.
6. **PR 6** — What is the canonical `_resolve_svg_units()` signature? It needs to handle: matplotlib SVG (pt viewBox), Inkscape SVG (mm or px viewBox), Illustrator SVG (px viewBox with `width="80mm"`), and the case where `width`/`height` and `viewBox` disagree on units.
7. **PR 6** — How should the composer behave when a schematic SVG declares no `viewBox`? Reject? Synthesize one from `width`/`height`? Spec doesn't say.
8. **PR 6** — Does the in-tree composer need a CSS/`<style>`-merge step, or are inline attributes enough for the schematics publiplots users actually produce? Spike used inline-only fixtures; production fixtures may differ.

---

## 6. Files touched in this spike

```
spikes/composer/
├── README.md
├── composer-spike.md             # THIS file
├── fixtures/
│   ├── canvas_with_slot.geom.json
│   ├── canvas_with_slot.pdf
│   ├── canvas_with_slot.py
│   ├── canvas_with_slot.svg
│   ├── schematic_complex.svg
│   └── schematic_simple.svg
├── outputs/                       # gitignored (.gitignore:222)
│   ├── path_a_composed.pdf
│   ├── path_a_composed_complex.pdf
│   ├── path_b_complex.pdf
│   ├── path_b_simple.pdf
│   └── path_c_composed.svg
├── path_a_pypdf/
│   ├── compose.py
│   ├── report.json
│   └── validate.py
├── path_b_cairosvg/
│   ├── compose.py
│   ├── report.json
│   └── validate.py
└── path_c_in_tree_svg/
    ├── compose.py
    ├── report.json
    └── validate.py
```

**Committed:** all `.py`, `.md`, `.json`, and the fixture PDFs/SVGs under `fixtures/` (16 files total via `git ls-files spikes/composer/`).

**Gitignored** (per `.gitignore:222-224`):
- `spikes/composer/outputs/` (compositor outputs — regenerable from compose.py)
- `spikes/composer/**/__pycache__/`
- `spikes/composer/**/*.pyc`

`report.json` files under each path's directory ARE committed — they are the programmatic evidence record.
