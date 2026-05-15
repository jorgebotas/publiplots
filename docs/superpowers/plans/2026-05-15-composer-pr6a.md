# Composer PR 6a Implementation Plan — vector-SVG compositing pipeline

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land in-tree vector-SVG output for the Composer — `canvas.savefig('fig.svg')` produces a real, vector-preserving SVG with embedded SVG/raster schematics, via lxml + cairosvg. After this PR:

1. **`canvas.savefig('fig.svg')`** dispatches to a new `compositing/svg.py` module instead of raising `NotImplementedError`. The pipeline:
   - Renders matplotlib's empty-slot Figure to an SVG byte buffer with full byte-determinism (`plt.rcParams["svg.hashsalt"]` pinned + `metadata={"Date": None}`).
   - Parses the canvas SVG via **lxml** (preserves source namespace prefixes — fixes spike Finding 6 ElementTree `xlink:` → `ns4:` rewrite hazard).
   - For each PanelImage panel: loads the schematic to an SVG element (cairosvg-rasterize for raster sources; in-tree splice for SVG; cairosvg-PDF→SVG round-trip for PDF), computes the mm→canvas-user-unit transform via the new `_resolve_svg_units()` helper (fixes spike Finding 7), wraps the schematic content in a `<g transform="...">` element, and inserts it into the canvas tree.
   - Serializes via `lxml.etree.tostring(..., pretty_print=False, xml_declaration=True)` → bytes → write to disk.
2. **`compositing/svg.py`** — new module mirroring `compositing/pdf.py`'s public surface: `savefig_svg(figure, path, *, panels, strict_vectors, metadata_date)`.
3. **`_resolve_svg_units()`** — new helper in `compositing/_geometry.py` (extends, doesn't replace, the PDF-only helpers). Detects canvas viewBox unit + schematic viewBox dims independently of `width`/`height` declarations; returns canvas-user-unit transforms (Finding 7).
4. **Spike Finding 2 SVG determinism contract** — the SVG path passes `metadata={"Date": None}` on `fig.savefig` AND sets `plt.rcParams["svg.hashsalt"]` to a deterministic string before `savefig` (otherwise `<defs>` IDs randomize on every render).
5. **`strict_vectors`** flag continues to gate fallback behavior. For SVG output: `True` raises `ComposerVectorError` on schematic-load failure; `False` rasterizes the schematic to an embedded `<image>` data-URI and emits `UserWarning`. Same shape as PR 5's PDF path.
6. **Golden-SVG tests** populate `tests/composer/golden/svg/` (NEW directory). Modes mirror PR 5's `assert_pdf_matches`: `viewbox` (canvas SVG `viewBox` matches `figure_size_mm` × user-unit factor); `structure` (the schematic `<g>` wrapper exists in the output XPath); `render_compare` (rasterize both via cairosvg at 200 DPI, compare via `compare_images`). Note: `viewbox` and `render_compare` are the meaningful modes for SVG; `structure` here checks XPath presence of the inserted `<g>`, distinct from the PDF-XObject check.
7. **`_pillow_to_pdf_bytes` extraction** — pull the duplicated Pillow→PDF logic from `_resources.py:117-128` and `pdf.py:_raster_fallback` into a shared helper (deferred follow-up from PR 5 code-quality review). The SVG path also uses Pillow→PDF for raster sources (transitively, through cairosvg's PDF normalization), so the extraction helps the SVG path immediately.

**Architecture:** All SVG compositing lives under `src/publiplots/composer/compositing/` (existing subpackage from PR 5). The public surface is the existing `canvas.savefig()` — no new top-level API. PR 6a ships only `svg.py` + extends `_geometry.py` and `_resources.py`; `embed_figure` (PR 6b), raster polish + TIFF + CMYK + `save_multiple` (PR 6b) are explicitly OUT.

```
src/publiplots/composer/
├── _save.py                           # MODIFY — replace NotImplementedError SVG branch with dispatch into compositing.svg.savefig_svg
└── compositing/
    ├── __init__.py                    # MODIFY — re-export savefig_svg
    ├── svg.py                         # NEW — savefig_svg + helpers (mirror pdf.py shape)
    ├── _resources.py                  # MODIFY — extract _pillow_to_pdf_bytes; add load_schematic_as_svg_element
    ├── _geometry.py                   # MODIFY — add _resolve_svg_units + compute_svg_transform helpers
    └── pdf.py                         # MODIFY (tiny) — call _pillow_to_pdf_bytes from _resources.py instead of inline duplication
```

**Tech Stack:** **lxml ≥ 4.9** (NEW dependency in `[composer]` extra). cairosvg ≥ 2.7 already present from PR 5. **lxml is a NET-NEW install footprint** (~5 MB wheel) — verified empirically: `uv pip show lxml` returns "Package(s) not found" and cairosvg's `Requires` is `cairocffi, cssselect2, defusedxml, pillow, tinycss2` (cairosvg uses `defusedxml` + stdlib ElementTree internally, NOT lxml). Decision: **adopt lxml directly** despite the new install cost. Rationale: (a) spike Finding 6 is structurally about ET dropping `xlink:` prefixes — manually pre-registering with `ET.register_namespace('xlink', '...')` works for known namespaces but breaks for any new namespace upstream introduces; lxml's source-prefix preservation is the production-safe choice. (b) lxml's XPath + namespace-map machinery makes `assert_svg_matches` mode='structure' clean instead of brittle. (c) The `[composer]` install is opt-in, so core users don't pay this cost. Document in CHANGELOG that the install footprint grew.

**Spec reference:** `docs/superpowers/specs/2026-05-14-composer-design.md`:
- §"PR 6" contract (lines 650-653) — note PR 6a covers ONLY the SVG composer slice; embed_figure + raster polish defer to PR 6b
- §"Architecture" (lines 80-100) — compositing-is-post-savefig invariant
- §"Inviolate invariants" (line 104) — `canvas.savefig(p)` byte-deterministic given identical state, modulo `/CreationDate` (PDF) and `Date`/`hashsalt` (SVG)

**Spike reference:** `spikes/composer/composer-spike.md`:
- Finding 2: SVG byte-determinism — `metadata={"Date": None}` + `plt.rcParams["svg.hashsalt"] = "<deterministic>"` (BOTH required).
- Finding 6: ElementTree namespace rewrite hazard → use lxml.
- Finding 7: Path C unit-conversion bug → `_resolve_svg_units()` helper.
- Open question 7 (no `viewBox`): we will REJECT — `ComposerVectorError` with a clear message asking the user to add a viewBox or convert to PDF/PNG. Inkscape-only authoring is a follow-up if users hit it.
- Open question 8 (`<style>` merging): inline-attributes-only for PR 6a. CSS/`<style>` merge stays follow-up; document the limitation in the docstring.

---

## What's IN scope for PR 6a

- **`compositing/svg.py`** — new module:
  - `savefig_svg(figure, path, *, panels, strict_vectors, metadata_date)` — public entry point. Mirrors `savefig_pdf`'s shape exactly so `_save.dispatch_savefig` can call it identically.
  - `_compose_panel_into(canvas_tree, panel, *, strict_vectors)` — for each PanelImage, parse the schematic, build a `<g transform="...">` wrapper, append to the canvas root.
  - `_raster_fallback_to_image_element(path, slot_bbox_user)` — embeds a base64 `xlink:href="data:image/png;base64,..."` `<image>` element when `strict_vectors=False` and a vector load fails. Same fallback ergonomics as the PDF path.
- **`compositing/_geometry.py`** extensions:
  - `_resolve_svg_units(svg_root) → (vb_x: float, vb_y: float, vb_w: float, vb_h: float, mm_per_user_unit: float)` — inspect `viewBox` (`min-x min-y width height`, always in user units) + `width`/`height` attribute UNITS to deduce mm-per-user-unit. **Returns viewBox X/Y ORIGIN as well** — non-zero-origin viewBoxes (`viewBox="10 -5 100 80"`) are common in Inkscape/Illustrator-authored SVGs, and the canvas-side X/Y origin participates in the wrapper `<g>` translate calculation. Handles 4 unit cases: pt viewBox (matplotlib), mm viewBox (Inkscape), px viewBox (Illustrator + most browser-authored), unit-less (assume px → 96 DPI). Raises `ComposerVectorError` if no viewBox. Also raises on relative units (`em`, `ex`, `%`) on `width`/`height` since they require font-size context we can't resolve. Disagreement between `width`-derived and `height`-derived mm-per-user-unit by > 1% emits `UserWarning` and prefers `width`.
  - `compute_svg_transform(slot_bbox_mm, schematic_size_mm, canvas_mm_per_user_unit, canvas_vb_origin, *, align, clip) → (sx, sy, tx_user, ty_user)` — analogous to `compute_pdf_transform` but in canvas user-unit space and with **SVG y-axis TOP-DOWN** (vs PDF bottom-up). Takes `canvas_vb_origin = (vb_x, vb_y)` so non-zero-origin canvas viewBoxes are handled correctly. Same 9 align values × 3 clip values matrix.
  - Reuse the existing `_VALID_ALIGN` set + the same defensive align validation when `clip != 'stretch'`.
- **`compositing/_resources.py`** extensions:
  - `_pillow_to_pdf_bytes(path: Path, dpi: float = _DEFAULT_RASTER_DPI) → bytes` — extracted helper. Both `load_schematic_as_pdf_bytes` (existing, for raster→PDF) and `pdf.py:_raster_fallback` consume it. ~10 LOC of dedup eliminated.
  - `_pillow_to_data_uri(path: Path, dpi: float = _DEFAULT_RASTER_DPI) → str` — NEW helper. Returns `"data:image/png;base64,..."` for embedding into SVG `<image>` elements. Reads via Pillow → re-saves to PNG bytes → base64-encodes (NOT pass-through; Illustrator-exported PNGs may have non-sRGB profiles that break some browsers; re-saving normalizes).
  - `load_schematic_as_svg_element(path) → (lxml.etree._Element, intrinsic_size_mm: Tuple[float, float], source_kind: str)`:
    - `.svg`: parse via `lxml.etree.parse(...)`. Take the root, call `_resolve_svg_units` to get intrinsic size, return `(root, (width_mm, height_mm), 'vector')`. Detach from any parent.
    - `.pdf`: cairosvg can't read PDF. Round-trip: call `pypdf` to extract page dims (we already do this in `_resources.py:60-65`), then... actually, **PDF-source schematics CANNOT be cleanly embedded in SVG without rasterizing** (no library round-trips PDF→SVG with vector preservation; pdf2svg via inkscape is out-of-process and adds a system dep). **Decision:** PDF-source schematics in SVG output go through the **raster fallback** path (Pillow re-render via `pdf2image` if available, else `ComposerVectorError` if `strict_vectors=True`, else `ComposerVectorError` always — pdf2image is NOT in `[composer]` extras). Document this clearly in `PanelImage` docstring + composer-guide.
    - `.png`/`.jpg`/etc: build an `<image>` element with `xlink:href="data:image/png;base64,..."` via `_pillow_to_data_uri`. Return `(image_element, (width_mm_at_dpi, height_mm_at_dpi), 'raster')`.
- **`Canvas.savefig('fig.svg')` wiring** — `_save.dispatch_savefig` `_VECTOR_SVG_EXTS` branch replaces the `NotImplementedError` with:
  ```python
  from publiplots.composer.compositing.svg import savefig_svg
  savefig_svg(figure, p, panels=list(panels), strict_vectors=strict_vectors,
              metadata_date=metadata_date, **kwargs)
  ```
  Note: `metadata_creation_date` (PDF) and `metadata_date` (SVG) are DIFFERENT kwargs — the PDF metadata key is `/CreationDate`, the SVG metadata key is `Date`. They flow through dispatch as separate named params.
- **`dispatch_savefig` signature** — add `metadata_date: Optional[str] = None` named kwarg alongside the existing `metadata_creation_date`. Both consumed by their respective branches; neither forwarded into `**kwargs`. Update docstring.
- **SVG byte-determinism contract** — inside `savefig_svg`:
  ```python
  if metadata_date is None:
      date_value = _DEFAULT_DATE
  elif metadata_date == "omit":
      date_value = None  # mpl strips <dc:date> when key is None
  else:
      date_value = metadata_date  # caller-supplied literal
  with plt.rc_context({"svg.hashsalt": _SVG_HASHSALT}):
      figure.savefig(canvas_buf, format="svg",
                     metadata={"Date": date_value},
                     **savefig_kwargs)
  ```
  with `_SVG_HASHSALT = "publiplots-composer"` and `_DEFAULT_DATE = "2026-01-01T00:00:00"`. **Kwarg semantics:**
  - `metadata_date=None` (default) → write deterministic `_DEFAULT_DATE` literal.
  - `metadata_date="omit"` → matplotlib strips `<dc:date>` entirely (suppressed, NOT defaulted).
  - `metadata_date="<literal>"` → write that literal value.
  
  This avoids the ambiguity the architect flagged ("`None` could mean `auto-fill` OR `suppress` — neither is what we want by default"). Document all three cases in the kwarg's docstring. The `rc_context` ensures we don't leak the hashsalt globally.
- **`compositing/__init__.py`** — re-export `savefig_svg` alongside `savefig_pdf`.
- **Golden-SVG helper in `_helpers.py`**:
  - `assert_svg_matches(canvas, name, *, mode='viewbox', tol_user=0.5, tol_image=20)`:
    - `mode='viewbox'`: parse output via lxml, assert `viewBox` user-units match `canvas.figure_size_mm × mm_per_user_unit` to `tol_user` user-units.
    - `mode='structure'`: parse via lxml, run XPath `//svg:g[starts-with(@id, 'publiplots-panel-image-')]` and assert match count == number of PanelImage panels in the canvas. Wrapper `<g>` IDs follow the scheme `publiplots-panel-image-{idx}-{label_or_unlabeled}` where `idx` is the panel's position in the filtered `image_panels` list (0-based). Disambiguates `label=None`/`label=False` collisions and conforms to SVG `id`-must-be-unique constraint.
    - `mode='render_compare'`: rasterize both produced + golden via `cairosvg.svg2png` at 200 DPI, compare via `compare_images` with `tol=tol_image`.
  - Regen-on-missing semantics mirror `assert_pdf_matches` from PR 5.
  - `_svg_renderer_available()` helper for `render_compare` skip-when-cairosvg-missing (analogous to `_pdf_rasterizer_available`).
- **2 new gated golden SVGs** under `tests/composer/golden/svg/`:
  - `cell-2col-with-svg-schematic.svg` — one PanelImage SVG (the same `tests/composer/golden/fixtures/schematic.svg` from PR 5, with the `"PR5-svg-marker"` text marker preserved through the SVG composer).
  - `cell-2col-with-png-schematic.svg` — one PanelImage PNG (raster→data-URI path).
- **Reuse PR 5's compositions**: `cell-2col-with-svg-schematic` and `cell-2col-with-png-schematic` from `_compositions.py` are the same canvas builds; the goldens are just rendered to a different format. Add a `format=` parameter to the golden-fixture parametrize in tests, or split into `test_svg_compositing.py` mirroring `test_pdf_compositing.py`'s shape (the latter is cleaner).
- **`tests/composer/test_svg_compositing.py`** — parametrized over the 2 compositions × 3 modes, mirroring `test_pdf_compositing.py`'s structure (which is the canonical reference).
- **`tests/composer/test_compositing_geometry.py`** extension — add unit tests for `_resolve_svg_units` covering: pt viewBox (matplotlib), mm viewBox (Inkscape), px viewBox (Illustrator), missing viewBox → raises, viewBox+width/height disagreement (viewBox wins, with a `UserWarning` if width/height attribute carries an explicit unit that disagrees with the deduced mm-per-user-unit by > 1%). Plus `compute_svg_transform` for all 9 align × 3 clip combinations.
- **No-deps fallback test** — patches `lxml` import to fail; asserts a clear `ComposerVectorError` with `pip install publiplots[composer]` install hint. (lxml is a NEW dep — gate behind the install-extra branch like cairosvg/pypdf.)
- **Strict-vectors test** — `pp.Canvas(..., strict_vectors=True)` + an intentionally-corrupt SVG → `ComposerVectorError` for the SVG path. With `strict_vectors=False`, rasterizes to a `<image>` data-URI element + emits `UserWarning`.
- **Determinism regression test** — render the same canvas twice via `canvas.savefig('a.svg')` then `canvas.savefig('b.svg')`, assert byte-identical (within the metadata-date-pinning + hashsalt-pinning contract). This guards Finding 2.
- **One example**: extend `examples/composer/cell_2col_with_schematic.py` (from PR 5) to ALSO save to SVG. Render the SVG to `docs/images/composer/cell_2col_with_schematic.svg` (matplotlib's `sphinx-gallery` integration handles SVG natively).
- **Update PR 5 deferred follow-ups** — tick off `_pillow_to_pdf_bytes` extraction (now done in PR 6a).
- **CHANGELOG entry** under `[Unreleased] / Added`.
- **`publiplots-guide` skill update** — one-line addition mentioning vector SVG works (per spec line 689 — transitional bridge until PR 7's `composer-guide`).

## What's OUT of scope for PR 6a

- **`canvas.embed_figure(panel=, fig=)`** — PR 6b. The "kitchen sink" use case from the spec.
- **TIFF support / optional CMYK** — PR 6b raster polish.
- **`save_multiple(stem, formats=['pdf','svg','png'])`** — PR 6b. The current workaround `for ext in formats: canvas.savefig(...)` is sufficient for PR 6a.
- **`canvas.inspect()` + composer-guide skill** — PR 7.
- **`pp.Canvas` integration with `pp.legend(rows=, cols=, span=)`** — PR 7.
- **`<style>` / CSS merge for SVG schematics** — open question 8 from spike, deferred. Inline-attribute-only for PR 6a.
- **Inkscape-authored SVG without `viewBox`** — open question 7 from spike. We REJECT for now (`ComposerVectorError` with a clear "add a viewBox" message).
- **Removing `NotImplementedError` calls in `_save.py`** beyond the SVG branch — only the SVG branch needs removal; multi-format save dispatch is a PR 6b concern.

---

## Files touched

| File | Status | LOC est. |
|---|---|---|
| `src/publiplots/composer/compositing/svg.py` | NEW | ~250 |
| `src/publiplots/composer/compositing/_geometry.py` | MODIFY (extend) | +120 |
| `src/publiplots/composer/compositing/_resources.py` | MODIFY (extract + extend) | +60 / -10 |
| `src/publiplots/composer/compositing/__init__.py` | MODIFY (1-line) | +1 |
| `src/publiplots/composer/compositing/pdf.py` | MODIFY (1-line) | +1 / -10 |
| `src/publiplots/composer/_save.py` | MODIFY (replace NIE branch) | +12 / -5 |
| `pyproject.toml` | MODIFY (`[composer]` += `lxml>=4.9`) | +1 |
| `tests/composer/test_svg_compositing.py` | NEW | ~200 |
| `tests/composer/test_compositing_geometry.py` | MODIFY (extend) | +120 |
| `tests/composer/test_strict_vectors.py` | MODIFY (add SVG cases) | +40 |
| `tests/composer/golden/_helpers.py` | MODIFY (add `assert_svg_matches`) | +120 |
| `tools/composer/regen_fixtures.py` | MODIFY (add SVG diff signature) | +30 |
| `tests/composer/golden/svg/cell-2col-with-svg-schematic.svg` | NEW (gated) | — |
| `tests/composer/golden/svg/cell-2col-with-png-schematic.svg` | NEW (gated) | — |
| `examples/composer/cell_2col_with_schematic.py` | MODIFY (also save SVG) | +5 |
| `CHANGELOG.md` | MODIFY | +6 |
| `skills/publiplots-guide/SKILL.md` | MODIFY (1-line) | +1 |

**Total:** ~960 LOC + tests + 2 goldens. Current test count on `main` is **1340** (verified via `uv run pytest --collect-only -q`); PR 6a will land ~60 new tests for ~1400 total. PR 6a is smaller than PR 5 because the SVG path reuses PR 5's plumbing (`Panel.image_*` fields, `Canvas._strict_vectors`, dispatch wiring, golden-fixture parametrize harness).

---

## Tasks

### Task 1 — Extract `_pillow_to_pdf_bytes` helper + dedup PDF code path

- [x] Add `_pillow_to_pdf_bytes(path: Path, dpi: float = _DEFAULT_RASTER_DPI) → bytes` to `_resources.py`. Move the Pillow→PDF logic from `load_schematic_as_pdf_bytes` (raster branch) into the helper. Update the raster branch to call the helper.
- [x] Update `pdf.py:_raster_fallback` to call `_resources._pillow_to_pdf_bytes` instead of inlining Pillow.
- [x] Add `tests/composer/test_pillow_helper.py` (new, small): unit-test `_pillow_to_pdf_bytes` with PNG, JPG, TIFF inputs; assert bytes start with `b"%PDF-"` and the returned PDF's mediabox matches the expected size.
- [x] Verify all PR 5 PDF tests still pass.

### Task 2 — Add `_resolve_svg_units` + `compute_svg_transform` to `_geometry.py`

- [x] Implement `_resolve_svg_units(svg_root: lxml.etree._Element) → (vb_x: float, vb_y: float, vb_w: float, vb_h: float, mm_per_user_unit: float)`. Logic:
  - Read `viewBox` attribute (`min-x min-y width height`). Required — raise `ComposerVectorError` if missing.
  - Inspect `width`/`height` attributes for explicit units (`mm`, `pt`, `px`, `in`, `cm`, unitless).
  - Resolve mm-per-user-unit by parsing `width` (preferred) or `height` (fallback) and dividing by viewBox dimension.
  - If width/height unitless or absent: assume user units = px @ 96 DPI (mm-per-user-unit ≈ 25.4 / 96 = 0.2646).
  - If width/height present but their unit-derived mm-per-user-unit disagrees with another's by > 1%, emit `UserWarning` and prefer width.
  - Reject relative units (`em`, `ex`, `%`) on `width`/`height` — raise `ComposerVectorError("width/height with relative units {unit!r} cannot be resolved without font/parent context; use absolute units (mm/pt/px/cm/in) or remove the attributes to fall back to px @ 96 DPI.")`.
- [x] Implement `compute_svg_transform(slot_bbox_mm, schematic_size_mm, canvas_mm_per_user_unit, canvas_vb_origin, *, align, clip) → (sx, sy, tx_user, ty_user)`. Parallel to `compute_pdf_transform` but:
  - **SVG y-axis is TOP-DOWN** — `'top'`-words → 0, `'bottom'`-words → full slack (opposite of PDF).
  - Output is in canvas user-unit space (not pt).
  - `canvas_vb_origin = (vb_x, vb_y)` is added to the slot's user-unit position to handle non-zero-origin viewBoxes.
  - Same 9 × 3 align×clip matrix.
- [x] Reuse the existing `_VALID_ALIGN` set + defensive align validation when `clip != 'stretch'`.
- [x] **Failing tests first** in `test_compositing_geometry.py` covering: 4 unit cases for `_resolve_svg_units` (pt/mm/px/unitless), missing-viewBox raise, relative-unit raise (`em`/`%`), non-zero-origin viewBox round-trip, width/height-disagreement warning, all 9 × 3 align × clip combos for `compute_svg_transform` including non-zero canvas origin. Then implement.

### Task 3 — Add `lxml` to `[composer]` extras + import-failure path

- [x] Add `"lxml>=4.9"` to `pyproject.toml` `[composer]` block (alphabetized between `cairosvg` and `Pillow`).
- [x] Add an import-failure path in `compositing/svg.py` mirroring the cairosvg/pypdf patterns: try-except ImportError → raise `ComposerVectorError` with the install-extra hint.
- [x] **Failing test first**: `test_svg_compositing.py::test_no_lxml_raises_install_hint` patches `sys.modules['lxml']` to `None` to break subsequent imports, then `importlib.reload(publiplots.composer.compositing.svg)` so the patched ImportError actually bites (since `compositing/svg.py` will already have lxml in its namespace from the first import). Expect `ComposerVectorError` with the `pip install publiplots[composer]` substring.

### Task 4 — `load_schematic_as_svg_element` in `_resources.py`

- [x] Implement `load_schematic_as_svg_element(path) → (element, intrinsic_size_mm, source_kind)`:
  - `.svg`: lxml-parse, root element is the schematic. Use `_resolve_svg_units` for intrinsic size. Return `(root, (vb_w_mm, vb_h_mm), 'vector')`.
  - `.png`/`.jpg`/etc: build a synthetic `<image xlink:href="data:image/png;base64,..." width="..." height="..."/>` element via lxml. Use `_pillow_to_data_uri` (new helper, also in `_resources.py`). Compute intrinsic mm size from pixel dims × `(25.4 / dpi)`. Return `(image_element, (w_mm, h_mm), 'raster')`.
  - `.pdf`: explicitly raise `ComposerVectorError(f"PanelImage {label!r}: PDF schematic {path.name!r} cannot be embedded in SVG output. PDF→SVG vector round-trip is not supported by cairosvg or any in-tree library. Either (a) convert {path.name!r} to SVG first, or (b) use canvas.savefig('fig.pdf') for PDF output where this PanelImage works as expected.")` (panel label and output format both named in the error to make the failure mode obvious). Document the limitation in `PanelImage.path` docstring extension. **Strict-vectors+raster-fallback combo:** even with `strict_vectors=False`, PDF-in-SVG output errors. We do NOT auto-rasterize via pdf2image (poppler) because pdf2image is not in `[composer]`; that's a PR 7 polish if users hit it.
- [x] Implement `_pillow_to_data_uri(path, dpi=_DEFAULT_RASTER_DPI) → str`: open via Pillow → re-save to PNG bytes (normalizes color profile; Illustrator-exported PNGs sometimes have non-sRGB profiles that break browsers) → base64-encode → return `"data:image/png;base64,..."`.
- [x] **Failing tests first** in `test_compositing_resources.py` (extend the existing PR 5 file): all 3 branches × happy-path; `.pdf`-in-SVG raises; corrupt SVG raises wrapped `ComposerVectorError`; missing viewBox in SVG schematic raises (Task 2 helper coverage).

### Task 5 — `compositing/svg.py` — `savefig_svg` orchestrator

- [x] Implement `savefig_svg(figure, path, *, panels, strict_vectors=False, metadata_date=None, **savefig_kwargs)`:
  - Step 1: filter `panels` to image kind → `image_panels`.
  - Step 2: render Figure to SVG buffer via `plt.rc_context({"svg.hashsalt": _SVG_HASHSALT})` + `metadata={"Date": metadata_date or _DEFAULT_DATE}`.
  - Step 3: lxml-parse the canvas SVG buffer.
  - Step 4: for each PanelImage, call `_compose_panel_into(canvas_root, panel, strict_vectors=strict_vectors)`.
  - Step 5: serialize via `lxml.etree.tostring(canvas_tree, pretty_print=False, xml_declaration=True, standalone=False)` → write bytes to `path`.
- [x] Implement `_compose_panel_into(canvas_root, panel, *, strict_vectors)`:
  - Read `panel.image_path`, `panel.image_align`, `panel.image_clip`, `panel.bbox_mm`, `panel.label`.
  - Try `load_schematic_as_svg_element(path)`. On `ComposerVectorError`: if `strict_vectors=True`, re-raise; else emit `UserWarning` + call `_raster_fallback_to_image_element(path, slot_bbox_mm)`.
  - Compute mm-per-user-unit on the canvas via `_resolve_svg_units(canvas_root)`.
  - Compute transform via `compute_svg_transform(panel.bbox_mm, intrinsic_size_mm, canvas_mm_per_user_unit, align=, clip=)`.
  - Build `<g id="publiplots-panel-image-{idx}-{label_or_unlabeled}" transform="translate({tx}, {ty}) scale({sx}, {sy})">` wrapper, where `idx` is the panel's 0-based position in `image_panels` and `label_or_unlabeled = label if label not in (None, False) else 'unlabeled'`. (Mirrors the PDF orchestrator's `<unlabeled>` precedent at `pdf.py:154` while ensuring SVG `id` uniqueness.) Append the schematic element as a child.
  - Append the `<g>` to `canvas_root`.
- [x] Implement `_raster_fallback_to_image_element(path, slot_bbox_mm)` — last-ditch path: open via Pillow → re-save to PNG → base64-encode → build an `<image>` element sized to `slot_bbox_mm` (in canvas user units). On Pillow failure: raise `ComposerVectorError`.
- [x] **Failing tests first** in `test_svg_compositing.py`: golden-SVG match (mode=viewbox + structure); strict_vectors=True raises on corrupt SVG; strict_vectors=False warns + emits `<image>`; deterministic byte-equality across two saves.

### Task 6 — Wire `_save.dispatch_savefig` SVG branch + add `metadata_date` kwarg

- [x] Replace the SVG branch in `_save.py` with a `from publiplots.composer.compositing.svg import savefig_svg; savefig_svg(...)` call.
- [x] Add `metadata_date: Optional[str] = None` to `dispatch_savefig` signature, alongside the existing `metadata_creation_date`. Update docstring.
- [x] `Canvas.savefig` does NOT need a new kwarg — `metadata_date` flows through `**kwargs`. Verify by reading `canvas.py:946-983`. (PR 5 also kept `metadata_creation_date` out of the `Canvas.savefig` signature; it's a tests-only knob.)
- [x] **Failing test first** in `test_strict_vectors.py`: extend `test_strict_vectors_raises_on_corrupt_svg` to cover the SVG output path AND the PDF output path (currently PDF only).

### Task 7 — Golden-SVG infrastructure + assert_svg_matches

- [x] Add `assert_svg_matches(canvas, name, *, mode='viewbox', tol_user=0.5, tol_image=20)` to `tests/composer/golden/_helpers.py`. Modes: `viewbox`, `structure` (XPath presence of `//svg:g[@id='publiplots-panel-image-{label}']`), `render_compare` (cairosvg.svg2png both → compare_images).
- [x] Add `_svg_renderer_available()` helper for skip-when-cairosvg-missing on `render_compare` mode.
- [x] `tests/composer/golden/svg/.gitkeep` placeholder; goldens generated via regen workflow.
- [x] Extend `tools/composer/regen_fixtures.py`:
  - Add `_svg_structure_signature(svg_path) → (root_tag, viewbox, n_image_groups: int)` for non-byte-equality `--check` diff (analogous to `_pdf_structure_signature`).
  - Re-generate the 2 goldens under `tests/composer/golden/svg/` when the SVG path is touched.
- [x] **Failing tests first**: `test_golden_registry::test_assert_svg_matches_viewbox_mode_passes_for_known_good`, `test_assert_svg_matches_structure_mode_finds_panel_g`, `test_assert_svg_matches_render_compare_skips_when_no_cairosvg`.

### Task 8 — Golden SVG goldens + parametrized compositing tests

- [ ] Generate `cell-2col-with-svg-schematic.svg` and `cell-2col-with-png-schematic.svg` via `python tools/composer/regen_fixtures.py --only cell-2col-with-svg-schematic --only cell-2col-with-png-schematic`. Inspect manually:
  - SVG schematic golden: open in Firefox; verify the `"PR5-svg-marker"` text is selectable AND positioned in the schematic slot (NOT at world origin like spike Path C bug).
  - PNG schematic golden: open in Firefox; verify the `<image>` is in the slot, no aspect distortion (clip='fit' default).
- [ ] Commit both goldens.
- [ ] `test_svg_compositing.py` parametrized over `[(name, mode) for name in ['cell-2col-with-svg-schematic', 'cell-2col-with-png-schematic'] for mode in ['viewbox', 'structure', 'render_compare']]`. Note: structure mode IS meaningful for SVG (we ID-tag the wrapper `<g>`), unlike the PDF path where structure-on-SVG-source was rejected as defensively useless.

### Task 9 — Determinism regression guard + example + CHANGELOG + skill

- [ ] `test_svg_compositing.py::test_savefig_svg_byte_deterministic` — render twice; assert byte-identical.
- [ ] Extend `examples/composer/cell_2col_with_schematic.py` to ALSO save to SVG (`canvas.savefig(out / 'fig.svg')`).
- [ ] CHANGELOG `[Unreleased] / Added` entry.
- [ ] `skills/publiplots-guide/SKILL.md` — one-line addition: "vector SVG works (PR 6a)".
- [ ] **Final test run**: `uv run pytest tests/composer tests/test_legend_grid_scope.py -q` should report 1338 + new SVG tests = ~1400 total; 1 residplot pre-existing failure unchanged.

---

## Test plan summary

- `test_compositing_resources.py` — extend with SVG-element loader cases (3 source-ext branches × happy-path; PDF-in-SVG raises; corrupt SVG raises).
- `test_compositing_geometry.py` — extend with `_resolve_svg_units` (4 unit cases + missing-viewBox + disagreement warning) + `compute_svg_transform` (9 × 3 matrix; SVG-y-axis inversion vs PDF).
- `test_svg_compositing.py` (NEW) — orchestrator + parametrize over 2 compositions × 3 modes; strict_vectors raise/warn; no-deps fallback (lxml import patched); byte-determinism regression.
- `test_strict_vectors.py` — extend with SVG-output strict_vectors path.
- `test_pillow_helper.py` (NEW) — unit tests for the extracted `_pillow_to_pdf_bytes`.
- `test_golden_registry.py` — extend with `assert_svg_matches` mode tests.

---

## Architect blockers expected

These will be the first questions the architect raises. Pre-document positions:

1. **lxml in `[composer]`** — adds ~5 MB to install. Decision: yes, include directly. Rationale: cairosvg already pulls lxml transitively (cairocffi → cffi → ...); we're making the dep explicit, not new. Spike Finding 6 recommends lxml over stdlib ElementTree explicitly. ASK ARCHITECT to verify the transitive-dep claim empirically with `uv pip show lxml` post-install.
2. **PDF schematics in SVG output** — we explicitly reject (raise `ComposerVectorError`). Architect may ask: should we auto-rasterize via pdf2image when `strict_vectors=False`? Decision: NO, pdf2image is not in `[composer]`; the install footprint is poppler (system dep). Defer to PR 7 polish. Document the limitation in `PanelImage.path` docstring.
3. **`metadata_date` kwarg name** — collides visually with `metadata_creation_date` (PDF). Could combine into `metadata_timestamp` or similar. Decision: keep separate; matches matplotlib's actual key names (`Date` for SVG, `CreationDate` for PDF). Architect to confirm.
4. **SVG `<style>` / CSS merge** — open question 8 from spike. NOT in PR 6a scope (inline-attribute-only). Architect may ask: do real publiplots schematics use `<style>` blocks? Empirically, matplotlib's SVG output uses inline attributes for axes/lines, but Inkscape and Illustrator emit `<style>` blocks. Decision: defer; document the limitation.
5. **`_resolve_svg_units` returning mm-per-user-unit when viewBox is present but width/height absent** — spec says "user units = px @ 96 DPI" assumption. Architect may ask: is this the same convention as cairosvg's internal default? Decision: verify empirically during Task 2; if cairosvg uses 90 DPI (Cairo default before 1.16), match cairosvg's value to avoid render_compare drift.
6. **Strict-vectors fallback path emits `<image>` data-URI** — for SVG output, the fallback is "vector container with raster contents". Is that the right contract? Architect to confirm vs alternative (raise on raster-source-when-strict-vectors-required).

## Likely code-quality nits expected

- `savefig_svg` orchestrator may grow > 80 LOC; consider extracting `_render_canvas_to_svg_bytes` helper before review.
- `_resolve_svg_units` has 4-way unit branch; consider a small `_UNIT_TO_MM_PER_PX` lookup constant.
- `_compose_panel_into` and `_compose_panel_onto` (PDF) are structurally similar; tempting to extract a shared abstract base. RESIST — the PDF path uses `pypdf.Transformation` + `merge_transformed_page`; the SVG path uses an lxml `<g>` wrapper. Different enough that abstraction premature.
- `_pillow_to_data_uri` could colocate with `_pillow_to_pdf_bytes`. Both in `_resources.py`. Document the ordering.

---

## Conventional commits expected

- `refactor(composer): extract _pillow_to_pdf_bytes helper` (Task 1)
- `feat(composer): _resolve_svg_units + compute_svg_transform helpers` (Task 2)
- `chore(composer): add lxml to [composer] install extra` (Task 3)
- `feat(composer): load_schematic_as_svg_element in _resources` (Task 4)
- `feat(composer): savefig_svg orchestrator` (Task 5)
- `feat(composer): wire dispatch_savefig SVG branch + metadata_date` (Task 6)
- `test(composer): assert_svg_matches helper + golden-SVG infrastructure` (Task 7)
- `test(composer): cell-2col SVG goldens + compositing parametrize` (Task 8)
- `feat(composer): SVG byte-determinism regression test + example + skill update` (Task 9)
- `chore(changelog): PR 6a vector-SVG entry`

All with the em-dash convention used in PRs 1-5.
