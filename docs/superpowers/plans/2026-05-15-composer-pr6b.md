# Composer PR 6b Implementation Plan — `canvas.embed_figure` + raster polish

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the second slice of PR 6 — `canvas.embed_figure` (the spec's "kitchen sink" use case) + raster polish (TIFF compression, optional CMYK, `save_multiple`, optional external-sidecar raster for SVG). PR 6a already shipped the SVG composer + spike Findings 2/6/7 fixes; PR 6b builds on that infrastructure.

After this PR:

1. **`canvas.embed_figure(panel_label, fig)`** — post-staging method that attaches a `pp.subplots`-built (or any matplotlib) Figure into a previously-staged `PanelImage` slot. The Figure is rendered to a deterministic PDF/SVG byte buffer at compose time; the existing `compositing.pdf.savefig_pdf` and `compositing.svg.savefig_svg` orchestrators treat it like a schematic source.
2. **`PanelImage` accepts staging WITHOUT `path=`** — `pp.PanelImage(label='B', size=(70, 40))` is now valid IFF the panel will be filled by `canvas.embed_figure(...)` before savefig. **Sentinel choice (architect-fixed):** `path=""` cannot be used as the unfilled marker because `Path("")` resolves to `PosixPath('.')` (truthy, exists). Instead: change `PanelImage.path` default to `None: Optional[Union[str, Path]] = None`; the unfilled finalize state on `Panel` is `image_path: Optional[Path] = None`. Compositing pipelines check `panel.image_path is None and panel.embedded_figure is None` for the unfilled-AND-unembedded error, raising `ComposerVectorError("PanelImage 'B' has no path and no embedded figure; either pass path= at construction or call canvas.embed_figure(label, fig) before savefig.")`.
3. **`canvas.save_multiple(stem, formats=None, **kwargs)`** — sugar over `for ext in formats: canvas.savefig(f"{stem}.{ext}")`. Reuses the same `panels`/`strict_vectors` pipeline; equivalent to a Python loop but ergonomic for the journal-submission use case (Cell wants `.pdf`+`.tif`, Nature wants `.eps`+`.pdf`, etc). **Signature deliberately mirrors `pp.save_multiple` (the existing free function at `utils/io.py:140`)**: `formats=None` defaults to `['png', 'pdf']`. The Canvas method is the multi-panel-aware counterpart; `pp.save_multiple` continues to work on plain matplotlib figures. Document the parallel.
4. **CMYK output (raster only)** — `canvas.savefig(path, *, cmyk=False, tiff_compression='tiff_lzw')`:
   - `cmyk=True` + raster ext (`.tif`/`.tiff`/`.jpg`/`.jpeg`) → Pillow converts RGB→CMYK on save with the journal-default profile (FOGRA39 / U.S. Web Coated SWOP v2; verify which is the empirical journal preference at architect time).
   - `cmyk=True` + `.pdf` or `.svg` → raise `ValueError("cmyk=True is only valid for raster outputs (.tif/.tiff/.jpg/.jpeg); matplotlib's PDF/SVG backends emit RGB, and cairosvg cannot produce CMYK. Convert the matching raster output instead.")`.
   - `cmyk=True` + `.png` → raise `ValueError("PNG does not support CMYK; use .tif/.tiff/.jpg/.jpeg instead.")`.
   - `tiff_compression='tiff_lzw'` default for TIFF; user can pass `'tiff_deflate'`, `'raw'`, etc (Pillow's tiff compression vocab). Other ext branches ignore.
5. **External-sidecar raster for SVG (optional)** — `canvas.savefig(path, *, external_raster=False)`:
   - Default `False` (current behavior): SVG raster fallbacks emit inline base64 data-URI `<image>` elements.
   - `True`: when the SVG composer hits a raster source (PNG/JPG/etc), write the source to a sidecar PNG at `path.parent / f"{path.stem}-{idx}-{label}.png"` and emit `<image href="<sidecar>">` instead of inline. Avoids the ~5MB+ SVG bloat for high-DPI rasters.
   - `True` + non-SVG output → silent no-op (nothing to externalize for PDF, which already handles raster via bytes).

**Architecture:** All embed_figure plumbing flows through the existing PR 5/PR 6a compositing surface. `Panel` result records gain an `embedded_figure: Optional[Figure] = None` field; `compositing/pdf.py` and `compositing/svg.py` orchestrators check it before falling back to `image_path`-based loading. NO new compositing module; NO new top-level API beyond the 3 named methods (`embed_figure`, `save_multiple`) + 3 kwargs (`cmyk`, `tiff_compression`, `external_raster`).

```
src/publiplots/composer/
├── canvas.py                          # MODIFY — add embed_figure + save_multiple methods; thread cmyk + tiff_compression + external_raster through savefig
├── panels.py                          # MODIFY — relax PanelImage path-required validation; add embedded_figure field to Panel
├── _save.py                           # MODIFY — accept new kwargs; thread to compositing pipelines
└── compositing/
    ├── pdf.py                         # MODIFY — embedded_figure branch in _compose_panel_onto
    ├── svg.py                         # MODIFY — embedded_figure branch in _compose_panel_into; external_raster sidecar emission
    ├── _resources.py                  # MODIFY — add render_figure_to_pdf_bytes + render_figure_to_svg_element helpers
    └── _embed.py                      # NEW — Figure→PDF-bytes and Figure→SVG-element helpers (~80 LOC)
```

**Tech Stack:** No new dependencies. Reuses pypdf + cairosvg + lxml + Pillow already in `[composer]` extra. CMYK conversion uses Pillow's built-in `Image.convert('CMYK')` (FOGRA39 ICC profile lookup happens via `PIL.ImageCms` if available; degrades gracefully to Pillow's basic RGB→CMYK when ImageCms not present).

**Spec reference:** `docs/superpowers/specs/2026-05-14-composer-design.md`:
- §"PR 6" contract (lines 650-653) — embed_figure is the "kitchen sink" use case; raster polish + save_multiple round out the savefig matrix.
- §"Architecture" (lines 80-100) — compositing-is-post-savefig invariant.
- §"Inviolate invariants" (line 104) — `canvas.savefig(p)` byte-deterministic given identical state. embed_figure must preserve this: render the Figure into a bytes buffer using the same `metadata={"CreationDate": None}` (PDF) / `metadata={"Date": ...}` + `svg.hashsalt` (SVG) determinism contract that `savefig_pdf`/`savefig_svg` already use for the canvas Figure itself.

**User-decided design at plan time** (locked in pre-architect):
- `embed_figure(panel_label, fig)` post-staging mutation API (NOT `PanelImage(figure=fig)` construction-time, NOT a new `PanelEmbed` panel kind).
- `save_multiple(stem, formats=[...])` IS added — explicit user request despite YAGNI alternative.
- `cmyk=True` paired with PDF/SVG → raise (loud rejection over silent no-op).

---

## What's IN scope for PR 6b

- **`Canvas.embed_figure(self, label: Union[str, int], figure: matplotlib.figure.Figure) → None`** — public method:
  - Resolves `label` via the same lookup as `canvas[label]` — accepts str (resolved label) or int (insertion index). Raises `KeyError` on miss, mirroring `__getitem__`.
  - The resolved `Panel` MUST have `kind == 'image'` — else `TypeError("embed_figure: target panel 'B' is kind='axes', not 'image'; embed_figure only attaches to PanelImage slots.")`.
  - The resolved `Panel.embedded_figure` MUST be currently `None` — else `RuntimeError("embed_figure: panel 'B' already has an embedded figure; embed_figure is one-shot per panel.")`. Avoids silent overwrites.
  - Sets `Panel.embedded_figure = figure`. Frozen dataclass → use `object.__setattr__` (matches the precedent in `PanelImage.__post_init__`'s path-normalization).
  - Triggers `_finalize_if_needed()` — embed_figure on an empty canvas raises a clear `RuntimeError`.
  - Note: `embed_figure` does NOT validate that `figure` is a real `matplotlib.figure.Figure` instance via `isinstance` — duck-typing is enough; the compositing pipeline will call `figure.savefig(buf, format=...)` which fails loudly on non-Figure inputs.
- **`PanelImage.__post_init__` relaxation** — `path=None` is now legal at construction:
  - Change default: `path: Optional[Union[str, Path]] = None` (was `path: Union[str, Path] = ""`).
  - `__post_init__`: `if self.path is None: return` (skip ext + exists checks); else normalize to `Path` as before.
  - The "unfilled" sentinel is `path is None`. Document that an unfilled PanelImage MUST be paired with `canvas.embed_figure(label, fig)` before savefig.
  - Validation moves to **compose-time**: when Canvas materializes a Panel for an unfilled PanelImage, it stores `image_path=None`. The compositing pipelines check `panel.image_path is None and panel.embedded_figure is None` → raise `ComposerVectorError("PanelImage 'B' has no path and no embedded figure...")`.
  - Existing path-required tests in `test_panel_image.py` get a deprecation-style update: `PanelImage(label='A', size=(70,40))` (no path) is now valid construction; the save-time error is the new contract surface.
- **`Panel` result dataclass** — add `embedded_figure: Optional[matplotlib.figure.Figure] = None` field. Threaded through `Canvas._finalize_if_needed` from the staged PanelImage (default None); mutated by `Canvas.embed_figure` post-finalize via `object.__setattr__`.
- **`compositing/_embed.py`** — new module with two pure helpers:
  - `render_figure_to_pdf_bytes(figure: Figure, *, metadata_creation_date: Optional[str], **savefig_kwargs) → bytes` — render to a `BytesIO` PDF buffer with `metadata={"CreationDate": None}` and pinned producer; return bytes. Mirrors the determinism contract that `savefig_pdf` uses for the canvas Figure.
  - `render_figure_to_svg_bytes(figure: Figure, *, metadata_date: Optional[str], svg_hashsalt: str, **savefig_kwargs) → bytes` — render to a `BytesIO` SVG buffer wrapped in `plt.rc_context({"svg.hashsalt": svg_hashsalt})` + `metadata={"Date": ...}`. Mirrors the SVG determinism contract.
  - Both helpers reuse the constants from `compositing/svg.py` (`_SVG_HASHSALT`, `_DEFAULT_DATE`) and `compositing/pdf.py` (`_DEFAULT_CREATION_DATE`) — extract the constants into a shared `_constants.py` to avoid circular imports.
  - **Determinism gotcha for `pp.subplots`-built figures (architect-flagged):** `pp.subplots` attaches a `SubplotsAutoLayout` reactor to the figure (`fig._publiplots_auto_layout`). The reactor measures decoration tightbboxes at draw-time and resizes the figure mid-render, which can produce non-deterministic output if the measure depends on font cache state. Mitigation in `render_figure_to_*_bytes`: BEFORE calling `figure.savefig(...)`, explicitly call `figure.canvas.draw()` once to settle the reactor (or detect `_publiplots_auto_layout` and call its `finalize()` method if available); document that the reactor is allowed to run AT MOST ONCE per render, not multiple times. Add a determinism regression test that renders the same `pp.subplots`-built figure 100 times and asserts byte-identity.
- **`compositing/pdf.py:_compose_panel_onto` extension** — add an early branch:
  ```python
  if panel.embedded_figure is not None:
      # Render the embedded figure to a deterministic PDF buffer + treat
      # like a vector schematic source.
      pdf_bytes = render_figure_to_pdf_bytes(panel.embedded_figure)
  else:
      try:
          pdf_bytes, _kind = load_schematic_as_pdf_bytes(path)
      except ComposerVectorError as e:
          ...  # existing strict_vectors fallback
  ```
- **`compositing/svg.py:_compose_panel_into` extension** — analogous branch:
  ```python
  if panel.embedded_figure is not None:
      svg_bytes = render_figure_to_svg_bytes(panel.embedded_figure, ...)
      # Parse via lxml; root is the embedded figure's SVG root; treat
      # like a vector schematic.
      sch_element = lxml.etree.fromstring(svg_bytes)
      sch_size_mm = _resolve_svg_units(sch_element)[2:4] * mm_per_uu
      sch_kind = "vector"
  else:
      try:
          sch_element, sch_size_mm, sch_kind = load_schematic_as_svg_element(path, label=raw_label)
      except ComposerVectorError:
          ...  # existing strict_vectors fallback
  ```
- **`Canvas.save_multiple(self, stem: Union[str, Path], formats: Optional[Sequence[str]] = None, **kwargs) → List[Path]`** — sugar method:
  - Default `formats=None` → `['png', 'pdf']` (parity with `pp.save_multiple` free function at `utils/io.py:140-188`).
  - **Pre-validate** `formats` BEFORE writing any file: validate non-empty, all entries are str, no duplicates (architect-suggested), each ext is in `_RASTER_EXTS | _VECTOR_PDF_EXTS | _VECTOR_SVG_EXTS`. Raises `ValueError` with the offending ext named. Architect noted: this DEVIATES from `pp.save_multiple`'s permissive forward-and-let-matplotlib-raise; document the divergence in the docstring.
  - For each `ext` in `formats`, build `path = Path(stem).with_suffix(f".{ext.lstrip('.')}")` and call `canvas.savefig(path, **kwargs)`.
  - Returns the list of paths written.
  - kwargs are forwarded to ALL format iterations (so `cmyk=True` + `formats=['tif', 'pdf']` errors loudly on the PDF iteration AT canvas.savefig PRE-VALIDATION; the pre-validate above catches `cmyk=True + 'pdf' in formats` BEFORE writing the .tif. Document this partial-write-prevention.).
- **`Canvas.savefig(path, *, cmyk=False, tiff_compression='tiff_lzw', external_raster=False, **kwargs)`** — extend the public signature:
  - Pre-validate `cmyk` against ext: raise `ValueError` for PDF/SVG/PNG (PNG does not support CMYK).
  - Thread `cmyk`/`tiff_compression`/`external_raster` to `_save.dispatch_savefig` as named kwargs.
- **`_save.dispatch_savefig` extension** — accept `cmyk: bool = False`, `tiff_compression: str = "tiff_lzw"`, `external_raster: bool = False` as **named kwargs** alongside the existing `panels`/`strict_vectors`/`metadata_creation_date`/`metadata_date`. **Critical:** these kwargs MUST be CONSUMED at the dispatch layer (i.e., named-kwarg slot, not floating in `**kwargs`). The existing PR 5/6a code already uses this pattern for `panels`/`strict_vectors`/`metadata_creation_date`/`metadata_date`. If `cmyk`/`tiff_compression`/`external_raster` were left in `**kwargs`, they would leak to `pp.savefig` (raster branch), `savefig_pdf` (PDF), and `savefig_svg` (SVG), all of which forward `**kwargs` to `figure.savefig` → matplotlib raises `AttributeError` for unknown kwargs. Explicit consumption mapping:
  - **Raster branch**: consumes `cmyk` + `tiff_compression`; passes neither to `pp.savefig`.
  - **PDF branch (`savefig_pdf`)**: consumes none of the new 3 kwargs (PDF can't carry CMYK; sidecar PNGs are SVG-specific). If user passed `cmyk=True` here it should have been pre-validated by `Canvas.savefig`; defensively, `dispatch_savefig` raises if `cmyk=True` reaches the PDF branch (belt-and-braces).
  - **SVG branch (`savefig_svg`)**: consumes `external_raster`; passes neither `cmyk` nor `tiff_compression`. Same defensive raise on `cmyk=True`.
- **Raster CMYK conversion** — `_save.py` raster branch:
  - When `cmyk=True` and ext in `{.tif, .tiff, .jpg, .jpeg}`:
    - Render via `pp.savefig` to a temp `BytesIO` PNG buffer.
    - Open via Pillow → `Image.convert('CMYK')` (Pillow accepts RGBA→CMYK directly per architect verification; built-in sRGB→CMYK with no ICC profile in PR 6b. FOGRA39 / U.S. Web Coated SWOP v2 ICC profile bundling deferred to PR 7. Emit `UserWarning` when CMYK is requested without ImageCms profile so journal QA users can debug rejected submissions).
    - **Save with the correct Pillow format string**: NOT `ext.upper()` (architect found this fails for `.tif` → Pillow expects `'TIFF'`). Use an explicit ext→format mapping: `{".tif": "TIFF", ".tiff": "TIFF", ".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG"}`. Only TIFF accepts `compression=tiff_compression`; JPEG ignores it.
  - When `cmyk=False`: existing path unchanged for non-TIFF; for TIFF with non-default `tiff_compression`, render to PNG buffer and re-save as TIFF with the requested compression.
- **TIFF compression** — when ext is `.tif`/`.tiff` and `cmyk=False`:
  - `pp.savefig` doesn't currently expose a `pil_kwargs` knob for tiff_compression. Decision: post-render via PIL — render to PNG buffer, re-save as TIFF with the requested compression. (Architect to verify whether `pp.savefig` already supports `pil_kwargs={"compression": ...}`; if so, prefer the direct path and skip the re-render.)
- **External-sidecar raster for SVG** — `compositing/svg.py:_compose_panel_into` raster branch:
  - When `external_raster=True` is threaded through, bypass `_pillow_to_data_uri` and write `path.parent / f"{output_stem}-{idx}-{label}.png"` instead. Use Pillow `Image.save(format='PNG')`.
  - Emit `<image href="<sidecar-relative-path>">` element. Use a relative path so the SVG is portable when the parent dir is moved.
  - The SVG composer's `_raster_fallback_to_image_element` gains a `external_raster: bool` + `output_stem: Optional[Path]` param.
- **Tests:**
  - `test_embed_figure.py` (NEW) — embed_figure happy paths (PDF + SVG outputs), error cases (kind mismatch, double-embed, empty-canvas, panel not found), determinism regression (embedding the same figure twice produces byte-identical canvases).
  - `test_save_multiple.py` (NEW) — formats list happy path, empty-formats raise, unknown-ext raise, kwargs threaded (`cmyk=True` + mixed formats raises mid-loop).
  - `test_cmyk_output.py` (NEW) — `cmyk=True` + TIFF/JPEG round-trip preserves dimensions; `cmyk=True` + PDF/SVG/PNG raises ValueError with helpful message; tiff_compression flows through and produces smaller files for `tiff_lzw` vs `raw`.
  - `test_external_raster.py` (NEW) — sidecar PNG written; SVG XPath finds `<image href>` not inline; non-SVG ext silent-no-op.
  - `test_panel_image.py` extension — empty-path PanelImage construction now legal; finalize-time error if no embed; existing path-required tests stay valid (path now constructed but the PanelImage with path also works the same way as before).
  - `test_pdf_compositing.py` + `test_svg_compositing.py` extensions — embed_figure variant of an existing golden composition.
- **2-3 new gated goldens** — `cell-2col-with-embed-figure.{pdf,svg}` (PR 6b's headline composition: one PanelAxes scatter + one embed_figure of a `pp.subplots()`-built lineplot).
- **One example** — `examples/composer/cell_2col_with_embed.py` (kitchen-sink): build a separate `pp.subplots()` figure, embed into a Canvas slot, save to PDF + SVG.
- **CHANGELOG entry** under `[Unreleased] / Added`.
- **`publiplots-guide` skill update** — one-line addition mentioning embed_figure works (per spec line 689 transitional bridge until PR 7's `composer-guide`).

## What's OUT of scope for PR 6b

- **`canvas.inspect()` + composer-guide skill** — PR 7. The inspect schema + ReadTheDocs gallery integration ships there.
- **`pp.legend(rows=, cols=, span=)` × Canvas integration** — PR 7. Requires `pp.Canvas` to attach `_publiplots_axes` on the figure.
- **PDF→SVG vector round-trip via pdf2image (poppler)** — explicit PR 7 polish if users hit the PR 6a "PDF in SVG output" rejection.
- **`<style>` / CSS-block merge for SVG schematics** — open question 8 from spike, deferred to PR 7 + composer-guide warning.
- **Cross-figure embed_figure consistency** — embedding the SAME figure into TWO canvases does NOT mutate the original; embed_figure simply stores a reference. Document but don't enforce immutability of the figure.
- **`embed_figure` validation that Figure has been finalized** — embed_figure accepts a figure mid-render; the user is responsible for ensuring `fig.tight_layout()` has NOT been called (per the global rule [[feedback_plots_no_tight_layout]]).
- **Promoting `cmyk` / `tiff_compression` / `external_raster` to global rcParams** — they stay savefig-time kwargs.

---

## Files touched

| File | Status | LOC est. |
|---|---|---|
| `src/publiplots/composer/canvas.py` | MODIFY (add 2 methods, extend savefig) | +90 |
| `src/publiplots/composer/panels.py` | MODIFY (relax PanelImage path; add embedded_figure to Panel) | +20 / -5 |
| `src/publiplots/composer/_save.py` | MODIFY (thread 3 new kwargs; raster branch CMYK + TIFF compression) | +60 |
| `src/publiplots/composer/compositing/_constants.py` | NEW (extract `_DEFAULT_CREATION_DATE`, `_DEFAULT_DATE`, `_SVG_HASHSALT`) | ~10 |
| `src/publiplots/composer/compositing/_embed.py` | NEW (`render_figure_to_pdf_bytes` + `render_figure_to_svg_bytes`) | ~80 |
| `src/publiplots/composer/compositing/pdf.py` | MODIFY (embedded_figure branch) | +20 |
| `src/publiplots/composer/compositing/svg.py` | MODIFY (embedded_figure branch + external_raster sidecar) | +40 |
| `src/publiplots/composer/compositing/_resources.py` | MODIFY (re-export embed helpers? OR keep in _embed.py) | +5 / -0 |
| `tests/composer/test_embed_figure.py` | NEW | ~180 |
| `tests/composer/test_save_multiple.py` | NEW | ~100 |
| `tests/composer/test_cmyk_output.py` | NEW | ~120 |
| `tests/composer/test_external_raster.py` | NEW | ~80 |
| `tests/composer/test_panel_image.py` | MODIFY (empty-path construction tests) | +20 |
| `tests/composer/test_pdf_compositing.py` | MODIFY (embed_figure variant) | +30 |
| `tests/composer/test_svg_compositing.py` | MODIFY (embed_figure variant) | +30 |
| `tests/composer/golden/svg/cell-2col-with-embed-figure.svg` | NEW (gated, regen) | — |
| `tests/composer/golden/pdf/cell-2col-with-embed-figure.pdf` | NEW (gated, regen) | — |
| `tests/composer/golden/_compositions.py` | MODIFY (1 new entry) | +25 |
| `tools/composer/regen_fixtures.py` | MODIFY (handle embed_figure registry entry) | +20 |
| `examples/composer/cell_2col_with_embed.py` | NEW (kitchen-sink) | ~50 |
| `CHANGELOG.md` | MODIFY (`[Unreleased] / Added`) | +10 |
| `skills/publiplots-guide/SKILL.md` | MODIFY (1-line) | +1 |

**Total:** ~990 LOC + tests + 2 goldens + 1 example. Similar to PR 6a; smaller than PR 5. Current test count is **407** (composer + legend on origin/main `de35ea1`); PR 6b will land ~80 new tests for ~487 total in tests/composer + tests/test_legend_grid_scope.py.

---

## Tasks

### Task 1 — Extract `_constants.py` for shared determinism strings

- [ ] Create `src/publiplots/composer/compositing/_constants.py` with:
  ```python
  _DEFAULT_CREATION_DATE = "D:20260101000000Z"  # PDF /CreationDate pin
  _DEFAULT_DATE = "2026-01-01T00:00:00"          # SVG <dc:date> pin
  _SVG_HASHSALT = "publiplots-composer"          # SVG defs-id pin
  _PRODUCER = "publiplots-composer"              # PDF /Producer pin
  ```
- [ ] Update `compositing/pdf.py` to import from `_constants` instead of module-local constants.
- [ ] Update `compositing/svg.py` to import from `_constants`.
- [ ] **Failing test first**: `test_constants.py::test_determinism_constants_consistent` — assert constants from PR 5 + PR 6a still produce byte-identical PDFs/SVGs (regenerates 2 golden PDFs + 2 golden SVGs and diffs). Then implement the extraction.

### Task 2 — Relax `PanelImage` path-required validation

- [ ] Update `PanelImage.__post_init__` in `panels.py`:
  - Change default: `path: Optional[Union[str, Path]] = None` (was `""`).
  - When `self.path is None`: skip extension/exists validation entirely; do NOT call `object.__setattr__` to normalize.
  - When `self.path` is provided: existing path validation (ext, exists, normalize to `Path`).
  - Document the new contract: `path=None` is the "unfilled" sentinel; must be paired with `canvas.embed_figure(label, fig)` before savefig.
- [ ] Add `embedded_figure: Optional[Any] = None` field to `Panel` result dataclass (`Any` to avoid matplotlib import in `panels.py` even though matplotlib types appear elsewhere — keep panels.py minimal-import).
- [ ] Change `Panel.image_path` type to `Optional[Path] = None` (was `Optional[Any] = None`); explicitly accept `None` as the "unfilled" sentinel.
- [ ] **Failing tests first** in `test_panel_image.py`:
  - `test_panel_image_no_path_construction_legal` — `pp.PanelImage(label='A', size=(70,40))` succeeds; `record.path is None`.
  - `test_panel_image_path_validates_when_provided` — existing extension/exists validation still bites when path IS provided.
  - `test_panel_image_empty_string_path_still_rejected` — explicitly passing `path=""` raises (the deliberate `""` sentinel was the architect-found bug; reject it loudly so users don't develop reliance on the broken behavior).
- [ ] Implement.

### Task 3 — Add `embedded_figure` branch to `Panel` finalization

- [ ] Update `Canvas._finalize_if_needed` in `canvas.py:738-762` (the PanelImage finalize branch):
  - When the staged `PanelImage` has `path is None`, set `image_path=None` (the unfilled sentinel) and leave `embedded_figure=None` (default).
  - When path IS provided, finalize as before with `image_path=panel_input.path` (already a `Path`).
- [ ] **Failing test first**: `test_embed_figure.py::test_panel_image_no_path_finalizes_to_unfilled` — finalize an unfilled PanelImage; the Panel record has `embedded_figure=None` and `image_path is None`.
- [ ] Implement.

### Task 4 — `Canvas.embed_figure` method

- [ ] Add `embed_figure(self, label, figure)` to `Canvas` in `canvas.py`:
  - Trigger lazy finalization first (`self._finalize_if_needed()`).
  - Resolve `label` via the same lookup as `__getitem__`.
  - Validate `panel.kind == "image"` else raise `TypeError`.
  - Validate `panel.embedded_figure is None` else raise `RuntimeError("one-shot per panel")`.
  - Mutate via `object.__setattr__(panel, "embedded_figure", figure)`.
- [ ] **Failing tests first** in `test_embed_figure.py`:
  - `test_embed_figure_attaches_figure` — happy path: stage empty PanelImage, embed_figure, panel.embedded_figure is the figure.
  - `test_embed_figure_raises_on_axes_panel` — embed_figure on a PanelAxes target raises TypeError.
  - `test_embed_figure_raises_on_double_embed` — embed_figure twice on same panel raises RuntimeError.
  - `test_embed_figure_raises_on_empty_canvas` — embed_figure before any add_row raises RuntimeError.
  - `test_embed_figure_raises_on_unknown_label` — embed_figure with unknown label raises KeyError.
- [ ] Implement.

### Task 5 — `compositing/_embed.py` Figure→bytes helpers

- [ ] Create `compositing/_embed.py` with:
  - `render_figure_to_pdf_bytes(figure, *, metadata_creation_date=None, **savefig_kwargs) → bytes` — uses `_DEFAULT_CREATION_DATE` from `_constants`. Calls `figure.savefig(buf, format='pdf', metadata={"CreationDate": None}, **savefig_kwargs)`; opens via pypdf to inject pinned `/Producer` + `/CreationDate`; returns bytes.
  - `render_figure_to_svg_bytes(figure, *, metadata_date=None, **savefig_kwargs) → bytes` — uses `_DEFAULT_DATE` + `_SVG_HASHSALT` from `_constants`. Wraps in `plt.rc_context({"svg.hashsalt": _SVG_HASHSALT})` + `metadata={"Date": ...}` per the same 3-state semantics as PR 6a's `savefig_svg`.
- [ ] **Failing tests first** in `test_embed_figure.py`:
  - `test_render_figure_to_pdf_bytes_deterministic` — render same figure twice, byte-identical.
  - `test_render_figure_to_svg_bytes_deterministic` — render same figure twice, byte-identical.
  - `test_render_figure_to_pdf_bytes_starts_with_pdf_marker` — bytes start with `b"%PDF-"`.
  - `test_render_figure_to_svg_bytes_parses_via_lxml` — bytes parse cleanly via `lxml.etree.fromstring`.
- [ ] Implement.

### Task 6 — Wire `embedded_figure` branch in `compositing/pdf.py` + `compositing/svg.py`

- [ ] In `pdf.py:_compose_panel_onto`, add the early `if panel.embedded_figure is not None:` branch that calls `render_figure_to_pdf_bytes`.
- [ ] In `svg.py:_compose_panel_into`, add the analogous branch using `render_figure_to_svg_bytes` + `lxml.etree.fromstring` to get the schematic root element.
- [ ] **Failing tests first** in `test_pdf_compositing.py` + `test_svg_compositing.py`:
  - `test_savefig_pdf_with_embedded_figure_passes_mediabox_check` — Canvas with one embed_figure'd PanelImage saved to PDF; mediabox matches expected.
  - `test_savefig_svg_with_embedded_figure_passes_viewbox_check` — analogous for SVG.
  - `test_savefig_pdf_with_embed_figure_strict_vectors_raises_on_no_figure` — empty-path PanelImage + no embed_figure + savefig pdf → ComposerVectorError with embed_figure hint.
- [ ] Implement.

### Task 7 — `Canvas.save_multiple` method

- [ ] Add `save_multiple(self, stem, formats=None, **kwargs) → List[Path]` to `Canvas`:
  - Default `formats=None` → `['png', 'pdf']` (parity with `pp.save_multiple`).
  - **Pre-validate** all formats BEFORE writing: non-empty, str type, no duplicates, all in known ext sets. Pre-validate `cmyk=True` against the format list (raise if any ext is non-raster). Raises `ValueError` listing the offending ext(s).
  - For each `ext`, build `path = Path(stem).with_suffix(f".{ext.lstrip('.')}")`, call `self.savefig(path, **kwargs)`.
  - Return list of paths written.
- [ ] **Failing tests first** in `test_save_multiple.py`:
  - `test_save_multiple_writes_all_formats` — `formats=['pdf', 'svg', 'png']` writes 3 files.
  - `test_save_multiple_default_formats` — `formats=None` writes png + pdf.
  - `test_save_multiple_returns_paths` — return value matches written paths.
  - `test_save_multiple_empty_formats_raises` — `formats=[]` raises ValueError.
  - `test_save_multiple_unknown_ext_raises_pre_write` — `formats=['xyz']` raises ValueError; no file written.
  - `test_save_multiple_duplicates_raise` — `formats=['png', 'png']` raises ValueError.
  - `test_save_multiple_cmyk_with_vector_format_raises_pre_write` — `cmyk=True + formats=['tif','pdf']` raises ValueError BEFORE writing the .tif (no partial state).
- [ ] Implement.

### Task 8 — CMYK + TIFF compression on raster branch

- [ ] Update `_save.dispatch_savefig` to **explicitly consume** `cmyk: bool = False`, `tiff_compression: str = "tiff_lzw"`, `external_raster: bool = False` as named kwargs alongside the existing PR 5/6a kwargs. They MUST NOT leak into `**kwargs` (architect-found leak hazard).
- [ ] Add an ext→Pillow-format map: `_PILLOW_FORMAT = {".tif": "TIFF", ".tiff": "TIFF", ".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG"}` (architect found `ext.upper()` fails for `.tif`).
- [ ] Raster branch logic:
  - When `cmyk=True` and ext in `{.tif, .tiff, .jpg, .jpeg}`: render to PNG BytesIO via `pp.savefig`, open via Pillow, `convert('CMYK')`, save with `format=_PILLOW_FORMAT[ext]` (and `compression=tiff_compression` for TIFF only).
  - When `cmyk=True` and ext is PDF/SVG: raise `ValueError(f"cmyk=True is only valid for raster outputs (.tif/.tiff/.jpg/.jpeg); matplotlib's PDF/SVG backends emit RGB, and cairosvg cannot produce CMYK.")`.
  - When `cmyk=True` and ext is PNG: raise `ValueError("PNG does not support CMYK; use .tif/.tiff/.jpg/.jpeg instead.")`.
  - When `cmyk=False` and ext is `.tif`/`.tiff` AND `tiff_compression != _DEFAULT_TIFF_COMPRESSION`: render to PNG BytesIO, re-save as TIFF with `compression=tiff_compression`.
  - When `cmyk=False` and `tiff_compression` is default: existing `pp.savefig` path unchanged (architect verified pp.savefig does NOT accept `pil_kwargs` so the PNG round-trip is required for non-default compression).
- [ ] PDF + SVG branches: defensively raise `ValueError` if `cmyk=True` reaches them (belt-and-braces; should have been caught by Canvas.savefig).
- [ ] Update `Canvas.savefig` signature: add `cmyk: bool = False`, `tiff_compression: str = "tiff_lzw"` keyword args; thread through.
- [ ] **Failing tests first** in `test_cmyk_output.py`:
  - `test_savefig_tiff_cmyk_round_trip` — Canvas with PanelAxes scatter, save to .tif with cmyk=True; re-open via Pillow, assert mode == 'CMYK'.
  - `test_savefig_jpeg_cmyk_round_trip` — analogous for JPEG.
  - `test_savefig_pdf_cmyk_raises` — `cmyk=True` + .pdf → ValueError with the hint.
  - `test_savefig_svg_cmyk_raises` — analogous for .svg.
  - `test_savefig_png_cmyk_raises` — `cmyk=True` + .png → ValueError.
  - `test_savefig_tiff_compression_default_is_lzw` — open via Pillow, inspect `info["compression"] == "tiff_lzw"`.
  - `test_savefig_tiff_compression_raw_smaller_than_lzw_no` — actually `raw` is bigger than `tiff_lzw`; fix wording.
- [ ] Implement.

### Task 9 — `external_raster` sidecar option for SVG

- [ ] Update `_save.dispatch_savefig` to thread `external_raster: bool = False` to `savefig_svg`.
- [ ] Update `compositing/svg.py:savefig_svg` + `_compose_panel_into` to thread `external_raster` + the output path (for sidecar resolution).
- [ ] In `_raster_fallback_to_image_element`, when `external_raster=True`, write sidecar PNG and emit `<image href="...">` with relative path; else emit inline data-URI.
- [ ] **Failing tests first** in `test_external_raster.py`:
  - `test_external_raster_writes_sidecar_png` — Canvas with one PNG-source PanelImage; savefig svg + external_raster=True; verify sidecar PNG exists at expected path.
  - `test_external_raster_emits_relative_href` — XPath finds `<image href>` with relative path (no `data:` URI).
  - `test_external_raster_no_op_for_pdf` — savefig pdf + external_raster=True silently writes PDF (no sidecar).
  - `test_default_external_raster_false_uses_inline` — default behavior unchanged from PR 6a.
- [ ] Implement.

### Task 10 — Goldens + parametrized integration tests + example + CHANGELOG + skill

- [ ] Add `cell-2col-with-embed-figure` composition to `_compositions.py`:
  - Stage `pp.PanelAxes('A', size=(70, 50))` + `pp.PanelImage('B', size=(70, 50))`.
  - Build a side-figure: `fig, ax = pp.subplots(); ax.plot([1,2,3], [4,5,6])`.
  - `canvas.embed_figure('B', fig)`.
- [ ] Generate `cell-2col-with-embed-figure.{pdf,svg}` via regen tool.
- [ ] Manual viewer check: open both in Adobe Illustrator (PDF) and Firefox (SVG); verify the embedded figure renders correctly in the slot, vector-preserved.
- [ ] Extend `test_pdf_compositing.py` + `test_svg_compositing.py` parametrize with the new composition × 3 modes each.
- [ ] Create `examples/composer/cell_2col_with_embed.py`.
- [ ] CHANGELOG `[Unreleased] / Added` entry covering: embed_figure + save_multiple + cmyk + tiff_compression + external_raster.
- [ ] `skills/publiplots-guide/SKILL.md` — one-line update.
- [ ] **Final test run**: `uv run pytest tests/composer tests/test_legend_grid_scope.py -q`. Should pass; ~487 total tests.

---

## Architect blockers expected

These will be the architect's first questions. Pre-document positions:

1. **`pp.savefig` direct support for `pil_kwargs={"compression": ...}`** — if matplotlib's PIL backend already accepts this, the TIFF compression path can skip the PNG-buffer round-trip. Architect to verify empirically with `inspect.signature(pp.savefig)` or by reading `publiplots.utils.io`.
2. **CMYK profile choice** — FOGRA39 vs U.S. Web Coated SWOP v2 vs Japan Color 2001 Coated. Empirically, Cell + Nature accept all three; the question is which is the "least bad" default. Architect: defer profile bundling to PR 7 polish; PR 6b uses Pillow's default sRGB→CMYK conversion (no ICC profile).
3. **`embed_figure` figure persistence** — when the user calls `embed_figure(label, fig)` then later `fig.clear()` or modifies `fig.axes`, what's the expected behavior at savefig time? Decision: embed_figure stores a reference, NOT a snapshot. The user is responsible for not mutating the figure between `embed_figure` and `savefig`. Document this; do NOT defensively deep-copy.
4. **Pillow lazy import** — `_save.py`'s raster CMYK branch needs Pillow. Currently `_save.py` doesn't import Pillow at all (raster path goes through `pp.savefig`). Add lazy import inside the cmyk branch.
5. **`save_multiple` mid-failure semantics** — when iteration 2 of `formats=['tif', 'pdf']` raises (e.g., cmyk=True), do we leave the .tif file written, or rollback by deleting it? Decision: leave files written; document the partial-write semantics. Rollback is risky (race with the user's filesystem watch + tooling).
6. **`external_raster` for canvases with NO raster sources** — silent no-op (no sidecars written). Document.
7. **`Panel.embedded_figure` as `Any`** — avoids matplotlib import in `panels.py`. Architect may push for `Optional[Figure]` typed; verify the `panels.py` import surface.

## Likely code-quality nits expected

- `Canvas.embed_figure` and `Canvas.save_multiple` together push canvas.py LOC ~590-650 (currently 561 + ~90 from PR 6 work). Refactor candidate for PR 7.
- The 3 new savefig kwargs (`cmyk`, `tiff_compression`, `external_raster`) push `Canvas.savefig` signature complexity. Consider a `SaveOptions` dataclass for PR 7.
- `_constants.py` extraction is a good moment to also extract `MM2PT`, `_VALID_ALIGN`, `_VALID_CLIP` from `_geometry.py`. Resist scope creep — ONLY the determinism strings move in PR 6b.

---

## Conventional commits expected

- `refactor(composer): extract _constants module for determinism strings` (Task 1)
- `feat(composer): relax PanelImage path-required validation; add Panel.embedded_figure` (Task 2)
- `feat(composer): finalize empty-path PanelImage as unfilled slot` (Task 3)
- `feat(composer): canvas.embed_figure method` (Task 4)
- `feat(composer): render_figure_to_{pdf,svg}_bytes helpers` (Task 5)
- `feat(composer): wire embedded_figure branch in pdf + svg compositing` (Task 6)
- `feat(composer): canvas.save_multiple sugar` (Task 7)
- `feat(composer): cmyk + tiff_compression on raster savefig` (Task 8)
- `feat(composer): external_raster sidecar PNG for SVG output` (Task 9)
- `test(composer): cell-2col-with-embed-figure golden + example + CHANGELOG + skill update` (Task 10)

All with the em-dash convention.
