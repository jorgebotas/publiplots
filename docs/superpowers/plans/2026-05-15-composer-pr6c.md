# Composer PR 6c Implementation Plan — `embed_figure` `anchor=` kwarg + visual-regression test

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the visual regression caught by user inspection of PR 6b's kitchen-sink demo. PR 6b shipped `canvas.embed_figure` with `clip='fit'` semantics that anchor the side figure's *outer mediabox* to the slot — geometrically correct but visually wrong for paper figures, because the side figure's axes-data box ends up offset from Panel A's axes-data box. After PR 6c:

1. **`canvas.embed_figure(label, figure, *, anchor='figure')`** — new `anchor` kwarg with two values:
   - `anchor='figure'` (current PR 6b default, kept for backward compat): scales the side figure's outer mediabox to fit the slot; decorations live INSIDE the slot rect.
   - `anchor='axes'`: anchors the side figure's *axes-data box* to the slot rect. Side figure's decorations spill OUTSIDE the slot, into the canvas-allocated margin region (mirroring how `Panel A`'s matplotlib axes-data box plus decorations land in the canvas's reserved margins).
2. **Raise-on-decoration-overflow contract** for `anchor='axes'` — at compose time, measure the side figure's decoration extents (xlabel/ylabel/title/legend-outside) and compare against canvas's reserved margin allocation around the panel's axes-data rect. **Architect-found contract clarification:** `Panel.bbox_mm` is the axes-data rect itself (NOT a slot containing decorations); the canvas allocates `ylabel_space`/`xlabel_space`/`title_space`/`right` budgets AROUND each panel's bbox via `_layout.py:30-38`. PR 6c maps the side fig's axes-data box to `panel.bbox_mm` directly; decorations spill into the canvas-allocated margin budget (left of bbox for ylabel, below for xlabel, etc.). Overflow detection: side fig decoration extent (mediabox − axes-data bbox per side) compared against the corresponding canvas margin budget for that panel's row/column. If overflow on any side > tolerance, raise `ComposerVectorError` with side + mm overflow + actionable hint. NOTE: this contract is intentionally strict; PR 6d will follow up with auto-expansion of canvas margins to accommodate the overflow (full layout-engine polish), which RELAXES the raise into a silent re-resolve. PR 7 documents the post-6d final contract in the composer-guide skill.
3. **Visual-regression golden** — rasterize the canvas-with-embed-figure golden PDF via `pdftocairo` at 200 DPI, store as `tests/composer/golden/png_from_pdf/cell-2col-with-embed-figure.png`, and assert via matplotlib's `compare_images(tol=10)`. This is the test that should have caught PR 6b's broken output.
4. **Example fix** — switch `examples/composer/cell_2col_with_embed.py` to use `axes_size=(50, 30)` (avoids `pp.subplots`'s xtick-clipping layout bug at axes_size=(40,30)) AND `canvas.embed_figure(..., anchor='axes')` for proper axes-data alignment.
5. **Regenerate the golden PDF + SVG + sidecar PNG** via `tools/composer/regen_fixtures.py --only cell-2col-with-embed-figure`. Manually inspect the rasterized output before committing; both Claude (per [[feedback_visual_features_must_be_eyeballed]]) and the user double-check.

**Architecture:** Pure addition. `Canvas.embed_figure` gains the `anchor` kwarg; `Panel` result dataclass gains `embedded_figure_anchor: str = 'figure'`. Compositing pipelines (`compositing/pdf.py:_compose_panel_onto`, `compositing/svg.py:_compose_panel_into`) check `panel.embedded_figure_anchor` and route through a new `_compute_axes_anchor_transform` helper when `'axes'`. The new helper extracts the side figure's axes-data bbox via `ax.get_position()` (figure-fraction coords) → multiply by side figure size in pt → subtract from the side figure mediabox to get padding-around-axes; map to slot bbox.

```
src/publiplots/composer/
├── canvas.py                          # MODIFY — embed_figure gains `anchor=` kwarg
├── panels.py                          # MODIFY — Panel.embedded_figure_anchor field
└── compositing/
    ├── _embed.py                      # MODIFY — add extract_side_axes_bbox + check_decoration_overflow helpers
    ├── pdf.py                         # MODIFY — _compose_panel_onto checks anchor; routes through new transform
    └── svg.py                         # MODIFY — _compose_panel_into checks anchor; routes through new transform
```

**Tech Stack:** No new dependencies. Reuses pypdf + cairosvg + lxml + Pillow + matplotlib already in `[composer]` extra. `pdftocairo` (system poppler binary) is used by `tools/composer/regen_fixtures.py` for the visual-regression rasterization step — already a documented optional dep for PR 6a's `assert_pdf_matches(mode='render_compare')`. CI must have poppler installed for the visual-regression test to run; without it, `_pdf_rasterizer_available()` returns False and the test SKIPs (mirrors PR 6a's pattern).

**Spec reference:** Spec line 650-653 (PR 6 contract) is silent on the figure-vs-axes anchor question; PR 6b shipped `'figure'` semantics by default. PR 6c is a follow-on contract clarification, NOT a spec change.

**Sequencing:** PR 6c → PR 6d (auto-margin-expansion, relaxes the raise-on-overflow contract) → PR 7 (composer-guide skill documents the post-6d final contract + canvas.inspect + ICC + Canvas-legend + style-merge + PDF→SVG). User-decided at plan time.

**Issues NOT addressed by PR 6c (pre-existing, separately tracked):**
- `pp.subplots(axes_size=(40, 30))` produces a figure where the rightmost xtick label is clipped at the figure mediabox. Reproducible standalone (no embed_figure involved). This is a publiplots core layout bug unrelated to Composer; file as a separate followup. PR 6c works around it by using `axes_size=(50, 30)` in the example.
- `Canvas`'s 'A' label overlaps Panel A's axes spine when the canvas is unusually short (~67mm tall). Pre-existing PR 4.5 follow-up about decoration reservation.

---

## What's IN scope for PR 6c

- **`Canvas.embed_figure(self, label, figure, *, anchor='figure')`** — extend the existing PR 6b method:
  - Add `anchor: str = 'figure'` keyword-only parameter.
  - Validate `anchor in {'figure', 'axes'}` else raise `ValueError(f"anchor={anchor!r} invalid. Expected 'figure' or 'axes'.")`.
  - Store `panel.embedded_figure_anchor = anchor` via `object.__setattr__` (Panel is frozen).
  - Default `'figure'` preserves PR 6b's behavior; existing tests + goldens unchanged for that path.
- **`Panel.embedded_figure_anchor: str = 'figure'`** — new field on the `Panel` result dataclass (default matches PR 6b behavior).
- **`compositing/_embed.py:extract_side_axes_bbox(figure) → Tuple[float, float, float, float]`** — pure helper:
  - Returns `(left_pt, bottom_pt, width_pt, height_pt)` of the side figure's axes-data bbox in side-figure-mediabox coords (PDF bottom-up convention; pt units).
  - **Twin-axes deduplication (architect-fixed):** `ax.get_shared_x_axes()` is INCORRECT for separating twins from `subplots(sharex=True)` — both produce shared sibling groups. Instead use **position-rect equality**: dedup `fig.axes` by `tuple(round(c, 6) for c in ax.get_position().bounds)`. Twins overlay (identical position rect → de-duped to one entry); `subplots(sharex=True)` axes are disjoint (different position rects → kept separately). Take union over the de-duped list.
  - Side figure's size in inches × 72 → pt; multiply fraction-coord bbox by pt size to get absolute coords.
  - Returns `(0, 0, w_pt, h_pt)` of the mediabox if `fig.axes` is empty (degenerate; the caller should already have called `_settle_subplots_auto_layout`).
- **`compositing/_embed.py:check_decoration_overflow(figure, axes_bbox_pt, decoration_budget_mm, panel_label) → None`** — pure helper that raises on overflow:
  - `figure`: the (already-settled) side figure.
  - `axes_bbox_pt`: tuple from `extract_side_axes_bbox(figure)`.
  - `decoration_budget_mm`: dict from `Canvas._panel_decoration_budget_mm(panel)`, with keys `'left'`, `'right'`, `'top'`, `'bottom'`.
  - `panel_label`: for the error message (use `'<unlabeled>'` when label is None/False).
  - Computes side figure's decoration extents on each side (left/right/top/bottom in mm) as `(mediabox_extent - axes_data_extent) × (PT2MM = 25.4/72)`. The transform applied to the side fig at compose time scales these decoration extents proportionally; verify by including the post-transform scale factor in the overflow check.
  - For each side, compare `scaled_decoration_extent_mm` against `decoration_budget_mm[side]`.
  - Raises `ComposerVectorError` with structured message if any side overflows by more than 0.5 mm tolerance (defensive against float-rounding).
  - Message format: `f"PanelImage {panel_label!r} embed_figure(anchor='axes'): side figure decorations overflow canvas reservation by {overflow_mm:.2f} mm on the {side!r} side (decoration={decoration_mm:.2f} mm vs canvas reserved {budget_mm:.2f} mm). Either (a) shrink the side figure's {hint_decoration_kind}, (b) use anchor='figure' to fit the entire side figure inside the slot rect, or (c) increase the canvas's {side} margin reservation. PR 6d will auto-expand canvas margins to accommodate; until then, this raise is the contract."`
- **`compositing/pdf.py:_compose_panel_onto`** — extend the existing `embedded_figure` branch:
  - Read `panel.embedded_figure_anchor`.
  - If `'figure'`: existing PR 6b path — scale entire side mediabox to slot via `compute_pdf_transform` with `clip='fit'` semantics (unchanged).
  - If `'axes'`:
    - Call `_settle_subplots_auto_layout(side_fig)` (same as PR 6b — pre-render reactor settle).
    - Compute `axes_bbox_pt = extract_side_axes_bbox(side_fig)` from settled positions.
    - Compute `decoration_budget_mm = canvas._panel_decoration_budget_mm(panel)`.
    - `check_decoration_overflow(side_fig, axes_bbox_pt, decoration_budget_mm, panel.label)` — raises `ComposerVectorError` if overflow > tolerance.
    - Compute transform: scale = `panel.bbox_mm × MM2PT / axes_bbox_pt_size`; translate such that `axes_bbox_pt`'s bottom-left maps to `panel.bbox_mm.bottom_left × MM2PT` (PDF y bottom-up).
    - Apply via `pypdf.Transformation()`. Decorations spill at NEGATIVE offsets relative to the slot (left of slot for ylabel, below for xlabel) consuming canvas margin region.
- **`compositing/svg.py:_compose_panel_into`** — analogous extension for SVG output. Same `anchor='axes'` semantics; SVG y-axis top-down + lxml `<g transform>` instead of pypdf transform. Reuses `extract_side_axes_bbox` (pt-based, format-agnostic) → converts pt→user-units via the canvas's `mm_per_user_unit`.
- **No new accessor needed** — `Panel.bbox_mm` IS already the axes-data rect (architect-found, `_layout.py:30-38`). For PanelAxes, `panel.bbox_mm` matches matplotlib's `ax.get_position()` × figure_size. For PanelImage `embed_figure(anchor='axes')`, the side figure's axes-data box maps directly to `panel.bbox_mm`; decorations land at negative offsets (left/below/above the bbox) consuming the canvas's pre-allocated `ylabel_space`/`xlabel_space`/`title_space` margin budget. The transform calculation in PDF/SVG compositing branches uses `panel.bbox_mm` directly — no accessor wrapper needed.
- **Canvas margin budget exposure** — for `check_decoration_overflow`, we need to know the canvas's per-panel margin allocation (ylabel_space/xlabel_space/title_space/right) for the row containing the panel. Add a NEW private helper `Canvas._panel_decoration_budget_mm(self, panel) → Dict[str, float]` returning `{'left': mm, 'right': mm, 'top': mm, 'bottom': mm}` — the mm budget on each side of `panel.bbox_mm` that's available for decoration overflow before colliding with adjacent panels or canvas edge. Read from `CanvasGeometry` (already computed during finalize). This IS a new accessor, but it surfaces existing geometry rather than duplicating it.
- **`tools/composer/regen_fixtures.py`** — extend with a `--rasterize-pdf-goldens` mode (or auto-emit alongside the PDF regen):
  - For each PDF golden in the registry, after writing the PDF, run `pdftocairo -png -r 200 -singlefile <pdf> <pngstem>` to produce `tests/composer/golden/png_from_pdf/<name>.png`. Skip silently if `pdftocairo` is not on PATH.
  - The visual-regression test reads the rasterized PNG, calls the same `compare_images(tol=10)` machinery already in PR 4.5/6a's `assert_png_matches`.
- **`tests/composer/test_pdf_compositing.py`** — extend the parametrize with `mode='visual'`:
  - Renders the PDF, rasterizes via `pdftocairo`, compares to `golden/png_from_pdf/<name>.png` with `compare_images(tol=10)`.
  - Skips if `pdftocairo` is not available (mirrors PR 6a's `_pdf_rasterizer_available()` pattern).
  - Catches PR 6b-style visual regressions where structure tests pass but the render is broken.
- **`tests/composer/test_embed_figure.py`** — add coverage for the new kwarg:
  - `test_embed_figure_anchor_figure_default` — explicit `anchor='figure'` matches PR 6b default behavior.
  - `test_embed_figure_anchor_axes_aligns_axes_data_box` — verify the embedded axes-data box lands at the slot's axes-data sub-rect (within tolerance).
  - `test_embed_figure_anchor_axes_raises_on_decoration_overflow` — construct a side fig with a deliberately oversized ylabel/title; assert `ComposerVectorError` with the actionable hint.
  - `test_embed_figure_anchor_invalid_raises` — `anchor='invalid'` raises ValueError.
  - `test_embed_figure_anchor_axes_multi_axes_side_fig` — `pp.subplots(nrows=2)` side fig: union of axes positions becomes the bbox; visual check via the visual-regression test.
- **Example regeneration** — `examples/composer/cell_2col_with_embed.py`:
  - Switch to `pp.subplots(axes_size=(50, 30))` (was `(40, 30)`) — works around the publiplots core layout bug.
  - Switch to `canvas.embed_figure('B', side_fig, anchor='axes')` for proper axes alignment.
- **Golden regeneration** — `cell-2col-with-embed-figure.{pdf,svg}` regenerated with the new example. PNG sidecar `cell-2col-with-embed-figure.png` rasterized from the PDF. Manual inspection by Claude (rasterize + Read) and user (Adobe Illustrator + Firefox).
- **CHANGELOG entry** under `[Unreleased] / Fixed` (NOT `Added` — this is a visual-regression fix on PR 6b's just-shipped feature). Mention: `anchor=` kwarg added; default `'figure'` preserves PR 6b behavior.
- **`publiplots-guide` skill update** — extend the existing one-liner to mention `anchor='axes'` for proper axes-data alignment.
- **Memory pointer to [[feedback_visual_features_must_be_eyeballed]]** — already saved (2026-05-15); plan + CHANGELOG reference it.

## What's OUT of scope for PR 6c

- **Auto-margin-expansion when `anchor='axes'` decorations overflow** — PR 6d. PR 6c contract is "raise on overflow" with the actionable hint pointing at PR 6d.
- **`canvas.inspect()` schema** — PR 7.
- **`composer-guide` skill** — PR 7.
- **Canvas × `pp.legend(rows=, cols=, span=)` integration** — PR 7.
- **FOGRA39 / SWOP v2 ICC profile bundling** — PR 7.
- **SVG `<style>` / CSS-block schematic merge** — PR 7.
- **PDF→SVG round-trip via pdf2image** — PR 7.
- **Promoting `ComposerError` etc. to top-level `pp.*`** — PR 7.
- **Fixing `pp.subplots(axes_size=(40,30))`'s xtick-clipping layout bug** — out of Composer's lane; separately tracked publiplots core layout followup. PR 6c works around it by widening the example's axes_size.
- **Fixing PR 4.5's 'A' label overlap on short canvases** — pre-existing PR 4.5 follow-up; unrelated to embed_figure.
- **Visual-regression for ALL existing goldens** — PR 6c only adds the visual-regression test for `cell-2col-with-embed-figure`. Backfilling the rest of the goldens (PR 5's `cell-2col-with-svg-schematic`, PR 6a's two SVG goldens, etc.) is a separate test-coverage PR if user wants belt-and-braces; the lesson from PR 6b is that visual checks SHOULD be standard, but the rollout doesn't need to retroactively fix all of them.

---

## Files touched

| File | Status | LOC est. |
|---|---|---|
| `src/publiplots/composer/canvas.py` | MODIFY (`anchor=` kwarg + `_panel_axes_data_bbox_mm` accessor) | +40 |
| `src/publiplots/composer/panels.py` | MODIFY (`Panel.embedded_figure_anchor` field) | +5 |
| `src/publiplots/composer/compositing/_embed.py` | MODIFY (`extract_side_axes_bbox` + `check_decoration_overflow`) | +90 |
| `src/publiplots/composer/compositing/pdf.py` | MODIFY (anchor branch in `_compose_panel_onto`) | +25 |
| `src/publiplots/composer/compositing/svg.py` | MODIFY (anchor branch in `_compose_panel_into`) | +25 |
| `tests/composer/test_embed_figure.py` | MODIFY (5 new tests) | +120 |
| `tests/composer/test_pdf_compositing.py` | MODIFY (visual-regression mode) | +40 |
| `tests/composer/golden/_helpers.py` | MODIFY (`assert_pdf_matches` mode='visual' + `_visual_png_path` accessor) | +30 |
| `tools/composer/regen_fixtures.py` | MODIFY (`--rasterize-pdf-goldens` flag + auto-emit on PDF regen) | +30 |
| `tests/composer/golden/png_from_pdf/cell-2col-with-embed-figure.png` | NEW (gated, regen) | — |
| `tests/composer/golden/pdf/cell-2col-with-embed-figure.pdf` | REGENERATE | — |
| `tests/composer/golden/svg/cell-2col-with-embed-figure.svg` | REGENERATE | — |
| `tests/composer/golden/_compositions.py` | MODIFY (composition uses anchor='axes' + axes_size=(50,30)) | +5 / -2 |
| `examples/composer/cell_2col_with_embed.py` | MODIFY (axes_size + anchor) | +3 / -3 |
| `CHANGELOG.md` | MODIFY (`[Unreleased] / Fixed`) | +6 |
| `skills/publiplots-guide/SKILL.md` | MODIFY (1-line) | +1 |

**Total:** ~420 LOC + tests + 1 new sidecar PNG + 2 regenerated goldens. Smaller than PR 6a/6b. Test count target: 478 (current) + 5 new = 483 in `tests/composer + tests/test_legend_grid_scope.py`.

---

## Tasks

### Task 1 — `Panel.embedded_figure_anchor` field + `Canvas._panel_decoration_budget_mm` accessor

- [x] Add `embedded_figure_anchor: str = 'figure'` to `Panel` result dataclass in `panels.py` (frozen dataclass; default preserves PR 6b behavior).
- [x] Add `Canvas._panel_decoration_budget_mm(self, panel) → Dict[str, float]` accessor in `canvas.py`. Returns `{'left', 'right', 'top', 'bottom'}` — mm available for decoration overflow on each side of `panel.bbox_mm` before colliding with adjacent panels or canvas edges. Read from the cached `CanvasGeometry` (computed during `_finalize_if_needed`).
  - `left` budget = (panel.bbox_mm.x_left) − (left-neighbor's right edge OR canvas left edge). For Panel A in cell-2col, the left budget is the canvas's `outer_pad_left` + `ylabel_space` reservation.
  - `right` budget = analogous on the right.
  - `top` budget = (canvas_height_mm − panel.y_bottom − panel.h_mm) reservation; or the gap between this panel's top edge and the row above.
  - `bottom` budget = analogous on the bottom.
- [x] **Failing tests first** in `test_embed_figure.py`:
  - `test_panel_decoration_budget_mm_panel_axes_matches_canvas_margins` — for a PanelAxes in cell-2col, the budget on left equals the canvas's `outer_pad + ylabel_space` reservation.
  - `test_panel_decoration_budget_mm_two_panel_row` — for a 2-panel row, the right edge of Panel A's `right` budget meets the left edge of Panel B's `left` budget (with `hpad_mm` between).
  - `test_panel_decoration_budget_mm_panel_image_returns_same_as_axes` — PanelImage and PanelAxes get identical budget shapes for the same slot position.
- [x] Implement.

### Task 2 — `Canvas.embed_figure` `anchor=` kwarg

- [x] Add `anchor: str = 'figure'` keyword-only parameter to `embed_figure` signature.
- [x] Validate `anchor in {'figure', 'axes'}` else raise `ValueError`.
- [x] Store via `object.__setattr__(panel, 'embedded_figure_anchor', anchor)`.
- [x] **Failing tests first**:
  - `test_embed_figure_anchor_default_is_figure` — `canvas.embed_figure('B', fig)` (no kwarg) → `panel.embedded_figure_anchor == 'figure'`.
  - `test_embed_figure_anchor_explicit_figure` — `anchor='figure'` works, matches default.
  - `test_embed_figure_anchor_axes_stored` — `anchor='axes'` stores correctly.
  - `test_embed_figure_anchor_invalid_raises` — `anchor='invalid'` → ValueError naming valid options.
- [x] Implement.

### Task 3 — `extract_side_axes_bbox` + `check_decoration_overflow` helpers

- [x] Add `extract_side_axes_bbox(figure)` to `compositing/_embed.py`:
  - **Architect-corrected twin-axes dedup**: dedup `fig.axes` by position-rect equality, NOT by `get_shared_x_axes()` membership. Build `seen_rects: Set[Tuple] = set()`; for each `ax` in `fig.axes`, compute `rect_key = tuple(round(c, 6) for c in ax.get_position().bounds)`; skip if already in `seen_rects`, else add and include in the union. This correctly drops twins (overlay → identical bounds) and keeps `subplots(sharex=True)` (disjoint rects).
  - Compute union of de-duped `ax.get_position()` rects (figure-fraction).
  - Multiply by figure size in pt → `(left_pt, bottom_pt, w_pt, h_pt)` in side-figure-mediabox coords.
  - Returns `(0, 0, fig_w_pt, fig_h_pt)` if no axes.
- [x] Add `check_decoration_overflow(figure, slot_axes_data_bbox_mm, panel_label) → None`:
  - Compute side fig's decoration extents per side (left/right/top/bottom in mm).
  - Compare against the slot's decoration-region thickness (canvas reservation).
  - Raise `ComposerVectorError` if any side overflows by > 0.5 mm.
  - Error message names panel + side + decoration extent vs canvas reservation + actionable hint (anchor='figure' / shrink decoration / increase canvas margin / wait for PR 6d).
- [x] **Failing tests first** in `test_embed_figure.py`:
  - `test_extract_side_axes_bbox_single_axes` — happy path, single primary axes.
  - `test_extract_side_axes_bbox_multi_axes_uses_union_for_subplots_sharex` — `pp.subplots(nrows=2, sharex=True)`: returns the union covering both rows. CRITICAL: this is the architect-flagged case where the WRONG twin-filter (via `get_shared_x_axes()`) would have collapsed both rows to one bbox.
  - `test_extract_side_axes_bbox_skips_twin_axes` — `ax.twinx()` produces overlay → de-duped to a single bbox (not double-counted).
  - `test_extract_side_axes_bbox_disjoint_subplots_keeps_both` — `plt.subplots(nrows=2, ncols=2)` returns union of all 4 disjoint rects.
  - `test_check_decoration_overflow_passes_when_within_canvas_reservation` — happy path no raise.
  - `test_check_decoration_overflow_raises_on_oversized_ylabel` — deliberately wide ylabel → ComposerVectorError with `'left'` named in message.
  - `test_check_decoration_overflow_message_names_panel_and_side_and_hint` — error contains the actionable hint substrings (panel label, side, decoration mm vs reservation mm, both `anchor='figure'` and `PR 6d` substrings).
- [x] Implement.

### Task 4 — `compositing/pdf.py` anchor='axes' branch

- [x] Update `_compose_panel_onto`. Strict ORDERING (architect-flagged):
  1. Read `panel.embedded_figure_anchor`.
  2. If `'figure'`: existing PR 6b path (unchanged).
  3. If `'axes'`:
     a. **First:** `_settle_subplots_auto_layout(panel.embedded_figure)` — settles the SubplotsAutoLayout reactor (one `figure.canvas.draw()` call; idempotent for already-decorated figures per architect verification).
     b. **Then read** `axes_bbox_pt = extract_side_axes_bbox(panel.embedded_figure)` — uses settled `ax.get_position()`.
     c. Compute `decoration_budget_mm = canvas._panel_decoration_budget_mm(panel)`.
     d. Call `check_decoration_overflow(panel.embedded_figure, axes_bbox_pt, decoration_budget_mm, panel.label)` — raises `ComposerVectorError` BEFORE re-render.
     e. Compute the transform such that `axes_bbox_pt` maps to `panel.bbox_mm × MM2PT` (panel.bbox_mm IS the axes-data rect per `_layout.py:30-38`).
     f. **Last:** call `render_figure_to_pdf_bytes(panel.embedded_figure)` which re-settles + renders. Idempotent settle is fine.
     g. Apply transform via `pypdf.Transformation()` and `merge_transformed_page` onto target page.
- [x] **Failing tests first** in `test_embed_figure.py`:
  - `test_savefig_pdf_anchor_axes_aligns_axes_data` — render a Canvas with PanelAxes(A) + embed(B, anchor='axes'); rasterize via pdftocairo; verify A's axes-data y-extent matches B's axes-data y-extent within tolerance.
  - `test_savefig_pdf_anchor_axes_overflow_raises` — overflow case raises before write.
- [x] Implement.

### Task 5 — `compositing/svg.py` anchor='axes' branch

- [x] Symmetric to Task 4 but for SVG output. Convert pt→user-units via canvas's `mm_per_user_unit`.
- [x] **Failing tests first** in `test_embed_figure.py`:
  - `test_savefig_svg_anchor_axes_aligns_axes_data` — analogous SVG test.
- [x] Implement.

### Task 6 — Visual-regression test infrastructure

- [x] Add `mode='visual'` to `assert_pdf_matches` in `tests/composer/golden/_helpers.py`:
  - Rasterize the produced PDF via `pdftocairo` to a temp PNG at 200 DPI.
  - Compare against `tests/composer/golden/png_from_pdf/<name>.png` with `compare_images(tol=10)`.
  - Skip via `pytest.skip("pdftocairo not available")` if `_pdf_rasterizer_available()` returns False (reuse PR 6a helper).
  - Regen-on-missing semantics mirror existing modes.
- [x] Add `_visual_png_path(name)` helper to mirror `_png_path` / `_pdf_path`.
- [x] Extend `tools/composer/regen_fixtures.py` with `--rasterize-pdf-goldens` flag and auto-emission on PDF regen:
  - After writing each PDF golden, run `pdftocairo -png -r 200 -singlefile <pdf> golden/png_from_pdf/<name>` (skip if not available).
- [x] **Failing test first** in `test_pdf_compositing.py`:
  - `test_assert_pdf_matches_visual_mode_skips_when_no_pdftocairo` — patches `_pdf_rasterizer_available` to False; mode='visual' skips cleanly.
  - `test_assert_pdf_matches_visual_mode_compares_rasterized_pdf` — happy path with mocked golden PNG.
- [x] Implement.

### Task 7 — Regenerate the example + goldens

- [x] **First experiment** (architect-flagged C1): does `axes_size=(40, 30)` + `anchor='axes'` ALONE fix the visual issue? Test before bumping to `(50, 30)`. If yes, drop the axes_size change to keep the example showing realistic figure dimensions.
  - Run `examples/composer/cell_2col_with_embed.py` with current `axes_size=(40,30)` but switch to `anchor='axes'`.
  - Rasterize + Read.
  - If visual is OK, KEEP `axes_size=(40, 30)`. If still bleeding, switch to `(50, 30)` to dodge the publiplots core layout bug.
- [x] Update `examples/composer/cell_2col_with_embed.py`:
  - `canvas.embed_figure('B', side_fig, anchor='axes')` (was no kwarg).
  - `axes_size` based on the experiment above (settled on `(70, 50)`).
- [x] Update `tests/composer/golden/_compositions.py` — `cell-2col-with-embed-figure` build_fn mirrors the example exactly (same axes_size + anchor).
- [x] Run `python tools/composer/regen_fixtures.py --only cell-2col-with-embed-figure --rasterize-pdf-goldens` to regenerate PDF + SVG + sidecar PNG.
- [x] **Manual rasterize + Read by Claude** (per [[feedback_visual_features_must_be_eyeballed]]):
  - `pdftocairo -png -r 200 -singlefile tests/composer/golden/pdf/cell-2col-with-embed-figure.pdf /tmp/check_pdf` then `Read /tmp/check_pdf.png` to verify A and B axes-data align, no bleed, decorations land in canvas margin.
  - `cairosvg tests/composer/golden/svg/cell-2col-with-embed-figure.svg -o /tmp/check_svg.png --output-width 1200` then `Read`.
- [x] If visuals look wrong, STOP and diagnose before continuing.

### Task 8 — Add the parametrized visual-regression test for the new golden

- [x] Extend `test_pdf_compositing.py` parametrize: add `('cell-2col-with-embed-figure', 'visual')`.
- [x] Verify: `uv run pytest tests/composer/test_pdf_compositing.py -q` passes.

### Task 9 — User visual sign-off pause

- [ ] Send the rasterized PNG previews to the user via the chat (via Read tool — the harness renders PNG inline). Wait for sign-off before continuing to reviewers.
- [ ] If user requests changes, loop back to Task 7.

### Task 10 — CHANGELOG + skill update + final test run

- [x] CHANGELOG `[Unreleased] / Fixed` entry referencing the visual-regression and the new memory rule.
- [x] `skills/publiplots-guide/SKILL.md` — extend the one-liner: "embed_figure(anchor='axes') aligns the side figure's axes-data box to the slot for paper-figure axes alignment".
- [x] **Final test run**: `uv run pytest tests/composer tests/test_legend_grid_scope.py -q`. **507 passed (was 478, +29 new).**

---

## Architect blockers expected

These will be the architect's first questions. Pre-document positions:

1. **`Canvas._panel_axes_data_bbox_mm` API** — should this be public (`canvas.panel_axes_data_bbox_mm`)? Decision: keep private for now; PR 7's `canvas.inspect()` schema will expose the full per-panel bbox map publicly. Architect to confirm.
2. **`extract_side_axes_bbox` filtering** — `ax.get_shared_x_axes()` is the matplotlib API for shared/twin axes. Edge case: `ax.twinx()` returns a NEW axes that's NOT in the shared list (twins are shared on x but not in `shared_x_axes` for the principal axis). Architect to verify the filter logic.
3. **`pp.subplots`-built side fig's axes-data box AFTER reactor settle** — verify empirically that `ax.get_position()` returns POST-settle coords (i.e., after the SubplotsAutoLayout reactor has resized the figure based on tightbbox). If pre-settle, the bbox math will be wrong.
4. **Decoration overflow tolerance (0.5 mm)** — architect-corrected: 0.5 mm is NOT "consistent with PR 6a's `tol_pt=0.5`" (those are different units; 0.5 pt ≈ 0.176 mm, so 0.5 mm is ~3× looser). Justification for 0.5 mm independently: visual-polish tolerance, not byte-fidelity; ≈ 6 printer pixels at 300 DPI (single pixel = 0.085 mm). Tighter than 0.5 mm risks false-positive raises from float-rounding in geometry math; looser admits visible decoration cropping. Architect to confirm 0.5 mm is the right value or push for 0.25 mm.
5. **`ComposerVectorError` for decoration overflow** — semantically NOT a vector-load failure (that's the existing use of `ComposerVectorError`). Architect may push for a new `ComposerLayoutError` subclass. Decision: reuse `ComposerVectorError` for now to avoid scope creep; PR 7's exception promotion can rename if needed.
6. **`pdftocairo` available locally; CI status unverified** — local dev box has `/opt/conda/bin/pdftocairo` (architect-verified, poppler 25.02.0). PR 6a's `render_compare` already shells out to it; PR 6c re-uses the same surface. Visual-regression test SKIPs cleanly if missing. CI image presence verified via PR 6a's `render_compare` test results in CI logs (if those tests pass in CI, poppler IS present). Plan states this as a soft-verify, not a hard requirement.
7. **Should the visual-regression test be added to ALL existing goldens (PR 5/6a) as well?** — would catch any latent bugs. Decision: NO for PR 6c — bundle as a separate test-coverage PR if user wants. Keeping PR 6c focused.

## Likely code-quality nits expected

- `extract_side_axes_bbox` is a mid-complexity helper; ensure docstring covers the multi-axes + twins edge cases.
- `check_decoration_overflow` error message is multi-paragraph long; ensure it's in a constant or template, not inline.
- `_panel_axes_data_bbox_mm` private name + computation path may duplicate logic in `compute_canvas_geometry`; architect to verify.
- `mode='visual'` adds a 4th `assert_pdf_matches` mode; the helper is getting long. Consider extracting per-mode handlers if this grows further.

---

## Conventional commits expected

- Task 1: `feat(composer): Panel.embedded_figure_anchor field + Canvas axes-data bbox accessor`
- Task 2: `feat(composer): canvas.embed_figure anchor= kwarg`
- Task 3: `feat(composer): extract_side_axes_bbox + check_decoration_overflow helpers`
- Task 4: `feat(composer): pdf compositing anchor='axes' branch`
- Task 5: `feat(composer): svg compositing anchor='axes' branch`
- Task 6: `test(composer): assert_pdf_matches mode='visual' + regen --rasterize-pdf-goldens`
- Task 7: `fix(composer): regenerate cell-2col-with-embed-figure golden with anchor='axes'`
- Task 8: `test(composer): visual-regression parametrize for embed-figure golden`
- Task 10: `docs(composer): PR 6c CHANGELOG + skill one-liner`

All with the em-dash convention.
