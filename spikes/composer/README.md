# Composer Vector-Pipeline Spike

Validates three compositing paths from `docs/superpowers/specs/2026-05-14-composer-design.md`
before PR 1 of the Composer rollout.

## Paths

- **path_a_pypdf** — matplotlib PDF + external PDF schematic → composited PDF via pypdf.
- **path_b_cairosvg** — matplotlib PDF + external SVG schematic (cairosvg → PDF) → composited PDF.
- **path_c_in_tree_svg** — matplotlib SVG + external SVG schematic → composited SVG via in-tree XML insertion.

## Run

```bash
uv pip install pypdf cairosvg
cd spikes/composer
uv run python fixtures/canvas_with_slot.py
uv run python path_a_pypdf/compose.py    && uv run python path_a_pypdf/validate.py
uv run python path_b_cairosvg/compose.py && uv run python path_b_cairosvg/validate.py
uv run python path_c_in_tree_svg/compose.py && uv run python path_c_in_tree_svg/validate.py
```

Each `validate.py` writes a `report.json` next to it. See `composer-spike.md` for the
consolidated findings.
