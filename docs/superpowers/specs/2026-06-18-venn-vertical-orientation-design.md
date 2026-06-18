# Vertical 2-way Venn Diagram — Design

**Date:** 2026-06-18
**Status:** Approved (design); implementation pending
**Scope:** Add a `vertical` orientation for the 2-way Venn diagram in `publiplots`.

## Problem

`pp.venn()` only renders the 2-way Venn diagram horizontally — two circles
side-by-side. Users sometimes want the two circles stacked vertically (one on
top of the other), e.g. to fit a tall/narrow figure slot or to match a
top-to-bottom reading order. Venn-diagram coordinates are finicky (centers,
intersection-region labels, and set-name labels are all positioned by hand), so
the change must be made without disturbing the existing, working horizontal
layout or the 3/4/5-way layouts.

## Goal

Let a user request a vertical 2-way Venn with a single, discoverable parameter,
producing a correct layout (first set on top, intersection in the middle,
set-name labels at the outer ends) while keeping the horizontal default
byte-for-byte unchanged.

## Non-goals (YAGNI)

- No orientation support for 3/4/5-way diagrams. Their label positions are
  hand-tuned; rotating them is out of scope and risks breaking placement.
- No arbitrary rotation angle — only `"horizontal"` and `"vertical"`.
- No `axes_size` parameter on `venn`. Custom sizing stays consistent with every
  other plot function: compose `pp.subplots(axes_size=...)` and pass `ax=`.
  (Considered and explicitly dropped — see "Rejected alternatives".)

## Approach

**Rotation transform of the existing horizontal geometry.** Compute the current
horizontal 2-way geometry, then apply a single 90°-clockwise coordinate map to
the circle centers and intersection-label positions. Because vertical is
*provably* the horizontal layout rotated, the two orientations cannot silently
drift apart — the key risk given how finicky these coordinates are. Set-name
labels are the one exception: they are authored directly for the vertical case
(see below) rather than rotated, because the chosen placement ("outer ends")
differs from where a blind rotation would put them.

### Rotation map

90° clockwise: `(x, y) → (y, -x)`.

Applied to the horizontal 2-way geometry (centers at `(-x_dist, 0)` and
`(+x_dist, 0)`):

| Element            | Horizontal                         | Vertical (after map)             |
|--------------------|------------------------------------|----------------------------------|
| Set A center       | `(-x_dist, 0)`                     | `(0, +x_dist)`  — **top**        |
| Set B center       | `(+x_dist, 0)`                     | `(0, -x_dist)`  — **bottom**     |
| `"10"` (only A)    | `(-x_dist - a_radius/2, 0)`        | `(0, +x_dist + a_radius/2)`      |
| `"01"` (only B)    | `(+x_dist + b_radius/2, 0)`        | `(0, -x_dist - b_radius/2)`      |
| `"11"` (both)      | `(0, 0)`                           | `(0, 0)`                         |

First set on top is the chosen stacking order. Circle radii are equal (true
circles), so no per-ellipse rotation angle is needed; `theta_offset` stays `0`.

### Set-name labels

Horizontal places set labels *below* each circle. A blind rotation would move
them to the *left* of each circle, which is not the chosen design. Instead,
compute them directly for the "outer ends" placement:

- Set A (top): above its circle — `(0, +x_dist + a_radius + 0.3)`, va `bottom`.
- Set B (bottom): below its circle — `(0, -x_dist - b_radius - 0.3)`, va `top`.

`_get_set_label_alignments()` already derives horizontal/vertical text alignment
from label-vs-center geometry, so it adapts to these positions without change
(the new positions are directly above/below their centers → ha `center`).

## Components & changes

### 1. Public API — `src/publiplots/plot/venn/diagram.py`

`venn()` gains one keyword-only parameter:

```python
def venn(sets, labels=None, colors=None, alpha=None, ax=None,
         fmt="{size}", color_labels=True, orientation="horizontal"):
```

- `orientation: str = "horizontal"` — `"horizontal"` (default, current behavior)
  or `"vertical"`.
- Validation:
  - `orientation == "vertical"` and `n_sets != 2` →
    `ValueError("orientation='vertical' is only supported for 2-way Venn diagrams.")`
  - `orientation` not in `{"horizontal", "vertical"}` → `ValueError`.
- `orientation` is threaded into `get_geometry(n_sets, orientation=orientation)`.
- Docstring updated with the parameter and a vertical example.

### 2. Geometry — `src/publiplots/plot/venn/geometry.py`

```python
def compute_2way_geometry(overlap_size: float = 0.5,
                          orientation: str = "horizontal") -> ...:
```

- Build horizontal `circles`, `label_positions`, `set_label_positions` as today.
- If `orientation == "vertical"`: apply the 90°-CW map to circle centers and to
  every `label_positions` entry; replace `set_label_positions` with the
  directly-authored vertical positions above.
- `get_geometry(n_sets, orientation="horizontal")` threads `orientation` only
  into the `n_sets == 2` branch; n=3,4,5 ignore it (never reached with a
  non-default orientation because `venn` validates first).
- `get_coordinate_ranges` unchanged — it computes the bounding box from whatever
  centers it is given, so the tall/narrow vertical box is handled automatically.

### 3. Drawing — `src/publiplots/plot/venn/draw.py`

No changes. `init_axes` already calls `set_aspect("equal")`, so circles stay
circular and the tall vertical bounding box frames correctly.

## Backward compatibility

`orientation` defaults to `"horizontal"`, which reproduces the current geometry
exactly. No existing call site changes behavior.

## Testing

Unit tests (pytest, matching the existing venn test layout):

1. `compute_2way_geometry(orientation="vertical")`:
   - Set A center above Set B (`center_A.y > center_B.y`), both at `x ≈ 0`.
   - `"10"` label above origin, `"01"` below, `"11"` at origin.
   - Set labels at outer ends (A above top circle, B below bottom circle).
2. Rotation invariants: vertical circle centers and intersection-label
   positions equal the 90°-CW map of the horizontal ones; radii unchanged.
3. `get_geometry` output for n=3,4,5 is unaffected by the new signature.
4. `pp.venn(..., orientation="vertical")` with 2 sets returns an `Axes` with the
   expected ellipse centers and text positions.
5. `orientation="vertical"` with 3/4/5 sets → `ValueError`.
6. Invalid `orientation` string → `ValueError`.
7. Backward-compat: default `orientation="horizontal"` reproduces current
   geometry exactly.

Visual verification (per the standing "visual features must be eyeballed" rule):
render a vertical 2-way Venn to PDF, rasterize via `pdftocairo`, and Read the
PNG to confirm circles stack correctly, labels do not collide, and counts land
in the right regions.

## Docs

- Update the `venn` docstring (new parameter + vertical example).
- Add a `# %%` cell to `examples/plots/plot_17_venn_diagrams.py` demonstrating
  the vertical orientation.
- Refresh `skills/publiplots-guide/SKILL.md` if it carries venn's signature /
  Plots inventory (per the release-skills-refresh convention).

## Rejected alternatives

- **`axes_size` parameter on `venn`** — initially in scope, dropped. No other
  publiplots plot function accepts both `ax` and `axes_size` (`heatmap` takes
  `ax` only; `complex_heatmap` takes `axes_size` only as a figure-owning
  builder). Adding both would introduce a novel convention for marginal benefit;
  composing `pp.subplots(axes_size=...)` + `ax=` already covers custom sizing.
- **Hand-written `compute_2way_geometry_vertical()`** — duplicates the finicky
  label math and is the classic place two layouts silently diverge.
- **Generic rotation transform across all n** — over-engineered; high blast
  radius on hand-tuned 3/4/5-way label positions.
- **Ignore / warn (instead of raise) for n≥3 + vertical** — rejected in favor of
  fail-fast `ValueError`, matching publiplots' existing validation style.
