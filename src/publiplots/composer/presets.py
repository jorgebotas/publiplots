"""Canvas presets for the Composer.

PR 1 ships only the ``'custom'`` preset. Journal presets (Cell, Nature,
Nature Methods, Science) land in PR 2 with verified mm dimensions. To
add a journal preset later, add an entry to :data:`PRESETS` here and
extend :func:`resolve_preset` to handle its specific defaults.

Pure-Python module — no matplotlib imports. Pure data + a tiny resolver.
"""

from typing import Any, Dict, Optional


# PRESETS table — verified 2026-05-15.
#
# Sources:
#   Cell:           cell.com/figureguidelines (web.archive.org/2023-12-30 snapshot;
#                   live page is Cloudflare-blocked)
#   Nature:         nature.com/nature/for-authors/final-submission (live)
#   Nature Methods: nature.com/nmeth/submission-guidelines/aip-and-formatting (live)
#   Science:        science.org/...instructions-preparing-initial-manuscript
#                   (web.archive.org/2024-04-21 snapshot; live is Cloudflare-blocked)
#
# DERIVED values are flagged inline. Three exist:
#   - Cell max_height_mm=225 (Cell only specifies "fit on one 8.5×11 page";
#     225 is the conservative trim minus running heads/captions)
#   - Science max_height_mm=240 (Science quotes no max figure height; 240 is
#     the trim derivation from Science's ~9.75″ usable depth)
#   - Nature Methods 1-col & 1.5-col widths inherit from Nature family;
#     only the 180mm 2-col cap is journal-specific
#
# The `abc_default` and `label_size_pt` keys feed Canvas.label_style
# defaults (Task 5).

PRESETS: Dict[str, Dict[str, Any]] = {
    # 'custom' — caller supplies width, no journal constraints.
    "custom": {
        "default_width_mm": None,
        "max_height_mm": None,
        "abc_default": "upper",
        "label_size_pt": 9,
    },

    # --- Cell Press family ---
    # Source: cell.com/figureguidelines. Widths verified.
    "cell-1col": {
        "default_width_mm": 85.0,
        "max_height_mm": 225.0,  # DERIVED: Cell page-fit (no quoted max)
        "abc_default": "upper",
        "label_size_pt": 9,
    },
    "cell-1.5col": {
        "default_width_mm": 114.0,
        "max_height_mm": 225.0,  # DERIVED
        "abc_default": "upper",
        "label_size_pt": 9,
    },
    "cell-2col": {
        "default_width_mm": 174.0,
        "max_height_mm": 225.0,  # DERIVED
        "abc_default": "upper",
        "label_size_pt": 9,
    },

    # --- Nature ---
    # Source: nature.com author guide. 89/120-136/183mm, 247mm max.
    # We commit 120mm (lower bound of 120-136 range).
    "nature-1col": {
        "default_width_mm": 89.0,
        "max_height_mm": 247.0,
        "abc_default": "lower",
        "label_size_pt": 8,
    },
    "nature-1.5col": {
        "default_width_mm": 120.0,
        "max_height_mm": 247.0,
        "abc_default": "lower",
        "label_size_pt": 8,
    },
    "nature-2col": {
        "default_width_mm": 183.0,
        "max_height_mm": 247.0,
        "abc_default": "lower",
        "label_size_pt": 8,
    },

    # --- Nature Methods ---
    # Source: nature.com/nmeth AIP page. Only 180mm max-width is
    # journal-specific; 1-col + 1.5-col DERIVED from Nature inheritance.
    "nature-methods-1col": {
        "default_width_mm": 89.0,    # DERIVED from Nature family
        "max_height_mm": 247.0,      # DERIVED from Nature family
        "abc_default": "lower",
        "label_size_pt": 8,
    },
    "nature-methods-1.5col": {
        "default_width_mm": 120.0,   # DERIVED from Nature family
        "max_height_mm": 247.0,      # DERIVED from Nature family
        "abc_default": "lower",
        "label_size_pt": 8,
    },
    "nature-methods-2col": {
        "default_width_mm": 180.0,   # journal-specific (Nature is 183mm)
        "max_height_mm": 247.0,      # DERIVED from Nature family
        "abc_default": "lower",
        "label_size_pt": 8,
    },

    # --- Science (AAAS) ---
    # Source: science.org instructions. 57/121/184mm verified
    # (corrects ultraplot's rounded 55/120). 240mm DERIVED.
    "science-1col": {
        "default_width_mm": 57.0,
        "max_height_mm": 240.0,  # DERIVED: Science quotes no max
        "abc_default": "upper",
        "label_size_pt": 10,
    },
    "science-1.5col": {
        "default_width_mm": 121.0,
        "max_height_mm": 240.0,  # DERIVED
        "abc_default": "upper",
        "label_size_pt": 10,
    },
    "science-2col": {
        "default_width_mm": 184.0,
        "max_height_mm": 240.0,  # DERIVED
        "abc_default": "upper",
        "label_size_pt": 10,
    },
}


def resolve_preset(
    name: str,
    *,
    width: Optional[float],
) -> Dict[str, Any]:
    """Resolve a preset name + user width into a config dict.

    Parameters
    ----------
    name : str
        Preset key. Must be in :data:`PRESETS`.
    width : float or None
        User-supplied canvas width in mm. Required for ``'custom'``;
        optional for journal presets (which provide a default).

    Returns
    -------
    dict
        Keys: ``width_mm`` (float), ``max_height_mm`` (float or None),
        ``abc_default`` (str — 'upper', 'lower', or 'A.'),
        ``label_size_pt`` (int — preset-suggested label font size).

    Raises
    ------
    KeyError
        If ``name`` is not a known preset.
    ValueError
        If ``width`` is None for a preset that requires it
        (``'custom'`` only), or if ``width`` is non-positive.
    """
    if name not in PRESETS:
        raise KeyError(
            f"unknown preset {name!r}; known presets: {sorted(PRESETS)}"
        )
    spec = PRESETS[name]
    default_width = spec["default_width_mm"]
    if width is None:
        if default_width is None:
            raise ValueError(
                f"preset {name!r} has no default width; pass width=<mm>"
            )
        width = default_width
    if width <= 0:
        raise ValueError(f"width must be positive, got {width}")
    return {
        "width_mm": float(width),
        "max_height_mm": spec["max_height_mm"],
        "abc_default": spec["abc_default"],
        "label_size_pt": spec["label_size_pt"],
    }
