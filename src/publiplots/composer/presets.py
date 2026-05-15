"""Canvas presets for the Composer.

PR 1 ships only the ``'custom'`` preset. Journal presets (Cell, Nature,
Nature Methods, Science) land in PR 2 with verified mm dimensions. To
add a journal preset later, add an entry to :data:`PRESETS` here and
extend :func:`resolve_preset` to handle its specific defaults.

Pure-Python module — no matplotlib imports. Pure data + a tiny resolver.
"""

from typing import Any, Dict, Optional


# PR 1: only 'custom'. PR 2 adds: 'cell-1col', 'cell-1.5col', 'cell-2col',
# 'nature-1col', 'nature-1.5col', 'nature-2col', 'nature-methods-*',
# 'science-1col', 'science-2col'.
PRESETS: Dict[str, Dict[str, Any]] = {
    "custom": {
        # 'custom' has no preset width; the caller MUST supply width=.
        "default_width_mm": None,
        "max_height_mm": None,  # no enforcement for 'custom' in PR 1
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
        Preset key (e.g., ``'custom'``). Must be in :data:`PRESETS`.
    width : float or None
        User-supplied canvas width in mm. Required for ``'custom'``;
        optional for journal presets (which provide a default width).

    Returns
    -------
    dict
        Keys: ``width_mm`` (float), ``max_height_mm`` (float or None).

    Raises
    ------
    KeyError
        If ``name`` is not a known preset.
    ValueError
        If ``width`` is None for a preset that requires it, or if
        ``width`` is non-positive.
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
    }
