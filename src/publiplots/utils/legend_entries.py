"""
Shared legend-entry infrastructure.

Plot functions stash LegendEntry objects on ax._publiplots_legend_entries.
pp.legend(ax) reads from this store to render per-axis legends.
pp.legend_group(anchor=ax) aggregates entries across a grid of axes.
"""

import hashlib
from dataclasses import dataclass


_LEGEND_KINDS = ("hue", "size", "style", "marker", "hatch")


@dataclass(frozen=True)
class LegendEntry:
    """A single stashed legend entry on an axes.

    Attributes
    ----------
    name : str
        The variable name the user passed to the plot function
        (e.g. ``hue='treatment'`` -> ``name='treatment'``).
    kind : str
        One of ``"hue"``, ``"size"``, ``"style"``, ``"marker"``.
    handles : tuple
        Matplotlib-compatible handles. For continuous hue, the first
        handle is a ``ScalarMappable`` (see :func:`is_continuous_hue`).
    labels : tuple of str
        Display labels. Empty for continuous hue (colorbar path).
    signature : str
        Short hash of (kind, labels, handle-type + key visual props).
        Used by pp.legend_group for dedup and mismatch detection.
    """
    name: str
    kind: str
    handles: tuple
    labels: tuple
    signature: str

    @classmethod
    def build(cls, name, kind, handles, labels) -> "LegendEntry":
        """Construct an entry with a computed signature."""
        return cls(
            name=name,
            kind=kind,
            handles=tuple(handles),
            labels=tuple(labels),
            signature=_hash_handles(handles, labels),
        )


def _hash_handles(handles, labels) -> str:
    parts = []
    for h, lab in zip(handles, labels):
        parts.append(type(h).__name__)
        parts.append(str(lab))
        for attr in ("get_facecolor", "get_marker", "get_markersize",
                     "get_linewidth"):
            fn = getattr(h, attr, None)
            if fn is not None:
                try:
                    parts.append(repr(fn()))
                except Exception:
                    pass
    # If no labels but there ARE handles (continuous hue / colorbar),
    # include the handle types at least.
    if not labels and handles:
        for h in handles:
            parts.append(type(h).__name__)
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:12]


def stash_entry(ax, entry: LegendEntry) -> None:
    """Append an entry to ``ax._publiplots_legend_entries``.

    Creates the list attribute on first call. Order is preserved;
    later calls append.
    """
    existing = getattr(ax, "_publiplots_legend_entries", None)
    if existing is None:
        existing = []
        ax._publiplots_legend_entries = existing
    existing.append(entry)


def get_entries(ax) -> list:
    """Return the ordered list of entries stashed on ``ax``."""
    return list(getattr(ax, "_publiplots_legend_entries", []))


def resolve_legend_flags(legend) -> dict:
    """Convert ``legend=`` (bool | dict) to a per-kind include map.

    - ``True``  -> all kinds True
    - ``False`` -> all kinds False
    - ``dict``  -> as given; missing keys default to True
    """
    if legend is True:
        return {k: True for k in _LEGEND_KINDS}
    if legend is False:
        return {k: False for k in _LEGEND_KINDS}
    if isinstance(legend, dict):
        return {k: bool(legend.get(k, True)) for k in _LEGEND_KINDS}
    raise TypeError(
        f"legend must be bool or dict[str, bool], got {type(legend).__name__}"
    )


def entry_is_in_group(fig, entry: LegendEntry, ax=None) -> bool:
    """True if any legend_group on ``fig`` claims this entry.

    When ``ax`` is provided, the check is scoped: a group claims the
    entry only if the entry name matches AND ``ax`` falls within the
    group's ``axes=`` scope. First-registered wins on scope overlap.
    """
    groups = getattr(fig, "_publiplots_legend_groups", None)
    if not groups:
        return False
    for group in groups:
        if not group.claims(entry.name):
            continue
        if ax is None or group._scope_contains(ax):
            return True
    return False


def is_continuous_hue(handles) -> bool:
    """True if the handles list represents a continuous colormap.

    Detection is by the presence of a ``ScalarMappable`` as the first
    handle — categorical hue handles are publiplots' RectanglePatch /
    MarkerPatch / etc., never ScalarMappable.
    """
    if not handles:
        return False
    try:
        from matplotlib.cm import ScalarMappable
    except ImportError:
        return False
    return isinstance(handles[0], ScalarMappable)
