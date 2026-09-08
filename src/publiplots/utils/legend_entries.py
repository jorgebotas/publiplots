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


def mark_entry_rendered(ax, entry: LegendEntry) -> None:
    """Record that ``entry`` has been drawn into ``ax``'s own legend.

    The stash is cumulative and every plot call renders from it, so
    without this record the second call on an axes re-renders the first
    call's entries alongside its own — N calls producing ~N(N+1)/2
    legends, each reserving its own layout space (#227).

    Keyed by object identity, not by name: two calls that pass the same
    column (``hue='c'`` twice) stash two distinct entries and each is
    still owed one render. The entry is kept as the dict value so the
    key cannot be recycled onto a later object by ``id`` reuse.

    Only the per-axes render path records here. A figure-level group's
    ``_materialize`` and ``pp.legend(ax)``'s adopt rebuild deliberately
    render entries later than they were stashed, and neither consults
    this record.
    """
    rendered = getattr(ax, "_publiplots_rendered_entries", None)
    if rendered is None:
        rendered = {}
        ax._publiplots_rendered_entries = rendered
    rendered[id(entry)] = entry


def entry_is_rendered(ax, entry: LegendEntry) -> bool:
    """True if ``entry`` has already been drawn into ``ax``'s own legend."""
    rendered = getattr(ax, "_publiplots_rendered_entries", None)
    if not rendered:
        return False
    return id(entry) in rendered


def record_entry_kwargs(ax, entry: LegendEntry, legend_kws) -> None:
    """Record the ``legend_kws`` the plot call gave ``entry``.

    A ``LegendEntry`` carries only what a legend needs to be *drawn*
    (name, kind, handles, labels); the presentation choices — ``ncol``,
    ``height``/``width``, ``ticks``, the placement family — were consumed
    at render time and then lost. ``pp.legend(ax)`` re-renders a claimed
    entry from the stash, so with nothing recorded it could only rebuild
    a default artist and every forwarded key silently vanished (#258).

    Kept in a side table rather than on the frozen dataclass, the same
    shape as the render record (#227): the entry stays immutable and its
    ``signature`` — which feeds cross-axes dedup — keeps hashing only
    visual identity, so two axes that stash the same handles with
    different ``legend_kws`` still merge as one entry.

    Keyed by object identity with the entry retained in the value, so an
    ``id`` freed by garbage collection cannot be recycled onto a later
    object and hand it the wrong kwargs.

    First writer wins. Every plot call stashes its entries and then calls
    ``render_entries`` once, so the un-recorded entries at that point are
    exactly the ones this call stashed; a repeat pass over an entry that
    already has a record must not overwrite it with a later call's keys.
    """
    recorded = getattr(ax, "_publiplots_entry_kwargs", None)
    if recorded is None:
        recorded = {}
        ax._publiplots_entry_kwargs = recorded
    if id(entry) in recorded:
        return
    recorded[id(entry)] = (entry, dict(legend_kws or {}))


def get_entry_kwargs(ax, entry: LegendEntry) -> dict:
    """The ``legend_kws`` recorded for ``entry`` on ``ax`` (``{}`` if none)."""
    recorded = getattr(ax, "_publiplots_entry_kwargs", None)
    if not recorded:
        return {}
    found = recorded.get(id(entry))
    if found is None or found[0] is not entry:
        return {}
    return dict(found[1])


def axes_entry_kwargs(ax) -> dict:
    """Union of the ``legend_kws`` recorded for every entry on ``ax``.

    Used for the group-level (placement) half of the record, which has
    one value per group rather than one per artist. First writer wins on
    a conflict — the same rule :func:`record_entry_kwargs` applies per
    entry, and the same rule ``_get_or_create_per_axes_group`` already
    applies to the placement of a plot-created group ("they take effect
    only when the group is FIRST created").
    """
    recorded = getattr(ax, "_publiplots_entry_kwargs", None)
    if not recorded:
        return {}
    merged = {}
    for _entry, kws in recorded.values():
        for key, value in kws.items():
            merged.setdefault(key, value)
    return merged


def entries_owed_render(fig, ax, flags: dict) -> list:
    """The stashed entries ``ax`` still owes its own per-axes legend.

    The single answer to "should this axes draw this entry itself?".
    An entry is owed a render when its kind is enabled by ``legend=``,
    no figure-level group has claimed it, and it has not been rendered
    on this axes already.
    """
    return [
        entry for entry in get_entries(ax)
        if flags[entry.kind]
        and not entry_is_in_group(fig, entry, ax=ax)
        and not entry_is_rendered(ax, entry)
    ]


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


def entry_is_in_group(fig, entry: LegendEntry, ax=None, exclude=None) -> bool:
    """True if any legend_group on ``fig`` claims this entry.

    When ``ax`` is provided, the check is scoped: a group claims the
    entry only if the entry name matches AND ``ax`` falls within the
    group's ``axes=`` scope. First-registered wins on scope overlap.

    ``exclude`` skips one group — used by a per-axes group asking "does
    anyone *else* already own this?", which is the same question the
    plot path asks through :func:`entries_owed_render` (#233).
    """
    groups = getattr(fig, "_publiplots_legend_groups", None)
    if not groups:
        return False
    for group in groups:
        if exclude is not None and group is exclude:
            continue
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
