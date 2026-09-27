"""Tree construction and angular layout for wedge charts.

Pure data: this module creates no artists and its layout is plain
arithmetic. A frame becomes a list of :class:`WedgeNode`, and nodes
become a list of :class:`WedgePlacement` carrying angles in degrees and
radii in axes units. Keeping the layout free of artists is what lets the
tree be tested as data rather than by measuring a rendered figure.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from publiplots.annotate._builders import _categorical_order

_DIRECTIONS = ("clockwise", "counterclockwise")


@dataclass(frozen=True)
class WedgeNode:
    """One node of the composition tree.

    Fields:
        path: keys from the root ring to this node; ``len(path) == depth + 1``.
        value: aggregated value; for an internal node, the sum of its children.
        depth: 0 for the innermost ring.
        parent: the parent's path, or None at depth 0.
        sibling_index: position among the nodes sharing this parent.
        sibling_count: how many nodes share this parent.
    """

    path: Tuple[Any, ...]
    value: float
    depth: int
    parent: Optional[Tuple[Any, ...]]
    sibling_index: int
    sibling_count: int


@dataclass(frozen=True)
class WedgePlacement:
    """A node's angular and radial extent.

    ``theta0 < theta1`` always, matching the argument order of
    ``matplotlib.patches.Wedge(center, r, theta1, theta2)``.
    """

    node: WedgeNode
    theta0: float
    theta1: float
    r_inner: float
    r_outer: float


def build_nodes(
    data: pd.DataFrame,
    levels: List[str],
    value: Optional[str] = None,
    order: Optional[Dict[str, Sequence]] = None,
    sort: bool = False,
) -> List[WedgeNode]:
    """Aggregate ``data`` into a composition tree, in draw order.

    A parent's value is always the sum of its children; internal nodes are
    never read from the frame. Rows with a NaN value are dropped, as are
    groups summing to zero -- a zero wedge draws nothing, so a record for
    it would describe no artist.
    """
    if sort and order:
        raise ValueError(
            "sort=True cannot be combined with order=; pass one or the other."
        )
    if not levels:
        raise ValueError("at least one level is required")

    frame = data.loc[:, list(levels) + ([value] if value else [])].copy()
    if value is None:
        value = "_count"
        frame[value] = 1.0
    frame = frame.dropna(subset=[value] + list(levels))
    if (frame[value] < 0).any():
        raise ValueError(
            "pieplot: negative values are not supported -- a negative wedge "
            "has no meaning in a part-of-whole chart."
        )

    order = dict(order or {})
    leaf_sums = frame.groupby(list(levels), observed=True, dropna=True)[value].sum()

    nodes: List[WedgeNode] = []

    def _levels_at(prefix: Tuple[Any, ...]) -> List[Any]:
        """Distinct keys of the next level below ``prefix``, in draw order."""
        depth = len(prefix)
        col = levels[depth]
        mask = pd.Series(True, index=frame.index)
        for i, key in enumerate(prefix):
            mask &= frame[levels[i]] == key
        present = frame.loc[mask, col]
        if present.empty:
            return []
        return list(_categorical_order(present, order.get(col)))

    def _subtree_sum(prefix: Tuple[Any, ...]) -> float:
        if len(prefix) == len(levels):
            return float(leaf_sums.get(prefix if len(prefix) > 1 else prefix[0], 0.0))
        return sum(_subtree_sum(prefix + (k,)) for k in _levels_at(prefix))

    def _walk(prefix: Tuple[Any, ...]) -> None:
        keys = _levels_at(prefix)
        entries = [(k, _subtree_sum(prefix + (k,))) for k in keys]
        entries = [(k, v) for k, v in entries if v > 0]
        if sort:
            entries.sort(key=lambda kv: kv[1], reverse=True)
        count = len(entries)
        for index, (key, total) in enumerate(entries):
            path = prefix + (key,)
            nodes.append(WedgeNode(
                path=path,
                value=float(total),
                depth=len(prefix),
                parent=prefix if prefix else None,
                sibling_index=index,
                sibling_count=count,
            ))
            if len(path) < len(levels):
                _walk(path)

    _walk(())
    return nodes


def total_value(nodes: List[WedgeNode]) -> float:
    """Sum of the root ring, which is the whole the chart is a part of."""
    return float(sum(n.value for n in nodes if n.depth == 0))


def layout(
    nodes: List[WedgeNode],
    *,
    start_angle: float = 90.0,
    direction: str = "clockwise",
    donut: float = 0.0,
    n_levels: int = 1,
) -> List[WedgePlacement]:
    """Assign each node an angular span and a radial band."""
    if direction not in _DIRECTIONS:
        raise ValueError(
            f"direction must be one of {list(_DIRECTIONS)}; got {direction!r}"
        )
    if not 0.0 <= donut < 1.0:
        raise ValueError(f"donut must be in [0, 1); got {donut!r}")

    total = total_value(nodes)
    if total <= 0:
        return []

    by_parent: Dict[Optional[Tuple[Any, ...]], List[WedgeNode]] = {}
    for node in nodes:
        by_parent.setdefault(node.parent, []).append(node)

    # Fraction of the full circle at which each node's span begins, walking
    # the tree in draw order so a child starts where its parent does.
    starts: Dict[Tuple[Any, ...], float] = {}

    def _assign(parent: Optional[Tuple[Any, ...]], cursor: float) -> None:
        for node in by_parent.get(parent, []):
            starts[node.path] = cursor
            _assign(node.path, cursor)
            cursor += node.value / total

    _assign(None, 0.0)

    band = (1.0 - donut) / n_levels
    places: List[WedgePlacement] = []
    for node in nodes:
        frac0 = starts[node.path]
        frac1 = frac0 + node.value / total
        if direction == "clockwise":
            theta_a = start_angle - frac1 * 360.0
            theta_b = start_angle - frac0 * 360.0
        else:
            theta_a = start_angle + frac0 * 360.0
            theta_b = start_angle + frac1 * 360.0
        places.append(WedgePlacement(
            node=node,
            theta0=theta_a,
            theta1=theta_b,
            r_inner=donut + band * node.depth,
            r_outer=donut + band * (node.depth + 1),
        ))
    return places
