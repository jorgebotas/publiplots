"""Tree construction and angular layout for wedge charts.

Pure data: this module creates no artists and its layout is plain
arithmetic. A frame becomes a list of :class:`WedgeNode`, and nodes
become a list of :class:`WedgePlacement` carrying angles in degrees and
radii in axes units. Keeping the layout free of artists is what lets the
tree be tested as data rather than by measuring a rendered figure.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

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
        own_value: summed value of the rows that end at this node. A
            full-depth leaf, and a ragged leaf (no children, shallower than
            the last ring), have ``own_value == value``; a parent-leaf, a
            node with children that some rows also end at, has
            ``0 < own_value < value``; a pure internal node has 0.
    """

    path: Tuple[Any, ...]
    value: float
    depth: int
    parent: Optional[Tuple[Any, ...]]
    sibling_index: int
    sibling_count: int
    own_value: float = 0.0


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
    *,
    owner: str = "pieplot",
) -> List[WedgeNode]:
    """Aggregate ``data`` into a composition tree, in draw order.

    A parent's value is always the sum of its subtree; internal nodes are
    never read from the frame. Rows with a NaN value are dropped, as are
    groups summing to zero -- a zero wedge draws nothing, so a record for
    it would describe no artist.

    Branches may be ragged. A NaN at a deeper level ends the row's branch
    at the last non-NaN level, and its value still counts at every ancestor
    and in the node it ends at (``own_value``). When other rows continue
    past that node it is a parent-leaf: its ``value`` is the whole subtree
    and ``own_value`` is the part that ends there. A row that has a value
    at a ring after a missing one would skip a ring and raises.

    ``owner`` names the calling plot function as the prefix of error messages.
    """
    if sort and order:
        raise ValueError(
            f"{owner}: sort=True cannot be combined with order=; pass one or "
            f"the other.")
    if not levels:
        raise ValueError("at least one level is required")

    frame = data.loc[:, list(levels) + ([value] if value else [])].copy()
    if value is None:
        value = "_count"
        frame[value] = 1.0
    # A NaN value, or a NaN root, drops the row; a NaN deeper level ends the
    # branch there.
    frame = frame.dropna(subset=[value, levels[0]])
    if (frame[value] < 0).any():
        raise ValueError(
            f"{owner}: negative values are not supported -- a negative wedge "
            f"has no meaning in a part-of-whole chart.")
    if len(levels) > 1:
        missing = frame[list(levels)].isna().to_numpy()
        skipped = (missing[:, :-1] & ~missing[:, 1:]).any(axis=1)
        if skipped.any():
            row = frame.index[skipped].tolist()[0]
            raise ValueError(
                f"{owner}: row {row!r} has a missing value at one ring and a "
                f"value further out; a branch can end early but cannot skip "
                f"a ring.")
    order = dict(order or {})

    # Path -> summed value of the rows that END there, one pandas groupby
    # sum per depth: with a single level that is exactly the sum the pie
    # has always drawn, bit for bit, where a Python running total drifts
    # by float rounding over many rows.
    depth_of = frame[list(levels)].notna().sum(axis=1).to_numpy()
    own: Dict[Tuple[Any, ...], float] = {}
    for depth in range(1, len(levels) + 1):
        ending = frame[depth_of == depth]
        if ending.empty:
            continue
        sums = ending.groupby(list(levels[:depth]), observed=True,
                              sort=False, dropna=True)[value].sum()
        for key, v in sums.items():
            # One grouping column gives scalar keys, which may themselves
            # be tuples; several give one tuple per path.
            own[(key,) if depth == 1 else tuple(key)] = float(v)

    # Subtree totals, and each parent's present child keys, in one pass.
    subtree: Dict[Tuple[Any, ...], float] = {}
    for path, v in own.items():
        for k in range(1, len(path) + 1):
            subtree[path[:k]] = subtree.get(path[:k], 0.0) + v

    # Rows grouped by every prefix, so _categorical_order sees exactly the
    # rows under a parent -- the same input the previous walk gave it.
    _rows_under: Dict[Tuple[Any, ...], pd.DataFrame] = {}
    for depth in range(1, len(levels)):
        grouped = frame.groupby(list(levels[:depth]), observed=True,
                                dropna=True, sort=False)
        for key, sub in grouped:
            key = key if isinstance(key, tuple) else (key,)
            _rows_under[key] = sub

    def _children_in_order(prefix: Tuple[Any, ...]) -> List[Any]:
        column = levels[len(prefix)]
        if not prefix:
            present = frame[column]
        else:
            present = _rows_under[prefix][column]
        present = present.dropna()
        if present.empty:
            return []
        ordered = list(_categorical_order(present, order.get(column)))
        # A Categorical lists every declared category; keep the present ones.
        return [k for k in ordered if subtree.get(prefix + (k,), 0.0) > 0]

    nodes: List[WedgeNode] = []

    def _walk(prefix: Tuple[Any, ...]) -> None:
        if len(prefix) == len(levels):
            return
        entries = [(k, subtree[prefix + (k,)])
                   for k in _children_in_order(prefix)]
        if sort:
            entries.sort(key=lambda kv: kv[1], reverse=True)
        for index, (key, total) in enumerate(entries):
            path = prefix + (key,)
            nodes.append(WedgeNode(
                path=path, value=float(total), depth=len(prefix),
                parent=prefix if prefix else None,
                sibling_index=index, sibling_count=len(entries),
                own_value=float(own.get(path, 0.0)),
            ))
            _walk(path)

    _walk(())
    return nodes


def has_children(nodes: List[WedgeNode]) -> Set[Tuple[Any, ...]]:
    """Paths of the nodes that have at least one child."""
    return {n.parent for n in nodes if n.parent is not None}


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
