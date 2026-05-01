"""Dispatcher stub. Real implementation lands in Task 5."""
from typing import List

from matplotlib.axes import Axes
from matplotlib.text import Text


_STRATEGIES: dict = {}


def annotate(ax: Axes, kind: str = "bar_values", **kwargs) -> List[Text]:
    raise NotImplementedError("annotate() implementation lands in Task 5")
