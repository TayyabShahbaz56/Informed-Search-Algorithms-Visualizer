"""
Heuristic functions for Informed Search (GBFS and A*).
"""

import math
from typing import Tuple

Node = Tuple[int, int]


def manhattan(node: Node, goal: Node) -> float:
    """
    Manhattan Distance: D = |x1 - x2| + |y1 - y2|
    """
    r1, c1 = node
    r2, c2 = goal
    return abs(r1 - r2) + abs(c1 - c2)


def euclidean(node: Node, goal: Node) -> float:
    """
    Euclidean Distance: D = sqrt((x1 - x2)^2 + (y1 - y2)^2)
    """
    r1, c1 = node
    r2, c2 = goal
    return math.sqrt((r1 - r2) ** 2 + (c1 - c2) ** 2)


HEURISTICS = {
    "manhattan": manhattan,
    "euclidean": euclidean,
}
