"""
Greedy Best-First Search (GBFS).
Uses f(n) = h(n) only.
"""

import heapq
import time
from typing import Callable, Dict, List, Optional, Tuple

from grid_env import Grid
from view_gui import GridGUI
from search_utils import reconstruct_path, spawn_dynamic_obstacle
from heuristics import manhattan, euclidean, HEURISTICS

Node = Tuple[int, int]


def gbfs(
    grid: Grid,
    gui: GridGUI,
    heuristic: str = "manhattan",
    pause: float = 0.1,
    skip_follow: bool = False,
    metrics: Optional[Dict] = None,
    dynamic_prob: float = 0.0,
) -> List[Node]:
    """
    Greedy Best-First Search: f(n) = h(n).
    heuristic: "manhattan" or "euclidean"
    """
    h_fn = HEURISTICS.get(heuristic, manhattan)
    start_time = time.perf_counter()

    grid.clear_search_marks()
    start, goal = grid.start, grid.end

    # priority queue: (h(n), node)
    pq: List[Tuple[float, Node]] = []
    heapq.heappush(pq, (h_fn(start, goal), start))
    parent: Dict[Node, Optional[Node]] = {start: None}
    grid.mark_visit(start)
    expanded_count = 0

    while pq:
        _, current = heapq.heappop(pq)
        if current not in (start, goal):
            grid.grid[current] = Grid.EXPLORED
        expanded_count += 1

        if current == goal:
            break

        for nbr in grid.neighbors(current):
            if nbr not in parent:
                parent[nbr] = current
                grid.mark_visit(nbr)
                h = h_fn(nbr, goal)
                heapq.heappush(pq, (h, nbr))
                if nbr not in (start, goal):
                    grid.grid[nbr] = Grid.FRONTIER

        if dynamic_prob > 0:
            spawn_dynamic_obstacle(grid, dynamic_prob)
        gui.update(pause=pause)

    path = reconstruct_path(parent, start, goal)
    exec_time_ms = (time.perf_counter() - start_time) * 1000

    if path and not skip_follow:
        path_cost_val = len(path) - 1
        for node in path:
            if node not in (start, goal):
                grid.grid[node] = Grid.PATH
            gui.update(pause=pause)
    path_cost_val = len(path) - 1 if path else 0

    if metrics is not None:
        metrics["nodes_visited"] = expanded_count
        metrics["path_cost"] = path_cost_val
        metrics["exec_time_ms"] = exec_time_ms

    return path

