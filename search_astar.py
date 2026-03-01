"""
A* Search.
Uses f(n) = g(n) + h(n).
"""

import heapq
import time
from typing import Callable, Dict, List, Optional, Tuple

from grid_env import Grid
from view_gui import GridGUI
from search_utils import reconstruct_path, spawn_dynamic_obstacle
from heuristics import manhattan, euclidean, HEURISTICS

Node = Tuple[int, int]


def astar(
    grid: Grid,
    gui: GridGUI,
    heuristic: str = "manhattan",
    pause: float = 0.1,
    skip_follow: bool = False,
    metrics: Optional[Dict] = None,
    dynamic_prob: float = 0.0,
) -> List[Node]:
    """
    A* Search: f(n) = g(n) + h(n).
    heuristic: "manhattan" or "euclidean"
    """
    h_fn = HEURISTICS.get(heuristic, manhattan)
    start_time = time.perf_counter()

    grid.clear_search_marks()
    start, goal = grid.start, grid.end

    # priority queue: (f(n), g(n), node)
    pq: List[Tuple[float, int, Node]] = []
    g_start = 0
    f_start = g_start + h_fn(start, goal)
    heapq.heappush(pq, (f_start, g_start, start))
    parent: Dict[Node, Optional[Node]] = {start: None}
    g_score: Dict[Node, int] = {start: 0}
    grid.mark_visit(start)
    expanded_count = 0

    while pq:
        _, g_current, current = heapq.heappop(pq)
        if current not in (start, goal):
            grid.grid[current] = Grid.EXPLORED
        expanded_count += 1

        if current == goal:
            break

        for nbr in grid.neighbors(current):
            step_cost = 1
            g_new = g_current + step_cost
            if nbr not in g_score or g_new < g_score[nbr]:
                g_score[nbr] = g_new
                parent[nbr] = current
                grid.mark_visit(nbr)
                f_new = g_new + h_fn(nbr, goal)
                heapq.heappush(pq, (f_new, g_new, nbr))
                if nbr not in (start, goal):
                    grid.grid[nbr] = Grid.FRONTIER

        if dynamic_prob > 0:
            spawn_dynamic_obstacle(grid, dynamic_prob)
        gui.update(pause=pause)

    path = reconstruct_path(parent, start, goal)
    exec_time_ms = (time.perf_counter() - start_time) * 1000

    if path and not skip_follow:
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
