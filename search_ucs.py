import heapq
import time
from typing import Dict, Optional, Tuple, List

from grid_env import Grid
from view_gui import GridGUI
from search_utils import reconstruct_path, spawn_dynamic_obstacle

Node = Tuple[int, int]


def ucs(
    grid: Grid,
    gui: GridGUI,
    pause: float = 0.1,
    skip_follow: bool = False,
    metrics: Optional[Dict] = None,
    dynamic_prob: float = 0.0,
) -> List[Node]:
    """
    Uniform-Cost Search with unit step cost (like Dijkstra's algorithm).
    """
    t0 = time.perf_counter()
    grid.clear_search_marks()
    start, goal = grid.start, grid.end

    pq: List[Tuple[int, Node]] = []
    heapq.heappush(pq, (0, start))

    parent: Dict[Node, Optional[Node]] = {start: None}
    cost_so_far: Dict[Node, int] = {start: 0}
    grid.mark_visit(start)

    while pq:
        current_cost, current = heapq.heappop(pq)
        if current not in (start, goal):
            grid.grid[current] = Grid.EXPLORED

        if current == goal:
            break

        for nbr in grid.neighbors(current):
            new_cost = current_cost + 1
            if nbr not in cost_so_far or new_cost < cost_so_far[nbr]:
                cost_so_far[nbr] = new_cost
                parent[nbr] = current
                grid.mark_visit(nbr)
                heapq.heappush(pq, (new_cost, nbr))
                if nbr not in (start, goal):
                    grid.grid[nbr] = Grid.FRONTIER

        if dynamic_prob > 0:
            spawn_dynamic_obstacle(grid, dynamic_prob)
        gui.update(pause=pause)

    path = reconstruct_path(parent, start, goal)
    expanded = sum(1 for r in range(grid.rows) for c in range(grid.cols) if grid.grid[r, c] == Grid.EXPLORED)
    if metrics is not None:
        metrics["nodes_visited"] = expanded
        metrics["path_cost"] = len(path) - 1 if path else 0
        metrics["exec_time_ms"] = (time.perf_counter() - t0) * 1000

    if not path:
        return []

    if not skip_follow:
        for node in path:
            if node not in (start, goal):
                grid.grid[node] = Grid.PATH
            gui.update(pause=pause)

    return path


