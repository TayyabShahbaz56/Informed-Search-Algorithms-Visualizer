import time
from typing import Dict, Optional, Tuple, List

from grid_env import Grid
from view_gui import GridGUI
from search_utils import reconstruct_path, spawn_dynamic_obstacle

Node = Tuple[int, int]


def dfs(
    grid: Grid,
    gui: GridGUI,
    pause: float = 0.1,
    skip_follow: bool = False,
    metrics: Optional[Dict] = None,
    dynamic_prob: float = 0.0,
) -> List[Node]:
    """
    Depth-First Search using an explicit stack.
    """
    t0 = time.perf_counter()
    grid.clear_search_marks()
    start, goal = grid.start, grid.end

    stack: List[Node] = [start]
    parent: Dict[Node, Optional[Node]] = {start: None}
    grid.mark_visit(start)

    while stack:
        current = stack.pop()
        if current not in (start, goal):
            grid.grid[current] = Grid.EXPLORED

        if current == goal:
            break

        # push neighbours in reverse order so the first in movement order is expanded first
        neighbours = list(grid.neighbors(current))
        neighbours.reverse()
        for nbr in neighbours:
            if nbr not in parent:
                parent[nbr] = current
                grid.mark_visit(nbr)
                if nbr not in (start, goal):
                    grid.grid[nbr] = Grid.FRONTIER
                stack.append(nbr)

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


