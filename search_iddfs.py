import time
from typing import Dict, Optional, Tuple, List

from grid_env import Grid
from view_gui import GridGUI
from search_utils import reconstruct_path, spawn_dynamic_obstacle

Node = Tuple[int, int]


def _dls_step(
    grid: Grid,
    current: Node,
    goal: Node,
    depth_limit: int,
    parent: Dict[Node, Optional[Node]],
    gui: GridGUI,
    pause: float,
    dynamic_prob: float = 0.0,
) -> bool:
    if dynamic_prob > 0:
        spawn_dynamic_obstacle(grid, dynamic_prob)
    if current not in (grid.start, grid.end):
        grid.grid[current] = Grid.EXPLORED
    gui.update(pause=pause)

    if current == goal:
        return True
    if depth_limit == 0:
        return False

    for nbr in grid.neighbors(current):
        if nbr not in parent:
            parent[nbr] = current
            grid.mark_visit(nbr)
            if nbr not in (grid.start, grid.end):
                grid.grid[nbr] = Grid.FRONTIER
            gui.update(pause=pause)
            if _dls_step(
                    grid, nbr, goal, depth_limit - 1, parent, gui, pause, dynamic_prob
            ):
                return True
    return False


def iddfs(
    grid: Grid,
    gui: GridGUI,
    max_depth: int = 20,
    pause: float = 0.1,
    skip_follow: bool = False,
    metrics: Optional[Dict] = None,
    dynamic_prob: float = 0.0,
) -> List[Node]:
    """
    Iterative Deepening Depth-First Search.
    """
    t0 = time.perf_counter()
    start, goal = grid.start, grid.end

    for depth in range(max_depth + 1):
        grid.clear_search_marks()
        parent: Dict[Node, Optional[Node]] = {start: None}
        grid.mark_visit(start)

        found = _dls_step(grid, start, goal, depth, parent, gui, pause, dynamic_prob)

        if found:
            path = reconstruct_path(parent, start, goal)
            if not path:
                return []

            expanded = sum(1 for r in range(grid.rows) for c in range(grid.cols) if grid.grid[r, c] == Grid.EXPLORED)
            if metrics is not None:
                metrics["nodes_visited"] = expanded
                metrics["path_cost"] = len(path) - 1 if path else 0
                metrics["exec_time_ms"] = (time.perf_counter() - t0) * 1000

            if not skip_follow:
                for node in path:
                    if node not in (start, goal):
                        grid.grid[node] = Grid.PATH
                    gui.update(pause=pause)

            return path

    if metrics is not None:
        metrics["nodes_visited"] = 0
        metrics["path_cost"] = 0
        metrics["exec_time_ms"] = (time.perf_counter() - t0) * 1000
    return []


