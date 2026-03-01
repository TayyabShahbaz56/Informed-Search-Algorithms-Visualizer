import time
from typing import Dict, Optional, Tuple, List

from grid_env import Grid
from view_gui import GridGUI
from search_utils import reconstruct_path, spawn_dynamic_obstacle

Node = Tuple[int, int]


def _recursive_dls(
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
            if _recursive_dls(
                grid, nbr, goal, depth_limit - 1, parent, gui, pause, dynamic_prob
            ):
                return True
    return False


def dls(
    grid: Grid,
    gui: GridGUI,
    depth_limit: int,
    pause: float = 0.1,
    skip_follow: bool = False,
    metrics: Optional[Dict] = None,
    dynamic_prob: float = 0.0,
) -> List[Node]:
    """
    Depth-Limited Search (recursive DFS up to a given depth).
    """
    t0 = time.perf_counter()
    grid.clear_search_marks()
    start, goal = grid.start, grid.end

    parent: Dict[Node, Optional[Node]] = {start: None}
    grid.mark_visit(start)

    found = _recursive_dls(grid, start, goal, depth_limit, parent, gui, pause, dynamic_prob)

    if not found:
        return []

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


