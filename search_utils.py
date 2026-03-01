import random
from typing import Dict, List, Optional, Tuple, Callable, Any

from grid_env import Grid
from view_gui import GridGUI

Node = Tuple[int, int]


def reconstruct_path(parent: Dict[Node, Optional[Node]], start: Node, goal: Node) -> List[Node]:
    """
    Rebuild the path from start to goal using the parent dictionary
    filled by the search algorithms.
    """
    if goal not in parent:
        return []

    path: List[Node] = []
    cur: Optional[Node] = goal
    while cur is not None:
        path.append(cur)
        if cur == start:
            break
        cur = parent.get(cur)

    path.reverse()
    return path


def spawn_dynamic_obstacle(grid: Grid, prob: float) -> None:
    """With probability `prob`, add a dynamic wall on a random empty cell."""
    if prob <= 0.0:
        return
    if random.random() > prob:
        return

    empty = []
    for r in range(grid.rows):
        for c in range(grid.cols):
            if grid.grid[r, c] == Grid.EMPTY and (r, c) not in (grid.start, grid.end):
                empty.append((r, c))
    if empty:
        grid.add_dynamic_wall(random.choice(empty))


def follow_path_with_replanning(
    grid: Grid,
    gui: GridGUI,
    full_path: List[Node],
    search_fn: Callable[..., List[Node]],
    dynamic_prob: float,
    pause: float,
    **search_kwargs: Any,
) -> bool:
    """
    Simulate time passing with dynamic obstacles.

    At each \"tick\" we may spawn a new dynamic wall. If any cell on the current
    path becomes blocked, we re-run the given search function to find a new path.
    The start node on the grid remains visually fixed; only the green path updates.
    """
    if not full_path:
        return False

    start = grid.start
    goal = grid.end

    # Run for a limited number of ticks so the animation eventually finishes
    max_ticks = max(10, len(full_path) * 2)
    ticks = 0

    while ticks < max_ticks:
        spawn_dynamic_obstacle(grid, dynamic_prob)

        # if any node on the current path is now blocked, we need to re-plan
        blocked = any(not grid.is_free(n) for n in full_path if n not in (start, goal))
        if blocked:
            grid.clear_search_marks()
            # run search again to get a new path from the same start/goal
            new_path = search_fn(grid, gui, pause=pause, skip_follow=False, **search_kwargs)
            if not new_path:
                return False
            full_path = new_path

        gui.update(pause=pause)
        ticks += 1

    return True


def path_cost(path: List[Node]) -> int:
    """Return the total cost of the path (number of steps)."""
    return len(path) - 1 if path else 0
