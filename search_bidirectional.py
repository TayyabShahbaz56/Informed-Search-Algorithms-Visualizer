import time
from collections import deque
from typing import Dict, Optional, Tuple, List

from grid_env import Grid
from view_gui import GridGUI
from search_utils import spawn_dynamic_obstacle

Node = Tuple[int, int]


def _reconstruct_meeting_path(
    parent_start: Dict[Node, Optional[Node]],
    parent_goal: Dict[Node, Optional[Node]],
    meet: Node,
    start: Node,
    goal: Node,
) -> List[Node]:
    # path from start -> meet
    path_start: List[Node] = []
    cur: Optional[Node] = meet
    while cur is not None:
        path_start.append(cur)
        if cur == start:
            break
        cur = parent_start.get(cur)
    path_start.reverse()

    # path from meet -> goal
    path_goal: List[Node] = []
    cur = parent_goal.get(meet)
    while cur is not None:
        path_goal.append(cur)
        if cur == goal:
            break
        cur = parent_goal.get(cur)

    return path_start + path_goal


def bidirectional_search(
    grid: Grid,
    gui: GridGUI,
    pause: float = 0.1,
    skip_follow: bool = False,
    metrics: Optional[Dict] = None,
    dynamic_prob: float = 0.0,
) -> List[Node]:
    """
    Bidirectional BFS from start and goal simultaneously.
    """
    t0 = time.perf_counter()
    grid.clear_search_marks()
    start, goal = grid.start, grid.end

    q_start: deque[Node] = deque([start])
    q_goal: deque[Node] = deque([goal])

    parent_start: Dict[Node, Optional[Node]] = {start: None}
    parent_goal: Dict[Node, Optional[Node]] = {goal: None}
    grid.mark_visit(start)
    grid.mark_visit(goal)

    visited_start = {start}
    visited_goal = {goal}

    meet: Optional[Node] = None

    while q_start and q_goal and meet is None:
        # expand from start side
        for _ in range(len(q_start)):
            current = q_start.popleft()
            if current not in (start, goal):
                grid.grid[current] = Grid.EXPLORED
            if current in visited_goal:
                meet = current
                break
            for nbr in grid.neighbors(current):
                if nbr not in visited_start:
                    visited_start.add(nbr)
                    parent_start[nbr] = current
                    grid.mark_visit(nbr)
                    if nbr not in (start, goal):
                        grid.grid[nbr] = Grid.FRONTIER
                    q_start.append(nbr)

        if meet is not None:
            break

        # expand from goal side
        for _ in range(len(q_goal)):
            current = q_goal.popleft()
            if current not in (start, goal):
                grid.grid[current] = Grid.EXPLORED
            if current in visited_start:
                meet = current
                break
            for nbr in grid.neighbors(current):
                if nbr not in visited_goal:
                    visited_goal.add(nbr)
                    parent_goal[nbr] = current
                    grid.mark_visit(nbr)
                    if nbr not in (start, goal):
                        grid.grid[nbr] = Grid.FRONTIER
                    q_goal.append(nbr)

        if dynamic_prob > 0:
            spawn_dynamic_obstacle(grid, dynamic_prob)
        gui.update(pause=pause)

    if meet is None:
        if metrics is not None:
            metrics["nodes_visited"] = sum(1 for r in range(grid.rows) for c in range(grid.cols) if grid.grid[r, c] == Grid.EXPLORED)
            metrics["path_cost"] = 0
            metrics["exec_time_ms"] = (time.perf_counter() - t0) * 1000
        return []

    path = _reconstruct_meeting_path(parent_start, parent_goal, meet, start, goal)
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


