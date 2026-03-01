import numpy as np  # type: ignore


class Grid:
    """
    Represents the environment grid (static walls).

    Cell codes (kept intentionally simple for visualization):
    -  0: empty
    - -1: static wall
    -  1: start
    -  2: target
    -  3: frontier (in open list)
    -  4: explored (already expanded)
    -  5: final path
    """

    EMPTY = 0
    WALL = -1
    DYNAMIC_WALL = -2
    START = 1
    END = 2
    FRONTIER = 3
    EXPLORED = 4
    PATH = 5

    def __init__(self, rows: int = 8, cols: int = 8):
        self.rows = rows
        self.cols = cols
        self.grid = np.zeros((rows, cols), dtype=int)
        
        self.visit_order = np.full((rows, cols), -1, dtype=int)
        self._visit_counter = 0
    
        self.max_dynamic_walls: int | None = None

    
        self.start = (3, 5)
        self.end = (5, 1)
        self.static_walls: list = []
        self.dynamic_walls: set = set()

        # make a vertical wall in the middle (default)
        self.static_walls = [(i, 3) for i in range(1, 7)]

        self.reset()

    # ------------------------------------------------------------------
    # basic helpers
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """
        Reset all non-wall markings (frontier / explored / path) but keep
        static walls and the current start/end positions.
        """
        self.grid.fill(self.EMPTY)
        self.visit_order.fill(-1)
        self._visit_counter = 0

        # static walls
        for r, c in self.static_walls:
            self.grid[r, c] = self.WALL

        # dynamic walls
        for r, c in self.dynamic_walls:
            self.grid[r, c] = self.DYNAMIC_WALL

        # start / end
        sr, sc = self.start
        er, ec = self.end
        self.grid[sr, sc] = self.START
        self.grid[er, ec] = self.END

    def clear_search_marks(self) -> None:
        """Remove FRONTIER / EXPLORED / PATH marks from the grid."""
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r, c] in (self.FRONTIER, self.EXPLORED, self.PATH):
                    self.grid[r, c] = self.EMPTY
                # clear any visit labels for next run
                if self.visit_order[r, c] >= 0 and self.grid[r, c] == self.EMPTY:
                    self.visit_order[r, c] = -1
        
        sr, sc = self.start
        er, ec = self.end
        self.grid[sr, sc] = self.START
        self.grid[er, ec] = self.END
        # mark start node as first in visit order
        self.mark_visit(self.start)

    def in_bounds(self, node) -> bool:
        r, c = node
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_free(self, node) -> bool:
        """Walkable for the agent (not a wall or dynamic wall)."""
        r, c = node
        return self.grid[r, c] not in (self.WALL, self.DYNAMIC_WALL)

    def neighbors(self, node):
        """
        Clockwise movement order:

        1. Up
        2. Right
        3. Bottom
        4. Bottom-Right (diagonal)
        5. Left
        6. Top-Left (diagonal)

        """
        r, c = node
        moves = [
            (-1, 0),   # up
            (0, 1),    # right
            (1, 0),    # bottom
            (1, 1),    # bottom-right (main diagonal)
            (0, -1),   # left
            (-1, -1),  # top-left (main diagonal)
        ]
        for dr, dc in moves:
            nr, nc = r + dr, c + dc
            nxt = (nr, nc)
            if self.in_bounds(nxt) and self.is_free(nxt):
                yield nxt

    # ------------------------------------------------------------------
    # grid setup helpers
    # ------------------------------------------------------------------
    def set_dimensions(self, rows: int, cols: int) -> None:
        """Resize the grid and reinitialize."""
        self.rows = rows
        self.cols = cols
        self.grid = np.zeros((rows, cols), dtype=int)
        self.visit_order = np.full((rows, cols), -1, dtype=int)
        self._visit_counter = 0
        # place start and end in valid corners
        self.start = (0, 0)
        self.end = (rows - 1, cols - 1)
        self.static_walls = []
        self.dynamic_walls = set()
        self.reset()

    def generate_random_map(self, obstacle_density: float) -> None:
        """Generate random walls with given density (e.g. 0.3 = 30% walls)."""
        self.static_walls = []
        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) in (self.start, self.end):
                    continue
                if np.random.random() < obstacle_density:
                    self.static_walls.append((r, c))
        self.reset()

    def toggle_wall(self, node) -> None:
        """Toggle wall at node: add if empty, remove if wall. Ignores start/end."""
        r, c = node
        if node in (self.start, self.end):
            return
        if self.grid[r, c] == self.WALL:
            self.static_walls = [(a, b) for (a, b) in self.static_walls if (a, b) != node]
            self.grid[r, c] = self.EMPTY
        elif self.grid[r, c] == self.EMPTY:
            self.static_walls.append(node)
            self.grid[r, c] = self.WALL
        self.grid[self.start] = self.START
        self.grid[self.end] = self.END

    def add_dynamic_wall(self, node) -> None:
        if self.in_bounds(node) and node not in (self.start, self.end):
            if self.grid[node[0], node[1]] == self.EMPTY:
                self.grid[node[0], node[1]] = self.DYNAMIC_WALL
                self.dynamic_walls.add(node)

    def clear_dynamic_walls(self) -> None:
        for r, c in list(self.dynamic_walls):
            self.grid[r, c] = self.EMPTY
        self.dynamic_walls.clear()

    # ------------------------------------------------------------------
    # visit order helpers (for numeric labels in GUI)
    # ------------------------------------------------------------------
    def mark_visit(self, node) -> None:
        """Assign a unique incremental id the first time a node is discovered."""
        r, c = node
        if self.visit_order[r, c] == -1:
            self.visit_order[r, c] = self._visit_counter
            self._visit_counter += 1


