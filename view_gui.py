import time
import matplotlib.pyplot as plt  # type: ignore
import matplotlib.colors as mcolors  # type: ignore
import matplotlib.patches as mpatches  # type: ignore

from grid_env import Grid


class GridGUI:
    """
    Simple Matplotlib based GUI for visualizing search on the grid.

    Each update completely redraws the matrix so that the animation is easy
    to follow and consistent across algorithms.
    """

    # softer, modern colour palette (original light theme)
    COLOR_MAP = {
        Grid.EMPTY: "#f4f4f4",
        Grid.WALL: "#ff4d4f",
        Grid.DYNAMIC_WALL: "#b71c1c",
        Grid.START: "#00b894",
        Grid.END: "#0984e3",
        Grid.FRONTIER: "#ffeaa7",
        Grid.EXPLORED: "#a3e4d7",
        Grid.PATH: "#00c96b",
    }

    def __init__(self, grid: Grid, title: str = "AIPathFinder"):
        self.grid = grid
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.title = title
        self._metrics: dict | None = None

        # build a fixed colormap for the discrete codes
        ordered_keys = sorted(self.COLOR_MAP.keys())
        self._bounds = ordered_keys
        self._cmap = mcolors.ListedColormap([self.COLOR_MAP[k] for k in ordered_keys])
        self._norm = mcolors.BoundaryNorm(self._bounds + [self._bounds[-1] + 1], self._cmap.N)

    def show_initial(self):
        """Legacy helper (no-op in Tkinter app)."""
        self.update()

    def block_until_closed(self):
        """Legacy helper (no-op in Tkinter app)."""
        plt.show(block=True)

    def update(self, pause: float = 0.1):
        self.ax.clear()
        # leave some extra vertical space at the top for the legend
        self.ax.set_title(self.title, fontsize=16, fontweight="bold", pad=20)
        # draw subtle grid lines, hide tick labels
        self.ax.set_xticks(range(self.grid.cols))
        self.ax.set_yticks(range(self.grid.rows))
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        # thicker border and slightly darker outer background (light theme)
        self.ax.grid(which="both", color="#bfbfbf", linewidth=0.8)
        self.ax.set_facecolor("#d9d9d9")
        self.ax.set_aspect("equal")

        # compact legend explaining cell roles; place it outside the grid on the right
        legend_handles = [
            mpatches.Patch(
                facecolor=self.COLOR_MAP[Grid.PATH],
                edgecolor="black",
                label="Visited path",
            ),
            mpatches.Patch(
                facecolor=self.COLOR_MAP[Grid.FRONTIER],
                edgecolor="black",
                label="Frontier (in queue)",
            ),
            mpatches.Patch(
                facecolor=self.COLOR_MAP[Grid.EXPLORED],
                edgecolor="black",
                label="Expanded / explored",
            ),
            mpatches.Patch(
                facecolor=self.COLOR_MAP[Grid.WALL],
                edgecolor="black",
                label="Static obstacle",
            ),
            mpatches.Patch(
                facecolor=self.COLOR_MAP[Grid.DYNAMIC_WALL],
                edgecolor="black",
                label="Dynamic obstacle",
            ),
        ]
        self.ax.legend(
            handles=legend_handles,
            loc="upper right",
            bbox_to_anchor=(-0.02, 1.0),
            ncol=1,
            frameon=False,
            fontsize=8,
        )

        # metrics dashboard
        if self._metrics:
            m = self._metrics
            nodes = m.get("nodes_visited", "-")
            cost = m.get("path_cost", "-")
            t_ms = m.get("exec_time_ms")
            time_str = f"{t_ms:.1f} ms" if isinstance(t_ms, (int, float)) else str(t_ms)
            self.ax.text(
                0.5, -0.12,
                f"Nodes Visited: {nodes}  |  Path Cost: {cost}  |  Execution Time: {time_str}",
                ha="center", va="top", transform=self.ax.transAxes,
                fontsize=10, family="monospace",
            )

        vis = self.grid.grid
        self.ax.imshow(vis, cmap=self._cmap, norm=self._norm)

        # numeric labels (visit order) and S/T marker so it looks like the assignment mockup
        for r in range(self.grid.rows):
            for c in range(self.grid.cols):
                cell_code = self.grid.grid[r, c]
                visit_val = self.grid.visit_order[r, c]

                # numeric label for every cell
                if cell_code in (Grid.WALL, Grid.DYNAMIC_WALL):
                    txt = "-1"
                elif cell_code == Grid.PATH:
                    # path cells are always labelled 0
                    txt = "0"
                else:
                    txt = str(visit_val) if visit_val >= 0 else "0"

                self.ax.text(
                    c,
                    r,
                    txt,
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=10,
                )

                # overlays for start / end on top of cell colour
                if (r, c) == self.grid.start:
                    self.ax.text(
                        c,
                        r,
                        "S",
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=14,
                        fontweight="bold",
                    )
                if (r, c) == self.grid.end:
                    self.ax.text(
                        c,
                        r,
                        "T",
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=14,
                        fontweight="bold",
                    )

        if self.fig.canvas is not None:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        if pause and pause > 0:
            # small delay so the user can see the animation steps
            time.sleep(pause)

    def set_metrics(self, metrics: dict | None) -> None:
        """Update metrics for the dashboard display."""
        self._metrics = metrics


