"""
AIPathFinder - Full Application with Tkinter + Matplotlib
Dynamic grid sizing, random map, interactive editor, informed search (GBFS, A*), metrics.
"""

import time
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np

from grid_env import Grid
from view_gui import GridGUI
from search_bfs import bfs
from search_dfs import dfs
from search_ucs import ucs
from search_dls import dls
from search_iddfs import iddfs
from search_bidirectional import bidirectional_search
from search_gbfs import gbfs
from search_astar import astar
class App:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AIPathFinder - Dynamic Pathfinding Agent")
        self.root.geometry("900x750")
        self.root.minsize(700, 600)

        self.rows = 8
        self.cols = 8
        self.grid = Grid(rows=self.rows, cols=self.cols)
        self.gui = None
        self.canvas_widget = None
        self.edit_mode = True
        self._click_cid = None
        self._release_cid = None
        self._drag_which = None  # "start" or "end" when dragging

        self._build_ui()

    def _build_ui(self):
        # Control frame
        ctrl = ttk.Frame(self.root, padding=8)
        ctrl.pack(fill=tk.X)

        ttk.Label(ctrl, text="Grid:").pack(side=tk.LEFT, padx=4)
        self.rows_var = tk.StringVar(value="8")
        self.cols_var = tk.StringVar(value="8")
        ttk.Entry(ctrl, textvariable=self.rows_var, width=4).pack(side=tk.LEFT, padx=2)
        ttk.Label(ctrl, text="x").pack(side=tk.LEFT)
        ttk.Entry(ctrl, textvariable=self.cols_var, width=4).pack(side=tk.LEFT, padx=2)
        ttk.Button(ctrl, text="Resize", command=self._resize).pack(side=tk.LEFT, padx=8)

        ttk.Label(ctrl, text="Obstacle density (0-1):").pack(side=tk.LEFT, padx=(16, 4))
        self.density_var = tk.StringVar(value="0.2")
        ttk.Entry(ctrl, textvariable=self.density_var, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Button(ctrl, text="Random Map", command=self._random_map).pack(side=tk.LEFT, padx=4)

        ttk.Button(ctrl, text="Edit Map (click cells)", command=self._toggle_edit).pack(side=tk.LEFT, padx=8)

        ttk.Separator(ctrl, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        ttk.Label(ctrl, text="Algorithm:").pack(side=tk.LEFT, padx=4)
        self.algo_var = tk.StringVar(value="BFS")
        algo_combo = ttk.Combobox(
            ctrl, textvariable=self.algo_var, width=14,
            values=["BFS", "DFS", "UCS", "DLS", "IDDFS", "Bidirectional", "GBFS", "A*"],
            state="readonly",
        )
        algo_combo.pack(side=tk.LEFT, padx=2)
        algo_combo.bind("<<ComboboxSelected>>", self._on_algo_change)

        ttk.Label(ctrl, text="Heuristic:").pack(side=tk.LEFT, padx=(8, 4))
        self.heur_var = tk.StringVar(value="manhattan")
        self.heur_combo = ttk.Combobox(
            ctrl, textvariable=self.heur_var, width=10,
            values=["manhattan", "euclidean"], state="readonly",
        )
        self.heur_combo.pack(side=tk.LEFT, padx=2)
        self._on_algo_change(None)

        self.dynamic_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(ctrl, text="Dynamic obstacles", variable=self.dynamic_var).pack(side=tk.LEFT, padx=8)

        ttk.Button(ctrl, text="Run Search", command=self._run_search).pack(side=tk.LEFT, padx=8)
        ttk.Button(ctrl, text="Refresh Grid", command=self._refresh_grid).pack(side=tk.LEFT, padx=4)

        # Metrics frame
        self.metrics_var = tk.StringVar(value="Nodes Visited: -  |  Path Cost: -  |  Execution Time: -")
        ttk.Label(self.root, textvariable=self.metrics_var, font=("Consolas", 10)).pack(pady=4)

        # Create GridGUI (which owns the Matplotlib figure/axes)
        self._init_gui()

        # Embed the GridGUI figure inside Tkinter
        self.canvas_widget = FigureCanvasTkAgg(self.gui.fig, master=self.root)
        self.canvas_widget.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.canvas_widget.draw()

        self._update_figure()
        self._connect_click()

    def _init_gui(self):
        """Create the GridGUI wrapper around a single Matplotlib figure."""
        self.gui = GridGUI(self.grid, title="AIPathFinder")

    def _connect_click(self):
        if self._click_cid is not None:
            self.gui.fig.canvas.mpl_disconnect(self._click_cid)
        if self._release_cid is not None:
            self.gui.fig.canvas.mpl_disconnect(self._release_cid)
        self._click_cid = self.gui.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self._release_cid = self.gui.fig.canvas.mpl_connect("button_release_event", self._on_release)

    def _on_click(self, event):
        if not self.edit_mode or event.inaxes != self.gui.ax:
            return
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        col = int(round(x))
        row = int(round(y))
        if 0 <= row < self.grid.rows and 0 <= col < self.grid.cols:
            node = (row, col)
            # If clicking on S or T, start drag operation
            if node == self.grid.start:
                self._drag_which = "start"
                return
            if node == self.grid.end:
                self._drag_which = "end"
                return
            # Otherwise toggle wall
            self.grid.toggle_wall(node)
            self._update_figure()

    def _on_release(self, event):
        if not self.edit_mode or not self._drag_which or event.inaxes != self.gui.ax:
            self._drag_which = None
            return
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            self._drag_which = None
            return
        col = int(round(x))
        row = int(round(y))
        if 0 <= row < self.grid.rows and 0 <= col < self.grid.cols:
            dest = (row, col)
            # Do not move onto the other special node
            other = self.grid.end if self._drag_which == "start" else self.grid.start
            if dest != other:
                if self._drag_which == "start":
                    self.grid.start = dest
                else:
                    self.grid.end = dest
                self.grid.reset()
                self._update_figure()
        self._drag_which = None

    def _toggle_edit(self):
        self.edit_mode = not self.edit_mode
        status = "ON (click to toggle walls)" if self.edit_mode else "OFF"
        messagebox.showinfo("Edit Map", f"Edit mode: {status}")

    def _on_algo_change(self, e):
        algo = self.algo_var.get()
        if algo in ("GBFS", "A*"):
            self.heur_combo.configure(state="readonly")
        else:
            self.heur_var.set("manhattan")
            self.heur_combo.configure(state="disabled")

    def _resize(self):
        try:
            r = int(self.rows_var.get())
            c = int(self.cols_var.get())
            if r < 2 or c < 2 or r > 50 or c > 50:
                raise ValueError("Grid must be 2-50")
            self.rows, self.cols = r, c
            self.grid.set_dimensions(r, c)
            self._update_figure()
        except Exception as ex:
            messagebox.showerror("Resize", str(ex))

    def _random_map(self):
        try:
            d = float(self.density_var.get())
            if not 0 <= d <= 1:
                raise ValueError("Density must be 0-1")
            self.grid.generate_random_map(d)
            self._update_figure()
        except Exception as ex:
            messagebox.showerror("Random Map", str(ex))

    def _update_figure(self, metrics=None):
        self.gui._metrics = metrics
        self.gui.update(pause=0)
        self.canvas_widget.draw()

    def _refresh_grid(self):
        """Clear search marks and dynamic walls, keep static walls/start/end."""
        self.grid.clear_dynamic_walls()
        self.grid.clear_search_marks()
        self.metrics_var.set("Nodes Visited: -  |  Path Cost: -  |  Execution Time: -")
        self._update_figure()

    def _run_search(self):
        self.edit_mode = False
        algo = self.algo_var.get()
        heuristic = self.heur_var.get()
        dynamic = self.dynamic_var.get()
        pause = 0.1
        metrics = {}

        # Dynamic obstacle spawn probability (per search step)
        # Slightly lower so the spawning feels less chaotic.
        dynamic_prob = 0.3 if dynamic else 0.0

        # Build search function
        if algo == "BFS":
            def fn(g, gui, pause=0.1, skip_follow=False, dynamic_prob=0.0):
                return bfs(g, gui, pause=pause, skip_follow=skip_follow, metrics=metrics, dynamic_prob=dynamic_prob)
        elif algo == "DFS":
            def fn(g, gui, pause=0.1, skip_follow=False, dynamic_prob=0.0):
                return dfs(g, gui, pause=pause, skip_follow=skip_follow, metrics=metrics, dynamic_prob=dynamic_prob)
        elif algo == "UCS":
            def fn(g, gui, pause=0.1, skip_follow=False, dynamic_prob=0.0):
                return ucs(g, gui, pause=pause, skip_follow=skip_follow, metrics=metrics, dynamic_prob=dynamic_prob)
        elif algo == "DLS":
            def fn(g, gui, pause=0.1, skip_follow=False, dynamic_prob=0.0):
                return dls(g, gui, depth_limit=20, pause=pause, skip_follow=skip_follow, metrics=metrics, dynamic_prob=dynamic_prob)
        elif algo == "IDDFS":
            def fn(g, gui, pause=0.1, skip_follow=False, dynamic_prob=0.0):
                return iddfs(g, gui, max_depth=30, pause=pause, skip_follow=skip_follow, metrics=metrics, dynamic_prob=dynamic_prob)
        elif algo == "Bidirectional":
            def fn(g, gui, pause=0.1, skip_follow=False, dynamic_prob=0.0):
                return bidirectional_search(g, gui, pause=pause, skip_follow=skip_follow, metrics=metrics, dynamic_prob=dynamic_prob)
        elif algo == "GBFS":
            def fn(g, gui, pause=0.1, skip_follow=False, dynamic_prob=0.0):
                return gbfs(g, gui, heuristic=heuristic, pause=pause, skip_follow=skip_follow, metrics=metrics, dynamic_prob=dynamic_prob)
        elif algo == "A*":
            def fn(g, gui, pause=0.1, skip_follow=False, dynamic_prob=0.0):
                return astar(g, gui, heuristic=heuristic, pause=pause, skip_follow=skip_follow, metrics=metrics, dynamic_prob=dynamic_prob)
        else:
            messagebox.showerror("Error", "Unknown algorithm")
            return

        self.gui.title = f"AIPathFinder - {algo}"
        self.grid.clear_search_marks()
        self.grid.reset()
        self._update_figure()

        path = fn(self.grid, self.gui, pause=pause, dynamic_prob=dynamic_prob)

        self._update_figure(metrics)
        n = metrics.get("nodes_visited", "-")
        c = metrics.get("path_cost", "-")
        t = metrics.get("exec_time_ms", "-")
        t_str = f"{t:.1f} ms" if isinstance(t, (int, float)) else str(t)
        self.metrics_var.set(f"Nodes Visited: {n}  |  Path Cost: {c}  |  Execution Time: {t_str}")

        if not path:
            messagebox.showwarning("No Path", "No path found - target may be blocked.")
        else:
            messagebox.showinfo("Done", "Search complete.")

        # re-enable editing after search completes
        self.edit_mode = True

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = App()
    app.run()
