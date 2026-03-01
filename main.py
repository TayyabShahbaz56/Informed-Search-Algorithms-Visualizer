"""
AIPathFinder
============

Entry point for the full GUI application.

Usage (from this folder):
    python main.py

This simply launches the Tkinter-based UI defined in `app_main.py`.
"""

from app_main import App


def main() -> None:
    app = App()
    app.run()


if __name__ == "__main__":
    main()


