"""zylab GUI 入口（转发到 zylab.gui.app.main）."""

from __future__ import annotations

import sys

from zylab.gui.app import main as gui_main

__all__ = ["main"]


def main() -> int:
    """zylab GUI 入口（转发到 zylab.gui.app.main）."""
    return gui_main()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
