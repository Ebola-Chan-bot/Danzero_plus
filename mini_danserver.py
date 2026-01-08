"""Workspace-root entrypoint for the mini danserver.

This repo's original implementation lives at:
  wintest/mini_server/mini_danserver.py

Keeping a thin wrapper at workspace root makes it easier to run:
  python mini_danserver.py

without turning `wintest/` into a Python package.
"""

from __future__ import annotations

import os
import runpy
import sys
import asyncio


def main() -> None:
    repo_root = os.path.dirname(__file__)
    target = os.path.join(repo_root, "wintest", "mini_server", "mini_danserver.py")
    if not os.path.exists(target):
        raise SystemExit(f"Cannot find mini_danserver implementation at: {target}")

    # Forward CLI args to the underlying script.
    sys.argv[0] = target
    try:
      runpy.run_path(target, run_name="__main__")
    except (KeyboardInterrupt, asyncio.CancelledError, EOFError):
      print("已退出。")
      return


if __name__ == "__main__":
    main()
