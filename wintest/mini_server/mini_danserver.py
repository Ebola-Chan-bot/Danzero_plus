"""兼容入口：实现已迁移到仓库根目录的 mini_danserver.py。

请直接运行：
    python mini_danserver.py
"""

from __future__ import annotations

import os
import runpy
import sys

def main() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    target = os.path.join(repo_root, "mini_danserver.py")
    if not os.path.exists(target):
        raise SystemExit(f"Cannot find mini_danserver implementation at: {target}")

    sys.argv[0] = target
    runpy.run_path(target, run_name="__main__")


if __name__ == "__main__":
    main()
