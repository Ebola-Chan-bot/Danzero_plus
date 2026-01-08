"""工作区根目录入口：mini danserver 启动封装。

本仓库的主要实现位于：
  wintest/mini_server/mini_danserver.py

把一个薄封装放在工作区根目录，方便直接运行：
  python mini_danserver.py

同时避免把 wintest/ 变成 Python 包。
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

    # 将命令行参数透传给底层脚本。
    sys.argv[0] = target
    try:
      runpy.run_path(target, run_name="__main__")
    except (KeyboardInterrupt, asyncio.CancelledError, EOFError):
      print("已退出。")
      return


if __name__ == "__main__":
    main()
