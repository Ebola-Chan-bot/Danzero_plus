# mini_server（路线3：自建对局裁判，支持指定 AI 起手牌）

这个目录提供一个最小可用的 WebSocket 牌局服务端（替代 `danserver`），用于：
- **指定 AI（seat0）起手 27 张牌**（两副牌制，允许重复）。
- 其余 **seat1/2/3 在同一终端手动输入出牌**，观察 AI 反应。

## 依赖

需要 Python 包：
- `websockets`

安装：

```bash
pip install websockets
```

## 单终端模式（默认）

直接运行：

```bash
python mini_danserver.py
```

（等价于运行 `wintest/mini_server/mini_danserver.py`，只是入口放在工作区根目录更顺手。）

程序启动后会在终端提示你输入：
- 当前级牌 `curRank`（回车默认 2）
- AI(seat0) 起手 27 张牌（回车=随机）

然后它会在同一终端里自动启动：
- `wintest/torch/actor.py --seats 0`
- `wintest/torch/client1.py --seat 0`

接下来你就在 **同一个终端** 里按提示为 seat1/2/3 输入出牌：
- `PASS`（或 `P`）表示过
- 或输入卡牌列表，例如：`H3 H3`、`S7`、`SA HA CA H2`（会尝试识别为 Bomb/ThreeWithTwo 等）

## 不合并终端（可选）

如果你不想让程序自动启动 actor/client（想自己分开跑），用：

```bash
python wintest/mini_server/mini_danserver.py --no-spawn-ai
```

## 说明与限制

- 这是实验用最小裁判：只实现了常见牌型与一个“够用”的压制关系（用于过滤 AI 可选动作、校验人工输入）。
- 暂不实现 `episodeOver/gameResult` 完整协议；默认用 `--max-steps` 控制对局推进步数。
