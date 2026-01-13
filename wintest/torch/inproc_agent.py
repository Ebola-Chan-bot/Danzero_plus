from __future__ import annotations

# 说明：在 Python 3.13 下，如果没有 postponed annotations，
# 某些“前向引用”的类型注解会在运行时被求值，可能触发 NameError。
# 这里统一延后注解求值，保证 import inproc_agent 时不因注解而崩。

from dataclasses import dataclass
from functools import reduce
from random import randint
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import os
import io
import pickle
from pathlib import Path

from util import card2array, card2num
from model import MLPActorCritic, MLPQNetwork


RANK = {
    '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6, '8': 7, '9': 8,
    'T': 9, 'J': 10, 'Q': 11, 'K': 12, 'A': 13
}


ACTION_NUMBER = 2

# proc_universal 输出维度：
# - 12 维手牌结构标志（兼容原 torch 客户端）
# - 8 维：按同点数张数(1..8)分类的“平均大小”
# - 6 维：单顺/双顺/三顺的“最长连顺窗口”(长度+平均大小)
UNIVERSAL_DIM = 26


_RANKS_13 = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']


def _rank_char_to_index(r: Any) -> Optional[int]:
    if not isinstance(r, str) or len(r) != 1:
        return None
    try:
        return _RANKS_13.index(r)
    except ValueError:
        return None


def _encode_real_rank_counts(real_cards: List[int]) -> np.ndarray:
    """13维：从实体出牌(0..53)统计点数 2..A 的计数（不含大小王）。"""

    out = np.zeros(13, dtype=np.int8)
    for c in real_cards or []:
        try:
            ci = int(c)
        except Exception:
            continue
        if ci < 0:
            continue
        if ci >= 52:
            # 大小王（SB/HR）不计入 2..A 的点数直方图
            continue
        r = ci % 13
        out[r] = np.int8(min(10, int(out[r]) + 1))
    return out


def _encode_virtual_rank_counts(detail: Optional[dict], real_cards: Optional[List[int]] = None) -> np.ndarray:
    """13维：虚拟点数计数（用于支持九/十炸等 >8 的同点数张数）。

    优先使用 virtual_cards（含花色）来取点数；否则使用 virtual_ranks。
    不统计大小王（B/R）。
    """

    out = np.zeros(13, dtype=np.int8)
    if not isinstance(detail, dict):
        return out

    vc = detail.get('virtual_cards')
    if isinstance(vc, list):
        for c in vc:
            if not isinstance(c, str) or len(c) < 2:
                continue
            r = c[-1]
            idx = _rank_char_to_index(r)
            if idx is None:
                continue
            out[idx] = np.int8(min(10, int(out[idx]) + 1))
        return out

    vr = detail.get('virtual_ranks')
    if isinstance(vr, list):
        for r in vr:
            idx = _rank_char_to_index(r)
            if idx is None:
                continue
            out[idx] = np.int8(min(10, int(out[idx]) + 1))

    # 如果没有提供任何 virtual 信息，则回退为实体牌点数统计（保持“点数直方图”语义）。
    if int(out.sum()) == 0 and real_cards is not None:
        return _encode_real_rank_counts(real_cards)

    return out


def _encode_action_feature(real_cards: List[int], detail: Optional[dict]) -> np.ndarray:
    """动作特征：54(实体牌) + 13(虚拟点数计数) = 67维。

    说明：按规则“能补成同花顺就一律当同花顺炸弹”，无需编码虚拟花色。
    """

    real54 = card2array(real_cards)
    vr13 = _encode_virtual_rank_counts(detail, real_cards)
    return np.concatenate([real54, vr13]).astype(np.int8)


def _torch_load_bytes_cpu(b: bytes):
    bio = io.BytesIO(b)
    try:
        return torch.load(bio, map_location='cpu', weights_only=True)
    except TypeError:
        return torch.load(bio, map_location='cpu')


class _CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return _torch_load_bytes_cpu
        return super().find_class(module, name)


def _adapt_actor_state_dict_for_obs_dim(weights: dict, target_obs_dim: int) -> dict:
    """兼容加载 actor 权重（仅支持最新与次新观测维度）。

    支持的 actor 观测维度：
    - 最新：904（invariant=574）
    - 次新：901（invariant=571）

    更老版本（如 843/849/847 等）不再兼容：发现后直接报错，避免“加载成功但语义错位”。
    """

    if not isinstance(weights, dict):
        return weights

    wkey = 'shared.0.weight'
    if wkey not in weights:
        return weights

    w = weights.get(wkey)
    if not isinstance(w, torch.Tensor) or w.ndim != 2:
        return weights

    out_features, in_features = int(w.shape[0]), int(w.shape[1])
    if in_features == int(target_obs_dim):
        return weights

    supported = {901, 904}
    if int(target_obs_dim) not in supported:
        raise ValueError(f"unsupported target actor obs_dim={int(target_obs_dim)}; supported={sorted(supported)}")
    if int(in_features) not in supported:
        raise ValueError(
            f"unsupported saved actor obs_dim={int(in_features)} (first layer '{wkey}'); supported={sorted(supported)}"
        )

    if in_features > int(target_obs_dim):
        weights[wkey] = w[:, : int(target_obs_dim)].contiguous()
        return weights

    new_w = torch.zeros((out_features, int(target_obs_dim)), dtype=w.dtype)
    new_w[:, :in_features] = w
    weights[wkey] = new_w
    return weights


def _adapt_q_state_dict_for_obs_dim(weights: dict, target_obs_dim: int) -> dict:
    """兼容加载 q_network 权重（仅支持最新与次新观测维度）。

    支持的 q 观测维度：
    - 最新：752（invariant=574）
    - 次新：749（invariant=571）

    更老版本不再兼容。
    """

    if not isinstance(weights, dict):
        return weights

    wkey = 'q.q_net.0.weight'
    if wkey not in weights:
        return weights

    w = weights.get(wkey)
    if not isinstance(w, torch.Tensor) or w.ndim != 2:
        return weights

    out_features, in_features = int(w.shape[0]), int(w.shape[1])
    if in_features == int(target_obs_dim):
        return weights

    supported = {749, 752}
    if int(target_obs_dim) not in supported:
        raise ValueError(f"unsupported target q obs_dim={int(target_obs_dim)}; supported={sorted(supported)}")
    if int(in_features) not in supported:
        raise ValueError(
            f"unsupported saved q obs_dim={int(in_features)} (first layer '{wkey}'); supported={sorted(supported)}"
        )
    if in_features > int(target_obs_dim):
        # 观测维度缩小：对第一层权重做截断。
        weights[wkey] = w[:, : int(target_obs_dim)].contiguous()
        return weights

    new_w = torch.zeros((out_features, int(target_obs_dim)), dtype=w.dtype)
    new_w[:, :in_features] = w
    weights[wkey] = new_w
    return weights


class _InProcPlayer:
    def __init__(self, args: TorchInProcArgs, *, actor_obs_dim: int, q_obs_dim: int) -> None:
        base_dir = Path(__file__).resolve().parent

        # 模型初始化
        self.model_id = int(args.iter) * 2000 + 500
        self.model = MLPActorCritic((ACTION_NUMBER, int(actor_obs_dim)), ACTION_NUMBER)

        model_path = base_dir / 'models' / f'ppo{self.model_id}.pth'
        if model_path.exists():
            with model_path.open('rb') as f:
                new_weights = _CPU_Unpickler(f).load()
            new_weights = _adapt_actor_state_dict_for_obs_dim(new_weights, int(actor_obs_dim))
            self.model.set_weights(new_weights)

        self.model_q = MLPQNetwork(int(q_obs_dim))
        q_path = base_dir / 'q_network.ckpt'
        if q_path.exists():
            with q_path.open('rb') as f:
                tf_weights = pickle.load(f)
            self.model_q.load_tf_weights(tf_weights)


def _get_one_hot_array(num_left_cards: int, max_num_cards: int, flag: int):
    if flag == 0:  # 级数
        one_hot = np.zeros(max_num_cards)
        one_hot[num_left_cards - 1] = 1
    else:
        one_hot = np.zeros(max_num_cards + 1)  # 剩余牌数 0..max
        one_hot[num_left_cards] = 1
    return one_hot


def _card_list_to_54(cards: List[int]) -> np.ndarray:
    return card2array(cards)


@dataclass
class TorchInProcArgs:
    # 该参数对象需要提供 iter 字段
    iter: int = 0


class TorchInProcAgent:
    """进程内运行的 Torch AI（不使用 websockets / ZMQ / 多进程）。

    这是从原 torch WS 客户端状态机中抽出的最小实现，保留相同的 `prepare()` 特征构造，
    以保证模型行为一致。

    按需求：
    - 不做超时处理。
    - 任何异常直接抛出，让主进程崩溃（方便定位问题）。
    """

    def __init__(
        self,
        seat: int = 0,
        model_iter: int = 0,
        weights_override_path: Optional[str] = None,
        q_weights_override_path: Optional[str] = None,
    ):
        self.mypos = seat
        self.tribute_result = []

        # 记录最近动作的“虚拟补法信息”（用于在特征里表达九/十炸与同花顺等）。
        # 与 action_seq / history_action 对齐：每次 append 一条动作，就 append 一条 detail（或 None）。
        self.action_seq_detail: List[Optional[dict]] = []
        self.history_action_detail: Dict[int, List[Optional[dict]]] = {0: [], 1: [], 2: [], 3: []}

        # 公开信息：由于进贡/还贡产生的“明牌”（可见手牌）。
        # 仅记录“已知在某个座位手里”的牌（54 维计数，范围 0..2）。
        # 这些信息会在 prepare() 里通过独立的 known(明牌) 通道加入特征，
        # 不会混入“已出牌(played_cards)”统计。
        self.visible_handcards_by_pos: Dict[int, np.ndarray] = {
            0: np.zeros(54, dtype=np.int8),
            1: np.zeros(54, dtype=np.int8),
            2: np.zeros(54, dtype=np.int8),
            3: np.zeros(54, dtype=np.int8),
        }

        # 抗贡阶段（可选公开信息）；暂不直接进特征，但保留以便后续扩展。
        self.anti_pos: List[int] = []

        # 统计字段（用于特征）
        self.rank = 1
        self.oppo_rank = 1
        self.playing_self = -1
        self.count_A = 0
        self.count_A_self = 0
        self.count_A_oppo = 0

        # 对局状态（用于特征）
        self.history_action = {0: [], 1: [], 2: [], 3: []}
        self.action_seq: List[List[int]] = []
        self.other_left_hands = [2 for _ in range(54)]
        self.flag = 0
        self.action_order: List[int] = []
        self.remaining = {0: 27, 1: 27, 2: 27, 3: 27}
        self.over: List[int] = []

        # 观测拆分（保持旧特征顺序不变，在尾部追加“剩余牌库结构特征”以便兼容旧权重）：
        # - 不可变状态块（与候选动作无关）：574 维
        # - 可变块（候选动作会改变的部分）：152 维
        #   = my_hand_after(54) + universal_after(26) + last_action_after(67) + rel_greater_after(5)
        # - 追加尾部：deck_universal(26)，对所有候选动作相同
        # 额外不变特征：贡牌目标（上家/队友/下家）3 维
        # q_network 输入（每个候选动作一行）= invariant + variant + deck_universal = 752 维
        # actor 输入 = invariant + top2_variant(2*152) + deck_universal = 904 维
        self._invariant_dim = 574
        self._variant_dim = 54 + UNIVERSAL_DIM + 67 + 5
        self._deck_universal_dim = UNIVERSAL_DIM
        actor_obs_dim = int(self._invariant_dim + ACTION_NUMBER * self._variant_dim + self._deck_universal_dim)
        q_obs_dim = int(self._invariant_dim + self._variant_dim + self._deck_universal_dim)

        self._actor_obs_dim = int(actor_obs_dim)
        self._q_obs_dim = int(q_obs_dim)

        args = TorchInProcArgs(iter=model_iter)
        self.player = _InProcPlayer(args, actor_obs_dim=self._actor_obs_dim, q_obs_dim=self._q_obs_dim)

        # 可选：加载“进程内训练”产出的最新策略/价值网络权重。
        if weights_override_path:
            try:
                self._load_actor_weights(weights_override_path)
                if os.path.exists(weights_override_path):
                    print(f"[inproc] loaded actor weights: {weights_override_path}")
            except FileNotFoundError:
                pass

        # 可选：加载“进程内训练”产出的最新 q 网络权重（state_dict 格式）。
        self._q_weights_override_path = q_weights_override_path
        if q_weights_override_path and os.path.exists(q_weights_override_path):
            try:
                qw = self._torch_load_cpu(q_weights_override_path)
                if not isinstance(qw, dict):
                    raise TypeError(f"unexpected q weights type: {type(qw).__name__}")
                qw = _adapt_q_state_dict_for_obs_dim(qw, int(q_obs_dim))
                self.player.model_q.load_state_dict(qw)
                print(f"[inproc] loaded q_network weights: {q_weights_override_path}")
            except Exception as e:
                # q 权重缺失/损坏/维度不兼容时不影响运行：回退使用仓库自带的默认权重文件（q_network.ckpt）。
                print(f"[inproc] skip q override (will use default q_network.ckpt): {e}")

        # 策略梯度（PPO）轨迹缓存：[(观测向量, 合法动作掩码, 选择的动作编号), ...]
        # 动作编号的范围是 [0, ActionNumber)。
        self.trajectory: List[Tuple[np.ndarray, np.ndarray, int]] = []

        # q 网络轨迹缓存：[(候选动作特征批, 选择的动作索引), ...]
        # 这里的“动作索引”是 message['actionList'] 的原始索引（也是特征批的行号）。
        self.q_trajectory: List[Tuple[np.ndarray, int]] = []

    def _torch_load_cpu(self, path: str):
        with open(path, 'rb') as f:
            try:
                return torch.load(f, map_location='cpu', weights_only=True)
            except TypeError:
                return torch.load(f, map_location='cpu')

    def export_runtime_state(self) -> Dict[str, Any]:
        """导出“对局内可变状态”，用于服务端 GUI 撤销回退。

        仅包含与 `prepare()/act_play()/on_notify_play()` 相关的运行时字段，
        不包含网络权重/优化器等大对象。
        """

        def _copy_arr(a: Any) -> Any:
            try:
                return a.copy()
            except Exception:
                return a

        # trajectory/q_trajectory 里包含 numpy 数组，不能只做浅拷贝。
        traj = []
        try:
            for obs, mask, chosen in list(self.trajectory):
                traj.append((_copy_arr(obs), _copy_arr(mask), int(chosen)))
        except Exception:
            traj = []

        q_traj = []
        try:
            for xb, chosen_idx in list(self.q_trajectory):
                q_traj.append((_copy_arr(xb), int(chosen_idx)))
        except Exception:
            q_traj = []

        return {
            "mypos": int(self.mypos),
            "tribute_result": list(self.tribute_result),
            "action_seq_detail": [d if d is None else dict(d) for d in list(self.action_seq_detail)],
            "history_action_detail": {
                int(k): [d if d is None else dict(d) for d in list(v)] for k, v in dict(self.history_action_detail).items()
            },
            "visible_handcards_by_pos": {int(k): _copy_arr(v) for k, v in dict(self.visible_handcards_by_pos).items()},
            "anti_pos": list(self.anti_pos),
            "rank": int(self.rank),
            "oppo_rank": int(self.oppo_rank),
            "playing_self": int(self.playing_self),
            "count_A": int(self.count_A),
            "count_A_self": int(self.count_A_self),
            "count_A_oppo": int(self.count_A_oppo),
            "history_action": {int(k): [list(x) for x in list(v)] for k, v in dict(self.history_action).items()},
            "action_seq": [list(x) for x in list(self.action_seq)],
            "other_left_hands": list(self.other_left_hands),
            "flag": int(self.flag),
            "action_order": list(self.action_order),
            "remaining": {int(k): int(v) for k, v in dict(self.remaining).items()},
            "over": list(self.over),
            "trajectory": traj,
            "q_trajectory": q_traj,
        }

    def import_runtime_state(self, state: Dict[str, Any]) -> None:
        """从 export_runtime_state() 恢复对局内状态。"""

        if not isinstance(state, dict):
            return

        self.tribute_result = list(state.get("tribute_result", []))
        self.action_seq_detail = list(state.get("action_seq_detail", []))
        self.history_action_detail = {int(k): list(v) for k, v in (state.get("history_action_detail", {}) or {}).items()}

        vhp = {}
        for k, v in (state.get("visible_handcards_by_pos", {}) or {}).items():
            try:
                vhp[int(k)] = v.copy()
            except Exception:
                vhp[int(k)] = v
        if vhp:
            self.visible_handcards_by_pos = vhp

        self.anti_pos = list(state.get("anti_pos", []))
        self.rank = int(state.get("rank", self.rank))
        self.oppo_rank = int(state.get("oppo_rank", self.oppo_rank))
        self.playing_self = int(state.get("playing_self", self.playing_self))
        self.count_A = int(state.get("count_A", self.count_A))
        self.count_A_self = int(state.get("count_A_self", self.count_A_self))
        self.count_A_oppo = int(state.get("count_A_oppo", self.count_A_oppo))

        self.history_action = {int(k): [list(x) for x in list(v)] for k, v in (state.get("history_action", {}) or {}).items()}
        self.action_seq = [list(x) for x in list(state.get("action_seq", []))]
        self.other_left_hands = list(state.get("other_left_hands", self.other_left_hands))
        self.flag = int(state.get("flag", self.flag))
        self.action_order = list(state.get("action_order", []))
        self.remaining = {int(k): int(v) for k, v in (state.get("remaining", {}) or {}).items()}
        self.over = list(state.get("over", []))

        # 轨迹恢复
        self.trajectory = []
        for item in list(state.get("trajectory", [])):
            try:
                obs, mask, chosen = item
                self.trajectory.append((obs, mask, int(chosen)))
            except Exception:
                pass

        self.q_trajectory = []
        for item in list(state.get("q_trajectory", [])):
            try:
                xb, chosen_idx = item
                self.q_trajectory.append((xb, int(chosen_idx)))
            except Exception:
                pass

    def _load_actor_weights(self, path: str):
        weights = self._torch_load_cpu(path)
        if not isinstance(weights, dict):
            raise TypeError(f"unexpected weights type: {type(weights).__name__}")
        weights = _adapt_actor_state_dict_for_obs_dim(weights, int(self._actor_obs_dim))
        self.player.model.set_weights(weights)

    def save_actor_weights(self, path: str):
        # 仅保存策略/价值网络权重；q 网络权重另存。
        torch.save(self.player.model.get_weights(), path)

    def save_q_weights(self, path: str):
        # 保存 q 网络（state_dict 格式）。
        torch.save(self.player.model_q.state_dict(), path)

    def on_beginning(self, curRank: str, selfRank: str, oppoRank: str):
        # 与原客户端保持一致的计数逻辑
        self.count_A += int(curRank == 'A')
        self.count_A_self += int(selfRank == 'A')
        self.count_A_oppo += int(oppoRank == 'A')

        # 新一副牌开始：清空进/还贡产生的“明牌”缓存。
        # （在本实验服里每次启动基本都是新牌局；这里保守清空，避免跨局泄露。）
        try:
            for k in list(self.visible_handcards_by_pos.keys()):
                self.visible_handcards_by_pos[k].fill(0)
        except Exception:
            pass
        self.anti_pos = []

    def on_notify_tribute(self, result: List[List[Any]]):
        """处理 notify(tribute) 的公开信息。

        result: [[tribute_pos, receive_pos, 'S2'], ...]
        进贡牌是明牌：所有玩家都知道 receive_pos 手里新增了这张牌。
        """

        try:
            self.tribute_result = result
        except Exception:
            pass

        for item in result or []:
            try:
                _from, to_pos, card = item
                idxs = card2num([card])
                if idxs:
                    i = int(idxs[0])
                    to_pos_i = int(to_pos)
                    arr = self.visible_handcards_by_pos.get(to_pos_i)
                    if arr is not None:
                        arr[i] = np.int8(min(2, int(arr[i]) + 1))
            except Exception:
                continue

    def on_notify_back(self, result: List[List[Any]]):
        """处理 notify(back) 的公开信息。

        result: [[back_pos, receive_pos, 'S2'], ...]
        还贡牌同样是明牌：所有玩家都知道 receive_pos 手里新增了这张牌。
        """

        for item in result or []:
            try:
                _from, to_pos, card = item
                idxs = card2num([card])
                if idxs:
                    i = int(idxs[0])
                    to_pos_i = int(to_pos)
                    arr = self.visible_handcards_by_pos.get(to_pos_i)
                    if arr is not None:
                        arr[i] = np.int8(min(2, int(arr[i]) + 1))
            except Exception:
                continue

    def on_notify_anti(self, anti_pos: List[int], anti_nums: int = 0):
        """处理 notify(anti-tribute) 的公开信息。

        标准规则里“抗贡”是基于手里是否有大王等条件触发；
        协议只公开哪些座位抗贡，不一定公开具体牌。
        在本实验服里，我们按“抗贡会导致两张大王变成明牌”的约定：
        - 将对应座位的 HR（大王）计入 visible_handcards_by_pos
        - 这样无需为抗贡再单独加一条特征通道，直接复用 known(明牌) 通道即可。
        """

        try:
            self.anti_pos = [int(x) for x in (anti_pos or [])]
        except Exception:
            self.anti_pos = []

        # 注意：协议里的 antiNums 是“抗贡人数”，与 antiPos 的长度一致，不是“有几张大王”。
        # 这里按约定把“两张大王变明牌”映射到座位的可见手牌：
        # - 只有 1 家抗贡：该家持有两张大王 -> HR=2
        # - 有 2 家抗贡：两家各持有 1 张大王 -> HR=1
        # 其他情况（未知/异常）不做强行推断。
        n_players = len(self.anti_pos)
        if n_players == 1:
            per_player = 2
        elif n_players == 2:
            per_player = 1
        else:
            return

        for p in self.anti_pos:
            try:
                p_i = int(p)
                if not (0 <= p_i < 4):
                    continue
                arr = self.visible_handcards_by_pos.get(p_i)
                if arr is None:
                    continue
                # HR 在 54 维编码里是 53，计数上限 2
                arr[53] = np.int8(max(int(arr[53]), per_player))
            except Exception:
                continue

    def on_notify_play(self, curPos: int, curAction):
        """处理服务端 notify(play) 后的本地状态更新。

        curAction 的格式是：[type, key, cards, (可选 detail)]
        """
        just_play = int(curPos)
        real_action_cards: List[int] = []
        model_action_cards: List[int] = []
        detail: Optional[dict] = None
        try:
            cards = curAction[2] if curAction is not None else []
            real_action_cards = card2num(cards)

            # 不再把虚拟补法塞进 54 维花色槽位。
            # 这里的历史动作 54 维只记录“真实出牌的实体牌”，虚拟补法信息由 detail 单独记录。
            if isinstance(curAction, list) and len(curAction) >= 4 and isinstance(curAction[3], dict):
                detail = curAction[3]

            model_action_cards = list(real_action_cards)
        except Exception:
            real_action_cards = []
            model_action_cards = []
            detail = None

        if just_play != self.mypos:
            for ele in real_action_cards:
                self.other_left_hands[ele] -= 1

        # 若该座位之前有“明牌”记录，那么一旦这些牌被打出，就应从可见手牌中扣除。
        try:
            vis = self.visible_handcards_by_pos.get(just_play)
            if vis is not None:
                for ele in real_action_cards:
                    i = int(ele)
                    if 0 <= i < 54 and int(vis[i]) > 0:
                        vis[i] = np.int8(int(vis[i]) - 1)
        except Exception:
            pass

        # 注意：原客户端围绕 `over` / `flag` 有一套较复杂的顺序修正逻辑。
        # 这里保持原样，以尽量不改变模型输入分布。
        if len(self.over) == 0:
            self.action_order.append(just_play)
            self.action_seq.append(model_action_cards)
            self.action_seq_detail.append(detail)
            self.history_action[just_play].append(model_action_cards)
            self.history_action_detail[just_play].append(detail)
        elif len(self.over) == 1:
            if len(model_action_cards) > 0 and self.flag == 1:
                self.flag = 2
                if just_play == (self.over[0] + 3) % 4:
                    self.action_order.append(just_play)
                    self.action_seq.append(model_action_cards)
                    self.action_seq_detail.append(detail)
                    self.history_action[just_play].append(model_action_cards)
                    self.history_action_detail[just_play].append(detail)
                    self.action_order.append(self.over[0])
                    self.history_action[self.over[0]].append([-1])
                    self.action_seq.append([-1])
                    self.action_seq_detail.append(None)
                    self.history_action_detail[self.over[0]].append(None)
                else:
                    self.action_order.append(just_play)
                    self.action_seq.append(model_action_cards)
                    self.action_seq_detail.append(detail)
                    self.history_action[just_play].append(model_action_cards)
                    self.history_action_detail[just_play].append(detail)
            elif self.flag == 1 and (just_play + 1) % 4 == self.over[0]:
                self.flag = 2
                self.action_order.append(just_play)
                self.action_seq.append(model_action_cards)
                self.action_seq_detail.append(detail)
                self.history_action[just_play].append(model_action_cards)
                self.history_action_detail[just_play].append(detail)
                self.action_order.append(self.over[0])
                self.history_action[self.over[0]].append([-1])
                self.action_seq.append([-1])
                self.action_seq_detail.append(None)
                self.history_action_detail[self.over[0]].append(None)
                self.action_order.append((just_play + 2) % 4)
                self.history_action[(just_play + 2) % 4].append([])
                self.action_seq.append([])
                self.action_seq_detail.append(None)
                self.history_action_detail[(just_play + 2) % 4].append(None)
            elif just_play == (self.over[0] + 3) % 4 and self.flag == 2:
                self.action_order.append(just_play)
                self.action_seq.append(model_action_cards)
                self.action_seq_detail.append(detail)
                self.history_action[just_play].append(model_action_cards)
                self.history_action_detail[just_play].append(detail)
                self.action_order.append(self.over[0])
                self.history_action[self.over[0]].append([-1])
                self.action_seq.append([-1])
                self.action_seq_detail.append(None)
                self.history_action_detail[self.over[0]].append(None)
            else:
                self.action_order.append(just_play)
                self.action_seq.append(model_action_cards)
                self.action_seq_detail.append(detail)
                self.history_action[just_play].append(model_action_cards)
                self.history_action_detail[just_play].append(detail)
        elif len(self.over) == 2:
            if len(model_action_cards) > 0 and self.flag <= 2:
                if (just_play + 1) % 4 not in self.over:
                    self.flag = 3
                    self.action_order.append(just_play)
                    self.action_seq.append(model_action_cards)
                    self.action_seq_detail.append(detail)
                    self.history_action[just_play].append(model_action_cards)
                    self.history_action_detail[just_play].append(detail)
                else:
                    self.flag = 3
                    self.action_order.append(just_play)
                    self.action_seq.append(model_action_cards)
                    self.action_seq_detail.append(detail)
                    self.history_action[just_play].append(model_action_cards)
                    self.history_action_detail[just_play].append(detail)
                    self.action_order.append((just_play + 1) % 4)
                    self.history_action[(just_play + 1) % 4].append([-1])
                    self.action_seq.append([-1])
                    self.action_seq_detail.append(None)
                    self.history_action_detail[(just_play + 1) % 4].append(None)
                    self.action_order.append((just_play + 2) % 4)
                    self.history_action[(just_play + 2) % 4].append([-1])
                    self.action_seq.append([-1])
                    self.action_seq_detail.append(None)
                    self.history_action_detail[(just_play + 2) % 4].append(None)
            elif self.flag <= 2 and (just_play + 1) % 4 in self.over:
                self.flag = 3
                self.action_order.append(just_play)
                self.action_seq.append(model_action_cards)
                self.action_seq_detail.append(detail)
                self.history_action[just_play].append(model_action_cards)
                self.history_action_detail[just_play].append(detail)
                self.action_order.append((just_play + 1) % 4)
                self.history_action[(just_play + 1) % 4].append([-1])
                self.action_seq.append([-1])
                self.action_seq_detail.append(None)
                self.history_action_detail[(just_play + 1) % 4].append(None)
                self.action_order.append((just_play + 2) % 4)
                self.history_action[(just_play + 2) % 4].append([-1])
                self.action_seq.append([-1])
                self.action_seq_detail.append(None)
                self.history_action_detail[(just_play + 2) % 4].append(None)
                if just_play == (self.over[-1] + 2) % 4:
                    self.action_order.append((just_play + 3) % 4)
                    self.history_action[(just_play + 3) % 4].append([])
                    self.action_seq.append([])
                    self.action_seq_detail.append(None)
                    self.history_action_detail[(just_play + 3) % 4].append(None)
            elif (just_play + 1) % 4 in self.over and self.flag == 3:
                self.action_order.append(just_play)
                self.action_seq.append(model_action_cards)
                self.action_seq_detail.append(detail)
                self.history_action[just_play].append(model_action_cards)
                self.history_action_detail[just_play].append(detail)
                self.action_order.append((just_play + 1) % 4)
                self.history_action[(just_play + 1) % 4].append([-1])
                self.action_seq.append([-1])
                self.action_seq_detail.append(None)
                self.history_action_detail[(just_play + 1) % 4].append(None)
                self.action_order.append((just_play + 2) % 4)
                self.history_action[(just_play + 2) % 4].append([-1])
                self.action_seq.append([-1])
                self.action_seq_detail.append(None)
                self.history_action_detail[(just_play + 2) % 4].append(None)
            else:
                self.action_order.append(just_play)
                self.action_seq.append(model_action_cards)
                self.action_seq_detail.append(detail)
                self.history_action[just_play].append(model_action_cards)
                self.history_action_detail[just_play].append(detail)
        else:
            self.action_order.append(just_play)
            self.action_seq.append(model_action_cards)
            self.action_seq_detail.append(detail)
            self.history_action[just_play].append(model_action_cards)
            self.history_action_detail[just_play].append(detail)

        self.remaining[just_play] = max(0, int(self.remaining.get(just_play, 27)) - len(real_action_cards))
        if self.remaining[just_play] == 0 and just_play not in self.over:
            self.over.append(just_play)

        return

    def proc_universal(self, my_handcards: np.ndarray, rank: int):
        """提取 26 维“手牌结构/统计特征”。

        结构标志（前 12 维）：兼容原 torch 客户端的 proc_universal。

        新增统计：
        - 8 维“平均大小”：按同点数张数 g=1..8 分类统计平均大小；该类没有牌则为 0。
          （大小映射：2..A => 2..14；参谋(级牌) => 15；SB/HR => 16/17）
        - 6 维“最长连顺窗口”：单顺/双顺/三顺分别给出 (长度, 平均大小)。
          允许缺 1 张当且仅当手里有参谋（参谋视为万能补缺）。

        - rank 是 1..13（2..A），用于定位“参谋(级牌)”的位置并在统计时排除。
        """

        res = np.zeros(UNIVERSAL_DIM, dtype=np.int8)

        cur_rank = int(rank)
        if cur_rank < 1 or cur_rank > 13:
            return res

        # 是否有“参谋(级牌)”：用于 res[0] 标志位，并影响 rock_flag 的判定阈值。
        # 注意：这里沿用原实现对“级牌位置”的假设：rank 对应 (rank-1)*4 这一组。
        has_advisor = False
        try:
            has_advisor = int(my_handcards[(cur_rank - 1) * 4]) > 0
        except Exception:
            has_advisor = False

        # legacy flags (0..11)
        res[0] = 1 if has_advisor else 0

        # rock_flag：是否存在“同花色窗口 5 连”的形态（排除级牌位）。
        # - 有参谋时：允许“差一张就五连”（最多缺 1 张）
        # - 无参谋时：不考虑“差一张就五连”，仅认可“完整五连”（缺 0 张）
        allow_missing = 1 if has_advisor else 0
        rock_flag = 0
        for i in range(4):
            left, right = 0, 5
            temp = [
                int(my_handcards[i + j * 4]) if (i + j * 4) != (cur_rank - 1) * 4 else 0
                for j in range(5)
            ]
            while right <= 12:
                zero_num = temp.count(0)
                if zero_num <= allow_missing:
                    rock_flag = 1
                    break
                temp.append(int(my_handcards[i + right * 4]) if (i + right * 4) != (cur_rank - 1) * 4 else 0)
                temp.pop(0)
                left += 1
                right += 1
            if rock_flag == 1:
                break
        res[1] = rock_flag

        # 统计每个点数的张数（排除级牌位）。
        num_count = [0] * 13
        for i in range(4):
            for j in range(13):
                idx = i + j * 4
                if idx == (cur_rank - 1) * 4:
                    continue
                if int(my_handcards[idx]) != 0:
                    num_count[j] += 1

        num_max = max(num_count) if num_count else 0
        if num_max >= 6:
            res[2:8] = 1
        elif num_max == 5:
            res[3:8] = 1
        elif num_max == 4:
            res[4:8] = 1
        elif num_max == 3:
            res[5:8] = 1
        elif num_max == 2:
            res[6:8] = 1
        else:
            res[7] = 1

        temp_run = 0
        for i in range(13):
            if num_count[i] != 0:
                temp_run += 1
                if i >= 1:
                    if (num_count[i] == 2 and num_count[i - 1] >= 3) or (num_count[i] >= 3 and num_count[i - 1] == 2):
                        res[9] = 1
                    elif num_count[i] == 2 and num_count[i - 1] == 2:
                        res[11] = 1
                if i >= 2:
                    if (
                        (num_count[i - 2] == 1 and num_count[i - 1] >= 2 and num_count[i] >= 2)
                        or (num_count[i - 2] >= 2 and num_count[i - 1] == 1 and num_count[i] >= 2)
                        or (num_count[i - 2] >= 2 and num_count[i - 1] >= 2 and num_count[i] == 1)
                    ):
                        res[10] = 1
            else:
                temp_run = 0

        if temp_run >= 4:
            res[8] = 1

        # 新增统计特征
        try:
            cur_rank_idx = int(cur_rank) - 1  # 0..12

            # 统计每个点数的总张数（四花色求和；两副牌时每个花色槽位 0..2）
            totals_full = [0] * 13
            for ridx in range(13):
                base = ridx * 4
                cnt = 0
                for s in range(4):
                    cnt += int(my_handcards[base + s])
                totals_full[ridx] = int(cnt)

            sb = int(my_handcards[52]) if int(len(my_handcards)) > 52 else 0
            hr = int(my_handcards[53]) if int(len(my_handcards)) > 53 else 0

            # (1) 按同点数张数 g=1..8 分类的“平均大小”
            for g in range(1, 9):
                s_val = 0
                n_val = 0
                for ridx in range(13):
                    if int(totals_full[ridx]) != int(g):
                        continue
                    v = 15 if ridx == cur_rank_idx else int(ridx + 2)
                    s_val += int(v)
                    n_val += 1
                if int(sb) == int(g):
                    s_val += 16
                    n_val += 1
                if int(hr) == int(g):
                    s_val += 17
                    n_val += 1
                avg_g = int(round(float(s_val) / float(n_val))) if n_val > 0 else 0
                if avg_g < 0:
                    avg_g = 0
                if avg_g > 30:
                    avg_g = 30
                res[12 + (g - 1)] = np.int8(avg_g)

            # (2) 最长连顺窗口：单顺/双顺/三顺，各输出(长度, 平均大小)
            advisor_cnt = int(totals_full[cur_rank_idx]) if 0 <= cur_rank_idx < 13 else 0
            allow_missing_seq = 1 if advisor_cnt > 0 else 0

            # 连顺检测不把“参谋(级牌)”当作自身点数，参谋仅用于补缺。
            totals_no_advisor = list(totals_full)
            if 0 <= cur_rank_idx < 13:
                totals_no_advisor[cur_rank_idx] = 0

            def _best_seq_stats(k: int) -> Tuple[int, int]:
                present = [1 if int(totals_no_advisor[i]) >= int(k) else 0 for i in range(13)]
                m = int(allow_missing_seq)

                p2 = present + present
                left = 0
                zeros = 0
                best_len = 0
                best_start = 0
                for right in range(len(p2)):
                    if int(p2[right]) == 0:
                        zeros += 1
                    while zeros > m or (right - left + 1) > 13:
                        if int(p2[left]) == 0:
                            zeros -= 1
                        left += 1
                    if left >= 13:
                        break
                    cur_len = int(right - left + 1)
                    if cur_len > best_len:
                        best_len = cur_len
                        best_start = left
                    elif cur_len == best_len and cur_len > 0:
                        # tie-break：更大平均大小优先
                        s1 = 0
                        s2 = 0
                        for t in range(cur_len):
                            s1 += int((best_start + t) % 13) + 2
                            s2 += int((left + t) % 13) + 2
                        if s2 > s1:
                            best_start = left

                if best_len <= 0:
                    return 0, 0

                total_v = 0
                for t in range(best_len):
                    total_v += int((best_start + t) % 13) + 2
                avg_v = int(round(float(total_v) / float(best_len))) if best_len > 0 else 0
                if avg_v < 0:
                    avg_v = 0
                if avg_v > 30:
                    avg_v = 30
                return int(best_len), int(avg_v)

            for idx, k in enumerate((1, 2, 3)):
                l, a = _best_seq_stats(int(k))
                res[20 + idx * 2] = np.int8(min(30, max(0, int(l))))
                res[21 + idx * 2] = np.int8(min(30, max(0, int(a))))
        except Exception:
            for i in range(12, UNIVERSAL_DIM):
                res[i] = np.int8(0)

        return res

    def prepare(self, message: Dict[str, Any]) -> Dict[str, Any]:
        num_legal_actions = int(message['indexRange']) + 1

        # 候选动作：
        # - 实体牌用于 54 维（保持花色语义）
        # - virtual_ranks / virtual_cards 用于额外 13 维“点数直方图”（支持九/十炸；不编码虚拟花色）
        legal_actions_type: List[str] = []
        legal_actions_real: List[List[int]] = []
        legal_actions_detail: List[Optional[dict]] = []
        for act in message['actionList']:
            try:
                legal_actions_type.append(str(act[0]))
                cards = act[2]
                legal_actions_real.append(card2num(cards))
                if isinstance(act, list) and len(act) >= 4 and isinstance(act[3], dict):
                    legal_actions_detail.append(act[3])
                else:
                    legal_actions_detail.append(None)
            except Exception:
                legal_actions_type.append('')
                legal_actions_real.append([])
                legal_actions_detail.append(None)
        # 当前局面（采取动作前）的手牌。
        my_handcards = card2array(card2num(message['handCards']))

        # 对每个候选动作生成“采取动作后”的手牌与其派生结构特征（统计量也要同步更新）。
        cur_rank_i = RANK[str(message['curRank'])]
        my_hand_after_batch = np.repeat(my_handcards[np.newaxis, :], num_legal_actions, axis=0)
        universal_after_batch = np.zeros((num_legal_actions, UNIVERSAL_DIM), dtype=np.int8)
        for j, (t, real_cards) in enumerate(zip(legal_actions_type, legal_actions_real)):
            if str(t).upper() != 'PASS':
                played = card2array(real_cards)
                my_hand_after_batch[j, :] = np.maximum(0, my_handcards - played).astype(np.int8)
            universal_after_batch[j, :] = self.proc_universal(my_hand_after_batch[j, :], cur_rank_i)

        count_a = np.array([self.count_A])

        count_a_self = np.array([self.count_A_self])

        count_a_oppo = np.array([self.count_A_oppo])

        other_hands = []
        for i in range(54):
            if self.other_left_hands[i] == 1:
                other_hands.append(i)
            elif self.other_left_hands[i] == 2:
                other_hands.append(i)
                other_hands.append(i)
        other_handcards = card2array(other_hands)

        # 当前剩余牌库（不在自己手上、尚未被打出的牌）结构特征：用同一套算法计算。
        # 注意：为兼容旧权重，这 13 维作为“尾部追加特征”，不插入 invariant/variant 中间。
        deck_universal = self.proc_universal(other_handcards, cur_rank_i)

        if len(self.action_seq) > 0:
            last_action_54 = card2array(self.action_seq[-1])
            last_detail = None
            try:
                last_detail = self.action_seq_detail[-1] if len(self.action_seq_detail) == len(self.action_seq) else None
            except Exception:
                last_detail = None
            last_action = np.concatenate([last_action_54, _encode_virtual_rank_counts(last_detail, self.action_seq[-1])]).astype(np.int8)
        else:
            last_action = np.concatenate([card2array([-1]), np.zeros(13, dtype=np.int8)]).astype(np.int8)
        # 关键改动：x_batch 的每一行表示“做出该候选动作后的局面”。
        # - 若候选动作不是 PASS：上一手牌(last_action) 视为该候选动作；最大牌来源(greaterPos) 视为自己。
        # - 若候选动作是 PASS：上一手牌与最大牌来源保持不变。
        last_action_batch = np.zeros((num_legal_actions, 67), dtype=np.int8)
        for j, (t, real_cards, det) in enumerate(zip(legal_actions_type, legal_actions_real, legal_actions_detail)):
            if str(t).upper() == 'PASS':
                last_action_batch[j, :] = last_action
            else:
                last_action_batch[j, :] = _encode_action_feature(real_cards, det)

        if len(self.history_action[(self.mypos + 2) % 4]) > 0 and (self.mypos + 2) % 4 not in self.over:
            ta_54 = card2array(self.history_action[(self.mypos + 2) % 4][-1])
            ta_detail = None
            try:
                ta_detail = self.history_action_detail[(self.mypos + 2) % 4][-1]
            except Exception:
                ta_detail = None
            last_teammate_action = np.concatenate(
                [ta_54, _encode_virtual_rank_counts(ta_detail, self.history_action[(self.mypos + 2) % 4][-1])]
            ).astype(np.int8)
        else:
            last_teammate_action = np.concatenate([card2array([-1]), np.zeros(13, dtype=np.int8)]).astype(np.int8)

        down_num_cards_left = _get_one_hot_array(self.remaining[(self.mypos + 1) % 4], 27, 1)

        teammate_num_cards_left = _get_one_hot_array(self.remaining[(self.mypos + 2) % 4], 27, 1)

        up_num_cards_left = _get_one_hot_array(self.remaining[(self.mypos + 3) % 4], 27, 1)

        if len(self.history_action[(self.mypos + 1) % 4]) > 0:
            down_played_cards = card2array(reduce(lambda x, y: x + y, self.history_action[(self.mypos + 1) % 4]))
        else:
            down_played_cards = card2array([])

        if len(self.history_action[(self.mypos + 2) % 4]) > 0:
            teammate_played_cards = card2array(reduce(lambda x, y: x + y, self.history_action[(self.mypos + 2) % 4]))
        else:
            teammate_played_cards = card2array([])

        if len(self.history_action[(self.mypos + 3) % 4]) > 0:
            up_played_cards = card2array(reduce(lambda x, y: x + y, self.history_action[(self.mypos + 3) % 4]))
        else:
            up_played_cards = card2array([])

        # 独立的“公开明牌(known handcards)”通道：下家/对家/上家
        down_known = self.visible_handcards_by_pos.get((self.mypos + 1) % 4)
        teammate_known = self.visible_handcards_by_pos.get((self.mypos + 2) % 4)
        up_known = self.visible_handcards_by_pos.get((self.mypos + 3) % 4)
        if down_known is None:
            down_known = np.zeros(54, dtype=np.int8)
        if teammate_known is None:
            teammate_known = np.zeros(54, dtype=np.int8)
        if up_known is None:
            up_known = np.zeros(54, dtype=np.int8)
        down_known = down_known.astype(np.int8)
        teammate_known = teammate_known.astype(np.int8)
        up_known = up_known.astype(np.int8)

        self_rank = _get_one_hot_array(RANK[str(message['selfRank'])], 13, 0)
        self.rank = RANK[str(message['selfRank'])]

        oppo_rank = _get_one_hot_array(RANK[str(message['oppoRank'])], 13, 0)
        self.oppo_rank = RANK[str(message['oppoRank'])]

        cur_rank = _get_one_hot_array(RANK[str(message['curRank'])], 13, 0)

        # 当前最大牌出牌者（greaterPos）相对位置：
        # one-hot(4) 对应 [自己, 下家, 对家, 上家] + has_greater(1)
        gp = message.get('greaterPos', -1)
        rel_greater = np.zeros(5, dtype=np.float32)
        try:
            gp_i = int(gp)
        except Exception:
            gp_i = -1
        if 0 <= gp_i < 4:
            rel = (gp_i - int(self.mypos)) % 4
            rel_greater[int(rel)] = 1.0
            rel_greater[4] = 1.0

        rel_greater_i8 = rel_greater.astype(np.int8)
        rel_greater_after_batch = np.zeros((num_legal_actions, 5), dtype=np.int8)
        self_is_greater = np.array([1, 0, 0, 0, 1], dtype=np.int8)
        for j, t in enumerate(legal_actions_type):
            if str(t).upper() == 'PASS':
                rel_greater_after_batch[j, :] = rel_greater_i8
            else:
                rel_greater_after_batch[j, :] = self_is_greater

                # 贡牌目标（不区分进贡/还贡）：[上家, 队友, 下家]
                # - 非贡牌阶段：全 0
                # - 贡牌阶段：若 toPos 明确，则对应位置为 1
                # - 若目标不明确（例如“双下进贡”）：上家与下家都为 1
                tribute_target = np.zeros(3, dtype=np.int8)
                stage = str(message.get('stage', '')).lower()
                if stage in {'tribute', 'back'}:
                    up_pos = (int(self.mypos) + 3) % 4
                    teammate_pos = (int(self.mypos) + 2) % 4
                    down_pos = (int(self.mypos) + 1) % 4

                    def _mark_to(p: int) -> None:
                        if int(p) == int(up_pos):
                            tribute_target[0] = 1
                        elif int(p) == int(teammate_pos):
                            tribute_target[1] = 1
                        elif int(p) == int(down_pos):
                            tribute_target[2] = 1

                    to_pos = message.get('toPos', message.get('toSeat', message.get('targetPos')))
                    if to_pos is not None and str(to_pos) != '':
                        try:
                            _mark_to(int(to_pos))
                        except Exception:
                            pass
                    else:
                        cand = message.get('toPosCandidates', message.get('toSeatCandidates', message.get('targetCandidates')))
                        if isinstance(cand, list) and cand:
                            for p in cand:
                                try:
                                    _mark_to(int(p))
                                except Exception:
                                    continue
                        else:
                            ambiguous = bool(message.get('toPosAmbiguous', message.get('ambiguousTo', message.get('doubleDown'))))
                            if ambiguous:
                                tribute_target[0] = 1
                                tribute_target[2] = 1

        # 不可变状态块（与候选动作无关）：用于 actor 的“状态特征区”，以及 q 的公共部分。
        invariant = np.hstack(
            (
                other_handcards,
                last_teammate_action,
                down_played_cards,
                teammate_played_cards,
                up_played_cards,
                down_num_cards_left,
                teammate_num_cards_left,
                up_num_cards_left,
                self_rank,
                oppo_rank,
                cur_rank,
                down_known.astype(np.float32),
                teammate_known.astype(np.float32),
                up_known.astype(np.float32),
                count_a,
                count_a_self,
                count_a_oppo,
                tribute_target,
            )
        )
        invariant_batch = np.repeat(invariant[np.newaxis, :], num_legal_actions, axis=0).astype(np.int8)

        # 可变块（候选动作会改变的部分）：用于 actor 的“动作特征区”。
        variant_batch = np.hstack((my_hand_after_batch, universal_after_batch, last_action_batch, rel_greater_after_batch))

        deck_universal_batch = np.repeat(deck_universal[np.newaxis, :], num_legal_actions, axis=0).astype(np.int8)

        # q_network 的每行输入：invariant + variant（候选动作的后继局面）+ deck_universal(尾部)
        x_batch = np.hstack((invariant_batch, variant_batch, deck_universal_batch))
        # actor 的状态输入：只保留 invariant（不随候选动作变化）
        x_no_action = invariant.astype(np.float32, copy=False)

        return {
            'x_batch': x_batch.astype(np.int8),
            'x_no_action': x_no_action.astype(np.float32),
            'deck_universal': deck_universal.astype(np.float32, copy=False),
        }

    def build_obs_for_message(self, message: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """为 play 消息构造 (obs_vec, legal_mask, top_indexs)。

        这里复用与 act_play() 相同的特征/路径，这样离线回放可以在“不采样动作、不修改轨迹”的情况下
        重建训练轨迹。
        """

        if self.flag == 0:
            init_hand = card2num(message['handCards'])
            for ele in init_hand:
                self.other_left_hands[ele] -= 1
            self.flag = 1

        state = self.prepare(message)
        states = state['x_batch']
        state_no_action = state['x_no_action']
        deck_universal = state.get('deck_universal')
        if deck_universal is None:
            deck_universal = np.zeros((UNIVERSAL_DIM,), dtype=np.float32)

        legal_action = 2
        legal_index = np.ones(legal_action, dtype=np.float32)

        if len(states) >= legal_action:
            indexs = self.player.model_q.get_max_n_index(states, legal_action)
            dqn_states = np.asarray(states[indexs])
            # actor 的“动作特征区”：只拼接会随候选动作改变的那部分（variant）。
            top_variants = dqn_states[:, self._invariant_dim : self._invariant_dim + self._variant_dim].reshape(-1)
            obs_vec = np.concatenate((state_no_action, top_variants, deck_universal))
            return obs_vec.astype(np.float32), legal_index.astype(np.float32), list(map(int, indexs))

        legal_action = len(states)
        legal_index[legal_action:] = 0.0
        # 与 act_play() 保持相同的“槽位→索引”映射，确保一致性。
        indexs = list(range(2))
        top_indexs = self.player.model_q.get_max_n_index(states, 2)
        dqn_states = np.asarray(states[top_indexs])
        supple = np.zeros(int(self._variant_dim) * (2 - legal_action), dtype=np.float32)
        top_variants = dqn_states[:, self._invariant_dim : self._invariant_dim + self._variant_dim].reshape(-1)
        obs_prefix = np.concatenate((state_no_action, top_variants, supple))
        obs_vec = np.concatenate((obs_prefix, deck_universal))
        return obs_vec.astype(np.float32), legal_index.astype(np.float32), indexs

    def build_obs_for_message_with_candidates(
        self,
        message: Dict[str, Any],
        candidate_action_indices: List[int],
        *,
        action_number: int = 2,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """为 play 消息构造 (obs_vec, legal_mask)，但候选动作由调用方显式指定。

        用于离线回放/训练时复现“TopK候选动作特征拼接”的输入格式，同时允许
        将某个“指定动作”（例如人类真实选择）强制拼进候选集。

        - obs_vec 形状：invariant(571) + action_number * variant + deck_universal
        - legal_mask 形状：(action_number,)，无效槽位为 0。
        """

        if self.flag == 0:
            init_hand = card2num(message['handCards'])
            for ele in init_hand:
                self.other_left_hands[ele] -= 1
            self.flag = 1

        state = self.prepare(message)
        states = state['x_batch']
        state_no_action = state['x_no_action']
        deck_universal = state.get('deck_universal')
        if deck_universal is None:
            deck_universal = np.zeros((UNIVERSAL_DIM,), dtype=np.float32)

        k = int(action_number)
        if k <= 0:
            return state_no_action.astype(np.float32), np.zeros((0,), dtype=np.float32)

        legal_mask = np.zeros((k,), dtype=np.float32)
        variants: List[np.ndarray] = []

        n_actions = int(states.shape[0]) if isinstance(states, np.ndarray) and states.ndim == 2 else 0
        used: List[int] = []

        for slot in range(k):
            idx = None
            try:
                idx = int(candidate_action_indices[slot])
            except Exception:
                idx = None

            if idx is None or idx < 0 or idx >= n_actions:
                variants.append(np.zeros((int(self._variant_dim),), dtype=np.float32))
                continue

            # 避免重复候选导致 PPO 动作空间退化：第二个槽位若与前面重复，则标记为无效并置 0。
            if idx in used:
                variants.append(np.zeros((int(self._variant_dim),), dtype=np.float32))
                continue

            used.append(idx)
            legal_mask[slot] = 1.0
            v = states[idx, int(self._invariant_dim) : int(self._invariant_dim + self._variant_dim)].astype(
                np.float32, copy=False
            )
            variants.append(v)

        obs_vec = np.concatenate(
            [state_no_action.astype(np.float32, copy=False)] + variants + [deck_universal.astype(np.float32, copy=False)]
        )
        return obs_vec.astype(np.float32), legal_mask.astype(np.float32)

    def act_play(self, message: Dict[str, Any]) -> int:
        """处理 act(play) 消息并返回 actIndex。"""
        return self._act_generic(message, record=True)

    def act_tribute(self, message: Dict[str, Any]) -> int:
        """处理 act(tribute) 并返回 actIndex（进贡阶段：通常是从候选单牌中选 1 张）。"""
        return self._act_generic(message, record=False)

    def act_back(self, message: Dict[str, Any]) -> int:
        """处理 act(back) 并返回 actIndex（还贡阶段：从候选单牌中选 1 张）。"""
        return self._act_generic(message, record=False)

    def _act_generic(self, message: Dict[str, Any], *, record: bool) -> int:
        """统一的决策入口：play/tribute/back 都走同一套“q筛Top2 + policy选其一”。"""
        if self.flag == 0:
            init_hand = card2num(message['handCards'])
            for ele in init_hand:
                self.other_left_hands[ele] -= 1
            self.flag = 1

        if len(message['actionList']) == 1:
            return 0

        state = self.prepare(message)

        # 这里手写一份采样逻辑，便于记录精确的模型输入（用于训练/回放）。
        states = state['x_batch']
        legal_action = 2
        legal_index = np.ones(legal_action, dtype=np.float32)
        state_no_action = state['x_no_action']
        deck_universal = state.get('deck_universal')
        if deck_universal is None:
            deck_universal = np.zeros((UNIVERSAL_DIM,), dtype=np.float32)

        if len(states) >= legal_action:
            indexs = self.player.model_q.get_max_n_index(states, legal_action)
            dqn_states = np.asarray(states[indexs])
            top_variants = dqn_states[:, self._invariant_dim : self._invariant_dim + self._variant_dim].reshape(-1)
            obs_vec = np.concatenate((state_no_action, top_variants, deck_universal))
        else:
            legal_action = len(states)
            legal_index[legal_action:] = 0.0
            indexs = list(range(2))
            top_indexs = self.player.model_q.get_max_n_index(states, 2)
            dqn_states = np.asarray(states[top_indexs])
            supple = np.zeros(int(self._variant_dim) * (2 - legal_action), dtype=np.float32)
            top_variants = dqn_states[:, self._invariant_dim : self._invariant_dim + self._variant_dim].reshape(-1)
            obs_prefix = np.concatenate((state_no_action, top_variants, supple))
            obs_vec = np.concatenate((obs_prefix, deck_universal))

        chosen = int(self.player.model.step(obs_vec, legal_index))
        if record:
            self.trajectory.append((obs_vec.astype(np.float32), legal_index.astype(np.float32), chosen))
        act_index = int(indexs[chosen])
        try:
            xb = state.get('x_batch')
            if isinstance(xb, np.ndarray) and xb.ndim == 2 and 0 <= act_index < xb.shape[0]:
                if record:
                    self.q_trajectory.append((xb, act_index))
        except Exception:
            pass
        return act_index

    def train_q_on_trajectory(
        self,
        trajectory: List[Tuple[np.ndarray, int]],
        *,
        reward: float,
        lr: float = 5e-5,
        clear: bool = False,
        neg_k: int = 31,
    ) -> Dict[str, float]:
        """对 q_network 进行一次轻量更新（排序/打分损失）。

        将 q(s,a) 视为对合法动作列表（x_batch 的行）进行打分。
        为避免对超大动作集合做全量 softmax，这里采用负采样。

        - reward > 0：提高被选择动作的得分
        - reward < 0：降低被选择动作的得分
        """

        if not trajectory:
            return {"loss": 0.0, "n": 0}

        optimizer = torch.optim.Adam(self.player.model_q.parameters(), lr=lr)
        optimizer.zero_grad()

        losses: List[torch.Tensor] = []
        for xb, chosen_idx in trajectory:
            if not isinstance(xb, np.ndarray) or xb.ndim != 2:
                continue
            n_actions = int(xb.shape[0])
            if n_actions <= 0:
                continue
            c = int(chosen_idx)
            if c < 0 or c >= n_actions:
                continue

            if n_actions == 1:
                idx = np.array([0], dtype=np.int64)
            else:
                k = min(int(neg_k), n_actions - 1)
                # 从 [0, 动作数) 里采样负样本（排除已选动作）
                cand = np.random.choice(n_actions - 1, size=k, replace=False)
                cand = cand + (cand >= c)
                idx = np.concatenate([np.array([c], dtype=np.int64), cand.astype(np.int64)])

            x_sub = torch.tensor(xb[idx], dtype=torch.float32)
            q_sub = self.player.model_q.q(x_sub)
            logp = F.log_softmax(q_sub, dim=0)
            chosen_logp = logp[0]
            losses.append(-(float(reward) * chosen_logp))

        if not losses:
            return {"loss": 0.0, "n": 0}

        loss = torch.stack(losses).mean()
        loss.backward()
        optimizer.step()

        n = int(len(losses))
        if clear:
            trajectory.clear()
        return {"loss": float(loss.detach().cpu().item()), "n": n}

    def train_one_game(self, reward: float, lr: float = 1e-4) -> Dict[str, float]:
        """用已收集的轨迹做一次优化更新。

        按需求：每局结束后只更新一次，然后下一局立即使用更新后的权重。
        """

        return self.train_on_trajectory(self.trajectory, reward=reward, lr=lr, clear=True)

    def train_on_trajectory(
        self,
        trajectory: List[Tuple[np.ndarray, np.ndarray, int]],
        *,
        reward: float,
        lr: float = 1e-4,
        clear: bool = False,
    ) -> Dict[str, float]:
        """对外部提供的轨迹做一次训练更新。

        基础 reward 仍由外部传入（你的规则），但“回传到之前 step 的方式”采用全自动 PPO 的做法：
        仅在最后一步注入 reward，其余步 reward=0，然后用 GAE 计算 adv/ret。
        """

        if not trajectory:
            return {"loss": 0.0, "n": 0}

        optimizer = torch.optim.Adam(self.player.model.parameters(), lr=lr)
        optimizer.zero_grad()

        obs = torch.tensor(np.stack([t[0] for t in trajectory]), dtype=torch.float32)
        legal = torch.tensor(np.stack([t[1] for t in trajectory]), dtype=torch.float32)
        act = torch.tensor([t[2] for t in trajectory], dtype=torch.long)

        # 前向：策略输出 + 价值函数
        shared = self.player.model.shared(obs)
        logits = self.player.model.pi(shared)
        logits = logits - (1.0 - legal) * 1e8
        logp = F.log_softmax(logits, dim=-1)
        chosen_logp = logp.gather(1, act.view(-1, 1)).squeeze(1)

        v_pred = self.player.model.v(shared).squeeze(-1)

        # GAE（与全自动版本默认超参一致）
        gamma = 0.99
        lam = 0.95
        t = int(len(trajectory))
        rewards = torch.zeros(t, dtype=torch.float32)
        dones = torch.zeros(t, dtype=torch.float32)
        rewards[-1] = float(reward)
        dones[-1] = 1.0

        # 计算优势时对价值预测停止梯度（将其视为固定目标）
        values = torch.cat([v_pred.detach(), torch.zeros(1, dtype=torch.float32)], dim=0)
        deltas = rewards + gamma * values[1:] * (1.0 - dones) - values[:-1]
        adv = torch.zeros_like(rewards)
        lastgaelam = 0.0
        for i in reversed(range(t)):
            nextnonterminal = 1.0 - float(dones[i].item())
            lastgaelam = float(deltas[i].item()) + gamma * lam * nextnonterminal * lastgaelam
            adv[i] = lastgaelam
        ret = adv + v_pred.detach()

        # 损失
        pi_loss = -(adv.detach() * chosen_logp).mean()
        vf_loss = 0.5 * ((v_pred - ret.detach()) ** 2).mean()
        loss = pi_loss + vf_loss
        loss.backward()
        optimizer.step()

        n = int(len(trajectory))
        if clear:
            trajectory.clear()
        return {"loss": float(loss.detach().cpu().item()), "n": n}
