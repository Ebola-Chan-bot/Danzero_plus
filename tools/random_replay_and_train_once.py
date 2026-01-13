from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import re
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


# Ensure repo root is importable when running from tools/ via absolute path.
_ROOT = _repo_root()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# mini_danserver provides the authoritative legality / beating rules.
import mini_danserver as mds


def _torch_dir() -> Path:
    return _ROOT / "wintest" / "torch"


def _import_inproc_agent():
    # wintest/torch is not a package; add to sys.path for import.
    torch_dir = _torch_dir()
    sys.path.insert(0, str(torch_dir))
    from inproc_agent import TorchInProcAgent  # type: ignore
    from model import MLPQNetwork  # type: ignore

    return TorchInProcAgent, MLPQNetwork


@dataclass
class ReplayMeta:
    version: int
    created_at: str
    seed: int
    cur_rank: str
    self_rank: str
    oppo_rank: str
    start_pos: int
    python: str
    platform: str


def _action_sig(a: Any) -> Tuple[Any, ...]:
    """Stable signature for matching actions (including adviser detail if present)."""
    try:
        t = a[0]
        k = a[1]
        cards = tuple(sorted(a[2]))
        vr = None
        vc = None
        if isinstance(a, list) and len(a) >= 4 and isinstance(a[3], dict):
            vr0 = a[3].get("virtual_ranks")
            vc0 = a[3].get("virtual_cards")
            vr = tuple(vr0) if isinstance(vr0, list) else None
            vc = tuple(vc0) if isinstance(vc0, list) else None
        return (t, k, cards, vr, vc)
    except Exception:
        return (str(a),)


def _next_active(cur: int, remaining: Dict[int, int]) -> int:
    for i in range(1, 5):
        p = (cur + i) % 4
        if int(remaining.get(p, 0)) > 0:
            return p
    return cur


def _pick_next_leader(greater_pos: int, remaining: Dict[int, int]) -> int:
    if int(remaining.get(greater_pos, 0)) > 0:
        return int(greater_pos)
    teammate = (int(greater_pos) + 2) % 4
    if int(remaining.get(teammate, 0)) > 0:
        return int(teammate)
    # Fallback: pick any active
    for p in range(4):
        if int(remaining.get(p, 0)) > 0:
            return int(p)
    return int(greater_pos)


def _compute_training_events(finish_order: List[int]) -> List[Tuple[int, float]]:
    """Port of mini_danserver's in-proc reward shaping for one game."""
    if not finish_order:
        return []

    o = [int(x) for x in finish_order]
    enemies = {1, 3}

    if o[0] == 0:
        return [(0, 2.0)]

    if o[0] == 2:
        if len(o) >= 2 and o[1] == 0:
            return [(0, 3.0), (2, 3.0)]

        if len(o) >= 3 and o[2] == 0 and o[1] in enemies:
            return [(0, 2.0), (2, 2.0), (1, -2.0), (3, -2.0)]

        if len(o) >= 3 and 0 not in o and 1 in o and 3 in o:
            return [(0, 1.0), (2, 1.0), (1, -1.0), (3, -1.0)]

        return []

    if o[0] in enemies:
        first_enemy = o[0]
        other_enemy = 3 if first_enemy == 1 else 1

        if len(o) >= 2 and o[1] == 0:
            return [(first_enemy, 1.5), (0, -1.5)]

        if len(o) >= 2 and o[1] == other_enemy:
            return [(first_enemy, 3.0), (other_enemy, 3.0), (0, -3.0), (2, -3.0)]

        if len(o) >= 3 and o[1] == 2 and o[2] == 0:
            return [(first_enemy, 1.0), (other_enemy, 1.0), (2, -1.0), (0, -1.0)]

        if len(o) >= 3 and o[1] == 2 and o[2] == other_enemy and 0 not in o:
            return [(first_enemy, 2.0), (other_enemy, 2.0), (0, -2.0), (2, -2.0)]

        return []

    return []


def _find_latest_ppo_model_id(models_dir: Path) -> int:
    """Return the largest numeric id from ppo*.pth, defaulting to 500."""
    best = 500
    for p in models_dir.glob("ppo*.pth"):
        m = re.match(r"^ppo(\d+)\.pth$", p.name)
        if not m:
            continue
        try:
            mid = int(m.group(1))
        except Exception:
            continue
        best = max(best, mid)
    return best


def _model_iter_from_model_id(model_id: int) -> int:
    # inproc_agent uses: model_id = iter*2000 + 500
    if model_id < 500:
        return 0
    delta = model_id - 500
    if delta % 2000 != 0:
        return 0
    return int(delta // 2000)


def _export_q_to_tf_ckpt_list(model_q) -> List[np.ndarray]:
    """Export torch q_network weights to the legacy q_network.ckpt format (pickle list).

    That ckpt list is what wintest/torch/actor.py expects.
    """
    sd = model_q.state_dict()
    names = [
        "q.q_net.0.weight",
        "q.q_net.0.bias",
        "q.q_net.2.weight",
        "q.q_net.2.bias",
        "q.q_net.4.weight",
        "q.q_net.4.bias",
        "q.q_net.6.weight",
        "q.q_net.6.bias",
        "q.q_net.8.weight",
        "q.q_net.8.bias",
        "q.q_net.10.weight",
        "q.q_net.10.bias",
    ]

    out: List[np.ndarray] = []
    for n in names:
        w = sd[n].detach().cpu().numpy()
        if n.endswith("weight"):
            out.append(w.T.copy())
        else:
            out.append(w.copy())
    return out


def simulate_random_game(
    *,
    seed: int,
    cur_rank: str,
    self_rank: str,
    oppo_rank: str,
    start_pos: int,
) -> Dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)

    deck = mds.make_double_deck()
    random.shuffle(deck)
    # IMPORTANT: store initial hands explicitly for replay (do not infer later).
    hands: Dict[int, List[mds.Card]] = {
        0: list(deck[:27]),
        1: list(deck[27:54]),
        2: list(deck[54:81]),
        3: list(deck[81:108]),
    }
    initial_hands: Dict[int, List[mds.Card]] = {k: list(v) for k, v in hands.items()}

    remaining = {i: 27 for i in range(4)}
    finish_order: List[int] = []
    played_by_seat: Dict[int, List[mds.Card]] = {i: [] for i in range(4)}

    steps: List[Dict[str, Any]] = []
    play_history: List[Tuple[int, mds.Action]] = []

    cur_pos = int(start_pos)
    trick_id = 0
    greater_action: Optional[mds.Action] = None
    greater_pos: Optional[int] = None
    passes_since_play = 0
    turn = 0

    def team_done() -> bool:
        # Game ends once either team has both players finished.
        a_done = remaining[0] <= 0 and remaining[2] <= 0
        b_done = remaining[1] <= 0 and remaining[3] <= 0
        return bool(a_done or b_done)

    while not team_done():
        if remaining[cur_pos] <= 0:
            cur_pos = _next_active(cur_pos, remaining)
            continue

        hand = hands[cur_pos]
        all_actions = mds.generate_actions_from_hand(hand, cur_rank)

        if greater_action is None:
            # leader: must play something if possible
            action_list = all_actions if all_actions else [["PASS", "PASS", []]]
        else:
            filtered = [a for a in all_actions if mds.beats(a, greater_action, cur_rank)]
            action_list = [["PASS", "PASS", []]] + filtered
            if not filtered:
                action_list = [["PASS", "PASS", []]]

        chosen = random.choice(action_list)
        play_history.append((cur_pos, chosen))

        before_n = int(len(hand))
        if chosen[0] != "PASS":
            for c in list(chosen[2]):
                # remove actual cards (including advisers)
                try:
                    hand.remove(c)
                except Exception:
                    # Should not happen; keep simulation going.
                    pass
            played_by_seat[cur_pos].extend(list(chosen[2]))
            remaining[cur_pos] = max(0, remaining[cur_pos] - len(chosen[2]))

        after_n = int(len(hand))
        if remaining[cur_pos] <= 0 and cur_pos not in finish_order:
            finish_order.append(cur_pos)

        step = {
            "turn": turn,
            "trick_id": trick_id,
            "pos": cur_pos,
            "greater_pos": (greater_pos if greater_pos is not None else -1),
            "greater_action": (greater_action if greater_action is not None else ["PASS", "PASS", []]),
            "action": chosen,
            "action_human": mds.format_action_for_human(chosen, cur_rank),
            "hand_count_before": before_n,
            "hand_count_after": after_n,
            "remaining": {str(k): int(v) for k, v in remaining.items()},
        }
        steps.append(step)

        # Trick bookkeeping
        if chosen[0] == "PASS":
            passes_since_play += 1
        else:
            greater_action = chosen
            greater_pos = cur_pos
            passes_since_play = 0

        active_after = [p for p in range(4) if int(remaining[p]) > 0]
        leader_active = False
        if greater_pos is not None:
            leader_active = int(remaining.get(int(greater_pos), 0)) > 0
        required_passes = max(0, len(active_after) - (1 if leader_active else 0))
        if greater_action is not None and passes_since_play >= required_passes:
            # end trick
            trick_id += 1
            if greater_pos is None:
                cur_pos = _next_active(cur_pos, remaining)
            else:
                cur_pos = _pick_next_leader(int(greater_pos), remaining)
            greater_action = None
            greater_pos = None
            passes_since_play = 0
            turn += 1
            continue

        cur_pos = _next_active(cur_pos, remaining)
        turn += 1

        # Safety stop (should never hit)
        if turn > 2000:
            raise RuntimeError("simulation exceeded 2000 turns; aborting")

    return {
        # Canonical: initial hands for each seat (do not infer later).
        "initial_hands": {str(k): list(v) for k, v in initial_hands.items()},
        # Debug: remaining cards in hand at end of game.
        "hands_remaining": {str(k): list(v) for k, v in hands.items()},
        "played_by_seat": {str(k): list(v) for k, v in played_by_seat.items()},
        "finish_order": list(map(int, finish_order)),
        "play_history": [(int(p), a) for p, a in play_history],
        "steps": steps,
    }


def build_trajectories_from_replay(
    *,
    replay: Dict[str, Any],
    model_iter: int,
    cur_rank: str,
    self_rank: str,
    oppo_rank: str,
) -> Tuple[Dict[int, List[Tuple[np.ndarray, np.ndarray, int]]], Dict[int, List[Tuple[np.ndarray, int]]]]:
    TorchInProcAgent, _ = _import_inproc_agent()

    # Prefer explicit initial_hands; fall back to hands for older files.
    hands_src = replay.get("initial_hands") or replay.get("hands")
    if not isinstance(hands_src, dict):
        raise TypeError("replay file missing initial_hands/hands")
    hands = {int(k): list(v) for k, v in hands_src.items()}
    play_history = [(int(p), a) for (p, a) in replay["play_history"]]

    def build_for_seat(seat: int):
        agent = TorchInProcAgent(seat=seat, model_iter=model_iter)
        agent.on_beginning(curRank=str(cur_rank), selfRank=str(self_rank), oppoRank=str(oppo_rank))

        my_hand = list(hands[seat])
        remaining = {0: 27, 1: 27, 2: 27, 3: 27}
        greater_action: Optional[Any] = None
        greater_pos: Optional[int] = None
        passes_since_play = 0

        ppo_traj: List[Tuple[np.ndarray, np.ndarray, int]] = []
        q_traj: List[Tuple[np.ndarray, int]] = []

        for pos, act in play_history:
            pos = int(pos)

            if pos == seat:
                all_actions = mds.generate_actions_from_hand(my_hand, str(cur_rank))
                if greater_action is None:
                    action_list = all_actions
                else:
                    filtered = [a for a in all_actions if mds.beats(a, greater_action, str(cur_rank))]
                    action_list = [["PASS", "PASS", []]] + filtered
                if not action_list:
                    action_list = [["PASS", "PASS", []]]

                act_sig = _action_sig(act)
                chosen_idx: Optional[int] = None
                for i, cand in enumerate(action_list):
                    if _action_sig(cand) == act_sig:
                        chosen_idx = i
                        break

                msg = {
                    "type": "act",
                    "stage": "play",
                    "handCards": my_hand,
                    "actionList": action_list,
                    "indexRange": len(action_list) - 1,
                    "curRank": str(cur_rank),
                    "selfRank": str(self_rank),
                    "oppoRank": str(oppo_rank),
                    "curPos": pos,
                    "curAction": None,
                    "greaterPos": (greater_pos if greater_pos is not None else -1),
                    "greaterAction": (greater_action if greater_action is not None else ["PASS", "PASS", []]),
                }

                if chosen_idx is not None:
                    try:
                        obs_vec, legal_mask, top_indexs = agent.build_obs_for_message(msg)
                        if int(chosen_idx) in list(top_indexs):
                            chosen = int(list(top_indexs).index(int(chosen_idx)))
                            ppo_traj.append((obs_vec, legal_mask, chosen))
                    except Exception:
                        pass

                    try:
                        st = agent.prepare(msg)
                        xb = st.get("x_batch") if isinstance(st, dict) else None
                        if isinstance(xb, np.ndarray) and xb.ndim == 2 and 0 <= int(chosen_idx) < int(xb.shape[0]):
                            q_traj.append((xb, int(chosen_idx)))
                    except Exception:
                        pass

            # advance agent state
            try:
                agent.on_notify_play(pos, act)
            except Exception:
                pass

            # update hand/remaining for trick logic rebuild
            if pos == seat and isinstance(act, list) and len(act) >= 3 and act[0] != "PASS":
                for c in list(act[2]):
                    try:
                        my_hand.remove(c)
                    except Exception:
                        pass

            if isinstance(act, list) and len(act) >= 3 and act[0] != "PASS":
                remaining[pos] = max(0, int(remaining.get(pos, 27)) - len(act[2]))

            if isinstance(act, list) and len(act) >= 1 and act[0] == "PASS":
                passes_since_play += 1
            else:
                greater_action = act
                greater_pos = pos
                passes_since_play = 0

            active_after = [p for p in range(4) if int(remaining.get(p, 27)) > 0]
            leader_active = False
            try:
                if greater_pos is not None:
                    leader_active = int(remaining.get(int(greater_pos), 27)) > 0
            except Exception:
                leader_active = False
            required_passes = max(0, len(active_after) - (1 if leader_active else 0))
            if greater_action is not None and passes_since_play >= required_passes:
                greater_action = None
                greater_pos = None
                passes_since_play = 0

        return ppo_traj, q_traj

    ppo_by_seat: Dict[int, List[Tuple[np.ndarray, np.ndarray, int]]] = {}
    q_by_seat: Dict[int, List[Tuple[np.ndarray, int]]] = {}
    for s in range(4):
        ppo_traj, q_traj = build_for_seat(s)
        ppo_by_seat[s] = ppo_traj
        q_by_seat[s] = q_traj
    return ppo_by_seat, q_by_seat


def train_from_replay_once(
    *,
    replay: Dict[str, Any],
    cur_rank: str,
    self_rank: str,
    oppo_rank: str,
    model_id: int,
    inplace: bool,
) -> Dict[str, Any]:
    TorchInProcAgent, _ = _import_inproc_agent()

    models_dir = _torch_dir() / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    q_ckpt_path = _torch_dir() / "q_network.ckpt"
    ppo_path = models_dir / f"ppo{model_id}.pth"

    model_iter = _model_iter_from_model_id(model_id)

    ppo_by_seat, q_by_seat = build_trajectories_from_replay(
        replay=replay,
        model_iter=model_iter,
        cur_rank=cur_rank,
        self_rank=self_rank,
        oppo_rank=oppo_rank,
    )

    trainer = TorchInProcAgent(seat=0, model_iter=model_iter)

    finish_order = list(map(int, replay.get("finish_order", [])))
    events = _compute_training_events(finish_order)

    ppo_total_n = 0
    ppo_last_loss = 0.0
    for seat, rew in events:
        traj = ppo_by_seat.get(int(seat), [])
        if not traj:
            continue
        info = trainer.train_on_trajectory(traj, reward=float(rew), lr=1e-4, clear=False)
        ppo_total_n += int(info.get("n", 0))
        ppo_last_loss = float(info.get("loss", 0.0))

    q_total_n = 0
    q_last_loss = 0.0
    for seat, rew in events:
        if float(rew) == 0.0:
            continue
        q_traj = q_by_seat.get(int(seat), [])
        if not q_traj:
            continue
        q_info = trainer.train_q_on_trajectory(q_traj, reward=float(rew), lr=5e-5, clear=False)
        q_total_n += int(q_info.get("n", 0))
        q_last_loss = float(q_info.get("loss", 0.0))

    # Save weights (with backups)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_dir = models_dir / "backup" / stamp
    backup_dir.mkdir(parents=True, exist_ok=True)

    if ppo_path.exists():
        shutil.copy2(ppo_path, backup_dir / ppo_path.name)
    if q_ckpt_path.exists():
        shutil.copy2(q_ckpt_path, backup_dir / q_ckpt_path.name)

    # actor expects pickle stream for ppo*.pth
    new_actor_weights = trainer.player.model.get_weights()
    if inplace:
        out_ppo = ppo_path
    else:
        out_ppo = models_dir / f"ppo{model_id}_{stamp}.pth"
    with open(out_ppo, "wb") as f:
        pickle.dump(new_actor_weights, f, protocol=pickle.HIGHEST_PROTOCOL)

    # q_network.ckpt expects a pickle list of numpy arrays (legacy TF format)
    q_list = _export_q_to_tf_ckpt_list(trainer.player.model_q)
    if inplace:
        out_q = q_ckpt_path
    else:
        out_q = _torch_dir() / f"q_network_{stamp}.ckpt"
    with open(out_q, "wb") as f:
        pickle.dump(q_list, f, protocol=pickle.HIGHEST_PROTOCOL)

    return {
        "events": events,
        "ppo": {"n": ppo_total_n, "loss": ppo_last_loss, "saved": str(out_ppo), "backup_dir": str(backup_dir)},
        "dqn": {"n": q_total_n, "loss": q_last_loss, "saved": str(out_q), "backup_dir": str(backup_dir)},
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Generate a random legal GuanDan game replay (JSON) and then train PPO+DQN once from that replay. "
            "Uses mini_danserver rules and wintest/torch inproc training helpers."
        )
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cur-rank", type=str, default="2")
    p.add_argument("--self-rank", type=str, default="2")
    p.add_argument("--oppo-rank", type=str, default="2")
    p.add_argument("--start-pos", type=int, default=0)
    p.add_argument(
        "--replay-out",
        type=str,
        default="",
        help="Output path for replay json (default: replays/replay-<timestamp>.json)",
    )
    p.add_argument(
        "--train-only",
        type=str,
        default="",
        help="If set, skip generation and only train from this replay json.",
    )
    p.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite latest ppo*.pth and q_network.ckpt (creates backups). Default: false (save new files).",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    torch_models_dir = _torch_dir() / "models"
    torch_models_dir.mkdir(parents=True, exist_ok=True)
    latest_model_id = _find_latest_ppo_model_id(torch_models_dir)

    if args.train_only:
        replay_path = Path(args.train_only)
        with replay_path.open("r", encoding="utf-8") as f:
            doc = json.load(f)
        meta = doc.get("meta", {})
        cur_rank = str(meta.get("cur_rank", args.cur_rank))
        self_rank = str(meta.get("self_rank", args.self_rank))
        oppo_rank = str(meta.get("oppo_rank", args.oppo_rank))

        result = train_from_replay_once(
            replay=doc["replay"],
            cur_rank=cur_rank,
            self_rank=self_rank,
            oppo_rank=oppo_rank,
            model_id=latest_model_id,
            inplace=bool(args.inplace),
        )
        doc.setdefault("training", {})
        doc["training"]["run_at"] = datetime.now().isoformat(timespec="seconds")
        doc["training"]["result"] = result
        with replay_path.open("w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)
        print(f"trained_once=1 replay={str(replay_path)}")
        print(result)
        return

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.replay_out:
        replay_path = Path(args.replay_out)
    else:
        replay_path = _repo_root() / "replays" / f"replay-{stamp}.json"
    replay_path.parent.mkdir(parents=True, exist_ok=True)

    meta = ReplayMeta(
        version=1,
        created_at=datetime.now().isoformat(timespec="seconds"),
        seed=int(args.seed),
        cur_rank=str(args.cur_rank),
        self_rank=str(args.self_rank),
        oppo_rank=str(args.oppo_rank),
        start_pos=int(args.start_pos),
        python=sys.version.replace("\n", " "),
        platform=sys.platform,
    )

    replay = simulate_random_game(
        seed=int(args.seed),
        cur_rank=str(args.cur_rank),
        self_rank=str(args.self_rank),
        oppo_rank=str(args.oppo_rank),
        start_pos=int(args.start_pos),
    )

    # 1) Write replay first (human+machine readable JSON)
    doc: Dict[str, Any] = {
        "meta": asdict(meta),
        "summary": {
            "finish_order": replay.get("finish_order"),
            "latest_model_id_used": int(latest_model_id),
        },
        "replay": replay,
    }

    with replay_path.open("w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)

    # 2) Read it back, then train once using recorded game
    with replay_path.open("r", encoding="utf-8") as f:
        loaded = json.load(f)

    result = train_from_replay_once(
        replay=loaded["replay"],
        cur_rank=str(args.cur_rank),
        self_rank=str(args.self_rank),
        oppo_rank=str(args.oppo_rank),
        model_id=latest_model_id,
        inplace=bool(args.inplace),
    )

    loaded.setdefault("training", {})
    loaded["training"]["run_at"] = datetime.now().isoformat(timespec="seconds")
    loaded["training"]["result"] = result
    loaded.setdefault("summary", {})
    loaded["summary"]["events"] = result.get("events")

    with replay_path.open("w", encoding="utf-8") as f:
        json.dump(loaded, f, ensure_ascii=False, indent=2)

    print(f"replay_written={str(replay_path)}")
    print(result)


if __name__ == "__main__":
    main()
