import argparse
import os
import pickle

import torch

from model import MLPActorCritic


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Generate a random-initialized PPO policy checkpoint (ppo*.pth) "
            "compatible with wintest/torch/actor.py."
        )
    )
    p.add_argument(
        "--model-id",
        type=int,
        default=500,
        help="Model id used in filename: ppo{model_id}.pth (default: 500)",
    )
    p.add_argument(
        "--action-number",
        type=int,
        default=2,
        help="Top-K candidate actions used by the PPO head (default: 2)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for torch initialization (default: 0)",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "models"),
        help="Output directory for ppo*.pth (default: wintest/torch/models)",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    torch.manual_seed(args.seed)

    # Keep the architecture identical to what wintest/torch/actor.py constructs:
    # obs dim = 516 + action_number * 54
    obs_dim = 516 + args.action_number * 54
    model = MLPActorCritic((args.action_number, obs_dim), args.action_number)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"ppo{args.model_id}.pth")

    # actor.py loads via pickle.Unpickler, so we store a plain state_dict.
    with open(out_path, "wb") as f:
        pickle.dump(model.get_weights(), f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Wrote random PPO weights to: {out_path}")


if __name__ == "__main__":
    main()
