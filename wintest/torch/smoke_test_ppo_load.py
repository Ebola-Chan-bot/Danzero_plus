import argparse
import os
import pickle

import numpy as np
import torch

from model import MLPActorCritic


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Smoke test loading ppo*.pth and running one forward step")
    p.add_argument(
        "--model-path",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "models", "ppo500.pth"),
        help="Path to ppo*.pth produced by make_random_ppo_model.py",
    )
    p.add_argument("--action-number", type=int, default=2)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    obs_dim = 516 + args.action_number * 54
    model = MLPActorCritic((args.action_number, obs_dim), args.action_number)

    with open(args.model_path, "rb") as f:
        weights = pickle.load(f)
    model.set_weights(weights)

    # Dummy input shaped like actor.py feeds:
    # states = concat(x_no_action(516), top_actions(action_number*54)) => (obs_dim,)
    states = np.zeros((obs_dim,), dtype=np.float32)
    legal_index = np.ones((args.action_number,), dtype=np.float32)

    with torch.no_grad():
        action = model.step(states, legal_index)

    print("loaded_ok=1", "action=", int(action))
    print("torch_cuda_available=", torch.cuda.is_available())


if __name__ == "__main__":
    main()
