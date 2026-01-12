import time
from argparse import ArgumentParser
from random import randint

import numpy as np
import zmq
import pickle
import torch
import io
import sys
from pathlib import Path
from model import MLPActorCritic, MLPQNetwork


def _torch_load_bytes_cpu(b: bytes):
    bio = io.BytesIO(b)
    try:
        # PyTorch 新版本会建议显式设置 weights_only=True 以避免不安全反序列化告警。
        return torch.load(bio, map_location='cpu', weights_only=True)
    except TypeError:
        # 兼容旧版 torch（不支持 weights_only 参数）。
        return torch.load(bio, map_location='cpu')


def _dumps(obj) -> bytes:
    return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)


def _loads(data: bytes):
    return pickle.loads(data)

ActionNumber = 2

parser = ArgumentParser()
parser.add_argument('--ip', type=str, default='172.15.15.2',
                    help='IP address of learner server')
parser.add_argument('--data_port', type=int, default=5000,
                    help='Learner server port to send training data')
parser.add_argument('--param_port', type=int, default=5001,
                    help='Learner server port to subscribe model parameters')
parser.add_argument('--exp_path', type=str, default='/home/root/log',
                    help='Directory to save logging data, model parameters and config file')
parser.add_argument('--num_saved_ckpt', type=int, default=4,
                    help='Number of recent checkpoint files to be saved')
parser.add_argument('--observation_space', type=int, default=(567,),
                    help='The YAML configuration file')
parser.add_argument('--action_space', type=int, default=(5, 216),
                    help='The YAML configuration file')
parser.add_argument('--epsilon', type=float, default=0.01,
                    help='Epsilon')
parser.add_argument('--iter', type=int, default=0,
                    help='update steps for the tested model')
parser.add_argument('--seats', type=str, default='1,3',
                    help='Comma-separated seat indices to serve via ZMQ on ports 6000+seat. उदाहरण: 0 or 1,3')

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return _torch_load_bytes_cpu
        else: return super().find_class(module, name)

class Player():
    def __init__(self, args) -> None:
        base_dir = Path(__file__).resolve().parent
        # 模型初始化
        self.model_id = args.iter * 2000 + 500
        self.model = MLPActorCritic((ActionNumber, 516+ActionNumber * 54), ActionNumber)
        model_path = base_dir / 'models' / f'ppo{self.model_id}.pth'
        with model_path.open('rb') as f:
            new_weights = CPU_Unpickler(f).load()
        print('load model:', self.model_id)
        self.model.set_weights(new_weights)
        self.model_q = MLPQNetwork(567)
        q_path = base_dir / 'q_network.ckpt'
        with q_path.open('rb') as f:
            tf_weights = pickle.load(f)
        self.model_q.load_tf_weights(tf_weights)

    def sample(self, state) -> int:
        states = state['x_batch']
        legal_action = ActionNumber
        legal_index = np.ones(ActionNumber)
        state_no_action = state['x_no_action']
        if len(states) >= ActionNumber:
            indexs = self.model_q.get_max_n_index(states, ActionNumber)
            dqn_states = np.asarray(states[indexs])
            top_actions = dqn_states[:, -54:].flatten()
            states = np.concatenate((state_no_action, top_actions))

        elif len(states) < ActionNumber:
            legal_action = len(states)
            legal_index[legal_action:] = np.zeros(ActionNumber-legal_action)
            top_indexs = self.model_q.get_max_n_index(states, ActionNumber)
            dqn_states = np.asarray(states[top_indexs])
            top_actions = dqn_states[:,-54:].flatten()
            states = np.concatenate((state_no_action, top_actions)) # 把动作先添加进来
            supple = np.zeros(54 * (ActionNumber - legal_action))
            states = np.concatenate((states,supple))
            indexs = list(range(ActionNumber))

        action = self.model.step(states, legal_index)
        return indexs[action]


def _bind_rep_socket(context: zmq.Context, seat: int) -> zmq.Socket:
    socket = context.socket(zmq.REP)
    socket.linger = 0
    port = 6000 + seat
    try:
        socket.bind(f'tcp://*:{port}')
    except zmq.ZMQError as e:
        if getattr(e, 'errno', None) == zmq.EADDRINUSE or 'Address in use' in str(e):
            print(
                f"[actor] ZMQ bind failed: tcp://*:{port} already in use (seat={seat}).\n"
                f"- Likely another actor process is still running or a stale process holds the port.\n"
                f"- Fix: stop the process using the port, then restart actor."
            )
            if sys.platform.startswith('win'):
                print(
                    f"[actor] Windows check:\n"
                    f"  netstat -ano | findstr :{port}\n"
                    f"  taskkill /PID <PID> /F"
                )
            else:
                print(
                    f"[actor] Linux/macOS check:\n"
                    f"  lsof -i :{port}\n"
                    f"  kill -9 <PID>"
                )
        raise
    return socket


def run_server(seat_list, args):
    """Single-process ZMQ server for multiple seats.

    Previous implementation used multiprocessing, which can leave orphan processes
    on Windows when the parent exits abnormally (closing terminal, force-kill).
    A single-process poll loop avoids residual child processes entirely.
    """

    context = zmq.Context()
    poller = zmq.Poller()
    seat_by_socket = {}
    players = {}
    sockets = []

    try:
        for seat in seat_list:
            players[seat] = Player(args)
            sock = _bind_rep_socket(context, seat)
            sockets.append(sock)
            poller.register(sock, zmq.POLLIN)
            seat_by_socket[sock] = seat

        while True:
            events = dict(poller.poll(timeout=1000))
            for sock in list(events.keys()):
                if events.get(sock) != zmq.POLLIN:
                    continue
                seat = seat_by_socket[sock]
                state = _loads(sock.recv())
                action_index = players[seat].sample(state)
                sock.send(_dumps(action_index))
    finally:
        for sock in sockets:
            try:
                poller.unregister(sock)
            except Exception:
                pass
            try:
                sock.close(0)
            except Exception:
                pass
        try:
            context.term()
        except Exception:
            pass


def main():
    # 参数传递
    args, _ = parser.parse_known_args()

    try:
        seat_list = [int(s.strip()) for s in args.seats.split(',') if s.strip() != '']
    except ValueError:
        raise SystemExit(f"Invalid --seats '{args.seats}'. Expected comma-separated integers like '0' or '1,3'.")
    if not seat_list:
        raise SystemExit("--seats must specify at least one seat.")

    try:
        run_server(seat_list, args)
    except KeyboardInterrupt:
        return


if __name__ == '__main__':
    main()
