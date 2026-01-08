import json
from argparse import ArgumentParser
from queue import Queue
from threading import Event, Lock, Thread
from typing import Any

from ws4py.client.threadedclient import WebSocketClient


def _format_action(action: Any) -> str:
    """Best-effort formatter for an entry in message['actionList'].

    Most clients in this repo treat an action as a 3-tuple/list like:
      [actionType, rankOrKey, cards]
    but we keep this tolerant.
    """
    if isinstance(action, (list, tuple)):
        if len(action) >= 3:
            return f"{action[0]} {action[1]} {action[2]}"
        return str(list(action))
    return str(action)


class ManualClient(WebSocketClient):
    def __init__(self, url: str, seat: int, prompt_queue: "Queue[tuple[int, list]]", response_queue: "Queue[int]",
                 print_lock: Lock):
        super().__init__(url)
        self.seat = seat
        self._prompt_queue = prompt_queue
        self._response_queue = response_queue
        self._print_lock = print_lock
        self.my_pos = None

    def opened(self):
        with self._print_lock:
            print(f"[manual] connected seat={self.seat} url={self.url}")

    def closed(self, code, reason=None):
        with self._print_lock:
            print("Closed down", code, reason)

    def received_message(self, message):
        msg = json.loads(str(message))

        # Useful one-liners so you can see game progress.
        stage = msg.get("stage")
        if stage == "beginning":
            self.my_pos = msg.get("myPos")
            with self._print_lock:
                print(f"[manual] game beginning, myPos={self.my_pos}, curRank={msg.get('curRank')}")

        if stage == "play" and "curPos" in msg:
            cur_pos = msg.get("curPos")
            cur_action = msg.get("curAction")
            with self._print_lock:
                print(f"[table] pos{cur_pos} played: {cur_action}")

        # Decision point.
        if "actionList" not in msg:
            return

        action_list = msg["actionList"]
        # Hand off the prompt to the main thread so multiple seats can share one terminal.
        self._prompt_queue.put((self.seat, action_list))
        idx = self._response_queue.get()
        self.send(json.dumps({"actIndex": idx}))


def _parse_seats(text: str) -> list[int]:
    try:
        seats = [int(s.strip()) for s in text.split(',') if s.strip() != '']
    except ValueError:
        raise SystemExit(f"Invalid --seats '{text}'. Expected comma-separated integers like '1,2,3'.")
    for s in seats:
        if s not in (0, 1, 2, 3):
            raise SystemExit("--seats must be a subset of 0,1,2,3.")
    if len(set(seats)) != len(seats):
        raise SystemExit("--seats contains duplicates.")
    return seats


def _run_ws_client(ws: ManualClient, stop_event: Event, print_lock: Lock):
    try:
        ws.connect()
        ws.run_forever()
    except Exception as e:
        with print_lock:
            print(f"[manual] seat={ws.seat} websocket error: {e}")
    finally:
        stop_event.set()


def main():
    parser = ArgumentParser(description="Manual GuanDan websocket client (choose actIndex by typing).")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=23456)
    parser.add_argument("--seat", type=int, choices=[0, 1, 2, 3], required=True,
                        help="Seat/client id to join: 0..3 maps to /game/client0..3")
    parser.add_argument("--seats", type=str, default="",
                        help="Comma-separated seats to control in ONE terminal (overrides --seat). Example: 1,2,3")
    args = parser.parse_args()

    seats = _parse_seats(args.seats) if args.seats else [args.seat]

    print_lock = Lock()
    stop_event = Event()
    prompt_queue: "Queue[tuple[int, list]]" = Queue()
    response_queues: dict[int, "Queue[int]"] = {s: Queue() for s in seats}

    clients: dict[int, ManualClient] = {}
    threads: list[Thread] = []
    for seat in seats:
        url = f"ws://{args.host}:{args.port}/game/client{seat}"
        ws = ManualClient(url, seat=seat, prompt_queue=prompt_queue, response_queue=response_queues[seat],
                          print_lock=print_lock)
        clients[seat] = ws
        t = Thread(target=_run_ws_client, args=(ws, stop_event, print_lock), daemon=True)
        t.start()
        threads.append(t)

    with print_lock:
        if len(seats) == 1:
            print(f"[manual] controlling seat {seats[0]}")
        else:
            print(f"[manual] controlling seats {seats} in ONE terminal")

    try:
        while not stop_event.is_set():
            seat, action_list = prompt_queue.get()
            with print_lock:
                print(f"\n[manual] seat {seat} turn. legal actions:")
                for i, a in enumerate(action_list):
                    print(f"  {i:>3}: {_format_action(a)}")

            while True:
                raw = input(f"seat {seat} choose actIndex (empty=0): ").strip()
                if raw == "":
                    idx = 0
                    break
                try:
                    idx = int(raw)
                except ValueError:
                    with print_lock:
                        print("Please enter an integer.")
                    continue
                if 0 <= idx < len(action_list):
                    break
                with print_lock:
                    print(f"Out of range. Must be 0..{len(action_list)-1}.")

            response_queues[seat].put(idx)
    except KeyboardInterrupt:
        pass
    finally:
        for ws in clients.values():
            try:
                ws.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
