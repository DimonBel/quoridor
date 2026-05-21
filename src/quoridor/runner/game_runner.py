import json
import os
import time
from quoridor.core.state import GameState
from quoridor.core.moves import Move, MoveType
from quoridor.bots.base import Bot


class GameResult:
    __slots__ = ("winner", "moves", "move_history", "bot_names", "move_times", "seed")

    def __init__(self):
        self.winner: int = -1
        self.moves: int = 0
        self.move_history: list[str] = []
        self.bot_names: tuple[str, str] = ("", "")
        self.move_times: list[float] = []
        self.seed: int = 0


def _move_to_str(move: Move) -> str:
    if move.move_type == MoveType.PAWN:
        return f"P{move.row},{move.col}"
    elif move.move_type == MoveType.WALL_H:
        return f"WH{move.row},{move.col}"
    else:
        return f"WV{move.row},{move.col}"


def _str_to_move(s: str) -> Move:
    from quoridor.core.moves import pawn_move, wall_h, wall_v
    if s.startswith("P"):
        r, c = s[1:].split(",")
        return pawn_move(int(r), int(c))
    elif s.startswith("WH"):
        r, c = s[2:].split(",")
        return wall_h(int(r), int(c))
    else:
        r, c = s[2:].split(",")
        return wall_v(int(r), int(c))


def run_game(
    bot1: Bot,
    bot2: Bot,
    seed: int = 0,
    max_moves: int = 400,
    callback=None,
) -> GameResult:
    state = GameState(max_moves=max_moves)
    bots = [bot1, bot2]
    result = GameResult()
    result.bot_names = (bot1.name, bot2.name)
    result.seed = seed

    while not state.is_over():
        current = state.current_player
        bot = bots[current]
        t0 = time.perf_counter()
        move = bot.choose_move(state)
        t1 = time.perf_counter()
        elapsed = t1 - t0
        result.move_times.append(elapsed)
        state.make_move(move)
        result.move_history.append(_move_to_str(move))
        result.moves += 1
        if callback:
            callback(state, move, current, elapsed)

    result.winner = state.winner()
    return result


def save_game(result: GameResult, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = {
        "bot1": result.bot_names[0],
        "bot2": result.bot_names[1],
        "seed": result.seed,
        "winner": result.winner,
        "moves": result.moves,
        "move_times": result.move_times,
        "history": result.move_history,
    }
    with open(path, "w") as f:
        json.dump(data, f)


def load_game(path: str) -> dict:
    with open(path) as f:
        return json.load(f)
