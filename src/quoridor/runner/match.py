import time
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from quoridor.bots.registry import get_bot
from quoridor.runner.game_runner import run_game, GameResult


class MatchResult:
    __slots__ = ("bot1_name", "bot2_name", "games", "bot1_wins", "bot2_wins",
                 "draws", "avg_moves", "avg_move_time1", "avg_move_time2",
                 "worst_move_time1", "worst_move_time2")

    def __init__(self):
        self.bot1_name = ""
        self.bot2_name = ""
        self.games = 0
        self.bot1_wins = 0
        self.bot2_wins = 0
        self.draws = 0
        self.avg_moves = 0.0
        self.avg_move_time1 = 0.0
        self.avg_move_time2 = 0.0
        self.worst_move_time1 = 0.0
        self.worst_move_time2 = 0.0


def _run_single_game(args):
    bot1_name, bot1_params, bot2_name, bot2_params, seed, max_moves = args
    bot1 = get_bot(bot1_name, **bot1_params)
    bot2 = get_bot(bot2_name, **bot2_params)
    return run_game(bot1, bot2, seed=seed, max_moves=max_moves)


def run_match(
    bot1_name: str,
    bot2_name: str,
    games: int = 100,
    swap_colors: bool = True,
    parallel: bool = True,
    max_workers: int = None,
    max_moves: int = 400,
    bot1_params: dict = None,
    bot2_params: dict = None,
    base_seed: int = 42,
) -> MatchResult:
    bot1_params = bot1_params or {}
    bot2_params = bot2_params or {}
    result = MatchResult()
    result.bot1_name = bot1_name
    result.bot2_name = bot2_name

    game_args = []
    for i in range(games):
        seed = base_seed + i
        if swap_colors and i % 2 == 1:
            game_args.append((bot2_name, bot2_params, bot1_name, bot1_params, seed, max_moves))
        else:
            game_args.append((bot1_name, bot1_params, bot2_name, bot2_params, seed, max_moves))

    all_results: list[GameResult] = []
    total_move_times1 = []
    total_move_times2 = []
    total_moves = 0

    if parallel and games > 1:
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(_run_single_game, a) for a in game_args]
                for future in as_completed(futures):
                    gr = future.result()
                    all_results.append(gr)
        except Exception:
            parallel = False

    if not parallel or games <= 1:
        for args in game_args:
            gr = _run_single_game(args)
            all_results.append(gr)

    for i, gr in enumerate(all_results):
        swapped = swap_colors and (i % 2 == 1)
        if gr.winner == 0:
            if swapped:
                result.bot2_wins += 1
            else:
                result.bot1_wins += 1
        elif gr.winner == 1:
            if swapped:
                result.bot1_wins += 1
            else:
                result.bot2_wins += 1
        else:
            result.draws += 1
        total_moves += gr.moves

        if swapped:
            p1_times = gr.move_times[1::2]
            p2_times = gr.move_times[0::2]
        else:
            p1_times = gr.move_times[0::2]
            p2_times = gr.move_times[1::2]
        total_move_times1.extend(p1_times)
        total_move_times2.extend(p2_times)

    result.games = len(all_results)
    if result.games > 0:
        result.avg_moves = total_moves / result.games
    if total_move_times1:
        result.avg_move_time1 = sum(total_move_times1) / len(total_move_times1)
        result.worst_move_time1 = max(total_move_times1)
    if total_move_times2:
        result.avg_move_time2 = sum(total_move_times2) / len(total_move_times2)
        result.worst_move_time2 = max(total_move_times2)

    return result
