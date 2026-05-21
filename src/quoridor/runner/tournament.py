import json
import os
from quoridor.runner.match import run_match


class TournamentResult:
    __slots__ = ("bot_names", "results", "elo_ratings")

    def __init__(self, bot_names: list[str]):
        self.bot_names = bot_names
        self.results: dict[tuple[str, str], dict] = {}
        self.elo_ratings: dict[str, float] = {name: 1000.0 for name in bot_names}


def _update_elo(ratings: dict[str, float], bot1: str, bot2: str, winner: int, k: float = 32.0):
    r1 = ratings[bot1]
    r2 = ratings[bot2]
    e1 = 1.0 / (1.0 + 10 ** ((r2 - r1) / 400.0))
    e2 = 1.0 - e1
    if winner == 0:
        s1, s2 = 1.0, 0.0
    elif winner == 1:
        s1, s2 = 0.0, 1.0
    else:
        s1, s2 = 0.5, 0.5
    ratings[bot1] = r1 + k * (s1 - e1)
    ratings[bot2] = r2 + k * (s2 - e2)


def run_tournament(
    bot_names: list[str],
    bot_params: dict[str, dict] = None,
    games_per_pair: int = 200,
    max_moves: int = 400,
    base_seed: int = 42,
) -> TournamentResult:
    bot_params = bot_params or {}
    result = TournamentResult(bot_names)

    for i, b1 in enumerate(bot_names):
        for j, b2 in enumerate(bot_names):
            if i >= j:
                continue
            p1 = bot_params.get(b1, {})
            p2 = bot_params.get(b2, {})
            mr = run_match(
                b1, b2,
                games=games_per_pair,
                swap_colors=True,
                parallel=False,
                max_moves=max_moves,
                bot1_params=p1,
                bot2_params=p2,
                base_seed=base_seed + i * 100 + j,
            )
            pair_key = (b1, b2)
            result.results[pair_key] = {
                "bot1_wins": mr.bot1_wins,
                "bot2_wins": mr.bot2_wins,
                "draws": mr.draws,
                "avg_moves": mr.avg_moves,
                "avg_time1": mr.avg_move_time1,
                "avg_time2": mr.avg_move_time2,
            }
            total = mr.bot1_wins + mr.bot2_wins + mr.draws
            for _ in range(mr.bot1_wins):
                _update_elo(result.elo_ratings, b1, b2, 0)
            for _ in range(mr.bot2_wins):
                _update_elo(result.elo_ratings, b1, b2, 1)
            for _ in range(mr.draws):
                _update_elo(result.elo_ratings, b1, b2, -1)

    return result
