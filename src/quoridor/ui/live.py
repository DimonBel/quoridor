import time
from quoridor.core.state import GameState
from quoridor.core.moves import MoveType
from quoridor.bots.base import Bot
from quoridor.runner.game_runner import _move_to_str, _str_to_move, GameResult, run_game
from quoridor.ui.renderer import render_live


def run_game_live(
    bot1: Bot,
    bot2: Bot,
    delay_ms: int = 200,
    seed: int = 0,
    max_moves: int = 400,
) -> GameResult:
    def callback(state, move, player, elapsed):
        render_live(state, last_move=move, bot_names=(bot1.name, bot2.name))
        if delay_ms > 0:
            time.sleep(delay_ms / 1000.0)

    result = run_game(bot1, bot2, seed=seed, max_moves=max_moves, callback=callback)
    winner_name = (bot1.name, bot2.name)[result.winner] if result.winner >= 0 else "Draw"
    print(f"\n  Winner: {winner_name} in {result.moves} moves")
    return result
