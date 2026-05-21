import random
from quoridor.bots.base import Bot
from quoridor.bots.registry import register
from quoridor.core.state import GameState
from quoridor.core.moves import Move


@register
class RandomBot(Bot):
    name = "random"

    def __init__(self, **params):
        super().__init__(**params)
        self._rng = random.Random(params.get("seed"))

    def choose_move(self, state: GameState) -> Move:
        moves = state.legal_moves()
        return self._rng.choice(moves)
