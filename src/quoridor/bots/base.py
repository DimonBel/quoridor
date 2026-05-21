from abc import ABC, abstractmethod
from quoridor.core.moves import Move
from quoridor.core.state import GameState


class Bot(ABC):
    name: str = ""

    def __init__(self, **params):
        self.params = params

    @abstractmethod
    def choose_move(self, state: GameState) -> Move:
        ...
