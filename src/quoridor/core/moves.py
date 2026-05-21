from enum import Enum, auto
from typing import NamedTuple


class MoveType(Enum):
    PAWN = auto()
    WALL_H = auto()
    WALL_V = auto()


class Move(NamedTuple):
    move_type: MoveType
    row: int
    col: int

    @property
    def is_pawn(self) -> bool:
        return self.move_type == MoveType.PAWN

    @property
    def is_wall(self) -> bool:
        return self.move_type in (MoveType.WALL_H, MoveType.WALL_V)

    def __repr__(self) -> str:
        if self.move_type == MoveType.PAWN:
            return f"P({self.row},{self.col})"
        direction = "H" if self.move_type == MoveType.WALL_H else "V"
        return f"W{direction}({self.row},{self.col})"


def pawn_move(row: int, col: int) -> Move:
    return Move(MoveType.PAWN, row, col)


def wall_h(row: int, col: int) -> Move:
    return Move(MoveType.WALL_H, row, col)


def wall_v(row: int, col: int) -> Move:
    return Move(MoveType.WALL_V, row, col)


class Player(Enum):
    ONE = 0
    TWO = 1

    @property
    def opponent(self) -> "Player":
        return Player.TWO if self == Player.ONE else Player.ONE

    @property
    def goal_row(self) -> int:
        return 8 if self == Player.ONE else 0

    @property
    def start_row(self) -> int:
        return 0 if self == Player.ONE else 8

    @property
    def start_col(self) -> int:
        return 4


DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIR_NAMES = {(-1, 0): "N", (1, 0): "S", (0, -1): "W", (0, 1): "E"}
