# Move types as plain ints — no enum overhead in hot loops
PAWN = 0
WALL_H = 1
WALL_V = 2

# Move is a plain tuple: (move_type, row, col)
# Access: move[0]=type, move[1]=row, move[2]=col

def pawn_move(row: int, col: int) -> tuple:
    return (PAWN, row, col)

def wall_h(row: int, col: int) -> tuple:
    return (WALL_H, row, col)

def wall_v(row: int, col: int) -> tuple:
    return (WALL_V, row, col)

def move_repr(m: tuple) -> str:
    if m[0] == PAWN:
        return f"P({m[1]},{m[2]})"
    d = "H" if m[0] == WALL_H else "V"
    return f"W{d}({m[1]},{m[2]})"

# Keep Move as an alias for backward compat in type hints
Move = tuple

# MoveType compat shim — just an object with attributes matching the old enum
class _MoveTypeCompat:
    PAWN = PAWN
    WALL_H = WALL_H
    WALL_V = WALL_V

MoveType = _MoveTypeCompat()


class Player:
    """Player constants — no enum overhead."""
    __slots__ = ("value", "goal_row", "start_row", "start_col")

    def __init__(self, value, goal_row, start_row, start_col):
        self.value = value
        self.goal_row = goal_row
        self.start_row = start_row
        self.start_col = start_col

    @property
    def opponent(self):
        return P2 if self.value == 0 else P1

    def __eq__(self, other):
        if isinstance(other, Player):
            return self.value == other.value
        return self.value == other

    def __hash__(self):
        return self.value


P1 = Player(0, goal_row=8, start_row=0, start_col=4)
P2 = Player(1, goal_row=0, start_row=8, start_col=4)

# Indexed access: PLAYERS[0] = P1, PLAYERS[1] = P2
PLAYERS = (P1, P2)

# Keep old enum-style access
Player.ONE = P1
Player.TWO = P2

DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIR_NAMES = {(-1, 0): "N", (1, 0): "S", (0, -1): "W", (0, 1): "E"}
