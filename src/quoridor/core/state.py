from quoridor.core.moves import PAWN, WALL_H, WALL_V, Player
from quoridor.core.board import (
    BOARD_SIZE,
    sq,
    _SQ_ROW,
    wall_idx,
)
from quoridor.core.rules import (
    check_winner,
    generate_all_moves,
    generate_pawn_moves,
)


class UndoRecord:
    __slots__ = ("move", "h_walls", "v_walls", "positions", "walls_remaining", "current_player")

    def __init__(self, move, h_walls, v_walls, positions, walls_remaining, current_player):
        self.move = move
        self.h_walls = h_walls
        self.v_walls = v_walls
        self.positions = positions
        self.walls_remaining = walls_remaining
        self.current_player = current_player


class GameState:
    __slots__ = (
        "positions",
        "h_walls",
        "v_walls",
        "walls_remaining",
        "current_player",
        "move_count",
        "max_moves",
    )

    def __init__(
        self,
        board_size: int = BOARD_SIZE,
        walls_per_player: int = 10,
        max_moves: int = 400,
    ):
        p1_start = sq(Player.ONE.start_row, Player.ONE.start_col)
        p2_start = sq(Player.TWO.start_row, Player.TWO.start_col)
        self.positions = [p1_start, p2_start]
        self.h_walls: int = 0
        self.v_walls: int = 0
        self.walls_remaining: list[int] = [walls_per_player, walls_per_player]
        self.current_player: int = 0
        self.move_count: int = 0
        self.max_moves: int = max_moves

    @property
    def p1_pos(self) -> int:
        return self.positions[0]

    @property
    def p2_pos(self) -> int:
        return self.positions[1]

    @property
    def h_walls_count(self) -> int:
        return bin(self.h_walls).count("1")

    @property
    def v_walls_count(self) -> int:
        return bin(self.v_walls).count("1")

    def legal_moves(self) -> list:
        return generate_all_moves(
            self.current_player,
            (self.positions[0], self.positions[1]),
            self.h_walls,
            self.v_walls,
            (self.walls_remaining[0], self.walls_remaining[1]),
        )

    def pawn_moves_only(self) -> list:
        """Fast move gen — pawn moves only, no wall validation BFS."""
        cp = self.current_player
        return generate_pawn_moves(
            self.positions[cp],
            self.positions[1 - cp],
            self.h_walls,
            self.v_walls,
        )

    def make_move(self, move) -> UndoRecord:
        undo = UndoRecord(
            move,
            self.h_walls,
            self.v_walls,
            (self.positions[0], self.positions[1]),
            (self.walls_remaining[0], self.walls_remaining[1]),
            self.current_player,
        )
        mt = move[0]
        if mt == PAWN:
            self.positions[self.current_player] = sq(move[1], move[2])
        elif mt == WALL_H:
            wi = wall_idx(move[1], move[2])
            self.h_walls |= 1 << wi
            self.walls_remaining[self.current_player] -= 1
        elif mt == WALL_V:
            wi = wall_idx(move[1], move[2])
            self.v_walls |= 1 << wi
            self.walls_remaining[self.current_player] -= 1
        self.current_player = 1 - self.current_player
        self.move_count += 1
        return undo

    def unmake_move(self, undo: UndoRecord) -> None:
        self.h_walls = undo.h_walls
        self.v_walls = undo.v_walls
        self.positions[0] = undo.positions[0]
        self.positions[1] = undo.positions[1]
        self.walls_remaining[0] = undo.walls_remaining[0]
        self.walls_remaining[1] = undo.walls_remaining[1]
        self.current_player = undo.current_player
        self.move_count -= 1

    def winner(self) -> int:
        return check_winner(self.positions[0], self.positions[1])

    def is_over(self) -> bool:
        if self.winner() >= 0:
            return True
        if self.move_count >= self.max_moves:
            return True
        return False

    def clone(self) -> "GameState":
        gs = GameState.__new__(GameState)
        gs.positions = [self.positions[0], self.positions[1]]
        gs.h_walls = self.h_walls
        gs.v_walls = self.v_walls
        gs.walls_remaining = [self.walls_remaining[0], self.walls_remaining[1]]
        gs.current_player = self.current_player
        gs.move_count = self.move_count
        gs.max_moves = self.max_moves
        return gs
