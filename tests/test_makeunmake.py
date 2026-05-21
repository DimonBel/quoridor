import pytest
from quoridor.core.board import sq, BOARD_SIZE
from quoridor.core.moves import pawn_move, wall_h, wall_v
from quoridor.core.state import GameState


class TestMakeUnmakeExact:
    def _snapshot(self, state):
        return (
            state.h_walls,
            state.v_walls,
            state.positions[0],
            state.positions[1],
            state.walls_remaining[0],
            state.walls_remaining[1],
            state.current_player,
            state.move_count,
        )

    def test_single_pawn_move(self):
        state = GameState()
        snap = self._snapshot(state)
        undo = state.make_move(pawn_move(1, 4))
        state.unmake_move(undo)
        assert self._snapshot(state) == snap

    def test_single_wall_h(self):
        state = GameState()
        snap = self._snapshot(state)
        undo = state.make_move(wall_h(0, 0))
        state.unmake_move(undo)
        assert self._snapshot(state) == snap

    def test_single_wall_v(self):
        state = GameState()
        snap = self._snapshot(state)
        undo = state.make_move(wall_v(0, 0))
        state.unmake_move(undo)
        assert self._snapshot(state) == snap

    def test_ten_random_moves(self):
        import random
        random.seed(123)
        state = GameState()
        snap = self._snapshot(state)
        undos = []
        for _ in range(10):
            moves = state.legal_moves()
            if not moves or state.is_over():
                break
            move = random.choice(moves)
            undos.append(state.make_move(move))
        for undo in reversed(undos):
            state.unmake_move(undo)
        assert self._snapshot(state) == snap

    def test_thirty_random_moves(self):
        import random
        random.seed(999)
        state = GameState()
        snap = self._snapshot(state)
        undos = []
        for _ in range(30):
            moves = state.legal_moves()
            if not moves or state.is_over():
                break
            move = random.choice(moves)
            undos.append(state.make_move(move))
        for undo in reversed(undos):
            state.unmake_move(undo)
        assert self._snapshot(state) == snap

    def test_alternating_pawn_wall(self):
        state = GameState()
        snap = self._snapshot(state)
        undos = []
        undos.append(state.make_move(pawn_move(1, 4)))
        moves = state.legal_moves()
        wall_moves = [m for m in moves if m[0] != 0]  # not PAWN
        if wall_moves:
            undos.append(state.make_move(wall_moves[0]))
        for undo in reversed(undos):
            state.unmake_move(undo)
        assert self._snapshot(state) == snap
