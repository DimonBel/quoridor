from quoridor.bots.base import Bot
from quoridor.bots.registry import register
from quoridor.core.state import GameState
from quoridor.core.moves import PAWN
from quoridor.eval.heuristics import evaluate


def _sort_moves(moves: list) -> list:
    pawn = []
    walls = []
    for m in moves:
        if m[0] == PAWN:
            pawn.append(m)
        else:
            walls.append(m)
    pawn.extend(walls)
    return pawn


@register
class MinimaxBot(Bot):
    name = "minimax"

    def __init__(self, **params):
        super().__init__(**params)
        self.depth = params.get("depth", 2)
        self.path_weight = params.get("path_weight", 2.0)
        self.wall_weight = params.get("wall_weight", 0.5)

    def choose_move(self, state: GameState):
        moves = _sort_moves(state.legal_moves())
        if not moves:
            return moves[0]
        best_move = moves[0]
        best_score = -999999.0
        maximizing = state.current_player == 0
        for move in moves:
            undo = state.make_move(move)
            score = self._minimax(self.depth - 1, not maximizing, state)
            state.unmake_move(undo)
            if score > best_score:
                best_score = score
                best_move = move
        return best_move

    def _minimax(self, depth: int, maximizing: bool, state: GameState) -> float:
        winner = state.winner()
        if winner == 0:
            return 10000.0 + depth
        if winner == 1:
            return -10000.0 - depth
        if depth == 0 or state.is_over():
            return self._evaluate(state)

        moves = _sort_moves(state.legal_moves())
        if maximizing:
            best = -999999.0
            for move in moves:
                undo = state.make_move(move)
                val = self._minimax(depth - 1, False, state)
                state.unmake_move(undo)
                if val > best:
                    best = val
            return best
        else:
            best = 999999.0
            for move in moves:
                undo = state.make_move(move)
                val = self._minimax(depth - 1, True, state)
                state.unmake_move(undo)
                if val < best:
                    best = val
            return best

    def _evaluate(self, state: GameState) -> float:
        return evaluate(
            state.current_player,
            state.p1_pos, state.p2_pos,
            state.h_walls, state.v_walls,
            (state.walls_remaining[0], state.walls_remaining[1]),
            self.path_weight, self.wall_weight,
        )
