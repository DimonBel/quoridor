from quoridor.bots.base import Bot
from quoridor.bots.registry import register
from quoridor.core.state import GameState
from quoridor.core.moves import PAWN
from quoridor.eval.heuristics import evaluate

_INF = 999999.0


def _zobrist_hash(state: GameState) -> int:
    h = state.h_walls ^ (state.v_walls * 0x9E3779B97F4A7C15)
    h ^= state.p1_pos * 0x517CC1B727220A95
    h ^= state.p2_pos * 0x6C62272E07BB0142
    h ^= state.current_player * 0x4CF5AD432745937F
    return h & 0x7FFFFFFFFFFFFFFF


@register
class AlphaBetaBot(Bot):
    name = "alphabeta"

    def __init__(self, **params):
        super().__init__(**params)
        self.depth = params.get("depth", 3)
        self.path_weight = params.get("path_weight", 2.0)
        self.wall_weight = params.get("wall_weight", 0.5)
        self.root_top_k = params.get("root_top_k", 15)
        self.inner_top_k = params.get("inner_top_k", 8)
        self._tt: dict = {}

    def choose_move(self, state: GameState):
        moves = state.strategic_moves(top_k=self.root_top_k)
        if not moves:
            moves = state.pawn_moves_only()
        if not moves:
            return state.legal_moves()[0]
        self._tt.clear()
        self._move_cache: dict = {}
        best_move = moves[0]
        best_score = -_INF
        maximizing = state.current_player == 0
        alpha = -_INF
        beta = _INF
        for move in moves:
            undo = state.make_move(move)
            score = self._alphabeta(self.depth - 1, alpha, beta, not maximizing, state)
            state.unmake_move(undo)
            if score > best_score:
                best_score = score
                best_move = move
            if maximizing:
                alpha = max(alpha, score)
        return best_move

    def _alphabeta(
        self, depth: int, alpha: float, beta: float, maximizing: bool, state: GameState
    ) -> float:
        h = _zobrist_hash(state)
        tt_key = (h, depth, maximizing)
        tt = self._tt
        cached = tt.get(tt_key)
        if cached is not None:
            return cached

        winner = state.winner()
        if winner == 0:
            return 10000.0 + depth
        if winner == 1:
            return -10000.0 - depth
        if depth == 0 or state.is_over():
            val = self._evaluate(state)
            tt[tt_key] = val
            return val

        sk = state._state_key()
        cached = self._move_cache.get(sk)
        if cached is not None:
            moves = cached
        else:
            moves = state.strategic_moves(top_k=self.inner_top_k)
            if not moves:
                moves = state.pawn_moves_only()
            self._move_cache[sk] = moves

        if maximizing:
            val = -_INF
            for move in moves:
                undo = state.make_move(move)
                val = max(val, self._alphabeta(depth - 1, alpha, beta, False, state))
                state.unmake_move(undo)
                alpha = max(alpha, val)
                if alpha >= beta:
                    break
            tt[tt_key] = val
            return val
        else:
            val = _INF
            for move in moves:
                undo = state.make_move(move)
                val = min(val, self._alphabeta(depth - 1, alpha, beta, True, state))
                state.unmake_move(undo)
                beta = min(beta, val)
                if alpha >= beta:
                    break
            tt[tt_key] = val
            return val

    def _evaluate(self, state: GameState) -> float:
        return evaluate(
            state.current_player,
            state.p1_pos, state.p2_pos,
            state.h_walls, state.v_walls,
            (state.walls_remaining[0], state.walls_remaining[1]),
            self.path_weight, self.wall_weight,
            move_count=state.move_count,
        )
