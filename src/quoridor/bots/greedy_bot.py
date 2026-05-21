from quoridor.bots.base import Bot
from quoridor.bots.registry import register
from quoridor.core.state import GameState
from quoridor.core.pathfind import bfs_shortest_path
from quoridor.core.moves import Player


@register
class GreedyBot(Bot):
    name = "greedy"

    def __init__(self, **params):
        super().__init__(**params)
        self.path_weight = params.get("path_weight", 2.0)
        self.wall_weight = params.get("wall_weight", 1.0)
        self.top_k = params.get("top_k", 8)

    def choose_move(self, state: GameState):
        moves = state.strategic_moves(top_k=self.top_k)
        if not moves:
            moves = state.pawn_moves_only()
        if not moves:
            return state.legal_moves()[0]
        best_move = moves[0]
        best_score = -999999.0
        cp = state.current_player
        for move in moves:
            undo = state.make_move(move)
            score = self._evaluate(cp, state)
            state.unmake_move(undo)
            if score > best_score:
                best_score = score
                best_move = move
        return best_move

    def _evaluate(self, player: int, state: GameState) -> float:
        p1_dist = bfs_shortest_path(
            state.p1_pos, Player.ONE.goal_row, state.h_walls, state.v_walls
        )
        p2_dist = bfs_shortest_path(
            state.p2_pos, Player.TWO.goal_row, state.h_walls, state.v_walls
        )
        if player == 0:
            if p1_dist == 0:
                return 10000.0
            if p2_dist == 0:
                return -10000.0
            return self.path_weight * (p2_dist - p1_dist)
        else:
            if p2_dist == 0:
                return 10000.0
            if p1_dist == 0:
                return -10000.0
            return self.path_weight * (p1_dist - p2_dist)
