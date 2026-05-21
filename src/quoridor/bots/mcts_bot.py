import math
import random
import time
from quoridor.bots.base import Bot
from quoridor.bots.registry import register
from quoridor.core.state import GameState
from quoridor.core.moves import Move
from quoridor.eval.heuristics import evaluate


class _MCTSNode:
    __slots__ = ("move", "parent", "children", "wins", "visits", "untried_moves", "player_just_moved")

    def __init__(self, move=None, parent=None, untried_moves=None, player_just_moved=0):
        self.move = move
        self.parent = parent
        self.children: list[_MCTSNode] = []
        self.wins: float = 0.0
        self.visits: int = 0
        self.untried_moves = list(untried_moves) if untried_moves else []
        self.player_just_moved = player_just_moved

    def ucb1(self, exploration: float = 1.414) -> float:
        if self.visits == 0:
            return float("inf")
        return self.wins / self.visits + exploration * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )

    def select_child(self, exploration: float) -> "_MCTSNode":
        return max(self.children, key=lambda c: c.ucb1(exploration))

    def add_child(self, move: Move, untried_moves: list, player_just_moved: int) -> "_MCTSNode":
        child = _MCTSNode(
            move=move, parent=self,
            untried_moves=untried_moves,
            player_just_moved=player_just_moved,
        )
        self.untried_moves.remove(move)
        self.children.append(child)
        return child


@register
class MCTSBot(Bot):
    name = "mcts"

    def __init__(self, **params):
        super().__init__(**params)
        self.simulations = params.get("simulations", 1000)
        self.exploration = params.get("exploration", 1.414)
        self.rollout = params.get("rollout", "random")
        self._rng = random.Random(params.get("seed"))

    def choose_move(self, state: GameState) -> Move:
        moves = state.legal_moves()
        if len(moves) == 1:
            return moves[0]

        root = _MCTSNode(
            untried_moves=moves,
            player_just_moved=1 - state.current_player,
        )

        for _ in range(self.simulations):
            node = root
            sim_state = state.clone()

            while not node.untried_moves and node.children:
                node = node.select_child(self.exploration)
                sim_state.make_move(node.move)

            if node.untried_moves and not sim_state.is_over():
                move = self._rng.choice(node.untried_moves)
                sim_state.make_move(move)
                node = node.add_child(
                    move,
                    sim_state.legal_moves(),
                    1 - sim_state.current_player,
                )

            result = self._rollout(sim_state)

            while node is not None:
                node.visits += 1
                if result == node.player_just_moved:
                    node.wins += 1.0
                elif result == -1:
                    node.wins += 0.5
                node = node.parent

        best = max(root.children, key=lambda c: c.visits)
        return best.move

    def _rollout(self, state: GameState) -> int:
        while not state.is_over():
            moves = state.legal_moves()
            if not moves:
                break
            if self.rollout == "random":
                move = self._rng.choice(moves)
            else:
                move = self._heuristic_move(state, moves)
            state.make_move(move)
        return state.winner()

    def _heuristic_move(self, state: GameState, moves: list[Move]) -> Move:
        best = moves[0]
        best_score = -999999.0
        cp = state.current_player
        for move in moves[:10]:
            undo = state.make_move(move)
            score = evaluate(
                cp, state.p1_pos, state.p2_pos,
                state.h_walls, state.v_walls,
                (state.walls_remaining[0], state.walls_remaining[1]),
            )
            state.unmake_move(undo)
            if score > best_score:
                best_score = score
                best = move
        return best
