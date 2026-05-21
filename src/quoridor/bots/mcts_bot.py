import math
import random
from quoridor.bots.base import Bot
from quoridor.bots.registry import register
from quoridor.core.state import GameState
from quoridor.core.moves import PAWN


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

    def ucb1(self, exploration: float, log_parent: float) -> float:
        if self.visits == 0:
            return float("inf")
        return self.wins / self.visits + exploration * math.sqrt(log_parent / self.visits)

    def select_child(self, exploration: float) -> "_MCTSNode":
        log_p = math.log(self.visits)
        return max(self.children, key=lambda c: c.ucb1(exploration, log_p))

    def add_child(self, move, untried_moves: list, player_just_moved: int) -> "_MCTSNode":
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
        self.top_k = params.get("top_k", 10)
        self._rng = random.Random(params.get("seed"))

    def choose_move(self, state: GameState):
        moves = state.strategic_moves(top_k=self.top_k)
        if not moves:
            moves = state.pawn_moves_only()
        if len(moves) == 1:
            return moves[0]

        root = _MCTSNode(
            untried_moves=moves,
            player_just_moved=1 - state.current_player,
        )

        for _ in range(self.simulations):
            node = root
            sim_state = state.clone()

            # Selection
            while not node.untried_moves and node.children:
                node = node.select_child(self.exploration)
                sim_state.make_move(node.move)

            # Expansion — use strategic moves to keep tree narrow
            if node.untried_moves and not sim_state.is_over():
                move = self._rng.choice(node.untried_moves)
                sim_state.make_move(move)
                child_moves = sim_state.strategic_moves(top_k=self.top_k)
                if not child_moves:
                    child_moves = sim_state.pawn_moves_only()
                node = node.add_child(
                    move,
                    child_moves,
                    1 - sim_state.current_player,
                )

            # Rollout — pawn-only for speed
            result = self._rollout_fn(sim_state)

            # Backprop
            while node is not None:
                node.visits += 1
                if result == node.player_just_moved:
                    node.wins += 1.0
                elif result == -1:
                    node.wins += 0.5
                node = node.parent

        best = max(root.children, key=lambda c: c.visits)
        return best.move

    def _rollout_fn(self, state: GameState) -> int:
        """Fast pawn-only rollout — avoids wall move generation entirely."""
        rng_choice = self._rng.choice
        while not state.is_over():
            moves = state.pawn_moves_only()
            if not moves:
                break
            state.make_move(rng_choice(moves))
        return state.winner()
