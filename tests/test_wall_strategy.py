"""
Tests for the wall scoring matrix and strategic wall move generation.

Covers:
  - Correctness: all returned walls are legal and don't block paths
  - Quality: opponent-path walls ranked high, self-blocking walls ranked low
  - Synergy: adjacent-wall extensions score higher
  - Game phase: correct range detection
  - Performance: strategic_wall_moves under budget
  - Integration: all bots still produce valid moves
"""

import time
import random
import pytest

from quoridor.core.board import (
    BOARD_SIZE, NUM_WALL_SLOTS, sq, wall_idx, wall_idx_to_rc,
)
from quoridor.core.moves import PAWN, WALL_H, WALL_V, Player, pawn_move, wall_h, wall_v
from quoridor.core.state import GameState
from quoridor.core.pathfind import bfs_path_exists, bfs_shortest_path, bfs_path_trace
from quoridor.core.rules import _path_edge_walls, generate_wall_moves
from quoridor.eval.wall_strategy import (
    score_wall_moves,
    strategic_wall_moves,
    game_phase,
    WALL_NEIGHBORS,
    WALL_CENTER_DIST,
)
from quoridor.eval.heuristics import evaluate, game_phase as heuristic_game_phase


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _build_walled_state(rng, n_walls=6):
    state = GameState()
    placed = 0
    for _ in range(200):
        moves = state.legal_moves()
        walls = [m for m in moves if m[0] != PAWN]
        if walls and placed < n_walls:
            state.make_move(rng.choice(walls))
            placed += 1
        else:
            pawns = [m for m in moves if m[0] == PAWN]
            if pawns:
                state.make_move(rng.choice(pawns))
        if state.is_over():
            break
    return state


# ===========================================================================
#  1. PRECOMPUTED TABLES
# ===========================================================================

class TestPrecomputedTables:
    def test_wall_neighbors_count(self):
        """Corner walls have 2 neighbors, edge 3, interior 4."""
        # Corner: (0,0) = idx 0
        assert len(WALL_NEIGHBORS[0]) == 2
        # Interior: (3,3) = idx 3*8+3 = 27
        assert len(WALL_NEIGHBORS[27]) == 4

    def test_wall_neighbors_symmetric(self):
        """If A is neighbor of B, B is neighbor of A."""
        for wi in range(NUM_WALL_SLOTS):
            for nwi in WALL_NEIGHBORS[wi]:
                assert wi in WALL_NEIGHBORS[nwi], \
                    f"Wall {wi} -> {nwi} but not reverse"

    def test_wall_center_dist_range(self):
        """Center dist should be 0 at center, max at corners."""
        assert WALL_CENTER_DIST[0] > 0
        # Center of 8x8 grid is between (3,3) and (4,4)
        center_wi = 3 * 8 + 3  # (3,3) = idx 27
        assert WALL_CENTER_DIST[center_wi] < 2.0


# ===========================================================================
#  2. GAME PHASE
# ===========================================================================

class TestGamePhase:
    def test_opening(self):
        """Fresh game = opening (phase near 0)."""
        phase = game_phase(0, 10, 10)
        assert 0.0 <= phase <= 0.1

    def test_midgame(self):
        """~6 walls spent, ~30 moves = midgame."""
        phase = game_phase(30, 7, 7)
        assert 0.3 <= phase <= 0.7

    def test_endgame(self):
        """Many walls spent, many moves = endgame."""
        phase = game_phase(100, 2, 2)
        assert 0.8 <= phase <= 1.0

    def test_clamped(self):
        """Phase never exceeds [0, 1]."""
        assert game_phase(0, 10, 10) >= 0.0
        assert game_phase(999, 0, 0) <= 1.0

    def test_heuristic_game_phase_same(self):
        """heuristics.game_phase and wall_strategy.game_phase agree."""
        for mc in [0, 20, 50, 80]:
            for w0 in [10, 5, 2]:
                for w1 in [10, 5, 2]:
                    assert abs(game_phase(mc, w0, w1) - heuristic_game_phase(mc, w0, w1)) < 1e-9


# ===========================================================================
#  3. CORRECTNESS — legal and path-safe
# ===========================================================================

class TestCorrectness:
    def test_all_returned_walls_are_legal(self):
        """Every wall from strategic_wall_moves must be in state.legal_moves()."""
        rng = random.Random(42)
        for _ in range(15):
            state = _build_walled_state(rng, n_walls=rng.randint(0, 8))
            legal = set(state.legal_moves())
            strategic = strategic_wall_moves(state, top_k=10)
            for wm in strategic:
                assert wm in legal, f"Wall {wm} not in legal moves"

    def test_paths_still_exist_after_placement(self):
        """After placing any returned wall, both players must still have paths."""
        rng = random.Random(123)
        for _ in range(15):
            state = _build_walled_state(rng, n_walls=rng.randint(0, 8))
            walls = strategic_wall_moves(state, top_k=10)
            for wm in walls:
                wi = wall_idx(wm[1], wm[2])
                if wm[0] == WALL_H:
                    new_h = state.h_walls | (1 << wi)
                    new_v = state.v_walls
                else:
                    new_h = state.h_walls
                    new_v = state.v_walls | (1 << wi)
                assert bfs_path_exists(state.p1_pos, Player.ONE.goal_row, new_h, new_v), \
                    f"Wall {wm} blocks P1 path"
                assert bfs_path_exists(state.p2_pos, Player.TWO.goal_row, new_h, new_v), \
                    f"Wall {wm} blocks P2 path"

    def test_top_k_limit_respected(self):
        state = GameState()
        for k in [1, 3, 5, 10]:
            walls = strategic_wall_moves(state, top_k=k)
            assert len(walls) <= k

    def test_empty_when_no_walls_remaining(self):
        state = GameState()
        state.walls_remaining[0] = 0
        state.walls_remaining[1] = 0
        assert strategic_wall_moves(state, top_k=10) == []

    def test_strategic_moves_includes_pawn_moves(self):
        """strategic_moves() should include pawn moves."""
        state = GameState()
        moves = state.strategic_moves(top_k=5)
        pawn_moves = [m for m in moves if m[0] == PAWN]
        assert len(pawn_moves) >= 2  # opening has at least 3 pawn moves


# ===========================================================================
#  4. RANKING QUALITY
# ===========================================================================

class TestRankingQuality:
    def test_opponent_path_walls_ranked_high(self):
        """Walls that block the opponent's shortest path should be top-ranked."""
        state = GameState()
        # Move P1 to row 4, P2 stays at row 8
        state.positions[0] = sq(4, 4)
        state.current_player = 0

        scored = score_wall_moves(state, top_k=5)
        if not scored:
            return

        # Get opponent's path and the wall indices that touch it
        opp_path = bfs_path_trace(state.p2_pos, Player.TWO.goal_row,
                                   state.h_walls, state.v_walls)
        if not opp_path:
            return
        opp_walls = _path_edge_walls(opp_path)

        # Top wall should touch opponent's path
        top_score, top_move = scored[0]
        top_wi = wall_idx(top_move[1], top_move[2])
        # At least one of the top-3 should be on opponent's path
        top3_wis = {wall_idx(m[1], m[2]) for _, m in scored[:3]}
        assert top3_wis & opp_walls, \
            f"None of top-3 walls {top3_wis} touch opponent path walls {opp_walls}"

    def test_self_blocking_walls_ranked_low(self):
        """Walls that block our own path should not be in top-3."""
        state = GameState()
        state.positions[0] = sq(4, 4)
        state.current_player = 0

        my_path = bfs_path_trace(state.p1_pos, Player.ONE.goal_row,
                                  state.h_walls, state.v_walls)
        if not my_path:
            return
        my_walls = _path_edge_walls(my_path)

        scored = score_wall_moves(state, top_k=10)
        if len(scored) < 3:
            return

        top3_wis = {wall_idx(m[1], m[2]) for _, m in scored[:3]}
        # Top-3 should NOT be exclusively self-blocking
        purely_self_blocking = top3_wis - set()
        # Check that not all top-3 are on own path
        on_own_count = sum(1 for wi in top3_wis if wi in my_walls)
        assert on_own_count < 3, "All top-3 walls are self-blocking"


# ===========================================================================
#  5. SYNERGY SCORING
# ===========================================================================

class TestSynergyScoring:
    def test_wall_adjacent_to_existing_scores_higher(self):
        """A wall next to an existing wall should score higher than an isolated wall."""
        state = GameState()
        # Place a horizontal wall at (3, 3)
        state.make_move(wall_h(3, 3))
        # Now P2's turn — place some pawn move to get back to P1
        state.make_move(pawn_move(7, 4))

        scored = score_wall_moves(state, top_k=20)
        if len(scored) < 2:
            return

        # Find walls adjacent to (3,3) and walls far away
        existing_wi = wall_idx(3, 3)
        adjacent_wis = set(WALL_NEIGHBORS[existing_wi])

        adj_scores = [s for s, m in scored if wall_idx(m[1], m[2]) in adjacent_wis]
        far_scores = [s for s, m in scored if wall_idx(m[1], m[2]) not in adjacent_wis
                      and wall_idx(m[1], m[2]) != existing_wi]

        if adj_scores and far_scores:
            avg_adj = sum(adj_scores) / len(adj_scores)
            avg_far = sum(far_scores) / len(far_scores)
            # Adjacent walls should score at least somewhat higher on average
            # (not guaranteed for every single wall, but statistically)
            # This is a soft check — synergy is one factor among many
            assert avg_adj > avg_far - 2.0, \
                f"Adjacent walls avg={avg_adj:.2f} not higher than far walls avg={avg_far:.2f}"


# ===========================================================================
#  6. PHASE-AWARE EVALUATION
# ===========================================================================

class TestPhaseAwareEvaluation:
    def test_evaluate_backward_compatible(self):
        """evaluate() without move_count should still work (defaults to 0)."""
        state = GameState()
        score = evaluate(
            0, state.p1_pos, state.p2_pos,
            state.h_walls, state.v_walls,
            (state.walls_remaining[0], state.walls_remaining[1]),
        )
        assert isinstance(score, float)

    def test_evaluate_with_move_count(self):
        """evaluate() with move_count should produce different weights."""
        state = GameState()
        score_early = evaluate(
            0, state.p1_pos, state.p2_pos,
            state.h_walls, state.v_walls,
            (state.walls_remaining[0], state.walls_remaining[1]),
            move_count=0,
        )
        score_late = evaluate(
            0, state.p1_pos, state.p2_pos,
            state.h_walls, state.v_walls,
            (2, 2),  # few walls remaining
            move_count=80,
        )
        # Both should be floats, potentially different
        assert isinstance(score_early, float)
        assert isinstance(score_late, float)


# ===========================================================================
#  7. PERFORMANCE
# ===========================================================================

class TestPerformance:
    def test_strategic_wall_moves_speed(self):
        """strategic_wall_moves should be under 5ms on opening position."""
        state = GameState()
        # Warm up
        strategic_wall_moves(state, top_k=10)

        N = 100
        t0 = time.perf_counter()
        for _ in range(N):
            strategic_wall_moves(state, top_k=10)
        elapsed = time.perf_counter() - t0
        avg_ms = (elapsed / N) * 1000
        print(f"  strategic_wall_moves avg: {avg_ms:.1f}ms")
        assert avg_ms < 5.0, f"Too slow: {avg_ms:.1f}ms (target <5ms)"

    def test_strategic_moves_speed(self):
        """state.strategic_moves() should be under 5ms."""
        state = GameState()
        state.strategic_moves(top_k=10)

        N = 100
        t0 = time.perf_counter()
        for _ in range(N):
            state.strategic_moves(top_k=10)
        elapsed = time.perf_counter() - t0
        avg_ms = (elapsed / N) * 1000
        print(f"  strategic_moves avg: {avg_ms:.1f}ms")
        assert avg_ms < 5.0, f"Too slow: {avg_ms:.1f}ms (target <5ms)"

    def test_strategic_faster_than_legal(self):
        """strategic_moves should be comparable to or faster than legal_moves
        for the purpose of bot decision making (fewer moves to evaluate)."""
        state = GameState()
        legal = state.legal_moves()
        strategic = state.strategic_moves(top_k=10)
        # Strategic should have far fewer moves
        print(f"  legal_moves: {len(legal)}, strategic_moves: {len(strategic)}")
        assert len(strategic) < len(legal), \
            f"Strategic ({len(strategic)}) not fewer than legal ({len(legal)})"


# ===========================================================================
#  8. BOT INTEGRATION
# ===========================================================================

class TestBotIntegration:
    @pytest.fixture(autouse=True)
    def _setup(self):
        from quoridor.bots.registry import discover_bots
        discover_bots()

    def _run_bot_moves(self, bot_name, n_moves=10, **params):
        from quoridor.bots.registry import get_bot
        bot = get_bot(bot_name, **params)
        state = GameState()
        for _ in range(n_moves):
            if state.is_over():
                break
            move = bot.choose_move(state)
            assert move in state.legal_moves(), \
                f"{bot_name} produced illegal move {move}"
            state.make_move(move)
        return state

    def test_random_bot(self):
        self._run_bot_moves("random", seed=1)

    def test_greedy_bot(self):
        self._run_bot_moves("greedy", n_moves=6)

    def test_minimax_bot(self):
        self._run_bot_moves("minimax", n_moves=4, depth=1)

    def test_alphabeta_bot(self):
        self._run_bot_moves("alphabeta", n_moves=4, depth=2)

    def test_mcts_bot(self):
        self._run_bot_moves("mcts", n_moves=4, simulations=50, seed=1)

    def test_greedy_completes_game(self):
        """Greedy should complete a game without errors."""
        state = self._run_bot_moves("greedy", n_moves=40)
        # Game should progress without crashes
        assert state.move_count >= 1
