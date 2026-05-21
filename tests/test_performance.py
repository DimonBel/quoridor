"""
Performance tests for Quoridor engine.

Run with:  pytest tests/test_performance.py -v -s
Or:        quoridor test   (via CLI)

These tests measure raw throughput of every layer — from individual BFS calls
up through full bot-vs-bot games.  Each test prints a metric and enforces a
minimum-acceptable bar.  When something gets slower, the failing test tells
you exactly which layer regressed and by how much.

The thresholds are deliberately conservative (2-5x headroom above "unusably
slow") so they pass on modest hardware.  Tighten them as the engine improves.
"""

import time
import random
import statistics
import pytest

from quoridor.core.board import (
    BOARD_SIZE, NUM_WALL_SLOTS, sq, sq_to_rc, wall_idx, _SQ_ROW,
    NEIGHBOR_DIR, EDGE_H_MASK, EDGE_V_MASK,
)
from quoridor.core.moves import (
    PAWN, WALL_H, WALL_V, Player, pawn_move, wall_h, wall_v,
)
from quoridor.core.state import GameState
from quoridor.core.rules import (
    generate_pawn_moves, generate_wall_moves, generate_all_moves,
)
from quoridor.core.pathfind import (
    bfs_path_exists, bfs_shortest_path, bfs_path_trace, astar_shortest_path,
)
from quoridor.eval.heuristics import evaluate


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _timed(fn, iterations: int) -> tuple[float, float]:
    """Run *fn()* *iterations* times. Return (total_seconds, ops_per_second)."""
    t0 = time.perf_counter()
    for _ in range(iterations):
        fn()
    elapsed = time.perf_counter() - t0
    return elapsed, iterations / elapsed


def _run_random_game(rng: random.Random, use_walls: bool = True) -> int:
    """Play a full random game, return move count."""
    state = GameState()
    while not state.is_over():
        if use_walls:
            moves = state.legal_moves()
        else:
            moves = state.pawn_moves_only()
        if not moves:
            break
        state.make_move(rng.choice(moves))
    return state.move_count


def _build_walled_state(rng: random.Random, n_walls: int = 6) -> GameState:
    """Return a mid-game state with some walls placed."""
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


def _print_metric(name: str, value: float, unit: str):
    print(f"  {name:.<50s} {value:>10.1f} {unit}")


# ===========================================================================
#  1. PATHFINDING — the innermost hot loop
# ===========================================================================

class TestPathfindingPerformance:
    """BFS / A* are called hundreds of times per legal_moves().
    If these are slow everything is slow."""

    def test_bfs_exists_empty_board(self):
        """BFS path-exists on an open board (best case)."""
        N = 50_000
        start = sq(0, 4)
        elapsed, ops = _timed(lambda: bfs_path_exists(start, 8, 0, 0), N)
        _print_metric("bfs_path_exists (empty)", ops, "calls/s")
        assert ops > 10_000, f"BFS exists too slow: {ops:.0f} calls/s"

    def test_bfs_exists_walled(self):
        """BFS path-exists with a partial wall row (forces detour)."""
        h_walls = 0
        for c in range(BOARD_SIZE - 2):
            h_walls |= 1 << wall_idx(3, c)
        N = 50_000
        start = sq(0, 4)
        elapsed, ops = _timed(lambda: bfs_path_exists(start, 8, h_walls, 0), N)
        _print_metric("bfs_path_exists (walled)", ops, "calls/s")
        assert ops > 8_000, f"BFS exists (walled) too slow: {ops:.0f} calls/s"

    def test_bfs_shortest_empty(self):
        """BFS shortest path on an open board."""
        N = 50_000
        start = sq(0, 4)
        elapsed, ops = _timed(lambda: bfs_shortest_path(start, 8, 0, 0), N)
        _print_metric("bfs_shortest_path (empty)", ops, "calls/s")
        assert ops > 8_000, f"BFS shortest too slow: {ops:.0f} calls/s"

    def test_bfs_shortest_walled(self):
        """BFS shortest with walls forcing a longer path."""
        h_walls = 0
        for c in range(BOARD_SIZE - 2):
            h_walls |= 1 << wall_idx(4, c)
        N = 50_000
        start = sq(0, 4)
        elapsed, ops = _timed(lambda: bfs_shortest_path(start, 8, h_walls, 0), N)
        _print_metric("bfs_shortest_path (walled)", ops, "calls/s")
        assert ops > 6_000, f"BFS shortest (walled) too slow: {ops:.0f} calls/s"

    def test_bfs_trace_empty(self):
        """BFS path trace (returns full path list) on open board."""
        N = 30_000
        start = sq(0, 4)
        elapsed, ops = _timed(lambda: bfs_path_trace(start, 8, 0, 0), N)
        _print_metric("bfs_path_trace (empty)", ops, "calls/s")
        assert ops > 5_000, f"BFS trace too slow: {ops:.0f} calls/s"

    def test_astar_vs_bfs_parity(self):
        """A* must return same distance as BFS and not be dramatically slower."""
        N = 30_000
        start = sq(0, 4)
        _, bfs_ops = _timed(lambda: bfs_shortest_path(start, 8, 0, 0), N)
        _, astar_ops = _timed(lambda: astar_shortest_path(start, 8, 0, 0), N)
        _print_metric("astar_shortest_path (empty)", astar_ops, "calls/s")
        ratio = bfs_ops / astar_ops if astar_ops > 0 else 999
        _print_metric("BFS/A* speed ratio", ratio, "x")
        # A* can be slower due to heapq overhead but shouldn't be >5x worse
        assert ratio < 5.0, f"A* is {ratio:.1f}x slower than BFS — check heapq overhead"


# ===========================================================================
#  2. MOVE GENERATION — branching factor gatekeeper
# ===========================================================================

class TestMoveGenPerformance:
    """Move gen is called at every search node. Wall move gen is the usual
    bottleneck because it validates each candidate wall with BFS."""

    def test_pawn_movegen_empty(self):
        """Pawn moves only — no BFS, should be very fast."""
        state = GameState()
        N = 100_000
        pos, opp, hw, vw = state.p1_pos, state.p2_pos, state.h_walls, state.v_walls
        elapsed, ops = _timed(lambda: generate_pawn_moves(pos, opp, hw, vw), N)
        _print_metric("pawn movegen (empty)", ops, "calls/s")
        assert ops > 200_000, f"Pawn movegen too slow: {ops:.0f} calls/s"

    def test_pawn_movegen_with_walls(self):
        """Pawn moves with walls placed — tests edge-blocking lookups."""
        rng = random.Random(42)
        state = _build_walled_state(rng, n_walls=8)
        N = 100_000
        cp = state.current_player
        pos = state.positions[cp]
        opp = state.positions[1 - cp]
        elapsed, ops = _timed(
            lambda: generate_pawn_moves(pos, opp, state.h_walls, state.v_walls), N
        )
        _print_metric("pawn movegen (walled)", ops, "calls/s")
        assert ops > 150_000, f"Pawn movegen (walled) too slow: {ops:.0f} calls/s"

    def test_wall_movegen_empty(self):
        """Wall move gen on open board — every slot is free, many BFS checks."""
        state = GameState()
        N = 500
        elapsed, ops = _timed(
            lambda: generate_wall_moves(
                state.p1_pos, state.p2_pos, state.h_walls, state.v_walls, 10
            ), N
        )
        _print_metric("wall movegen (empty)", ops, "calls/s")
        assert ops > 100, f"Wall movegen too slow: {ops:.0f} calls/s"

    def test_wall_movegen_midgame(self):
        """Wall move gen in a mid-game state with some walls already placed."""
        rng = random.Random(42)
        state = _build_walled_state(rng, n_walls=8)
        N = 500
        elapsed, ops = _timed(
            lambda: generate_wall_moves(
                state.p1_pos, state.p2_pos,
                state.h_walls, state.v_walls,
                state.walls_remaining[state.current_player],
            ), N
        )
        _print_metric("wall movegen (midgame)", ops, "calls/s")
        assert ops > 80, f"Wall movegen (midgame) too slow: {ops:.0f} calls/s"

    def test_legal_moves_total(self):
        """Full legal_moves() = pawn + wall combined."""
        state = GameState()
        N = 500
        elapsed, ops = _timed(lambda: state.legal_moves(), N)
        _print_metric("legal_moves() (empty)", ops, "calls/s")
        assert ops > 100, f"legal_moves() too slow: {ops:.0f} calls/s"

    def test_wall_movegen_no_walls_remaining(self):
        """Wall gen with 0 walls remaining must be instant (short-circuit)."""
        state = GameState()
        state.walls_remaining[0] = 0
        N = 500_000
        elapsed, ops = _timed(
            lambda: generate_wall_moves(
                state.p1_pos, state.p2_pos, state.h_walls, state.v_walls, 0
            ), N
        )
        _print_metric("wall movegen (0 walls left)", ops, "calls/s")
        assert ops > 1_000_000, "Wall gen should short-circuit when 0 walls remaining"

    def test_move_count_sanity(self):
        """Verify the number of legal moves is in expected range."""
        state = GameState()
        moves = state.legal_moves()
        pawn_count = sum(1 for m in moves if m[0] == PAWN)
        wall_count = sum(1 for m in moves if m[0] != PAWN)
        _print_metric("Initial pawn moves", pawn_count, "moves")
        _print_metric("Initial wall moves", wall_count, "moves")
        assert 2 <= pawn_count <= 5
        # 64 slots x 2 orientations = 128, minus blocked ones
        assert wall_count > 100


# ===========================================================================
#  3. MAKE / UNMAKE — state mutation cost
# ===========================================================================

class TestMakeUnmakePerformance:
    """Make/unmake is the core loop of every search bot. It must be cheap."""

    def test_pawn_make_unmake(self):
        """Make + unmake a pawn move."""
        state = GameState()
        move = pawn_move(1, 4)
        N = 500_000
        def fn():
            undo = state.make_move(move)
            state.unmake_move(undo)
        elapsed, ops = _timed(fn, N)
        _print_metric("make+unmake pawn", ops, "ops/s")
        assert ops > 200_000, f"Make/unmake pawn too slow: {ops:.0f} ops/s"

    def test_wall_make_unmake(self):
        """Make + unmake a wall move."""
        state = GameState()
        move = wall_h(3, 3)
        N = 500_000
        def fn():
            undo = state.make_move(move)
            state.unmake_move(undo)
        elapsed, ops = _timed(fn, N)
        _print_metric("make+unmake wall", ops, "ops/s")
        assert ops > 200_000, f"Make/unmake wall too slow: {ops:.0f} ops/s"

    def test_clone_speed(self):
        """State.clone() — used by MCTS per simulation."""
        state = GameState()
        N = 500_000
        elapsed, ops = _timed(lambda: state.clone(), N)
        _print_metric("clone()", ops, "ops/s")
        assert ops > 200_000, f"Clone too slow: {ops:.0f} ops/s"


# ===========================================================================
#  4. HEURISTIC EVALUATION
# ===========================================================================

class TestHeuristicPerformance:
    """The evaluate() function is called at every leaf of minimax/alphabeta."""

    def test_evaluate_empty(self):
        state = GameState()
        N = 50_000
        elapsed, ops = _timed(
            lambda: evaluate(
                0, state.p1_pos, state.p2_pos,
                state.h_walls, state.v_walls,
                (state.walls_remaining[0], state.walls_remaining[1]),
            ), N
        )
        _print_metric("evaluate() (empty)", ops, "calls/s")
        assert ops > 8_000, f"Evaluate too slow: {ops:.0f} calls/s"

    def test_evaluate_walled(self):
        rng = random.Random(42)
        state = _build_walled_state(rng, n_walls=8)
        N = 30_000
        elapsed, ops = _timed(
            lambda: evaluate(
                state.current_player,
                state.p1_pos, state.p2_pos,
                state.h_walls, state.v_walls,
                (state.walls_remaining[0], state.walls_remaining[1]),
            ), N
        )
        _print_metric("evaluate() (walled)", ops, "calls/s")
        assert ops > 5_000, f"Evaluate (walled) too slow: {ops:.0f} calls/s"


# ===========================================================================
#  5. FULL GAME THROUGHPUT — the bottom-line number
# ===========================================================================

class TestGameThroughput:
    """End-to-end games. This is what benchmarks and tournaments actually run."""

    def test_pawn_only_games(self):
        """Pawn-only random games — engine ceiling without wall overhead."""
        rng = random.Random(42)
        N = 500
        t0 = time.perf_counter()
        total_moves = 0
        for _ in range(N):
            total_moves += _run_random_game(rng, use_walls=False)
        elapsed = time.perf_counter() - t0
        gps = N / elapsed
        avg_moves = total_moves / N
        _print_metric("Pawn-only random games", gps, "games/s")
        _print_metric("Avg moves/game (pawn-only)", avg_moves, "moves")
        assert gps > 500, f"Pawn-only games too slow: {gps:.0f} games/s"

    def test_full_random_games(self):
        """Full random games (pawn + wall moves) — the real bottleneck."""
        rng = random.Random(42)
        N = 50
        t0 = time.perf_counter()
        total_moves = 0
        for _ in range(N):
            total_moves += _run_random_game(rng, use_walls=True)
        elapsed = time.perf_counter() - t0
        gps = N / elapsed
        avg_moves = total_moves / N
        _print_metric("Full random games", gps, "games/s")
        _print_metric("Avg moves/game (full)", avg_moves, "moves")
        assert gps > 3, f"Full random games too slow: {gps:.0f} games/s"

    def test_wall_overhead_ratio(self):
        """Measure how much slower wall gen makes full games vs pawn-only.
        This ratio is the #1 diagnostic for wall-gen optimization."""
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        N_pawn = 200
        N_full = 30
        t0 = time.perf_counter()
        for _ in range(N_pawn):
            _run_random_game(rng1, use_walls=False)
        pawn_time = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(N_full):
            _run_random_game(rng2, use_walls=True)
        full_time = time.perf_counter() - t0
        pawn_gps = N_pawn / pawn_time
        full_gps = N_full / full_time
        ratio = pawn_gps / full_gps if full_gps > 0 else 999
        _print_metric("Pawn-only throughput", pawn_gps, "games/s")
        _print_metric("Full throughput", full_gps, "games/s")
        _print_metric("Wall overhead ratio", ratio, "x slower")
        # Before optimization this was ~360x. After smart wall gen it should be <200x.
        assert ratio < 300, (
            f"Wall overhead is {ratio:.0f}x — wall move generation is the bottleneck. "
            f"Check that generate_wall_moves() uses path-based pruning."
        )


# ===========================================================================
#  6. BOT PERFORMANCE — per-move decision time
# ===========================================================================

class TestBotPerformance:
    """Time a single choose_move() call for each bot. This catches regressions
    in bot logic — a minimax depth-2 call shouldn't take 10 seconds."""

    @pytest.fixture(autouse=True)
    def _setup_bots(self):
        from quoridor.bots.registry import discover_bots
        discover_bots()

    def _time_bot_move(self, bot_name: str, n_moves: int = 5, **params) -> list[float]:
        """Play n_moves and return per-move times."""
        from quoridor.bots.registry import get_bot
        bot = get_bot(bot_name, **params)
        state = GameState()
        times = []
        for _ in range(n_moves):
            if state.is_over():
                break
            t0 = time.perf_counter()
            move = bot.choose_move(state)
            times.append(time.perf_counter() - t0)
            state.make_move(move)
            # Opponent plays a random pawn move
            if not state.is_over():
                pawns = state.pawn_moves_only()
                if pawns:
                    state.make_move(random.choice(pawns))
        return times

    def test_random_bot(self):
        times = self._time_bot_move("random", n_moves=20, seed=1)
        avg_ms = statistics.mean(times) * 1000
        max_ms = max(times) * 1000
        _print_metric("random avg move", avg_ms, "ms")
        _print_metric("random max move", max_ms, "ms")
        assert avg_ms < 100, f"Random bot too slow: {avg_ms:.1f}ms avg"

    def test_greedy_bot(self):
        times = self._time_bot_move("greedy", n_moves=10)
        avg_ms = statistics.mean(times) * 1000
        max_ms = max(times) * 1000
        _print_metric("greedy avg move", avg_ms, "ms")
        _print_metric("greedy max move", max_ms, "ms")
        assert avg_ms < 2000, f"Greedy bot too slow: {avg_ms:.1f}ms avg"

    def test_minimax_d1(self):
        times = self._time_bot_move("minimax", n_moves=5, depth=1)
        avg_ms = statistics.mean(times) * 1000
        max_ms = max(times) * 1000
        _print_metric("minimax(d=1) avg move", avg_ms, "ms")
        _print_metric("minimax(d=1) max move", max_ms, "ms")
        assert avg_ms < 3000, f"Minimax d=1 too slow: {avg_ms:.1f}ms avg"

    def test_minimax_d2(self):
        times = self._time_bot_move("minimax", n_moves=3, depth=2)
        avg_ms = statistics.mean(times) * 1000
        max_ms = max(times) * 1000
        _print_metric("minimax(d=2) avg move", avg_ms, "ms")
        _print_metric("minimax(d=2) max move", max_ms, "ms")
        assert avg_ms < 30_000, f"Minimax d=2 too slow: {avg_ms:.1f}ms avg"

    def test_alphabeta_d2(self):
        times = self._time_bot_move("alphabeta", n_moves=3, depth=2)
        avg_ms = statistics.mean(times) * 1000
        max_ms = max(times) * 1000
        _print_metric("alphabeta(d=2) avg move", avg_ms, "ms")
        _print_metric("alphabeta(d=2) max move", max_ms, "ms")
        assert avg_ms < 15_000, f"AlphaBeta d=2 too slow: {avg_ms:.1f}ms avg"

    def test_alphabeta_faster_than_minimax(self):
        """Alpha-beta pruning should beat plain minimax at same depth."""
        mm_times = self._time_bot_move("minimax", n_moves=3, depth=2)
        ab_times = self._time_bot_move("alphabeta", n_moves=3, depth=2)
        mm_avg = statistics.mean(mm_times)
        ab_avg = statistics.mean(ab_times)
        speedup = mm_avg / ab_avg if ab_avg > 0 else 0
        _print_metric("minimax(d=2) avg", mm_avg * 1000, "ms")
        _print_metric("alphabeta(d=2) avg", ab_avg * 1000, "ms")
        _print_metric("AB speedup over minimax", speedup, "x")
        # Alpha-beta should be at least a bit faster due to pruning
        assert speedup > 0.8, (
            f"AlphaBeta is not faster than minimax ({speedup:.1f}x) — "
            f"check move ordering and pruning logic"
        )

    def test_mcts_100_sims(self):
        times = self._time_bot_move("mcts", n_moves=3, simulations=100, seed=1)
        avg_ms = statistics.mean(times) * 1000
        max_ms = max(times) * 1000
        _print_metric("mcts(100 sims) avg move", avg_ms, "ms")
        _print_metric("mcts(100 sims) max move", max_ms, "ms")
        assert avg_ms < 5000, f"MCTS 100 sims too slow: {avg_ms:.1f}ms avg"


# ===========================================================================
#  7. SCALING — how performance degrades as game progresses
# ===========================================================================

class TestScaling:
    """Catch performance cliffs. Some operations get dramatically slower
    as walls accumulate — this detects that."""

    def test_bfs_scaling_with_walls(self):
        """BFS should not get dramatically slower with more walls."""
        start = sq(0, 4)
        rng = random.Random(42)

        # Measure BFS at 0, 4, 8 walls
        results = []
        state = GameState()
        for n_walls in [0, 4, 8]:
            while bin(state.h_walls | state.v_walls).count("1") < n_walls:
                wi = rng.randint(0, NUM_WALL_SLOTS - 1)
                bit = 1 << wi
                if not (state.h_walls & bit) and not (state.v_walls & bit):
                    if rng.random() < 0.5:
                        if bfs_path_exists(start, 8, state.h_walls | bit, state.v_walls):
                            state.h_walls |= bit
                    else:
                        if bfs_path_exists(start, 8, state.h_walls, state.v_walls | bit):
                            state.v_walls |= bit

            N = 20_000
            _, ops = _timed(
                lambda: bfs_shortest_path(start, 8, state.h_walls, state.v_walls), N
            )
            results.append((n_walls, ops))
            _print_metric(f"BFS shortest ({n_walls} walls)", ops, "calls/s")

        # Should not degrade more than 3x from 0 to 8 walls
        if results[0][1] > 0:
            ratio = results[0][1] / results[-1][1]
            _print_metric("Degradation 0->8 walls", ratio, "x")
            assert ratio < 5.0, (
                f"BFS degrades {ratio:.1f}x with 8 walls — "
                f"check if path traversal has quadratic behavior"
            )

    def test_legal_moves_scaling(self):
        """legal_moves() with increasing wall count."""
        rng = random.Random(42)
        results = []
        for n_walls in [0, 4, 8]:
            state = _build_walled_state(rng, n_walls=n_walls)
            N = 100
            _, ops = _timed(lambda: state.legal_moves(), N)
            results.append((n_walls, ops))
            _print_metric(f"legal_moves ({n_walls} walls)", ops, "calls/s")

        if results[0][1] > 0:
            ratio = results[0][1] / results[-1][1]
            _print_metric("Degradation 0->8 walls", ratio, "x")


# ===========================================================================
#  8. MEMORY — catch accidental allocations in hot loops
# ===========================================================================

class TestMemoryEfficiency:
    """Verify that make/unmake doesn't leak state and that the UndoRecord
    is lightweight."""

    def test_make_unmake_no_state_growth(self):
        """After N make/unmake pairs, state should be identical to start."""
        state = GameState()
        rng = random.Random(42)
        snap = (
            state.h_walls, state.v_walls,
            state.positions[0], state.positions[1],
            state.walls_remaining[0], state.walls_remaining[1],
            state.current_player, state.move_count,
        )
        for _ in range(1000):
            moves = state.legal_moves()
            if not moves or state.is_over():
                break
            move = rng.choice(moves)
            undo = state.make_move(move)
            state.unmake_move(undo)
        after = (
            state.h_walls, state.v_walls,
            state.positions[0], state.positions[1],
            state.walls_remaining[0], state.walls_remaining[1],
            state.current_player, state.move_count,
        )
        assert snap == after, "State drifted after make/unmake cycles"

    def test_undo_record_is_lightweight(self):
        """UndoRecord should use __slots__ — no __dict__."""
        from quoridor.core.state import UndoRecord
        move = pawn_move(1, 4)
        undo = UndoRecord(move, 0, 0, (4, 76), (10, 10), 0)
        assert not hasattr(undo, "__dict__"), "UndoRecord should use __slots__"


# ===========================================================================
#  9. SMART WALL GEN VALIDATION — correctness of the optimization
# ===========================================================================

class TestSmartWallGenCorrectness:
    """The smart wall gen skips BFS for walls far from the shortest path.
    This must not generate illegal moves (walls that fully block a player)."""

    def test_all_generated_walls_are_legal(self):
        """Every wall move from generate_wall_moves must leave both players
        with a path to their goal."""
        rng = random.Random(42)
        for _ in range(20):
            state = _build_walled_state(rng, n_walls=rng.randint(0, 10))
            walls = generate_wall_moves(
                state.p1_pos, state.p2_pos,
                state.h_walls, state.v_walls,
                state.walls_remaining[state.current_player],
            )
            for wm in walls:
                if wm[0] == WALL_H:
                    wi = wall_idx(wm[1], wm[2])
                    new_h = state.h_walls | (1 << wi)
                    assert bfs_path_exists(state.p1_pos, Player.ONE.goal_row, new_h, state.v_walls), \
                        f"Wall H({wm[1]},{wm[2]}) blocks P1"
                    assert bfs_path_exists(state.p2_pos, Player.TWO.goal_row, new_h, state.v_walls), \
                        f"Wall H({wm[1]},{wm[2]}) blocks P2"
                elif wm[0] == WALL_V:
                    wi = wall_idx(wm[1], wm[2])
                    new_v = state.v_walls | (1 << wi)
                    assert bfs_path_exists(state.p1_pos, Player.ONE.goal_row, state.h_walls, new_v), \
                        f"Wall V({wm[1]},{wm[2]}) blocks P1"
                    assert bfs_path_exists(state.p2_pos, Player.TWO.goal_row, state.h_walls, new_v), \
                        f"Wall V({wm[1]},{wm[2]}) blocks P2"

    def test_no_legal_walls_missed(self):
        """Smart wall gen must not miss any legal wall.
        Compare against brute-force (check every slot with BFS)."""
        rng = random.Random(123)
        for _ in range(10):
            state = _build_walled_state(rng, n_walls=rng.randint(0, 6))
            smart_walls = set()
            for wm in generate_wall_moves(
                state.p1_pos, state.p2_pos,
                state.h_walls, state.v_walls,
                state.walls_remaining[state.current_player],
            ):
                smart_walls.add(wm)

            # Brute force
            brute_walls = set()
            occupied = state.h_walls | state.v_walls
            for wi in range(NUM_WALL_SLOTS):
                if (1 << wi) & occupied:
                    continue
                from quoridor.core.board import wall_idx_to_rc
                wr, wc = wall_idx_to_rc(wi)
                new_h = state.h_walls | (1 << wi)
                if bfs_path_exists(state.p1_pos, Player.ONE.goal_row, new_h, state.v_walls) and \
                   bfs_path_exists(state.p2_pos, Player.TWO.goal_row, new_h, state.v_walls):
                    brute_walls.add(wall_h(wr, wc))
                new_v = state.v_walls | (1 << wi)
                if bfs_path_exists(state.p1_pos, Player.ONE.goal_row, state.h_walls, new_v) and \
                   bfs_path_exists(state.p2_pos, Player.TWO.goal_row, state.h_walls, new_v):
                    brute_walls.add(wall_v(wr, wc))

            missing = brute_walls - smart_walls
            extra = smart_walls - brute_walls
            assert not missing, f"Smart gen missed {len(missing)} legal walls: {list(missing)[:5]}"
            assert not extra, f"Smart gen produced {len(extra)} illegal walls: {list(extra)[:5]}"


# ===========================================================================
#  10. SUMMARY — quick diagnostic dashboard
# ===========================================================================

class TestPerformanceSummary:
    """Runs a compact suite and prints a one-screen dashboard.
    Use this as the first thing to check after any engine change."""

    def test_dashboard(self):
        print("\n" + "=" * 65)
        print("  QUORIDOR PERFORMANCE DASHBOARD")
        print("=" * 65)

        state = GameState()
        rng = random.Random(42)

        # BFS
        N = 30_000
        _, bfs_ops = _timed(lambda: bfs_shortest_path(state.p1_pos, 8, 0, 0), N)
        _print_metric("BFS shortest (empty board)", bfs_ops, "calls/s")

        # Move gen
        N = 300
        _, movegen_ops = _timed(lambda: state.legal_moves(), N)
        _print_metric("legal_moves() (empty)", movegen_ops, "calls/s")

        # Make/unmake
        move = pawn_move(1, 4)
        N = 300_000
        def mu():
            undo = state.make_move(move)
            state.unmake_move(undo)
        _, mu_ops = _timed(mu, N)
        _print_metric("make+unmake pawn", mu_ops, "ops/s")

        # Pawn-only games
        N = 200
        t0 = time.perf_counter()
        for _ in range(N):
            _run_random_game(rng, use_walls=False)
        pawn_gps = N / (time.perf_counter() - t0)
        _print_metric("Pawn-only random games", pawn_gps, "games/s")

        # Full games
        N = 20
        t0 = time.perf_counter()
        for _ in range(N):
            _run_random_game(rng, use_walls=True)
        full_gps = N / (time.perf_counter() - t0)
        _print_metric("Full random games", full_gps, "games/s")

        ratio = pawn_gps / full_gps if full_gps > 0 else 999
        _print_metric("Wall overhead ratio", ratio, "x")

        print("-" * 65)
        if ratio > 200:
            print("  >>> BOTTLENECK: wall move generation (wall overhead > 200x)")
            print("      Focus on generate_wall_moves() and BFS call count")
        elif bfs_ops < 10_000:
            print("  >>> BOTTLENECK: BFS pathfinding is slow")
            print("      Check _neighbors inlining and visited-array allocation")
        elif movegen_ops < 50:
            print("  >>> BOTTLENECK: legal_moves() overall throughput")
            print("      Profile generate_wall_moves vs generate_pawn_moves")
        else:
            print("  >>> No obvious bottleneck — performance looks healthy")
        print("=" * 65)
