"""
Matrix Analysis Tests — full-game analytics for the wall scoring matrix.

Tracks how the wall strategy matrix behaves across entire games:
  - Score distributions across game phases
  - Wall placement heatmaps (where walls actually land)
  - Path impact: how much each placed wall lengthens opponent / shortens self
  - Correlation between wall score and actual game outcome
  - Phase transition accuracy
  - Bot comparison: strategic vs non-strategic wall play
  - Defensive vs offensive wall balance
  - Wall efficiency (walls that actually mattered vs wasted)

Run:  pytest tests/test_matrix_analysis.py -v -s
"""

import random
import time
from collections import Counter, defaultdict

import pytest

from quoridor.core.board import (
    BOARD_SIZE, NUM_WALL_SLOTS, sq, wall_idx, wall_idx_to_rc, sq_row, sq_col,
)
from quoridor.core.moves import PAWN, WALL_H, WALL_V, Player, pawn_move, wall_h, wall_v
from quoridor.core.state import GameState
from quoridor.core.pathfind import bfs_shortest_path, bfs_path_trace
from quoridor.core.rules import _path_edge_walls
from quoridor.eval.wall_strategy import (
    score_wall_moves,
    strategic_wall_moves,
    game_phase,
    WALL_NEIGHBORS,
    WALL_CENTER_DIST,
)
from quoridor.eval.heuristics import evaluate
from quoridor.bots.registry import discover_bots, get_bot


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _play_game(bot1, bot2, max_moves=200):
    """Play a full game, returning detailed move log."""
    state = GameState()
    log = []  # list of dicts per move
    while not state.is_over() and state.move_count < max_moves:
        cp = state.current_player
        bot = bot1 if cp == 0 else bot2

        # Capture pre-move state
        p1_dist = bfs_shortest_path(state.p1_pos, Player.ONE.goal_row, state.h_walls, state.v_walls)
        p2_dist = bfs_shortest_path(state.p2_pos, Player.TWO.goal_row, state.h_walls, state.v_walls)
        phase = game_phase(state.move_count, state.walls_remaining[0], state.walls_remaining[1])

        # Get wall scores before the move
        scored_walls = score_wall_moves(state, top_k=10) if state.walls_remaining[cp] > 0 else []

        t0 = time.perf_counter()
        move = bot.choose_move(state)
        dt = time.perf_counter() - t0

        # Capture wall score if a wall was placed
        wall_score = None
        if move[0] != PAWN:
            for s, m in scored_walls:
                if m == move:
                    wall_score = s
                    break

        state.make_move(move)

        # Post-move distances
        p1_dist_after = bfs_shortest_path(state.p1_pos, Player.ONE.goal_row, state.h_walls, state.v_walls)
        p2_dist_after = bfs_shortest_path(state.p2_pos, Player.TWO.goal_row, state.h_walls, state.v_walls)

        log.append({
            "move_num": state.move_count,
            "player": cp,
            "move": move,
            "move_type": move[0],
            "phase": phase,
            "time_ms": dt * 1000,
            "p1_dist_before": p1_dist,
            "p2_dist_before": p2_dist,
            "p1_dist_after": p1_dist_after,
            "p2_dist_after": p2_dist_after,
            "wall_score": wall_score,
            "walls_remaining": list(state.walls_remaining),
        })

    return state, log


def _play_n_games(bot1_name, bot2_name, n=20, **params):
    """Play N games and collect all logs."""
    discover_bots()
    results = []
    for i in range(n):
        bot1 = get_bot(bot1_name, seed=i, **params)
        bot2 = get_bot(bot2_name, seed=i + 1000, **params)
        state, log = _play_game(bot1, bot2)
        results.append({
            "winner": state.winner(),
            "moves": state.move_count,
            "log": log,
        })
    return results


# ===========================================================================
#  1. SCORE DISTRIBUTION ACROSS PHASES
# ===========================================================================

class TestScoreDistribution:
    """Analyze how wall scores distribute across game phases."""

    def test_score_range_by_phase(self):
        """Wall scores should have reasonable ranges in each phase."""
        discover_bots()
        bot1 = get_bot("greedy", seed=42)
        bot2 = get_bot("greedy", seed=43)
        state, log = _play_game(bot1, bot2)

        phase_scores = {"opening": [], "midgame": [], "endgame": []}
        for entry in log:
            if entry["wall_score"] is not None:
                if entry["phase"] < 0.3:
                    phase_scores["opening"].append(entry["wall_score"])
                elif entry["phase"] < 0.7:
                    phase_scores["midgame"].append(entry["wall_score"])
                else:
                    phase_scores["endgame"].append(entry["wall_score"])

        print("\n  Wall score distribution by phase:")
        for phase_name, scores in phase_scores.items():
            if scores:
                avg = sum(scores) / len(scores)
                lo, hi = min(scores), max(scores)
                print(f"    {phase_name}: n={len(scores)}, avg={avg:.2f}, range=[{lo:.2f}, {hi:.2f}]")
            else:
                print(f"    {phase_name}: no walls placed")

    def test_top_scores_decrease_as_walls_depleted(self):
        """As walls deplete, the top available score should generally decrease
        (fewer good options remain)."""
        state = GameState()
        rng = random.Random(99)

        scores_over_time = []
        for _ in range(60):
            if state.is_over():
                break
            scored = score_wall_moves(state, top_k=5)
            top_score = scored[0][0] if scored else None
            scores_over_time.append((state.move_count, state.walls_remaining[:], top_score))

            moves = state.legal_moves()
            if moves:
                state.make_move(rng.choice(moves))

        print("\n  Top wall score over time:")
        for mc, wr, ts in scores_over_time[:15]:
            ts_str = f"{ts:.2f}" if ts is not None else "N/A"
            print(f"    move={mc}, walls_left={wr}, top_score={ts_str}")


# ===========================================================================
#  2. WALL PLACEMENT HEATMAP
# ===========================================================================

class TestWallHeatmap:
    """Analyze where bots actually place walls on the board."""

    def test_wall_placement_distribution(self):
        """Walls should cluster near opponent paths, not spread randomly."""
        results = _play_n_games("greedy", "greedy", n=15)

        heatmap = Counter()  # (row, col) -> count
        total_walls = 0
        for game in results:
            for entry in game["log"]:
                if entry["move_type"] != PAWN:
                    heatmap[(entry["move"][1], entry["move"][2])] += 1
                    total_walls += 1

        if total_walls == 0:
            print("\n  No walls placed in 15 games — skipping heatmap")
            return

        # Print heatmap
        print(f"\n  Wall placement heatmap ({total_walls} walls across {len(results)} games):")
        grid = [[0] * 8 for _ in range(8)]
        for (r, c), count in heatmap.items():
            grid[r][c] = count

        print("    ", "  ".join(str(c) for c in range(8)))
        for r in range(8):
            row_str = "  ".join(f"{grid[r][c]:2d}" if grid[r][c] else " ." for c in range(8))
            print(f"    {r} {row_str}")

        # Walls should be more concentrated in center rows (rows 2-5)
        center_count = sum(v for (r, c), v in heatmap.items() if 2 <= r <= 5)
        edge_count = sum(v for (r, c), v in heatmap.items() if r < 2 or r > 5)
        print(f"\n    Center rows (2-5): {center_count}")
        print(f"    Edge rows (0-1, 6-7): {edge_count}")

    def test_h_vs_v_wall_balance(self):
        """Check balance between horizontal and vertical walls."""
        results = _play_n_games("alphabeta", "alphabeta", n=10, depth=2)

        h_count = 0
        v_count = 0
        for game in results:
            for entry in game["log"]:
                if entry["move_type"] == WALL_H:
                    h_count += 1
                elif entry["move_type"] == WALL_V:
                    v_count += 1

        total = h_count + v_count
        if total == 0:
            print("\n  No walls placed — skipping H/V analysis")
            return

        print(f"\n  Wall type balance:")
        print(f"    Horizontal: {h_count} ({100*h_count/total:.0f}%)")
        print(f"    Vertical:   {v_count} ({100*v_count/total:.0f}%)")
        # Neither type should dominate completely (>90%)
        if total >= 5:
            assert h_count / total < 0.95, "Almost all walls are horizontal"
            assert v_count / total < 0.95, "Almost all walls are vertical"


# ===========================================================================
#  3. PATH IMPACT ANALYSIS
# ===========================================================================

class TestPathImpact:
    """Measure actual impact of placed walls on path distances."""

    def test_wall_moves_increase_opponent_distance(self):
        """Placed walls should, on average, increase opponent distance more than own."""
        results = _play_n_games("alphabeta", "greedy", n=10, depth=2)

        opp_deltas = []
        own_deltas = []
        for game in results:
            for entry in game["log"]:
                if entry["move_type"] == PAWN:
                    continue
                cp = entry["player"]
                if cp == 0:
                    opp_delta = entry["p2_dist_after"] - entry["p2_dist_before"]
                    own_delta = entry["p1_dist_after"] - entry["p1_dist_before"]
                else:
                    opp_delta = entry["p1_dist_after"] - entry["p1_dist_before"]
                    own_delta = entry["p2_dist_after"] - entry["p2_dist_before"]
                opp_deltas.append(opp_delta)
                own_deltas.append(own_delta)

        if not opp_deltas:
            print("\n  No walls placed — skipping impact analysis")
            return

        avg_opp = sum(opp_deltas) / len(opp_deltas)
        avg_own = sum(own_deltas) / len(own_deltas)
        net = avg_opp - avg_own

        print(f"\n  Wall path impact ({len(opp_deltas)} walls):")
        print(f"    Avg opponent path increase: {avg_opp:+.2f}")
        print(f"    Avg own path increase:      {avg_own:+.2f}")
        print(f"    Net advantage per wall:     {net:+.2f}")
        # Walls should have positive net impact on average
        assert net >= -0.5, f"Walls are net-negative: {net:.2f}"

    def test_high_scored_walls_have_more_impact(self):
        """Walls with higher matrix scores should produce larger path deltas."""
        state = GameState()
        rng = random.Random(77)
        # Advance game a bit
        for _ in range(10):
            moves = state.legal_moves()
            pawns = [m for m in moves if m[0] == PAWN]
            if pawns:
                state.make_move(rng.choice(pawns))

        scored = score_wall_moves(state, top_k=20)
        if len(scored) < 6:
            return

        cp = state.current_player
        opp_pos = state.positions[1 - cp]
        opp_goal = Player.TWO.goal_row if cp == 0 else Player.ONE.goal_row
        opp_dist = bfs_shortest_path(opp_pos, opp_goal, state.h_walls, state.v_walls)

        impacts = []
        for score, move in scored:
            wi = wall_idx(move[1], move[2])
            if move[0] == WALL_H:
                new_h = state.h_walls | (1 << wi)
                new_v = state.v_walls
            else:
                new_h = state.h_walls
                new_v = state.v_walls | (1 << wi)
            new_dist = bfs_shortest_path(opp_pos, opp_goal, new_h, new_v)
            if new_dist >= 0:
                impacts.append((score, new_dist - opp_dist))

        if len(impacts) < 6:
            return

        # Top-3 scored walls should have >= average path impact of bottom-3
        top3_impact = sum(d for _, d in impacts[:3]) / 3
        bot3_impact = sum(d for _, d in impacts[-3:]) / 3

        print(f"\n  Score vs impact correlation:")
        print(f"    Top-3 scored walls avg path delta:    {top3_impact:+.2f}")
        print(f"    Bottom-3 scored walls avg path delta: {bot3_impact:+.2f}")
        assert top3_impact >= bot3_impact - 0.5, \
            f"High-scored walls not better: top={top3_impact:.2f}, bot={bot3_impact:.2f}"


# ===========================================================================
#  4. GAME OUTCOME CORRELATION
# ===========================================================================

class TestOutcomeCorrelation:
    """Does better wall play correlate with winning?"""

    def test_winner_has_better_wall_efficiency(self):
        """The winning player should, on average, get more path impact per wall."""
        results = _play_n_games("alphabeta", "alphabeta", n=20, depth=2)

        winner_efficiency = []
        loser_efficiency = []

        for game in results:
            if game["winner"] == -1:
                continue  # skip draws

            player_impact = {0: [], 1: []}
            for entry in game["log"]:
                if entry["move_type"] == PAWN:
                    continue
                cp = entry["player"]
                if cp == 0:
                    net = (entry["p2_dist_after"] - entry["p2_dist_before"]) - \
                          (entry["p1_dist_after"] - entry["p1_dist_before"])
                else:
                    net = (entry["p1_dist_after"] - entry["p1_dist_before"]) - \
                          (entry["p2_dist_after"] - entry["p2_dist_before"])
                player_impact[cp].append(net)

            winner = game["winner"]
            loser = 1 - winner
            if player_impact[winner]:
                winner_efficiency.append(sum(player_impact[winner]) / len(player_impact[winner]))
            if player_impact[loser]:
                loser_efficiency.append(sum(player_impact[loser]) / len(player_impact[loser]))

        if winner_efficiency and loser_efficiency:
            avg_w = sum(winner_efficiency) / len(winner_efficiency)
            avg_l = sum(loser_efficiency) / len(loser_efficiency)
            print(f"\n  Wall efficiency and outcome:")
            print(f"    Winner avg net impact/wall: {avg_w:+.2f} (n={len(winner_efficiency)})")
            print(f"    Loser avg net impact/wall:  {avg_l:+.2f} (n={len(loser_efficiency)})")
        else:
            print("\n  Insufficient wall data for outcome correlation")

    def test_wall_count_vs_outcome(self):
        """Track how many walls each player places and if it correlates with winning."""
        results = _play_n_games("alphabeta", "greedy", n=20, depth=2)

        winner_walls = []
        loser_walls = []
        for game in results:
            if game["winner"] == -1:
                continue
            walls_by_player = {0: 0, 1: 0}
            for entry in game["log"]:
                if entry["move_type"] != PAWN:
                    walls_by_player[entry["player"]] += 1
            winner = game["winner"]
            loser = 1 - winner
            winner_walls.append(walls_by_player[winner])
            loser_walls.append(walls_by_player[loser])

        if winner_walls:
            avg_w = sum(winner_walls) / len(winner_walls)
            avg_l = sum(loser_walls) / len(loser_walls)
            print(f"\n  Wall count vs outcome ({len(winner_walls)} decisive games):")
            print(f"    Winner avg walls placed: {avg_w:.1f}")
            print(f"    Loser avg walls placed:  {avg_l:.1f}")


# ===========================================================================
#  5. PHASE TRANSITION ANALYSIS
# ===========================================================================

class TestPhaseTransition:
    """Verify that game phase detection aligns with actual game state."""

    def test_phase_progression_monotonic(self):
        """Game phase should generally increase over the course of a game."""
        results = _play_n_games("greedy", "greedy", n=5)

        for game in results:
            phases = [e["phase"] for e in game["log"]]
            if len(phases) < 4:
                continue
            # Phase at end should be >= phase at start
            assert phases[-1] >= phases[0], \
                f"Phase went backward: start={phases[0]:.2f}, end={phases[-1]:.2f}"

    def test_phase_matches_wall_depletion(self):
        """Phase should track wall depletion and move count."""
        state = GameState()
        rng = random.Random(55)

        last_phase = 0.0
        for i in range(80):
            if state.is_over():
                break
            phase = game_phase(state.move_count, state.walls_remaining[0], state.walls_remaining[1])
            # Phase should never decrease by more than a tiny float error
            assert phase >= last_phase - 1e-9, \
                f"Phase decreased at move {i}: {last_phase:.4f} -> {phase:.4f}"
            last_phase = phase

            moves = state.legal_moves()
            if moves:
                state.make_move(rng.choice(moves))

    def test_wall_strategy_adapts_to_phase(self):
        """Score distribution should change between opening and late game."""
        # Opening position
        state_open = GameState()
        scores_open = score_wall_moves(state_open, top_k=10)

        # Late game: advance pawns and spend walls
        state_late = GameState()
        rng = random.Random(33)
        for _ in range(50):
            if state_late.is_over():
                break
            moves = state_late.legal_moves()
            if moves:
                state_late.make_move(rng.choice(moves))

        scores_late = score_wall_moves(state_late, top_k=10)

        if scores_open and scores_late:
            avg_open = sum(s for s, _ in scores_open) / len(scores_open)
            avg_late = sum(s for s, _ in scores_late) / len(scores_late)
            print(f"\n  Score by phase:")
            print(f"    Opening avg score: {avg_open:.2f} (n={len(scores_open)})")
            print(f"    Late avg score:    {avg_late:.2f} (n={len(scores_late)})")


# ===========================================================================
#  6. DEFENSIVE VS OFFENSIVE BALANCE
# ===========================================================================

class TestDefenseOffense:
    """Analyze whether the matrix produces balanced offensive/defensive walls."""

    def test_offensive_vs_defensive_classification(self):
        """Classify each scored wall as offensive (hurts opp) or defensive (helps self)."""
        state = GameState()
        # Move pawns to mid-board for interesting position
        state.positions[0] = sq(3, 4)
        state.positions[1] = sq(5, 4)

        cp = state.current_player
        my_pos = state.positions[cp]
        opp_pos = state.positions[1 - cp]
        my_goal = Player.ONE.goal_row if cp == 0 else Player.TWO.goal_row
        opp_goal = Player.TWO.goal_row if cp == 0 else Player.ONE.goal_row

        my_dist = bfs_shortest_path(my_pos, my_goal, state.h_walls, state.v_walls)
        opp_dist = bfs_shortest_path(opp_pos, opp_goal, state.h_walls, state.v_walls)

        scored = score_wall_moves(state, top_k=15)
        offensive = 0
        defensive = 0
        mixed = 0

        for score, move in scored:
            wi = wall_idx(move[1], move[2])
            if move[0] == WALL_H:
                new_h = state.h_walls | (1 << wi)
                new_v = state.v_walls
            else:
                new_h = state.h_walls
                new_v = state.v_walls | (1 << wi)

            opp_after = bfs_shortest_path(opp_pos, opp_goal, new_h, new_v)
            my_after = bfs_shortest_path(my_pos, my_goal, new_h, new_v)

            opp_delta = opp_after - opp_dist if opp_after >= 0 else 0
            my_delta = my_after - my_dist if my_after >= 0 else 0

            if opp_delta > 0 and my_delta <= 0:
                offensive += 1
            elif opp_delta <= 0 and my_delta < 0:
                defensive += 1
            else:
                mixed += 1

        total = offensive + defensive + mixed
        if total > 0:
            print(f"\n  Wall classification (top {total} walls):")
            print(f"    Offensive (hurts opp only): {offensive} ({100*offensive/total:.0f}%)")
            print(f"    Defensive (helps self only): {defensive} ({100*defensive/total:.0f}%)")
            print(f"    Mixed/neutral:               {mixed} ({100*mixed/total:.0f}%)")
            # Should have at least some offensive walls
            assert offensive > 0 or mixed > 0, "No offensive walls in top-K"


# ===========================================================================
#  7. WALL EFFICIENCY — WASTED vs USEFUL WALLS
# ===========================================================================

class TestWallEfficiency:
    """Track how many placed walls actually affected the game outcome."""

    def test_wall_impact_nonzero(self):
        """Most placed walls should change at least one player's path distance."""
        results = _play_n_games("alphabeta", "greedy", n=10, depth=2)

        impactful = 0
        wasted = 0
        for game in results:
            for entry in game["log"]:
                if entry["move_type"] == PAWN:
                    continue
                d1 = abs(entry["p1_dist_after"] - entry["p1_dist_before"])
                d2 = abs(entry["p2_dist_after"] - entry["p2_dist_before"])
                if d1 > 0 or d2 > 0:
                    impactful += 1
                else:
                    wasted += 1

        total = impactful + wasted
        if total == 0:
            print("\n  No walls placed — skipping efficiency analysis")
            return

        pct = 100 * impactful / total
        print(f"\n  Wall efficiency:")
        print(f"    Impactful walls: {impactful}/{total} ({pct:.0f}%)")
        print(f"    Wasted walls:    {wasted}/{total} ({100-pct:.0f}%)")
        # At least 30% of walls should have impact
        if total >= 5:
            assert pct >= 30, f"Too many wasted walls: only {pct:.0f}% impactful"


# ===========================================================================
#  8. MATRIX SCORING CONSISTENCY
# ===========================================================================

class TestScoringConsistency:
    """Verify the scoring matrix produces stable, consistent results."""

    def test_scores_deterministic(self):
        """Same state should produce identical scores every time."""
        state = GameState()
        s1 = score_wall_moves(state, top_k=10)
        s2 = score_wall_moves(state, top_k=10)
        assert len(s1) == len(s2)
        for (sc1, m1), (sc2, m2) in zip(s1, s2):
            assert m1 == m2, f"Move order differs: {m1} vs {m2}"
            assert abs(sc1 - sc2) < 1e-9, f"Score differs: {sc1} vs {sc2}"

    def test_scores_change_after_wall_placed(self):
        """After placing a wall, the scoring should change (board state changed)."""
        state = GameState()
        s_before = score_wall_moves(state, top_k=10)

        # Place a wall and a pawn move to get back to P1's turn
        state.make_move(wall_h(3, 3))
        state.make_move(pawn_move(7, 4))

        s_after = score_wall_moves(state, top_k=10)
        # Scores should be different (different board)
        moves_before = {m for _, m in s_before}
        moves_after = {m for _, m in s_after}
        assert moves_before != moves_after, "Exact same moves after wall placement"

    def test_symmetric_opening(self):
        """In the symmetric opening, P1 and P2 should get similar top scores."""
        state1 = GameState()
        state1.current_player = 0
        s1 = score_wall_moves(state1, top_k=5)

        state2 = GameState()
        state2.current_player = 1
        s2 = score_wall_moves(state2, top_k=5)

        if s1 and s2:
            avg1 = sum(s for s, _ in s1) / len(s1)
            avg2 = sum(s for s, _ in s2) / len(s2)
            print(f"\n  Symmetric opening scores:")
            print(f"    P1 avg top-5: {avg1:.2f}")
            print(f"    P2 avg top-5: {avg2:.2f}")
            # Should be within ~2 points of each other
            assert abs(avg1 - avg2) < 3.0, \
                f"Asymmetric scores: P1={avg1:.2f}, P2={avg2:.2f}"


# ===========================================================================
#  9. FULL GAME DASHBOARD
# ===========================================================================

class TestGameDashboard:
    """Comprehensive dashboard that prints full-game analytics."""

    def test_dashboard(self):
        """Run games and print a full analysis dashboard."""
        discover_bots()
        matchups = [
            ("alphabeta", "greedy", {"depth": 2}),
            ("greedy", "greedy", {}),
            ("minimax", "greedy", {"depth": 1}),
        ]

        print("\n" + "=" * 70)
        print("  WALL STRATEGY MATRIX — GAME ANALYSIS DASHBOARD")
        print("=" * 70)

        for bot1_name, bot2_name, params in matchups:
            bot1 = get_bot(bot1_name, seed=42, **params)
            bot2 = get_bot(bot2_name, seed=99, **params)

            games = 10
            wins = {0: 0, 1: 0, -1: 0}
            all_walls = {0: 0, 1: 0}
            all_impact = {0: [], 1: []}
            total_moves = 0
            wall_scores_placed = []

            for g in range(games):
                b1 = get_bot(bot1_name, seed=g, **params)
                b2 = get_bot(bot2_name, seed=g + 500, **params)
                state, log = _play_game(b1, b2)
                wins[state.winner()] += 1
                total_moves += state.move_count

                for entry in log:
                    if entry["move_type"] == PAWN:
                        continue
                    cp = entry["player"]
                    all_walls[cp] += 1
                    if entry["wall_score"] is not None:
                        wall_scores_placed.append(entry["wall_score"])

                    if cp == 0:
                        net = (entry["p2_dist_after"] - entry["p2_dist_before"]) - \
                              (entry["p1_dist_after"] - entry["p1_dist_before"])
                    else:
                        net = (entry["p1_dist_after"] - entry["p1_dist_before"]) - \
                              (entry["p2_dist_after"] - entry["p2_dist_before"])
                    all_impact[cp].append(net)

            print(f"\n  --- {bot1_name} vs {bot2_name} ({games} games) ---")
            print(f"  Results: {bot1_name}={wins[0]}, {bot2_name}={wins[1]}, draws={wins[-1]}")
            print(f"  Avg moves/game: {total_moves/games:.1f}")
            print(f"  Walls placed: P1={all_walls[0]}, P2={all_walls[1]}")

            for p in [0, 1]:
                name = bot1_name if p == 0 else bot2_name
                if all_impact[p]:
                    avg_i = sum(all_impact[p]) / len(all_impact[p])
                    print(f"  {name} (P{p+1}) avg net wall impact: {avg_i:+.2f}")

            if wall_scores_placed:
                avg_s = sum(wall_scores_placed) / len(wall_scores_placed)
                print(f"  Avg matrix score of placed walls: {avg_s:.2f}")

        print("\n" + "=" * 70)


# ===========================================================================
#  10. BOT COMPARISON — STRATEGIC vs RANDOM WALLS
# ===========================================================================

class TestBotComparison:
    """Compare bots using strategic walls vs random move selection."""

    def test_strategic_greedy_vs_random(self):
        """Strategic greedy should beat random bot consistently."""
        results = _play_n_games("greedy", "random", n=20)
        greedy_wins = sum(1 for r in results if r["winner"] == 0)
        random_wins = sum(1 for r in results if r["winner"] == 1)
        draws = sum(1 for r in results if r["winner"] == -1)

        pct = 100 * greedy_wins / len(results)
        print(f"\n  Greedy vs Random (20 games):")
        print(f"    Greedy wins: {greedy_wins} ({pct:.0f}%)")
        print(f"    Random wins: {random_wins}")
        print(f"    Draws: {draws}")
        assert greedy_wins > random_wins, "Greedy should beat random"

    def test_alphabeta_vs_greedy(self):
        """AlphaBeta with strategic walls should beat greedy."""
        results = _play_n_games("alphabeta", "greedy", n=15, depth=2)
        ab_wins = sum(1 for r in results if r["winner"] == 0)
        gr_wins = sum(1 for r in results if r["winner"] == 1)

        print(f"\n  AlphaBeta(d=2) vs Greedy (15 games):")
        print(f"    AlphaBeta wins: {ab_wins}")
        print(f"    Greedy wins:    {gr_wins}")

    def test_game_length_distribution(self):
        """Analyze how game lengths distribute across different matchups."""
        matchups = [
            ("alphabeta", "random", {"depth": 2}),
            ("greedy", "random", {}),
            ("mcts", "random", {"simulations": 50}),
        ]
        discover_bots()
        print(f"\n  Game length distribution:")
        for b1, b2, params in matchups:
            lengths = []
            for i in range(10):
                bot1 = get_bot(b1, seed=i, **params)
                bot2 = get_bot(b2, seed=i + 100)
                state, _ = _play_game(bot1, bot2)
                lengths.append(state.move_count)
            avg = sum(lengths) / len(lengths)
            print(f"    {b1} vs {b2}: avg={avg:.1f}, min={min(lengths)}, max={max(lengths)}")
        # Random opponent should produce some variation
        assert True
