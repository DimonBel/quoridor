"""
Wall Scoring Matrix — cheaply ranks wall moves by strategic impact.

Two-pass approach:
  Pass 1: O(1) heuristic per wall (path proximity, synergy, zone, center)
  Pass 2: BFS verification on top-2K candidates only

Usage:
  walls = strategic_wall_moves(state, top_k=10)  # list of move tuples
  scored = score_wall_moves(state, top_k=10)      # list of (score, move)
"""

from quoridor.core.board import (
    BOARD_SIZE,
    NUM_WALL_SLOTS,
    _SQ_ROW,
    wall_idx,
    wall_idx_to_rc,
)
from quoridor.core.moves import PAWN, WALL_H, WALL_V, Player, wall_h, wall_v
from quoridor.core.pathfind import bfs_path_trace, bfs_shortest_path, bfs_path_exists
from quoridor.core.rules import _path_edge_walls, generate_wall_moves

_WS = BOARD_SIZE - 1  # wall grid width = 8


# ---------------------------------------------------------------------------
#  Precomputed tables (built once at import)
# ---------------------------------------------------------------------------

def _build_wall_neighbors():
    """For each wall index, its adjacent wall indices (up/down/left/right)."""
    table = [None] * NUM_WALL_SLOTS
    for wi in range(NUM_WALL_SLOTS):
        wr, wc = divmod(wi, _WS)
        neighbors = []
        if wc > 0:
            neighbors.append(wi - 1)       # left
        if wc < _WS - 1:
            neighbors.append(wi + 1)       # right
        if wr > 0:
            neighbors.append(wi - _WS)     # up
        if wr < _WS - 1:
            neighbors.append(wi + _WS)     # down
        table[wi] = tuple(neighbors)
    return tuple(table)


def _build_wall_center_dist():
    """Manhattan distance from each wall slot to the center of the wall grid."""
    cr, cc = 3.5, 3.5  # center of 8x8 wall grid
    table = [0.0] * NUM_WALL_SLOTS
    for wi in range(NUM_WALL_SLOTS):
        wr, wc = divmod(wi, _WS)
        table[wi] = abs(wr - cr) + abs(wc - cc)
    return tuple(table)


WALL_NEIGHBORS = _build_wall_neighbors()
WALL_CENTER_DIST = _build_wall_center_dist()
_MAX_CENTER_DIST = max(WALL_CENTER_DIST)


# ---------------------------------------------------------------------------
#  Game phase
# ---------------------------------------------------------------------------

def game_phase(move_count: int, walls_remaining_0: int, walls_remaining_1: int) -> float:
    """Return 0.0 (opening) to 1.0 (endgame)."""
    walls_spent = 20 - walls_remaining_0 - walls_remaining_1
    phase = (walls_spent / 12.0) * 0.5 + (move_count / 80.0) * 0.5
    if phase < 0.0:
        return 0.0
    if phase > 1.0:
        return 1.0
    return phase


# ---------------------------------------------------------------------------
#  Core scoring
# ---------------------------------------------------------------------------

def score_wall_moves(state, top_k: int = 10):
    """Score and rank wall moves. Returns [(score, move), ...] descending.

    Pass 1: cheap heuristic (no BFS per wall).
    Pass 2: BFS verification on top 2*K candidates.
    """
    cp = state.current_player
    my_pos = state.positions[cp]
    opp_pos = state.positions[1 - cp]
    my_goal = Player.ONE.goal_row if cp == 0 else Player.TWO.goal_row
    opp_goal = Player.TWO.goal_row if cp == 0 else Player.ONE.goal_row
    h_walls = state.h_walls
    v_walls = state.v_walls
    wr = state.walls_remaining[cp]

    if wr <= 0:
        return []

    # Get legal wall moves
    wall_moves = generate_wall_moves(
        state.p1_pos, state.p2_pos, h_walls, v_walls, wr,
    )
    if not wall_moves:
        return []

    # Trace paths once
    my_path = bfs_path_trace(my_pos, my_goal, h_walls, v_walls)
    opp_path = bfs_path_trace(opp_pos, opp_goal, h_walls, v_walls)

    opp_path_walls = _path_edge_walls(opp_path) if opp_path else set()
    my_path_walls = _path_edge_walls(my_path) if my_path else set()

    # Baseline distances
    my_dist = len(my_path) - 1 if my_path else 99
    opp_dist = len(opp_path) - 1 if opp_path else 99

    occupied = h_walls | v_walls
    opp_row = _SQ_ROW[opp_pos]

    phase = game_phase(state.move_count, state.walls_remaining[0], state.walls_remaining[1])

    # --- Pass 1: cheap heuristic scoring ---
    scored = []
    for move in wall_moves:
        wi = wall_idx(move[1], move[2])
        score = 0.0

        # 1. On-opponent-path bonus
        if wi in opp_path_walls:
            score += 4.0

        # 2. On-own-path penalty
        if wi in my_path_walls:
            score -= 3.0

        # 3. Synergy: count adjacent occupied walls
        for nwi in WALL_NEIGHBORS[wi]:
            if (1 << nwi) & occupied:
                score += 0.5

        # 4. Zone proximity to opponent
        wr_pos, _ = wall_idx_to_rc(wi)
        row_dist = abs(wr_pos - opp_row)
        if row_dist <= 1:
            score += 2.0
        elif row_dist <= 2:
            score += 1.0
        elif row_dist <= 3:
            score += 0.5

        # 5. Center bonus
        score += 1.0 - WALL_CENTER_DIST[wi] / _MAX_CENTER_DIST

        # 6. Phase modifier: in endgame, amplify on-path score, dampen speculative
        if phase > 0.5:
            if wi in opp_path_walls:
                score *= 1.0 + (phase - 0.5)  # up to 1.5x for on-path walls
            else:
                score *= 1.0 - (phase - 0.5) * 0.5  # down to 0.75x for off-path

        scored.append((score, move))

    # Sort descending by score
    scored.sort(key=lambda x: x[0], reverse=True)

    # --- Pass 2: BFS verification on top 2*K candidates ---
    verify_count = min(2 * top_k, len(scored))
    if verify_count == 0:
        return scored[:top_k]

    verified = []
    for i in range(verify_count):
        heuristic_score, move = scored[i]
        wi = wall_idx(move[1], move[2])

        if move[0] == WALL_H:
            new_h = h_walls | (1 << wi)
            new_v = v_walls
        else:
            new_h = h_walls
            new_v = v_walls | (1 << wi)

        opp_dist_after = bfs_shortest_path(opp_pos, opp_goal, new_h, new_v)
        my_dist_after = bfs_shortest_path(my_pos, my_goal, new_h, new_v)

        if opp_dist_after < 0 or my_dist_after < 0:
            # Shouldn't happen (walls are pre-validated) but be safe
            continue

        delta = (opp_dist_after - opp_dist) - (my_dist_after - my_dist)
        final_score = 0.3 * heuristic_score + 0.7 * delta
        verified.append((final_score, move))

    # Also include remaining unverified walls (with heuristic score discounted)
    for i in range(verify_count, len(scored)):
        heuristic_score, move = scored[i]
        verified.append((heuristic_score * 0.3, move))

    verified.sort(key=lambda x: x[0], reverse=True)
    return verified[:top_k]


def strategic_wall_moves(state, top_k: int = 10) -> list:
    """Return top-K wall moves ranked by strategic value. Just the moves, no scores."""
    if state.walls_remaining[state.current_player] <= 0:
        return []
    scored = score_wall_moves(state, top_k=top_k)
    return [move for _, move in scored]
