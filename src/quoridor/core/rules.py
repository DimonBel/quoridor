from quoridor.core.moves import PAWN, WALL_H, WALL_V, pawn_move, wall_h, wall_v, Player
from quoridor.core.board import (
    BOARD_SIZE,
    NUM_WALL_SLOTS,
    NEIGHBOR_DIR,
    EDGE_H_MASK,
    EDGE_V_MASK,
    _SQ_ROW,
    sq,
    sq_to_rc,
    wall_idx,
    wall_idx_to_rc,
    is_edge_blocked,
)
from quoridor.core.pathfind import bfs_path_exists, bfs_path_trace


def check_winner(p1_pos: int, p2_pos: int) -> int:
    if _SQ_ROW[p1_pos] == Player.ONE.goal_row:
        return 0
    if _SQ_ROW[p2_pos] == Player.TWO.goal_row:
        return 1
    return -1


def generate_pawn_moves(
    pos: int,
    opp_pos: int,
    h_walls: int,
    v_walls: int,
) -> list:
    moves = []
    nd = NEIGHBOR_DIR[pos]
    hm = EDGE_H_MASK[pos]
    vm = EDGE_V_MASK[pos]
    opp_nd = NEIGHBOR_DIR[opp_pos]

    for di in range(4):
        ns = nd[di]
        if ns < 0:
            continue
        h = hm[di]
        if h and h & h_walls:
            continue
        v = vm[di]
        if v and v & v_walls:
            continue
        if ns != opp_pos:
            moves.append(pawn_move(_SQ_ROW[ns], ns % BOARD_SIZE))
        else:
            opp_hm = EDGE_H_MASK[opp_pos]
            opp_vm = EDGE_V_MASK[opp_pos]
            jump_target = opp_nd[di]
            if jump_target >= 0:
                oh = opp_hm[di]
                ov = opp_vm[di]
                if (not oh or not (oh & h_walls)) and (not ov or not (ov & v_walls)):
                    moves.append(pawn_move(_SQ_ROW[jump_target], jump_target % BOARD_SIZE))
                else:
                    for ldi in range(4):
                        if ldi == di:
                            continue
                        lt = opp_nd[ldi]
                        if lt < 0 or lt == pos:
                            continue
                        lh = opp_hm[ldi]
                        lv = opp_vm[ldi]
                        if lh and lh & h_walls:
                            continue
                        if lv and lv & v_walls:
                            continue
                        moves.append(pawn_move(_SQ_ROW[lt], lt % BOARD_SIZE))
            else:
                for ldi in range(4):
                    if ldi == di:
                        continue
                    lt = opp_nd[ldi]
                    if lt < 0 or lt == pos:
                        continue
                    lh = opp_hm[ldi]
                    lv = opp_vm[ldi]
                    if lh and lh & h_walls:
                        continue
                    if lv and lv & v_walls:
                        continue
                    moves.append(pawn_move(_SQ_ROW[lt], lt % BOARD_SIZE))
    return moves


def _path_edge_walls(path: list[int]) -> set[int]:
    """Given a path (list of squares), return wall indices that block any edge on the path."""
    wall_set = set()
    for i in range(len(path) - 1):
        s = path[i]
        ns = path[i + 1]
        # Figure out direction from s to ns
        diff = ns - s
        if diff == -BOARD_SIZE:
            di = 0  # N
        elif diff == BOARD_SIZE:
            di = 1  # S
        elif diff == -1:
            di = 2  # W
        elif diff == 1:
            di = 3  # E
        else:
            continue
        # All wall indices in the h_mask and v_mask for this edge
        hm = EDGE_H_MASK[s][di]
        vm = EDGE_V_MASK[s][di]
        mask = hm | vm
        while mask:
            wi = (mask & -mask).bit_length() - 1
            wall_set.add(wi)
            mask &= mask - 1
    return wall_set


def generate_wall_moves(
    p1_pos: int,
    p2_pos: int,
    h_walls: int,
    v_walls: int,
    walls_remaining: int,
) -> list:
    if walls_remaining <= 0:
        return []

    occupied = h_walls | v_walls
    moves = []

    # Get shortest paths for both players — we only need to validate walls
    # that could potentially block one of these paths
    p1_path = bfs_path_trace(p1_pos, Player.ONE.goal_row, h_walls, v_walls)
    p2_path = bfs_path_trace(p2_pos, Player.TWO.goal_row, h_walls, v_walls)

    # Collect wall indices that touch the paths
    critical_walls = set()
    if p1_path:
        critical_walls |= _path_edge_walls(p1_path)
    if p2_path:
        critical_walls |= _path_edge_walls(p2_path)

    for wi in range(NUM_WALL_SLOTS):
        if (1 << wi) & occupied:
            continue

        # Try horizontal wall
        new_h = h_walls | (1 << wi)
        # Only need path validation if this wall touches a critical edge
        if wi in critical_walls:
            if bfs_path_exists(p1_pos, Player.ONE.goal_row, new_h, v_walls) and \
               bfs_path_exists(p2_pos, Player.TWO.goal_row, new_h, v_walls):
                wr, wc = wall_idx_to_rc(wi)
                moves.append(wall_h(wr, wc))
        else:
            wr, wc = wall_idx_to_rc(wi)
            moves.append(wall_h(wr, wc))

        # Try vertical wall
        new_v = v_walls | (1 << wi)
        if wi in critical_walls:
            if bfs_path_exists(p1_pos, Player.ONE.goal_row, h_walls, new_v) and \
               bfs_path_exists(p2_pos, Player.TWO.goal_row, h_walls, new_v):
                wr, wc = wall_idx_to_rc(wi)
                moves.append(wall_v(wr, wc))
        else:
            wr, wc = wall_idx_to_rc(wi)
            moves.append(wall_v(wr, wc))

    return moves


def generate_all_moves(
    current_player: int,
    positions: tuple[int, int],
    h_walls: int,
    v_walls: int,
    walls_remaining: tuple[int, int],
) -> list:
    pos = positions[current_player]
    opp = positions[1 - current_player]
    wr = walls_remaining[current_player]
    pawn_moves = generate_pawn_moves(pos, opp, h_walls, v_walls)
    wall_moves = generate_wall_moves(positions[0], positions[1], h_walls, v_walls, wr)
    return pawn_moves + wall_moves
