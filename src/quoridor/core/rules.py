from quoridor.core.moves import Move, MoveType, Player, pawn_move, wall_h, wall_v
from quoridor.core.board import (
    BOARD_SIZE,
    NUM_WALL_SLOTS,
    NEIGHBOR_DIR,
    EDGE_H_MASK,
    EDGE_V_MASK,
    sq,
    sq_to_rc,
    wall_idx,
    wall_idx_to_rc,
    is_edge_blocked,
)
from quoridor.core.pathfind import bfs_path_exists


def check_winner(p1_pos: int, p2_pos: int) -> int:
    if sq_to_rc(p1_pos)[0] == Player.ONE.goal_row:
        return 0
    if sq_to_rc(p2_pos)[0] == Player.TWO.goal_row:
        return 1
    return -1


def generate_pawn_moves(
    pos: int,
    opp_pos: int,
    h_walls: int,
    v_walls: int,
) -> list[Move]:
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
            r, c = sq_to_rc(ns)
            moves.append(pawn_move(r, c))
        else:
            opp_hm = EDGE_H_MASK[opp_pos]
            opp_vm = EDGE_V_MASK[opp_pos]
            jump_di = di
            jump_target = opp_nd[jump_di]
            if jump_target >= 0:
                oh = opp_hm[jump_di]
                ov = opp_vm[jump_di]
                if (not oh or not (oh & h_walls)) and (not ov or not (ov & v_walls)):
                    r, c = sq_to_rc(jump_target)
                    moves.append(pawn_move(r, c))
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
                        r, c = sq_to_rc(lt)
                        moves.append(pawn_move(r, c))
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
                    r, c = sq_to_rc(lt)
                    moves.append(pawn_move(r, c))
    return moves


def _wall_valid(wi: int, h_walls: int, v_walls: int) -> bool:
    if (1 << wi) & (h_walls | v_walls):
        return False
    return True


def generate_wall_moves(
    p1_pos: int,
    p2_pos: int,
    h_walls: int,
    v_walls: int,
    walls_remaining: int,
) -> list[Move]:
    if walls_remaining <= 0:
        return []
    moves = []
    for wi in range(NUM_WALL_SLOTS):
        if (1 << wi) & (h_walls | v_walls):
            continue
        wr, wc = wall_idx_to_rc(wi)
        new_h = h_walls | (1 << wi)
        if bfs_path_exists(p1_pos, Player.ONE.goal_row, new_h, v_walls) and \
           bfs_path_exists(p2_pos, Player.TWO.goal_row, new_h, v_walls):
            moves.append(wall_h(wr, wc))
        new_v = v_walls | (1 << wi)
        if bfs_path_exists(p1_pos, Player.ONE.goal_row, h_walls, new_v) and \
           bfs_path_exists(p2_pos, Player.TWO.goal_row, h_walls, new_v):
            moves.append(wall_v(wr, wc))
    return moves


def generate_all_moves(
    current_player: int,
    positions: tuple[int, int],
    h_walls: int,
    v_walls: int,
    walls_remaining: tuple[int, int],
) -> list[Move]:
    pos = positions[current_player]
    opp = positions[1 - current_player]
    wr = walls_remaining[current_player]
    pawn_moves = generate_pawn_moves(pos, opp, h_walls, v_walls)
    wall_moves = generate_wall_moves(positions[0], positions[1], h_walls, v_walls, wr)
    return pawn_moves + wall_moves
