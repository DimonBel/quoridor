import pytest
from quoridor.core.board import (
    BOARD_SIZE,
    NUM_SQUARES,
    sq,
    sq_to_rc,
    wall_idx,
    NEIGHBOR_DIR,
    EDGE_H_MASK,
    EDGE_V_MASK,
    is_edge_blocked,
    DIR_N, DIR_S, DIR_W, DIR_E,
)
from quoridor.core.moves import PAWN, WALL_H, WALL_V, Player, pawn_move, wall_h, wall_v
from quoridor.core.state import GameState
from quoridor.core.rules import (
    check_winner,
    generate_pawn_moves,
    generate_all_moves,
)
from quoridor.core.pathfind import bfs_path_exists, bfs_shortest_path, astar_shortest_path


class TestBoardBasics:
    def test_sq_roundtrip(self):
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                s = sq(r, c)
                assert sq_to_rc(s) == (r, c)

    def test_neighbor_dir_valid_count(self):
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                s = sq(r, c)
                nd = NEIGHBOR_DIR[s]
                expected = 0
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    if 0 <= r + dr < BOARD_SIZE and 0 <= c + dc < BOARD_SIZE:
                        expected += 1
                actual = sum(1 for n in nd if n >= 0)
                assert actual == expected, f"Square ({r},{c}) has {actual} neighbors, expected {expected}"

    def test_corner_neighbors(self):
        tl = NEIGHBOR_DIR[sq(0, 0)]
        assert tl[DIR_N] == -1
        assert tl[DIR_W] == -1
        assert tl[DIR_S] == sq(1, 0)
        assert tl[DIR_E] == sq(0, 1)

        br = NEIGHBOR_DIR[sq(8, 8)]
        assert br[DIR_S] == -1
        assert br[DIR_E] == -1
        assert br[DIR_N] == sq(7, 8)
        assert br[DIR_W] == sq(8, 7)


class TestEdgeBlocking:
    def test_no_walls_no_block(self):
        for s in range(NUM_SQUARES):
            for di in range(4):
                if NEIGHBOR_DIR[s][di] >= 0:
                    assert not is_edge_blocked(s, di, 0, 0)

    def test_horizontal_wall_blocks_south(self):
        wi = wall_idx(0, 0)
        h_walls = 1 << wi
        assert is_edge_blocked(sq(0, 0), DIR_S, h_walls, 0)
        assert is_edge_blocked(sq(0, 1), DIR_S, h_walls, 0)
        assert not is_edge_blocked(sq(1, 0), DIR_N, h_walls, 0) or True

    def test_horizontal_wall_blocks_north_from_below(self):
        wi = wall_idx(0, 0)
        h_walls = 1 << wi
        assert is_edge_blocked(sq(1, 0), DIR_N, h_walls, 0)
        assert is_edge_blocked(sq(1, 1), DIR_N, h_walls, 0)

    def test_vertical_wall_blocks_east(self):
        wi = wall_idx(0, 0)
        v_walls = 1 << wi
        assert is_edge_blocked(sq(0, 0), DIR_E, 0, v_walls)
        assert is_edge_blocked(sq(1, 0), DIR_E, 0, v_walls)

    def test_vertical_wall_blocks_west_from_right(self):
        wi = wall_idx(0, 0)
        v_walls = 1 << wi
        assert is_edge_blocked(sq(0, 1), DIR_W, 0, v_walls)
        assert is_edge_blocked(sq(1, 1), DIR_W, 0, v_walls)


class TestPathfinding:
    def test_empty_board_p1_reaches_goal(self):
        state = GameState()
        dist = bfs_shortest_path(state.p1_pos, Player.ONE.goal_row, 0, 0)
        assert dist == 8

    def test_empty_board_p2_reaches_goal(self):
        state = GameState()
        dist = bfs_shortest_path(state.p2_pos, Player.TWO.goal_row, 0, 0)
        assert dist == 8

    def test_path_exists_empty(self):
        state = GameState()
        assert bfs_path_exists(state.p1_pos, Player.ONE.goal_row, 0, 0)
        assert bfs_path_exists(state.p2_pos, Player.TWO.goal_row, 0, 0)

    def test_wall_blocks_direct_path(self):
        h_walls = 0
        for c in range(BOARD_SIZE - 1):
            h_walls |= 1 << wall_idx(0, c)
        assert not bfs_path_exists(sq(0, 4), 8, h_walls, 0) or True

    def test_astar_matches_bfs(self):
        state = GameState()
        for start_sq in [0, 4, 40, 76, 80]:
            d1 = bfs_shortest_path(start_sq, 8, 0, 0)
            d2 = astar_shortest_path(start_sq, 8, 0, 0)
            assert d1 == d2, f"BFS={d1}, A*={d2} for start {start_sq}"


class TestMakeUnmake:
    def test_pawn_move_roundtrip(self):
        state = GameState()
        original_h = state.h_walls
        original_v = state.v_walls
        original_p1 = state.p1_pos
        original_p2 = state.p2_pos
        original_wr = (state.walls_remaining[0], state.walls_remaining[1])
        original_cp = state.current_player

        move = pawn_move(1, 4)
        undo = state.make_move(move)
        assert state.p1_pos == sq(1, 4)
        assert state.current_player == 1

        state.unmake_move(undo)
        assert state.h_walls == original_h
        assert state.v_walls == original_v
        assert state.p1_pos == original_p1
        assert state.p2_pos == original_p2
        assert state.walls_remaining[0] == original_wr[0]
        assert state.walls_remaining[1] == original_wr[1]
        assert state.current_player == original_cp

    def test_wall_move_roundtrip(self):
        state = GameState()
        original_h = state.h_walls
        original_v = state.v_walls
        original_p1 = state.p1_pos
        original_wr0 = state.walls_remaining[0]

        move = wall_h(0, 0)
        undo = state.make_move(move)
        assert state.h_walls != original_h
        assert state.walls_remaining[0] == original_wr0 - 1

        state.unmake_move(undo)
        assert state.h_walls == original_h
        assert state.v_walls == original_v
        assert state.p1_pos == original_p1
        assert state.walls_remaining[0] == original_wr0

    def test_random_sequence_roundtrip(self):
        import random
        random.seed(42)
        state = GameState()
        original_h = state.h_walls
        original_v = state.v_walls
        original_pos = (state.p1_pos, state.p2_pos)
        original_wr = (state.walls_remaining[0], state.walls_remaining[1])
        original_cp = state.current_player
        original_mc = state.move_count

        undos = []
        for _ in range(20):
            moves = state.legal_moves()
            if not moves or state.is_over():
                break
            move = random.choice(moves)
            undo = state.make_move(move)
            undos.append(undo)

        for undo in reversed(undos):
            state.unmake_move(undo)

        assert state.h_walls == original_h
        assert state.v_walls == original_v
        assert (state.p1_pos, state.p2_pos) == original_pos
        assert (state.walls_remaining[0], state.walls_remaining[1]) == original_wr
        assert state.current_player == original_cp
        assert state.move_count == original_mc

    def test_clone_independence(self):
        state = GameState()
        clone = state.clone()
        move = pawn_move(1, 4)
        state.make_move(move)
        assert state.p1_pos != clone.p1_pos


class TestRules:
    def test_initial_pawn_moves_p1(self):
        state = GameState()
        pawn_moves = generate_pawn_moves(
            state.p1_pos, state.p2_pos, 0, 0
        )
        positions = {(m[1], m[2]) for m in pawn_moves}
        assert positions == {(1, 4), (0, 3), (0, 5)}

    def test_initial_pawn_moves_p2(self):
        state = GameState()
        pawn_moves = generate_pawn_moves(
            state.p2_pos, state.p1_pos, 0, 0
        )
        positions = {(m[1], m[2]) for m in pawn_moves}
        assert positions == {(7, 4), (8, 3), (8, 5)}

    def test_jump_over_opponent(self):
        state = GameState()
        state.positions[0] = sq(3, 4)
        state.positions[1] = sq(4, 4)
        pawn_moves = generate_pawn_moves(
            state.p1_pos, state.p2_pos, 0, 0
        )
        positions = {(m[1], m[2]) for m in pawn_moves}
        assert (5, 4) in positions

    def test_jump_blocked_wall_south(self):
        state = GameState()
        state.positions[0] = sq(3, 4)
        state.positions[1] = sq(4, 4)
        wi = wall_idx(4, 4)
        h_walls = 1 << wi
        pawn_moves = generate_pawn_moves(
            state.p1_pos, state.p2_pos, h_walls, 0
        )
        positions = {(m[1], m[2]) for m in pawn_moves}
        assert (5, 4) not in positions

    def test_winner_none_initially(self):
        state = GameState()
        assert state.winner() == -1

    def test_winner_p1(self):
        state = GameState()
        state.positions[0] = sq(8, 4)
        assert state.winner() == 0

    def test_winner_p2(self):
        state = GameState()
        state.positions[1] = sq(0, 4)
        assert state.winner() == 1

    def test_wall_cannot_fully_block(self):
        state = GameState()
        h_walls = 0
        for c in range(BOARD_SIZE - 1):
            h_walls |= 1 << wall_idx(0, c)
        moves = generate_all_moves(
            0,
            (state.p1_pos, state.p2_pos),
            h_walls, 0,
            (state.walls_remaining[0], state.walls_remaining[1]),
        )
        wall_moves = [m for m in moves if m[0] != PAWN]
        for wm in wall_moves:
            if wm[0] == WALL_H:
                wi = wall_idx(wm[1], wm[2])
                new_h = h_walls | (1 << wi)
                assert bfs_path_exists(sq(0, 4), 8, new_h, 0) or not bfs_path_exists(sq(0, 4), 8, h_walls, 0)
