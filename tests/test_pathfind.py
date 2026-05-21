import pytest
from quoridor.core.board import sq, wall_idx, BOARD_SIZE
from quoridor.core.pathfind import bfs_path_exists, bfs_shortest_path, astar_shortest_path
from quoridor.core.moves import Player


class TestBFSPathExists:
    def test_same_row_is_goal(self):
        assert bfs_path_exists(sq(8, 4), 8, 0, 0)

    def test_empty_board_path_exists(self):
        assert bfs_path_exists(sq(0, 4), 8, 0, 0)
        assert bfs_path_exists(sq(8, 4), 0, 0, 0)

    def test_full_row_block_no_path(self):
        h_walls = 0
        for c in range(BOARD_SIZE - 1):
            h_walls |= 1 << wall_idx(3, c)
        start = sq(0, 4)
        assert not bfs_path_exists(start, 8, h_walls, 0)

    def test_partial_row_block_has_path(self):
        h_walls = 0
        for c in range(BOARD_SIZE - 2):
            h_walls |= 1 << wall_idx(3, c)
        assert bfs_path_exists(sq(0, 4), 8, h_walls, 0)


class TestBFSShortestPath:
    def test_empty_board_straight(self):
        assert bfs_shortest_path(sq(0, 4), 8, 0, 0) == 8
        assert bfs_shortest_path(sq(4, 4), 8, 0, 0) == 4

    def test_already_at_goal(self):
        assert bfs_shortest_path(sq(8, 3), 8, 0, 0) == 0


class TestAStarShortestPath:
    def test_matches_bfs_empty(self):
        for r in range(0, BOARD_SIZE, 2):
            for c in range(0, BOARD_SIZE, 2):
                s = sq(r, c)
                d_bfs = bfs_shortest_path(s, 8, 0, 0)
                d_astar = astar_shortest_path(s, 8, 0, 0)
                assert d_bfs == d_astar, f"Mismatch at ({r},{c}): bfs={d_bfs} astar={d_astar}"

    def test_with_walls(self):
        h_walls = 0
        h_walls |= 1 << wall_idx(0, 3)
        h_walls |= 1 << wall_idx(0, 4)
        start = sq(0, 4)
        d_bfs = bfs_shortest_path(start, 8, h_walls, 0)
        d_astar = astar_shortest_path(start, 8, h_walls, 0)
        assert d_bfs == d_astar
