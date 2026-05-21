from quoridor.core.board import sq_to_rc, BOARD_SIZE
from quoridor.core.moves import Player
from quoridor.core.pathfind import bfs_shortest_path


def path_diff(
    p1_pos: int,
    p2_pos: int,
    h_walls: int,
    v_walls: int,
) -> float:
    p1_dist = bfs_shortest_path(p1_pos, Player.ONE.goal_row, h_walls, v_walls)
    p2_dist = bfs_shortest_path(p2_pos, Player.TWO.goal_row, h_walls, v_walls)
    if p1_dist < 0:
        return -1000.0
    if p2_dist < 0:
        return 1000.0
    return float(p2_dist - p1_dist)


def evaluate(
    current_player: int,
    p1_pos: int,
    p2_pos: int,
    h_walls: int,
    v_walls: int,
    walls_remaining: tuple[int, int],
    path_weight: float = 2.0,
    wall_weight: float = 0.5,
    position_weight: float = 0.1,
) -> float:
    p1_dist = bfs_shortest_path(p1_pos, Player.ONE.goal_row, h_walls, v_walls)
    p2_dist = bfs_shortest_path(p2_pos, Player.TWO.goal_row, h_walls, v_walls)
    if p1_dist < 0:
        return -10000.0 if current_player == 0 else 10000.0
    if p2_dist < 0:
        return 10000.0 if current_player == 0 else -10000.0
    if p1_dist == 0:
        return 10000.0 if current_player == 0 else -10000.0
    if p2_dist == 0:
        return -10000.0 if current_player == 0 else 10000.0

    score = path_weight * (p2_dist - p1_dist)
    r1, _ = sq_to_rc(p1_pos)
    r2, _ = sq_to_rc(p2_pos)
    score += position_weight * (r1 - (8 - r2))
    score += wall_weight * (walls_remaining[0] - walls_remaining[1])
    if current_player == 1:
        score = -score
    return score
