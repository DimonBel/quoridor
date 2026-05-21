from collections import deque
from heapq import heappush, heappop
from quoridor.core.board import (
    BOARD_SIZE,
    NUM_SQUARES,
    NEIGHBOR_DIR,
    EDGE_H_MASK,
    EDGE_V_MASK,
    sq_to_rc,
)


def _neighbors(s: int, h_walls: int, v_walls: int):
    nd = NEIGHBOR_DIR[s]
    hm = EDGE_H_MASK[s]
    vm = EDGE_V_MASK[s]
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
        yield ns


def bfs_path_exists(
    start: int,
    goal_row: int,
    h_walls: int,
    v_walls: int,
) -> bool:
    r, _ = sq_to_rc(start)
    if r == goal_row:
        return True
    visited = bytearray(NUM_SQUARES)
    visited[start] = 1
    queue = deque([start])
    while queue:
        s = queue.popleft()
        for ns in _neighbors(s, h_walls, v_walls):
            if visited[ns]:
                continue
            nr, _ = sq_to_rc(ns)
            if nr == goal_row:
                return True
            visited[ns] = 1
            queue.append(ns)
    return False


def bfs_shortest_path(
    start: int,
    goal_row: int,
    h_walls: int,
    v_walls: int,
) -> int:
    r, _ = sq_to_rc(start)
    if r == goal_row:
        return 0
    visited = bytearray(NUM_SQUARES)
    visited[start] = 1
    queue = deque([(start, 0)])
    while queue:
        s, dist = queue.popleft()
        for ns in _neighbors(s, h_walls, v_walls):
            if visited[ns]:
                continue
            nr, _ = sq_to_rc(ns)
            if nr == goal_row:
                return dist + 1
            visited[ns] = 1
            queue.append((ns, dist + 1))
    return -1


def astar_shortest_path(
    start: int,
    goal_row: int,
    h_walls: int,
    v_walls: int,
) -> int:
    sr, _ = sq_to_rc(start)
    if sr == goal_row:
        return 0
    dist = [255] * NUM_SQUARES
    dist[start] = 0
    heap = [(abs(sr - goal_row), 0, start)]
    while heap:
        _, d, s = heappop(heap)
        if d > dist[s]:
            continue
        r, _ = sq_to_rc(s)
        if r == goal_row:
            return d
        for ns in _neighbors(s, h_walls, v_walls):
            nd = d + 1
            if nd < dist[ns]:
                dist[ns] = nd
                nr, _ = sq_to_rc(ns)
                f = nd + abs(nr - goal_row)
                heappush(heap, (f, nd, ns))
    return -1
