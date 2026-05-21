from collections import deque
from quoridor.core.board import (
    BOARD_SIZE,
    NUM_SQUARES,
    NEIGHBOR_DIR,
    EDGE_H_MASK,
    EDGE_V_MASK,
    _SQ_ROW,
)

# Local refs for speed — avoid module-level dict lookups in hot loops
_ND = NEIGHBOR_DIR
_HM = EDGE_H_MASK
_VM = EDGE_V_MASK
_ROW = _SQ_ROW
_GOAL_START = tuple(r * BOARD_SIZE for r in range(BOARD_SIZE))
_GOAL_END = tuple(r * BOARD_SIZE + BOARD_SIZE for r in range(BOARD_SIZE))

# Precompute goal-row square sets for fast checking
_GOAL_ROW_SET = tuple(
    frozenset(range(r * BOARD_SIZE, r * BOARD_SIZE + BOARD_SIZE))
    for r in range(BOARD_SIZE)
)


def bfs_path_exists(
    start: int,
    goal_row: int,
    h_walls: int,
    v_walls: int,
) -> bool:
    if _ROW[start] == goal_row:
        return True
    goal_lo = goal_row * BOARD_SIZE
    goal_hi = goal_lo + BOARD_SIZE
    visited = bytearray(NUM_SQUARES)
    visited[start] = 1
    queue = deque()
    queue.append(start)
    nd = _ND
    hm = _HM
    vm = _VM
    row = _ROW
    while queue:
        s = queue.popleft()
        s_nd = nd[s]
        s_hm = hm[s]
        s_vm = vm[s]
        # Unrolled 4 directions — no loop overhead
        # DIR_N = 0
        ns = s_nd[0]
        if ns >= 0 and not visited[ns]:
            h = s_hm[0]
            if not (h and h & h_walls):
                v = s_vm[0]
                if not (v and v & v_walls):
                    if goal_lo <= ns < goal_hi:
                        return True
                    visited[ns] = 1
                    queue.append(ns)
        # DIR_S = 1
        ns = s_nd[1]
        if ns >= 0 and not visited[ns]:
            h = s_hm[1]
            if not (h and h & h_walls):
                v = s_vm[1]
                if not (v and v & v_walls):
                    if goal_lo <= ns < goal_hi:
                        return True
                    visited[ns] = 1
                    queue.append(ns)
        # DIR_W = 2
        ns = s_nd[2]
        if ns >= 0 and not visited[ns]:
            h = s_hm[2]
            if not (h and h & h_walls):
                v = s_vm[2]
                if not (v and v & v_walls):
                    if goal_lo <= ns < goal_hi:
                        return True
                    visited[ns] = 1
                    queue.append(ns)
        # DIR_E = 3
        ns = s_nd[3]
        if ns >= 0 and not visited[ns]:
            h = s_hm[3]
            if not (h and h & h_walls):
                v = s_vm[3]
                if not (v and v & v_walls):
                    if goal_lo <= ns < goal_hi:
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
    if _ROW[start] == goal_row:
        return 0
    goal_lo = goal_row * BOARD_SIZE
    goal_hi = goal_lo + BOARD_SIZE
    # Use flat distance array instead of (node, dist) tuples
    dist = bytearray(NUM_SQUARES)  # 0 = unvisited (except start)
    dist[start] = 1  # store dist+1 so 0 means unvisited
    queue = deque()
    queue.append(start)
    nd = _ND
    hm = _HM
    vm = _VM
    while queue:
        s = queue.popleft()
        d = dist[s]  # this is actual_dist + 1
        s_nd = nd[s]
        s_hm = hm[s]
        s_vm = vm[s]
        # Unrolled 4 directions
        ns = s_nd[0]
        if ns >= 0 and not dist[ns]:
            h = s_hm[0]
            if not (h and h & h_walls):
                v = s_vm[0]
                if not (v and v & v_walls):
                    if goal_lo <= ns < goal_hi:
                        return d  # d = actual_dist + 1, and we need +1 more = d
                    dist[ns] = d + 1
                    queue.append(ns)
        ns = s_nd[1]
        if ns >= 0 and not dist[ns]:
            h = s_hm[1]
            if not (h and h & h_walls):
                v = s_vm[1]
                if not (v and v & v_walls):
                    if goal_lo <= ns < goal_hi:
                        return d
                    dist[ns] = d + 1
                    queue.append(ns)
        ns = s_nd[2]
        if ns >= 0 and not dist[ns]:
            h = s_hm[2]
            if not (h and h & h_walls):
                v = s_vm[2]
                if not (v and v & v_walls):
                    if goal_lo <= ns < goal_hi:
                        return d
                    dist[ns] = d + 1
                    queue.append(ns)
        ns = s_nd[3]
        if ns >= 0 and not dist[ns]:
            h = s_hm[3]
            if not (h and h & h_walls):
                v = s_vm[3]
                if not (v and v & v_walls):
                    if goal_lo <= ns < goal_hi:
                        return d
                    dist[ns] = d + 1
                    queue.append(ns)
    return -1


def bfs_path_trace(
    start: int,
    goal_row: int,
    h_walls: int,
    v_walls: int,
) -> list[int] | None:
    """BFS that returns the actual path as a list of squares, or None if no path.
    Used for smart wall generation — we need to know which squares are on the path."""
    if _ROW[start] == goal_row:
        return [start]
    goal_lo = goal_row * BOARD_SIZE
    goal_hi = goal_lo + BOARD_SIZE
    prev = [-1] * NUM_SQUARES
    prev[start] = start
    queue = deque()
    queue.append(start)
    nd = _ND
    hm = _HM
    vm = _VM
    found = -1
    while queue:
        s = queue.popleft()
        s_nd = nd[s]
        s_hm = hm[s]
        s_vm = vm[s]
        # Unrolled 4 directions
        ns = s_nd[0]
        if ns >= 0 and prev[ns] < 0:
            h = s_hm[0]
            if not (h and h & h_walls):
                v = s_vm[0]
                if not (v and v & v_walls):
                    prev[ns] = s
                    if goal_lo <= ns < goal_hi:
                        found = ns
                        break
                    queue.append(ns)
        ns = s_nd[1]
        if ns >= 0 and prev[ns] < 0:
            h = s_hm[1]
            if not (h and h & h_walls):
                v = s_vm[1]
                if not (v and v & v_walls):
                    prev[ns] = s
                    if goal_lo <= ns < goal_hi:
                        found = ns
                        break
                    queue.append(ns)
        ns = s_nd[2]
        if ns >= 0 and prev[ns] < 0:
            h = s_hm[2]
            if not (h and h & h_walls):
                v = s_vm[2]
                if not (v and v & v_walls):
                    prev[ns] = s
                    if goal_lo <= ns < goal_hi:
                        found = ns
                        break
                    queue.append(ns)
        ns = s_nd[3]
        if ns >= 0 and prev[ns] < 0:
            h = s_hm[3]
            if not (h and h & h_walls):
                v = s_vm[3]
                if not (v and v & v_walls):
                    prev[ns] = s
                    if goal_lo <= ns < goal_hi:
                        found = ns
                        break
                    queue.append(ns)
    if found < 0:
        return None
    path = []
    s = found
    while s != start:
        path.append(s)
        s = prev[s]
    path.append(start)
    path.reverse()
    return path


def astar_shortest_path(
    start: int,
    goal_row: int,
    h_walls: int,
    v_walls: int,
) -> int:
    from heapq import heappush, heappop
    sr = _ROW[start]
    if sr == goal_row:
        return 0
    goal_lo = goal_row * BOARD_SIZE
    goal_hi = goal_lo + BOARD_SIZE
    dist = [255] * NUM_SQUARES
    dist[start] = 0
    heap = [(abs(sr - goal_row), 0, start)]
    nd = _ND
    hm = _HM
    vm = _VM
    row = _ROW
    while heap:
        _, d, s = heappop(heap)
        if d > dist[s]:
            continue
        if goal_lo <= s < goal_hi:
            return d
        s_nd = nd[s]
        s_hm = hm[s]
        s_vm = vm[s]
        for di in range(4):
            ns = s_nd[di]
            if ns < 0:
                continue
            h = s_hm[di]
            if h and h & h_walls:
                continue
            v = s_vm[di]
            if v and v & v_walls:
                continue
            nd2 = d + 1
            if nd2 < dist[ns]:
                dist[ns] = nd2
                f = nd2 + abs(row[ns] - goal_row)
                heappush(heap, (f, nd2, ns))
    return -1
