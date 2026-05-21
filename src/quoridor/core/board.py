BOARD_SIZE = 9
NUM_SQUARES = BOARD_SIZE * BOARD_SIZE
NUM_WALL_SLOTS = (BOARD_SIZE - 1) * (BOARD_SIZE - 1)

DIR_N, DIR_S, DIR_W, DIR_E = 0, 1, 2, 3
DIR_DELTAS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def sq(row: int, col: int) -> int:
    return row * BOARD_SIZE + col


def sq_to_rc(s: int) -> tuple[int, int]:
    return divmod(s, BOARD_SIZE)


def in_bounds(row: int, col: int) -> bool:
    return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE


def wall_idx(row: int, col: int) -> int:
    return row * (BOARD_SIZE - 1) + col


def wall_in_bounds(row: int, col: int) -> bool:
    return 0 <= row < BOARD_SIZE - 1 and 0 <= col < BOARD_SIZE - 1


def wall_idx_to_rc(wi: int) -> tuple[int, int]:
    return divmod(wi, BOARD_SIZE - 1)


def _build_neighbor_dir():
    table = [None] * NUM_SQUARES
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            s = sq(r, c)
            n = sq(r - 1, c) if r > 0 else -1
            so = sq(r + 1, c) if r < BOARD_SIZE - 1 else -1
            w = sq(r, c - 1) if c > 0 else -1
            e = sq(r, c + 1) if c < BOARD_SIZE - 1 else -1
            table[s] = (n, so, w, e)
    return table


def _build_edge_masks():
    h_masks = [[0] * 4 for _ in range(NUM_SQUARES)]
    v_masks = [[0] * 4 for _ in range(NUM_SQUARES)]
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            s = sq(r, c)
            if r > 0:
                m = 0
                if c < BOARD_SIZE - 1:
                    m |= 1 << wall_idx(r - 1, c)
                if c > 0:
                    m |= 1 << wall_idx(r - 1, c - 1)
                h_masks[s][DIR_N] = m
            if r < BOARD_SIZE - 1:
                m = 0
                if c < BOARD_SIZE - 1:
                    m |= 1 << wall_idx(r, c)
                if c > 0:
                    m |= 1 << wall_idx(r, c - 1)
                h_masks[s][DIR_S] = m
            if c > 0:
                m = 0
                if r < BOARD_SIZE - 1:
                    m |= 1 << wall_idx(r, c - 1)
                if r > 0:
                    m |= 1 << wall_idx(r - 1, c - 1)
                v_masks[s][DIR_W] = m
            if c < BOARD_SIZE - 1:
                m = 0
                if r < BOARD_SIZE - 1:
                    m |= 1 << wall_idx(r, c)
                if r > 0:
                    m |= 1 << wall_idx(r - 1, c)
                v_masks[s][DIR_E] = m
    return h_masks, v_masks


NEIGHBOR_DIR: tuple[tuple[int, int, int, int], ...] = tuple(_build_neighbor_dir())
_raw_h, _raw_v = _build_edge_masks()
EDGE_H_MASK: tuple[tuple[int, int, int, int], ...] = tuple(tuple(m) for m in _raw_h)
EDGE_V_MASK: tuple[tuple[int, int, int, int], ...] = tuple(tuple(m) for m in _raw_v)


def is_edge_blocked(s: int, di: int, h_walls: int, v_walls: int) -> bool:
    hm = EDGE_H_MASK[s][di]
    if hm and hm & h_walls:
        return True
    vm = EDGE_V_MASK[s][di]
    if vm and vm & v_walls:
        return True
    return False
