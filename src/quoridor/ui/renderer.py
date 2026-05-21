from quoridor.core.board import (
    BOARD_SIZE,
    sq_to_rc,
    wall_idx,
)
from quoridor.core.state import GameState

P1_COLOR = "\033[94m"
P2_COLOR = "\033[91m"
WALL_COLOR = "\033[33;1m"
DIM = "\033[2m"
RESET = "\033[0m"
BOLD = "\033[1m"

# Cell content
EMPTY_CELL = f"{DIM} · {RESET}"
P1_CELL = f"{P1_COLOR}{BOLD} 1 {RESET}"
P2_CELL = f"{P2_COLOR}{BOLD} 2 {RESET}"

# Grid characters (dim for normal grid lines)
H_LINE = f"{DIM}───{RESET}"
V_LINE = f"{DIM}│{RESET}"
CROSS = f"{DIM}┼{RESET}"
T_DOWN = f"{DIM}┬{RESET}"
T_UP = f"{DIM}┴{RESET}"
T_RIGHT = f"{DIM}├{RESET}"
T_LEFT = f"{DIM}┤{RESET}"
TOP_L = f"{DIM}┌{RESET}"
TOP_R = f"{DIM}┐{RESET}"
BOT_L = f"{DIM}└{RESET}"
BOT_R = f"{DIM}┘{RESET}"

# Wall characters (bright, bold)
WALL_H = f"{WALL_COLOR}━━━{RESET}"
WALL_V = f"{WALL_COLOR}┃{RESET}"
WALL_CROSS = f"{WALL_COLOR}╋{RESET}"


def _has_h_wall(state: GameState, r: int, c: int) -> bool:
    """Check if horizontal wall segment between row r and r+1 at column c."""
    if r < 0 or r >= BOARD_SIZE - 1:
        return False
    # A horizontal wall at (wr, wc) covers columns wc and wc+1
    if c > 0 and c - 1 < BOARD_SIZE - 1:
        wi = wall_idx(r, c - 1)
        if state.h_walls & (1 << wi):
            return True
    if c < BOARD_SIZE - 1:
        wi = wall_idx(r, c)
        if state.h_walls & (1 << wi):
            return True
    return False


def _has_v_wall(state: GameState, r: int, c: int) -> bool:
    """Check if vertical wall segment between col c and c+1 at row r."""
    if c < 0 or c >= BOARD_SIZE - 1:
        return False
    # A vertical wall at (wr, wc) covers rows wr and wr+1
    if r > 0 and r - 1 < BOARD_SIZE - 1:
        wi = wall_idx(r - 1, c)
        if state.v_walls & (1 << wi):
            return True
    if r < BOARD_SIZE - 1:
        wi = wall_idx(r, c)
        if state.v_walls & (1 << wi):
            return True
    return False


def render_board(
    state: GameState,
    last_move=None,
    bot_names: tuple[str, str] = ("P1", "P2"),
) -> str:
    p1_r, p1_c = sq_to_rc(state.p1_pos)
    p2_r, p2_c = sq_to_rc(state.p2_pos)
    lines = []

    # Header
    lines.append(
        f"  {BOLD}{bot_names[0]}{RESET} vs {BOLD}{bot_names[1]}{RESET}"
        f"  │  Move {state.move_count}"
    )
    lines.append(
        f"  Walls: {P1_COLOR}{bot_names[0]}={state.walls_remaining[0]}{RESET}  "
        f"{P2_COLOR}{bot_names[1]}={state.walls_remaining[1]}{RESET}  "
        f"Turn: {'P1' if state.current_player == 0 else 'P2'}"
    )
    lines.append("")

    # Column numbers
    col_header = "    " + "   ".join(f"{c}" for c in range(BOARD_SIZE))
    lines.append(col_header)

    # Top border
    top = f"  {TOP_L}" + T_DOWN.join([H_LINE] * BOARD_SIZE) + f"{TOP_R}"
    lines.append(top)

    for r in range(BOARD_SIZE):
        # Cell row
        row_parts = []
        for c in range(BOARD_SIZE):
            if r == p1_r and c == p1_c:
                cell = P1_CELL
            elif r == p2_r and c == p2_c:
                cell = P2_CELL
            else:
                cell = EMPTY_CELL

            row_parts.append(cell)

            if c < BOARD_SIZE - 1:
                if _has_v_wall(state, r, c):
                    row_parts.append(WALL_V)
                else:
                    row_parts.append(V_LINE)

        lines.append(f" {r} {V_LINE}" + "".join(row_parts) + f"{V_LINE}")

        # Separator row between rows
        if r < BOARD_SIZE - 1:
            sep = f"  {T_RIGHT}"
            for c in range(BOARD_SIZE):
                h_wall = _has_h_wall(state, r, c)
                sep += WALL_H if h_wall else H_LINE

                if c < BOARD_SIZE - 1:
                    # Intersection: check surrounding walls
                    h_left = _has_h_wall(state, r, c)
                    h_right = _has_h_wall(state, r, c + 1)
                    v_up = _has_v_wall(state, r, c)
                    v_down = _has_v_wall(state, r + 1, c)
                    if h_left or h_right or v_up or v_down:
                        sep += WALL_CROSS
                    else:
                        sep += CROSS
            sep += f"{T_LEFT}"
            lines.append(sep)

    # Bottom border
    bot = f"  {BOT_L}" + T_UP.join([H_LINE] * BOARD_SIZE) + f"{BOT_R}"
    lines.append(bot)

    return "\n".join(lines)


def render_live(state: GameState, last_move=None, bot_names=("P1", "P2")):
    print("\033[2J\033[H", end="")
    board = render_board(state, last_move, bot_names)
    print(board)
    if last_move:
        print(f"  Last: {last_move}")
