from quoridor.core.board import (
    BOARD_SIZE,
    sq_to_rc,
    wall_idx,
    sq,
)
from quoridor.core.moves import MoveType
from quoridor.core.state import GameState

P1_COLOR = "\033[94m"
P2_COLOR = "\033[91m"
WALL_H_COLOR = "\033[33m"
WALL_V_COLOR = "\033[36m"
RESET = "\033[0m"
LAST_MOVE_COLOR = "\033[7m"


def render_board(
    state: GameState,
    last_move=None,
    bot_names: tuple[str, str] = ("P1", "P2"),
) -> str:
    p1_r, p1_c = sq_to_rc(state.p1_pos)
    p2_r, p2_c = sq_to_rc(state.p2_pos)
    lines = []
    lines.append(f"  {bot_names[0]} vs {bot_names[1]}  |  Move {state.move_count}")
    lines.append(
        f"  Walls: {bot_names[0]}={state.walls_remaining[0]}  "
        f"{bot_names[1]}={state.walls_remaining[1]}  "
        f"Turn: {'P1' if state.current_player == 0 else 'P2'}"
    )
    lines.append("")
    header = "    " + " ".join(f" {c}" for c in range(BOARD_SIZE))
    lines.append(header)

    for r in range(BOARD_SIZE):
        row_str = f" {r}  "
        for c in range(BOARD_SIZE):
            if r == p1_r and c == p1_c:
                row_str += f"{P1_COLOR}(1){RESET}"
            elif r == p2_r and c == p2_c:
                row_str += f"{P2_COLOR}(2){RESET}"
            else:
                row_str += "( )"
            if c < BOARD_SIZE - 1:
                wi = wall_idx(r, c)
                if state.v_walls & (1 << wi):
                    row_str += f"{WALL_V_COLOR}║{RESET}"
                else:
                    row_str += " "
        lines.append(row_str)

        if r < BOARD_SIZE - 1:
            wall_str = "    "
            for c in range(BOARD_SIZE):
                wi = wall_idx(r, c)
                if state.h_walls & (1 << wi):
                    wall_str += f"{WALL_H_COLOR}═══{RESET}"
                else:
                    wall_str += "   "
                if c < BOARD_SIZE - 1:
                    if state.h_walls & (1 << wi):
                        wall_str += f"{WALL_H_COLOR}═{RESET}"
                    else:
                        wall_str += " "
            lines.append(wall_str)

    return "\n".join(lines)


def render_live(state: GameState, last_move=None, bot_names=("P1", "P2")):
    print("\033[2J\033[H", end="")
    board = render_board(state, last_move, bot_names)
    print(board)
    if last_move:
        print(f"  Last: {last_move}")
