from quoridor.core.state import GameState
from quoridor.core.moves import Move
from quoridor.runner.game_runner import _str_to_move, load_game
from quoridor.ui.renderer import render_live


def replay_game(path: str) -> None:
    data = load_game(path)
    history = data["history"]
    bot_names = (data.get("bot1", "P1"), data.get("bot2", "P2"))
    state = GameState(max_moves=data.get("moves", 400))
    render_live(state, bot_names=bot_names)
    print(f"\n  Loaded game: {bot_names[0]} vs {bot_names[1]}")
    print(f"  Total moves: {len(history)}")
    print(f"  Original winner: P{data['winner']+1}" if data.get("winner", -1) >= 0 else "  Draw")
    print("  Press Enter to step forward, 'q' to quit...")

    for i, move_str in enumerate(history):
        move = _str_to_move(move_str)
        state.make_move(move)
        render_live(state, last_move=move, bot_names=bot_names)
        print(f"\n  Move {i+1}/{len(history)}: {move_str}")
        try:
            cmd = input("  [Enter]=next q=quit> ").strip().lower()
            if cmd == "q":
                break
        except (EOFError, KeyboardInterrupt):
            break
