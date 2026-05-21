import argparse
import sys
import yaml
import os

from quoridor.bots.registry import get_bot, discover_bots
from quoridor.runner.game_runner import run_game, save_game
from quoridor.runner.match import run_match
from quoridor.runner.tournament import run_tournament
from quoridor.bench.benchmark import benchmark_match, format_benchmark
from quoridor.bench.tuner import run_tuning, format_tuning_results
from quoridor.bench.elo import EloTable
from quoridor.persistence.logger import GameLogger


def _load_config(path: str = None) -> dict:
    defaults = {
        "game": {"board_size": 9, "walls_per_player": 10, "max_moves": 400},
        "bots": {},
        "match": {"games": 100, "swap_colors": True, "parallel": True, "max_workers": None},
    }
    if path and os.path.exists(path):
        with open(path) as f:
            user_config = yaml.safe_load(f) or {}
        for section in defaults:
            if section in user_config:
                if isinstance(defaults[section], dict):
                    defaults[section].update(user_config[section])
                else:
                    defaults[section] = user_config[section]
    return defaults


def _cmd_play(args, config):
    bot1_params = config["bots"].get(args.bot1, {})
    bot2_params = config["bots"].get(args.bot2, {})
    b1 = get_bot(args.bot1, seed=1, **bot1_params)
    b2 = get_bot(args.bot2, seed=2, **bot2_params)

    if args.watch:
        from quoridor.ui.live import run_game_live
        result = run_game_live(
            b1, b2,
            delay_ms=args.delay,
            seed=args.seed,
            max_moves=config["game"]["max_moves"],
        )
    else:
        result = run_game(b1, b2, seed=args.seed, max_moves=config["game"]["max_moves"])
        winner_name = (b1.name, b2.name)[result.winner] if result.winner >= 0 else "Draw"
        print(f"Winner: {winner_name} in {result.moves} moves")

    if args.save:
        logger = GameLogger(config.get("logging", {}).get("log_dir", "logs"))
        path = logger.log_game(result)
        print(f"Game saved to {path}")


def _cmd_sim(args, config):
    bot1_params = config["bots"].get(args.bot1, {})
    bot2_params = config["bots"].get(args.bot2, {})
    result = benchmark_match(
        args.bot1, args.bot2,
        games=args.games,
        bot1_params=bot1_params,
        bot2_params=bot2_params,
        swap_colors=config["match"]["swap_colors"],
        max_moves=config["game"]["max_moves"],
        parallel=config["match"]["parallel"],
        max_workers=config["match"]["max_workers"],
    )
    print(format_benchmark(result))

    if args.save_elo:
        elo = EloTable(args.save_elo)
        elo.load()
        for _ in range(result["bot1_wins"]):
            elo.update(args.bot1, args.bot2, 0)
        for _ in range(result["bot2_wins"]):
            elo.update(args.bot1, args.bot2, 1)
        for _ in range(result["draws"]):
            elo.update(args.bot1, args.bot2, -1)
        elo.save()
        print(f"\n{elo.format()}")


def _cmd_tourney(args, config):
    bot_names = [b.strip() for b in args.bots.split(",")]
    bot_params = config.get("bots", {})
    games_per_pair = args.games or config.get("tournament", {}).get("games_per_pair", 200)

    result = run_tournament(
        bot_names,
        bot_params=bot_params,
        games_per_pair=games_per_pair,
        max_moves=config["game"]["max_moves"],
    )

    print("\nTournament Results:")
    print("=" * 50)
    for (b1, b2), data in sorted(result.results.items()):
        total = data["bot1_wins"] + data["bot2_wins"] + data["draws"]
        wr1 = data["bot1_wins"] / total if total else 0
        print(f"  {b1} vs {b2}: {data['bot1_wins']}-{data['bot2_wins']}-{data['draws']} ({wr1:.1%})")

    print("\nFinal ELO Ratings:")
    for name, rating in sorted(result.elo_ratings.items(), key=lambda x: -x[1]):
        print(f"  {name:15s} {rating:.0f}")


def _cmd_tune(args, config):
    with open(args.config) as f:
        tune_config = yaml.safe_load(f)

    results = run_tuning(
        bot_name=tune_config["bot"],
        search_space=tune_config["search_space"],
        opponent_name=tune_config.get("opponent", "greedy"),
        games_per_config=tune_config.get("games_per_config", 500),
        method=tune_config.get("method", "grid"),
    )
    print()
    print(format_tuning_results(results))


def _cmd_replay(args, config):
    from quoridor.ui.replay import replay_game
    replay_game(args.file)


def _cmd_test(args, config):
    import subprocess
    cmd = [sys.executable, "-m", "pytest", "tests/test_performance.py", "-v", "-s"]
    if args.filter:
        cmd.extend(["-k", args.filter])
    if args.dashboard:
        cmd.extend(["-k", "test_dashboard"])
    sys.exit(subprocess.call(cmd))


def _cmd_bots(args, config):
    discover_bots()
    from quoridor.bots.registry import BOTS
    print("Available bots:")
    for name in sorted(BOTS.keys()):
        print(f"  {name}")


def main():
    parser = argparse.ArgumentParser(prog="quoridor", description="Quoridor Bot Platform")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file path")
    sub = parser.add_subparsers(dest="command")

    play_p = sub.add_parser("play", help="Play a single game")
    play_p.add_argument("--bot1", default="random")
    play_p.add_argument("--bot2", default="random")
    play_p.add_argument("--watch", action="store_true")
    play_p.add_argument("--delay", type=int, default=200)
    play_p.add_argument("--seed", type=int, default=42)
    play_p.add_argument("--save", action="store_true")

    sim_p = sub.add_parser("sim", help="Run simulation headless")
    sim_p.add_argument("--bot1", default="random")
    sim_p.add_argument("--bot2", default="greedy")
    sim_p.add_argument("--games", type=int, default=100)
    sim_p.add_argument("--save-elo", default=None, help="Path to ELO file")

    tourney_p = sub.add_parser("tourney", help="Round-robin tournament")
    tourney_p.add_argument("--bots", default="random,greedy")
    tourney_p.add_argument("--games", type=int, default=None)

    tune_p = sub.add_parser("tune", help="Parameter tuning")
    tune_p.add_argument("--config", default="configs/tuning.yaml", dest="config")

    replay_p = sub.add_parser("replay", help="Replay a saved game")
    replay_p.add_argument("--file", required=True)

    test_p = sub.add_parser("test", help="Run performance tests and diagnostics")
    test_p.add_argument("-k", "--filter", default=None, help="Filter tests by keyword (e.g. 'bfs', 'bot', 'scaling')")
    test_p.add_argument("--dashboard", action="store_true", help="Run only the summary dashboard")

    bots_p = sub.add_parser("bots", help="List available bots")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    config = _load_config(args.config if hasattr(args, "config") else "configs/default.yaml")

    discover_bots()

    commands = {
        "play": _cmd_play,
        "sim": _cmd_sim,
        "tourney": _cmd_tourney,
        "tune": _cmd_tune,
        "replay": _cmd_replay,
        "test": _cmd_test,
        "bots": _cmd_bots,
    }
    cmd_func = commands.get(args.command)
    if cmd_func:
        cmd_func(args, config)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
