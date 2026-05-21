import itertools
import time
from quoridor.runner.match import run_match
from quoridor.bench.elo import EloTable


def _grid_search(space: dict) -> list[dict]:
    keys = list(space.keys())
    values = [space[k] if isinstance(space[k], list) else [space[k]] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _random_search(space: dict, n: int, rng) -> list[dict]:
    configs = []
    for _ in range(n):
        config = {}
        for k, v in space.items():
            if isinstance(v, list):
                config[k] = rng.choice(v)
            else:
                config[k] = v
        configs.append(config)
    return configs


def run_tuning(
    bot_name: str,
    search_space: dict,
    opponent_name: str = "greedy",
    games_per_config: int = 500,
    method: str = "grid",
    opponent_params: dict = None,
    base_seed: int = 42,
) -> list[dict]:
    import random
    rng = random.Random(base_seed)

    if method == "grid":
        configs = _grid_search(search_space)
    else:
        configs = _random_search(search_space, n=min(50, games_per_config), rng=rng)

    results = []
    total = len(configs)
    for i, params in enumerate(configs):
        t0 = time.perf_counter()
        mr = run_match(
            bot_name, opponent_name,
            games=games_per_config,
            swap_colors=True,
            parallel=False,
            bot1_params=params,
            bot2_params=opponent_params or {},
            base_seed=base_seed + i * 1000,
        )
        elapsed = time.perf_counter() - t0
        win_rate = mr.bot1_wins / mr.games if mr.games else 0
        results.append({
            "params": params,
            "wins": mr.bot1_wins,
            "losses": mr.bot2_wins,
            "draws": mr.draws,
            "win_rate": win_rate,
            "avg_moves": mr.avg_moves,
            "avg_time_ms": mr.avg_move_time1 * 1000,
            "elapsed_s": elapsed,
        })
        print(f"  [{i+1}/{total}] {params} -> win_rate={win_rate:.1%} ({elapsed:.1f}s)")

    results.sort(key=lambda r: (-r["win_rate"], r["avg_time_ms"]))
    return results


def format_tuning_results(results: list[dict]) -> str:
    lines = ["Tuning Results (sorted by win rate):", "=" * 60]
    for i, r in enumerate(results):
        lines.append(
            f"  #{i+1} win={r['win_rate']:.1%} "
            f"W/D/L={r['wins']}/{r['draws']}/{r['losses']} "
            f"avg_t={r['avg_time_ms']:.1f}ms "
            f"params={r['params']}"
        )
    return "\n".join(lines)
