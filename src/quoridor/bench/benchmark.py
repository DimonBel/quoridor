import time
import statistics
from quoridor.runner.match import run_match, MatchResult


def benchmark_match(
    bot1_name: str,
    bot2_name: str,
    games: int = 1000,
    confidence: bool = True,
    **kwargs,
) -> dict:
    t0 = time.perf_counter()
    mr = run_match(bot1_name, bot2_name, games=games, **kwargs)
    elapsed = time.perf_counter() - t0

    result = {
        "bot1": bot1_name,
        "bot2": bot2_name,
        "games": mr.games,
        "bot1_wins": mr.bot1_wins,
        "bot2_wins": mr.bot2_wins,
        "draws": mr.draws,
        "win_rate_1": mr.bot1_wins / mr.games if mr.games else 0,
        "win_rate_2": mr.bot2_wins / mr.games if mr.games else 0,
        "avg_moves": mr.avg_moves,
        "avg_move_time_1_ms": mr.avg_move_time1 * 1000,
        "avg_move_time_2_ms": mr.avg_move_time2 * 1000,
        "worst_move_time_1_ms": mr.worst_move_time1 * 1000,
        "worst_move_time_2_ms": mr.worst_move_time2 * 1000,
        "total_time_s": elapsed,
        "games_per_second": mr.games / elapsed if elapsed > 0 else 0,
    }

    if confidence and mr.games > 0:
        p = result["win_rate_1"]
        n = mr.games
        z = 1.96
        se = (p * (1 - p) / n) ** 0.5
        result["win_rate_1_ci"] = (max(0, p - z * se), min(1, p + z * se))

    return result


def format_benchmark(result: dict) -> str:
    lines = [
        f"Benchmark: {result['bot1']} vs {result['bot2']}",
        f"  Games: {result['games']}",
        f"  {result['bot1']} wins: {result['bot1_wins']} ({result['win_rate_1']:.1%})",
        f"  {result['bot2']} wins: {result['bot2_wins']} ({result['win_rate_2']:.1%})",
        f"  Draws: {result['draws']}",
        f"  Avg moves/game: {result['avg_moves']:.1f}",
        f"  Avg move time: {result['avg_move_time_1_ms']:.1f}ms / {result['avg_move_time_2_ms']:.1f}ms",
        f"  Worst move time: {result['worst_move_time_1_ms']:.1f}ms / {result['worst_move_time_2_ms']:.1f}ms",
        f"  Total time: {result['total_time_s']:.1f}s ({result['games_per_second']:.1f} games/s)",
    ]
    if "win_rate_1_ci" in result:
        lo, hi = result["win_rate_1_ci"]
        lines.append(f"  95% CI for {result['bot1']}: [{lo:.1%}, {hi:.1%}]")
    return "\n".join(lines)
