# Quoridor Bot Platform

Terminal-based Python platform for fast Quoridor simulation between pluggable bots, with full benchmarking, parameter tuning, and performance diagnostics.

## Quick Start

```bash
# Install (requires Python 3.11+)
pip install -e ".[dev]"

# List available bots
quoridor bots

# Watch a live game
quoridor play --bot1 random --bot2 greedy --watch --delay 200

# Run a headless simulation (100 games)
quoridor sim --bot1 random --bot2 greedy --games 100

# Round-robin tournament
quoridor tourney --bots random,greedy,minimax,alphabeta,mcts --games 200

# Parameter tuning
quoridor tune --config configs/tuning.yaml

# Replay a saved game
quoridor replay --file logs/game_000000.json
```

## Available Bots

| Bot | Description |
|-----|-------------|
| `random` | Picks any legal move at random |
| `greedy` | 1-ply search minimizing own shortest path |
| `minimax` | Depth-N minimax with make/unmake |
| `alphabeta` | Alpha-beta pruning + Zobrist transposition table |
| `mcts` | UCB1 selection with pawn-only rollouts for speed |

## Architecture

```
src/quoridor/
├── core/        # Engine — board, state, moves, rules, pathfinding (no I/O)
├── bots/        # Plugin bots — one file each, auto-discovered
├── eval/        # Shared heuristics (path-diff, position, wall-count)
├── runner/      # Game runner, match harness, tournament
├── ui/          # Grid renderer with box-drawing chars, live mode, replay
├── bench/       # Benchmarking, ELO ratings, parameter tuner
├── persistence/ # Game logger, leaderboard
└── cli.py       # Entry point
```

**Engine/benchmark separation:** The core engine runs headless with zero output. Rendering is an optional UI layer. This lets you run thousands of silent games for benchmarking and 1 pretty game for watching.

## Performance

Key engine optimizations and current throughput numbers:

| Layer | Metric |
|-------|--------|
| BFS pathfinding | ~40,000 calls/s |
| Pawn move gen | ~900,000 calls/s |
| Make/unmake | ~1,000,000 ops/s |
| Pawn-only random games | ~2,500 games/s |
| Full random games (with walls) | ~22 games/s |
| AlphaBeta d=2 move | ~260 ms |
| MCTS 100 sims move | ~340 ms |

### Design Highlights

- **Bitmask walls** — horizontal and vertical walls stored as 64-bit integers; edge blocking is a single bitwise AND
- **Precomputed tables** — neighbor/edge geometry and row lookups computed once at import
- **Make/unmake** — search algorithms mutate and restore state, no deep copies
- **Smart wall generation** — only validates walls near players' shortest paths (skips BFS for non-critical walls)
- **Plain int move encoding** — moves are `(type, row, col)` tuples with int constants, no enum overhead
- **Inlined BFS** — 4 directions unrolled, flat `bytearray` distance tracking, no generator overhead
- **Pawn-only MCTS rollouts** — rollouts skip wall generation entirely for fast playouts
- **Move ordering** — alpha-beta tries pawn moves before walls for better cutoffs

## Running Tests

```bash
# Correctness tests (fast)
pytest tests/ --ignore=tests/test_performance.py -v

# Performance tests — full diagnostic suite
quoridor test

# Just the summary dashboard
quoridor test --dashboard

# Filter by area (bfs, bot, scaling, wallgen, memory, etc.)
quoridor test -k bfs
quoridor test -k bot
quoridor test -k scaling
quoridor test -k "smart_wall"
```

### Performance Test Suite

The `quoridor test` command runs 35 performance tests across 10 categories:

| Category | What it measures | Why it matters |
|----------|-----------------|----------------|
| **Pathfinding** | BFS/A* throughput on empty and walled boards | BFS is called hundreds of times per `legal_moves()` |
| **Move generation** | Pawn and wall movegen speed, empty vs midgame | Wall gen is the #1 bottleneck (triggers BFS per candidate) |
| **Make/unmake** | State mutation and restore throughput | Core loop of every search bot |
| **Heuristic eval** | `evaluate()` calls per second | Called at every leaf of minimax/alphabeta |
| **Game throughput** | Full games/s, pawn-only games/s, wall overhead ratio | The bottom-line metric for simulation speed |
| **Bot performance** | Per-move time for each bot at various depths | Catches regressions in bot logic |
| **Scaling** | How BFS and legal_moves degrade with more walls | Detects performance cliffs as game progresses |
| **Memory** | Make/unmake state drift, UndoRecord weight | Catches accidental allocations in hot loops |
| **Smart wall gen** | Correctness validation against brute-force | Ensures optimization doesn't generate illegal moves |
| **Dashboard** | One-screen summary with bottleneck diagnosis | First thing to check after any engine change |

Each test prints its metric and enforces a minimum-acceptable threshold. When something regresses, the failing test names the exact layer and suggests where to look.

## Configuration

Edit `configs/default.yaml` for game settings, bot parameters, and match options. Use `configs/tuning.yaml` for parameter sweep definitions.

## Adding a Bot

Drop a file in `src/quoridor/bots/`:

```python
from quoridor.bots.base import Bot
from quoridor.bots.registry import register

@register
class MyBot(Bot):
    name = "mybot"

    def choose_move(self, state):
        moves = state.legal_moves()
        # your logic here
        return moves[0]
```

No other files need to change. The registry auto-discovers it.
