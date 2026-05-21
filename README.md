# Quoridor Bot Platform

Terminal-based Python platform for fast Quoridor simulation between pluggable bots, with full benchmarking and parameter tuning.

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
| `mcts` | UCB1 selection with random or greedy rollouts |

## Architecture

```
src/quoridor/
├── core/        # Engine — board, state, moves, rules, pathfinding (no I/O)
├── bots/        # Plugin bots — one file each, auto-discovered
├── eval/        # Shared heuristics (path-diff, position, wall-count)
├── runner/      # Game runner, match harness, tournament
├── ui/          # ASCII renderer, live mode, replay
├── bench/       # Benchmarking, ELO ratings, parameter tuner
├── persistence/ # Game logger, leaderboard
└── cli.py       # Entry point
```

**Engine/benchmark separation:** The core engine runs headless with zero output. Rendering is an optional UI layer. This lets you run 100k silent games for benchmarking and 1 pretty game for watching.

## Design Highlights

- **Bitmask walls** — horizontal and vertical walls stored as 64-bit integers; edge blocking is a single bitwise AND
- **Precomputed tables** — neighbor/edge geometry computed once at import
- **Make/unmake** — search algorithms mutate and restore state, no deep copies
- **BFS + A\*** — path existence and shortest path with incremental checks
- **Config-driven** — all bot parameters in `configs/default.yaml`
- **Plugin registry** — add a bot by dropping a file in `bots/`

## Running Tests

```bash
pytest tests/ -v
```

## Configuration

Edit `configs/default.yaml` for game settings, bot parameters, and match options. Use `configs/tuning.yaml` for parameter sweep definitions.
