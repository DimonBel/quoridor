import json
import os
from quoridor.runner.game_runner import GameResult, save_game


class GameLogger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self._counter = 0

    def log_game(self, result: GameResult) -> str:
        filename = f"game_{self._counter:06d}.json"
        path = os.path.join(self.log_dir, filename)
        save_game(result, path)
        self._counter += 1
        return path


class Leaderboard:
    def __init__(self, path: str = "leaderboard.json"):
        self.path = path
        self.data: dict[str, dict] = {}

    def load(self) -> None:
        if os.path.exists(self.path):
            with open(self.path) as f:
                raw = json.load(f)
            for name, entry in raw.get("entries", {}).items():
                self.data[name] = entry

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        entries = {}
        for name in sorted(self.data):
            entries[name] = self.data[name]
        with open(self.path, "w") as f:
            json.dump({"entries": entries}, f, indent=2)

    def update(self, name: str, won: bool, opponent: str, move_time: float) -> None:
        if name not in self.data:
            self.data[name] = {"wins": 0, "losses": 0, "draws": 0, "games": 0, "total_time": 0.0}
        d = self.data[name]
        d["games"] += 1
        if won:
            d["wins"] += 1
        else:
            d["losses"] += 1
        d["total_time"] += move_time

    def format(self) -> str:
        lines = ["Leaderboard:", "-" * 50]
        sorted_entries = sorted(self.data.items(), key=lambda x: -x[1]["wins"])
        for name, d in sorted_entries:
            wr = d["wins"] / d["games"] if d["games"] else 0
            avg_t = d["total_time"] / d["games"] if d["games"] else 0
            lines.append(f"  {name:15s} {d['wins']:4d}W/{d['losses']:4d}L ({wr:.1%}) avg={avg_t*1000:.1f}ms")
        return "\n".join(lines)
