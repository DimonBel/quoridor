import json
import os


_K_FACTOR = 32.0
_INITIAL_ELO = 1000.0


def compute_elo_update(rating_a: float, rating_b: float, result: int) -> tuple[float, float]:
    ea = 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))
    eb = 1.0 - ea
    if result == 0:
        sa, sb = 1.0, 0.0
    elif result == 1:
        sa, sb = 0.0, 1.0
    else:
        sa, sb = 0.5, 0.5
    new_a = rating_a + _K_FACTOR * (sa - ea)
    new_b = rating_b + _K_FACTOR * (sb - eb)
    return new_a, new_b


class EloTable:
    def __init__(self, path: str = "leaderboard.json"):
        self.path = path
        self.ratings: dict[str, float] = {}

    def load(self) -> None:
        if os.path.exists(self.path):
            with open(self.path) as f:
                data = json.load(f)
            self.ratings = data.get("ratings", {})

    def save(self) -> None:
        data = {
            "ratings": self.ratings,
        }
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)

    def ensure(self, name: str) -> None:
        if name not in self.ratings:
            self.ratings[name] = _INITIAL_ELO

    def update(self, bot1: str, bot2: str, winner: int) -> None:
        self.ensure(bot1)
        self.ensure(bot2)
        r1, r2 = compute_elo_update(self.ratings[bot1], self.ratings[bot2], winner)
        self.ratings[bot1] = r1
        self.ratings[bot2] = r2

    def sorted_ratings(self) -> list[tuple[str, float]]:
        return sorted(self.ratings.items(), key=lambda x: -x[1])

    def format(self) -> str:
        lines = ["ELO Leaderboard:", "-" * 30]
        for name, rating in self.sorted_ratings():
            lines.append(f"  {name:15s} {rating:.0f}")
        return "\n".join(lines)
