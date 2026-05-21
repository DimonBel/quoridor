import importlib
import pkgutil
from quoridor.bots.base import Bot

BOTS: dict[str, type[Bot]] = {}


def register(cls: type[Bot]) -> type[Bot]:
    BOTS[cls.name] = cls
    return cls


def discover_bots() -> None:
    import quoridor.bots as pkg
    for info in pkgutil.iter_modules(pkg.__path__):
        if info.name in ("base", "registry", "__init__"):
            continue
        importlib.import_module(f"quoridor.bots.{info.name}")


def get_bot(name: str, **params) -> Bot:
    if not BOTS:
        discover_bots()
    cls = BOTS[name]
    return cls(**params)
