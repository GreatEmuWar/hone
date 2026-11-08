"""A small spec for checking the Measure workflow: the quick suite, random search and the genetic optimizer, a few seeds."""
from hone.benchmarks import DEFAULT_OPTIMIZERS, quick_problems


def problems():
    return quick_problems()


def optimizers():
    return {name: DEFAULT_OPTIMIZERS[name] for name in ("random", "genetic")}


SEEDS = 4
BUDGETS = (10,)


def report(runs):
    return f"{len(runs)} runs, from {len({r.problem for r in runs})} problems.\n"
