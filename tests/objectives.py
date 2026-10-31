"""Objective functions defined at module level, so worker processes can import them."""
import math
import os
import threading
import time

from hone import EvaluationFailed


def sphere(x, y):
    return x**2 + y**2


def fails_right_of_zero(x, y):
    """Sphere, except that every evaluation with x > 0 fails, half by exception and half by NaN."""
    if x > 0:
        if y > 0:
            raise EvaluationFailed("simulated crash")
        return math.nan
    return x**2 + y**2


def always_fails(x, y):
    raise EvaluationFailed("nothing converges")


def raises_a_bug(x, y):
    raise ZeroDivisionError("a bug in the objective")


def hangs_right_of_zero(x, y):
    """Sphere, except that every evaluation with x > 0 hangs."""
    if x > 0:
        time.sleep(60)
    return x**2 + y**2


def crashes_right_of_zero(x, y):
    """Sphere, except that every evaluation with x > 0 kills its process, as a segfaulting simulation would."""
    if x > 0:
        os._exit(1)
    return x**2 + y**2


def raises_while_others_hang(x, y):
    """A bug on one side, and evaluations that would run for a minute on the other."""
    if x > 0:
        raise ZeroDivisionError("a bug in the objective")
    time.sleep(60)
    return 0.0


class UnpicklableError(Exception):
    def __init__(self):
        super().__init__("carries a lock")
        self.lock = threading.Lock()


def raises_unpicklable(x, y):
    raise UnpicklableError()


class SlowToStart:
    """A callable that takes a while to unpickle, as a worker importing a heavy simulation package would."""

    def __init__(self, function, delay):
        self.function, self.delay = function, delay

    def __call__(self, params):
        return self.function(**params)

    def __setstate__(self, state):
        time.sleep(state["delay"])
        self.__dict__.update(state)


class CannotStart:
    """A callable that fails to unpickle in a worker, as an objective whose module won't import would."""

    def __init__(self):
        self.package = "simulation"  # pickle only calls __setstate__ when there is state

    def __call__(self, params):
        return 0.0

    def __setstate__(self, state):
        raise ImportError("a missing simulation package")


def slow_sphere(x, y):
    """Sphere, taking a tenth of a second per evaluation."""
    time.sleep(0.1)
    return x**2 + y**2


def sometimes_slow(x, y):
    """Sphere, taking a third of a second for about one point in four and almost no time for the rest."""
    time.sleep(0.3 if int(abs(x) * 1e6) % 4 == 0 else 0.005)
    return x**2 + y**2
