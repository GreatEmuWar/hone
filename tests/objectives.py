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
