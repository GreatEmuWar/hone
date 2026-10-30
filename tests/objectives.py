"""Objective functions defined at module level, so worker processes can import them."""
import math

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
