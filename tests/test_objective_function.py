import pytest

from hone.objective_function import ObjectiveFunction
from hone.parameters import FixedParameter, RangeParameter, SearchSpace

XY = SearchSpace([RangeParameter("x", 0, 1), RangeParameter("y", 0, 1)])


def objective(function):
    return ObjectiveFunction(name=function.__name__, function=function, search_space=XY)


def test_matching_signature_accepted():
    def f(x, y):
        return x + y

    objective(f)


def test_extra_keyword_arguments_accepted():
    def f(**params):
        return 0

    objective(f)


def test_defaulted_argument_may_be_left_out():
    def f(x, y, scale=1):
        return scale * (x + y)

    objective(f)


def test_unknown_parameter_rejected():
    def f(x):
        return x

    with pytest.raises(ValueError, match="does not accept parameters: y"):
        objective(f)


def test_missing_parameter_rejected():
    def f(x, y, z):
        return x + y + z

    with pytest.raises(ValueError, match="missing from the search space: z"):
        objective(f)


def test_fixed_parameters_count_toward_signature():
    def f(x, mode):
        return x

    ObjectiveFunction(
        name="f",
        function=f,
        search_space=SearchSpace([RangeParameter("x", 0, 1), FixedParameter("mode", "fast")]),
    )
