"""Without fork, workers get the objective by pickle, so one that can't be pickled by name is refused up front."""
import multiprocessing
import subprocess
import sys
import textwrap

import pytest

from hone import Constraint, SearchSpace

from conftest import bounded_parameters, make_optimizer
import objectives

SETTINGS = dict(generations=2, population_size=4)
NOT_FORK = [
    pytest.param(
        method,
        marks=pytest.mark.skipif(method not in multiprocessing.get_all_start_methods(), reason=f"no {method} here"),
    )
    for method in ("forkserver", "spawn")
]
needs_fork = pytest.mark.skipif("fork" not in multiprocessing.get_all_start_methods(), reason="no fork here")


@pytest.fixture
def default_start_method(monkeypatch):
    """Make the pool's default start method the one named, and fail the test if a worker process is created."""
    real = multiprocessing.get_context

    def use(method, allow_workers=False):
        def get_context(name=None):
            context = real(method if name is None else name)
            if not allow_workers:
                def no_process(*args, **kwargs):
                    pytest.fail("a worker process was created")

                monkeypatch.setattr(context, "Process", no_process, raising=False)
            return context

        monkeypatch.setattr(multiprocessing, "get_context", get_context)

    return use


def a_lambda_objective():
    return lambda x, y: x**2 + y**2


def a_closure_objective():
    offset = 1.0

    def objective(x, y):
        return x**2 + y**2 + offset

    return objective


@pytest.mark.parametrize("method", NOT_FORK)
@pytest.mark.parametrize("make_objective", [a_lambda_objective, a_closure_objective])
def test_an_objective_that_cannot_be_pickled_is_refused_before_a_worker_starts(
    method, make_objective, default_start_method
):
    default_start_method(method)
    with pytest.raises(TypeError, match="separate process"):
        make_optimizer(**SETTINGS).optimize(make_objective(), workers=2)


@pytest.mark.parametrize("method", NOT_FORK)
def test_a_lambda_constraint_is_refused_before_a_worker_starts(method, default_start_method):
    default_start_method(method)
    space = SearchSpace(bounded_parameters(), constraints=[Constraint.function(lambda x, y: x + y - 8)])
    with pytest.raises(TypeError, match="separate process"):
        make_optimizer(space, **SETTINGS).optimize(objectives.sphere, workers=2)


@pytest.mark.parametrize("method", NOT_FORK)
def test_a_module_level_objective_runs_in_workers(method, default_start_method):
    default_start_method(method, allow_workers=True)
    in_workers = make_optimizer(**SETTINGS).optimize(objectives.sphere, workers=2)
    assert in_workers == make_optimizer(**SETTINGS).optimize(objectives.sphere)


@needs_fork
@pytest.mark.parametrize("make_objective", [a_lambda_objective, a_closure_objective])
def test_forked_workers_still_take_a_lambda_and_a_closure(make_objective, default_start_method):
    default_start_method("fork", allow_workers=True)
    objective = make_objective()
    in_workers = make_optimizer(**SETTINGS).optimize(objective, workers=2)
    assert in_workers == make_optimizer(**SETTINGS).optimize(objective)


@pytest.mark.parametrize("method", NOT_FORK)
def test_the_message_names_the_fix(method, default_start_method):
    default_start_method(method)
    with pytest.raises(TypeError) as raised:
        make_optimizer(**SETTINGS).optimize(a_lambda_objective(), workers=2)
    message = str(raised.value)
    assert "top level of a .py file" in message
    assert "constraint" in message
    assert "fork" in message
    assert isinstance(raised.value.__cause__, Exception)
    assert str(raised.value.__cause__) in message


@pytest.mark.parametrize("method", NOT_FORK)
def test_a_function_defined_at_a_prompt_is_refused(method):
    script = textwrap.dedent(
        f"""
        import multiprocessing
        multiprocessing.set_start_method({method!r})
        from hone import SearchSpace
        from hone.config import GeneticOptimizerConfig
        from hone.optimizers import GeneticOptimizer
        from hone.parameters import RangeParameter

        def objective(x, y):
            return x**2 + y**2

        space = [RangeParameter("x", -5, 5), RangeParameter("y", -5, 5)]
        optimizer = GeneticOptimizer(search_space=space, config=GeneticOptimizerConfig(seed=0, generations=2, population_size=4))
        try:
            optimizer.optimize(objective, workers=2)
        except TypeError as error:
            print(error)
        else:
            print("NOT REFUSED")
        """
    )
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=120)
    assert "NOT REFUSED" not in done.stdout, done.stderr
    assert "top level of a .py file" in done.stdout, done.stdout + done.stderr
