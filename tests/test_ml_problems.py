"""The machine learning tuning problems: real models, scored by cross-validation."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn")

from hone.benchmarks import ml_problems, run_benchmark  # noqa: E402
from hone.benchmarks.__main__ import main  # noqa: E402
from hone.benchmarks.ml import BUDGETS, OPTIMA  # noqa: E402

PROBLEMS = ml_problems()


def test_every_problem_has_a_reference_optimum():
    assert {p.name.removesuffix("-flat") for p in PROBLEMS} == set(OPTIMA)
    assert all(0 <= p.optimum < 1 for p in PROBLEMS)


@pytest.mark.parametrize("problem", PROBLEMS, ids=lambda p: p.name)
def test_evaluations_are_scores_no_better_than_the_reference(problem):
    params = problem.search_space.sample(np.random.default_rng(0))
    value = problem.function(**params)
    assert 0 <= value <= 1.5, "1 − accuracy is in [0, 1]; 1 − R² can exceed 1 for a poor model"
    assert value >= problem.optimum - 1e-12
    assert problem.function(**params) == value, "fixed folds make every evaluation repeatable"


def test_small_budgets_only():
    assert max(BUDGETS) <= 50


def test_cli_runs_the_ml_suite(tmp_path, capsys):
    results = tmp_path / "ml.md"
    assert main(["--ml", "--seeds", "1", "--budgets", "1", "--markdown", str(results)]) == 0
    text = results.read_text()
    assert "5 problems, 1 seeds" in text and "| svm-breast-cancer |" in text


def test_the_iris_example_runs():
    example = Path(__file__).parent.parent / "examples" / "tune_iris.py"
    output = subprocess.run([sys.executable, str(example)], capture_output=True, text=True, check=True).stdout
    assert output.startswith("accuracy 0.9")


def test_the_model_choice_example_runs():
    example = Path(__file__).parent.parent / "examples" / "choose_a_model.py"
    output = subprocess.run([sys.executable, str(example)], capture_output=True, text=True, check=True).stdout
    accuracy = float(output.split()[1])
    assert accuracy >= 0.97, output


def test_runs_with_every_optimizer():
    runs = run_benchmark(PROBLEMS[:1], seeds=range(1), budgets_per_dimension=[2])
    assert {r.optimizer for r in runs} == {"genetic", "cma-es", "bayesian", "random"}
