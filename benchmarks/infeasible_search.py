"""The test of how to search before anything is feasible: item 1 of docs/design/output-constraints-quality.md.

The current method maximizes the probability that every limit holds. A and B replace it only
until a feasible design is found, as subclasses here, so the shipped optimizer is what is
measured as the current method. The rule that decides between them is fixed in the design doc.

    python benchmarks/infeasible_search.py run --jobs 16
    python benchmarks/infeasible_search.py report
"""
import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.special import ndtri
from scipy.stats import binomtest, qmc

from hone.benchmarks.outputs import output_problems
from hone.benchmarks.problems import Problem
from hone.benchmarks.runner import BAYESIAN_MAX_EVALUATIONS, SkipBudget, run_benchmark
from hone.config import BayesianConfig
from hone.optimizers._gp import GaussianProcess
from hone.optimizers.bayesian import (
    FIT_EVERY_UNTIL,
    REFIT_EVERY,
    UNCOMPUTED_MARGIN,
    BayesianOptimizer,
    _believed,
    _bilog,
    _log_expected_improvement,
)

BATCH_SIZES = (1, 4)
BUDGETS = (10, 20, 50, 100)
SEEDS = range(30)
CHECK_SEEDS = range(30, 60)  # the current method against itself, to see what the guard can resolve
TOLERANCE = 1e-4
GUARD = 1.1
ALPHA = 0.05
SOONER = 0.8  # a median first feasible design at most this share of the current method's
MORE_OFTEN = 0.1  # or at least this much more of the runs finding one
QMC_SAMPLES = 128
NOTHING_IMPROVES = -1e4  # B's score where no sample improves, below any log improvement, added to the log probability of feasibility
RUNS = Path(__file__).parent / "infeasible-search-runs.json"


class ViolationModel(BayesianOptimizer):
    """A: expected improvement in a Gaussian process of the bilog of the total violation, until something is feasible."""

    def reset(self) -> None:
        super().reset()
        self._violation_theta: np.ndarray | None = None
        self._violation_fitted_at = 0
        self._violation_model: tuple[int, GaussianProcess] | None = None

    def _fitted_violation(self, X: np.ndarray) -> GaussianProcess:
        """The model of every real evaluation, fitted once per set of them, as a batch's candidates share it."""
        if self._violation_model is not None and self._violation_model[0] == len(X):
            return self._violation_model[1]
        values = np.array(self._violations, dtype=np.float64)
        finite = np.isfinite(values)
        values[~finite] = max(values[finite].max(), UNCOMPUTED_MARGIN) if finite.any() else UNCOMPUTED_MARGIN
        model = GaussianProcess(X, _bilog(values), self._violation_theta)
        if self._violation_theta is None or len(X) <= FIT_EVERY_UNTIL or len(X) - self._violation_fitted_at >= REFIT_EVERY:
            model.fit(restarts=2 if self._violation_theta is None else 0, rng=self.rng)
            self._violation_theta, self._violation_fitted_at = model.theta, len(X)
        self._violation_model = (len(X), model)
        return model

    def _maximize_improvement(self, model, best, constraints, leaders):
        if best is not None or not constraints:
            return super()._maximize_improvement(model, best, constraints, leaders)
        evaluated = len(self._X)
        violation = self._fitted_violation(model.X[:evaluated])
        lowest = float(violation.z.min())
        for x in model.X[evaluated:]:  # candidates still pending or already chosen for this batch
            violation = _believed(violation, x)
        return self._search(lambda X: _log_expected_improvement(violation, X, lowest), leaders, model.X)


class SampledViolation(BayesianOptimizer):
    """B: quasi-Monte Carlo expected improvement in the total violation, from the models of each limit, until something is feasible.

    The samples are fixed for the run, so the acquisition is smooth enough for _refine's finite
    differences. Where no sample improves it would be exactly zero and give no direction, so
    there it falls back on the probability of feasibility, below every point that can improve.
    """

    def reset(self) -> None:
        super().reset()
        count = len(self.search_space.output_constraints)
        sobol = qmc.Sobol(count, seed=self.config.seed).random(QMC_SAMPLES) if count else np.zeros((QMC_SAMPLES, 0))
        self._draws = ndtri(np.clip(sobol, 1e-9, 1 - 1e-9))

    def _maximize_improvement(self, model, best, constraints, leaders):
        if best is not None or not constraints:
            return super()._maximize_improvement(model, best, constraints, leaders)
        lowest = min(v for v in self._violations if math.isfinite(v)) if any(math.isfinite(v) for v in self._violations) else math.inf
        log_feasibility = self._log_feasibility(constraints)
        return self._search(self._improvement(constraints, lowest, log_feasibility), leaders, model.X)

    def _improvement(self, constraints, lowest: float, log_feasibility: Callable) -> Callable[[np.ndarray], np.ndarray]:
        draws = self._draws

        def acquisition(X: np.ndarray) -> np.ndarray:
            violation = np.zeros((len(X), len(draws)))
            for j, constraint_model in constraints:
                mean, sd = constraint_model.predict_original(X)
                sampled = mean[:, None] + sd[:, None] * draws[:, j]
                # The inverse of the bilog, kept in range: anything past the lowest violation improves nothing anyway.
                violation += np.maximum(np.expm1(np.minimum(sampled, 50.0)), 0.0)
            improvement = np.maximum(lowest - violation, 0.0).mean(axis=1) if math.isfinite(lowest) else np.zeros(len(X))
            improves = improvement > 0
            scores = log_feasibility(X) + NOTHING_IMPROVES
            scores[improves] = np.log(improvement[improves])
            return scores

        return acquisition


def _factory(optimizer_class: type[BayesianOptimizer], size: int) -> Callable[[Problem, int, int], BayesianOptimizer]:
    def factory(problem: Problem, budget: int, seed: int) -> BayesianOptimizer:
        if budget > BAYESIAN_MAX_EVALUATIONS:
            raise SkipBudget(f"{budget} evaluations is more than Bayesian optimization is run with")
        return optimizer_class(problem.search_space, BayesianConfig(max_evaluations=budget, population_size=size, seed=seed))

    return factory


METHODS: dict[str, type[BayesianOptimizer]] = {"current": BayesianOptimizer, "a": ViolationModel, "b": SampledViolation}


def run(jobs: int, seeds: range, check_seeds: range, budgets: tuple[int, ...], problems: list[Problem], path: Path) -> None:
    """Every method's runs, written after each so a stopped run loses little. Each stage's time is printed, since a run doesn't record its own."""
    rows: list[dict] = []
    stages = [(f"{name}-{size}", name, size, seeds) for name in METHODS for size in BATCH_SIZES]
    stages += [(f"check-{size}", "current", size, check_seeds) for size in BATCH_SIZES]
    for label, method, size, stage_seeds in stages:
        start = time.perf_counter()
        runs = run_benchmark(problems, {label: _factory(METHODS[method], size)}, seeds=stage_seeds, budgets_per_dimension=budgets, jobs=jobs)
        rows += [
            {
                "problem": r.problem, "label": label, "budget": r.budget_per_dimension, "seed": r.seed,
                "error": r.error, "first_feasible": r.first_feasible,
            }
            for r in runs
        ]
        path.write_text(json.dumps(rows))
        print(f"{label}: {len(runs)} runs in {time.perf_counter() - start:.0f}s", flush=True)


def _holm(p_values: dict, alpha: float = ALPHA) -> set:
    """The keys significant at alpha after Holm's correction for testing them all."""
    significant = set()
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    for rank, (key, p) in enumerate(ordered):
        if p > alpha / (len(ordered) - rank):
            break
        significant.add(key)
    return significant


def _sign_test(alternative: list[float], current: list[float]) -> tuple[float, int, int]:
    """The exact two-sided sign test's p-value, and the seeds where the alternative came sooner and later; ties are dropped."""
    sooner = sum(a < c for a, c in zip(alternative, current))
    later = sum(a > c for a, c in zip(alternative, current))
    return (binomtest(sooner, sooner + later).pvalue if sooner + later else 1.0), sooner, later


def _first(row: dict) -> float:
    return math.inf if row["first_feasible"] is None else float(row["first_feasible"])


def _median_error(rows: list[dict]) -> float:
    return statistics.median(max(row["error"], TOLERANCE) for row in rows)


def _guard(alternative: dict, current: dict, problem: str, size: int, labels: tuple[str, str], budgets: list[int]) -> tuple[float, bool]:
    """The geometric mean of the ratio of median errors over budgets, and whether a cell found nothing where the current method did."""
    ratios, lost = [], False
    for budget in budgets:
        a = _median_error(alternative[problem, labels[0], budget])
        c = _median_error(current[problem, labels[1], budget])
        if math.isinf(a) and math.isfinite(c):
            lost = True
        ratios.append(1.0 if math.isinf(a) or math.isinf(c) else a / c)
    return math.exp(statistics.fmean(math.log(ratio) for ratio in ratios)), lost


def report(path: Path) -> str:
    """The decision by the design doc's rule, as markdown."""
    cells: dict[tuple, list[dict]] = {}
    for row in json.loads(path.read_text()):
        cells.setdefault((row["problem"], row["label"], row["budget"]), []).append(row)
    for rows in cells.values():
        rows.sort(key=lambda row: row["seed"])
    problems = sorted({key[0] for key in cells})
    budgets_of = {problem: sorted({key[2] for key in cells if key[0] == problem}) for problem in problems}
    lines = []
    alarms = []
    for size in BATCH_SIZES:
        for problem in problems:
            ratio, lost = _guard(cells, cells, problem, size, (f"check-{size}", f"current-{size}"), budgets_of[problem])
            if ratio > GUARD or lost:
                alarms.append(f"{problem} at batch size {size}: ×{ratio:.2f}" + (", found nothing where the other seeds did" if lost else ""))
    lines += ["## Noise check: the current method against itself on other seeds", ""]
    lines += [f"- {alarm}" for alarm in alarms] or ["Every problem within the guard."]
    lines.append("")

    for method in ("a", "b"):
        p_values = {}
        details = {}
        for size in BATCH_SIZES:
            for problem in problems:
                for budget in budgets_of[problem]:
                    alternative = cells[problem, f"{method}-{size}", budget]
                    current = cells[problem, f"current-{size}", budget]
                    first_a, first_c = [_first(r) for r in alternative], [_first(r) for r in current]
                    p, sooner, later = _sign_test(first_a, first_c)
                    share_a = sum(math.isfinite(r["error"]) for r in alternative) / len(alternative)
                    share_c = sum(math.isfinite(r["error"]) for r in current) / len(current)
                    median_a, median_c = statistics.median(first_a), statistics.median(first_c)
                    large = (median_a <= SOONER * median_c and median_a < median_c) or share_a - share_c >= MORE_OFTEN
                    p_values[problem, size, budget] = p
                    details[problem, size, budget] = (p, sooner, later, median_a, median_c, share_a, share_c, large)
        significant = _holm(p_values)
        gains = [key for key in significant if details[key][1] > details[key][2] and details[key][7]]
        harms = [key for key in significant if details[key][2] > details[key][1]]

        failures = []
        rows = ["| Problem | Batch | ×error |", "| --- | --- | --- |"]
        for size in BATCH_SIZES:
            for problem in problems:
                ratio, lost = _guard(cells, cells, problem, size, (f"{method}-{size}", f"current-{size}"), budgets_of[problem])
                if ratio > GUARD or lost:
                    failures.append((problem, size))
                flag = " **fail**" if ratio > GUARD or lost else ""
                rows.append(f"| {problem} | {size} | {ratio:.2f}{flag}{' (lost)' if lost else ''} |")

        verdict = "ships" if gains and not failures else "does not ship"
        lines += [f"## Method {method.upper()}: {verdict}", ""]
        lines += [f"{len(p_values)} cells; {len(gains)} with a gain, {len(harms)} significantly later, {len(failures)} guard failures.", ""]
        lines += ["| Problem | Batch | Budget | Sooner / later | Median first | Share found | Holm | Large |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for key in sorted(significant, key=lambda k: p_values[k]):
            p, sooner, later, median_a, median_c, share_a, share_c, large = details[key]
            lines.append(
                f"| {key[0]} | {key[1]} | {key[2]} | {sooner} / {later} | {median_a:g} vs {median_c:g} | {share_a:.2f} vs {share_c:.2f} | yes | {'yes' if large else 'no'} |"
            )
        lines += ["", *rows, ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "report", "smoke"])
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    if args.command == "run":
        run(args.jobs, SEEDS, CHECK_SEEDS, BUDGETS, output_problems(), RUNS)
    elif args.command == "smoke":
        problems = [p for p in output_problems() if p.name in {"pocket-2d-1in100", "g07", "tension-spring-outputs"}]
        path = Path("/tmp/infeasible-smoke.json")
        run(args.jobs, range(2), range(2, 4), (10,), problems, path)
        print(report(path))
    else:
        print(report(RUNS))


if __name__ == "__main__":
    main()
