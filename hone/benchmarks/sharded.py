"""A benchmark shared among several machines: each runs one shard of it, and the shards are put together afterwards.

A spec is a Python file that defines what to run:

    def problems() -> list[Problem]: ...
    def optimizers() -> dict[str, OptimizerFactory]: ...
    SEEDS = 30                      # seeds 0 to SEEDS - 1
    BUDGETS = (10, 20)              # evaluations per dimension
    def report(runs: list[Run]) -> str: ...   # optional: markdown added to the results, such as a check of pass criteria

Run one shard of it, and put the shards together:

    python -m hone.benchmarks.sharded run benchmarks/specs/smoke.py --shard 0 --of 4 --out runs-0.json
    python -m hone.benchmarks.sharded merge benchmarks/specs/smoke.py runs-*.json --summaries s.json --markdown r.md

A spec whose problems() returns multi-objective problems (MultiProblem) runs through run_multi_benchmark,
and its summaries are those of summarize_multi; everything else is the same.

merge refuses to proceed unless the shards hold exactly the runs the spec makes, none missing and
none twice. The Measure workflow in .github/workflows/measure.yml does both on GitHub's runners.
"""
import argparse
import importlib.util
import json
import sys
import time
from importlib.metadata import version
from pathlib import Path
from types import ModuleType
from typing import Any

from .multi import MultiProblem, MultiRun, multi_markdown, multi_tasks, run_multi_benchmark, summarize_multi
from .runner import Run, benchmark_tasks, run_benchmark, save_summaries, summarize, to_markdown


def load_spec(path: str | Path) -> ModuleType:
    """The spec file at path, imported as a module, checked for what it must define."""
    loaded = importlib.util.spec_from_file_location(f"_benchmark_spec_{Path(path).stem}", path)
    if loaded is None or loaded.loader is None:
        raise ValueError(f"can't import {path}")
    module = importlib.util.module_from_spec(loaded)
    loaded.loader.exec_module(module)
    missing = [name for name in ("problems", "optimizers", "SEEDS", "BUDGETS") if not hasattr(module, name)]
    if missing:
        raise ValueError(f"{path} doesn't define {', '.join(missing)}")
    return module


def is_multi(spec: ModuleType) -> bool:
    """Whether the spec's problems have several objectives."""
    return any(isinstance(problem, MultiProblem) for problem in spec.problems())


def expected_runs(spec: ModuleType) -> set[tuple[str, str, int, int]]:
    """(problem, optimizer, budget per dimension, seed) of every run the spec makes."""
    problems = spec.problems()
    make = multi_tasks if is_multi(spec) else benchmark_tasks
    tasks = make(problems, spec.optimizers(), range(spec.SEEDS), spec.BUDGETS)
    return {(problems[index].name, name, budget, seed) for index, name, budget, seed in tasks}


def run_shard(spec: ModuleType, shard: int, of: int, jobs: int = 1) -> list[Run] | list[MultiRun]:
    run = run_multi_benchmark if is_multi(spec) else run_benchmark
    return run(spec.problems(), spec.optimizers(), range(spec.SEEDS), spec.BUDGETS, jobs=jobs, shard=(shard, of))


def merge(spec: ModuleType, shards: list[list[Any]]) -> list[Run] | list[MultiRun]:
    """Every shard's runs together, checked against the runs the spec makes."""
    runs: list[Any] = [run for shard in shards for run in shard]
    keys = [(r.problem, r.optimizer, r.budget_per_dimension, r.seed) for r in runs]
    expected = expected_runs(spec)
    twice = len(keys) - len(set(keys))
    missing, extra = expected - set(keys), set(keys) - expected
    if twice or missing or extra:
        raise ValueError(f"the shards don't match the spec: {len(missing)} runs missing, {len(extra)} not in it, {twice} repeated")
    return sorted(runs, key=lambda r: (r.problem, r.optimizer, r.budget_per_dimension, r.seed))


def _save_runs(runs: list[Run] | list[MultiRun], path: str | Path) -> None:
    Path(path).write_text(json.dumps([run.model_dump() for run in runs]) + "\n")


def _load_runs(path: str | Path, multi: bool = False) -> list[Run] | list[MultiRun]:
    kind: Any = MultiRun if multi else Run
    return [kind.model_validate(run) for run in json.loads(Path(path).read_text())]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m hone.benchmarks.sharded", description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run one shard of a spec")
    run.add_argument("spec")
    run.add_argument("--shard", type=int, required=True, help="which shard, from 0")
    run.add_argument("--of", type=int, required=True, help="how many shards in all")
    run.add_argument("--jobs", type=int, default=1, help="share this shard's runs among this many processes (where they can fork)")
    run.add_argument("--out", required=True, help="write the shard's runs as JSON to this file")
    put = commands.add_parser("merge", help="put shards together, checked against the spec")
    put.add_argument("spec")
    put.add_argument("shards", nargs="+", help="the shards' JSON files")
    put.add_argument("--runs", help="write every run as JSON to this file")
    put.add_argument("--summaries", help="write the summaries as JSON to this file")
    put.add_argument("--markdown", help="write the results table, and the spec's report, to this file")
    put.add_argument("--tolerance", type=float, default=1e-4, help="error at or below which a run counts as solved")
    args = parser.parse_args(argv)
    spec = load_spec(args.spec)

    if args.command == "run":
        start = time.perf_counter()
        runs = run_shard(spec, args.shard, args.of, args.jobs)
        _save_runs(runs, args.out)
        print(f"shard {args.shard} of {args.of}: {len(runs)} runs in {time.perf_counter() - start:.0f}s")
        return 0

    try:
        multi = is_multi(spec)
        runs = merge(spec, [_load_runs(path, multi) for path in args.shards])
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    if multi:
        summaries: Any = summarize_multi(runs)  # type: ignore[arg-type]
        text = multi_markdown(summaries)
    else:
        summaries = summarize(runs, args.tolerance)  # type: ignore[arg-type]
        text = to_markdown(summaries, args.tolerance)
    if hasattr(spec, "report"):
        text += "\n" + spec.report(runs)
    print(text)
    if args.runs:
        _save_runs(runs, args.runs)
    if args.summaries:
        if multi:
            Path(args.summaries).write_text(json.dumps([s.model_dump() for s in summaries]) + "\n")
        else:
            save_summaries(summaries, args.summaries)
    if args.markdown:
        Path(args.markdown).write_text(
            f"# Benchmark results\n\nGenerated by `python -m hone.benchmarks.sharded` from `{args.spec}` with hone "
            f"{version('hone-python')}: {len(runs)} runs in {len(args.shards)} shards.\n\n{text}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
