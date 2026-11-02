"""Benchmark problems and a runner for comparing optimizers. Run `python -m hone.benchmarks --help`."""
from .conditional import conditional_problems
from .constrained import constrained_problems
from .ml import ml_problems
from .multi import MultiProblem, multi_problems, run_multi_benchmark, summarize_multi
from .outputs import noisy_output_problems, output_problems
from .problems import Problem, noisy_problems, quick_problems, standard_problems
from .runner import (
    BUDGETS,
    DEFAULT_OPTIMIZERS,
    Run,
    SkipBudget,
    Summary,
    batch_optimizers,
    load_summaries,
    regressions,
    run_benchmark,
    save_summaries,
    summarize,
    to_markdown,
    wins,
)
