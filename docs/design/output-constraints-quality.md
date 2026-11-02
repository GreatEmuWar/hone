# Output constraints, quality improvements: design

Status: in progress, for 0.15.0. Step 0, the measuring tools, is being built; the items below are hypotheses until measured.

[Output constraints](output-constraints.md) shipped in 0.14.0 with a list of known weaknesses. This release measures each one and fixes those the measurements justify. The rule throughout is measure first, with the pass criteria fixed before seeing the results.

## Decisions taken at the start

1. **The final round defaults to on when noise and output constraints combine.** Otherwise `Result.feasible` rests on one noisy draw of the chosen point. It costs budget, and on ordinary noisy runs the final round hasn't paid for itself, so its cost is measured here too.
2. **Noise with output constraints is for a single objective at first.** The multi-objective optimizers keep their refusal; `nsga2.py` and `multi.py` have noise paths of their own.
3. **Pass criteria.** A change to how a feasible design is found (item 1) ships if it finds one more often or sooner without worsening final error by more than ×1.1. `scale=` (item 3) ships only if the test shows a factor of 2 or more. Decisions are made over 30 seeds rather than the usual 10.

## Step 0: measuring tools, no optimizer changes

- `Summary` gains the share of runs that found a feasible design and the median evaluations to the first one, a miss counting as not found, so the median is infinite when most runs miss. `Run` records when each run first found one.
- New problems whose feasible region is known or tiny: balls of known volume, so the feasible fraction is exact; two limits whose boundaries cross at the optimum, so the active limit switches; a deceptive one, a wide basin where the violation falls but never reaches zero beside a small feasible pocket, to give the current method, which explores where the models are least sure, a fair chance; and g06, g07 and g10 from the CEC 2006 constrained suite, whose feasible fractions are measured rather than remembered.
- Noise on problems with output constraints crashed the runner: `_with_noise` added a float to a dict, and the true value it scores by was a dict. Noise is now added to the objective and to each output separately, and a run is scored by the true value and the true feasibility of the design it returns.
- Bayesian optimization runs at batch sizes 4 and 8 as well as 1 on these problems. The `bayesian` factory only used the default batch size, so batches that bunch together were invisible.
- `run_benchmark` only parallelizes where it can fork, so full runs are serial on Windows; they run in the Docker image, WSL or CI.

What the first run of the suite showed, at 0.14.0 (details in `benchmarks/README.md`):

- Maximizing the probability of feasibility already finds a feasible design on every problem with a tiny region, g06, g07 and g10 included, in every run within about 30 evaluations, where no other optimizer finds one at the same budgets. Item 1's alternative has to beat that, so its pass criterion stands as written and the likely outcome is that the current method survives.
- The deceptive pocket defeats everything at small budgets, and CMA-ES at large ones: it follows the violation into the basin and sits there without restarting. That makes pocket-2d-1in100 the direct test for item 5, and the problem on which item 1's alternatives would have to show a gain.
- Batches of 8 are 10 to 300 times worse than one at a time at 10 evaluations per dimension, confirming item 6's cost; the hypothesis about the believer is unchanged and still to be confirmed by logging u.

## 1. Before anything is feasible

With every margin positive, the probability that a design is feasible is Φ(−μ/σ) for each limit, and log Φ(−μ/σ) ≈ −μ²/(2σ²) in the tail, so a large σ wins and the search goes to the corners, where the models are least sure. Two alternatives are compared over 30 paired seeds:

- A: one new Gaussian process on the bilog of the total violation, reusing `_log_expected_improvement`. Simple; the total violation has kinks where the active limit changes.
- B: quasi-Monte Carlo expected improvement in the total violation from the existing per-limit models. Bounded integrand; needs fixed samples or the noise swamps the finite differences in `_refine`; exactly zero where no sample improves.

The simplest method that finds a feasible design more often or sooner, without worsening final error by more than ×1.1, is kept. The current method may survive: it already handles the spring at 10 evaluations per dimension.

## 2. Noise with output constraints

Average the outputs, not the margins: the limit is on the true output, so a limit judged on the average output is the right one, and the average of the margin converges to the margin of the average anyway.

The gap: Bayesian optimization refuses to evaluate a point it has seen, continuous CMA-ES essentially never repeats one, and the final round is off by default, so nearly every point is judged on one draw. Picking the best apparently feasible point also favours luck on both value and margin, and optima sit on the boundary, so a true margin of zero is a coin flip however often it is averaged. The benchmark reports true feasibility alongside error.

Code to change: `_ranked_keys` sorts on average value alone and needs (violation of the averages, average); `result()` takes `feasible` and `best_outputs` from `_best`, while `best_params` can be a different point under noise; `_refresh_averages` must refresh violations; the noise branch of Bayesian optimization picks its incumbent from all points, infeasible included.

## 3. `scale=` for limits that aren't numbers, gated

With no scale, raw-unit margins reach the bilog transform, and standardizing afterwards can't undo it, so the model's shape depends on the units. Test `stress - 250 <= 0` against `stress <= 250`; under about ×2 it doesn't ship. Runs after items 1 and 2. If it ships, it is explicit, since inferred scales would change the ranking mid-run; an expression can't take a keyword, so something like `(stress <= 50 * width).scaled(100)`.

## 4. `choose_optimizer` with output constraints

On the pressure vessel with its limits on outputs, Bayesian optimization at 200 evaluations (31.6) beats CMA-ES at 4,000 (277); the welded beam goes the other way (0.0112 against 3e-7). Measure seconds per candidate up to 500 evaluations (k + 1 Gaussian processes at O(n³)) and state the rule in those terms. Count step 0's problems too. Change the rule only for spaces with output constraints.

## 5. CMA-ES flat-run restart

Track the best violation while infeasible with the same flatness test. `_recent_bests` only grows on feasible generations, so its window can span a long infeasible gap: clear the history when the run moves between infeasible and feasible. Add a direct test that a flat violation triggers a restart.

## 6. Batches that bunch together

Hypothesis: the kriging believer keeps the mean at x and only shrinks the variance; where μ is far below the best, expected improvement is about best − μ, nearly independent of σ, so the next candidate lands adjacent. A large learned noise term worsens it. Confirm by logging u at the chosen points before picking a fix. Candidates: the constant liar (Ginsbourger et al., 2010), local penalization (González et al., 2016), a minimum distance in length scales. Measure the smallest pairwise distance in a batch and the error at sizes 4 and 8. Guard: batch size 1 with nothing pending must be exactly unchanged.

## Delivery

Separate pull requests: (1) step 0, the measuring tools with a new baseline; (2) item 1; (3) item 2; (4) items 5 and 6; (5) item 4 last, because the earlier changes alter what it measures. Item 3 only if its test shows a need.
