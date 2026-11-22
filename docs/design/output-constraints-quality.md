# Output constraints, quality improvements: design

Status: in progress, for 0.15.0. Step 0, the measuring tools, and step 1b, the cost of a Bayesian candidate, are done, item 2, noise with output constraints, is built, and item 6, batches that bunch together, is fixed with the constant liar, and item 7, a margin of safety for limits under noise, is built; item 1 was measured and closed, the current method staying, and item 5 was measured and not shipped. The rest are hypotheses until measured.

[Output constraints](output-constraints.md) shipped in 0.14.0 with a list of known weaknesses. This release measures each one and fixes those the measurements justify. The rule throughout is measure first, with the pass criteria fixed before seeing the results.

## Decisions taken at the start

1. **The final round defaults to on when noise and output constraints combine.** Otherwise `Result.feasible` rests on one noisy draw of the chosen point. It costs budget, and on ordinary noisy runs the final round hasn't paid for itself, so its cost is measured here too.
2. **Noise with output constraints covers one objective and several, in this release.** A single objective comes first, as item 2, to settle the averaging rules and measure the final round's cost; several objectives follow in a pull request of their own, after item 7, since `multi.py`, `nsga2.py` and `multi_bayesian.py` need their own averaging, a final round that doesn't exist yet, and a benchmark.
3. **Pass criteria.** A change to how a feasible design is found (item 1) ships if it finds one more often or sooner without worsening final error by more than ×1.1. `scale=` (item 3) ships only if the test shows a factor of 2 or more. Decisions are made over 30 seeds rather than the usual 10.
4. **Criteria are set above measured noise.** Item 1 found that 30 unpaired seeds can't resolve ×1.1 between a method and itself, so later criteria are set above a paired noise floor, or judge by absolute outcomes such as lost designs and first-feasible rates.

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

## Step 1b: the cost of a Bayesian candidate

Added by the roadmap's review of 2 Nov 2026. A timing harness, `python -m hone.benchmarks --cost`, gives seconds per candidate against evaluations (50 to 500) and limits on outputs (0 to 8), at batch sizes 1 and 8, with one profiled run, saved as `benchmarks/cost-summaries.json`. It sets a ceiling: every later item reports its seconds per candidate against the table, and may not raise it past ×1.2 at 500 evaluations without sign-off. `--baseline benchmarks/cost-summaries.json` checks it. 0.16.0's first step starts from this table rather than measuring its own.

Each cell is the mean over 3 seeds. One seed alone varies by up to about 15% between seeds at 500 evaluations, which is too close to ×1.2 for a change that moves the run the way a new seed does; the same seed repeats within about 5%.

Measured on an Apple M2 under macOS 27.2 at 0.14.0 with item 2 (details in `benchmarks/README.md`): at 500 evaluations a candidate takes 0.08 seconds with no limits and 0.78 with eight, about the same in batches of 8, and the cold fit about 10 seconds with eight limits. Searching the models costs more than fitting them: about four fifths of a profiled run of g07 at 200 evaluations, and most of the steady-state time at 500 with eight limits, where refitting takes about a third. An operating system update on the same machine moved the times by 3% to 25%, so the table is remeasured whenever the system changes. That bears on 0.16.0's first step, whose first planned change is refitting less often.

## 1. Before anything is feasible: measured, the current method stays

Step 0's baseline already pointed this way: the current rule found a feasible design in every run on every problem with a tiny feasible region. The full test below, over 30 paired seeds, confirmed it, and neither alternative ships. The hypothesis and the alternatives stay below, with their numbers, as the record of what was tried.

With every margin positive, the probability that a design is feasible is Φ(−μ/σ) for each limit, and log Φ(−μ/σ) ≈ −μ²/(2σ²) in the tail, so a large σ wins and the search goes to the corners, where the models are least sure. Two alternatives are compared over 30 paired seeds:

- A: one new Gaussian process on the bilog of the total violation, reusing `_log_expected_improvement`. Simple; the total violation has kinks where the active limit changes.
- B: quasi-Monte Carlo expected improvement in the total violation from the existing per-limit models. Bounded integrand; needs fixed samples or the noise swamps the finite differences in `_refine`; exactly zero where no sample improves.

The simplest method that finds a feasible design more often or sooner, without worsening final error by more than ×1.1, is kept. The current method may survive: it already handles the spring at 10 evaluations per dimension.

### The test, fixed before any alternative was run

Written before A or B existed. The only results seen by then were step 0's 10 seeds of the current method, which is why the deceptive pocket is named above.

**Runs.** The current method, A and B, in Bayesian optimization only, on all 14 problems of `output_problems()`, at batch sizes 1 and 4, at 10, 20, 50 and 100 evaluations per dimension up to its limit of 200 evaluations, with seeds 0 to 29. The same seed gives each method the same first points, so the seeds are paired. A cell is one problem, batch size and budget.

**Noise check.** The current method also runs with seeds 30 to 59, and is scored against itself on seeds 0 to 29 by the guard below. If it fails its own guard, ×1.1 is finer than 30 seeds can resolve: no decision is made from the guard, and the user is asked.

**Gain.** An alternative needs at least one cell where both hold:

- It finds a feasible design sooner across seeds: an exact two-sided sign test on the paired evaluations to the first feasible design, a miss counting as infinity and ties dropped, is significant at 0.05 after Holm's correction over all of that alternative's cells, and favours the alternative.
- The gain is large enough to matter: its median first feasible design comes at least 20% sooner, or it finds one in at least 0.1 more of the runs.

**Guard.** For every problem at each batch size, the geometric mean over budgets of the alternative's median error divided by the current method's is at most 1.1. Errors below the benchmark's tolerance of 1e-4 count as 1e-4, so two solved runs tie. In a cell where both medians are infinite, or only the current method's is, the ratio counts as 1; a cell where only the alternative's is infinite fails the guard outright.

**Decision.** An alternative that shows a gain and passes every guard ships; if both do, A, the simpler. Otherwise the current method stays, and this section records why. Seconds per candidate are reported but don't decide. These numbers aren't changed after the results are seen: a result that seems to call for it goes to the user.

If an alternative ships, `MultiObjectiveBayesian` takes it too, since it searches on the same probability until something is feasible. No multi-objective benchmark has output constraints yet, so there it is tested for working rather than measured; the benchmark for noise with several objectives measures it.

### Result: the current method stays

Neither alternative ships, and no optimizer code changed. All 9,840 runs finished; they are in `benchmarks/infeasible-search-runs.json`, the report is in `benchmarks/infeasible-search-results.md`, and `benchmarks/infeasible_search.py` reproduces both.

The rule didn't reach this verdict by itself. The noise check failed, below, so the rule made no decision from the guard and the user was asked. The user chose to keep the current method without a run with more seeds, on the evidence that follows. A 2-seed smoke test of the script was also seen after the test was committed; it changed nothing here.

| | A | B |
| --- | --- | --- |
| Cells with a gain, of 82 | 4 | 7 |
| Cells significantly later | 20 | 8 |
| Guard rows failed, of 28 | 11 | 9 |
| Cells that lost the feasible design | 17 | 5 |

- **The gains are real and narrow.** On ball-2d-1in100 at batch size 1, A and B both reach a first feasible design at a median of 9 evaluations against 14.5. On tension-spring-outputs at batch size 4, B does at 16 against 20. The gain cells are these results at different budgets, since a first feasible design doesn't depend on the budget once it is found.
- **The deceptive pocket defeats both.** On pocket-2d-1in100 at 100 evaluations per dimension, the current method finds a feasible design in every run at both batch sizes. A and B find one in 1 run of 30 (3%) at every budget: the same seed, whose first spread of points landed in the pocket, so neither ever found it by searching. The likely reason is the one the problem was built to expose: following the falling violation into the basin.
- **A also misses thin regions.** On g06 at batch size 1, A finds a feasible design in 0%, 0%, 3% and 7% of runs at 10, 20, 50 and 100 evaluations per dimension, and at batch size 4 in 0%, 0%, 13% and 43%; the current method finds one in 100% at batch size 1 and in 93% to 100% at batch size 4. On g10, A finds none in any run at either batch size, against 100%; on g07 at batch size 4, 23% and 80% against 100%. A possible reason, not tested: a model of the total violation can't predict below zero, so it has no reason to aim inside a thin region.
- **B loses less, but still loses.** Beyond the pocket, it finds a design in 37% of runs on g06 at batch size 4 and 10 evaluations per dimension, against 93%, and comes later on g07 (a median of 32.5 against 28 at batch size 1, and 42 against 34 at batch size 4).
- **Cost didn't decide.** At 30 seeds on 14 processes, the batch size 1 stages took 2,528 s for the current method, 2,185 s for A and 2,478 s for B; the batch size 4 stages took 2,381 s, 1,807 s and 2,356 s. These are whole stages on a shared machine, not seconds per candidate.

Lost designs are the evidence the verdict rests on, rather than the ×1.1 ratios. No lost cell appeared in the noise check, against 17 for A and 5 for B. That comparison was made after the results were seen, and isn't part of the rule fixed beforehand.

### Finding: what 30 seeds can resolve

Separately from the verdict: the current method on seeds 30 to 59, scored against itself on seeds 0 to 29, failed its own ×1.1 guard on 5 of 28 rows: tension-spring-outputs ×1.40, welded-beam-outputs ×1.36, ball-2d-1in10 ×1.24 and pocket-2d-1in100 ×1.20 at batch size 4, and welded-beam-outputs ×1.13 at batch size 1. So a median-error ratio near ×1.1 can't be resolved at 30 seeds, especially at batch size 4.

This is one comparison, between unpaired sets of seeds, while the methods' comparisons are paired, so it may overstate the noise they face; this data can't say by how much.

For the items that follow, the noise estimate is designed in from the start. Both methods run on the same seeds, and the unchanged method also runs against itself on a paired draw: the same seed and first generation, with a different random stream for the search after it, since an identical seed would otherwise repeat the run exactly. Each ratio is then set above that floor, or the criterion uses absolute outcomes, such as lost designs and first-feasible rates, where a ratio can't be resolved.

## 2. Noise with output constraints

Average the outputs, not the margins: the limit is on the true output, so a limit judged on the average output is the right one, and the average of the margin converges to the margin of the average anyway.

The gap: Bayesian optimization refuses to evaluate a point it has seen, continuous CMA-ES essentially never repeats one, and the final round is off by default, so nearly every point is judged on one draw. Picking the best apparently feasible point also favours luck on both value and margin, and optima sit on the boundary, so a true margin of zero is a coin flip however often it is averaged. The benchmark reports true feasibility alongside error.

Code to change: `_ranked_keys` sorts on average value alone and needs (violation of the averages, average); `result()` takes `feasible` and `best_outputs` from `_best`, while `best_params` can be a different point under noise; `_refresh_averages` must refresh violations; the noise branch of Bayesian optimization picks its incumbent from all points, infeasible included.

Built. `Optimizer` keeps every set of outputs seen at each point beside its values, averages them by output, and judges the violation on the averages; `_ranked_keys` sorts by (violation of the averages, average value); the result, the progress and the threshold come from the chosen point's averages; and `_refresh_averages`, which the genetic optimizer calls before breeding, refreshes outputs and violations too. Bayesian optimization keeps each row's key, re-judges every row by its point's current averages before proposing, and takes its incumbent from the rows that pass; with none, it looks for a feasible design, as without noise. The final round defaults on through `final_evaluations=None`: a tenth of `max_evaluations`, at least 10 and at most half. `Result.infeasible` still counts the draws that broke a limit. The optimizers of several objectives keep their refusal, by `supports_noisy_outputs`.

Measured on the three noisy engineering designs over 30 seeds at 10, 20, 50 and 100 evaluations per dimension, three ways: the new default (averaging, final round on), averaging alone (`final_evaluations=0`), and believing every draw (`noise=False`). The share of runs whose returned design truly satisfies every limit, default / averaging alone / every draw believed:

| Evaluations per dimension | Bayesian, vessel | Bayesian, spring | Bayesian, beam | CMA-ES, vessel | CMA-ES, spring | CMA-ES, beam |
| ---: | --- | --- | --- | --- | --- | --- |
| 10 | 87% / 63% / 67% | 90% / 100% / 97% | 93% / 63% / 77% | 100% / 100% / 100% | 40% / 57% / 57% | 97% / 90% / 90% |
| 20 | 60% / 53% / 30% | 80% / 70% / 67% | 80% / 33% / 53% | 93% / 93% / 93% | 80% / 83% / 83% | 97% / 97% / 97% |
| 50 | 43% / 17% / 7% | 50% / 47% / 30% | 70% / 40% / 33% | 97% / 97% / 97% | 100% / 93% / 93% | 100% / 97% / 97% |
| 100 | | | | 100% / 73% / 73% | 100% / 97% / 97% | 100% / 70% / 70% |

- **The final round is what makes the difference, and mostly for Bayesian optimization.** It converges onto the boundary of a limit, where one draw passes half the time, and it never evaluates a point twice during the search, so averaging alone changes little for it; the final round's repeated draws are what catch a design that only seemed to pass. CMA-ES and random search never repeat a point either, so averaging alone leaves their results exactly as they were, and the final round shows at 100 evaluations per dimension, where CMA-ES too has reached the boundary: 100% against 73% on the vessel and the beam, with lower error. Decision 1 stands.
- **Its cost is the budget it takes.** On the spring at 10 evaluations per dimension the floor of 10 takes a third of the 30 evaluations, and the share of feasible results for CMA-ES, the genetic optimizer and random search falls from 57% to 40%. Elsewhere median errors are up to about 15% higher than with the whole budget searched, the price of the re-evaluations.
- **Bayesian optimization's feasible share still falls with the budget**, to 43% to 70% at 50 evaluations per dimension even with the final round. The better it finds the optimum, the closer the design sits to the boundary, and a true margin of zero is a coin flip however often it is averaged. A margin of safety under noise, such as requiring a design's average margin to clear the limit by its standard error, is the next question for noise with output constraints; it isn't part of this release.
- **The genetic optimizer loses by running with `noise=True` here.** Its noise settings, a fixed mutation of 0.1 and no adaptive step, were chosen on the noisy suite, whose noise hides the differences near the optimum; at about 1% noise they cost more than averaging gains, and its median errors at 50 and 100 evaluations per dimension are 1.5 to 3 times those of believing every draw. Which settings suit small noise is a question for the noisy suite, not this one.

## 3. `scale=` for limits that aren't numbers, gated

With no scale, raw-unit margins reach the bilog transform, and standardizing afterwards can't undo it, so the model's shape depends on the units. Test `stress - 250 <= 0` against `stress <= 250`; under about ×2 it doesn't ship. Runs after items 1 and 2. If it ships, it is explicit, since inferred scales would change the ranking mid-run; an expression can't take a keyword, so something like `(stress <= 50 * width).scaled(100)`.

## 4. `choose_optimizer` with output constraints

On the pressure vessel with its limits on outputs, Bayesian optimization at 200 evaluations (31.6) beats CMA-ES at 4,000 (277); the welded beam goes the other way (0.0112 against 3e-7). Measure seconds per candidate up to 500 evaluations (k + 1 Gaussian processes at O(n³)) and state the rule in those terms. Count step 0's problems too. Change the rule only for spaces with output constraints.

### The current rule on suites it wasn't chosen from

The rule was set from the standard suite. Checked on 21 Nov 2026 against the saved summaries of the ML, constrained and outputs suites, 10 seeds each, by asking in each cell whether the rule's pick has the lowest median error of the optimizers run. Errors below 1e-8 count as equal, as in the overview, and batched Bayesian runs count as Bayesian.

- **ML tuning: 13 of 15 cells.** Both misses are at 10 evaluations per dimension (20 and 80 evaluations), by ×1.5, which is one misclassified sample in the validation set. The rule holds.
- **Constrained, limits on inputs and the engineering designs: 57 of 65.** Past 200 evaluations the misses are:
  - the boundary ellipsoid at 50 per dimension in 5 and 10 dimensions, where CMA-ES is ×2.1 and ×3.3 behind the genetic optimizer;
  - the pressure vessel, which has integer variables, so the rule picks the genetic optimizer. CMA-ES beats it by ×1.07 to ×1.09, and with its limits on outputs by ×1.7 to ×2.2.
- **Outputs, step 0's problems and the engineering designs with limits on outputs: 64 of 70.** The misses are:
  - pocket-2d at 20 and 40 evaluations, where no optimizer's median run is feasible;
  - pocket-2d at 2,000, where CMA-ES is ×479 behind the genetic optimizer;
  - the pressure vessel at 400 and 4,000, where CMA-ES beats the genetic pick by ×2.2 and ×2.5;
  - the spring at 300, by ×1.04.

Up to 200 evaluations the rule's pick, Bayesian optimization, is best in every constrained and outputs cell but the two pocket cells where nothing is feasible. The question item 4 asks is past 200, where Bayesian optimization isn't run.

**Observations, recorded and not acted on:**
- **The pressure vessel.** CMA-ES beats the genetic optimizer the rule picks for its integer variables: by ×1.07 at 400 evaluations and ×1.09 at 4,000 with limits on inputs, and by ×2.2 (1,300 against 2,870) and ×1.7 to ×2.5 (193 to 277 against 481) at 400 and 4,000 with limits on outputs. No other problem has both integers and limits on outputs, so a rule from it would rest on one problem.
- **pocket-2d at 2,000 evaluations.** CMA-ES's median error is 5.26, ×479 the genetic optimizer's 0.011.
- **The boundary ellipsoid at 50 per dimension.** CMA-ES is behind the genetic optimizer: 33.1 against 15.4 in 5 dimensions (×2.1), and 44.3 against 13.6 in 10 (×3.3).

The last two are CMA-ES behind the genetic optimizer where the feasible region is thin or its optimum lies on a boundary. They go to the roadmap as an experiment, not into this item.

### The design

With output constraints, Bayesian optimization stays the choice up to B evaluations, past the 200 it has without them. B is chosen from 300 and 500 by the measurement below, or stays at 200. Nothing past 500 is considered, since step 1b measured cost only that far. Without output constraints the rule is unchanged. So is a choice parameter with `choose_n`, which still rules Bayesian optimization out. `choose_pareto_optimizer` is unchanged too.

**The cost, stated in seconds.** From `benchmarks/cost-results.md` (10 dimensions, batches of 1), a candidate at 500 evaluations costs:
- 0.15 s with 1 limit;
- 0.41 s with 4 limits;
- 0.78 s with 8 limits.

At 200 evaluations the figures are 0.04 s, 0.10 s and 0.18 s.

Summed over a run, Bayesian optimization's own time comes to these totals. They are interpolated linearly between the measured budgets, which overstates them, since the cost grows faster than linearly:

| Evaluations | 1 limit | 4 limits | 8 limits |
| ---: | ---: | ---: | ---: |
| 200 | 5 s | 11 s | 19 s |
| 300 | 11 s | 27 s | 47 s |
| 500 | 35 s | 88 s | 163 s |

*Found on 21 Nov 2026: the harness understates a real run.* A full run of Bayesian optimization on ball-2d-1in10 at 500 evaluations spent 145 s proposing candidates. The harness's curve on the same machine (2 dimensions, 1 limit) sums to about 77 s. Profiling puts 90% of the run in refining the acquisition with L-BFGS-B. Both make about 5.5 starts per candidate, but a start takes about 121 steps in the real run against 54 in the harness. A converged run's points cluster near the optimum, so its acquisition surface is sharper than the harness's history of points spread over the space. Main is not slower than 0.14.0 on the same machine: 0.46 s against 0.50 s per candidate at 500 evaluations, 10 dimensions and 1 limit. The harness stays the reference for the ×1.2 ceiling, which is relative. The totals above are superseded for users: the totals the docs state come from timed real runs on GitHub's runners (below), not from the harness.

The user-facing docs state the rule with the total per run, as "at most about", so a user whose objective is cheap can see when to choose otherwise. The figures are for 10 dimensions on the machine in `cost-results.md`.

### The measurement

- **Problems.** Every problem in `output_problems()`: step 0's 11 and the three engineering designs, 14 in all.
- **Budgets.** 300 and 500 evaluations on every problem, fixed totals rather than per dimension, since the rule is stated in evaluations.
- **Runs.** 30 seeds each, for Bayesian optimization, CMA-ES and the genetic optimizer. The three noisy designs are run too, for the record, and not judged.
- **Where it runs.** A spec on the Measure workflow, which lifts the benchmark's cap of 200 on Bayesian optimization for this spec only. The default benchmark's cap stays at 200.
- **Timing, for the docs, not a criterion.** The benchmark's runs are many to a runner, so they aren't timed. A second spec times Bayesian optimization alone:
  - each of the 14 problems, 5 seeds, at 500 evaluations, with the cumulative seconds also recorded at evaluation 300 in the same run;
  - one run at a time per runner, on shards spread over GitHub's ubuntu-latest runners.

  Nothing in Bayesian optimization is set from the budget without noise, so the first 300 evaluations of a 500 run are a 300 run. It reports the median and 90th percentile per problem at 300 and 500, with each runner's processor, since the hardware varies. If the 90th percentile at 500 with 8 limits is over about 10 minutes, that goes to the user with the verdict.

The rule's current pick in a cell is CMA-ES, or the genetic optimizer if the space has integer or choice variables.

### Pass criteria, fixed before measuring

B isn't held out. It is chosen and judged on the same 14 problems, unlike step 1's check. The number of problems is the statistic, as it was when the rule was first set. Only 300 and 500 are measured, so a cutoff between them rests on the result being monotone in the budget.

Criteria 1 and 2 as below were reviewed by Hone Applied Mathematician and decided by the user on 21 Nov 2026, before measuring, with ×2 as the size of a loss that goes to the user.

At each budget b, 300 and 500, the unit is the problem:

1. **Error.** Bayesian optimization has a lower median error than the current pick on at least 11 of the 14 problems, and on at least 4 of the 6 that aren't step 0's shapes: g06, g07, g10, the pressure vessel, the spring and the welded beam. Errors below 1e-8 count as equal, and a tie counts against it.

   The problems aren't independent: step 0's 8 shapes (balls, corners and pockets) are about 3 families, so there are about 9 to 12 independent units. The second count keeps the shapes from carrying the verdict together. If the problems were independent and the two optimizers equally good, 11 or more of 14 would happen by chance with probability 0.029, from binomial(14, ½). If Bayesian optimization truly wins each problem 8 times in 10, the criterion passes with probability 0.70; at 9 in 10, 0.96.
2. **Feasibility.** On a problem where Bayesian optimization's share of truly feasible runs is more than 3 of 30 below the current pick's:
   - it **fails** if the pick's share is at least 27 of 30;
   - it **goes to the user** if the pick's share is under 27 of 30.

   The two shares are unpaired. If they are truly equal, the criterion trips by chance with probability 0.6% at 97%, 6% at 90%, 13% at 80% and 18% at 50%. It catches a true deficit of 10 points 35% of the time and of 20 points 88% of the time. So it detects about 20 points, not 10.
3. **Size of a loss.** On a problem where Bayesian optimization's median error is more than ×2 the pick's, the numbers go to the user, even when criterion 1 passes. Infeasible runs count as an infinite error.

   For each problem, a two-sided Mann–Whitney test at 0.05 is reported, with infeasible runs as infinite, and does not bind. It is exact at 30 against 30. About 1 of the 14 problems comes out clear by chance.
4. **Nothing else moves.** For budgets from 1 to 10,000, a test checks that each of these returns the same class with the same config as before:
   - `choose_optimizer` on every search space without output constraints;
   - `choose_optimizer` on a space with a choice parameter that has `choose_n`, with or without output constraints;
   - `choose_pareto_optimizer` on every space.

**Which B ships.** A budget *passes* if it meets criterion 1 and fails nothing under criterion 2. It *goes to the user* if it passes but has a loss under criterion 3 or a feasibility trip that goes to the user under criterion 2. In this order:

| | Outcomes | Ships |
| --- | --- | --- |
| 1 | A budget that would decide B goes to the user | The user decides, with that budget's numbers |
| 2 | b = 300 and b = 500 both pass | 500 |
| 3 | Only b = 300 passes | 300 |
| 4 | b = 500 passes and b = 300 doesn't | The user decides, with the numbers |
| 5 | Neither passes | 200 stays and the rule is unchanged |

Every problem's medians, feasible shares, Mann–Whitney p-values and the losing problems are reported beside the verdict.

## 5. CMA-ES flat-run restart

Track the best violation while infeasible with the same flatness test. `_recent_bests` only grows on feasible generations, so its window can span a long infeasible gap: clear the history when the run moves between infeasible and feasible. Add a direct test that a flat violation triggers a restart.

Pass criteria, fixed before measuring: over 30 seeds on the outputs suite at every budget, CMA-ES finds a feasible design on pocket-2d-1in100 in more runs, or as often and sooner, at 1,000 evaluations per dimension, and its median error on no other problem and budget worsens by more than ×1.1. Without output constraints every violation is 0, so nothing changes, which a test checks.

Measured, and not shipped. Over 30 seeds on the seventeen problems at all five budgets, the change altered one result: pocket-2d-1in100 at 1,000 evaluations per dimension. There, runs that found the pocket found it sooner, a median of 1,263 evaluations against 1,803, but fewer found it, 16 of 30 against 19, and the median error rose from 4.8 to 8.7. That fails the criterion. Seed by seed, both versions restarted twice per run on average: the restart for a step size collapsed already fired in the basin, and the flat violation only fires it earlier. Earlier restarts reshuffle which runs happen to start near the pocket, 9 seeds finding it only before the change and 6 only after, rather than finding it more often. What would help is a restart that avoids the basin it left, not one that leaves sooner; that is a change to the restart strategy and is left for later. The known weakness stands: CMA-ES's flat-run restart looks only at feasible bests.

## 6. Batches that bunch together

Hypothesis: the kriging believer keeps the mean at x and only shrinks the variance; where μ is far below the best, expected improvement is about best − μ, nearly independent of σ, so the next candidate lands adjacent. A large learned noise term worsens it. Confirm by logging u at the chosen points before picking a fix. Candidates: the constant liar (Ginsbourger et al., 2010), local penalization (González et al., 2016), a minimum distance in length scales. Measure the smallest pairwise distance in a batch and the error at sizes 4 and 8. Guard: batch size 1 with nothing pending must be exactly unchanged.

Pass criteria, fixed before measuring: over 30 seeds on the outputs suite and the quick suite, at batch sizes 4 and 8 and at 10 and 20 evaluations per dimension, the geometric mean of a fix's median errors relative to the believer's is at most ×0.9 at every size and budget (errors floored at 1e-8, over the problems where both medians are finite), and no problem's share of runs that find a feasible design falls by more than one run in ten. Of the candidates that pass, the simplest ships. Step 1b's cost ceiling is part of the criteria too: a fix may not raise seconds per candidate past ×1.2 of step 1b's table at 500 evaluations without sign-off.

The hypothesis is confirmed, except for the noise term. Logging batches of 8 at 10 evaluations per dimension on ball-5d-1in100, corner-5d, sphere-5d and rastrigin-5d, five seeds each: the first candidate of a batch has a median u of 0.1 to 3.7, and the later ones 0.4 to 386, with σ shrunk to about 0.005 and within 0.001 to 0.01 length scales of an earlier candidate. The learned noise stayed at its floor of about 1e-6, so a large noise term isn't the cause on these problems. Without output constraints a second mechanism adds to it: the local search starts near the five lowest values in the believed model, and a believed point was among them for 70% to 100% of later candidates.

Three candidates were built behind a temporary switch, each against the believer:

- the constant liar: each chosen point believed at the incumbent's value;
- the believer capped at the incumbent, max(μ, best);
- leaders: the believer, with the local search started only from evaluated points.

The two that lie also start the local search from evaluated points only. Pending points are lied about the same way, and with nothing feasible yet the believer is kept. The median smallest distance between two candidates of a batch of 8, in the unit cube, over the same four problems and five seeds:

| Strategy | ball-5d-1in100 | corner-5d | sphere-5d | rastrigin-5d |
| --- | ---: | ---: | ---: | ---: |
| believer | 0.0001 | 0.0001 | 0.0000 | 0.0000 |
| constant liar | 0.036 | 0.048 | 0.056 | 0.077 |
| capped | 0.033 | 0.049 | 0.019 | 0.111 |
| leaders | 0.0001 | 0.0001 | 0.0000 | 0.0000 |

Measured against the criteria over 30 seeds on the outputs suite, its noisy designs included, and the quick suite: 23 problems, four strategies at sizes 4 and 8, 10 and 20 evaluations per dimension, 11,040 runs. The geometric mean of each strategy's median errors relative to the believer's, over the problems where both are finite:

| Strategy | Size 4, 10 per dimension | Size 4, 20 | Size 8, 10 | Size 8, 20 |
| --- | ---: | ---: | ---: | ---: |
| constant liar | ×0.60 | ×0.81 | ×0.47 | ×0.59 |
| capped | ×0.58 | ×0.81 | ×0.47 | ×0.55 |
| leaders | ×1.01 | ×0.99 | ×0.98 | ×1.01 |

- **Leaders alone fail**: starting the search from evaluated points doesn't spread the batch, and changes nothing.
- **The two that lie pass on error everywhere**, and gain most where the believer bunched worst: at size 8 and 10 per dimension the constant liar's median error is ×0.065 of the believer's on sphere-5d and ball-5d-1in100, and ×0.13 on g07. Its worst cells are corner-2d at size 8, ×2.7, and pressure-vessel-outputs at size 8 and 20 per dimension, ×1.5; the criterion is on the geometric mean, which these don't move past ×0.9.
- **On feasibility, every run of every strategy found a feasible design.** The shares that fell are on the noisy designs, where `feasible` is the share of runs whose returned design truly satisfies every limit. At 30 seeds the constant liar fell past one run in ten in one cell, noisy-pressure-vessel at size 4 and 20 per dimension, 63% against 77%, and the capped believer in two. Since the cell's difference was within chance at 30 seeds, the noisy designs were rerun for the believer and the constant liar on 30 more seeds, as the user decided on 8 Nov 2026, and the criterion applied to the 60-seed shares. There no cell falls by more than 6 runs in 60, so the constant liar passes, but at the limit on the pressure vessel at size 8 and 20 per dimension, 67% against 77%, and the vessel at 20 per dimension falls at both sizes. It is the boundary effect of item 2: a design nearer the optimum sits nearer the limit, where one noisy draw passes about half the time. The constant liar's median error is lower than the believer's in every one of those cells, so the margin of safety under noise, an open decision on the roadmap, is where that belongs, not here.
- **Within the cost ceiling.** At 500 evaluations, seconds per candidate are ×0.98 to ×1.08 of step 1b's table at every count of limits and both batch sizes.

The constant liar ships, as the simpler of the two that lie: it needs no prediction at the chosen point. The switch is gone. With one candidate and nothing pending it never lies, so batches of 1 choose exactly as before, which a test checks by refusing the lie; another checks that a batch of 8 on a 5-dimensional sphere keeps its candidates more than 0.02 apart in the unit cube, which the believer failed on three of four seeds. The constraints' models still believe their candidates: a limit's margin has no incumbent to lie at, and the objective's model alone spread the batches. The multi-objective Bayesian optimizer keeps its own batch strategy, which wasn't measured here.

While checking that the final code repeats the measured runs, seeded Bayesian runs on rastrigin-5d turned out not to repeat exactly on macOS, on main as well: two runs in one process part at the 65th evaluation by about 1e-9, with BLAS held to one thread. The cause is Apple's Accelerate: its single right-hand-side triangular solve, and some matrix-vector products and dot products, give last-bit results that depend on where in memory an operand starts, from about 64 rows up, and small NumPy arrays start wherever the allocator puts them. Every repeatability test stopped below 64 evaluations. Fixed after item 6: the Gaussian process places those vectors on a 256-byte boundary, and a test checks its results are the same bits wherever its inputs sit in memory.

## 7. A margin of safety for limits under noise

Added on 8 Nov 2026 from item 2's measurement, and taken into 0.15.0 by the user's decision, ahead of noise with several objectives, which can then reuse it. Design, for sign-off before it is built.

The problem. With noise, a limit is judged on the average of a design's outputs. The best design usually sits on a limit, where its true margin is near zero, so whether its average passes is close to a coin flip however often it is measured, and the better the optimizer, the closer it sits. On the noisy engineering designs, only 43% to 70% of Bayesian optimization's returned designs truly satisfy every limit at 50 evaluations per dimension, and `Result.feasible` says True for most of the rest. A result that breaks a limit while reporting that it doesn't is the worst kind of wrong answer.

The rule. A design passes a limit only if the margin of its average outputs clears it by k standard errors:

    margin of the average outputs + k × s / √n ≤ 0

where n is the number of times the design was evaluated, s is the noise in that limit's margin, and k is the quantile at `feasibility_confidence` of the t distribution with the limit's pooled degrees of freedom. This is the margin of the averages, not the average of the margins, as item 2 decided; for a limit that is linear in the outputs the two are the same. The first version used the normal quantile, k = Φ⁻¹(confidence), so 0.95 gave k ≈ 1.64 whatever the degrees of freedom. With s estimated from only a few repeats, that is too small. At the minimum of 3 degrees of freedom, a design sitting exactly on the limit passes with about 10% probability, not 5%: 90% per limit, not 95%. The t quantile, 2.35 at 3 degrees of freedom, approaching 1.64 as they grow, gives the stated 95%. The user adopted it on 20 Nov 2026, along with renaming the setting. As a design is measured more often, its standard error shrinks and it may sit closer to the limit. The violation used by Deb's rule becomes the relative violation of the shifted margin, so ranking, the result, `Result.feasible`, progress and the threshold all use the same test.

Where s comes from. Few designs are evaluated twice during a search, so s can't be each design's own spread. It is pooled: for each limit, the within-design variance of its margin over every design evaluated at least twice, weighted by degrees of freedom. That assumes the noise in a limit's margin is about the same everywhere, which suits a simulator's or a rig's noise and is stated in the docs. Margins rather than outputs are pooled, so a limit on several outputs, such as `Output("a") <= Output("b")`, gets the noise of its own difference. Until there are at least 3 degrees of freedom for a limit, there is no margin on it and the current rule applies.

A limit that varies with the design, such as `Output("stress") <= Output("strength")`, or one whose bound is computed from the parameters, has no fixed size to scale its margin by. So its margin is pooled and widened in the outputs' own units, which assumes the noise in that difference is about the same at every design. Where the noise grows with the quantities, as it usually does, write the limit as a ratio, such as `Output("stress") / Output("strength") <= 1`. It then has a fixed size of 1, and its margin is relative like that of any limit that is a number. The docs say so.

The final round in two halves. Today the final round re-evaluates up to 5 finalists chosen before any repeats exist, then returns the best of them by average. Under the margin, the first half of the reserve re-evaluates the finalists as now, which gives the pooled estimate its degrees of freedom; the second half re-ranks every evaluated design by its averages with the margin, as every optimizer ranks its finalists, and spends the rest on the new leaders that are likeliest to pass, so a design with some slack can still win if the first finalists all sat on the limit. The result is the best finalist that passes with the margin, or, if none does, the least violating, with `feasible=False`.

What the confidence means. It is per limit, per design, and before selection. A design that truly breaks a limit passes that limit's test with probability at most 1 − `feasibility_confidence`, each time it is judged. It says nothing directly about the result, which is chosen as the best of many designs judged this way, and selection favours those that passed by luck. On a toy problem whose search converges onto one limit, Hone Applied Mathematician ran Bayesian optimization on 150 seeds at 0.95: of the 41 results that truly broke the limit, 8 still reported feasible, about 20% where a design fixed in advance would give at most 5%. Those 8 were 5% of all runs, so this agrees with item 7's measured criterion, at most 10% of results reporting feasible while truly breaking a limit. The docs state the guarantee as per limit and per design, and the 10% as what was measured, not promised.

The setting. `feasibility_confidence: float | None` on every config (named `confidence` when item 7 shipped, and renamed before 0.15.0 was released, since `sensitivity()` has a `confidence` of its own), used only with noise and output constraints. The default is 0.95. `None` turns the margin off and gives exactly the behaviour of item 2. It joins the run's settings, so a checkpoint saved without it is refused as set up differently, as `final_evaluations` was.

Pass criteria, fixed before measuring. Over 30 seeds on the three noisy engineering designs at 10, 20, 50 and 100 evaluations per dimension, for Bayesian optimization (up to its 200-evaluation cap), CMA-ES, the genetic optimizer and random search, against item 2's behaviour (`confidence=None`):

- **Truly feasible results.** The share of runs whose returned design truly satisfies every limit rises in every cell where it was under 90%, and falls by no more than one run in ten anywhere.
- **Honest reporting.** The share of runs where `Result.feasible` is True but the design truly breaks a limit is at most 10% in every cell; item 2 is measured on this too, for comparison.
- **What it costs.** The median error, counting a truly infeasible result as infinite as the benchmark does, worsens by no more than ×1.1 in any cell. Among runs truly feasible under both, the median error worsens by no more than ×1.5, which bounds how conservative the margin makes a design.
- **Nothing else moves.** Without noise, or without output constraints, seeded runs are exactly unchanged, which tests check.
- **Within the cost ceiling,** though the margin adds no model, so the cost table isn't expected to move.

If 0.95 fails the criteria and 0.9 or 0.99 passes, the passing value becomes the default, with the measurement stating why. The run is a spec on the Measure workflow.

Measured, and shipped with 0.95 as the default by the user's decision of 8 Nov 2026, past six of the criteria. On GitHub's runners by the Measure workflow, 5,400 runs, the full table in `benchmarks/margin-results.md`.

The first version failed 19 checks at 0.95, two kinds. Nine were cells where the margin changed nothing, mostly the spring at small budgets, where the search often finds nothing feasible to protect; the criterion that a share under 90% must rise failed them for staying the same, a flaw in how it was written, and the user clarified that a cell the margin leaves unchanged seed for seed needn't rise. The rest were Bayesian optimization made too conservative, up to 5.8% above the optimum against 0.7%: the second half ranked its candidates with the margins they had, so a design evaluated once, with a wide margin, could only lead if it sat far from the limit. Ranking them with the standard error they would have after their share of the half, and concentrating the half on two designs, brought the failures at 0.95 to six. The same measurement also found GitHub's runners differ in processor, and OpenBLAS's arithmetic with them, so the workflow now deals each seed's runs of every optimizer to one shard; the numbers below are from that run.

Bayesian optimization, truly feasible runs of 30, the median distance above the optimum among them, and the share that report feasible but truly break a limit:

| Problem, per dimension | Average alone | confidence 0.95 |
| --- | --- | --- |
| Pressure vessel, 10 | 26, 3.7%, 13% | 30, 4.8%, 0% |
| Pressure vessel, 20 | 19, 1.4%, 37% | 26, 2.3%, 10% |
| Pressure vessel, 50 | 14, 1.0%, 43% | 19, 0.9%, 7% |
| Spring, 20 | 26, 2.3%, 13% | 29, 4.4%, 0% |
| Spring, 50 | 14, 0.8%, 53% | 21, 1.1%, 3% |
| Welded beam, 20 | 25, 1.8%, 17% | 28, 2.4%, 3% |
| Welded beam, 50 | 23, 1.2%, 20% | 27, 1.4%, 3% |

- **Honest reporting passes everywhere.** At 0.95 no cell of any optimizer has more than 10% of results reporting feasible while truly breaking a limit, against up to 53% before.
- **Truly feasible results rise** wherever the margin applies and the share was under 90%, except the genetic optimizer on the spring at 100 per dimension, where runs changed and the share stayed at 87%. CMA-ES, the genetic optimizer and random search pay almost nothing: their median errors are within ×1.1 everywhere but CMA-ES on the beam at 100 per dimension at 0.99.
- **Bayesian optimization pays 0.1 to 2 percentage points of objective,** past the bounds in five checks at small budgets: median error ×1.24 on the vessel at 10 per dimension, ×1.65 on the spring and ×1.21 on the beam at 20, and among runs feasible both ways ×1.67 on the vessel and ×1.76 on the spring at 20. The bounds were set before seeing what a statistical margin costs; it costs about what a safety factor does, and the user judged it worth it.
- **0.9 and 0.99 are no better.** 0.9 fails five checks and lets the vessel at 50 per dimension fall; 0.99 fails fourteen, costing Bayesian optimization up to ×2.3. So 0.95 stays.
- **Bayesian optimization still misses on the vessel at 50 per dimension,** 19 of 30: the margin rests on the noise it can estimate from a final round of 20 evaluations, and its designs are the closest to the limits. More evaluations in the final round would help, at the search's expense.

What it doesn't do. Feasibility as a probability from the models, which would use the constraints' Gaussian processes rather than repeats, stays for later; so does a noise level given by the user, which a pooled estimate makes unnecessary when there are repeats.

## 8. Noise with several objectives

Decision 2's second half: noise with output constraints for `MultiObjectiveBayesian`, `NSGA2` and `MultiObjectiveRandomSearch`, reusing items 2 and 7. Hone Applied Mathematician reviewed a first draft on 20 Nov 2026, and the review's changes are folded in. The user signed the design off on 20 Nov 2026, adding the second reserve size below.

### What happens today

- **The refusal.** `OptimizerCore.__init__` raises `UnsupportedSearchSpace` for output constraints with `noise=True` unless the class sets `supports_noisy_outputs`, which only the single-objective `Optimizer` does. `MultiObjectiveConfig` also rejects any `final_evaluations` above 0 ("doesn't apply to several objectives yet"). It inherits `feasibility_confidence` from `OptimizerConfig`, but nothing reads it.
- **Noise without output constraints already works.** `MultiObjectiveOptimizer._record` keeps an `_Observed` per point with running sums of its objective vector. With noise, a point's values are the average of its evaluations, the front is recomputed from the averages (`_front_stale`), and each `ParetoPoint` carries its standard errors. `NSGA2._next_population` sets each pool member's values to its point's current average before choosing survivors. `MultiObjectiveBayesian` adds one row to `_Y` per evaluation, holding the point's running average at that time, as single-objective Bayesian optimization does with `individual.fitness`.
- **What output constraints do, without noise.** `_record` judges the violation on the single draw. A draw that breaks a limit never reaches `_observed`. It counts in `_infeasible`, its values go to `_infeasible_values` for the reference point, and the individual keeps its values and violation for ranking. `ParetoPoint.outputs` holds the first draw's outputs. NSGA-II ranks by constrained domination on each individual's `violation`. Multi-objective Bayesian optimization models each margin with `_constraint_models` and takes `_violations` row by row from that draw. It measures improvement only from feasible rows, multiplied by the probability of feasibility.
- **Why lifting the refusal alone would be wrong.** A point's average would count only its passing draws, which biases it toward passing. Its outputs would be one draw's, and its place on the front would rest on that draw. Item 7 measured the cost with one objective sitting on a limit: up to 53% of results were reported feasible while truly breaking a limit. A front is worse, because most of its points usually sit on some limit.

### The design

**Averaging per point, as in item 2, for all three optimizers.** Every successful draw of a point enters its `_Observed`: objectives as now, plus a list of its outputs. Whether the point passes is judged on its average outputs, with item 7's margin. The front is the non-dominated set of the points that pass, with dominance judged on the average objectives as today. `ParetoPoint.outputs` becomes the average outputs. `ParetoResult.infeasible` still counts the draws that broke a limit, as `Result.infeasible` does. `ParetoResult.feasible` says whether any point passes on its averages. Without noise, each point is evaluated once, and all of this reduces to what happens today.

**One implementation of the rule.** `_average_outputs`, `_violation`, `_limit_noise` and `_margin_on` move from `base.Optimizer` into a mixin in `core.py`, used by both families, along with their storage (`_output_observations`, `_points`, `_noise_cache`). Three things are then literally the same code for both: each limit's pooled noise, the `MARGIN_FREEDOM` threshold, and the margin of k standard errors per limit, with k the t quantile at the limit's pooled degrees of freedom, as section 7 now states. The move must leave every single-objective seeded run exactly unchanged.

**NSGA-II.** Before choosing survivors, each pool member's values, outputs and violation are refreshed from its point's averages, as the genetic optimizer's `_refresh_averages` does. Constrained domination is then judged on the averages.

**Multi-objective Bayesian optimization**, mirroring item 2's changes to `BayesianOptimizer`:
- Keep each row's point key. Before proposing, re-judge every row's violation from its point's current averages, with the margin.
- Improvement counts only from rows that pass.
- With noise, the incumbent comes from the model, not from the draws, since the lowest draw is likely a lucky one. With `parego`, it is the lowest predicted combination among the passing rows. With `models`, it is the combination of the predicted means at those rows.
- With nothing passing, the search for a feasible design is unchanged, as item 1 decided.
- Batches keep the kriging believer. Item 6's constant liar was measured on one objective only, and moving it here would be a separate change.

**Multi-objective random search** needs nothing beyond the averaging.

**A final round for a front.** With noise and output constraints, `final_evaluations=None` keeps back a reserve; otherwise it keeps back none. Two sizes of reserve are measured, and the rule for choosing between them was fixed on 20 Nov 2026, before measuring (see the pass criteria):
- *a tenth*, the same as for one objective: a tenth of `max_evaluations`, at least 10 and at most half;
- *a fifth*, sized for a front: a fifth of `max_evaluations`, at least 20 and at most half.

The refusal in `MultiObjectiveConfig` goes. `feasibility_confidence` defaults to 0.95, and `None` turns the margin off. With the margin on and enough reserve, the round comes in two halves, as in item 7:

- *First half: the front's points, to measure the noise.* The finalists are chosen greedily by hypervolume contribution, so they spread along the front instead of crowding one end. Their number is capped so that each gets at least two evaluations. Beyond the cap, every limit is given at least `MARGIN_FREEDOM` (3) degrees of freedom among the finalists where it applies: if the chosen finalists don't give a limit that, the next by contribution among the points where it applies is added. Otherwise a conditional limit would silently have no margin.
- *Second half: the points whose pass is least certain.* Every evaluated point is re-judged on its averages with the margin, using the standard error it would have after its share of the half (`more`, as item 7 does). The front is recomputed. The half then goes to the front points whose widened margin, in standard errors, is smallest on any limit, and only `SECOND_HALF_FRONT_POINTS = 4` of them: count = max(1, min(4, half // 3)). Four, because three repeats halve a point's one-draw standard error (1/√4), and four points keep more of a two- or three-objective front than its ends; item 7 found that spreading the half thinly leaves every standard error wide.
- *The result* is the front of the points that pass with the margin on their averages. A point the round never reached is still judged at n = 1, where the margin is widest, so it stays on the front only if it clears every limit by a lot.

With the margin off, there is one round, item 2's behaviour: the first half's rule takes the whole reserve.

**Checkpoints.** `final_evaluations` and `feasibility_confidence` are already fields of every multi-objective config, so they are already part of a checkpoint's setup. No multi-objective run could combine noise with output constraints until now, so no saved checkpoint changes meaning. A checkpoint saved before `feasibility_confidence` existed is already refused as set up differently, as for one objective; a test pins that.

**Names.** New code and fields use the final names (`max_evaluations`, `batch_size`, `best_value`). Existing names, such as `population_size` on `MultiObjectiveBayesianConfig`, stay until 0.22.0.

### What the rule guarantees, and what it doesn't

These go in the docs as well as here.

- **A rate per point, not per front.** Each limit keeps its own margin, uncorrected for the number of limits. A design that truly breaks any one limit passes with probability at most 1 − `feasibility_confidence`, since it has to pass that limit's test. This is the intersection-union argument. So the bound is on the rate per reported point, judged before selection, as section 7 says for one objective. It is not a bound on the chance that every reported point is feasible. The cost is power: a design with k standard errors of slack on each of m limits passes them all with probability about 0.5^m.
- **Selection is stronger on a front.** A point that passes a limit by luck dominates its truly feasible neighbours and pushes them off the front. On OSY, whose whole front sits on its limits, criterion 1 at 10% may fail for this reason. That is expected, and the margin isn't to be tuned around it.
- **Concentrated repeats bias what survives.** Repeated points regress toward their means, while an unrepeated point keeps its one lucky objective draw and dominates too often. Unrepeated points survive only well inside the limits, so the front leans toward the interior. And exposing one false pass brings back its dominated neighbours at n = 1, where most of them fail, so a region of the front can vanish. Spending the second half in two or three rounds, re-ranking between them, might reduce this. It isn't part of this design, and is left as a possible later experiment.
- **Dominance is judged on averages and ignores objective noise.** A point can sit on the front because of a lucky objective draw, as it can today with noise and no output constraints. The margin protects the limits, not the objectives. This stays as it is and is stated in the docs.
- **The reserve was sized for one result.** In most Bayesian cells it is the floor of 10, so each half is 5: two finalists, then one front point. At those budgets, expect reported fronts of two or three measured points plus interior points, and criterion 3 failing or unresolved there. The lever is the reserve's size for a front, so the user decided at sign-off, on 20 Nov 2026, to measure a fifth (at least 20) beside the tenth, and let the pass criteria choose.
- **The pooled noise is stretched further.** Pooling each limit's noise over every repeated point assumes the noise is about the same everywhere, and a front spans much more of the space than one optimum. The benchmark's noise has a constant standard deviation, so it can't show this failing. An optional cell with noise proportional to each output would test it.

### The benchmark

Two problems with known fronts, made noisy with their limits as outputs, plus the cantilever:

- **OSY** (Osyczka and Kundu, 1995): 6 variables, 2 objectives and 6 limits, all reported as outputs. Its whole front lies on the limits' boundaries, in five pieces with an analytic form, so its true hypervolume is computed from that form. It is the hard case for honest reporting, since every front point sits on a limit.
- **TNK** (Tanaka et al., 1995): 2 variables, 2 objectives and 2 limits, with a disconnected front on the wavy limit. With two variables it gives Bayesian optimization four budgets under its 200-evaluation cap, where OSY's six leave it two, and it costs little to run.
- **The cantilever**: three objectives and a choice of material. The benchmark's variant, `cantilever_outputs()` in `hone.benchmarks`, reports the stress as an output. `examples/cantilever_beam.py` is unchanged and keeps its function constraint. With two continuous variables per material, the true front comes from a dense noise-free grid of each material, built once and saved.

Noise is added to every objective and every output, scaled as `ENGINEERING_NOISE` scales it for the noisy engineering designs, at about 1% of each quantity's typical value. **The noise has its own seed**, separate from the optimizer's. Today `runner.py` seeds the noise from the run's seed, and the optimizer gets the same seed. In the multi-objective runner the run's seed stays the noise seed and the key that pairs runs, while a spec's optimizer factory may give the optimizer a different seed; each run records both. Single-objective measurements are unchanged.

The runs use 30 seeds at 10, 20, 50 and 100 evaluations per dimension, with Bayesian optimization run up to its cap. The optimizers are NSGA-II, multi-objective Bayesian optimization with both `parego` and `models`, and random search. Each runs four ways, on the same optimizer and noise seeds:
- the new default with a tenth kept back: averaging, the final round, and the margin at 0.95;
- the same with a fifth kept back, everything else as the default;
- averaging alone: `final_evaluations=0` and the margin off, for the record;
- every draw believed: `noise=False`. This is the baseline, the only way to run these problems today.

Each reported front point is re-evaluated without noise. For each run, the measurement records:
- the front's size;
- how many of its points truly satisfy every limit;
- whether the run holds any truly feasible point at all;
- the *true gap*: the true front's scaled hypervolume less that of the truly feasible reported points, at their noise-free values.

The scaling and the reference point are fixed per problem from the true front, never from a run, and a point beyond the reference counts for nothing. The true gap doesn't penalise a falsely reported point beyond leaving it out, so it is read together with criterion 1.

**The floor run** comes before any comparison. The baseline runs against itself. The noise seed stays at s for both runs of a pair. The second run uses optimizer seed s + 1000 and takes the first run's first generation as its initial points, so both evaluate the same first generation in the same order and see the same draws on it. After that they differ only in the search, and the noise on later evaluations is independent. Per cell, the floor run applies criterion 3's statistics to these pairs. That shows whether 30 seeds can resolve δ in that cell: 30 pairs resolve about half the standard deviation of the paired differences. A cell that can't resolve δ is reported as such before the comparison runs.

There is a limit for random search. With the same optimizer seed, the default and the baseline propose identical designs and differ only in the final round and the judging. A floor from a different search stream is therefore much wider than that comparison's real spread under no change, and criterion 3 is lenient for random search.

**The floor, measured** on 20 Nov 2026 by the Measure workflow: 2,520 runs, the table in `benchmarks/multi-floor-results.md`. 21 of 42 cells resolve δ.
- *The cantilever resolves everywhere:* all 14 cells, with U − L at most 0.013 against a δ of 0.026.
- *TNK resolves for Bayesian optimization at 50 and 100 evaluations per dimension* (both methods), and nowhere for NSGA-II.
- *OSY resolves almost nowhere:* U − L runs from 0.045 to 0.12 against a δ of 0.019. Its baseline fronts are small, with medians of 1 to 14 points, and its true gaps large: 0.12 to 0.27 for Bayesian optimization, 0.5 to 0.97 for NSGA-II and random search.
- *Four of the cells that resolve say nothing:* random search where the whole budget fits in its first generation of 50, or nearly does (OSY at 10, TNK at 10 and 20, the cantilever at 10). There the pair's runs are identical and every difference is 0.

So in about half the cells, criterion 3 can't resolve δ at 30 seeds. Under the rule for choosing the reserve, those cells count as not met, and the choice goes to the user. The baseline's own figures also show what the comparison starts from. Believing every draw, Bayesian optimization reports 14% to 39% of its front points on OSY truly infeasible on average per run, and up to 15% on TNK at 100 per dimension. NSGA-II and random search stay under 3%, mostly because their fronts are small and well inside the limits.

### Pass criteria, fixed before measuring

Per problem, optimizer and budget (a *cell*), each reserve is compared with the baseline, paired by seed, and judged on every criterion below on its own:

1. **Honest reporting.** Read two ways:
   - pooled over a cell's 30 runs, at most 10% of reported front points truly break a limit;
   - per run, the share of its reported points that truly break a limit; the mean of these shares over the cell's runs with a non-empty front is at most 10%.

   Runs with an empty front are counted separately, not as passes. The number of runs with any truly infeasible reported point is reported, as a figure, not a criterion. Each cell's median front size and median number of truly feasible reported points are reported beside the baseline's, so a front that passes by shrinking shows.
2. **Truly feasible runs.** The share of runs whose front holds at least one truly feasible point rises in every cell where it was under 90%, and falls by no more than one run in ten anywhere. A cell the change leaves seed-for-seed unchanged need not rise, as the user clarified for item 7.
3. **Hypervolume, the guard on shrinkage.** δ, the tolerated loss, is 2% of the true front's scaled hypervolume, fixed by the user on 20 Nov 2026. Per cell, sort the 30 paired differences in true gap (default less baseline). The 20th smallest is a 95% upper bound on their median, and the 11th smallest a 95% lower bound (binomial(30, ½): P(X ≤ 10) = 0.049). The cell **passes** if the 20th is at most δ and **fails** if the 11th is above δ. Otherwise it is **unresolved**, and goes to the user with its numbers. Item 7 found that a statistical margin costs about what a safety factor does, so a failure comes to the user too, rather than being tuned away.
4. **Nothing else moves.** Seeded multi-objective runs without noise, or with noise and no output constraints, are exactly unchanged. So is every seeded single-objective run, after the shared code moves. Tests check both.
5. **Cost.** Step 1b timed only single-objective Bayesian optimization, so `hone.benchmarks.cost` gains multi-objective cells. They time seconds per `MultiObjectiveBayesian` candidate at 50 to 500 evaluations, with 0 and 8 limits on outputs, at batch sizes 1 and 8, for both methods. Main and the branch are timed in the same session on the same machine. Neither method may exceed ×1.2 of main at 500 evaluations without the user's sign-off. Re-judging rows adds one pass over the points per proposal, so no change is expected.

**Which reserve ships**, decided on 20 Nov 2026, before measuring. A reserve passes if it meets criteria 1 to 3 in every cell, a cell left unresolved by criterion 3 counting as not met, and criteria 4 and 5:
- if both pass, the smaller reserve, a tenth, ships;
- if only one passes, that one ships;
- if neither passes, the numbers go to the user.

If a reserve fails only through unresolved cells, the choice goes to the user with the numbers rather than to the other reserve by default: an unresolved cell means the user decides.

**Cells the floor can't resolve**, decided by the user on 20 Nov 2026, after the floor was measured and before the comparison runs, on Hone Applied Mathematician's rule.

Criterion 3's verdicts hold in every cell. A fail (the 11th smallest paired difference in true gap above δ) is a valid statement that the median loss exceeds δ however wide the cell's floor is, and so is a pass (the 20th smallest at most δ). A cell the floor marks unresolvable is simply one where neither is likely, and when neither occurs it is decided on criteria 1 and 2. In those cells the PR also reports, for the baseline and each reserve, the median number of truly feasible reported points and the number of seeds in which the default keeps at least half of the baseline's; a cell where fewer than 11 of 30 seeds do is flagged for the user, not failed.

The floor's verdicts name in advance the cells where the reserve is chosen on counts. They do not decide whether criterion 3 is judged: its pass and fail stand wherever they occur. Those cells are the ones listed as unresolvable in `benchmarks/multi-floor-results.md`, fixed by that measurement and not re-judged from the comparison. The four random-search cells whose paired floor runs were identical (OSY at 10 evaluations per dimension, TNK at 10 and 20, the cantilever at 10) carry no information about width, so they are treated as the comparison finds them, with no special exclusion.

*Clearly more.* Per cell and seed, c is the number of reported front points that truly satisfy every limit; a run with an empty front has c = 0. For each of the 30 seeds, d = c with the larger reserve less c with the smaller, the two runs sharing the optimizer seed and the noise seed. The larger reserve gives *clearly more* in a cell when the 11th smallest d is at least 1, that is, when at least 20 of the 30 seeds have strictly more truly feasible points with it. It gives *clearly fewer* when the 20th smallest d is at most −1. A seed where the two are equal counts against both. The larger reserve ships if it gives clearly more in at least half of the cells the floor marks unresolvable and clearly fewer in none of them, and if it passes criteria 1 and 2 wherever the smaller one does; otherwise the smaller ships.

The test is exact at 30 pairs: with no difference, P ≤ 0.049 of a cell coming out clearly more. Ties count against the claim. And "at least half" of the 17 or so cells is required, because one of them would come out clearly more by chance.

The rule above for which reserve ships still applies, with one reading: in a cell the floor marks unresolvable, an unresolved criterion 3 doesn't count as not met. Such a cell is decided on criteria 1 and 2, and between the reserves on the counts, as just stated. A fail there still counts as a fail.

*The order, fixed with Hone Tech Lead on 20 Nov 2026.* Each reserve gets one outcome:
- **pass:** it meets every criterion, as read above;
- **fail:** it fails at least one criterion outright in some cell, including a criterion 3 fail anywhere;
- **undecided:** it fails nothing outright, but has unresolved criterion 3 cells outside the floor's list.

Then, in this order:

| | Outcomes | Ships |
| --- | --- | --- |
| 1 | Either reserve undecided | The user decides, with the numbers |
| 2 | Both pass | The fifth, if it gives clearly more in at least half of the floor-unresolvable cells, clearly fewer in none, and passes criteria 1 and 2 wherever the tenth does; otherwise the tenth |
| 3 | One passes, the other fails | The one that passes |
| 4 | Both fail | The user decides, with the numbers |

The first row comes first, so the fifth can't be chosen over a tenth that was merely undecided.

The runs are specs on the Measure workflow, which first needs to run multi-objective specs; see below.

### What it takes

- **Benchmark infrastructure:**
  - `MultiProblem` gains `noise`, `output_noise` and outputs.
  - The specs gain a noise seed apart from the optimizer's.
  - `MultiRun` gains the true gap, the front size, the number of truly feasible front points, and whether any point is truly feasible.
  - New problems `osy()`, `tnk()` and `cantilever_outputs()`, with a test of each true front.
  - `hone.benchmarks.sharded` and the Measure workflow accept specs whose problems are `MultiProblem`s.
- **The shared mixin,** moved out of `base.Optimizer`, with every single-objective result unchanged.
- **The averaging:** in `MultiObjectiveOptimizer._record`, the refresh in NSGA-II, and the re-judged rows and incumbent in multi-objective Bayesian optimization.
- **The final round** for a front, the validator change, and `supports_noisy_outputs = True` on the three optimizers.
- **The specs:** the floor run, the comparison and the cost cells.
- **The docs:** the README's noise section, including what the rule guarantees and what it doesn't, the docstrings and the changelog.

Delivery: three pull requests in order, each reviewable alone, by the user's decision of 20 Nov 2026.
- **(a) The mixin move,** a pure refactor: every seeded result unchanged, for one objective and several. It starts only after the t-quantile pull request (`margin-t-quantile`) merges, since both change `base.py` around the margin code.
- **(b) The benchmark infrastructure and the floor run,** with no optimizer change. The floor runs on the Measure workflow while (c) is built.
- **(c) The feature:** the averaging, the final round, the validator, `supports_noisy_outputs`, the comparison spec, the cost cells and the docs.

### Result: the tenth ships

Measured, and shipped with a tenth kept back by the user's decision of 21 Nov 2026. The Measure workflow made 5,040 runs on GitHub's runners, and the full table is in `benchmarks/multi-compare-results.md`. Both reserves failed some cells, which is branch 4, so the user decided. The user accepted the tenth's 16 failing cells as a cost of honest fronts, as for item 7.

- **Honest reporting passes everywhere,** for both reserves. On OSY, the share of Bayesian optimization's front points that truly break a limit falls from 14–38% to at most 1.4%. On TNK at 100 per dimension it falls from 14% to 2–3%.
- **Criterion 3 fails in 10 cells, which are accepted.** Eight are Bayesian optimization at 10–20 per dimension on OSY and TNK, where a reserve of 10 shrinks the median front from about 6 points to 2–3. The other two are NSGA-II at 100 per dimension on both problems.
- **Criterion 2 fails in 6 cells, which are accepted.** These are NSGA-II and random search at 10–20 per dimension, where more fronts come back empty.
- **The cantilever passes every cell.**
- **The fifth is rejected.** It fails 21 cells, two of them on the cantilever. In the cells the floor can't resolve, it gives clearly fewer truly feasible points than the tenth in 4 and clearly more in none.
- **Cost is within ×1.2 in every cell,** ×0.91 to ×1.08 against main's noise-free cells at 500 evaluations.

## Delivery

Separate pull requests, in the roadmap's order: (1) step 0, the measuring tools with a new baseline, done; (2) item 1, measured, the current method staying; (3) step 1b, the cost of a Bayesian candidate measured, which sets the ceiling for every later item; (4) item 2, with item 5's measurement, which didn't pass; (5) item 6, taken ahead of noise with several objectives on 8 Nov 2026 since its measurement had started; (6) item 7, the margin of safety, added on 8 Nov 2026; (6b) the margin's t quantile and the rename to `feasibility_confidence`, decided on 20 Nov 2026 from item 7's review; (7) noise with several objectives, which reuses it, as three pull requests: the refactor, the benchmark and its floor, then the feature; (8) item 4 last, because the earlier changes alter what it measures. Item 3 only if its test shows a need.
