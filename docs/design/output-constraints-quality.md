# Output constraints, quality improvements: design

Status: in progress, for 0.15.0. Step 0, the measuring tools, and step 1b, the cost of a Bayesian candidate, are done, item 2, noise with output constraints, is built, and item 6, batches that bunch together, is fixed with the constant liar, and item 7, a margin of safety for limits under noise, is built; item 1 is set aside, and item 5 was measured and not shipped. The rest are hypotheses until measured.

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

## Step 1b: the cost of a Bayesian candidate

Added by the roadmap's review of 2 Nov 2026. A timing harness, `python -m hone.benchmarks --cost`, gives seconds per candidate against evaluations (50 to 500) and limits on outputs (0 to 8), at batch sizes 1 and 8, with one profiled run, saved as `benchmarks/cost-summaries.json`. It sets a ceiling: every later item reports its seconds per candidate against the table, and may not raise it past ×1.2 at 500 evaluations without sign-off. `--baseline benchmarks/cost-summaries.json` checks it. 0.16.0's first step starts from this table rather than measuring its own.

Each cell is the mean over 3 seeds. One seed alone varies by up to about 15% between seeds at 500 evaluations, which is too close to ×1.2 for a change that moves the run the way a new seed does; the same seed repeats within about 5%.

Measured on an Apple M2 under macOS 27.2 at 0.14.0 with item 2 (details in `benchmarks/README.md`): at 500 evaluations a candidate takes 0.08 seconds with no limits and 0.78 with eight, about the same in batches of 8, and the cold fit about 10 seconds with eight limits. Searching the models costs more than fitting them: about four fifths of a profiled run of g07 at 200 evaluations, and most of the steady-state time at 500 with eight limits, where refitting takes about a third. An operating system update on the same machine moved the times by 3% to 25%, so the table is remeasured whenever the system changes. That bears on 0.16.0's first step, whose first planned change is refitting less often.

## 1. ~~Before anything is feasible~~ Set aside

Set aside after step 0's baseline: the current rule found a feasible design in every run on every problem with a tiny feasible region, and the alternatives tried didn't beat it. The deceptive pocket, the one problem where an alternative could show a gain, is item 5's test instead. The hypothesis and the alternatives stay below for whoever picks it up.

With every margin positive, the probability that a design is feasible is Φ(−μ/σ) for each limit, and log Φ(−μ/σ) ≈ −μ²/(2σ²) in the tail, so a large σ wins and the search goes to the corners, where the models are least sure. Two alternatives are compared over 30 paired seeds:

- A: one new Gaussian process on the bilog of the total violation, reusing `_log_expected_improvement`. Simple; the total violation has kinks where the active limit changes.
- B: quasi-Monte Carlo expected improvement in the total violation from the existing per-limit models. Bounded integrand; needs fixed samples or the noise swamps the finite differences in `_refine`; exactly zero where no sample improves.

The simplest method that finds a feasible design more often or sooner, without worsening final error by more than ×1.1, is kept. The current method may survive: it already handles the spring at 10 evaluations per dimension.

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

Pass criteria, signed off on 8 Nov 2026 before measuring. The outputs suite, its noisy designs included, at 10, 20, 50 and 100 evaluations per dimension, on the Measure workflow (`benchmarks/specs/choose_outputs.py`): Bayesian optimization with the benchmark's cap lifted to 1,000 evaluations, CMA-ES, the genetic optimizer and random search, over 30 seeds, except Bayesian optimization past 200 evaluations, which runs 10 seeds by the user's decision, to keep it to about a third of the runner time. Three rules for spaces with limits on outputs, picking Bayesian optimization up to 200 evaluations (R0, today's), 500 (R1) or 1,000 (R2), and otherwise what `choose_optimizer` picks past 200. Each is scored as the existing test scores today's rule: over the problem and budget cells where some optimizer finds a feasible design, the geometric mean of the pick's median error over the best optimizer's, errors floored at 1e-8, with cells where the pick finds nothing feasible counted on their own.

- The larger of R1 and R2 whose geometric mean is at most ×0.9 of R0's ships; if neither, the rule stays.
- It may lose no cell: no more cells where its pick finds nothing feasible than R0's, and wherever it picks Bayesian optimization where R0 doesn't, Bayesian optimization finds a feasible design at least as often as R0's pick, within one run in ten.
- Held out: today's rule, scored the same way from the saved summaries of the machine learning and constrained suites, which weren't used to choose it, keeps its picks within ×1.5 of the best by geometric mean, as the existing test requires on the 36 problems. A failure there is reported and doesn't block the change for limits on outputs.
- The rule is also stated in seconds: the docstring and README give Bayesian optimization's seconds per candidate at the chosen limit, from step 1b's table, so a user can judge it against an evaluation's own time.

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

The rule. A design passes a limit only if its average margin clears it by k standard errors:

    average margin + k × s / √n ≤ 0

where n is the number of times the design was evaluated, s is the noise in that limit's margin, and k = Φ⁻¹(confidence), so `confidence=0.95` gives k ≈ 1.64 per limit. As a design is measured more often, its standard error shrinks and it may sit closer to the limit. The violation used by Deb's rule becomes the relative violation of the shifted margin, so ranking, the result, `Result.feasible`, progress and the threshold all use the same test.

Where s comes from. Few designs are evaluated twice during a search, so s can't be each design's own spread. It is pooled: for each limit, the within-design variance of its margin over every design evaluated at least twice, weighted by degrees of freedom. That assumes the noise in a limit's margin is about the same everywhere, which suits a simulator's or a rig's noise and is stated in the docs. Margins rather than outputs are pooled, so a limit on several outputs, such as `Output("a") <= Output("b")`, gets the noise of its own difference. Until there are at least 3 degrees of freedom for a limit, there is no margin on it and the current rule applies.

The final round in two halves. Today the final round re-evaluates up to 5 finalists chosen before any repeats exist, then returns the best of them by average. Under the margin, the first half of the reserve re-evaluates the finalists as now, which gives the pooled estimate its degrees of freedom; the second half re-ranks every evaluated design by its averages with the margin, as every optimizer ranks its finalists, and spends the rest on the new leaders that are likeliest to pass, so a design with some slack can still win if the first finalists all sat on the limit. The result is the best finalist that passes with the margin, or, if none does, the least violating, with `feasible=False`.

The setting. `confidence: float | None` on every config, used only with noise and output constraints. The default is 0.95. `None` turns the margin off and gives exactly the behaviour of item 2. It joins the run's settings, so a checkpoint saved without it is refused as set up differently, as `final_evaluations` was.

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

## Delivery

Separate pull requests, in the roadmap's order: (1) step 0, the measuring tools with a new baseline, done; (2) item 1, set aside; (3) step 1b, the cost of a Bayesian candidate measured, which sets the ceiling for every later item; (4) item 2, with item 5's measurement, which didn't pass; (5) item 6, taken ahead of noise with several objectives on 8 Nov 2026 since its measurement had started; (6) item 7, the margin of safety, added on 8 Nov 2026; (7) noise with several objectives, which reuses it; (8) item 4 last, because the earlier changes alter what it measures. Item 3 only if its test shows a need.
