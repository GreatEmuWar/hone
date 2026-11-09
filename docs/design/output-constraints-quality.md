# Output constraints, quality improvements: design

Status: in progress, for 0.15.0. Step 0, the measuring tools, and step 1b, the cost of a Bayesian candidate, are done, item 2, noise with output constraints, is built, and item 6, batches that bunch together, is fixed with the constant liar; item 1 is set aside, and item 5 was measured and not shipped. Item 3's gate was measured and passed, for Bayesian optimization, and awaits a decision. The rest are hypotheses until measured.

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

**Pass criteria, fixed before measuring** (8 Nov 2026). Each problem of the outputs suite is run as written and rewritten, with every output limit that has a size, a number on one side, turned into the same limit with nothing on that side: `Output("x") <= 250` becomes `Output("x") - 250 <= 0`, and `Output("x") >= 100` becomes `Output("x") - 100 >= 0`. The two have the same feasible set, which the spec checks on 300 random points of each problem, but the rewritten one's limits all have size 0, so its relative violations and the Bayesian model's margins are in raw units. Limits that already have size 0 stay as they are, such as the pressure vessel's `shell / 16 >= Output("shell_needed")`; the pockets, whose only limit is `Output("quantity") <= 0`, have nothing to rewrite and are left out, and so are the spring and the corners, whose limits all have size 1, so that dividing by it changes nothing and they would run exactly as written. That leaves 9 problems: the pressure vessel, the welded beam, the four balls, g06, g07 and g10. Bayesian optimization, CMA-ES and the genetic optimizer, with the runner's settings, at 10, 20 and 50 evaluations per dimension, Bayesian optimization only up to its 200 evaluations; 30 seeds.

For each optimizer and budget: the geometric mean over problems of the ratio of median errors, rewritten to as written, each error floored at 1e-8, over the problems where both medians are finite; and the share of runs that end feasible, averaged over the problems. `scale=` is needed if, for any optimizer and budget, that ratio is ×2 or more, or the average feasible share of the rewritten problems is lower by more than 1 run in 10. Otherwise item 3 is measured and not needed. Single problems with a ratio of ×2 or more, or a feasible share lower by more than 1 run in 10, are listed but don't decide it: at 30 seeds one problem's share moves that much by chance.

Measured with `benchmarks/specs/scale.py` on the Measure workflow, 4,620 runs in 40 shards ([run 42547180219](https://github.com/GreatEmuWar/hone/actions/runs/42547180219)). The geometric mean of median errors, rewritten to as written, and the average feasible share, as written against rewritten:

| Optimizer | 10 per dimension | 20 | 50 |
| --- | ---: | ---: | ---: |
| Bayesian optimization | ×3.12, 98.5% / 95.2% | ×3.20, 100% / 100% | ×2.79, 100% / 100% (5 problems) |
| CMA-ES | ×0.99, 53.3% / 53.0% | ×1.02, 65.6% / 67.8% | ×0.90, 91.1% / 87.8% |
| genetic optimizer | ×1.02, 51.1% / 51.5% | ×1.01, 64.1% / 65.2% | ×0.94, 82.2% / 78.5% |

- **Bayesian optimization passes the gate at every budget**, so by decision 3 `scale=` is justified. The loss is on the problems whose limits are far from 1 in size: the welded beam (13,600, 30,000, 0.25, 6,000) ×22, ×62 and ×5.2 at 10, 20 and 50 per dimension, the pressure vessel's volume (1,296,000) ×8.4, ×14 and ×24, g06 ×19 and ×7.3, g07 ×10 and ×2.1, g10 ×2.3 and ×2.0. On the balls, whose radius is 0.6 to 2.9, it is ×0.3 to ×1.4, within what seeds do. The likely cause, not measured separately: through the bilog transform a margin of thousands becomes a few units of log, nearly flat across most of the box and steep only near the boundary, which a Gaussian process fits badly; with the relative margin the boundary sits where the transform is still nearly linear. Feasibility hardly suffers: every rewritten run finds a feasible design except at 10 per dimension on g06 (87%) and ball-5d-1in1000 (70% against 87%).
- **CMA-ES and the genetic optimizer don't need it.** With one limit, Deb's rule ranks infeasible designs in the same order either way: the genetic optimizer ran the balls identically, seed for seed, and CMA-ES all but 15 of its 360 runs on them, with the same medians. Where the sizes differ, summing raw violations lets the largest limit dominate; the clearest case is g10, whose `g6 <= -1250000` swamps the others, at 50 per dimension: CMA-ES finds a feasible design in 43% of runs against 67%, and the genetic optimizer in none against 47%. On g07 it goes the other way, 27% to 30% against 3% to 10% at 20 per dimension. The averages stay within 1 run in 10.

Verdict: **the gate is passed, for Bayesian optimization's model of each limit.** Nothing is implemented here; whether `scale=` ships, and in what form, is for the lead to decide. The measurement says what it would need to do: the rewritten limit with a scale of the written limit's size is the written limit, so an explicit scale on a limit that varies or is written with nothing on one side recovers exactly the ×3. It matters most for Bayesian optimization, and for Deb's rule only where limits of very different sizes are summed.

## 4. `choose_optimizer` with output constraints

On the pressure vessel with its limits on outputs, Bayesian optimization at 200 evaluations (31.6) beats CMA-ES at 4,000 (277); the welded beam goes the other way (0.0112 against 3e-7). Measure seconds per candidate up to 500 evaluations (k + 1 Gaussian processes at O(n³)) and state the rule in those terms. Count step 0's problems too. Change the rule only for spaces with output constraints.

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

## Delivery

Separate pull requests, in the roadmap's order: (1) step 0, the measuring tools with a new baseline, done; (2) item 1, set aside; (3) step 1b, the cost of a Bayesian candidate measured, which sets the ceiling for every later item; (4) item 2, with item 5's measurement, which didn't pass; (5) item 6, taken ahead of noise with several objectives on 8 Nov 2026 since its measurement had started; (6) noise with several objectives; (7) item 4 last, because the earlier changes alter what it measures. Item 3 only if its test shows a need; it did, for Bayesian optimization, and the decision is open.
