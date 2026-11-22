# Bayesian optimization's run time with output constraints

Measured by the Measure workflow, run 44973122042, on choose-outputs-measure at 2f50643, from `benchmarks/specs/choose_outputs_timing.py`: 70 runs, one at a time per runner, on GitHub's ubuntu-latest runners. Every run, with its machine, is in `choose-outputs-timing-runs.json`.

## Bayesian optimization's run time

Median and 90th percentile over the seeds. The 90th percentile is the nearest-rank one, sorted(x)[ceil(0.9 n) - 1], which is the largest value at n = 5.

| Problem | Dimensions | Output limits | Median at 300 | p90 at 300 | Median at 500 | p90 at 500 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ball-2d-1in10 | 2 | 1 | 46.5 s | 60.9 s (1.0 min) | 122.0 s (2.0 min) | 145.9 s (2.4 min) |
| ball-2d-1in100 | 2 | 1 | 28.4 s | 34.3 s | 72.6 s (1.2 min) | 90.4 s (1.5 min) |
| ball-5d-1in100 | 5 | 1 | 60.4 s (1.0 min) | 84.5 s (1.4 min) | 169.1 s (2.8 min) | 235.9 s (3.9 min) |
| ball-5d-1in1000 | 5 | 1 | 33.5 s | 51.1 s | 100.0 s (1.7 min) | 137.0 s (2.3 min) |
| corner-2d | 2 | 2 | 26.9 s | 58.6 s | 70.0 s (1.2 min) | 184.2 s (3.1 min) |
| corner-5d | 5 | 2 | 40.0 s | 71.1 s (1.2 min) | 159.0 s (2.7 min) | 239.2 s (4.0 min) |
| g06 | 2 | 2 | 22.3 s | 25.1 s | 72.4 s (1.2 min) | 81.6 s (1.4 min) |
| g07 | 10 | 8 | 248.1 s (4.1 min) | 294.1 s (4.9 min) | 883.6 s (14.7 min) | 1150.8 s (19.2 min) |
| g10 | 8 | 6 | 98.2 s (1.6 min) | 166.1 s (2.8 min) | 463.1 s (7.7 min) | 586.6 s (9.8 min) |
| pocket-2d-1in100 | 2 | 1 | 11.9 s | 16.6 s | 41.2 s | 63.0 s (1.1 min) |
| pocket-5d-1in100 | 5 | 1 | 44.0 s | 52.6 s | 121.7 s (2.0 min) | 148.2 s (2.5 min) |
| pressure-vessel-outputs | 4 | 3 | 87.7 s (1.5 min) | 120.4 s (2.0 min) | 241.4 s (4.0 min) | 321.5 s (5.4 min) |
| tension-spring-outputs | 3 | 3 | 75.0 s (1.3 min) | 114.6 s (1.9 min) | 213.1 s (3.6 min) | 302.6 s (5.0 min) |
| welded-beam-outputs | 4 | 4 | 59.5 s | 89.2 s (1.5 min) | 162.8 s (2.7 min) | 220.0 s (3.7 min) |

Machines:

- AMD EPYC 7763 64-Core Processor: 22 runs
- AMD EPYC 9V45 96-Core Processor: 16 runs
- AMD EPYC 9V74 80-Core Processor: 20 runs
- INTEL(R) XEON(R) PLATINUM 8573C: 4 runs
- Intel(R) Xeon(R) 6973P-C: 2 runs
- Intel(R) Xeon(R) 6975P-C: 2 runs
- Intel(R) Xeon(R) Platinum 8370C CPU @ 2.80GHz: 4 runs

Wall time including the objective, which is close to zero on these problems; one run at a time per runner.

Over about 10 minutes at 8 limits: goes to the user. (g07)

