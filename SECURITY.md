# Security policy

## Supported versions

hone is before 1.0, so only the latest minor release gets security fixes. Fixes ship as a patch release of it, such as 0.12.1, to GitHub and PyPI as `hone-python`.

| Version | Supported |
| --- | --- |
| Latest minor release (see [releases](https://github.com/GreatEmuWar/hone/releases)) | Yes |
| Anything older | No: please upgrade |

## Reporting a vulnerability

Please don't open a public issue. Report it privately through GitHub instead:

**[Report a vulnerability](https://github.com/GreatEmuWar/hone/security/advisories/new)** (the repository's Security tab, then "Report a vulnerability").

Include what you can of:

- the version of hone, Python and your operating system
- what an attacker controls, such as a file hone loads or a value passed to it
- the steps or a short script that reproduces it, and what happens

hone is maintained by one person, so replies are on a best-effort basis. You should hear back within a week. Once the problem is confirmed, the fix and a GitHub security advisory are published together, and you're credited in both unless you'd rather not be.

## What counts

hone runs the objective function you give it, so some behaviour that would be a vulnerability elsewhere is the point here. In particular these are by design, not vulnerabilities:

- **Your own code runs.** Objective functions and `Constraint.function` constraints are ordinary Python, run in your process or in worker processes hone starts.
- **Checkpoints are pickles.** Loading one with `load_checkpoint()` can run code, as the README and its docstring say: only load checkpoints you saved yourself.

These would be vulnerabilities, and are worth reporting:

- running code, or reading or writing files hone wasn't asked to, when loading a result (`Result.load`, `ParetoResult.load`) or a config (`from_file`), which are JSON and YAML and are meant to be safe to load from anywhere
- a checkpoint, result or other file written somewhere other than the path given, or with permissions that expose it
- worker processes that keep running, or keep resources, after `optimize()` has returned or been stopped
- a problem in a dependency that hone's own use of it makes exploitable. A problem in the dependency alone belongs with that project.

The examples, tests and saved benchmark results in the repository aren't part of the installed package; problems there are welcome as ordinary issues. `hone.benchmarks`, which is installed, is in scope like the rest of `hone`.
