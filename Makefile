.DEFAULT_GOAL := test

# uv installs and runs everything. Poetry 2 still works as a fallback: run poetry lock once, then
#   make test-fast RUNNER="poetry run" INSTALL="poetry install" DOCS_INSTALL="poetry install --with docs"
RUNNER ?= uv run
INSTALL ?= uv sync --locked
DOCS_INSTALL ?= uv sync --locked --group docs

.PHONY: install
install:
	@$(INSTALL)

.PHONY: test
test: install
	@$(RUNNER) pytest
	@$(RUNNER) pytest -n 0 -m timing

# For local iteration: leaves out the tests marked slow (10 seconds or more, set from durations measured in Docker). CI and make test run everything.
.PHONY: test-fast
test-fast: install
	@$(RUNNER) pytest -m "not slow"

.PHONY: lint
lint: install
	@$(RUNNER) ruff check hone tests pbt
	@$(RUNNER) mypy

.PHONY: docs
docs:
	@$(DOCS_INSTALL)
	@NO_MKDOCS_2_WARNING=1 $(RUNNER) mkdocs serve

.PHONY: docs-build
docs-build:
	@$(DOCS_INSTALL)
	@NO_MKDOCS_2_WARNING=1 $(RUNNER) mkdocs build --strict

.PHONY: benchmark
benchmark: install
	@$(RUNNER) python -m hone.benchmarks --jobs 4 --markdown benchmarks/results.md --save benchmarks/full-summaries.json

.PHONY: benchmark-ml
benchmark-ml: install
	@$(RUNNER) python -m hone.benchmarks --ml --jobs 4 --markdown benchmarks/ml-results.md --save benchmarks/ml-summaries.json

.PHONY: benchmark-noisy
benchmark-noisy: install
	@$(RUNNER) python -m hone.benchmarks --noisy --jobs 4 --markdown benchmarks/noisy-results.md --save benchmarks/noisy-summaries.json

.PHONY: benchmark-multi
benchmark-multi: install
	@$(RUNNER) python -m hone.benchmarks --multi --jobs 4 --markdown benchmarks/multi-results.md --save benchmarks/multi-summaries.json

.PHONY: benchmark-conditional
benchmark-conditional: install
	@$(RUNNER) python -m hone.benchmarks --conditional --jobs 4 --markdown benchmarks/conditional-results.md --save benchmarks/conditional-summaries.json

.PHONY: benchmark-constrained
benchmark-constrained: install
	@$(RUNNER) python -m hone.benchmarks --constrained --jobs 4 --markdown benchmarks/constrained-results.md --save benchmarks/constrained-summaries.json

.PHONY: benchmark-cost
benchmark-cost: install
	@$(RUNNER) python -m hone.benchmarks --cost --markdown benchmarks/cost-results.md --save benchmarks/cost-summaries.json

.PHONY: benchmark-outputs
benchmark-outputs: install
	@$(RUNNER) python -m hone.benchmarks --outputs --jobs 4 --markdown benchmarks/outputs-results.md --save benchmarks/outputs-summaries.json

# Runs nothing: tabulates the summaries that benchmark-constrained and benchmark-outputs save.
.PHONY: benchmark-engineering
benchmark-engineering: install
	@$(RUNNER) python -m hone.benchmarks --engineering --markdown benchmarks/engineering-results.md

.PHONY: docker-build
docker-build:
	@docker build -t hone .

.PHONY: docker-test
docker-test: docker-build
	@docker run --rm hone

.PHONY: docker-shell
docker-shell: docker-build
	@docker run --rm -it hone bash

# Former name of the test target.
.PHONY: run-tests
run-tests: test
