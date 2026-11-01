.DEFAULT_GOAL := test

.PHONY: install
install:
	@poetry install

.PHONY: test
test: install
	@poetry run pytest

.PHONY: lint
lint: install
	@poetry run ruff check hone tests pbt
	@poetry run mypy

.PHONY: docs
docs:
	@poetry install --with docs
	@NO_MKDOCS_2_WARNING=1 poetry run mkdocs serve

.PHONY: docs-build
docs-build:
	@poetry install --with docs
	@NO_MKDOCS_2_WARNING=1 poetry run mkdocs build --strict

.PHONY: benchmark
benchmark: install
	@poetry run python -m hone.benchmarks --jobs 4 --markdown benchmarks/results.md --save benchmarks/full-summaries.json

.PHONY: benchmark-ml
benchmark-ml: install
	@poetry run python -m hone.benchmarks --ml --jobs 4 --markdown benchmarks/ml-results.md --save benchmarks/ml-summaries.json

.PHONY: benchmark-noisy
benchmark-noisy: install
	@poetry run python -m hone.benchmarks --noisy --jobs 4 --markdown benchmarks/noisy-results.md --save benchmarks/noisy-summaries.json

.PHONY: benchmark-multi
benchmark-multi: install
	@poetry run python -m hone.benchmarks --multi --jobs 4 --markdown benchmarks/multi-results.md --save benchmarks/multi-summaries.json

.PHONY: benchmark-conditional
benchmark-conditional: install
	@poetry run python -m hone.benchmarks --conditional --jobs 4 --markdown benchmarks/conditional-results.md --save benchmarks/conditional-summaries.json

.PHONY: benchmark-constrained
benchmark-constrained: install
	@poetry run python -m hone.benchmarks --constrained --jobs 4 --markdown benchmarks/constrained-results.md --save benchmarks/constrained-summaries.json

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
