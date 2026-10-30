.DEFAULT_GOAL := test

.PHONY: install
install:
	@poetry install

.PHONY: test
test: install
	@poetry run pytest

.PHONY: lint
lint: install
	@poetry run ruff check hone tests

.PHONY: benchmark
benchmark: install
	@poetry run python -m hone.benchmarks --jobs 4 --markdown benchmarks/results.md --save benchmarks/full-summaries.json

.PHONY: benchmark-ml
benchmark-ml: install
	@poetry run python -m hone.benchmarks --ml --jobs 4 --markdown benchmarks/ml-results.md --save benchmarks/ml-summaries.json

.PHONY: benchmark-noisy
benchmark-noisy: install
	@poetry run python -m hone.benchmarks --noisy --jobs 4 --markdown benchmarks/noisy-results.md --save benchmarks/noisy-summaries.json

.PHONY: docker-build
docker-build:
	@docker build -t hone .

.PHONY: docker-test
docker-test: docker-build
	@docker run --rm hone

# Former name of the test target.
.PHONY: run-tests
run-tests: test
