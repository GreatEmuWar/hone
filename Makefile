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

.PHONY: docker-build
docker-build:
	@docker build -t hone .

.PHONY: docker-test
docker-test: docker-build
	@docker run --rm hone

# Former name of the test target.
.PHONY: run-tests
run-tests: test
