.DEFAULT_GOAL := run-tests

.PHONY: build
build:
	@echo "Building..."
	@docker-compose build

.PHONY: install-requirements
install-requirements:
	@echo "Installing requirements..."
	@pip install --upgrade pip
	@pip install poetry
	@poetry install


.PHONY: run-tests
run-tests: install-requirements
	@echo "Running tests..."
	@python -m unittest discover -s tests -p '*_test.py'
