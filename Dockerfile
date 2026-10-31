FROM python:3.14-slim

WORKDIR /src

RUN python -m pip install --no-cache-dir poetry \
    && poetry config virtualenvs.create false

# Install dependencies first, so editing the code does not invalidate this layer.
COPY ./pyproject.toml ./poetry.lock ./README.md ./LICENSE /src/
RUN poetry install --no-root --no-interaction --no-ansi

COPY ./hone /src/hone
COPY ./tests /src/tests
COPY ./examples /src/examples
# The saved benchmark results, which test that choose_optimizer's rule is still backed by them.
COPY ./benchmarks/*-summaries.json /src/benchmarks/
RUN poetry install --only-root --no-interaction --no-ansi

CMD ["pytest"]
