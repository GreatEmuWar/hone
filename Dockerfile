# Pinned by digest, which Dependabot keeps current.
FROM python:3.15-slim@sha256:65a65fc7b30b06cb6df4930fd4650697cb23f4c2698ab81fc87e4b031bc36928

WORKDIR /src

# Poetry and what it needs, each checked against its hash.
COPY ./.github/requirements/build.txt /tmp/build.txt
RUN python -m pip install --no-cache-dir --require-hashes -r /tmp/build.txt \
    && poetry config virtualenvs.create false

# Install dependencies first, so editing the code does not invalidate this layer.
COPY ./pyproject.toml ./poetry.lock ./README.md ./LICENSE /src/
RUN poetry install --no-root --no-interaction --no-ansi

COPY ./hone /src/hone
COPY ./tests /src/tests
COPY ./pbt/tests /src/pbt/tests
COPY ./examples /src/examples
# The saved benchmark results, which test that choose_optimizer's rule is still backed by them.
COPY ./benchmarks/*-summaries.json /src/benchmarks/
# The noisy multi-objective floor's runs, which the comparison spec's tests recompute its unresolvable cells from.
COPY ./benchmarks/multi-floor-runs.json /src/benchmarks/
# The specs for measurements shared among machines, which the image can run and its tests load.
COPY ./benchmarks/specs /src/benchmarks/specs
RUN poetry install --only-root --no-interaction --no-ansi

# Run the tests as an ordinary user, who owns /src for pytest's cache.
RUN useradd --create-home hone && chown -R hone:hone /src
USER hone

CMD ["sh", "-c", "pytest && pytest -n 0 -m timing"]
