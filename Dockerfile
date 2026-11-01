# Pinned by digest, which Dependabot keeps current.
FROM python:3.14-slim@sha256:cd89c30eda26ee911a4611b18680086e0ea42e6f5f8cbe653d07fa7958c9cec8

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
RUN poetry install --only-root --no-interaction --no-ansi

# Run the tests as an ordinary user, who owns /src for pytest's cache.
RUN useradd --create-home hone && chown -R hone:hone /src
USER hone

CMD ["pytest"]
