FROM python:3.14-slim

WORKDIR /src

COPY  ./hone /src/hone
COPY ./pyproject.toml ./poetry.lock ./README.md /src/

RUN apt-get update && apt-get install -y
RUN python -m pip install --upgrade pip
RUN python -m pip install poetry
RUN poetry config virtualenvs.create false
RUN poetry install --only main --no-interaction --no-ansi

CMD ["python", "-m", "unittest", "discover", "-s", "hone/tests"]