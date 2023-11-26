FROM python:3.11-bullseye

WORKDIR /src

COPY  ./hone /src/hone
COPY ./pyproject.toml /src

RUN apt-get update && apt-get install -y
RUN python -m pip install --upgrade pip
RUN python -m pip install poetry
RUN poetry config virtualenvs.create false
RUN poetry install --no-dev --no-interaction --no-ansi

CMD ["python", "-m", "unittest", "discover", "-s", "hone/tests"]