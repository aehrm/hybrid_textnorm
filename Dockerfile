FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

RUN apt-get update && apt-get install -y curl

ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    POETRY_NO_INTERACTION=1

ARG USER=evaluator
ARG UID=""

RUN if [ -z "${UID}" ]; then adduser ${USER} --home /docker_home --disabled-password --gecos ""; \
        else adduser ${USER} --uid ${UID} --home /docker_home --disabled-password --gecos ""; fi
USER ${USER}

RUN curl -sSL https://install.python-poetry.org | python -
ENV PATH="${PATH}:/docker_home/.local/bin"
RUN poetry config virtualenvs.in-project true

WORKDIR /hybrid_textnorm

COPY pyproject.toml .
COPY poetry.lock .
RUN poetry install --no-root

COPY . /hybrid_textnorm

ENTRYPOINT ["poetry", "run"]