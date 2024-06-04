FROM pytorch/pytorch:2.3.0-cuda11.8-cudnn8-runtime

RUN apt-get update && apt-get install -y curl

ARG USER=evaluator
ARG UID=""

RUN if [ -z "${UID}" ]; then adduser ${USER} --home /docker_home --disabled-password --gecos ""; \
        else adduser ${USER} --uid ${UID} --home /docker_home --disabled-password --gecos ""; fi
USER ${USER}


ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    POETRY_NO_INTERACTION=1

RUN curl -sSL https://install.python-poetry.org | python -
ENV PATH="${PATH}:/docker_home/.local/bin"

WORKDIR /docker_home/hybrid_textnorm

COPY pyproject.toml .
COPY poetry.lock .
RUN poetry install --no-root

COPY . /docker_home/hybrid_textnorm

ENTRYPOINT ["poetry", "run"]
