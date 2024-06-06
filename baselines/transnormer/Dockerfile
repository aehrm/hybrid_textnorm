FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

RUN apt update && apt install -y git
RUN git clone https://github.com/ybracke/transnormer /repo
WORKDIR /repo
RUN git checkout 0913278
COPY 0001-fix-code.patch 0001-fix-code.patch
RUN git apply 0001-fix-code.patch

WORKDIR /transnormer
RUN mv /repo/* .


RUN pip install --upgrade pip
RUN pip install -e .

COPY training_config.toml training_config.toml
COPY test_config.toml test_config.toml
COPY run_train_and_eval.sh run_train_and_eval.sh

ENTRYPOINT ["/transnormer/run_train_and_eval.sh"]
