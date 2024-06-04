FROM greegorey/csmtiser

RUN git clone https://github.com/clarinsi/csmtiser /temp

RUN mv /temp/* /csmtiser

WORKDIR /csmtiser

COPY 0001-fix-code.patch 0001-fix-code.patch
RUN git apply 0001-fix-code.patch

COPY config.yml config.yml
COPY run_train_and_eval.sh run_train_and_eval.sh

ENTRYPOINT ["/csmtiser/run_train_and_eval.sh"]
