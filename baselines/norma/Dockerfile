FROM mbollmann/norma


COPY ./norma.cfg /home
COPY ./run_train_and_eval.sh /home

RUN wget 'https://gist.github.com/MarvinJWendt/2f4f4154b8ae218600eb091a5706b5f4/raw/36b70dd6be330aa61cd4d4cdfda6234dcb0b8784/wordlist-german.txt' -O /home/wordlist-german.txt

ENTRYPOINT ["/home/run_train_and_eval.sh"]
