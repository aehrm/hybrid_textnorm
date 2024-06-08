#!/bin/bash

python preprocess.py config.yml

tail -F mert-work/extract.err &
tail -F mert-work/mert.log &
python train.py config.yml

cp /dataset/test.csmtiser_token.orig .
python normalise.py config.yml ./test.csmtiser_token.orig

cp ./test.csmtiser_token.orig.norm /output/test.csmtiser_token.pred
