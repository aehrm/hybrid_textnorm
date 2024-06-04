#!/bin/bash

python preprocess.py config.yml

tail -F mert-work/extract.err &
tail -F mert-work/mert.log &
python train.py config.yml

cp /dataset/test.csmtiser_sent.orig .
python normalise.py config.yml ./test.csmtiser_sent.orig

cp ./test.csmtiser_sent.orig.norm /output/test.csmtiser_sent.pred
