#!/bin/bash

normalize -c norma.cfg -f /dataset/train.norma.parallel -s -t --saveonexit
normalize -c norma.cfg -f /dataset/test.norma.orig -s | tee /proc/self/fd/2 | cut -d$'\t' -f1 > /output/test.norma.pred
