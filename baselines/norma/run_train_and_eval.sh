#!/bin/bash


norma_lexicon -w wordlist-german.txt -a words.fsm -l words.sym -c
normalize -c norma.cfg -f /dataset/train.norma.parallel -s -t --saveonexit
normalize -c norma.cfg -f /dataset/test.norma.orig -s | tee /proc/self/fd/2 | cut -d$'\t' -f1 > /output/test.norma.pred
