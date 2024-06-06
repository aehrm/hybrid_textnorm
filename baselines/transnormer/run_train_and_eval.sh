#!/bin/bash

python src/transnormer/models/train_model.py
python src/transnormer/models/generate.py -c test_config.toml -o /output/test.transnormer.pred
