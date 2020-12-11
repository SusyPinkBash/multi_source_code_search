#!/bin/bash

rm res/*.pkl
python3 src/extract_data.py $1
python3 src/search_data.py $2
python3 src/prec_recall.py res/data.csv res/ground-truth.txt